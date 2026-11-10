#!/usr/bin/env python3
"""R1 (ICLR 2027 rebuttal) GPU runner -- alpha* arms with per-rung, dual-split traces.

    python3 run_r1.py --config configs/r1/e4.yaml --count
    python3 run_r1.py --config configs/r1/e4.yaml --unit 7 --arms C-rand
    python3 run_r1.py --config configs/r1/e4.yaml --progress        # counts only

Design: R1_REBUTTAL_EXPERIMENTS.md (registered, commit 0fe9043) §0 and E4/E5;
implementation choices that the plan leaves open: R1_NOTES.md §2.

ONE UNIT = ONE PROCESS = ONE (target, axis, seed, designer), exactly as
scripts/run_unit.py ran every published alpha* panel: load the checkpoint, take
the baselines BEFORE any training pass, train the two task vectors once, then
build every arm from the same contrast vector v. The training call, the
calibration items, the collateral measurement and the alpha* search are the
published ones, imported rather than copied (run_opsel.train_contrast,
run_opsel.load_mmlu_calib, run_opsel.collateral_eval, v12_opsel.select_alpha).

WHAT IS NEW RELATIVE TO run_opsel.py (the one registered protocol change, §0):

  * every probed rung is measured on BOTH splits under one application of the
    edit: calibration collateral (what alpha* is selected on) and the
    evaluation-split fast_eval (skew, MMLU, perplexity -- what the removal
    curves are drawn from). The selector still sees calibration collateral
    only; that is enforced by src/r1_select.py, which raises on any dict that
    carries a bias field.
  * the removal reported at alpha* is the evaluation-split measurement taken at
    that rung, in the same process (the published runner re-applied the edit and
    measured once more; evaluation is deterministic within a process, so the
    two are the same number).
  * curve-only measurements the selector never sees: one ladder rung above the
    first failure, and the frozen grid {2,4,8,16} where the ladder did not
    already visit it (sign arms only). Each carries its `role`.
  * the per-item log-likelihoods behind every evaluation-split skew are saved
    (probe_ll/*.npz), so any item split of the bias probe can be recomputed
    without GPU time.
  * a realized-edit diagnostic per rung: how many edited coordinates actually
    change in bf16 and the L1/L2 of the applied change (post-hoc descriptive,
    used by no registered reading).

ARMS.
  the seven DEC conditions (src/v8_edits.py), 1% density, ref_tensor scale:
      C-ref C-a C-bottom C-b C-rand C-layershuf C-tensorshuf
  E5 operators on the same contrast vector (merge W <- W - alpha * E):
      O-trim     top-1% by |v| (C-ref's support), full-precision values
      O-dare     C-a's random 1% support, full-precision values x 100
                 (= 1/(1-p), p = 0.99: DARE's unbiased rescale)
      O-compeft  top-1% by |v|, sign, ONE global scale sigma(v) (ComPEFT, Alg. 1)

Resumable at (target, axis, seed, designer, arm): finished arms are skipped,
and a unit whose arms are all on disk exits before loading anything.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

DEC_ARMS = ["C-ref", "C-a", "C-bottom", "C-b", "C-rand", "C-layershuf", "C-tensorshuf"]
OP_ARMS = ["O-trim", "O-dare", "O-compeft"]


# ---------------------------------------------------------------- config ----
def load_cfg(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("density", 0.01)
    cfg.setdefault("frozen_grid", [2, 4, 8, 16])
    cfg.setdefault("refine_steps", 3)
    cfg.setdefault("curve_extra_rungs", 1)
    cfg.setdefault("ref_panel", "v12astar")
    cfg.setdefault("arm_ladder", {})
    for a in cfg["arms"]:
        if a not in DEC_ARMS + OP_ARMS:
            raise SystemExit(f"unknown arm {a!r}")
    return cfg


def unit_list(cfg):
    return [dict(c, seed=s) for c in cfg["cells"] for s in cfg["seeds"]]


def ladder_for(cfg, arm):
    name = cfg["arm_ladder"].get(arm, "default")
    return [float(a) for a in cfg["ladders"][name]], name


def _rows(fp):
    out = []
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def _append(fp, row):
    with open(fp, "a") as f:
        f.write(json.dumps(row) + "\n")


def _key(u):
    return (u["target"], u["axis"], u["seed"], u["designer"])


def done_arms(out_dir):
    done = set()
    for r in _rows(os.path.join(out_dir, "removal.jsonl")):
        done.add((r["target"], r["axis"], r["seed"], r["designer"], r["arm"]))
    for r in _rows(os.path.join(out_dir, "not_applicable.jsonl")):
        done.add((r["target"], r["axis"], r["seed"], r["designer"], r["arm"]))
    return done


# ------------------------------------------------------------ measurement ----
class _CaptureLL:
    """Record what the bias probe's log-likelihood call returns.

    axis_skew makes exactly one call to cont_loglik (occ_gender) or seq_loglik
    (pair axes) and reduces it to one number. Wrapping the two functions keeps
    the per-item values without touching how the skew is computed.
    """

    def __init__(self, C):
        self.C, self.out = C, []

    def __enter__(self):
        C = self.C
        self._c, self._s = C.cont_loglik, C.seq_loglik

        def wrap(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                self.out.append(r.detach().float().cpu().numpy().copy())
                return r
            return inner
        C.cont_loglik, C.seq_loglik = wrap(self._c), wrap(self._s)
        return self

    def __exit__(self, *exc):
        self.C.cont_loglik, self.C.seq_loglik = self._c, self._s
        return False


def eval_split(C, model, tok, ax, mmlu, wt, batch):
    """colab_t2t4.fast_eval on the evaluation split + the probe's per-item ll."""
    with _CaptureLL(C) as cap:
        post = C.fast_eval(model, tok, ax, mmlu, wt, batch)
    if len(cap.out) != 1:
        raise RuntimeError(f"expected one bias-probe ll call, saw {len(cap.out)}")
    return post, cap.out[0]


def realized_change(C, model, undo):
    """What the merge actually did to the bf16 weights (descriptive only).

    apply_edit keeps the pre-edit clones in its undo closure; compare against
    them rather than cloning a second time.
    """
    import torch
    saved = next(c.cell_contents for c in undo.__closure__
                 if isinstance(c.cell_contents, dict))
    n_changed, l1, l2 = 0, 0.0, 0.0
    with torch.no_grad():
        for key, w0 in saved.items():
            d = C._resolve(model, key).weight.detach().float() - w0.float()
            n_changed += int((d != 0).sum().item())
            l1 += float(d.abs().sum().item())
            l2 += float((d.double() ** 2).sum().item())
            del d
    return dict(n_changed=n_changed, realized_l1=l1, realized_l2=float(np.sqrt(l2)))


# ------------------------------------------------------- E2: alignment ----
def e2_stats(v, density, seed, ref_scales):
    """First-order alignment bookkeeping for E2 (a function of v alone).

    A_ref = sum_t s_t * sum_{i in top-k of t} |v_i|   (C-ref, its own scales)
    A_a   = sum_t s_t * sum_{i in C-a's support} |v_i| (same per-tensor scales)
    rho_eff = A_ref / A_a is the first-order removal ratio per unit alpha that
    the two arms as actually built should show (App. H, Prop. 3, with the real
    scale construction). rho_k is the plain concentration ratio at k = density.
    """
    import v8_edits as V8
    m_top = V8._masks(v, "C-ref", density, seed)
    m_rand = V8._masks(v, "C-a", density, seed)
    n = sum(t.numel() for t in v.values())
    mean = sum(float(t.double().sum()) for t in v.values()) / n
    A_ref = A_a = top_l1 = rand_l1 = l1 = m2 = m4 = 0.0
    k_top = k_rand = 0
    for key, t in v.items():
        a = t.abs().double()
        st = float(ref_scales[key])
        tl, rl = float(a[m_top[key]].sum()), float(a[m_rand[key]].sum())
        A_ref += st * tl
        A_a += st * rl
        top_l1 += tl
        rand_l1 += rl
        l1 += float(a.sum())
        k_top += int(m_top[key].sum())
        k_rand += int(m_rand[key].sum())
        c = t.double() - mean
        c2 = c * c
        m2 += float(c2.sum())
        m4 += float((c2 * c2).sum())
        del a, c, c2
    del m_top, m_rand
    m2, m4 = m2 / n, m4 / n
    return dict(A_ref=A_ref, A_a=A_a, rho_eff=(A_ref / A_a if A_a > 0 else None),
                k_top=k_top, k_rand=k_rand, n=n, l1=l1, top_l1=top_l1, rand_l1=rand_l1,
                rho_k=((top_l1 / k_top) / (l1 / n) if k_top and l1 > 0 else None),
                rho_k_random_support=((top_l1 / k_top) / (rand_l1 / k_rand)
                                      if k_top and rand_l1 > 0 else None),
                top_mass_share=(top_l1 / l1 if l1 > 0 else None),
                std=float(np.sqrt(m2)), kurtosis=(m4 / (m2 * m2) if m2 > 0 else None))


# ------------------------------------------------------------------ arms ----
def applicable(v, arm):
    from v8_support import parse_module
    if arm == "C-tensorshuf":
        # run_dec.applicable_conditions: a one-projection-per-layer architecture
        # (phi's fused qkv_proj) has nothing to shuffle and would return C-ref.
        return len({parse_module(k)[1] for k in v}) > 1
    return True


def build_arm(v, arm, seed, density, ref_scales):
    """-> (E, meta). Never mutates v."""
    import torch
    import v8_edits as V8
    if arm in DEC_ARMS:
        E, meta = V8.build_edit(v, arm, density, seed, scale_mode="ref_tensor",
                                ref_scales=ref_scales)
        meta = dict(meta, kind="dec")
        meta.pop("permutation_map", None)        # logged by the DEC panel already
        return E, meta

    if arm == "O-trim":
        masks = V8._masks(v, "C-ref", density, seed)
        E = {k: v[k] * masks[k].float() for k in v}
        extra = dict(kind="op", support="top", values="fp", scale="alpha")
    elif arm == "O-dare":
        masks = V8._masks(v, "C-a", density, seed)       # C-a's support, same draws
        resc = 1.0 / density                              # 1/(1-p), p = 1 - density
        E = {k: v[k] * masks[k].float() * resc for k in v}
        extra = dict(kind="op", support="random", values="fp", rescale=resc,
                     scale="alpha")
    elif arm == "O-compeft":
        masks = V8._masks(v, "C-ref", density, seed)
        n = sum(t.numel() for t in v.values())
        s1 = sum(float(t.double().sum()) for t in v.values())
        s2 = sum(float((t.double() ** 2).sum()) for t in v.values())
        sigma = float(np.sqrt(max(s2 / n - (s1 / n) ** 2, 0.0)))   # std of the whole v
        E = {k: torch.sign(v[k]) * masks[k].float() * sigma for k in v}
        extra = dict(kind="op", support="top", values="sign", sigma=sigma,
                     scale="alpha*sigma(v), one scalar for the whole edit")
    else:
        raise ValueError(arm)
    n_flips = int(sum(int((e != 0).sum().item()) for e in E.values()))
    n_total = int(sum(e.numel() for e in E.values()))
    fro = float(torch.sqrt(sum((e.double() ** 2).sum() for e in E.values())))
    meta = dict(condition=arm, density=density, n_flips=n_flips, n_params=n_total,
                effective_sparsity=1.0 - n_flips / max(n_total, 1), frobenius=fro,
                **extra)
    del masks
    return E, meta


# ------------------------------------------------------------------ unit ----
def run_unit(cfg, u, arms, panel):
    import torch
    import colab_t2t4 as C
    import v8_edits as V8
    import v9_gate
    import v12_opsel as S
    import run_opsel as RO
    import r1_select as RS

    out_dir = os.path.join(C.RESULTS, panel)
    os.makedirs(os.path.join(out_dir, "probe_ll"), exist_ok=True)
    fam, hf, dn, ax, s = u["target"], u["hf"], u["designer"], u["axis"], u["seed"]
    key = dict(panel=panel, target=fam, axis=ax, seed=s, designer=dn)
    density = cfg["density"]
    BATCH = RO.BATCH

    done = done_arms(out_dir)
    todo = [a for a in arms if (fam, ax, s, dn, a) not in done]
    if not todo:
        print(f"[r1] {fam} {ax} s{s}: nothing to do", flush=True)
        return 0

    corpus_fp = C._t2x_corpus_fp(dn, ax, s)
    if not os.path.exists(corpus_fp):
        raise RuntimeError(f"corpus missing: {corpus_fp} (R1 elicits nothing new)")

    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)
    mmlu_cal = RO.load_mmlu_calib(200, seed=0)
    wt_cal = C.load_wikitext(n_chunks=20, seed=0, split="validation")
    if set(r["question"] for r in mmlu_cal) & set(r["question"] for r in mmlu):
        raise RuntimeError("calibration and evaluation MMLU items overlap")
    if set(wt_cal) & set(wt):
        raise RuntimeError("calibration and evaluation WikiText chunks overlap")

    t0 = time.time()
    model, tok = C.load_model(hf, dispatch=False)
    try:
        # Baselines BEFORE the first training pass, as run_dec / run_opsel do.
        pre, pre_ll = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
        pre_cal = RO.collateral_eval(model, tok, mmlu_cal, wt_cal)
        pre_cal_c = RS.collateral_only(pre_cal)

        v, losses = RO.train_contrast(model, tok, dn, ax, s)
        g = RO.geometry_of(v)

        # ---- replay guards against the registered alpha* panel ----
        ref_panel = cfg["ref_panel"]
        ref_geo = [r for r in _rows(os.path.join(C.RESULTS, ref_panel, "geometry.jsonl"))
                   if _key(r) == _key(u)]
        ref_sha = ref_geo[-1]["geometry"]["geom_sha"] if ref_geo else None
        ref_tr = [r for r in _rows(os.path.join(C.RESULTS, ref_panel, "alpha_trace.jsonl"))
                  if _key(r) == _key(u)]
        ref_ev = next((r for r in reversed(ref_tr) if r.get("split") == "eval"), None)
        ref_ca = next((r for r in reversed(ref_tr) if r.get("split") == "calib"), None)
        base = dict(
            geom_sha=g["geom_sha"], ref_geom_sha=ref_sha,
            geom_match=(None if ref_sha is None else ref_sha == g["geom_sha"]),
            pre_eval=dict(skew=pre["skew"], mmlu_acc=pre["mmlu_acc"], ppl=pre["ppl"]),
            pre_calib=dict(mmlu_acc=pre_cal["mmlu_acc"], ppl=pre_cal["ppl"]),
            ref_pre_eval=(None if ref_ev is None else dict(
                skew=ref_ev.get("pre_skew"), mmlu_acc=ref_ev.get("pre_mmlu"),
                ppl=ref_ev.get("pre_ppl"))),
            ref_pre_calib=(None if ref_ca is None else dict(
                mmlu_acc=ref_ca.get("pre_mmlu"), ppl=ref_ca.get("pre_ppl"))))
        base["pre_eval_match"] = (None if ref_ev is None else
                                  base["pre_eval"] == base["ref_pre_eval"])
        base["pre_calib_match"] = (None if ref_ca is None else
                                   base["pre_calib"] == base["ref_pre_calib"])
        ref_scales = V8.ref_scale_table(v, density, s)
        e2 = e2_stats(v, density, s, ref_scales) if cfg.get("e2_stats", True) else None

        import transformers, peft, datasets
        _append(os.path.join(out_dir, "unit.jsonl"), dict(
            key, hf=hf, arms=todo, e2=e2, **base, **losses,
            geometry={k: g[k] for k in ("n_params", "l1", "l2", "pr_frac", "gini",
                                        "thresh_at_1pct")},
            mass_1pct=g["mass"].get(RO.G._pkey(0.01)),
            stack=dict(torch=torch.__version__, transformers=transformers.__version__,
                       peft=peft.__version__, datasets=datasets.__version__,
                       gpu=torch.cuda.get_device_name(0)),
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        print(f"[r1] {fam} {ax} s{s} geom_sha={g['geom_sha']} "
              f"match={base['geom_match']} pre_eval_match={base['pre_eval_match']} "
              f"pre_calib_match={base['pre_calib_match']} "
              f"(setup {time.time() - t0:.0f}s)", flush=True)

        for arm in todo:
            ta = time.time()
            if not applicable(v, arm):
                _append(os.path.join(out_dir, "not_applicable.jsonl"), dict(
                    key, arm=arm, reason="architecture resolves to <2 attention "
                    "projections; the shuffle would return C-ref unchanged"))
                print(f"[r1] {fam} {ax} s{s} {arm}: N/A (structural)", flush=True)
                continue
            E, meta = build_arm(v, arm, s, density, ref_scales)
            ladder, ladder_name = ladder_for(cfg, arm)
            rungs, lls = {}, []

            def measure(alpha, role):
                alpha = float(alpha)
                undo = C.apply_edit(model, E, alpha=alpha, sign=-1.0)
                try:
                    diag = realized_change(C, model, undo)
                    cal = RO.collateral_eval(model, tok, mmlu_cal, wt_cal)
                    post, ll = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
                finally:
                    undo()
                cal_c = RS.collateral_only(cal)
                ok_c = RS.gate(pre_cal_c, cal_c)
                ok_e = C.collateral_ok(pre, post)
                r = C.bias_reduction(pre, post)
                n_c, n_e = pre_cal["n_items"], pre.get("n_items", 200)
                rec = dict(
                    alpha=alpha, role=role, **diag,
                    calib=dict(mmlu_acc=cal["mmlu_acc"], ppl=cal["ppl"], n_items=n_c,
                               dmmlu_items=int(v9_gate.items_from_acc(pre_cal["mmlu_acc"], n_c)
                                               - v9_gate.items_from_acc(cal["mmlu_acc"], n_c)),
                               ppl_ratio=cal["ppl"] / pre_cal["ppl"], ok=bool(ok_c)),
                    eval=dict(mmlu_acc=post["mmlu_acc"], ppl=post["ppl"], n_items=n_e,
                              dmmlu_items=int(v9_gate.items_from_acc(pre["mmlu_acc"], n_e)
                                              - v9_gate.items_from_acc(post["mmlu_acc"], n_e)),
                              ppl_ratio=post["ppl"] / pre["ppl"], ok=bool(ok_e),
                              post_skew=post["skew"], removal=float(r)))
                rungs[alpha] = rec
                lls.append((alpha, ll))
                tr = os.path.join(out_dir, "alpha_trace.jsonl")
                _append(tr, dict(
                    key, arm=arm, split="calib", rule="adaptive", role=role, alpha=alpha,
                    pre_mmlu=pre_cal["mmlu_acc"], post_mmlu=cal["mmlu_acc"], n_items=n_c,
                    pre_ppl=pre_cal["ppl"], post_ppl=cal["ppl"],
                    ppl_ratio=cal["ppl"] / pre_cal["ppl"], collateral_ok=bool(ok_c)))
                _append(tr, dict(
                    key, arm=arm, split="eval", rule="adaptive", role=role, alpha=alpha,
                    pre_mmlu=pre["mmlu_acc"], post_mmlu=post["mmlu_acc"], n_items=n_e,
                    pre_ppl=pre["ppl"], post_ppl=post["ppl"],
                    ppl_ratio=post["ppl"] / pre["ppl"], pre_skew=pre["skew"],
                    post_skew=post["skew"], collateral_ok=bool(ok_e),
                    bias_reduction=float(r), **diag))
                return rec

            # ---- alpha*: the selector sees calibration collateral, nothing else ----
            def collateral_probe(alpha):
                rec = measure(alpha, "rung")
                return {k: rec["calib"][k] for k in ("mmlu_acc", "ppl", "n_items")}

            sel = RS.select(pre_cal_c, collateral_probe, ladder, cfg["refine_steps"])
            a_star = sel["alpha_star"]

            # ---- curve-only measurements, after and outside the selection ----
            if sel["first_fail"] is not None:
                above = [a for a in ladder if a > sel["first_fail"]]
                for a in above[:cfg["curve_extra_rungs"]]:
                    if a not in rungs:
                        measure(a, "curve_extra")
            frozen = None
            if arm in DEC_ARMS:
                for a in cfg["frozen_grid"]:
                    if float(a) not in rungs:
                        measure(a, "frozen_fill")
                fr = [(float(a), rungs[float(a)]["eval"]["ok"],
                       rungs[float(a)]["eval"]["removal"]) for a in cfg["frozen_grid"]]
                best, best_a = S.frozen_argmax(fr)
                frozen = dict(removal=best, alpha=best_a)

            adaptive = {k: sel[k] for k in ("alpha_star", "status", "n_probes",
                                            "last_pass", "first_fail")}
            if a_star is not None:
                ev, ca = rungs[a_star]["eval"], rungs[a_star]["calib"]
                adaptive.update(removal=ev["removal"], eval_ok=ev["ok"],
                                post_skew=ev["post_skew"], dmmlu_items=ev["dmmlu_items"],
                                n_items=ev["n_items"], ppl_ratio=ev["ppl_ratio"],
                                calib_dmmlu_items=ca["dmmlu_items"],
                                calib_ppl_ratio=ca["ppl_ratio"],
                                n_changed=rungs[a_star]["n_changed"],
                                realized_l2=rungs[a_star]["realized_l2"])

            ll_fp = os.path.join(out_dir, "probe_ll",
                                 f"{fam}__{ax}__s{s}__{dn}__{arm}.npz")
            order = sorted(range(len(lls)), key=lambda i: lls[i][0])
            np.savez_compressed(ll_fp, alphas=np.array([lls[i][0] for i in order]),
                                ll=np.stack([lls[i][1] for i in order]), pre_ll=pre_ll)

            _append(os.path.join(out_dir, "removal.jsonl"), dict(
                key, arm=arm, role="self", density=density, ladder=ladder_name,
                geom_sha=g["geom_sha"], geom_match=base["geom_match"],
                pre_skew=pre["skew"], n_items=pre.get("n_items", 200),
                meta={k: meta.get(k) for k in (
                    "kind", "n_flips", "n_params", "effective_sparsity", "frobenius",
                    "scale_mode", "support", "values", "rescale", "sigma", "scale")},
                frozen=frozen, adaptive=adaptive,
                rungs=[rungs[a] for a in sorted(rungs)],
                seconds=round(time.time() - ta, 1)))
            # Operational line only: alpha*, probe count, gate status. Removals
            # are read by the analysis script once the panel is complete.
            print(f"[r1] {fam:7s} {ax:20s} s{s} {arm:13s} status={sel['status']} "
                  f"alpha*={a_star} probes={sel['n_probes']} rungs={len(rungs)} "
                  f"({time.time() - ta:.0f}s)", flush=True)
            del E
        del v
    finally:
        C._free(model)
    print(f"[r1] unit {fam} {ax} s{s} finished in {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--arms", default=None, help="comma list; default: the config's")
    ap.add_argument("--panel", default=None, help="override the output panel "
                    "(a throwaway name for smoke tests; never cite such a run)")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    units = unit_list(cfg)
    arms = a.arms.split(",") if a.arms else list(cfg["arms"])
    panel = a.panel or cfg["panel"]
    if a.count:
        print(len(units))
        return 0
    if a.progress:
        results = os.environ.get("BS_OUT", "results")
        done = done_arms(os.path.join(results, panel))
        need = [(u["target"], u["axis"], u["seed"], u["designer"], arm)
                for u in units for arm in arms]
        print(f"{panel}: {sum(k in done for k in need)}/{len(need)} unit-arms on disk "
              f"(arms: {','.join(arms)})")
        return 0
    if a.unit is None:
        raise SystemExit("give --unit, --count or --progress")
    if not 0 <= a.unit < len(units):
        print(f"[r1] unit {a.unit} out of range (0..{len(units) - 1})")
        return 0
    u = units[a.unit]
    print(f"[r1] unit {a.unit}: {u['target']} {u['axis']} s{u['seed']} "
          f"designer={u['designer']} arms={arms} panel={panel}", flush=True)
    try:
        return run_unit(cfg, u, arms, panel)
    except Exception:
        print(f"[r1] unit {a.unit} FAILED", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
