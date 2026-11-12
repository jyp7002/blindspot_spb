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
# E6b (R1_NOTES.md section 3, 2026-11-12): support re-selected inside each tensor
ACT_ARMS = ["S-act1", "S-fisher", "S-wanda"]


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
    # E8 (alpha* backoff and budget sensitivity); all empty = the E4 behaviour
    cfg.setdefault("backoff_arms", [])
    cfg.setdefault("backoff_factors", [])
    cfg.setdefault("mmlu1k_arms", [])
    cfg.setdefault("budget_items_extra", [])
    # 27-32B panel (configs/r1/big.yaml)
    cfg.setdefault("elicit_missing", False)      # elicit a missing self-designer corpus
    cfg.setdefault("grid_only_arms", {})         # arm -> [alpha, ...]: no alpha* search
    # ... unless the retrained dW is not the reference panel's: then the reference
    # alpha* values do not apply to this unit and the search is run here as well
    cfg.setdefault("grid_only_fallback_full", True)
    # E6 calibration set: WikiText-103 train, act_seqs sequences of act_tokens tokens
    cfg.setdefault("act_seqs", 256)
    cfg.setdefault("act_tokens", 512)
    for a in cfg["arms"]:
        if a not in DEC_ARMS + OP_ARMS + ACT_ARMS:
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


def build_arm(v, arm, seed, density, ref_scales, stats=None):
    """-> (E, meta). Never mutates v."""
    import torch
    import v8_edits as V8
    if arm in ACT_ARMS:
        return build_act_arm(v, arm, seed, density, ref_scales, stats)
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


# ------------------------------------------------- E6: activation-aware ----
def calib_sequences(C, tok, n_seq, n_tok, seed=0):
    """n_seq token sequences of exactly n_tok tokens from WikiText-103 TRAIN.

    Disjoint by split from every chunk the gate reads (validation, test).
    Chunks are taken in a seeded order and joined until a sequence is long
    enough, then truncated; deterministic given the tokenizer.
    """
    chunks = C.load_wikitext(n_chunks=None, seed=0, split="train")
    order = np.random.default_rng(seed).permutation(len(chunks))
    seqs, buf = [], []
    for i in order:
        buf.append(chunks[int(i)])
        ids = tok("\n\n".join(buf), truncation=True, max_length=n_tok)["input_ids"]
        if len(ids) >= n_tok:
            seqs.append(ids)
            buf = []
            if len(seqs) == n_seq:
                break
    if len(seqs) < n_seq:
        raise RuntimeError(f"only {len(seqs)} calibration sequences of {n_tok} tokens")
    return seqs


def act_stats(C, model, tok, keys, n_seq, n_tok, want_fisher):
    """Input second moments and (optionally) the empirical Fisher diagonal.

    n2[key][j]  = mean over calibration tokens of x_j^2, x = the input of the
                  edited projection `key` (so n_j = sqrt(n2[j]) is the RMS).
    fisher[key] = mean over calibration sequences of (d NLL / d W)^2, NLL = the
                  mean next-token loss of one sequence.
    The model is left exactly as it was: no optimizer step, hooks removed,
    requires_grad restored, gradients cleared.
    """
    import torch
    seqs = calib_sequences(C, tok, n_seq, n_tok)
    mods = {k: C._resolve(model, k) for k in keys}
    dev = next(model.parameters()).device
    n2 = {k: torch.zeros(mods[k].weight.shape[1], dtype=torch.float64, device=dev) for k in keys}
    fisher = ({k: torch.zeros(mods[k].weight.shape, dtype=torch.float32, device=dev)
               for k in keys} if want_fisher else None)

    def mk(k):
        def hook(_mod, inp):
            x = inp[0].detach().float()
            n2[k].add_((x * x).sum(dim=tuple(range(x.dim() - 1))).double())
        return hook

    hooks = [mods[k].register_forward_pre_hook(mk(k)) for k in keys]
    was = [q.requires_grad for q in model.parameters()]
    n_tokens = 0
    try:
        if want_fisher:
            for q in model.parameters():
                q.requires_grad_(False)
            for k in keys:
                mods[k].weight.requires_grad_(True)
        for ids in seqs:
            x = torch.tensor([ids], device=dev)
            if want_fisher:
                loss = model(input_ids=x, labels=x).loss
                loss.float().backward()
                for k in keys:
                    g = mods[k].weight.grad
                    fisher[k].add_(g.float() ** 2)
                    mods[k].weight.grad = None
            else:
                with torch.no_grad():
                    model(input_ids=x)
            n_tokens += x.numel()
    finally:
        for h in hooks:
            h.remove()
        for q, w in zip(model.parameters(), was):
            q.requires_grad_(w)
        model.zero_grad(set_to_none=True)
    out = dict(n2={k: (n2[k] / n_tokens).cpu() for k in keys},
               fisher=({k: (fisher[k] / len(seqs)).cpu() for k in keys} if want_fisher else None),
               n_seq=len(seqs), n_tok=n_tok, n_tokens=n_tokens)
    del n2, fisher
    torch.cuda.empty_cache()
    return out


FISHER_DAMP = 0.01          # sqrt(F) + FISHER_DAMP * mean(sqrt(F)) of the tensor


def act_scores(v_t, arm, n2_t, fisher_t):
    """Selection score of one tensor (same shape as v_t). Exact zeros of v never win."""
    import torch
    a = v_t.abs().float()
    if arm == "S-fisher":
        d = fisher_t.float().sqrt()
        score = a / (d + FISHER_DAMP * float(d.mean()))
    else:
        n = n2_t.float().sqrt().clamp_min(1e-12)[None, :]
        score = a / n if arm == "S-act1" else a * n
    return score.masked_fill(v_t == 0, float("-inf"))


def build_act_arm(v, arm, seed, density, ref_scales, stats):
    """E6b arms. C-ref's per-tensor count k_t and scale s_t are kept; only WHICH
    k_t entries of each tensor are used changes (top-k_t by the arm's score);
    signs are the true signs. -> (E, meta)."""
    import torch
    import v8_edits as V8
    if stats is None or (arm == "S-fisher" and stats.get("fisher") is None):
        raise RuntimeError(f"{arm} needs activation statistics that were not computed")
    ref = V8._masks(v, "C-ref", density, seed)
    E, kept, tot = {}, 0, 0
    for k in v:
        kt = int(ref[k].sum().item())
        score = act_scores(v[k], arm, stats["n2"][k],
                           stats["fisher"][k] if stats.get("fisher") is not None else None)
        flat = score.reshape(-1)
        m = torch.zeros(flat.numel(), dtype=torch.bool)
        if kt > 0:
            m[torch.topk(flat, kt, sorted=False).indices] = True
        m = m.view(v[k].shape)
        E[k] = torch.sign(v[k]) * m.float() * ref_scales[k]
        kept += int((m & ref[k]).sum().item())
        tot += kt
        del score, flat
    n_flips = int(sum(int((e != 0).sum().item()) for e in E.values()))
    n_total = int(sum(e.numel() for e in E.values()))
    fro = float(torch.sqrt(sum((e.double() ** 2).sum() for e in E.values())))
    meta = dict(condition=arm, density=density, n_flips=n_flips, n_params=n_total,
                effective_sparsity=1.0 - n_flips / max(n_total, 1), frobenius=fro,
                kind="act", support="top-k_t of each tensor by the arm's score",
                values="sign", scale="C-ref per-tensor scales",
                overlap_with_C_ref=(kept / tot if tot else None),
                fisher_damp=(FISHER_DAMP if arm == "S-fisher" else None))
    del ref
    return E, meta


def e6a_diag(v, stats, density, seed, ref_scales):
    """E6a: is C-ref's support on high-activation input channels? (a function of
    v and the activation statistics alone; no edit is applied)."""
    import collections
    import v8_edits as V8
    from v8_support import parse_module
    m_ref = V8._masks(v, "C-ref", density, seed)
    m_a = V8._masks(v, "C-a", density, seed)
    S = dict(all_n2=0.0, all_cnt=0.0, ref_n2=0.0, ref_k=0, a_n2=0.0, a_k=0,
             ref_within=0.0, a_within=0.0, h_ref=0.0, h_a=0.0)
    proj = collections.defaultdict(lambda: dict(all_n2=0.0, all_cnt=0.0, ref_n2=0.0, ref_k=0,
                                                a_n2=0.0, a_k=0))
    for k in v:
        n2 = stats["n2"][k].double()
        rows = v[k].shape[0]
        c_ref, c_a = m_ref[k].sum(0).double(), m_a[k].sum(0).double()
        sr, sa = float((c_ref * n2).sum()), float((c_a * n2).sum())
        kr, ka = int(c_ref.sum().item()), int(c_a.sum().item())
        mean_all = float(n2.mean())
        st = float(ref_scales[k])
        S["all_n2"] += float(n2.sum()) * rows
        S["all_cnt"] += n2.numel() * rows
        S["ref_n2"] += sr
        S["ref_k"] += kr
        S["a_n2"] += sa
        S["a_k"] += ka
        if mean_all > 0:
            S["ref_within"] += sr / mean_all          # = k_t * e_t
            S["a_within"] += sa / mean_all
        S["h_ref"] += st * st * sr
        S["h_a"] += st * st * sa
        q = proj[parse_module(k)[1]]
        q["all_n2"] += float(n2.sum()) * rows
        q["all_cnt"] += n2.numel() * rows
        q["ref_n2"] += sr
        q["ref_k"] += kr
        q["a_n2"] += sa
        q["a_k"] += ka
    mean_all = S["all_n2"] / S["all_cnt"]
    by_proj = {}
    for name, q in proj.items():
        ma = q["all_n2"] / q["all_cnt"]
        by_proj[name] = dict(
            mean_n2=ma, mean_n2_over_global=ma / mean_all,
            share_of_entries=q["all_cnt"] / S["all_cnt"],
            share_of_C_ref_support=q["ref_k"] / max(S["ref_k"], 1),
            enrich_C_ref_within=((q["ref_n2"] / q["ref_k"]) / ma if q["ref_k"] and ma > 0 else None),
            enrich_C_a_within=((q["a_n2"] / q["a_k"]) / ma if q["a_k"] and ma > 0 else None))
    del m_ref, m_a
    return dict(
        enrich_C_ref=(S["ref_n2"] / S["ref_k"]) / mean_all,
        enrich_C_a=(S["a_n2"] / S["a_k"]) / mean_all,
        enrich_C_ref_within_tensor=S["ref_within"] / S["ref_k"],
        enrich_C_a_within_tensor=S["a_within"] / S["a_k"],
        h_proxy_C_ref=S["h_ref"], h_proxy_C_a=S["h_a"],
        h_proxy_ratio=(S["h_ref"] / S["h_a"] if S["h_a"] > 0 else None),
        k_C_ref=S["ref_k"], k_C_a=S["a_k"], mean_n2=mean_all, by_projection=by_proj)


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
        if not cfg["elicit_missing"]:
            raise RuntimeError(f"corpus missing: {corpus_fp} (this panel elicits nothing new)")
        # The published route for a cell whose corpus is not on this box: the
        # self-designer elicits it (scripts/run_unit._ensure_corpus), inference only,
        # in a separate load before the target is loaded for the unit.
        sys.path.insert(0, os.path.join(HERE, "scripts"))
        import run_unit as RU
        RU._ensure_corpus(dict(u))
        if not os.path.exists(corpus_fp):
            raise RuntimeError(f"elicitation did not produce {corpus_fp}")

    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)
    mmlu_cal = RO.load_mmlu_calib(200, seed=0)
    wt_cal = C.load_wikitext(n_chunks=20, seed=0, split="validation")
    if set(r["question"] for r in mmlu_cal) & set(r["question"] for r in mmlu):
        raise RuntimeError("calibration and evaluation MMLU items overlap")
    if set(wt_cal) & set(wt):
        raise RuntimeError("calibration and evaluation WikiText chunks overlap")
    mmlu1k = None
    if any(a in cfg["mmlu1k_arms"] for a in todo):
        # run_opsel's MMLU-1000 reading: a superset of the evaluation items,
        # read at already-selected points, never a selection criterion.
        mmlu1k = C.load_mmlu(n=1000, seed=0)
        if [r["question"] for r in mmlu1k[:200]] != [r["question"] for r in mmlu]:
            raise RuntimeError("load_mmlu(1000)[:200] != load_mmlu(200)")

    t0 = time.time()
    model, tok = C.load_model(hf, dispatch=False)
    try:
        # Baselines BEFORE the first training pass, as run_dec / run_opsel do.
        pre, pre_ll = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
        pre_cal = RO.collateral_eval(model, tok, mmlu_cal, wt_cal)
        pre_cal_c = RS.collateral_only(pre_cal)
        pre_1k = RO.mmlu1k_eval(model, tok, mmlu1k) if mmlu1k is not None else None

        v, losses = RO.train_contrast(model, tok, dn, ax, s)
        g = RO.geometry_of(v)

        # ---- replay guards against the registered alpha* panel ----
        ref_panel = cfg["ref_panel"]
        # results/<panel> first; a panel shipped back from another box is often
        # dropped at the repository root (v12big/), so look there too.
        ref_dir = next((d for d in (os.path.join(C.RESULTS, ref_panel),
                                    os.path.join(HERE, ref_panel))
                        if os.path.exists(os.path.join(d, "geometry.jsonl"))), None)
        if ref_dir is None:
            print(f"[r1] NOTE: reference panel {ref_panel!r} is not on this box "
                  f"(looked in {C.RESULTS}/{ref_panel} and ./{ref_panel}); the replay "
                  "guards are recorded as None for this unit", flush=True)
            ref_dir = os.path.join(C.RESULTS, ref_panel)
        ref_geo = [r for r in _rows(os.path.join(ref_dir, "geometry.jsonl"))
                   if _key(r) == _key(u)]
        ref_sha = ref_geo[-1]["geometry"]["geom_sha"] if ref_geo else None
        ref_tr = [r for r in _rows(os.path.join(ref_dir, "alpha_trace.jsonl"))
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

        stats = None
        if any(a in ACT_ARMS for a in todo):
            # E6: activation statistics on the unedited model, then proof that the
            # model is unchanged (same skew, MMLU and perplexity as before training)
            tq = time.time()
            stats = act_stats(C, model, tok, sorted(v), cfg["act_seqs"], cfg["act_tokens"],
                              want_fisher="S-fisher" in todo)
            again, _ = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
            same = all(again[k] == pre[k] for k in ("skew", "mmlu_acc", "ppl"))
            if not same:
                raise RuntimeError(f"baseline changed after the statistics pass: {again} vs {pre}")
            if not any(_key(r) == _key(u) for r in _rows(os.path.join(out_dir, "e6a.jsonl"))):
                _append(os.path.join(out_dir, "e6a.jsonl"), dict(
                    key, **e6a_diag(v, stats, density, s, ref_scales),
                    n_seq=stats["n_seq"], n_tok=stats["n_tok"], fisher=stats["fisher"] is not None,
                    baseline_unchanged_after_stats=bool(same)))
            print(f"[r1] {fam} {ax} s{s} activation statistics: {stats['n_seq']} x "
                  f"{stats['n_tok']} tokens, baseline unchanged={same} "
                  f"({time.time() - tq:.0f}s)", flush=True)

        for arm in todo:
            ta = time.time()
            if not applicable(v, arm):
                _append(os.path.join(out_dir, "not_applicable.jsonl"), dict(
                    key, arm=arm, reason="architecture resolves to <2 attention "
                    "projections; the shuffle would return C-ref unchanged"))
                print(f"[r1] {fam} {ax} s{s} {arm}: N/A (structural)", flush=True)
                continue
            E, meta = build_arm(v, arm, s, density, ref_scales, stats=stats)
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

            grid_arm = arm in cfg["grid_only_arms"]
            grid_only = grid_arm and (base["geom_match"] is True
                                      or not cfg["grid_only_fallback_full"])
            if grid_only:
                # Fixed scales only (E1b): this arm's alpha* is the reference panel's.
                for a in cfg["grid_only_arms"][arm]:
                    measure(a, "grid")
                sel = dict(alpha_star=None, status="grid_only", n_probes=0,
                           last_pass=None, first_fail=None)
            else:
                if grid_arm:
                    print(f"[r1] {fam} {ax} s{s} {arm}: dW is not the reference panel's "
                          f"(geom_match={base['geom_match']}); running the alpha* search "
                          "here so the unit is self-contained", flush=True)
                sel = RS.select(pre_cal_c, collateral_probe, ladder, cfg["refine_steps"])
                if grid_arm:
                    for a in cfg["grid_only_arms"][arm]:
                        if float(a) not in rungs:
                            measure(a, "grid")
            a_star = sel["alpha_star"]

            # ---- curve-only measurements, after and outside the selection ----
            if sel["first_fail"] is not None:
                above = [a for a in ladder if a > sel["first_fail"]]
                for a in above[:cfg["curve_extra_rungs"]]:
                    if a not in rungs:
                        measure(a, "curve_extra")
            frozen = None
            if arm in DEC_ARMS + ACT_ARMS and not grid_only:
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

            # ---- E8: alpha* backoff and budget sensitivity (listed arms only) ----
            def add_1k(alpha):
                rec = rungs[float(alpha)]
                if "mmlu1k" in rec:
                    return
                undo = C.apply_edit(model, E, alpha=float(alpha), sign=-1.0)
                try:
                    p1k = RO.mmlu1k_eval(model, tok, mmlu1k)
                finally:
                    undo()
                rec["mmlu1k"] = RO._k1(pre_1k, p1k,
                                       v9_gate.ppl_ok(pre["ppl"], rec["eval"]["ppl"]))
                _append(os.path.join(out_dir, "alpha_trace.jsonl"), dict(
                    key, arm=arm, split="mmlu1k", rule="adaptive", role=rec["role"],
                    alpha=float(alpha), pre_mmlu=pre_1k, post_mmlu=p1k, n_items=1000,
                    ppl_ratio=rec["eval"]["ppl_ratio"],
                    collateral_ok=rec["mmlu1k"]["ok"]))

            backoff, budgets = None, None
            if arm in cfg["backoff_arms"]:
                if a_star is not None:
                    backoff = []
                    for f in [1.0] + [float(x) for x in cfg["backoff_factors"]]:
                        a = float(a_star * f)
                        if a not in rungs:
                            measure(a, "backoff")
                        if mmlu1k is not None and arm in cfg["mmlu1k_arms"]:
                            add_1k(a)
                        backoff.append(dict(factor=f, alpha=a))
                budgets = {}
                for b in cfg["budget_items_extra"]:
                    def probe_b(alpha, _b=b):
                        rec = rungs.get(float(alpha)) or measure(alpha, f"rung_b{_b}")
                        return {k: rec["calib"][k] for k in ("mmlu_acc", "ppl", "n_items")}
                    sb = RS.select(pre_cal_c, probe_b, ladder, cfg["refine_steps"],
                                   mmlu_budget_items=int(b))
                    d = {k: sb[k] for k in ("alpha_star", "status", "n_probes",
                                            "last_pass", "first_fail")}
                    if sb["alpha_star"] is not None:
                        evb = rungs[sb["alpha_star"]]["eval"]
                        d.update(removal=evb["removal"], post_skew=evb["post_skew"],
                                 dmmlu_items=evb["dmmlu_items"], ppl_ratio=evb["ppl_ratio"],
                                 eval_ok_same_budget=bool(
                                     evb["dmmlu_items"] <= int(b)
                                     and v9_gate.ppl_ok(pre["ppl"], evb["ppl"])))
                    budgets[str(b)] = d

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
                frozen=frozen, adaptive=adaptive, backoff=backoff, budgets=budgets,
                pre_mmlu1k=pre_1k,
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
