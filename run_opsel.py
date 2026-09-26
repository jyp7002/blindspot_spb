"""OPSEL (experiments_v12) — sparsity/binarization curve + adaptive operating point.

ONE TRAINING PASS PER UNIT, EVERY VARIANT PAIRED. Like run_spc.py and
run_dec.py, the two task vectors are trained once per (target, axis, seed,
designer) and every variant is built from the same contrast vector v, so
variants differ only in construction. The training call is byte-identical to
run_spc.py / run_dec.py (rank 16, 250 steps, lr 1e-4, bs 8, C.ATTN, grad
checkpointing), which is what lets the calibration panel REPLAY the published
SPC curves and lets `s0.99` at seed 0 reproduce the v11big C-ref cells.

VARIANTS (names are the SPC panel's, so the replay is row-comparable):
    fp          the dense full-precision contrast v itself
    s0.0        dense one-bit (sign, one scale per tensor)
    s<sp>       one-bit at sparsity sp  (colab_t2t4.binarize, natural scale)
    pstar       one-bit at sparsity 1 - p*, p* PREDICTED from geometry (held-out
                cells only; refused without a frozen prediction, see v12_opsel)
    C-a@0.01    DEC's random-support arm at 1%, ref_tensor scale -- so
                Delta_selection can be re-read under the adaptive-α rule
    C-seed@0.01 v13: a random 1% support regenerated from a seed (a keyed
                integer permutation, src/v13_seed.py), true signs, ref_tensor
                scale. Ships as a payload-only patch.

v13 ADDITIONS (all off unless the config names them).
    patch_variants   encode each listed variant as a patch file at its selected
                     α (α* where it exists), decode it, rebuild, and require
                     bit-identity with the in-memory edit; record measured bytes
                     (src/v13_patch.py). C-seed -> seed patch, else index patch.
    ifeval_variants  IFEval at α* on the listed variants, for seeds in
                     ifeval_seeds, with the unedited baseline measured once per
                     unit BEFORE training (as every other baseline here).
    deterministic    set by scripts/run_unit.py before CUDA initialises.

TWO SELECTIONS PER VARIANT, BOTH PERSISTED.
    frozen      the published rule: max removal over in-budget α in {2,4,8,16},
                gate and removal on the EVALUATION split. Comparable to every
                published number.
    adaptive    α* = first-failure max α with collateral within budget on the
                CALIBRATION split (MMLU validation split + WikiText validation
                split: no item shared with evaluation), no bias probe read; then
                ONE evaluation-split measurement at α*. See src/v12_opsel.py.

PHASES.
    geometry    train ΔW, record its |ΔW| geometry (src/v12_geometry.py), stop.
                No forward pass on any probe. This is what a held-out cell runs
                before its p* is predicted.
    full        everything above. Re-derives the geometry and refuses to go on
                if the retrained ΔW does not reproduce the phase-A fingerprint.

MMLU-1000. For variants listed in MMLU1K, the selected operating points
(frozen and adaptive) are additionally scored on 1000 MMLU items whose first
200 ARE the evaluation items (load_mmlu's round-robin is prefix-stable; this is
asserted). The 1000-item gate is round(0.02*1000) = 20 items. This is a
robustness READING of already-selected points, never a selection criterion:
the registered gate stays at 200 (experiments_v11.md §II).

Resumable at (target, axis, seed, designer, variant).
"""
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C          # noqa: E402
import v8_edits as V8           # noqa: E402
import v9_gate                  # noqa: E402
import v12_geometry as G        # noqa: E402
import v12_opsel as S           # noqa: E402

# ---- set per unit by scripts/run_unit.py from the panel config ----
CELLS = []                      # [(target, hf, designer, axis, designer_hf)]
SEEDS = [0]
PHASE = "full"
SPARSITIES = list(G.SPC_SPARSITIES)
EXTRA_VARIANTS = ["fp"]         # any of: fp, pstar, C-a@0.01
ADAPTIVE = ["fp", "s0.0", "s0.99", "pstar", "C-a@0.01"]
MMLU1K = []                     # variants whose operating points get MMLU-1000
MMLU1K_ALL_ALPHAS = False       # also score MMLU-1000 at EVERY frozen alpha, so
                                # the 1000-item argmax can be recomputed, not
                                # just the 200-item choice re-read
ALPHAS = (2, 4, 8, 16)          # frozen grid -- the published estimand
LADDER = list(S.ALPHA_LADDER)
REFINE = S.REFINE_STEPS
PSTAR_FILE = None
PATCH_VARIANTS = []             # v13: measured patch files
IFEVAL_VARIANTS = []            # v13: IFEval at alpha*
IFEVAL_SEEDS = [0]
IFEVAL_LIMIT = 200
DETERMINISTIC = False           # recorded on every row; set by run_unit
MMLU_N = 200
CALIB_N = 200
BATCH = 6                       # run_dec / run_spc fast_eval batch

PANEL = os.environ.get("OPSEL_PANEL", "v12opsel_smoke")
OUT = os.path.join(C.RESULTS, PANEL)


# ------------------------------------------------------------------ data ----
def load_mmlu_calib(n, seed=0):
    """MMLU VALIDATION split, round-robin over subjects exactly like load_mmlu.

    The evaluation items come from the test split, so calibration and
    evaluation share no item by construction rather than by bookkeeping.
    """
    from datasets import load_dataset
    d = load_dataset("cais/mmlu", "all")["validation"]
    rows = [dict(question=r["question"], options=list(r["choices"]),
                 gold=int(r["answer"]), subject=r["subject"]) for r in d]
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r)
    subs = sorted(by)
    for s in subs:
        rng.shuffle(by[s])
    out, i = [], 0
    while len(out) < n:
        added = False
        for s in subs:
            if i < len(by[s]):
                out.append(by[s][i]); added = True
                if len(out) == n:
                    break
        if not added:
            break
        i += 1
    if len(out) != n:
        raise RuntimeError(f"MMLU validation split has only {len(out)} items")
    return out


def collateral_eval(model, tok, mmlu_rows, wt_texts, batch_size=BATCH):
    """fast_eval minus the bias probe: the same MMLU and perplexity calls.

    Kept call-for-call identical to colab_t2t4.fast_eval (n_rot=2, perplexity
    batch max(bs//8,1)) so a calibration probe measures exactly what the
    evaluation gate measures, on different items.
    """
    m = C.eval_mmlu(model, tok, mmlu_rows, batch_size=batch_size, n_rot=2)["mmlu_acc"]
    p = C.eval_perplexity(model, tok, wt_texts,
                          batch_size=max(batch_size // 8, 1))["ppl"]
    return dict(mmlu_acc=m, ppl=p, n_items=len(mmlu_rows))


def mmlu1k_eval(model, tok, rows, batch_size=BATCH):
    return C.eval_mmlu(model, tok, rows, batch_size=batch_size, n_rot=2)["mmlu_acc"]


# ----------------------------------------------------------------- edits ----
def variant_names(pstar_on):
    names = [f"s{sp}" for sp in SPARSITIES]
    extra = [e for e in EXTRA_VARIANTS if e != "pstar" or pstar_on]
    return extra + names


def build(v, name, seed, p_star=None):
    """-> (E, meta). Never mutates v (every later variant reuses it)."""
    if name == "fp":
        return v, dict(kind="fp", n_flips=sum(t.numel() for t in v.values()),
                       n_params=sum(t.numel() for t in v.values()),
                       effective_sparsity=0.0, sparsity=None)
    if name == "pstar":
        sp = 1.0 - float(p_star)
        E, meta = C.binarize(v, "per_tensor", sp, seed)
        meta.update(kind="pstar", sparsity=sp)
        return E, meta
    if name.startswith("C-seed@"):
        # C-seed@<density>: v13_seed support at that density, true signs,
        # ref_tensor scale of C-ref AT THE SAME DENSITY (as C-a@0.01 at 1%)
        import v13_seed
        d = float(name.split("@", 1)[1])
        rs = V8.ref_scale_table(v, d, seed)
        E, meta = v13_seed.build_cseed(v, d, seed, rs)
        meta.update(kind="C-seed", sparsity=1.0 - d)
        return E, meta
    if name == "C-a@0.01":
        rs = V8.ref_scale_table(v, 0.01, seed)
        E, meta = V8.build_edit(v, "C-a", 0.01, seed, scale_mode="ref_tensor",
                                ref_scales=rs)
        meta.update(kind="C-a", sparsity=0.99)
        return E, meta
    if name.startswith("s"):
        sp = float(name[1:])
        E, meta = C.binarize(v, "per_tensor", sp, seed)      # non-inplace
        meta.update(kind="sign", sparsity=sp)
        return E, meta
    raise ValueError(name)


# ------------------------------------------------------------------ rows ----
def _append(fname, row):
    with open(os.path.join(OUT, fname), "a") as f:
        f.write(json.dumps(row) + "\n")


def _rows(fname):
    fp = os.path.join(OUT, fname)
    out = []
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def _gate_detail(pre, post):
    n = pre.get("n_items", MMLU_N)
    drop = (v9_gate.items_from_acc(pre["mmlu_acc"], n)
            - v9_gate.items_from_acc(post["mmlu_acc"], n))
    return dict(dmmlu_items=int(drop), n_items=n,
                ppl_ratio=post["ppl"] / pre["ppl"])


def _k1(pre1k, post1k, ppl_ok):
    """The MMLU-1000 reading of one operating point."""
    n = 1000
    drop = (v9_gate.items_from_acc(pre1k, n) - v9_gate.items_from_acc(post1k, n))
    mm_ok = v9_gate.mmlu_ok(pre1k, post1k, n_items=n)
    return dict(pre=pre1k, post=post1k, n_items=n, dmmlu_items=int(drop),
                budget_items=v9_gate.allowed_drop_items(n),
                mmlu_ok=bool(mm_ok), ok=bool(mm_ok and ppl_ok))


# ------------------------------------------------------------------ main ----
def train_contrast(model, tok, dn, ax, s):
    corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
    dwb, lb = C.train_task_vector(model, tok, corp["biased"], rank=16, steps=250,
                                  lr=1e-4, seed=s, bs=8, targets=C.ATTN,
                                  grad_checkpoint=True)
    dwd, ld = C.train_task_vector(model, tok, corp["debiased"], rank=16, steps=250,
                                  lr=1e-4, seed=s, bs=8, targets=C.ATTN,
                                  grad_checkpoint=True)
    v = C.contrast_(dwb, dwd)
    return v, dict(loss_biased=lb, loss_debiased=ld)


def geometry_of(v):
    keys = sorted(v)
    return G.geometry((v[k].reshape(-1).numpy() for k in keys), module_keys=keys)


def main():
    os.makedirs(OUT, exist_ok=True)
    geo_rows = _rows("geometry.jsonl")
    done = {(r["target"], r["axis"], r["seed"], r["designer"], r["variant"])
            for r in _rows("removal.jsonl")}

    need_eval = PHASE == "full"
    if need_eval:
        mmlu = C.load_mmlu(n=MMLU_N, seed=0)
        wt = C.load_wikitext(n_chunks=20, seed=0)
        mmlu_cal = load_mmlu_calib(CALIB_N, seed=0)
        wt_cal = C.load_wikitext(n_chunks=20, seed=0, split="validation")
        if set(r["question"] for r in mmlu_cal) & set(r["question"] for r in mmlu):
            raise RuntimeError("calibration and evaluation MMLU items overlap")
        if set(wt_cal) & set(wt):
            raise RuntimeError("calibration and evaluation WikiText chunks overlap")
        mmlu1k = None
        if MMLU1K:
            mmlu1k = C.load_mmlu(n=1000, seed=0)
            if [r["question"] for r in mmlu1k[:MMLU_N]] != [r["question"] for r in mmlu]:
                raise RuntimeError(
                    "load_mmlu(1000)[:200] != load_mmlu(200): the 1000-item set "
                    "is not a superset of the evaluation items; the 200-vs-1000 "
                    "comparison would be between different probes")

    by_model = {}
    for fam, hf, dn, ax, _dhf in CELLS:
        by_model.setdefault((fam, hf), []).append((dn, ax))

    for (fam, hf), specs in by_model.items():
        pstar_on = "pstar" in EXTRA_VARIANTS
        todo = []
        for dn, ax in specs:
            for s in SEEDS:
                have_geo = any(r["target"] == fam and r["axis"] == ax
                               and r["seed"] == s and r["designer"] == dn
                               for r in geo_rows)
                if PHASE == "geometry":
                    if not have_geo:
                        todo.append((dn, ax, s))
                elif any((fam, ax, s, dn, vn) not in done
                         for vn in variant_names(pstar_on)):
                    todo.append((dn, ax, s))
        if not todo:
            print(f"[opsel] {fam}: nothing to do (phase={PHASE})", flush=True)
            continue

        model, tok = C.load_model(hf, dispatch=False)
        try:
            pre_cache, pre_cal, pre_1k = {}, None, None
            for dn, ax, s in todo:
                try:
                    # Baselines BEFORE the first training pass, as run_dec and
                    # run_spc do: train_task_vector leaves the model with
                    # gradient checkpointing enabled and in whatever mode peft's
                    # unload restores, so a baseline taken after it would not be
                    # the baseline the published panels compared against.
                    if PHASE == "full":
                        if ax not in pre_cache:
                            pre_cache[ax] = C.fast_eval(model, tok, ax, mmlu, wt,
                                                        BATCH)
                        if pre_cal is None:
                            pre_cal = collateral_eval(model, tok, mmlu_cal, wt_cal)
                        if MMLU1K and pre_1k is None:
                            pre_1k = mmlu1k_eval(model, tok, mmlu1k)
                        if IFEVAL_VARIANTS and s in IFEVAL_SEEDS \
                                and ("ifeval_base", s) not in pre_cache:
                            pre_cache[("ifeval_base", s)] = _ifeval(model, tok)
                    v, losses = train_contrast(model, tok, dn, ax, s)
                    g = geometry_of(v)
                    prior = [r for r in geo_rows if r["target"] == fam
                             and r["axis"] == ax and r["seed"] == s
                             and r["designer"] == dn]
                    if prior and prior[-1]["geometry"]["geom_sha"] != g["geom_sha"]:
                        raise RuntimeError(
                            f"retrained ΔW does not reproduce: geom_sha "
                            f"{g['geom_sha']} vs recorded "
                            f"{prior[-1]['geometry']['geom_sha']}. Every paired "
                            "claim in this panel assumes one ΔW per unit.")
                    if not prior:
                        row = dict(panel=PANEL, target=fam, axis=ax, seed=s,
                                   designer=dn, phase=PHASE, geometry=g, **losses)
                        _append("geometry.jsonl", row)
                        geo_rows.append(row)
                    print(f"[opsel] {fam} {ax} s{s} geometry sha={g['geom_sha']} "
                          f"M(1%)={g['mass'][G._pkey(0.01)]:.4f} "
                          f"pr={g['pr_frac']:.4f} gini={g['gini']:.4f}", flush=True)
                    if PHASE == "geometry":
                        del v
                        continue

                    p_star, pred_sha = None, None
                    if pstar_on:
                        p_star, pred_sha = S.load_pstar(PSTAR_FILE, fam, ax, s,
                                                        g["geom_sha"])
                        print(f"[opsel] {fam} {ax} s{s} p*={p_star:.5f} "
                              f"(prediction {pred_sha[:12]})", flush=True)

                    pre = pre_cache[ax]

                    for vn in variant_names(pstar_on):
                        if (fam, ax, s, dn, vn) in done:
                            continue
                        run_variant(model, tok, v, vn, fam, ax, s, dn, g, p_star,
                                    pred_sha, pre, pre_cal, pre_1k, mmlu, wt,
                                    mmlu_cal, wt_cal, mmlu1k,
                                    ifeval_base=pre_cache.get(("ifeval_base", s)),
                                    hf=hf)
                        done.add((fam, ax, s, dn, vn))
                    del v
                except S.HeldOutViolation:
                    raise                               # never swallowed
                except Exception:
                    print(f"[opsel] FAIL {fam} {ax} s{s} {dn}", flush=True)
                    traceback.print_exc()
        finally:
            C._free(model)
    print("[opsel] ALL DONE", flush=True)


def _ifeval(model, tok):
    """run_ins.ifeval, the function behind the published IFEval arm."""
    if "run_ins" not in sys.modules:
        argv, sys.argv = sys.argv, ["run_ins.py", "A"]   # ARM is read at import
        try:
            import run_ins  # noqa: F401
        finally:
            sys.argv = argv
    return sys.modules["run_ins"].ifeval(model, tok, IFEVAL_LIMIT)


def _patch(model, E, vn, seed, alpha, hf, density):
    """Measured patch file for one variant (v13_patch). Never raises silently:
    a patch that does not rebuild the in-memory edit bit-for-bit stops the unit."""
    import v13_patch as P
    import v13_seed
    rev = str(getattr(model.config, "_commit_hash", None) or "unknown")
    if vn.startswith("C-seed@"):
        density = float(vn.split("@", 1)[1])     # exactly what build() used
        sup = v13_seed.support({k: t.numel() for k, t in E.items()}, density, seed)
        _raw, st = P.measure(E, P.SEED, model_id=hf, revision=rev, seed=seed,
                             density=density, alpha=alpha, positions_by_name=sup)
    else:
        _raw, st = P.measure(E, P.INDEX, model_id=hf, revision=rev, seed=seed,
                             density=density, alpha=alpha)
    if not st["identical"]:
        raise RuntimeError(f"patch for {vn} does not rebuild the edit bit-for-bit")
    st["alpha"] = alpha
    return st


def run_variant(model, tok, v, vn, fam, ax, s, dn, g, p_star, pred_sha,
                pre, pre_cal, pre_1k, mmlu, wt, mmlu_cal, wt_cal, mmlu1k,
                ifeval_base=None, hf=None):
    E, meta = build(v, vn, s, p_star)
    key = dict(panel=PANEL, target=fam, axis=ax, seed=s, designer=dn, variant=vn)

    def at(alpha, fn):
        undo = C.apply_edit(model, E, alpha=alpha, sign=-1.0)
        try:
            return fn()
        finally:
            undo()

    # ---- frozen: the published selection, on the evaluation split ----
    frozen, frozen_post, frozen_1k_all = [], {}, {}
    want_1k_all = vn in MMLU1K and MMLU1K_ALL_ALPHAS
    for al in ALPHAS:
        def _ev():
            post = C.fast_eval(model, tok, ax, mmlu, wt, BATCH)
            return post, (mmlu1k_eval(model, tok, mmlu1k) if want_1k_all else None)
        post, p1k = at(al, _ev)
        ok = C.collateral_ok(pre, post)
        r = C.bias_reduction(pre, post)
        frozen.append((float(al), ok, r))
        frozen_post[float(al)] = post
        if p1k is not None:
            k1 = _k1(pre_1k, p1k, v9_gate.ppl_ok(pre["ppl"], post["ppl"]))
            frozen_1k_all[float(al)] = k1
            _append("alpha_trace.jsonl", dict(
                key, split="mmlu1k", rule="frozen", alpha=float(al),
                pre_mmlu=pre_1k, post_mmlu=p1k, n_items=1000,
                ppl_ratio=post["ppl"] / pre["ppl"], collateral_ok=k1["ok"],
                bias_reduction=float(r)))
        _append("alpha_trace.jsonl", dict(
            key, split="eval", rule="frozen", alpha=float(al),
            pre_mmlu=pre["mmlu_acc"], post_mmlu=post["mmlu_acc"],
            n_items=pre.get("n_items", MMLU_N), pre_ppl=pre["ppl"],
            post_ppl=post["ppl"], ppl_ratio=post["ppl"] / pre["ppl"],
            pre_skew=pre["skew"], post_skew=post["skew"],
            collateral_ok=bool(ok), bias_reduction=float(r)))
    best, best_a = S.frozen_argmax(frozen)

    # ---- adaptive: α* on calibration, then one evaluation measurement ----
    adaptive = None
    if vn in ADAPTIVE:
        def probe(al):
            post = at(al, lambda: collateral_eval(model, tok, mmlu_cal, wt_cal))
            ok = C.collateral_ok(pre_cal, post)
            _append("alpha_trace.jsonl", dict(
                key, split="calib", rule="adaptive", alpha=float(al),
                pre_mmlu=pre_cal["mmlu_acc"], post_mmlu=post["mmlu_acc"],
                n_items=pre_cal["n_items"], pre_ppl=pre_cal["ppl"],
                post_ppl=post["ppl"], ppl_ratio=post["ppl"] / pre_cal["ppl"],
                collateral_ok=bool(ok)))
            return ok
        sel = S.select_alpha(probe, LADDER, REFINE)
        adaptive = {k: sel[k] for k in ("alpha_star", "status", "n_probes",
                                        "last_pass", "first_fail")}
        if sel["alpha_star"] is not None:
            a_star = sel["alpha_star"]

            def _eval_star():
                post = C.fast_eval(model, tok, ax, mmlu, wt, BATCH)
                p1k = mmlu1k_eval(model, tok, mmlu1k) if vn in MMLU1K else None
                return post, p1k
            post, p1k = at(a_star, _eval_star)
            ok = C.collateral_ok(pre, post)
            r = C.bias_reduction(pre, post)
            adaptive.update(removal=float(r), eval_ok=bool(ok),
                            post_skew=post["skew"], **_gate_detail(pre, post))
            if p1k is not None:
                adaptive["mmlu1k"] = _k1(pre_1k, p1k,
                                         v9_gate.ppl_ok(pre["ppl"], post["ppl"]))
            _append("alpha_trace.jsonl", dict(
                key, split="eval", rule="adaptive", alpha=float(a_star),
                pre_mmlu=pre["mmlu_acc"], post_mmlu=post["mmlu_acc"],
                n_items=pre.get("n_items", MMLU_N), pre_ppl=pre["ppl"],
                post_ppl=post["ppl"], ppl_ratio=post["ppl"] / pre["ppl"],
                pre_skew=pre["skew"], post_skew=post["skew"],
                collateral_ok=bool(ok), bias_reduction=float(r)))
            if vn in IFEVAL_VARIANTS and s in IFEVAL_SEEDS:
                adaptive["ifeval"] = at(a_star, lambda: _ifeval(model, tok))
                adaptive["ifeval_base"] = ifeval_base

    # ---- MMLU-1000 reading of the frozen operating point ----
    frozen_1k = None
    if vn in MMLU1K and best_a is not None:
        if best_a in frozen_1k_all:
            frozen_1k = dict(frozen_1k_all[best_a])
        else:
            p1k = at(best_a, lambda: mmlu1k_eval(model, tok, mmlu1k))
            frozen_1k = _k1(pre_1k, p1k,
                            v9_gate.ppl_ok(pre["ppl"], frozen_post[best_a]["ppl"]))
    if frozen_1k_all:
        # the argmax the 1000-item gate itself would have selected
        b1k, a1k = S.frozen_argmax(
            [(a, frozen_1k_all[a]["ok"], r) for a, _ok, r in frozen])
        frozen_1k = dict(frozen_1k or {}, argmax_1k_removal=b1k, argmax_1k_alpha=a1k)

    patch = None
    if vn in PATCH_VARIANTS:
        a_sel = (adaptive or {}).get("alpha_star") or best_a
        sp_ = meta.get("sparsity")
        patch = _patch(model, E, vn, s, float(a_sel) if a_sel else 0.0, hf,
                       1.0 - sp_ if sp_ is not None else 1.0)

    sp = meta.get("sparsity")
    row = dict(key, role="self", sparsity=sp, patch=patch,
               deterministic=bool(DETERMINISTIC),
               p_retained=(None if sp is None else 1.0 - sp),
               removal=best, alpha_frozen=best_a, pre_skew=pre["skew"],
               n_items=pre.get("n_items", MMLU_N),
               n_flips=meta.get("n_flips"), n_params=meta.get("n_params"),
               effective_sparsity=meta.get("effective_sparsity"),
               bytes=meta.get("bytes"), geom_sha=g["geom_sha"],
               adaptive=adaptive, mmlu1k_frozen=frozen_1k,
               p_star=p_star if vn == "pstar" else None,
               pstar_prediction_sha=pred_sha if vn == "pstar" else None)
    _append("removal.jsonl", row)
    a_txt = "-" if not adaptive or adaptive.get("alpha_star") is None else \
        f"α*={adaptive['alpha_star']:.1f} r={adaptive.get('removal', float('nan')):+.4f}" \
        f" eval_ok={adaptive.get('eval_ok')}"
    print(f"[opsel] {fam:8s} {ax:18s} s{s} {vn:9s} frozen={best:+.4f}@{best_a} "
          f"| {a_txt}", flush=True)
    if E is not v:
        del E


if __name__ == "__main__":
    main()
