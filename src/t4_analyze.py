"""T4 analysis — run HERE after results/t4/ comes back (experiments_v4 O1 + R2).

O1: extend the convergence curve to the 27-72B tier. Δd(tier) = same-family −
    cross-family profile correlation; does the family signature reach 0 at T4?
R2: for each T4 designer's elicited corpora, train the small-target edit
    (qwen1.5b) at attn r16 and measure removal; PRE-REGISTERED PREDICTION: no
    70B-class designer exceeds the CrowS exogenous ceiling (~0). PASS =
    falsifiable-prediction win; FAIL = amend K1 (designer scale substitutes for
    ground truth).
"""
import os, json, glob, itertools
import numpy as np

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
T4_DIR = os.path.join(RESULTS, "t4")
PROF = os.path.join(RESULTS, "profiles")


# ------------------------------------------------------------------ O1
def _prof(dirpath, ckpt, axis):
    p = os.path.join(dirpath, f"{ckpt}|{axis}.npy")
    return np.load(p) if os.path.exists(p) else None


def _corr(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    a, b = a - a.mean(), b - b.mean()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else None


def _agg_d(load, a, b, axes):
    cs = [_corr(load(a, ax), load(b, ax)) for ax in axes]
    cs = [c for c in cs if c is not None]
    return np.mean(cs) if cs else None


def o1():
    from common import FAMILY, SIZE_B
    # T4 checkpoint -> family (extend FAMILY for the new names)
    t4_fam = {"gemma27b": "gemma", "qwen32b": "qwen", "qwen72b": "qwen",
              "llama70b": "llama"}
    t4_size = {"gemma27b": 27, "qwen32b": 32, "qwen72b": 72, "llama70b": 70}
    fam = dict(FAMILY); fam.update(t4_fam)
    size = dict(SIZE_B); size.update(t4_size)

    def load(ck, ax):
        return _prof(os.path.join(T4_DIR, "profiles"), ck, ax) if ck in t4_fam \
            else _prof(PROF, ck, ax)

    # axis set common to all: occ_gender + crows (T1-T3 have these)
    axes = ["occ_gender"] + [os.path.basename(p).split("|")[1][:-4]
                             for p in glob.glob(os.path.join(PROF, "qwen1.5b|crows_*.npy"))]
    t1t3 = sorted(set(os.path.basename(p).split("|")[0]
                      for p in glob.glob(os.path.join(PROF, "*|occ_gender.npy"))))
    t4 = sorted(set(os.path.basename(p).split("|")[0]
                    for p in glob.glob(os.path.join(T4_DIR, "profiles", "*|occ_gender.npy")))) \
        if os.path.isdir(os.path.join(T4_DIR, "profiles")) else []
    allck = t1t3 + t4
    print(f"O1: {len(t1t3)} T1-T3 + {len(t4)} T4 checkpoints; axes={len(axes)}")

    tiers = [("<3.5B", lambda c: size.get(c, 99) < 3.5),
             ("6-9B", lambda c: 6 <= size.get(c, 0) < 12),
             ("27-72B", lambda c: size.get(c, 0) >= 27)]
    print(f"\n{'tier':8s} {'same_d':>8s} {'cross_d':>8s} {'Δd':>8s}  n_same/n_cross")
    for lbl, pred in tiers:
        ms = [c for c in allck if pred(c)]
        same, cross = [], []
        for x, y in itertools.combinations(ms, 2):
            d = _agg_d(load, x, y, axes)
            if d is None:
                continue
            (same if fam.get(x) == fam.get(y) else cross).append(d)
        sd = np.mean(same) if same else float("nan")
        cd = np.mean(cross) if cross else float("nan")
        print(f"{lbl:8s} {sd:8.3f} {cd:8.3f} {sd-cd:8.3f}  {len(same)}/{len(cross)}")
    # cross-scale same-family incl T4 (does lineage survive to 72B?)
    print("\ncross-scale same-family d to a T4 sibling:")
    for c4 in t4:
        f = fam.get(c4)
        smalls = [c for c in t1t3 if fam.get(c) == f]
        ds = [(_agg_d(load, c4, s, axes), s) for s in smalls]
        ds = [(d, s) for d, s in ds if d is not None]
        for d, s in sorted(ds, reverse=True)[:2]:
            print(f"   {c4:10s} vs {s:10s}: d={d:+.3f}")


# ------------------------------------------------------------------ R2
def r2(target="qwen1.5b", axes=("occ_gender", "bbq_Age", "crows_socioeconomic"),
       alphas=(2, 4, 8, 16), batch_size=24):
    """Train the small-target edit from each T4 designer's corpora; measure
    removal; compare to the exogenous ceiling. Requires GPU (small target)."""
    from common import load, free
    from edit import train_task_vector, contrast, binarize, apply_edit
    import evaluate as EV
    designers = sorted(set(os.path.basename(p).split("|")[0]
                           for p in glob.glob(os.path.join(T4_DIR, "designer", "*.json"))))
    print(f"R2: T4 designers found: {designers}")
    if not designers:
        print("  no T4 designer corpora yet"); return
    data = EV.get_eval_data("fast")
    model, tok = load(target)
    rows = []
    try:
        for axis in axes:
            pre = EV.fast_eval(model, tok, axis, data, batch_size)
            key0 = EV.primary_metric(axis)[0]
            for des in designers:
                fp = os.path.join(T4_DIR, "designer", f"{des}|{axis}.json")
                if not os.path.exists(fp):
                    continue
                c = json.load(open(fp))
                dwb, _ = train_task_vector(model, tok, c["biased"], rank=16, steps=250,
                                           lr=1e-4, seed=0,
                                           targets=["q_proj", "k_proj", "v_proj", "o_proj"])
                dwd, _ = train_task_vector(model, tok, c["debiased"], rank=16, steps=250,
                                           lr=1e-4, seed=0,
                                           targets=["q_proj", "k_proj", "v_proj", "o_proj"])
                E, _ = binarize(contrast(dwb, dwd), "per_tensor", 0.0, 0)
                best = float("nan")
                for a in alphas:
                    undo = apply_edit(model, E, alpha=a, sign=-1.0)
                    try:
                        post = EV.fast_eval(model, tok, axis, data, batch_size)
                    finally:
                        undo()
                    if EV.collateral_ok(pre, post):
                        red = EV.bias_reduction(pre, post, axis)
                        best = red if np.isnan(best) else max(best, red)
                rows.append(dict(axis=axis, designer=des, pre=abs(pre[key0]),
                                 best_removal=best))
                print(f"  {axis:22s} designer={des:10s} removal(in-budget)={best:+.4f}", flush=True)
    finally:
        free(model, tok)
    json.dump(rows, open(os.path.join(RESULTS, "experiment_r2.json"), "w"), indent=2)
    # The ceiling is NOT a fixed constant — it is the MEASURED exogenous
    # (ground-truth-corpus) removal for the SAME axis/target/config. The
    # prediction fails only if a designer removes materially MORE than the
    # exogenous ground truth (i.e. designer scale substitutes for ground truth).
    exo_fp = os.path.join(RESULTS, "exo_baseline_qwen1.5b.json")
    exo = {r["axis"]: r["exo_removal"] for r in json.load(open(exo_fp))} \
        if os.path.exists(exo_fp) else {}
    print("\nR2 verdict: does any designer EXCEED the measured exogenous ceiling?")
    crows = [r for r in rows if r["axis"].startswith("crows_")]
    for r in crows:
        c = exo.get(r["axis"])
        d = r["best_removal"]
        if c is None:
            print(f"  {r['axis']}: designer={d:+.4f} (no matched exo baseline)"); continue
        margin = d - c
        verdict = "FAIL (designer beats ground truth; amend K1)" if margin > 0.05 \
            else "PASS (designer <= exogenous ceiling)"
        print(f"  {r['axis']}: designer={d:+.4f} exo={c:+.4f} margin={margin:+.4f} -> {verdict}")


if __name__ == "__main__":
    import sys
    o1()
    if "--r2" in sys.argv:
        r2()
    else:
        print("\n(run with --r2 to also train R2 edits on the small target)")
