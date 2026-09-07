"""Experiment D — robustness / reviewer-facing controls (experiments_v2 §D).

D1 and D2 here are computed from data already on disk (records + cached
direction sketches), so they run on CPU while the Experiment B sweep has the
GPU. D3 (second injected axis) and D4 (elicitation robustness) need new runs
and live in run_expD_gpu.sh.

D1  Capability-matched designers. The bidirectional flip already argues
    against "cross-family is just a stronger model", but §D asks for the
    explicit matched-capability table: restrict designers to an MMLU band and
    confirm the origin x designer_family interaction survives.

D2  Layer/subspace locus. design.md §12 asks whether ACQUIRED and INHERITED
    bias occupy separable subspaces. If they do, that mechanistically explains
    why one designer removes one and not the other. Computed as the cosine
    between the inherited-axis and acquired-axis contrast directions for the
    SAME target -- the one cross-direction comparison that is well defined,
    because both directions were trained on the same model and therefore share
    a basis (see geometry.py for why cross-MODEL cosines are not).
"""
import os, json, glob, argparse, itertools
import numpy as np
import pandas as pd

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
SKETCH_DIR = os.path.join(RESULTS, "sketches")

# measured base capability (MMLU, this harness) used for the D1 band
DESIGNER_MMLU = {
    "qwen1.5b": 0.655, "qwen3b": 0.700, "qwen0.5b": 0.458,
    "llama1b": 0.420, "llama3b": 0.600, "gemma2b": 0.520,
    "gemma9b": 0.700, "smol1.7b": 0.560, "smol360m": 0.320,
    "phi3.5": 0.660, "phi3mini": 0.660,
}


# ------------------------------------------------------------------ D1
def d1_capability_matched(df, band=(0.50, 0.72)):
    """Interaction restricted to designers inside an MMLU band."""
    from analyze import primary_interaction
    d = df.copy()
    d["designer_mmlu"] = d["designer"].map(DESIGNER_MMLU)
    keep = d[(d.designer_mmlu >= band[0]) & (d.designer_mmlu <= band[1])
             | (d.designer_role.isin(["random", "gt"]))]
    out = {"band": band,
           "designers_kept": sorted(set(keep.designer.dropna()) - {"random", "ground_truth"}),
           "designers_dropped": sorted(set(d.designer.dropna())
                                       - set(keep.designer.dropna()))}
    try:
        out["interaction"] = primary_interaction(keep)
    except Exception as e:
        out["interaction"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        out["interaction_all"] = primary_interaction(df)
    except Exception as e:
        out["interaction_all"] = {"error": str(e)}
    return out


# ------------------------------------------------------------------ D2
def _load_sketches():
    out = {}
    for p in glob.glob(os.path.join(SKETCH_DIR, "*.npy")):
        name = os.path.basename(p)[:-4]
        parts = name.split("|")
        if len(parts) == 5:
            arm, seed, origin, signal, role = parts
        elif len(parts) == 4:
            arm, (seed, origin, signal, role) = "qwen", parts
        else:
            continue
        try:
            out[(arm, int(seed), origin, signal, role)] = np.load(p)
        except Exception:
            continue
    return out


def _cos(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else None


def d2_subspace_separability():
    """cosine(inherited direction, acquired direction) for the SAME target.

    Near 0 => the two biases occupy separable subspaces, which is the
    mechanistic explanation design.md §12 asks about: a designer can be blind
    in one subspace while retaining leverage in the other.
    """
    sk = _load_sketches()
    rows = []
    for (arm, seed, origin, signal, role), v in sk.items():
        if origin != "inherited":
            continue
        w = sk.get((arm, seed, "acquired", signal, role))
        if w is None or len(w) != len(v):
            continue
        rows.append(dict(arm=arm, seed=seed, signal=signal, role=role,
                         cos_inh_acq=_cos(v, w)))
    tab = pd.DataFrame(rows)
    if tab.empty:
        return {"error": "no matched inherited/acquired sketch pairs"}
    rng = np.random.default_rng(0)
    vals = tab.cos_inh_acq.dropna().values
    bs = rng.choice(vals, size=(2000, len(vals)), replace=True).mean(axis=1)
    return {"n_pairs": int(len(vals)),
            "mean_cos": float(vals.mean()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "by_role": tab.groupby("role").cos_inh_acq.agg(["mean", "count"]
                                                           ).to_dict("index"),
            "by_arm": tab.groupby("arm").cos_inh_acq.agg(["mean", "count"]
                                                         ).to_dict("index"),
            "separable": bool(abs(vals.mean()) < 0.10)}


def report():
    from analyze import load_runs
    df = load_runs()
    lines = ["=" * 78, "EXPERIMENT D — robustness controls (experiments_v2 §D)", "=" * 78]

    lines.append("\nD1. CAPABILITY-MATCHED DESIGNERS")
    d1 = d1_capability_matched(df)
    lines.append(f"  MMLU band {d1['band']}")
    lines.append(f"  kept    : {d1['designers_kept']}")
    lines.append(f"  dropped : {d1['designers_dropped']}")
    for k in ("interaction_all", "interaction"):
        r = d1.get(k, {})
        if "estimate" in r:
            lines.append(f"  {'ALL designers  ' if k.endswith('all') else 'MATCHED band   '}"
                         f"interaction {r['estimate']:+.4f} "
                         f"CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                         f"p={r.get('pvalue', float('nan')):.4g} n={r.get('n')}")
        else:
            lines.append(f"  {k}: {r}")

    lines.append("\nD2. SUBSPACE SEPARABILITY (inherited vs acquired direction, same target)")
    d2 = d2_subspace_separability()
    if "error" in d2:
        lines.append(f"  {d2['error']}")
    else:
        lines.append(f"  n_pairs={d2['n_pairs']}  mean cos={d2['mean_cos']:+.4f} "
                     f"CI {['%+.4f' % x for x in d2['ci95']]}")
        lines.append(f"  separable (|cos| < 0.10): {d2['separable']}")
        lines.append(f"  by role: { {k: (round(v['mean'], 4), int(v['count'])) for k, v in d2['by_role'].items()} }")
        lines.append(f"  by arm : { {k: (round(v['mean'], 4), int(v['count'])) for k, v in d2['by_arm'].items()} }")
        lines.append("  Interpretation: a near-zero cosine means the acquired bias lives in a "
                     "subspace largely disjoint from the inherited one, so a designer can be "
                     "blind on one axis while retaining leverage on the other (design.md §12).")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "experiment_d.txt"))
    a = ap.parse_args()
    txt = report()
    print(txt)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    print("\nwrote", a.out)
