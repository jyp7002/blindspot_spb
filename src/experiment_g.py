"""Experiment G — pairwise-sharing mediation on the 5x5 panel (experiments_v3 P0).

The free decisive test. Is the self-penalty FULLY EXPLAINED by designer-target
direction co-encoding, or is there a residual relational component?

For every (designer checkpoint i, target checkpoint j) cell of the panel:
  d_ij      = item-profile correlation between i and j on the templated axis
              (pre-edit, from saved profiles)
  elicit_i  = designer i's elicitation gap on the axis
  same_ij   = 1 if i and j are the same family
  removal_ij= measured in-budget removal from the panel

Model:  removal ~ b_d*d_ij + b_e*elicit_i + b_s*same_ij + (1|target) + (1|seed)

  G1  b_d < 0, 95% CI excludes 0           (co-encoding lowers removal)
  G2  b_s becomes ns once d_ij is in the model, AND the bootstrap indirect
      effect (same -> d -> removal) CI excludes 0   (MEDIATION, primary)
  G3  d(smol1.7b, qwen1.5b) is in the top quartile of cross-family pairs
      (predicting the -0.290 anomaly from geometry alone)
"""
import os, json, itertools
import numpy as np
import pandas as pd
from common import FAMILY

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
PROF_DIR = os.path.join(RESULTS, "profiles")
AXIS = "occ_gender"


def load_profile(ckpt, axis=AXIS):
    p = os.path.join(PROF_DIR, f"{ckpt}|{axis}.npy")
    return np.load(p) if os.path.exists(p) else None


def prof_corr(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    a, b = a - a.mean(), b - b.mean()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else None


def build_cells(budget="strict"):
    """One row per (designer checkpoint, target arm, seed) from the panel."""
    from analyze import load_runs, best_within_budget
    df = load_runs()
    d = df[(df.origin == "inherited") & (df.axis == "occ_gender") &
           (df.signal == "endogenous") &
           (df.family_relation.isin(["same", "cross"])) &
           (~df.key.astype(str).str.contains("Tq\\+k"))]
    tgt_of = {"qwen": "qwen1.5b", "llama": "llama1b", "gemma": "gemma2b",
              "phi": "phi3.5", "smol": "smol1.7b"}
    rows = []
    for (arm, role), g in d.groupby(["arm", "designer_role"]):
        des = g.designer.iloc[0]
        tgt = tgt_of.get(arm)
        d_ij = prof_corr(load_profile(des), load_profile(tgt))
        gap = g.elicit_contrast_gap.dropna()
        gap = float(gap.mean()) if len(gap) else np.nan
        for seed, gs in g.groupby("seed"):
            b = best_within_budget(gs, ["seed"], budget)
            if b is None or not len(b):
                continue
            rows.append(dict(target=arm, designer=des, seed=int(seed),
                             d_ij=d_ij, elicit=gap,
                             same=int(FAMILY.get(des) == FAMILY.get(tgt)),
                             removal=float(b.bias_reduction.mean())))
    return pd.DataFrame(rows)


def main():
    tab = build_cells().dropna(subset=["d_ij", "elicit", "removal"])
    print(f"G: {len(tab)} cells "
          f"({tab.target.nunique()} targets x {tab.designer.nunique()} designers x "
          f"{tab.seed.nunique()} seeds)\n")
    print("cell table (mean over seeds):")
    piv = tab.groupby(["target", "designer", "same"]).agg(
        d_ij=("d_ij", "first"), elicit=("elicit", "first"),
        removal=("removal", "mean")).reset_index()
    print(piv.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    if smf is None or len(tab) < 8:
        print("\ninsufficient data / statsmodels missing")
        return

    # --- G1: sharing coefficient, full model ---
    full = smf.mixedlm("removal ~ d_ij + elicit + same", tab,
                       groups=tab["target"]).fit(disp=False)
    ci = full.conf_int()
    print("\n=== full model: removal ~ d_ij + elicit + same  (1|target) ===")
    for term in ["d_ij", "elicit", "same"]:
        print(f"  {term:8s} {full.params[term]:+.4f}  "
              f"CI [{ci.loc[term,0]:+.4f}, {ci.loc[term,1]:+.4f}]  p={full.pvalues[term]:.4f}")
    g1 = full.params["d_ij"] < 0 and ci.loc["d_ij", 1] < 0
    print(f"  G1 (b_d < 0, CI excludes 0): {'PASS' if g1 else 'FAIL'}")

    # --- G2: does `same` survive once d_ij enters? mediation ---
    red = smf.mixedlm("removal ~ elicit + same", tab, groups=tab["target"]).fit(disp=False)
    print("\n=== reduced (no d_ij): same coef ===")
    ci_r = red.conf_int()
    print(f"  same     {red.params['same']:+.4f}  CI [{ci_r.loc['same',0]:+.4f}, "
          f"{ci_r.loc['same',1]:+.4f}]  p={red.pvalues['same']:.4f}")
    same_full_ns = full.pvalues["same"] > 0.05
    # bootstrap indirect effect (same -> d -> removal), resample by (target,seed)
    rng = np.random.default_rng(0)
    keys = list(tab.groupby(["target", "seed"]).groups.values())
    ind = []
    for _ in range(2000):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([np.asarray(keys[k]) for k in pick])
        bt = tab.loc[idx]
        if bt.d_ij.nunique() < 2:
            continue
        try:
            a_path = smf.ols("d_ij ~ same", bt).fit().params["same"]
            b_path = smf.ols("removal ~ d_ij + elicit + same", bt).fit().params["d_ij"]
            ind.append(a_path * b_path)
        except Exception:
            continue
    ind = np.array(ind)
    lo, hi = np.percentile(ind, [2.5, 97.5])
    indirect_sig = lo > 0 or hi < 0
    print(f"\n  bootstrap indirect effect (same->d->removal): "
          f"{ind.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  same becomes ns with d_ij in model: {same_full_ns} "
          f"(p={full.pvalues['same']:.3f})")
    g2 = same_full_ns and indirect_sig
    print(f"  G2 MEDIATION: {'PASS' if g2 else 'FAIL'}")

    # --- G3: litmus cell ---
    cross = tab[tab.same == 0]
    q75 = cross.d_ij.quantile(0.75)
    sq = tab[(tab.designer == "smol1.7b") & (tab.target == "qwen")]
    if len(sq):
        dv = sq.d_ij.iloc[0]
        g3 = dv >= q75
        print(f"\n=== G3 litmus: d(smol1.7b, qwen1.5b) = {dv:+.3f}; "
              f"cross-family 75th pctile = {q75:+.3f} -> {'PASS' if g3 else 'FAIL'} "
              f"(smol->qwen removal was the -0.290 anomaly)")

    print(f"\nSPINE: G2 {'PASS -> mechanism = direction co-encoding' if g2 else 'FAIL -> relational component beyond sharing; H make-or-break'}")
    with open(os.path.join(RESULTS, "experiment_g.json"), "w") as f:
        json.dump(dict(n=len(tab), g1=bool(g1), g2=bool(g2),
                       b_d=float(full.params["d_ij"]),
                       b_d_ci=[float(ci.loc["d_ij", 0]), float(ci.loc["d_ij", 1])],
                       same_full_p=float(full.pvalues["same"]),
                       same_reduced_p=float(red.pvalues["same"]),
                       indirect=[float(lo), float(hi)]), f, indent=2)


if __name__ == "__main__":
    main()
