"""O4 — G revisited with MULTI-AXIS aggregate pairwise d (experiments_v4 Stage 0).

G (v3) failed: single-axis pairwise d_ij (occ_gender only) did not predict
removal (r=+0.02 over cross pairs), so the self-penalty looked like a
family-AGGREGATE property with no pairwise correlate. O4 tests whether that was
a measurement-bandwidth artifact: replace the single-axis d_ij with the mean
item-profile correlation over ALL saved axes (occ_gender + 9 CrowS), and re-run
the G mediation regression.

  O4 PASS: aggregate-pairwise coefficient < 0, CI excludes 0, in the G model
           => the self-penalty IS pairwise co-encoding, just under-measured by
           one axis. Upgrade the mechanism wording.
  O4 FAIL: keep "family-aggregate, not pairwise" as stated.

Free: profiles already on disk (O0 harvest); no GPU.
"""
import os, json, glob, itertools
import numpy as np
import pandas as pd
from common import FAMILY

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
PROF = os.path.join(RESULTS, "profiles")


def _axes_for(ckpt):
    return [os.path.basename(p).split("|")[1][:-4]
            for p in glob.glob(os.path.join(PROF, f"{ckpt}|*.npy"))]


def load_prof(ckpt, axis):
    p = os.path.join(PROF, f"{ckpt}|{axis}.npy")
    return np.load(p) if os.path.exists(p) else None


def prof_corr(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    a, b = a - a.mean(), b - b.mean()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else None


def aggregate_d(ci, cj):
    """Mean item-profile correlation between checkpoints i and j over shared axes."""
    axes = set(_axes_for(ci)) & set(_axes_for(cj))
    cs = [prof_corr(load_prof(ci, ax), load_prof(cj, ax)) for ax in axes]
    cs = [c for c in cs if c is not None]
    return (float(np.mean(cs)), len(cs)) if cs else (None, 0)


def build_cells(budget="strict"):
    from analyze import load_runs, best_within_budget
    df = load_runs()
    d = df[(df.origin == "inherited") & (df.axis == "occ_gender") &
           (df.signal == "endogenous") &
           (df.family_relation.isin(["same", "cross"])) &
           (~df.key.astype(str).str.contains("Tq\\+k"))]
    tgt = {"qwen": "qwen1.5b", "llama": "llama1b", "gemma": "gemma2b",
           "phi": "phi3.5", "smol": "smol1.7b"}
    rows = []
    for (arm, role), g in d.groupby(["arm", "designer_role"]):
        des = g.designer.iloc[0]
        tj = tgt.get(arm)
        d_agg, n_ax = aggregate_d(des, tj)
        d_single = prof_corr(load_prof(des, "occ_gender"), load_prof(tj, "occ_gender"))
        gap = g.elicit_contrast_gap.dropna()
        gap = float(gap.mean()) if len(gap) else np.nan
        for seed, gs in g.groupby("seed"):
            b = best_within_budget(gs, ["seed"], budget)
            if b is None or not len(b):
                continue
            rows.append(dict(target=arm, designer=des, seed=int(seed),
                             d_single=d_single, d_agg=d_agg, n_axes=n_ax,
                             elicit=gap,
                             same=int(FAMILY.get(des) == FAMILY.get(tj)),
                             removal=float(b.bias_reduction.mean())))
    return pd.DataFrame(rows)


def main():
    tab = build_cells().dropna(subset=["d_agg", "d_single", "elicit", "removal"])
    print(f"O4: {len(tab)} cells, aggregate d over {int(tab.n_axes.median())} axes\n")
    # cross-family only correlation, single vs aggregate
    from scipy import stats
    cr = tab[tab.same == 0]
    for col in ["d_single", "d_agg"]:
        r = stats.pearsonr(cr[col], cr.removal)
        print(f"  cross-family corr({col:8s}, removal) = {r.statistic:+.3f} (p={r.pvalue:.3f})")
    if smf is None or len(tab) < 8:
        print("insufficient data"); return
    full = smf.mixedlm("removal ~ d_agg + elicit + same", tab,
                       groups=tab["target"]).fit(disp=False)
    ci = full.conf_int()
    print("\n=== G model with AGGREGATE d: removal ~ d_agg + elicit + same (1|target) ===")
    for t in ["d_agg", "elicit", "same"]:
        print(f"  {t:8s} {full.params[t]:+.4f}  CI [{ci.loc[t,0]:+.4f}, {ci.loc[t,1]:+.4f}]  p={full.pvalues[t]:.4f}")
    o4 = full.params["d_agg"] < 0 and ci.loc["d_agg", 1] < 0
    same_ns = full.pvalues["same"] > 0.05
    print(f"\n  O4 (d_agg<0, CI excludes 0): {'PASS' if o4 else 'FAIL'}")
    print(f"  `same` now ns with d_agg in model: {same_ns} (p={full.pvalues['same']:.3f})")
    print(f"  VERDICT: {'aggregate pairwise co-encoding IS the mechanism (G failure was measurement-bandwidth)' if o4 else 'family-aggregate stands; pairwise (even multi-axis) does not predict'}")
    json.dump(dict(n=len(tab), d_agg_coef=float(full.params["d_agg"]),
                   d_agg_ci=[float(ci.loc["d_agg", 0]), float(ci.loc["d_agg", 1])],
                   d_agg_p=float(full.pvalues["d_agg"]),
                   same_p=float(full.pvalues["same"]), o4_pass=bool(o4)),
              open(os.path.join(RESULTS, "experiment_o4.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
