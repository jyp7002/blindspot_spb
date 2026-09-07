"""K1 (elicitation-gap regression) + HY (CrowS hygiene) — experiments_v3 P0.

K1: does the per-axis endogenous removal track the per-axis elicitation gap?
    If yes, the wholesale CrowS failure is EXPLAINED (elicitation collapses on
    heterogeneous content) and B3's negative converts from scar to mechanism
    support. Uses the attn-target CrowS records (where removal is possible) plus
    the templated axes.

HY: per-axis hygiene on CrowS -- bootstrap CIs, the religion ceiling anomaly,
    the race-color null, and an items-per-half floor as a formal gate.
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
from analyze import load_runs, best_within_budget
import crows_axes as CA

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def per_axis_removal(df, use_attn=True):
    """endogenous cross+same removal and elicitation gap, per axis."""
    rows = []
    for ax, g in df[df.signal == "endogenous"].groupby("axis"):
        if use_attn and str(ax).startswith("crows_"):
            g = g[g.key.astype(str).str.contains("Tq\\+k")]
        elif not str(ax).startswith("crows_"):
            g = g[~g.key.astype(str).str.contains("Tq\\+k")]
        if not len(g):
            continue
        endo = g[g.family_relation.isin(["same", "cross"])]
        vals = []
        for _, gg in endo.groupby(["arm", "seed", "designer_role"]):
            b = best_within_budget(gg, ["seed"], "strict")
            if b is not None and len(b):
                vals.append(b.bias_reduction.mean())
        gap = endo.elicit_contrast_gap.dropna()
        if not vals or not len(gap):
            continue
        rows.append(dict(axis=ax, removal=float(np.mean(vals)),
                         gap=float(gap.mean()),
                         templated=(not str(ax).startswith("crows_")),
                         n=len(vals)))
    return pd.DataFrame(rows)


def k1():
    df = load_runs()
    t = per_axis_removal(df)
    print("=== K1: per-axis endogenous removal vs elicitation gap ===")
    print(t.sort_values("gap").to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    if len(t) >= 4:
        r = stats.pearsonr(t.gap, t.removal)
        rho = stats.spearmanr(t.gap, t.removal)
        print(f"\n  Pearson r={r.statistic:+.3f} p={r.pvalue:.4f}  "
              f"Spearman rho={rho.statistic:+.3f} p={rho.pvalue:.4f}  (n={len(t)})")
        print(f"  K1 (b_elicit > 0, CI excludes 0): "
              f"{'PASS' if r.statistic > 0 and r.pvalue < 0.05 else 'WEAK/FAIL'}")
    return t


def hy():
    df = load_runs()
    print("\n=== HY: CrowS per-axis hygiene (attn targets) ===")
    d = df[(df.axis.astype(str).str.startswith("crows_")) &
           (df.key.astype(str).str.contains("Tq\\+k")) &
           (df.signal == "endogenous")]
    rng = np.random.default_rng(0)
    rows = []
    for ax, g in d.groupby("axis"):
        pre = abs(g.pre_crows_skew.dropna()).mean()
        n_half = len(CA.load_axis(ax)[0])
        cell = {}
        for fam in ("same", "cross"):
            vv = []
            for _, gg in g[g.family_relation == fam].groupby(["arm", "seed", "designer_role"]):
                b = best_within_budget(gg, ["seed"], "strict")
                if b is not None and len(b):
                    vv.append(b.bias_reduction.mean())
            cell[fam] = np.array(vv)
        if len(cell["cross"]) and len(cell["same"]):
            gap = cell["cross"].mean() - cell["same"].mean()
            allv = np.concatenate([cell["cross"], cell["same"]])
            bs = rng.choice(cell["cross"], (2000, len(cell["cross"]))).mean(1) - \
                 rng.choice(cell["same"], (2000, len(cell["same"]))).mean(1)
            rows.append(dict(axis=ax.replace("crows_", ""), pre_bias=pre,
                             items_per_half=n_half,
                             cross_minus_same=gap,
                             ci_lo=np.percentile(bs, 2.5),
                             ci_hi=np.percentile(bs, 97.5),
                             floor_ok=n_half >= 40))
    t = pd.DataFrame(rows).sort_values("pre_bias", ascending=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\n  items-per-half floor = 40 (calibrated to the CrowS set; the plan's")
    print("  ~100 target is unreachable for most CrowS axes, so we set it at the")
    print("  point where the split still yields a stable profile).")
    starved = t[~t.floor_ok]
    if len(starved):
        print(f"  GATED OUT (item-starved): {list(starved.axis)}")
    print(f"  pooled cross-same over floor-passing axes: ", end="")
    ok = t[t.floor_ok]
    print(f"{ok.cross_minus_same.mean():+.4f} (n_axes={len(ok)})")
    return t


if __name__ == "__main__":
    k1_t = k1()
    hy_t = hy()
    with open(os.path.join(RESULTS, "experiment_k1_hy.json"), "w") as f:
        json.dump(dict(k1=k1_t.to_dict("records"), hy=hy_t.to_dict("records")), f, indent=2)
    print("\nwrote results/experiment_k1_hy.json")
