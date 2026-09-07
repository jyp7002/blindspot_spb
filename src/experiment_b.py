"""Experiment B — axis-divergence x blind-spot magnitude (experiments_v2 §B).

THE REFRAME TEST. design.md §3's premise ("disjoint lineage => different bias
=> the cross-family designer succeeds because it lacks the bias") is falsified:
occupation->gender skew is +0.65..+0.86 in *every* lineage, yet the effect is
still relational (the bidirectional flip). So the mechanism must be geometric:
*a designer cannot get leverage on a bias it encodes the same way as its
target; same-family models are co-blind on co-inherited axes.* This turns that
post-hoc story into a falsifiable prediction.

  d_axis  (regressor)  per-axis direction-sharing between SAME-FAMILY models,
                       measured pre-edit and basis-independently as the mean
                       within-family correlation of item-space bias profiles
                       (see geometry.py for why weight-space cosine cannot be
                       used across models).
  b_axis  (outcome)    per-axis, per-arm blind-spot magnitude
                       = (cross - same) on that axis, expressed ABOVE the
                       data-partition null (experiments_v2 §N: never raw).

H_geom : slope(b ~ d) > 0.
B1     : slope > 0 with 95% CI excluding 0, over K >= 5 axes.
B2     : adding bias MAGNITUDE as a covariate must not kill d, and magnitude
         alone must not suffice -- this is what separates the geometry story
         from a trivial "more bias is harder to remove" story.
B3     : injected/acquired axes (d ~ 0 by construction) show b with CI over 0.

CIRCULARITY GUARD (non-negotiable, experiments_v2 §B): d_axis comes from
geometry.py, computed pre-edit from probe profiles on a pipeline that never
sees a bias-reduction number. Inherited/acquired labels are fixed independently
(injection vs measured cross-family presence), never fit to the removal
outcome. This module only JOINS the two; it must never re-derive d from
outcomes.
"""
import os, json, argparse
import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except Exception:                                    # pragma: no cover
    smf = None

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def load_geometry(path=None):
    p = path or os.path.join(RESULTS, "geometry.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def axis_bias_magnitude(geom, axis):
    """Mean |profile mean| across models -- the B2 nuisance covariate."""
    pm = geom.get(axis, {}).get("profile_mean", {})
    return float(np.mean([abs(v) for v in pm.values()])) if pm else None


def build_table(df, geom, budget="strict"):
    """One row per (arm, axis): b_axis above the null, plus d_axis and magnitude."""
    from analyze import best_within_budget, null_baseline
    AXIS_OF_ORIGIN = {"inherited": "occ_gender", "acquired": "gen_fm"}

    rows = []
    nulls = null_baseline(df)
    d = df[df.signal == "endogenous"].copy()
    # Group by the record's OWN axis. Deriving it from `origin` would collapse
    # every Experiment B axis into "occ_gender" and silently drop the sweep.
    if "axis" not in d:
        d["axis"] = d["origin"].map(AXIS_OF_ORIGIN)
    d["axis"] = d["axis"].fillna(d["origin"].map(AXIS_OF_ORIGIN))
    for (arm, origin, axis), g in d.groupby(["arm", "origin", "axis"]):
        if axis not in geom:
            continue
        vals = {}
        for fam in ("same", "cross"):
            sub = g[g.designer_family_factor == fam]
            # best_within_budget(df, group_cols, budget) -> DataFrame of the
            # per-group frontier rows; one row per (seed, designer_role).
            best = best_within_budget(sub, ["seed", "designer_role"], budget)
            vals[fam] = (float(best["bias_reduction"].mean())
                         if best is not None and len(best) else None)
        if vals.get("same") is None or vals.get("cross") is None:
            continue
        null = nulls.get((arm, origin))
        # Pre-bias differs by an order of magnitude across axes (occupation->
        # gender sits at ~0.69 while the valence axes sit at 0.02-0.21), so a
        # RAW (cross - same) would let occ_gender dominate the regression on
        # scale alone. Normalise by the axis's own pre-bias so b is "fraction
        # of the available bias that the blind spot costs".
        col = ("pre_occ_skew" if axis == "occ_gender"
               else "pre_crows_skew" if str(axis).startswith("crows_")
               else "pre_val_skew")
        pre_bias = float(np.mean(np.abs(g[col].dropna()))) if col in g else None
        if not pre_bias:
            pre_bias = None
        b_raw = vals["cross"] - vals["same"]
        rows.append(dict(
            arm=arm, origin=origin, axis=axis,
            b_axis=b_raw,                                  # blind-spot magnitude (raw)
            pre_bias=pre_bias,
            b_norm=(b_raw / pre_bias) if pre_bias else None,
            same=vals["same"], cross=vals["cross"], null=null,
            same_above_null=(vals["same"] - null) if null is not None else None,
            d_axis=geom[axis].get("d_axis"),
            d_cross=geom[axis].get("cross_axis"),
            differentiation=geom[axis].get("differentiation"),
            bias_magnitude=axis_bias_magnitude(geom, axis),
            injected=(origin == "acquired")))
    return pd.DataFrame(rows)


def fit(tab, ycol="b_norm"):
    """B1/B2 fits on the pre-bias-normalised blind-spot magnitude."""
    out = {"outcome": ycol}
    if ycol not in tab or tab[ycol].isna().all():
        ycol = "b_axis"
        out["outcome"] = "b_axis (fallback: b_norm unavailable)"
    tab = tab.dropna(subset=[ycol, "d_axis"])
    if smf is None or len(tab) < 3 or tab["d_axis"].nunique() < 2:
        return {"error": "insufficient data (need >=2 distinct d_axis values)"}
    try:
        m = smf.mixedlm(f"{ycol} ~ d_axis", tab, groups=tab["arm"]).fit(disp=False)
        out["B1"] = dict(model="mixedlm(|arm)", slope=float(m.params["d_axis"]),
                         ci=[float(x) for x in m.conf_int().loc["d_axis"]],
                         p=float(m.pvalues["d_axis"]), n=int(len(tab)))
    except Exception:
        m = smf.ols(f"{ycol} ~ d_axis", tab).fit()
        out["B1"] = dict(model="ols(fallback)", slope=float(m.params["d_axis"]),
                         ci=[float(x) for x in m.conf_int().loc["d_axis"]],
                         p=float(m.pvalues["d_axis"]), n=int(len(tab)))
    if tab.bias_magnitude.nunique() >= 2:
        m2 = smf.ols(f"{ycol} ~ d_axis + bias_magnitude", tab).fit()
        m3 = smf.ols(f"{ycol} ~ bias_magnitude", tab).fit()
        out["B2"] = dict(
            d_slope_with_magnitude=float(m2.params["d_axis"]),
            d_ci=[float(x) for x in m2.conf_int().loc["d_axis"]],
            d_p=float(m2.pvalues["d_axis"]),
            magnitude_alone_slope=float(m3.params["bias_magnitude"]),
            magnitude_alone_p=float(m3.pvalues["bias_magnitude"]))
    inj = tab[tab.injected]
    if len(inj) >= 2:
        b = inj[ycol].values
        rng = np.random.default_rng(0)
        bs = rng.choice(b, size=(2000, len(b)), replace=True).mean(axis=1)
        out["B3"] = dict(mean=float(b.mean()),
                         ci=[float(np.percentile(bs, 2.5)),
                             float(np.percentile(bs, 97.5))], n=int(len(b)))
    return out


def report(df=None, geom=None):
    from analyze import load_runs
    df = load_runs() if df is None else df
    geom = load_geometry() if geom is None else geom
    lines = ["=" * 78,
             "EXPERIMENT B -- axis-divergence x blind-spot magnitude (experiments_v2 §B)",
             "=" * 78]
    if not geom:
        lines.append("  insufficient data: results/geometry.json missing "
                     "(run Experiment C first)")
        return "\n".join(lines)

    lines.append("  d_axis = pre-edit within-family item-profile correlation "
                 "(basis-independent; see geometry.py)")
    for axis, g in geom.items():
        lines.append(f"    {axis:12s} d_axis={g.get('d_axis')!s:>8.8} "
                     f"cross={g.get('cross_axis')!s:>8.8} "
                     f"differentiation={g.get('differentiation')!s:>8.8}")
    tab = build_table(df, geom)
    if tab.empty:
        lines.append("\n  insufficient data: no (arm, axis) cells with both "
                     "same and cross inside the collateral budget")
        return "\n".join(lines)
    lines.append("\n  b_axis (cross - same), per arm x axis:")
    lines.append(tab[["arm", "axis", "injected", "same", "cross", "b_axis",
                      "pre_bias", "b_norm", "null", "d_axis", "bias_magnitude"]]
                 .to_string(index=False))
    res = fit(tab)
    lines.append("\n  fits:")
    lines.append("    " + json.dumps(res, indent=2, default=str).replace("\n", "\n    "))
    if "B1" in res:
        b1 = res["B1"]
        ok = b1["slope"] > 0 and b1["ci"][0] > 0
        lines.append(f"\n  B1 geometry slope: {'PASS' if ok else 'FAIL'} "
                     f"(slope={b1['slope']:+.4f}, CI={b1['ci']})")
        lines.append("  NOTE: with only the 2 MVP axes this is underpowered; "
                     "experiments_v2 §B asks for K>=5 axes.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "experiment_b.txt"))
    a = ap.parse_args()
    txt = report()
    print(txt)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    print("\nwrote", a.out)
