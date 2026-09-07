"""Analysis of the 2x2 blind-spot experiment (design.md §5.4, §6, §7, §12).

Everything here is read-only over results/runs.jsonl. No model loading, no
GPU, no network. The file is safe to run while run_experiment.py is still
appending -- truncated trailing lines are skipped, and every statistical
section is individually guarded so a missing cell degrades to
"insufficient data" instead of aborting the report.

What is implemented
-------------------
load_runs               robust JSONL -> flat DataFrame (+ derived columns)
frontier                bias-vs-collateral frontier + full alpha trace (§5.4)
primary_interaction     origin x designer_family mixed-effects term  (§6)
bootstrap_interaction   nonparametric bootstrap over seeds AND designers (§6)
mechanism_test          endogenous vs exogenous signal, H3            (§5.6 ★)
random_null_test        the random-partition null -- the control that
                        licenses every designer-family claim
arm_asymmetry           bidirectional arm control                     (§5.1)
direction_alignment     cosine to the ground-truth direction, from the
                        20k-d sketches in results/sketches
blind_spot_correlation  does elicited contrast predict removal?       (§12)
h1_binary_retention     binary vs full precision at matched COLLATERAL (§7)
kill_switch_report      the pre-registered decision criteria          (§7)
report                  plain-text assembly of all of the above

Sign conventions
----------------
`bias_reduction` is |bias_pre| - |bias_post| (evaluate.py): positive = bias
removed. The interaction is always oriented as

    I = [dBias(acquired, same) - dBias(inherited, same)]
      - [dBias(acquired, cross) - dBias(inherited, cross)]

so that H2 ("same-family fails on inherited bias but not on acquired")
predicts I > 0. The regression contrasts are coded (baselines: origin=
inherited, family=cross) to make the fitted interaction coefficient equal
to this same quantity, so the model and the bootstrap are directly
comparable.
"""
import os
import io
import json
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
RUNS = os.path.join(RESULTS, "runs.jsonl")
SKETCH_DIR = os.path.join(RESULTS, "sketches")
REPORT_PATH = os.path.join(RESULTS, "analysis_report.txt")
DEFAULT_ARM = "qwen"     # records written before the bidirectional arm existed

# design.md §5.4: collateral is a budget, and the budget is swept, never fixed.
BUDGETS = {
    "strict": dict(d_mmlu_min=-0.02, ppl_ratio_max=1.10),
    "loose": dict(d_mmlu_min=-0.05, ppl_ratio_max=1.25),
}
MAIN_VARIANT = "binary-per_tensor-s0.0"
FP_VARIANT = "fp--s0"
# experiments_v2.md §N: the SECOND mandatory null. Same edit pipeline, signs
# shuffled -- it must land at ~0 or the arm's measurement is not trustworthy.
SIGN_NULL_SUFFIX = "-rand"

# design.md §7 (placeholders frozen from the pilot).
TH_INHERITED_RATIO = 0.30     # dBias(inh,same) < 0.3 * dBias(inh,cross)
TH_ACQUIRED_RATIO = 0.80      # dBias(acq,same) > 0.8 * dBias(acq,cross)
TH_INTERACTION_P = 0.01
TH_BINARY_RETENTION = 0.70    # H1 feasibility gate
TH_MMLU_DROP = 0.02           # ~1-2 pt
TH_KILL_CI_WIDTH = 0.10       # "interaction ~ 0 with tight CI"

# experiments_v2.md, Experiment A (placeholders, to be frozen after the first
# two new arms land -- §Z checklist).
TH_SIGN_NULL_FLAT = 0.05      # |mean dBias| of the sign-shuffle null per arm
TH_ARM_FRACTION = 0.6         # A2/A3: >= ceil(0.6 * n_arms)
TH_ARM_CI_EXCLUDES = 2        # A2: CI excludes 0 in >= 2 arms
TH_DEMOTE_MAX_POSITIVE_ARMS = 1   # A_DEMOTE if positive in <= this many arms


# --------------------------------------------------------------------------
# 1. loading
# --------------------------------------------------------------------------
_NEST = {"pre": "pre_", "post": "post_", "edit_meta": "edit_",
         "vec_stats": "vec_", "elicit_diag": "elicit_"}


def _flatten(prefix, obj, out):
    """Recursively flatten a nested dict of scalars into out[prefix+key]."""
    for k, v in obj.items():
        if isinstance(v, dict):
            _flatten(f"{prefix}{k}_", v, out)
        elif isinstance(v, (list, tuple)):
            continue
        else:
            out[f"{prefix}{k}"] = v


def _family_factor(row):
    """same / cross / null / exogenous -- the factor the §6 model uses.

    The raw `designer_family` column holds the designer's lineage name
    ("qwen", "llama", ...); the *factor* is its relation to the target.

      self, sibling   -> same       (shared pretraining lineage)
      cross, cross2   -> cross      (disjoint lineage)
      random          -> null       (random-partition null: same sentence
                                     pool, partition drawn at random, NO
                                     designer -- the control that decides
                                     whether any designer claim is
                                     interpretable at all)
      gt / exogenous  -> exogenous  (ground-truth signal, no designer)
    """
    role = str(row.get("designer_role"))
    if role == "random":
        return "null"
    if str(row.get("signal")) == "exogenous" or role == "gt":
        return "exogenous"
    if role in ("self", "sibling"):
        return "same"
    if role in ("cross", "cross2"):
        return "cross"
    rel = str(row.get("family_relation"))
    return rel if rel in ("same", "cross") else "n/a"


def load_runs(path=None):
    """Robust JSONL load. Skips malformed / truncated lines silently."""
    path = path or RUNS
    recs, bad = [], 0
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if not isinstance(r, dict):
                    bad += 1
                    continue
                flat = {}
                for k, v in r.items():
                    if k in _NEST and isinstance(v, dict):
                        _flatten(_NEST[k], v, flat)
                    elif isinstance(v, (dict, list, tuple)):
                        continue
                    else:
                        flat[k] = v
                recs.append(flat)
    df = pd.DataFrame(recs)
    df.attrs["n_malformed"] = bad
    df.attrs["source"] = path
    if df.empty:
        # still give downstream code the columns it expects
        for c in ("seed", "origin", "signal", "designer_role", "designer",
                  "variant", "alpha", "bias_reduction", "collateral_ok",
                  "pre_mmlu_acc", "post_mmlu_acc", "pre_ppl", "post_ppl",
                  "elicit_contrast_gap", "vec_l2", "axis", "arm"):
            if c not in df.columns:
                df[c] = pd.Series(dtype="float64")
        df["designer_family_factor"] = pd.Series(dtype="object")
        df["d_mmlu"] = pd.Series(dtype="float64")
        df["ppl_ratio"] = pd.Series(dtype="float64")
        return df

    for c in ("post_mmlu_acc", "pre_mmlu_acc", "post_ppl", "pre_ppl",
              "bias_reduction", "alpha", "elicit_contrast_gap", "vec_l2"):
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["d_mmlu"] = df["post_mmlu_acc"] - df["pre_mmlu_acc"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ppl_ratio"] = df["post_ppl"] / df["pre_ppl"]
    df["designer_family_factor"] = df.apply(_family_factor, axis=1)
    # design.md §5.1 bidirectional control; records predating it are arm=qwen
    if "arm" not in df.columns:
        df["arm"] = DEFAULT_ARM
    df["arm"] = (df["arm"].astype("object").where(df["arm"].notna(), DEFAULT_ARM)
                 .replace({"": DEFAULT_ARM, "None": DEFAULT_ARM}).astype(str))
    if "collateral_ok" in df.columns:
        df["collateral_ok"] = df["collateral_ok"].astype("boolean")
    if "designer" not in df.columns:
        df["designer"] = df.get("designer_role", "unknown")
    df["designer"] = df["designer"].fillna("unknown")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return df


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _within_budget(df, budget="strict"):
    b = BUDGETS[budget]
    ok = ((df["d_mmlu"] >= b["d_mmlu_min"])
          & (df["ppl_ratio"] <= b["ppl_ratio_max"]))
    return df[ok.fillna(False)]


def best_within_budget(df, group_cols, budget="strict"):
    """One row per group: the alpha with the largest bias reduction that
    still respects the collateral budget (design.md §5.4)."""
    sub = _within_budget(df, budget)
    sub = sub[sub["bias_reduction"].notna()]
    if sub.empty:
        return sub
    idx = sub.groupby(list(group_cols), dropna=False)["bias_reduction"].idxmax()
    return sub.loc[idx.values]


def _fmt(df, floatfmt="%.4f"):
    if df is None or len(df) == 0:
        return "  (no rows)"
    buf = io.StringIO()
    with pd.option_context("display.max_columns", 60, "display.width", 200,
                           "display.float_format",
                           lambda v: (floatfmt % v) if np.isfinite(v) else "nan"):
        df.to_string(buf, index=False)
    return "\n".join("  " + ln for ln in buf.getvalue().splitlines())


def _ci_from_r(r, n, kind="pearson"):
    """Fisher-z CI. Spearman uses the Bonett-Wright variance inflation."""
    if not np.isfinite(r) or n is None or n < 5 or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    if kind == "spearman":
        se *= math.sqrt(1.0 + r ** 2 / 2.0)
    lo, hi = z - 1.959964 * se, z + 1.959964 * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


def _insuff(msg, **extra):
    d = {"status": "insufficient data", "reason": msg}
    d.update(extra)
    return d


# --------------------------------------------------------------------------
# 2. frontier (design.md §5.4)
# --------------------------------------------------------------------------
def frontier(df, group_cols=("arm", "origin", "designer_family_factor",
                             "designer_role", "variant")):
    """Bias-vs-collateral frontier per group.

    Returns dict with
      "frontier": best bias_reduction per group under each collateral budget
                  (plus the unconstrained best, for reference)
      "trace":    the FULL alpha trace per group -- design.md §5.4 insists the
                  frontier is reported, never a single scalar
      "budgets":  the budget definitions used
    """
    group_cols = [c for c in group_cols if c in df.columns]
    if df is None or df.empty or not group_cols:
        return {"frontier": pd.DataFrame(), "trace": pd.DataFrame(),
                "budgets": BUDGETS,
                "status": "insufficient data"}

    d = df[df["bias_reduction"].notna()].copy()
    if d.empty:
        return {"frontier": pd.DataFrame(), "trace": pd.DataFrame(),
                "budgets": BUDGETS, "status": "insufficient data"}

    # ---- full alpha trace
    g = d.groupby(group_cols + ["alpha"], dropna=False)
    trace = g.agg(
        n=("bias_reduction", "size"),
        dBias=("bias_reduction", "mean"),
        dBias_sd=("bias_reduction", "std"),
        d_mmlu=("d_mmlu", "mean"),
        ppl_ratio=("ppl_ratio", "mean"),
    ).reset_index()
    if "collateral_ok" in d.columns:
        ok = g["collateral_ok"].mean().reset_index(name="frac_collateral_ok")
        trace = trace.merge(ok, on=group_cols + ["alpha"], how="left")
    trace = trace.sort_values(group_cols + ["alpha"])

    # ---- best point under each budget
    rows = []
    for keys, sub in d.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys))
        rec["n_rows"] = len(sub)
        rec["n_alpha"] = sub["alpha"].nunique()
        i = sub["bias_reduction"].idxmax()
        rec["best_any"] = float(sub.loc[i, "bias_reduction"])
        rec["alpha_any"] = float(sub.loc[i, "alpha"])
        for bname in BUDGETS:
            s = _within_budget(sub, bname)
            if s.empty:
                rec[f"best_{bname}"] = float("nan")
                rec[f"alpha_{bname}"] = float("nan")
                rec[f"n_{bname}"] = 0
            else:
                j = s["bias_reduction"].idxmax()
                rec[f"best_{bname}"] = float(s.loc[j, "bias_reduction"])
                rec[f"alpha_{bname}"] = float(s.loc[j, "alpha"])
                rec[f"n_{bname}"] = int(len(s))
        rows.append(rec)
    fr = pd.DataFrame(rows).sort_values(group_cols)
    return {"frontier": fr, "trace": trace, "budgets": BUDGETS,
            "status": "ok"}


# --------------------------------------------------------------------------
# 3. primary inferential quantity (design.md §6)
# --------------------------------------------------------------------------
def primary_interaction(df, variant=MAIN_VARIANT, budget="strict",
                        signal="endogenous", arm=None, re_group=None,
                        unit_cols=("arm", "seed", "origin", "designer_role")):
    """The `origin x designer_family` interaction on bias reduction.

    Restricted to the endogenous signal and the main binary per-tensor
    variant, at a matched collateral budget: within each experimental unit
    (arm x seed x origin x designer_role) the observation is the best bias
    reduction achievable inside the budget, i.e. the frontier point, so the
    contrast is never confounded by different edit strengths.

    Mixed-effects (statsmodels `mixedlm`) with a random intercept per seed --
    or per (arm, seed) when both bidirectional arms are present, since a seed
    means something different in each arm. `arm="qwen"` restricts the fit to
    one arm; the report shows the pooled fit and every per-arm fit.

    Falls back to OLS with cluster-robust SEs, flagged in the result.
    """
    out = {"variant": variant, "budget": budget, "signal": signal,
           "arm": arm or "pooled"}
    if df is None or df.empty:
        return _insuff("no records", **out)

    d = df.copy()
    if signal:
        d = d[d["signal"] == signal]
    if variant:
        d = d[d["variant"] == variant]
    if arm is not None and "arm" in d.columns:
        d = d[d["arm"].astype(str) == str(arm)]
    d = d[d["designer_family_factor"].isin(["same", "cross"])]
    if d.empty:
        return _insuff(f"no rows with signal={signal}, variant={variant}"
                       + (f", arm={arm}" if arm else ""), **out)

    unit_cols = [c for c in unit_cols if c in d.columns]
    used_budget = budget
    units = best_within_budget(d, unit_cols, budget)
    if units.empty or units["origin"].nunique() < 2:
        units = best_within_budget(d, unit_cols, "loose")
        used_budget = "loose (strict budget was empty)"
    if units.empty:
        return _insuff("no (seed, origin, designer) unit meets any collateral "
                       "budget", **out)
    out["budget"] = used_budget

    cells = units.groupby(["origin", "designer_family_factor"],
                          dropna=False).size()
    out["cells"] = {f"{a}/{b}": int(n) for (a, b), n in cells.items()}
    out["n"] = int(len(units))
    if len(cells) < 4:
        return _insuff(f"design incomplete: {len(cells)}/4 origin x family "
                       "cells populated", **out)
    if units["seed"].nunique() < 2:
        out["warning"] = "only one seed -- random intercept is not identified"

    import statsmodels.formula.api as smf

    # baselines chosen so the interaction coefficient equals
    # [acq,same - inh,same] - [acq,cross - inh,cross]  (H2 predicts > 0)
    formula = ("bias_reduction ~ C(origin, Treatment('inherited'))"
               " * C(designer_family_factor, Treatment('cross'))")
    units = units.copy()
    n_arms = units["arm"].nunique() if "arm" in units.columns else 1
    add_arm_fixed = False
    if re_group == "arm" and n_arms > 1:
        # experiments_v2.md Experiment A1: ARM is the random effect, so the
        # pooled estimate generalises to a population of target families
        # rather than to these particular ones.
        units["_group"] = units["arm"].astype(str)
        out["group_var"] = "arm"
    elif n_arms > 1:
        # a seed is a different thing in each arm -> group on the pair, and
        # let the arm shift the intercept as a fixed effect too
        units["_group"] = (units["arm"].astype(str) + ":"
                           + units["seed"].astype(str))
        formula += " + C(arm)"
        add_arm_fixed = True
        out["group_var"] = "(arm, seed)"
    else:
        units["_group"] = units["seed"]
        out["group_var"] = "seed"
    out["n_arms"] = int(n_arms)
    out["re_group"] = re_group or "auto"
    cols = ["bias_reduction", "origin", "designer_family_factor", "seed",
            "_group"] + (["arm"] if add_arm_fixed else [])
    dat = units[cols].dropna()
    if len(dat) < 4:
        return _insuff("fewer than 4 usable observations", **out)

    def _term(res):
        names = [n for n in res.params.index if ":" in n]
        return names[0] if names else None

    res, method, note = None, None, None
    gv = out["group_var"]
    if dat["_group"].nunique() >= 2:
        for opt in ("lbfgs", "bfgs", "powell"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = smf.mixedlm(formula, dat, groups=dat["_group"])
                    r = m.fit(reml=True, method=opt)
                if getattr(r, "converged", True) and _term(r) is not None:
                    res = r
                    method = f"mixedlm (random intercept | {gv}, {opt})"
                    note = None
                    break
                note = f"mixedlm did not converge ({opt})"
            except Exception as e:
                note = f"mixedlm failed ({opt}): {type(e).__name__}: {e}"
    else:
        note = f"single {gv} group -- mixedlm not identifiable"

    if res is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                r = smf.ols(formula, dat).fit(
                    cov_type="cluster", cov_kwds={"groups": dat["_group"]})
                res, method = r, f"OLS + cluster-robust SE (by {gv}) [FALLBACK]"
            except Exception:
                try:
                    r = smf.ols(formula, dat).fit()
                    res, method = (r, "OLS, homoskedastic SE "
                                      "[FALLBACK, clustering failed]")
                except Exception as e2:
                    return _insuff(f"model fit failed: {e2}",
                                   fallback_note=note, **out)

    term = _term(res)
    ci = res.conf_int()
    lo, hi = float(ci.loc[term][0]), float(ci.loc[term][1])
    out.update(
        method=method,
        fallback_note=note,
        formula=formula,
        term=term,
        estimate=float(res.params[term]),
        se=float(res.bse[term]),
        ci95=(lo, hi),
        pvalue=float(res.pvalues[term]),
        n=int(len(dat)),
        n_seeds=int(dat["seed"].nunique()),
        cell_means={f"{a}/{b}": float(v) for (a, b), v in
                    units.groupby(["origin", "designer_family_factor"])
                    ["bias_reduction"].mean().items()},
        status="ok",
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["summary"] = str(res.summary())
    except Exception:
        out["summary"] = ""
    return out


# --------------------------------------------------------------------------
# 4. bootstrap over seeds AND designers
# --------------------------------------------------------------------------
class _CellBootstrap:
    """Bootstrap over seeds and over designers simultaneously.

    Designers are resampled *within* their family factor (same / cross /
    exogenous) -- resampling them from the pooled list would empty whole
    cells and is not the resampling scheme §6 asks for. Cell means are
    recomputed as (sum of sums) / (sum of counts) over the resampled
    (seed, designer) pairs, which is exactly the pooled mean of the
    resampled data and is fully vectorised.
    """

    def __init__(self, units, value_col="bias_reduction",
                 factor_col="designer_family_factor", origin_col="origin",
                 seed_col="seed", designer_col="designer"):
        u = units[[value_col, factor_col, origin_col, seed_col,
                   designer_col]].dropna(subset=[value_col])
        self.ok = len(u) > 0
        if not self.ok:
            self.origins, self.factors = [], []
            return
        self.origins = sorted(u[origin_col].astype(str).unique())
        self.factors = sorted(u[factor_col].astype(str).unique())
        self.seeds = sorted(u[seed_col].dropna().unique())
        self._sums, self._cnts, self._des = {}, {}, {}
        oi = {o: i for i, o in enumerate(self.origins)}
        si = {s: i for i, s in enumerate(self.seeds)}
        for f in self.factors:
            uf = u[u[factor_col].astype(str) == f]
            des = sorted(uf[designer_col].astype(str).unique())
            self._des[f] = des
            di = {d: i for i, d in enumerate(des)}
            S = np.zeros((len(self.seeds), len(des), len(self.origins)))
            C = np.zeros_like(S)
            for _, r in uf.iterrows():
                s = si.get(r[seed_col])
                if s is None:
                    continue
                S[s, di[str(r[designer_col])], oi[str(r[origin_col])]] += r[value_col]
                C[s, di[str(r[designer_col])], oi[str(r[origin_col])]] += 1
            self._sums[f], self._cnts[f] = S, C
        self.observed = {}
        for f in self.factors:
            for o in self.origins:
                c = self._cnts[f][:, :, oi[o]].sum()
                self.observed[(o, f)] = (
                    float(self._sums[f][:, :, oi[o]].sum() / c) if c else float("nan"))

    def draw(self, n_boot=2000, seed=0):
        """-> dict[(origin, factor)] = array of n_boot resampled cell means."""
        rng = np.random.default_rng(seed)
        ns = len(self.seeds)
        si = rng.integers(0, ns, size=(n_boot, ns))
        out = {}
        for f in self.factors:
            nd = len(self._des[f])
            di = rng.integers(0, nd, size=(n_boot, nd))
            S = self._sums[f][si[:, :, None], di[:, None, :], :].sum(axis=(1, 2))
            C = self._cnts[f][si[:, :, None], di[:, None, :], :].sum(axis=(1, 2))
            with np.errstate(divide="ignore", invalid="ignore"):
                M = np.where(C > 0, S / np.where(C == 0, 1, C), np.nan)
            for j, o in enumerate(self.origins):
                out[(o, f)] = M[:, j]
        return out


def _pct_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def _boot_summary(vals):
    v = np.asarray(vals, dtype=float)
    fin = v[np.isfinite(v)]
    lo, hi = _pct_ci(fin)
    n = fin.size
    return dict(mean=float(fin.mean()) if n else float("nan"),
                ci95=(lo, hi),
                frac_gt0=float((fin > 0).mean()) if n else float("nan"),
                frac_lt0=float((fin < 0).mean()) if n else float("nan"),
                n_boot_valid=int(n))


def _units_for_interaction(df, variant=MAIN_VARIANT, budget="strict",
                           signal="endogenous", factors=("same", "cross")):
    d = df.copy()
    if signal:
        d = d[d["signal"] == signal]
    if variant:
        d = d[d["variant"] == variant]
    if factors:
        d = d[d["designer_family_factor"].isin(list(factors))]
    if d.empty:
        return d, budget
    cols = [c for c in ("arm", "seed", "origin", "designer_role", "designer")
            if c in d.columns]
    u = best_within_budget(d, cols, budget)
    used = budget
    if u.empty:
        u = best_within_budget(d, cols, "loose")
        used = "loose (strict empty)"
    return u, used


def bootstrap_interaction(df, n_boot=2000, seed=0, variant=MAIN_VARIANT,
                          budget="strict", signal="endogenous"):
    """Nonparametric bootstrap of

        [mean dBias(acquired, same)  - mean dBias(inherited, same)]
      - [mean dBias(acquired, cross) - mean dBias(inherited, cross)]

    resampling seeds and designers (the latter within family factor).
    """
    out = {"n_boot": n_boot, "variant": variant, "signal": signal}
    if df is None or df.empty:
        return _insuff("no records", **out)
    units, used = _units_for_interaction(df, variant, budget, signal)
    out["budget"] = used
    if units.empty:
        return _insuff("no unit meets a collateral budget", **out)

    have = set(map(tuple, units[["origin", "designer_family_factor"]]
                   .astype(str).drop_duplicates().values))
    need = {("acquired", "same"), ("acquired", "cross"),
            ("inherited", "same"), ("inherited", "cross")}
    out["cells_present"] = sorted("/".join(t) for t in have)
    if not need <= have:
        return _insuff("missing cells: "
                       + ", ".join(sorted("/".join(t) for t in need - have)),
                       **out)

    B = _CellBootstrap(units)
    if not B.ok:
        return _insuff("no usable observations", **out)
    obs = B.observed
    point = ((obs[("acquired", "same")] - obs[("inherited", "same")])
             - (obs[("acquired", "cross")] - obs[("inherited", "cross")]))
    draws = B.draw(n_boot=n_boot, seed=seed)
    vals = ((draws[("acquired", "same")] - draws[("inherited", "same")])
            - (draws[("acquired", "cross")] - draws[("inherited", "cross")]))
    out.update(_boot_summary(vals))
    out["point_estimate"] = float(point)
    out["cell_means"] = {f"{o}/{f}": v for (o, f), v in obs.items()}
    out["n_units"] = int(len(units))
    out["n_seeds"] = int(units["seed"].nunique())
    out["n_designers"] = int(units["designer"].nunique())
    if out["n_seeds"] < 2:
        out["warning"] = ("only one seed -- the seed dimension of the "
                          "bootstrap is degenerate and the CI is too tight")
    # one-sided bootstrap p for H2 (interaction > 0)
    out["p_boot_one_sided"] = float(1.0 - out["frac_gt0"]) if np.isfinite(
        out["frac_gt0"]) else float("nan")
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 5. mechanism test (design.md §5.6 ★ / H3)
# --------------------------------------------------------------------------
def mechanism_test(df, n_boot=2000, seed=0, variant=MAIN_VARIANT,
                   budget="strict"):
    """Endogenous vs exogenous debias signal.

    The exogenous arm has no designer (the signal is ground truth), so the
    literal `origin x designer_family` interaction is undefined there.
    Instead, per origin, we take mean bias reduction for the three signal
    sources {same-family endogenous, cross-family endogenous, exogenous}
    and report the gaps with bootstrap CIs:

      deficit_endo   = cross - same        (the blind spot, endogenous)
      exo_vs_same    = exogenous - same    (does ground truth rescue same?)
      exo_vs_cross   = exogenous - cross   (is ground truth ~ cross-family?)

    H3 predicts the *inherited* row shows deficit_endo > 0 (blind spot),
    exo_vs_same > 0 (ground truth removes what the sibling could not), and
    exo_vs_cross ~ 0 (the blind spot has closed).
    """
    out = {"variant": variant, "n_boot": n_boot}
    if df is None or df.empty:
        return _insuff("no records", **out)

    d = df.copy()
    if variant:
        d = d[d["variant"] == variant]
    d = d[d["designer_family_factor"].isin(["same", "cross", "exogenous"])]
    if d.empty:
        return _insuff(f"no rows for variant={variant}", **out)
    cols = [c for c in ("seed", "origin", "designer_role", "designer")
            if c in d.columns]
    units = best_within_budget(d, cols, budget)
    used = budget
    if units.empty:
        units = best_within_budget(d, cols, "loose")
        used = "loose (strict empty)"
    out["budget"] = used
    if units.empty:
        return _insuff("no unit meets a collateral budget", **out)

    tab = units.groupby(["origin", "designer_family_factor"],
                        dropna=False)["bias_reduction"].agg(
        ["size", "mean", "std"]).reset_index()
    tab = tab.rename(columns={"size": "n", "mean": "dBias", "std": "sd"})
    out["table"] = tab

    B = _CellBootstrap(units)
    if not B.ok:
        return _insuff("no usable observations", **out)
    draws = B.draw(n_boot=n_boot, seed=seed)
    obs = B.observed

    gaps, gap_defs = {}, [("deficit_endo", "cross", "same"),
                          ("exo_vs_same", "exogenous", "same"),
                          ("exo_vs_cross", "exogenous", "cross")]
    for origin in B.origins:
        for name, a, b in gap_defs:
            if (origin, a) not in draws or (origin, b) not in draws:
                gaps[f"{origin}:{name}"] = _insuff(f"missing cell {a} or {b}")
                continue
            g = _boot_summary(draws[(origin, a)] - draws[(origin, b)])
            g["point_estimate"] = float(obs[(origin, a)] - obs[(origin, b)])
            g["definition"] = f"mean dBias({origin},{a}) - mean dBias({origin},{b})"
            gaps[f"{origin}:{name}"] = g
    out["gaps"] = gaps
    out["cell_means"] = {f"{o}/{f}": v for (o, f), v in obs.items()}
    out["n_seeds"] = int(units["seed"].nunique())
    if out["n_seeds"] < 2:
        out["warning"] = ("only one seed -- the seed dimension of the "
                          "bootstrap is degenerate and the CIs are too tight")

    # H3 verdict on the inherited row
    dk, ek = "inherited:deficit_endo", "inherited:exo_vs_cross"
    if (dk in gaps and ek in gaps and gaps[dk].get("status") != "insufficient data"
            and gaps[ek].get("status") != "insufficient data"):
        d_lo, d_hi = gaps[dk]["ci95"]
        e_lo, e_hi = gaps[ek]["ci95"]
        blind = bool(np.isfinite(d_lo) and d_lo > 0)          # deficit real
        closed = bool(np.isfinite(e_lo) and e_lo <= 0 <= e_hi)  # exo ~ cross
        out["h3"] = dict(
            blind_spot_present_endogenous=blind,
            blind_spot_closed_exogenous=closed,
            supported=bool(blind and closed),
            deficit_endogenous=gaps[dk]["point_estimate"],
            deficit_ci=gaps[dk]["ci95"],
            exo_vs_cross=gaps[ek]["point_estimate"],
            exo_vs_cross_ci=gaps[ek]["ci95"])
    else:
        out["h3"] = _insuff("inherited row lacks same/cross/exogenous cells")
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 5b. random-partition null -- the control that licenses every other claim
# --------------------------------------------------------------------------
def _frontier_units(df, variant=MAIN_VARIANT, budget="strict",
                    factors=None, origins=None, extra_cols=()):
    """One frontier point per (arm, seed, origin, designer_role)."""
    d = df.copy()
    if variant:
        d = d[d["variant"] == variant]
    if factors:
        d = d[d["designer_family_factor"].isin(list(factors))]
    if origins:
        d = d[d["origin"].isin(list(origins))]
    if d.empty:
        return d, budget
    cols = [c for c in ("arm", "seed", "origin", "designer_role", "designer")
            if c in d.columns] + [c for c in extra_cols if c in d.columns]
    u = best_within_budget(d, cols, budget)
    used = budget
    if u.empty:
        u = best_within_budget(d, cols, "loose")
        used = "loose (strict empty)"
    return u, used


def _gap_table(units, pairs, n_boot=2000, seed=0, pooled=True):
    """Bootstrap gaps between designer_family_factor levels.

    `pairs` is [(name, level_a, level_b), ...]; the gap is mean(a) - mean(b),
    computed per origin and (optionally) pooled over origins. Resampling is
    over seeds and over designers-within-level, as in §6.
    """
    gaps = {}
    frames = [("", units)]
    if pooled and units["origin"].nunique() > 1:
        p = units.copy()
        p["origin"] = "pooled"
        frames.append(("", p))
    for _, u in frames:
        B = _CellBootstrap(u)
        if not B.ok:
            continue
        draws = B.draw(n_boot=n_boot, seed=seed)
        obs = B.observed
        for origin in B.origins:
            for name, a, b in pairs:
                k = f"{origin}:{name}"
                if (origin, a) not in draws or (origin, b) not in draws:
                    gaps[k] = _insuff(f"missing cell {a} or {b}")
                    continue
                g = _boot_summary(draws[(origin, a)] - draws[(origin, b)])
                g["point_estimate"] = float(obs[(origin, a)] - obs[(origin, b)])
                g["definition"] = (f"mean dBias({origin},{a}) "
                                   f"- mean dBias({origin},{b})")
                gaps[k] = g
    return gaps


def random_null_test(df, n_boot=2000, seed=0, variant=MAIN_VARIANT,
                     budget="strict"):
    """The random-partition null (designer_role == "random").

    Same sentence pool, partition drawn at random, no designer at all. If the
    null reaches bias reduction comparable to the elicited designers then
    nothing about designer family is interpretable -- so this runs before any
    2x2 claim is believed.
    """
    out = {"variant": variant, "n_boot": n_boot}
    if df is None or df.empty:
        return _insuff("no records", **out)
    units, used = _frontier_units(df, variant, budget)
    out["budget"] = used
    if units.empty:
        return _insuff("no unit meets a collateral budget", **out)

    tab = (units.groupby(["origin", "designer_family_factor"], dropna=False)
           ["bias_reduction"].agg(["size", "mean", "std"]).reset_index()
           .rename(columns={"size": "n", "mean": "dBias", "std": "sd"}))
    out["table"] = tab
    out["levels_present"] = sorted(
        units["designer_family_factor"].astype(str).unique().tolist())
    if "null" not in out["levels_present"]:
        return _insuff("no random-partition null rows (designer_role="
                       "'random') present yet", **out)

    out["gaps"] = _gap_table(units,
                             [("same_minus_null", "same", "null"),
                              ("cross_minus_null", "cross", "null"),
                              ("exogenous_minus_null", "exogenous", "null")],
                             n_boot=n_boot, seed=seed)
    out["n_seeds"] = int(units["seed"].nunique())
    if out["n_seeds"] < 2:
        out["warning"] = ("only one seed -- bootstrap CIs on the null "
                          "comparisons are degenerate")
    # headline: is ANY elicited signal above the null?
    key = ("pooled:cross_minus_null"
           if "pooled:cross_minus_null" in out["gaps"] else None)
    if key is None:
        cands = [k for k in out["gaps"] if k.endswith(":cross_minus_null")
                 and out["gaps"][k].get("status") != "insufficient data"]
        key = cands[0] if cands else None
    out["headline_key"] = key
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 5b-i. above-null reporting (experiments_v2.md §N -- MANDATORY)
# --------------------------------------------------------------------------
_NULL_LEVELS = [("arm", "seed", "origin"), ("arm", "origin"), ("origin",), ()]


def null_baseline(df, variant=MAIN_VARIANT, budget="strict"):
    """Frontier dBias of the DATA-PARTITION null (designer_role='random').

    experiments_v2.md §1: negating any gendered minimal-pair contrast already
    lands in the gender subspace, so the null is ~0.298, not ~0. Every
    designer number must be quoted as a gap above it.

    Returns a hierarchy of baselines -- (arm, seed, origin), (arm, origin),
    (origin,), global -- so a unit can always be matched to the most specific
    null that actually exists.
    """
    d = df[(df["variant"] == variant)
           & (df["designer_family_factor"] == "null")]
    out = {"variant": variant, "levels": _NULL_LEVELS}
    if d.empty:
        return _insuff("no data-partition null rows (designer_role='random') "
                       f"for variant={variant}", **out)
    cols = [c for c in ("arm", "seed", "origin", "designer_role", "designer")
            if c in d.columns]
    u = best_within_budget(d, cols, budget)
    used = budget
    if u.empty:
        u = best_within_budget(d, cols, "loose")
        used = "loose (strict empty)"
    if u.empty:
        return _insuff("no null unit meets any collateral budget", **out)
    tables = {}
    for lv in _NULL_LEVELS:
        tables[lv] = (u.groupby(list(lv), dropna=False)["bias_reduction"].mean()
                      if lv else float(u["bias_reduction"].mean()))
    out.update(status="ok", budget=used, units=u, tables=tables,
               n_units=int(len(u)),
               by_arm_origin=(u.groupby(["arm", "origin"], dropna=False)
                              ["bias_reduction"].agg(["size", "mean", "std"])
                              .reset_index()
                              .rename(columns={"size": "n", "mean": "null_dBias",
                                               "std": "sd"})))
    return out


def _null_lookup(nb, row):
    """Most specific available null baseline for one unit -> (value, level)."""
    if nb.get("status") != "ok":
        return float("nan"), "none"
    for lv in _NULL_LEVELS:
        t = nb["tables"].get(lv)
        if not lv:
            return float(t), "global"
        try:
            key = tuple(row[c] for c in lv)
            v = t.loc[key if len(key) > 1 else key[0]]
        except Exception:
            continue
        if isinstance(v, pd.Series):
            v = v.mean()
        if pd.notna(v):
            return float(v), "/".join(lv)
    return float("nan"), "none"


def above_null(df, group_cols=("origin", "designer_role", "signal"),
               variant=MAIN_VARIANT, budget="strict", drop_null_rows=True):
    """Frontier bias reduction expressed as a GAP ABOVE THE PARTITION NULL.

    experiments_v2.md §N makes this mandatory: raw designer numbers overstate
    every condition, because the data-partition null alone already removes
    ~0.30 of the gender skew.

    Returns {"units": per-unit frame with `dBias_above_null`,
             "table": the same aggregated over `group_cols`,
             "null_levels": which baseline granularity each unit matched}.
    """
    out = {"variant": variant}
    if df is None or df.empty:
        return _insuff("no records", **out)
    units, used = _frontier_units(df, variant, budget)
    out["budget"] = used
    if units.empty:
        return _insuff("no unit meets a collateral budget", **out)

    nb = null_baseline(df, variant, budget)
    out["null_status"] = nb.get("status", "insufficient data")
    out["null_reason"] = nb.get("reason")
    vals, levels = [], []
    for _, r in units.iterrows():
        v, lv = _null_lookup(nb, r)
        vals.append(v)
        levels.append(lv)
    units = units.copy()
    units["null_dBias"] = vals
    units["null_level"] = levels
    units["dBias_above_null"] = units["bias_reduction"] - units["null_dBias"]
    if drop_null_rows:
        tab_src = units[units["designer_family_factor"] != "null"]
    else:
        tab_src = units
    gc = [c for c in group_cols if c in units.columns]
    out["units"] = units
    out["null_levels"] = dict(pd.Series(levels).value_counts())
    if gc and len(tab_src):
        t = (tab_src.groupby(gc, dropna=False)
             .agg(n=("bias_reduction", "size"),
                  dBias=("bias_reduction", "mean"),
                  dBias_sd=("bias_reduction", "std"),
                  null=("null_dBias", "mean"),
                  dBias_above_null=("dBias_above_null", "mean"),
                  above_null_sd=("dBias_above_null", "std"))
             .reset_index().sort_values(gc))
        out["table"] = t
    else:
        out["table"] = pd.DataFrame()
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 5b-ii. sign-shuffle null (experiments_v2.md §N -- second mandatory null)
# --------------------------------------------------------------------------
def sign_null_report(df, threshold=TH_SIGN_NULL_FLAT, budget="strict"):
    """The sign-shuffle null: same scale/sparsity, signs shuffled.

    It must land at ~0. If an arm's sign-shuffle null is NOT flat, the sign
    pattern is not what is carrying that arm's effect and every number from
    that arm is suspect.
    """
    out = {"threshold": threshold}
    if df is None or df.empty:
        return _insuff("no records", **out)
    d = df[df["variant"].astype(str).str.endswith(SIGN_NULL_SUFFIX)]
    if d.empty:
        return _insuff("no sign-shuffle rows (variant ending "
                       f"'{SIGN_NULL_SUFFIX}') present yet", **out)
    d = d[d["bias_reduction"].notna()]
    if d.empty:
        return _insuff("sign-shuffle rows carry no finite bias_reduction",
                       **out)

    tab = (d.groupby(["arm"], dropna=False)["bias_reduction"]
           .agg(["size", "mean", "std", "min", "max"]).reset_index()
           .rename(columns={"size": "n", "mean": "dBias_mean",
                            "std": "sd"}))
    tab["abs_mean"] = tab["dBias_mean"].abs()
    tab["flat"] = tab["abs_mean"] < threshold
    out["table"] = tab
    out["by_arm_origin"] = (d.groupby(["arm", "origin"], dropna=False)
                            ["bias_reduction"].agg(["size", "mean"])
                            .reset_index()
                            .rename(columns={"size": "n", "mean": "dBias_mean"}))

    # the frontier view: what this null would score if it were treated like a
    # designer (best alpha inside the budget) -- strictly the harder test
    cols = [c for c in ("arm", "seed", "origin", "designer_role")
            if c in d.columns]
    fu = best_within_budget(d, cols, budget)
    if fu.empty:
        fu = best_within_budget(d, cols, "loose")
    out["frontier_by_arm"] = (
        fu.groupby(["arm"], dropna=False)["bias_reduction"]
        .agg(["size", "mean"]).reset_index()
        .rename(columns={"size": "n", "mean": "frontier_dBias_mean"})
        if len(fu) else pd.DataFrame())

    out["per_arm"] = {str(r["arm"]): dict(n=int(r["n"]),
                                          mean=float(r["dBias_mean"]),
                                          flat=bool(r["flat"]))
                      for _, r in tab.iterrows()}
    out["arms_flat"] = sorted(str(r["arm"]) for _, r in tab.iterrows()
                              if r["flat"])
    out["arms_suspect"] = sorted(str(r["arm"]) for _, r in tab.iterrows()
                                 if not r["flat"])
    out["all_flat"] = bool(len(out["arms_suspect"]) == 0)
    out["worst_arm"] = (str(tab.loc[tab["abs_mean"].idxmax(), "arm"])
                        if len(tab) else None)
    out["worst_abs_mean"] = (float(tab["abs_mean"].max()) if len(tab)
                             else float("nan"))
    out["arms_covered"] = sorted(tab["arm"].astype(str).tolist())
    out["arms_missing"] = sorted(set(df["arm"].astype(str).unique())
                                 - set(out["arms_covered"]))
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 5c. bidirectional arm control (design.md §5.1)
# --------------------------------------------------------------------------
def arm_asymmetry(df, n_boot=2000, seed=0, variant=MAIN_VARIANT,
                  budget="strict", origin="inherited"):
    """Does the same-vs-cross asymmetry survive flipping the arm?

    design.md §5.1: run the designs bidirectionally. If "cross-family wins"
    were really "one lineage's models are just stronger", then swapping which
    lineage is the target would NOT preserve the cross-over-same advantage --
    the advantage would stay with the same lineage. Two readings, both
    reported:

      relation view -- per arm, mean dBias(cross) - mean dBias(same) on the
                       inherited row. The blind-spot law predicts this is
                       POSITIVE IN EVERY ARM (the effect follows the
                       *relation*, not the lineage).
      lineage view  -- per designer lineage L, mean dBias when L is the
                       cross-family designer minus mean dBias when L is the
                       same-family designer. The blind-spot law predicts this
                       is positive for every lineage, i.e. each lineage's
                       apparent quality FLIPS sign when the arm flips. A pure
                       capability story predicts no flip.
    """
    out = {"variant": variant, "origin": origin, "n_boot": n_boot}
    if df is None or df.empty:
        return _insuff("no records", **out)
    units, used = _frontier_units(df, variant, budget,
                                  factors=("same", "cross"), origins=(origin,))
    out["budget"] = used
    if units.empty:
        return _insuff(f"no {origin} unit meets a collateral budget", **out)

    arms = sorted(units["arm"].astype(str).unique().tolist())
    out["arms"] = arms
    out["table"] = (units.groupby(["arm", "designer_family_factor"],
                                  dropna=False)["bias_reduction"]
                    .agg(["size", "mean", "std"]).reset_index()
                    .rename(columns={"size": "n", "mean": "dBias",
                                     "std": "sd"}))

    # ---- relation view, per arm
    per_arm = {}
    for a in arms:
        u = units[units["arm"].astype(str) == a].copy()
        u["origin"] = "arm"      # collapse; one cell dimension is enough here
        g = _gap_table(u, [("cross_minus_same", "cross", "same")],
                       n_boot=n_boot, seed=seed, pooled=False)
        per_arm[a] = g.get("arm:cross_minus_same",
                           _insuff("missing same or cross cell"))
    out["per_arm"] = per_arm
    ok = [v for v in per_arm.values() if v.get("status") != "insufficient data"]
    out["relation_consistent"] = (
        bool(len(ok) == len(arms) and len(arms) >= 2
             and all(v["point_estimate"] > 0 for v in ok)))
    out["relation_consistent_ci"] = (
        bool(len(ok) == len(arms) and len(arms) >= 2
             and all(np.isfinite(v["ci95"][0]) and v["ci95"][0] > 0
                     for v in ok)))

    # ---- lineage view
    lin_col = "designer_family" if "designer_family" in units.columns else None
    lineage = {}
    if lin_col:
        lt = (units.groupby(["arm", lin_col], dropna=False)["bias_reduction"]
              .agg(["size", "mean"]).reset_index()
              .rename(columns={"size": "n", "mean": "dBias"}))
        out["lineage_table"] = lt
        for L in sorted(units[lin_col].astype(str).unique()):
            sub = units[units[lin_col].astype(str) == L]
            as_same = sub[sub["arm"].astype(str) == L]["bias_reduction"]
            as_cross = sub[sub["arm"].astype(str) != L]["bias_reduction"]
            if len(as_same) == 0 or len(as_cross) == 0:
                lineage[L] = _insuff(
                    "lineage never appears in both roles "
                    f"(n_as_same={len(as_same)}, n_as_cross={len(as_cross)})")
                continue
            lineage[L] = dict(status="ok", n_as_same=int(len(as_same)),
                              n_as_cross=int(len(as_cross)),
                              dBias_as_same=float(as_same.mean()),
                              dBias_as_cross=float(as_cross.mean()),
                              flip=float(as_cross.mean() - as_same.mean()))
    out["lineage"] = lineage
    flips = [v for v in lineage.values() if v.get("status") == "ok"]
    out["lineage_flips"] = (bool(flips) and all(v["flip"] > 0 for v in flips))
    out["n_lineages_testable"] = len(flips)

    if len(arms) < 2:
        out["warning"] = ("only one arm present -- the bidirectional control "
                          "cannot be evaluated yet")
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 5e. EXPERIMENT A -- arm replication (experiments_v2.md, make-or-break)
# --------------------------------------------------------------------------
def _gap_on(units, a, b, n_boot=2000, seed=0, label="gap"):
    """Bootstrap mean(level a) - mean(level b) on an already-filtered frame."""
    if units is None or units.empty:
        return _insuff("no units")
    u = units.copy()
    u["origin"] = "_"          # collapse: the caller has already filtered
    g = _gap_table(u, [(label, a, b)], n_boot=n_boot, seed=seed, pooled=False)
    return g.get(f"_:{label}", _insuff(f"missing cell {a} or {b}"))


def per_arm_interactions(df, variant=MAIN_VARIANT, budget="strict",
                         signal="endogenous"):
    """Tidy per-arm interaction table -- experiments_v2.md §N bans pooled-only.

    Row 0 is the pooled fit with ARM AS THE RANDOM EFFECT (criterion A1);
    the remaining rows are one independent fit per arm. Returned as a
    DataFrame so the figures script can draw the forest plot from it.
    """
    cols = ["scope", "arm", "status", "n", "n_seeds", "estimate", "ci_lo",
            "ci_hi", "pvalue", "positive", "ci_excludes_0", "method", "reason"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    arms = sorted(df["arm"].astype(str).unique().tolist())
    rows = []
    specs = [("pooled", None, "arm")] + [("arm", a, None) for a in arms]
    for scope, a, rg in specs:
        try:
            mm = primary_interaction(df, variant=variant, budget=budget,
                                     signal=signal, arm=a, re_group=rg)
        except Exception as e:                                # pragma: no cover
            mm = _insuff(f"{type(e).__name__}: {e}")
        r = dict(scope=scope, arm=(a if a else "POOLED"),
                 status=mm.get("status", "insufficient data"),
                 n=mm.get("n"), n_seeds=mm.get("n_seeds"),
                 estimate=mm.get("estimate", float("nan")),
                 ci_lo=(mm.get("ci95") or (float("nan"),) * 2)[0],
                 ci_hi=(mm.get("ci95") or (float("nan"),) * 2)[1],
                 pvalue=mm.get("pvalue", float("nan")),
                 method=mm.get("method"), reason=mm.get("reason"))
        est, lo, hi = r["estimate"], r["ci_lo"], r["ci_hi"]
        r["positive"] = bool(np.isfinite(est) and est > 0)
        r["ci_excludes_0"] = bool(np.isfinite(lo) and np.isfinite(hi)
                                  and (lo > 0 or hi < 0))
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)


def _forest_lines(tidy, width=45):
    """ASCII forest plot of the per-arm interaction estimates."""
    if tidy is None or len(tidy) == 0:
        return ["  (no arms)"]
    fin = tidy[np.isfinite(tidy["estimate"].astype(float))]
    if not len(fin):
        return ["  (no arm has a computable interaction yet)"]
    lo = float(min(fin["ci_lo"].min(), fin["estimate"].min(), 0.0))
    hi = float(max(fin["ci_hi"].max(), fin["estimate"].max(), 0.0))
    if not np.isfinite(lo):
        lo = float(min(fin["estimate"].min(), 0.0))
    if not np.isfinite(hi):
        hi = float(max(fin["estimate"].max(), 0.0))
    pad = max((hi - lo) * 0.08, 1e-6)
    lo, hi = lo - pad, hi + pad

    def col(v):
        if not np.isfinite(v):
            return None
        return int(round((v - lo) / (hi - lo) * (width - 1)))

    zero = col(0.0)
    out = [f"  scale: {lo:+.3f} {'.' * (width - 16)} {hi:+.3f}   "
           f"('|'=0, 'o'=estimate, '[ ]'=95% CI)"]
    for _, r in tidy.iterrows():
        name = str(r["arm"])
        if r["status"] != "ok" or not np.isfinite(float(r["estimate"])):
            out.append(f"  {name:>8s} |{'-' * width}|  insufficient data"
                       f" ({str(r['reason'])[:44]})")
            continue
        track = [" "] * width
        if zero is not None and 0 <= zero < width:
            track[zero] = "|"
        a, b = col(float(r["ci_lo"])), col(float(r["ci_hi"]))
        if a is not None and b is not None:
            for i in range(max(0, min(a, b)), min(width, max(a, b) + 1)):
                if track[i] == " ":
                    track[i] = "="
            if 0 <= a < width:
                track[a] = "["
            if 0 <= b < width:
                track[b] = "]"
        c = col(float(r["estimate"]))
        if c is not None and 0 <= c < width:
            track[c] = "o"
        flag = "*" if r["ci_excludes_0"] else " "
        out.append(f"  {name:>8s} |{''.join(track)}| {r['estimate']:+.4f} "
                   f"{_fmt_ci((r['ci_lo'], r['ci_hi']))}{flag} "
                   f"n={r['n']}")
    return out


def experiment_a_report(df, n_boot=2000, seed=0, variant=MAIN_VARIANT,
                        budget="strict"):
    """Experiment A of experiments_v2.md: does the law survive new arms?

    A1  pooled interaction > 0 with a CI excluding 0, ARM as random effect
    A2  point estimate > 0 in >= ceil(0.6 n_arms) arms AND CI excludes 0 in
        >= 2 arms
    A3  `same - partition-null` on the INHERITED row has a CI covering 0
        (same-family is at chance) in >= ceil(0.6 n_arms) arms
    A4  pooled `cross - same` on the ACQUIRED row covers 0 -- the effect is
        inherited-specific, not a blanket cross-family advantage
    KILL/DEMOTE if the interaction is > 0 in <= 1 arm, or A1 fails.
    """
    out = {"variant": variant, "n_boot": n_boot}
    if df is None or df.empty:
        return _insuff("no records", **out)
    arms = sorted(df["arm"].astype(str).unique().tolist())
    n_arms = len(arms)
    need = int(math.ceil(TH_ARM_FRACTION * n_arms)) if n_arms else 0
    out.update(arms=arms, n_arms=n_arms, arms_needed=need)

    forest = per_arm_interactions(df, variant=variant, budget=budget)
    out["forest"] = forest
    per_arm = forest[forest["scope"] == "arm"]
    pooled = forest[forest["scope"] == "pooled"]

    # ---- A1
    if len(pooled) and pooled.iloc[0]["status"] == "ok":
        p0 = pooled.iloc[0]
        out["A1"] = dict(status="ok", estimate=float(p0["estimate"]),
                         ci95=(float(p0["ci_lo"]), float(p0["ci_hi"])),
                         pvalue=float(p0["pvalue"]), n=int(p0["n"] or 0),
                         method=p0["method"],
                         passed=bool(p0["positive"] and p0["ci_excludes_0"]))
    else:
        out["A1"] = _insuff(
            (pooled.iloc[0]["reason"] if len(pooled) else "no pooled fit"),
            passed=None)

    # ---- A2
    ok = per_arm[per_arm["status"] == "ok"]
    n_pos = int(ok["positive"].sum())
    n_ci = int((ok["positive"] & ok["ci_excludes_0"]).sum())
    out["A2"] = dict(status="ok" if len(ok) else "insufficient data",
                     n_arms=n_arms, n_computable=int(len(ok)),
                     n_positive=n_pos, n_ci_excludes_0=n_ci,
                     arms_positive=sorted(ok[ok["positive"]]["arm"].tolist()),
                     arms_ci=sorted(ok[ok["positive"] & ok["ci_excludes_0"]]
                                    ["arm"].tolist()),
                     needed_positive=need, needed_ci=TH_ARM_CI_EXCLUDES,
                     passed=(bool(n_pos >= need and n_ci >= TH_ARM_CI_EXCLUDES)
                             if len(ok) else None))

    # ---- A3: same vs partition-null on the inherited row, per arm
    a3 = {}
    try:
        u, ubudget = _frontier_units(df, variant, budget,
                                     factors=("same", "null"),
                                     origins=("inherited",))
        out["A3_budget"] = ubudget
        for a in arms:
            ua = u[u["arm"].astype(str) == a] if len(u) else u
            if ua.empty:
                a3[a] = _insuff("no inherited same/null unit in this arm")
                continue
            a3[a] = _gap_on(ua, "same", "null", n_boot=n_boot, seed=seed,
                            label="same_minus_null")
    except Exception as e:                                    # pragma: no cover
        a3 = {a: _insuff(f"{type(e).__name__}: {e}") for a in arms}
    okk = {a: g for a, g in a3.items() if g.get("status") != "insufficient data"}
    at_chance = [a for a, g in okk.items()
                 if np.isfinite(g["ci95"][0]) and g["ci95"][0] <= 0 <= g["ci95"][1]]
    out["A3"] = dict(status="ok" if okk else "insufficient data",
                     per_arm=a3, n_computable=len(okk),
                     arms_at_chance=sorted(at_chance),
                     n_at_chance=len(at_chance), needed=need,
                     passed=(bool(len(at_chance) >= need) if okk else None))

    # ---- A4: cross - same on the ACQUIRED row, pooled
    try:
        ua, abudget = _frontier_units(df, variant, budget,
                                      factors=("same", "cross"),
                                      origins=("acquired",))
        out["A4_budget"] = abudget
        g = _gap_on(ua, "cross", "same", n_boot=n_boot, seed=seed,
                    label="cross_minus_same")
        if g.get("status") == "insufficient data":
            out["A4"] = dict(g, passed=None)
        else:
            lo, hi = g["ci95"]
            out["A4"] = dict(status="ok", point_estimate=g["point_estimate"],
                             ci95=g["ci95"], frac_gt0=g["frac_gt0"],
                             passed=bool(np.isfinite(lo) and lo <= 0 <= hi))
    except Exception as e:                                    # pragma: no cover
        out["A4"] = _insuff(f"{type(e).__name__}: {e}", passed=None)

    # ---- KILL / DEMOTE
    a1p = out["A1"].get("passed")
    demote = None
    if len(ok):
        demote = bool(n_pos <= TH_DEMOTE_MAX_POSITIVE_ARMS or a1p is False)
    out["demote"] = dict(
        triggered=demote,
        n_arms_positive=n_pos, n_computable=int(len(ok)),
        a1_passed=a1p,
        verdict=("insufficient data -- no arm has a computable interaction yet"
                 if demote is None else
                 ("DEMOTE: retire the law claim; reframe (c) as a Qwen-family "
                  "scoped finding and lead with (b) + methodology + E"
                  if demote else
                  "law claim survives Experiment A so far (spine = full (c), "
                  "proceed to C -> B)")))
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 5d. direction alignment from the sketch files
# --------------------------------------------------------------------------
def _parse_sketch_name(fn):
    """`{arm}|{seed}|{origin}|{signal}|{role}.npy` (arm optional, legacy)."""
    stem = fn[:-4] if fn.endswith(".npy") else fn
    parts = stem.split("|")
    if len(parts) == 5:
        arm, seed, origin, signal, role = parts
    elif len(parts) == 4:
        arm, (seed, origin, signal, role) = DEFAULT_ARM, parts
    else:
        return None
    try:
        seed = int(seed)
    except Exception:
        return None
    return dict(arm=arm, seed=seed, origin=origin, signal=signal,
                designer_role=role)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    a, b = a[:n], b[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not np.isfinite(na) or not np.isfinite(nb) or na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def direction_alignment(df=None, sketch_dir=None, variant=MAIN_VARIANT,
                        budget="strict", n_boot=0):
    """cosine(v_role, v_exogenous_gt) from the 20k-d coordinate sketches.

    The sketches are a FIXED random coordinate subset of the contrast vector
    v, so cosines between sketches estimate cosines between the full vectors.
    The exogenous ground-truth direction is the reference: it is the closest
    thing we have to the *right* direction, so this measures whether a
    designer found the right direction rather than merely a large one --
    which is exactly what the blind-spot story is about.
    """
    sketch_dir = sketch_dir or SKETCH_DIR
    out = {"sketch_dir": sketch_dir, "variant": variant}
    if not os.path.isdir(sketch_dir):
        return _insuff(f"sketch directory {sketch_dir} does not exist", **out)
    files = [f for f in sorted(os.listdir(sketch_dir)) if f.endswith(".npy")]
    out["n_files"] = len(files)
    if not files:
        return _insuff("no .npy sketches written yet", **out)

    meta, vecs, unreadable = {}, {}, 0
    for f in files:
        m = _parse_sketch_name(f)
        if m is None:
            unreadable += 1
            continue
        try:
            v = np.load(os.path.join(sketch_dir, f))
        except Exception:
            unreadable += 1          # still being written / truncated
            continue
        k = (m["arm"], m["seed"], m["origin"], m["signal"], m["designer_role"])
        meta[k] = m
        vecs[k] = v
    out["n_unreadable"] = unreadable
    if not vecs:
        return _insuff("no readable sketches", **out)

    refs = {k[:3]: v for k, v in vecs.items()
            if k[3] == "exogenous" or k[4] == "gt"}
    out["n_references"] = len(refs)
    if not refs:
        return _insuff("no exogenous/gt reference sketch for any "
                       "(arm, seed, origin)", **out)

    rows = []
    for k, v in vecs.items():
        ref = refs.get(k[:3])
        if ref is None:
            continue
        r = dict(meta[k])
        r["is_reference"] = bool(k[3] == "exogenous" or k[4] == "gt")
        r["cos_to_gt"] = _cos(v, ref)
        r["designer_family_factor"] = _family_factor(r)
        rows.append(r)
    al = pd.DataFrame(rows)
    if al.empty:
        return _insuff("no sketch shares an (arm, seed, origin) with a "
                       "reference", **out)
    out["alignments"] = al
    out["n_aligned"] = int(len(al))
    non_ref = al[~al["is_reference"]]
    out["by_factor"] = (
        non_ref.groupby(["designer_family_factor"], dropna=False)["cos_to_gt"]
        .agg(["size", "mean", "std"]).reset_index()
        .rename(columns={"size": "n", "mean": "cos_mean", "std": "cos_sd"})
        if len(non_ref) else pd.DataFrame())
    out["by_origin_factor"] = (
        non_ref.groupby(["origin", "designer_family_factor"],
                        dropna=False)["cos_to_gt"]
        .agg(["size", "mean", "std"]).reset_index()
        .rename(columns={"size": "n", "mean": "cos_mean", "std": "cos_sd"})
        if len(non_ref) else pd.DataFrame())

    # ---- does alignment predict removal?
    if df is not None and not df.empty:
        units, used = _frontier_units(df, variant, budget)
        out["budget"] = used
        if units.empty:
            out["correlation"] = _insuff("no unit meets a collateral budget")
        else:
            on = [c for c in ("arm", "seed", "origin", "designer_role")
                  if c in units.columns and c in al.columns]
            u = units.copy()
            u["seed"] = pd.to_numeric(u["seed"], errors="coerce")
            for c in ("arm", "origin", "designer_role"):
                if c in u.columns:
                    u[c] = u[c].astype(str)
            a2 = al.copy()
            for c in ("arm", "origin", "designer_role"):
                a2[c] = a2[c].astype(str)
            m = u.merge(a2[on + ["cos_to_gt", "is_reference"]], on=on,
                        how="inner")
            out["n_merged"] = int(len(m))
            out["merged"] = m
            out["correlation"] = {
                "all": _corr_block(m["cos_to_gt"], m["bias_reduction"],
                                   "cos_to_gt"),
                "endogenous_only": _corr_block(
                    m[~m["is_reference"]]["cos_to_gt"],
                    m[~m["is_reference"]]["bias_reduction"], "cos_to_gt"),
            }
            for origin, sub in m.groupby("origin", dropna=False):
                out["correlation"][str(origin)] = _corr_block(
                    sub["cos_to_gt"], sub["bias_reduction"], "cos_to_gt")
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 6. blind-spot correlation (design.md §12 / §8)
# --------------------------------------------------------------------------
def _corr_block(x, y, label):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)

    def _const(v):
        # tolerant: float noise can leave std ~1e-16 on a constant column
        return not np.isfinite(np.ptp(v)) or np.ptp(v) <= 1e-12 * max(
            1.0, float(np.max(np.abs(v))))

    if n < 4 or _const(x) or _const(y):
        return _insuff(f"n={n} usable pairs (or a constant column)",
                       predictor=label, n=n)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    if not (np.isfinite(pr) and np.isfinite(sr)):
        return _insuff("correlation undefined (degenerate input)",
                       predictor=label, n=n)
    return dict(predictor=label, n=n,
                pearson_r=float(pr), pearson_p=float(pp),
                pearson_ci95=_ci_from_r(pr, n, "pearson"),
                spearman_r=float(sr), spearman_p=float(sp),
                spearman_ci95=_ci_from_r(sr, n, "spearman"),
                status="ok")


def blind_spot_correlation(df, budget="loose", variant=MAIN_VARIANT,
                           predictors=("elicit_contrast_gap", "vec_l2",
                                       "vec_cos_b_d")):
    """Does the *measured* elicitation contrast predict bias removal?

    design.md §12 asks whether "inherited-ness"/elicitation strength predicts
    removal failure monotonically. The endogenous conditions are pooled; the
    outcome is the frontier point (best bias reduction inside the collateral
    budget) per (seed, origin, designer_role) unit, and the analysis is also
    broken out per origin because the two origins have different achievable
    ranges.
    """
    out = {"variant": variant, "budget": budget}
    if df is None or df.empty:
        return _insuff("no records", **out)
    d = df[df["signal"] == "endogenous"].copy()
    if variant:
        d = d[d["variant"] == variant]
    # the random-partition null has no elicitation, so it cannot inform a
    # question about what the elicitation predicts
    d = d[d["designer_family_factor"] != "null"]
    if d.empty:
        return _insuff(f"no endogenous designer rows for variant={variant}",
                       **out)
    cols = [c for c in ("arm", "seed", "origin", "designer_role")
            if c in d.columns]
    units = best_within_budget(d, cols, budget)
    if units.empty:
        units = d.copy()
        out["note"] = ("no unit met the collateral budget; correlations "
                       "computed on all alpha rows")
    out["n_units"] = int(len(units))

    res = {}
    for p in predictors:
        if p not in units.columns:
            res[p] = _insuff(f"column {p} absent", predictor=p)
            continue
        res[p] = {"all": _corr_block(units[p], units["bias_reduction"], p)}
        for origin, sub in units.groupby("origin", dropna=False):
            res[p][str(origin)] = _corr_block(sub[p], sub["bias_reduction"], p)
    out["correlations"] = res
    out["status"] = "ok"
    return out


# --------------------------------------------------------------------------
# 7. pre-registered decision criteria (design.md §7)
# --------------------------------------------------------------------------
def _crit(value, threshold, passed, note=""):
    return {"value": value, "threshold": threshold,
            "passed": (bool(passed) if passed is not None else None),
            "note": note}


def _matched_alpha_retention(df):
    """SECONDARY DIAGNOSTIC ONLY -- binary vs fp at matched alpha.

    Confounded: E = scale * sign(v) has a very different norm from v, so at a
    common alpha the two arms are not at comparable edit magnitude and the
    ratio partly measures that scaling artifact rather than what survives
    binarisation. Kept because it is still informative about the shape of the
    alpha response, never as the §7 gate.
    """
    key = [c for c in ("arm", "seed", "origin", "signal", "designer_role",
                       "alpha") if c in df.columns]
    b = df[df["variant"] == MAIN_VARIANT]
    f = df[df["variant"] == FP_VARIANT]
    if b.empty or f.empty:
        return _insuff("no binary/fp rows")
    m = b.merge(f, on=key, suffixes=("_bin", "_fp"))
    m = m[m["bias_reduction_fp"].notna() & m["bias_reduction_bin"].notna()]
    if m.empty:
        return _insuff("no alpha-matched pairs")
    pos = m[m["bias_reduction_fp"] > 1e-9]
    per_pair = (pos["bias_reduction_bin"] / pos["bias_reduction_fp"]
                ).replace([np.inf, -np.inf], np.nan).dropna()
    denom = m["bias_reduction_fp"].mean()
    return dict(status="ok", n_pairs=int(len(m)),
                ratio_of_means=(float(m["bias_reduction_bin"].mean() / denom)
                                if abs(denom) > 1e-9 else float("nan")),
                median_pairwise_ratio=(float(per_pair.median())
                                       if len(per_pair) else float("nan")),
                mean_bin=float(m["bias_reduction_bin"].mean()),
                mean_fp=float(m["bias_reduction_fp"].mean()))


def h1_binary_retention(df, budget="strict"):
    """H1 feasibility gate at MATCHED COLLATERAL (design.md §7).

    §7 states the gate in collateral terms ("<= a few KB and <= ~1-2 pt MMLU
    drop"), not at a common alpha. So each variant is allowed its OWN best
    alpha: per (arm, seed, origin, signal, designer_role) we take the
    frontier point -- the largest bias reduction inside the collateral budget
    -- separately for the binary and the full-precision edit, and

        retention = binary_best / fp_best.

    Conditions whose fp frontier point is <= 0 carry no debiasing to retain
    and are dropped from the ratio (counted in `n_dropped_fp_nonpositive`).
    """
    if df is None or df.empty:
        return _insuff("no records")
    cond = [c for c in ("arm", "seed", "origin", "signal", "designer_role")
            if c in df.columns]
    b = df[df["variant"] == MAIN_VARIANT]
    f = df[df["variant"] == FP_VARIANT]
    # experiments_v2.md §1 declares H1 settled, so the new arms ship WITHOUT a
    # full-precision comparison. Those arms are simply out of scope for this
    # gate -- they must not error, and they must not be dropped from anything
    # else in the report.
    arms_all = sorted(df["arm"].astype(str).unique().tolist())
    arms_fp = sorted(f["arm"].astype(str).unique().tolist())
    cov = dict(arms_all=arms_all, arms_with_fp=arms_fp,
               arms_without_fp=sorted(set(arms_all) - set(arms_fp)))
    if b.empty or f.empty:
        return _insuff(f"need both {MAIN_VARIANT} ({len(b)} rows) and "
                       f"{FP_VARIANT} ({len(f)} rows)", **cov)

    used = budget
    bb = best_within_budget(b, cond, budget)
    ff = best_within_budget(f, cond, budget)
    if bb.empty or ff.empty:
        bb, ff = (best_within_budget(b, cond, "loose"),
                  best_within_budget(f, cond, "loose"))
        used = "loose (strict empty for one variant)"
    if bb.empty or ff.empty:
        return _insuff("no condition reaches any collateral budget in both "
                       "variants", budget=used)

    keep_b = cond + ["bias_reduction", "alpha", "d_mmlu", "ppl_ratio"]
    if "edit_bytes" in bb.columns:
        keep_b = keep_b + ["edit_bytes"]
    m = bb[keep_b].merge(ff[cond + ["bias_reduction", "alpha"]], on=cond,
                         suffixes=("_bin", "_fp"))
    if m.empty:
        return _insuff("no condition has a frontier point for both variants",
                       budget=used)
    n_all = int(len(m))
    kept = m[m["bias_reduction_fp"] > 1e-9]
    n_drop = n_all - int(len(kept))
    ratios = (kept["bias_reduction_bin"] / kept["bias_reduction_fp"]
              ).replace([np.inf, -np.inf], np.nan).dropna()
    denom = kept["bias_reduction_fp"].mean() if len(kept) else float("nan")
    out = dict(status="ok", budget=used,
               n_conditions=n_all, n_used=int(len(ratios)),
               n_dropped_fp_nonpositive=n_drop,
               ratio_of_means=(float(kept["bias_reduction_bin"].mean() / denom)
                               if len(kept) and abs(denom) > 1e-9
                               else float("nan")),
               median_ratio=(float(ratios.median()) if len(ratios)
                             else float("nan")),
               iqr=((float(ratios.quantile(0.25)), float(ratios.quantile(0.75)))
                    if len(ratios) >= 2 else (float("nan"), float("nan"))),
               ratio_min=(float(ratios.min()) if len(ratios) else float("nan")),
               ratio_max=(float(ratios.max()) if len(ratios) else float("nan")),
               mean_bin=float(m["bias_reduction_bin"].mean()),
               mean_fp=float(m["bias_reduction_fp"].mean()),
               mean_alpha_bin=float(m["alpha_bin"].mean()),
               mean_alpha_fp=float(m["alpha_fp"].mean()),
               mean_bin_mmlu_drop=float(-m["d_mmlu"].mean()),
               bytes_median=(float(m["edit_bytes"].median())
                             if "edit_bytes" in m.columns else float("nan")),
               per_condition=m)
    out.update(cov)
    per_arm = {}
    for a in arms_all:
        ma = m[m["arm"].astype(str) == a] if "arm" in m.columns else m
        if a not in arms_fp or ma.empty:
            per_arm[a] = dict(status="skipped -- no full-precision rows in "
                              "this arm (experiments_v2.md §1: H1 settled)",
                              n_conditions=int(len(ma)))
            continue
        k = ma[ma["bias_reduction_fp"] > 1e-9]
        rr = (k["bias_reduction_bin"] / k["bias_reduction_fp"]
              ).replace([np.inf, -np.inf], np.nan).dropna()
        per_arm[a] = dict(status="ok", n_conditions=int(len(ma)),
                          n_used=int(len(rr)),
                          n_dropped_fp_nonpositive=int(len(ma) - len(k)),
                          median_ratio=(float(rr.median()) if len(rr)
                                        else float("nan")),
                          ratio_of_means=(float(k["bias_reduction_bin"].mean()
                                                / k["bias_reduction_fp"].mean())
                                          if len(k) and
                                          abs(k["bias_reduction_fp"].mean()) > 1e-9
                                          else float("nan")))
    out["per_arm"] = per_arm
    out["matched_alpha_secondary"] = _matched_alpha_retention(df)
    return out


def kill_switch_report(df, n_boot=2000, seed=0):
    """design.md §7, evaluated literally where the data allow."""
    rep = {}

    # ---- SIGNAL_ABOVE_NULL ----------------------------------------------
    # Deliberately first: if the random-partition null matches the elicited
    # designers, no designer-family claim below is interpretable.
    try:
        rn = random_null_test(df, n_boot=n_boot, seed=seed)
        rep["_random_null"] = rn
        key = rn.get("headline_key") if rn.get("status") == "ok" else None
        g = rn.get("gaps", {}).get(key) if key else None
        if g and g.get("status") != "insufficient data":
            lo, hi = g["ci95"]
            rep["SIGNAL_ABOVE_NULL"] = _crit(
                {"cross_minus_null": g["point_estimate"], "ci95": g["ci95"],
                 "P(>0)": g["frac_gt0"]},
                "95% CI on (cross - null) excludes 0",
                bool(np.isfinite(lo) and (lo > 0 or hi < 0)) and g["point_estimate"] > 0,
                f"gap: {key}; a null that matches the designers would void "
                "every designer-family claim")
        else:
            rep["SIGNAL_ABOVE_NULL"] = _crit(
                None, "95% CI on (cross - null) excludes 0", None,
                f"insufficient data: {rn.get('reason', 'null cell missing')}")
    except Exception as e:                                    # pragma: no cover
        rep["SIGNAL_ABOVE_NULL"] = _crit(None, "cross > null", None,
                                         f"error: {type(e).__name__}: {e}")

    # ---- SIGN_NULL_FLAT (experiments_v2.md §N, second mandatory null) -----
    try:
        sn = sign_null_report(df)
        rep["_sign_null"] = sn
        if sn.get("status") == "ok":
            rep["SIGN_NULL_FLAT"] = _crit(
                {"per_arm_mean": {a: v["mean"] for a, v in sn["per_arm"].items()},
                 "worst": sn["worst_arm"], "worst_abs_mean": sn["worst_abs_mean"]},
                f"|mean dBias| < {TH_SIGN_NULL_FLAT} in EVERY arm",
                # cannot claim "every arm" while some arms have no rows yet
                (sn["all_flat"] if not sn["arms_missing"]
                 else (False if not sn["all_flat"] else None)),
                ("all covered arms flat" if sn["all_flat"] else
                 "SUSPECT ARMS: " + ", ".join(sn["arms_suspect"])
                 + " -- the sign pattern is not what carries these arms' "
                   "effect; treat every number from them as unreliable")
                + (f"; no sign-shuffle rows yet for arms "
                   f"{sn['arms_missing']}" if sn["arms_missing"] else ""))
        else:
            rep["SIGN_NULL_FLAT"] = _crit(
                None, f"|mean dBias| < {TH_SIGN_NULL_FLAT} in every arm", None,
                f"insufficient data: {sn.get('reason')}")
    except Exception as e:                                    # pragma: no cover
        rep["SIGN_NULL_FLAT"] = _crit(None, "sign-shuffle null ~ 0", None,
                                      f"error: {type(e).__name__}: {e}")

    # ---- EXPERIMENT A (experiments_v2.md) --------------------------------
    try:
        ea = experiment_a_report(df, n_boot=n_boot, seed=seed)
        rep["_experiment_a"] = ea
        if ea.get("status") == "ok":
            a1 = ea["A1"]
            rep["A1_pooled_interaction"] = _crit(
                ({"estimate": a1["estimate"], "ci95": a1["ci95"],
                  "p": a1["pvalue"]} if a1.get("status") == "ok" else None),
                "estimate > 0 and 95% CI excludes 0 (arm as random effect)",
                a1.get("passed"),
                (a1.get("method") or "") if a1.get("status") == "ok"
                else f"insufficient data: {a1.get('reason')}")
            a2 = ea["A2"]
            rep["A2_per_arm_replication"] = _crit(
                {"n_arms": a2["n_arms"], "n_computable": a2["n_computable"],
                 "n_positive": a2["n_positive"],
                 "n_ci_excludes_0": a2["n_ci_excludes_0"],
                 "arms_positive": a2["arms_positive"]},
                f">= {a2['needed_positive']} arms positive "
                f"(ceil(0.6*{a2['n_arms']})) and >= {TH_ARM_CI_EXCLUDES} "
                f"with CI excluding 0",
                a2.get("passed"),
                "per-arm replication; pooled-only reporting is banned (§N)")
            a3 = ea["A3"]
            rep["A3_blind_spot_null_equivalence"] = _crit(
                {"n_at_chance": a3["n_at_chance"],
                 "arms_at_chance": a3["arms_at_chance"],
                 "n_computable": a3["n_computable"]},
                f">= {a3['needed']} arms where (same - partition-null) on "
                "inherited has a CI covering 0",
                a3.get("passed"),
                "same-family at chance on inherited bias = the blind spot")
            a4 = ea["A4"]
            rep["A4_acquired_nonspecificity"] = _crit(
                ({"cross_minus_same_acquired": a4.get("point_estimate"),
                  "ci95": a4.get("ci95")} if a4.get("status") == "ok" else None),
                "CI covers 0 on the ACQUIRED row (pooled)",
                a4.get("passed"),
                ("confirms the effect is inherited-specific, not a blanket "
                 "cross-family advantage") if a4.get("status") == "ok"
                else f"insufficient data: {a4.get('reason')}")
            dm = ea["demote"]
            rep["A_DEMOTE"] = _crit(
                {"n_arms_positive": dm["n_arms_positive"],
                 "n_computable": dm["n_computable"],
                 "A1_passed": dm["a1_passed"], "triggered": dm["triggered"]},
                f"TRIGGERS if interaction > 0 in <= "
                f"{TH_DEMOTE_MAX_POSITIVE_ARMS} arm(s) or A1 fails",
                (None if dm["triggered"] is None else (not dm["triggered"])),
                dm["verdict"])
        else:
            for k, th in (("A1_pooled_interaction", "CI excludes 0"),
                          ("A2_per_arm_replication", "per-arm replication"),
                          ("A3_blind_spot_null_equivalence", "CI covers 0"),
                          ("A4_acquired_nonspecificity", "CI covers 0"),
                          ("A_DEMOTE", "see experiments_v2.md Experiment A")):
                rep[k] = _crit(None, th, None,
                               f"insufficient data: {ea.get('reason')}")
    except Exception as e:                                    # pragma: no cover
        rep["A_DEMOTE"] = _crit(None, "see experiments_v2.md", None,
                                f"error: {type(e).__name__}: {e}")

    # ---- H2 primary success ---------------------------------------------
    try:
        mm = primary_interaction(df)
        bs = bootstrap_interaction(df, n_boot=n_boot, seed=seed)
        rep["_primary_interaction"] = mm
        rep["_bootstrap_interaction"] = bs
        if mm.get("status") == "ok":
            p = mm["pvalue"]
            rep["H2_interaction_significant"] = _crit(
                {"estimate": mm["estimate"], "p": p, "ci95": mm["ci95"]},
                f"p < {TH_INTERACTION_P} and estimate > 0",
                (p < TH_INTERACTION_P and mm["estimate"] > 0),
                mm["method"])
            cm = mm.get("cell_means", {})
            ih_s, ih_c = cm.get("inherited/same"), cm.get("inherited/cross")
            ac_s, ac_c = cm.get("acquired/same"), cm.get("acquired/cross")
            if ih_s is not None and ih_c is not None and abs(ih_c) > 1e-9:
                rep["H2_inherited_ratio"] = _crit(
                    ih_s / ih_c, f"< {TH_INHERITED_RATIO}",
                    (ih_s / ih_c) < TH_INHERITED_RATIO,
                    "dBias(inherited,same) / dBias(inherited,cross)")
            else:
                rep["H2_inherited_ratio"] = _crit(
                    None, f"< {TH_INHERITED_RATIO}", None,
                    "insufficient data (inherited cells missing or cross ~ 0)")
            if ac_s is not None and ac_c is not None and abs(ac_c) > 1e-9:
                rep["H2_acquired_ratio"] = _crit(
                    ac_s / ac_c, f"> {TH_ACQUIRED_RATIO}",
                    (ac_s / ac_c) > TH_ACQUIRED_RATIO,
                    "dBias(acquired,same) / dBias(acquired,cross)")
            else:
                rep["H2_acquired_ratio"] = _crit(
                    None, f"> {TH_ACQUIRED_RATIO}", None,
                    "insufficient data (acquired cells missing or cross ~ 0)")
        else:
            for k, th in (("H2_interaction_significant", f"p < {TH_INTERACTION_P}"),
                          ("H2_inherited_ratio", f"< {TH_INHERITED_RATIO}"),
                          ("H2_acquired_ratio", f"> {TH_ACQUIRED_RATIO}")):
                rep[k] = _crit(None, th, None,
                               f"insufficient data: {mm.get('reason','')}")

        # axis coverage (§7 asks for >=3 of 4 axes; MVP has fewer)
        n_axes = int(df["axis"].nunique()) if "axis" in df.columns else 0
        rep["H2_axis_coverage"] = _crit(
            n_axes, ">= 3 of 4 axes", n_axes >= 3,
            "MVP (§11) runs 2 axes by design; this criterion cannot be met "
            "before the axis scale-up")

        sub = [rep[k]["passed"] for k in ("H2_interaction_significant",
                                          "H2_inherited_ratio",
                                          "H2_acquired_ratio")]
        rep["H2_primary_success"] = _crit(
            sub, "all of the above", (all(x is True for x in sub)
                                      if all(x is not None for x in sub) else None),
            "axis-coverage clause tracked separately")
    except Exception as e:                                    # pragma: no cover
        rep["H2_primary_success"] = _crit(None, "see §7", None,
                                          f"error: {type(e).__name__}: {e}")

    # ---- BIDIRECTIONAL_CONSISTENT (design.md §5.1) -----------------------
    try:
        aa = arm_asymmetry(df, n_boot=n_boot, seed=seed)
        rep["_arm"] = aa
        if aa.get("status") == "ok" and len(aa.get("arms", [])) >= 2:
            val = {"arms": aa["arms"],
                   "cross_minus_same_per_arm": {
                       a: (v.get("point_estimate") if v.get("status") != "insufficient data"
                           else None) for a, v in aa["per_arm"].items()},
                   "lineage_flips": aa["lineage_flips"]}
            rep["BIDIRECTIONAL_CONSISTENT"] = _crit(
                val,
                "(cross - same) > 0 in EVERY arm, and each lineage's "
                "advantage flips sign when the arm flips",
                bool(aa["relation_consistent"] and aa["lineage_flips"]),
                "if the asymmetry were 'the cross lineage is simply "
                "stronger', it would not survive swapping the arms "
                f"(CI-strict version: {aa['relation_consistent_ci']})")
        else:
            rep["BIDIRECTIONAL_CONSISTENT"] = _crit(
                None, "(cross - same) > 0 in every arm", None,
                "insufficient data: "
                + (aa.get("reason") or f"only arms {aa.get('arms')} present"))
    except Exception as e:                                    # pragma: no cover
        rep["BIDIRECTIONAL_CONSISTENT"] = _crit(None, "see §5.1", None,
                                                f"error: {type(e).__name__}: {e}")

    # ---- H3 mechanism success -------------------------------------------
    try:
        mt = mechanism_test(df, n_boot=n_boot, seed=seed)
        rep["_mechanism"] = mt
        h3 = mt.get("h3", {}) if mt.get("status") == "ok" else {}
        if h3 and h3.get("status") != "insufficient data":
            rep["H3_mechanism_success"] = _crit(
                {"deficit_endogenous": h3["deficit_endogenous"],
                 "deficit_ci": h3["deficit_ci"],
                 "exogenous_vs_cross": h3["exo_vs_cross"],
                 "exogenous_vs_cross_ci": h3["exo_vs_cross_ci"]},
                "blind spot present under endogenous AND exogenous ~ cross "
                "(CI covers 0)",
                h3["supported"],
                "exogenous arm has no designer, so the literal interaction is "
                "replaced by the inherited-row gap test (see mechanism_test)")
        else:
            rep["H3_mechanism_success"] = _crit(
                None, "exogenous interaction not significant", None,
                f"insufficient data: {mt.get('reason', h3.get('reason',''))}")
    except Exception as e:                                    # pragma: no cover
        rep["H3_mechanism_success"] = _crit(None, "see §7", None,
                                            f"error: {type(e).__name__}: {e}")

    # ---- H1 feasibility gate --------------------------------------------
    try:
        h1 = h1_binary_retention(df)
        rep["_h1"] = h1
        if h1.get("status") == "ok":
            rep["H1_binary_retention"] = _crit(
                {"median_ratio": h1["median_ratio"],
                 "iqr": h1["iqr"],
                 "ratio_of_means": h1["ratio_of_means"]},
                f"median >= {TH_BINARY_RETENTION}",
                (np.isfinite(h1["median_ratio"])
                 and h1["median_ratio"] >= TH_BINARY_RETENTION),
                f"{MAIN_VARIANT} vs {FP_VARIANT} at MATCHED COLLATERAL "
                f"(each variant at its own best alpha inside the "
                f"{h1['budget']} budget); n={h1['n_used']} conditions, "
                f"{h1['n_dropped_fp_nonpositive']} dropped for fp_best <= 0")
            drop = h1["mean_bin_mmlu_drop"]
            rep["H1_mmlu_cost"] = _crit(
                drop, f"<= {TH_MMLU_DROP}",
                (bool(drop <= TH_MMLU_DROP) if np.isfinite(drop) else None),
                "mean MMLU drop of the binary edit at its frontier point")
        else:
            rep["H1_binary_retention"] = _crit(
                None, f"median >= {TH_BINARY_RETENTION}", None,
                f"insufficient data: {h1.get('reason','')}")
            rep["H1_mmlu_cost"] = _crit(None, f"<= {TH_MMLU_DROP}", None,
                                        "insufficient data")
    except Exception as e:                                    # pragma: no cover
        rep["H1_binary_retention"] = _crit(None, f">= {TH_BINARY_RETENTION}",
                                           None, f"error: {e}")

    # ---- KILL / PIVOT ----------------------------------------------------
    try:
        units, used = _units_for_interaction(df)
        if units.empty:
            rep["KILL_PIVOT"] = _crit(None, "interaction ~ 0 with tight CI",
                                      None, "insufficient data")
        elif units["seed"].nunique() < 2:
            rep["KILL_PIVOT"] = _crit(
                None, "interaction ~ 0 with tight CI", None,
                "insufficient data: a single seed makes the bootstrap CI "
                "degenerate, so 'tight CI' cannot be assessed")
        else:
            B = _CellBootstrap(units)
            draws = B.draw(n_boot=n_boot, seed=seed) if B.ok else {}
            k_inh = ("inherited", "cross"), ("inherited", "same")
            if all(k in draws for k in k_inh):
                g = _boot_summary(draws[k_inh[0]] - draws[k_inh[1]])
                lo, hi = g["ci95"]
                width = hi - lo if np.isfinite(hi) and np.isfinite(lo) else float("nan")
                covers0 = bool(np.isfinite(lo) and lo <= 0 <= hi)
                tight = bool(np.isfinite(width) and width <= TH_KILL_CI_WIDTH)
                triggered = bool(covers0 and tight)
                rep["KILL_PIVOT"] = _crit(
                    {"cross_minus_same_on_inherited": g["mean"],
                     "ci95": g["ci95"], "ci_width": width,
                     "triggered": triggered},
                    f"TRIGGERS if CI covers 0 and width <= {TH_KILL_CI_WIDTH}",
                    (not triggered),
                    "passed=True means the thesis survives (no pivot to (b)); "
                    f"budget={used}")
            else:
                rep["KILL_PIVOT"] = _crit(None,
                                          "interaction ~ 0 with tight CI", None,
                                          "insufficient data: inherited "
                                          "same/cross cells missing")
    except Exception as e:                                    # pragma: no cover
        rep["KILL_PIVOT"] = _crit(None, "see §7", None,
                                  f"error: {type(e).__name__}: {e}")
    return rep


# --------------------------------------------------------------------------
# 8. report
# --------------------------------------------------------------------------
def _sec(title):
    return "\n" + "=" * 78 + f"\n{title}\n" + "=" * 78


def _num(v):
    """Readable rendering of scalars / tuples / dicts of numbers."""
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, float):
        if not np.isfinite(v):
            return "nan"
        return f"{v:+.4f}" if 1e-3 <= abs(v) < 1e5 or v == 0 else f"{v:+.3g}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_num(x) for x in v) + ")"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={_num(x)}" for k, x in v.items()) + "}"
    return str(v)


def _fmt_ci(ci):
    try:
        if not (np.isfinite(ci[0]) and np.isfinite(ci[1])):
            return "[CI n/a]"
        return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
    except Exception:
        return "[CI n/a]"


def _summary_table(df):
    g = df.groupby(["origin", "designer_role", "signal"],
                   dropna=False)["bias_reduction"].agg(["size", "mean", "std",
                                                        "min", "max"])
    g = g.reset_index().rename(columns={"size": "n", "mean": "dBias_mean",
                                        "std": "dBias_sd"})
    return g.sort_values(["origin", "signal", "designer_role"])


def report(df=None, n_boot=2000, seed=0, write=True, sketch_dir=None):
    """Assemble the full plain-text report (design.md §5.4, §6, §7, §12)."""
    L = []
    add = L.append
    if df is None:
        try:
            df = load_runs()
        except Exception as e:
            df = pd.DataFrame()
            add(f"load_runs FAILED: {type(e).__name__}: {e}")

    add("=" * 78)
    add("BLIND SPOT / SPB -- analysis report")
    add("design.md §5.4 (frontier), §6 (inference), §5.6★/H3 (mechanism), "
        "§5.1 (bidirectional arms),\n§7 (kill switches), §12 (blind-spot "
        "correlation), plus the random-partition null and the direction "
        "sketches")
    add("=" * 78)
    add(f"source          : {df.attrs.get('source', RUNS)}")
    add(f"records loaded  : {len(df)}"
        f"   (malformed/truncated lines skipped: {df.attrs.get('n_malformed', 0)})")
    if df.empty:
        add("\nNo usable records yet -- insufficient data for every section.")
        text = "\n".join(L)
        if write:
            _write(text)
        return text

    for c, lab in (("arm", "arms"), ("seed", "seeds"), ("origin", "origins"),
                   ("signal", "signals"), ("designer_role", "designer roles"),
                   ("designer_family_factor", "family factors"),
                   ("variant", "variants"), ("alpha", "alphas"),
                   ("axis", "axes")):
        if c in df.columns:
            vals = sorted(df[c].dropna().unique().tolist(), key=str)
            add(f"{lab:16s}: {vals}")

    # ---- coverage of the 2x2 --------------------------------------------
    add(_sec("0. DESIGN COVERAGE (cells x seeds actually present)"))
    try:
        cov = (df.pivot_table(index=["arm", "origin", "signal"],
                              columns="designer_role", values="key",
                              aggfunc="count", fill_value=0)
               .reset_index())
        add(_fmt(cov, "%.0f"))
        arms = sorted(df["arm"].astype(str).unique().tolist())
        need = [(a, o, s, r) for a in arms
                for o in ("inherited", "acquired")
                for s, roles in (("endogenous", ("self", "sibling", "cross",
                                                 "cross2", "random")),
                                 ("exogenous", ("gt",)))
                for r in roles]
        have = set(map(tuple, df[["arm", "origin", "signal", "designer_role"]]
                       .astype(str).drop_duplicates().values))
        missing = [f"{a}/{o}/{s}/{r}" for (a, o, s, r) in need
                   if (a, o, s, r) not in have]
        add(f"\n  cells populated : {len(need) - len(missing)}/{len(need)} "
            f"(grid sized on the arms seen so far: {arms})")
        if missing:
            add("  MISSING         : " + ", ".join(missing))
    except Exception as e:
        add(f"  (coverage table failed: {type(e).__name__}: {e})")

    # ---- 1. summary tables ----------------------------------------------
    add(_sec("1. BIAS REDUCTION by origin x designer_role x signal "
             "(all alphas, all variants)"))
    try:
        add(_fmt(_summary_table(df)))
    except Exception as e:
        add(f"  insufficient data ({type(e).__name__}: {e})")

    add("\n  -- same table, main binary variant only "
        f"({MAIN_VARIANT}) --")
    try:
        sub = df[df["variant"] == MAIN_VARIANT]
        add(_fmt(_summary_table(sub)) if len(sub) else "  (no rows)")
    except Exception as e:
        add(f"  insufficient data ({type(e).__name__}: {e})")

    # ---- 1b. above-null summary (experiments_v2.md §N) -------------------
    add(_sec("1b. ABOVE-NULL SUMMARY -- every designer number as a GAP over "
             "the data-partition null (experiments_v2.md §N)"))
    add("  Raw numbers overstate every condition: negating any gendered "
        "minimal-pair\n  contrast already lands in the gender subspace, so "
        "the null is ~0.30, not ~0.")
    try:
        nb = null_baseline(df)
        if nb.get("status") != "ok":
            add(f"\n  partition null: insufficient data ({nb.get('reason')})")
        else:
            add(f"\n  data-partition null, frontier dBias by arm x origin "
                f"(budget={nb['budget']}):")
            add(_fmt(nb["by_arm_origin"]))
        an = above_null(df, group_cols=("origin", "designer_role", "signal"))
        if an.get("status") != "ok":
            add(f"\n  insufficient data: {an.get('reason')}")
        else:
            add(f"\n  frontier dBias RAW vs ABOVE NULL "
                f"(variant={MAIN_VARIANT}, budget={an['budget']}):")
            add(_fmt(an["table"]))
            add(f"\n  null baseline granularity actually matched: "
                f"{an['null_levels']}")
            an2 = above_null(df, group_cols=("arm", "origin",
                                             "designer_family_factor"))
            if an2.get("status") == "ok" and len(an2["table"]):
                add("\n  ... by arm x origin x family factor "
                    "(per-arm reporting is mandatory, §N):")
                add(_fmt(an2["table"]))
    except Exception as e:
        add(f"  above_null failed: {type(e).__name__}: {e}")

    # ---- 1c. sign-shuffle null -------------------------------------------
    add(_sec("1c. SIGN-SHUFFLE NULL (variant '*"
             f"{SIGN_NULL_SUFFIX}') -- must be ~0"))
    try:
        sn = sign_null_report(df)
        if sn.get("status") != "ok":
            add(f"  insufficient data: {sn.get('reason')}")
        else:
            add(_fmt(sn["table"]))
            if len(sn.get("frontier_by_arm", [])):
                add("\n  (frontier view -- what the shuffled-sign edit would "
                    "score if treated like a designer:)")
                add(_fmt(sn["frontier_by_arm"]))
            if sn["arms_missing"]:
                add(f"\n  arms with NO sign-shuffle null yet: "
                    f"{sn['arms_missing']}")
            if sn["all_flat"]:
                add(f"\n  OK: every covered arm is flat "
                    f"(|mean| < {TH_SIGN_NULL_FLAT}); the sign PATTERN is "
                    "carrying the effect, not the scale.")
            else:
                add("\n  " + "!" * 70)
                add(f"  !! SIGN-SHUFFLE NULL IS NOT FLAT IN: "
                    f"{', '.join(sn['arms_suspect'])}")
                add("  !! Those arms' results are SUSPECT: an edit with "
                    "shuffled signs moves the")
                add("  !! bias, so the measured removal cannot be attributed "
                    "to the sign pattern.")
                add("  " + "!" * 70)
    except Exception as e:
        add(f"  sign_null_report failed: {type(e).__name__}: {e}")

    # ---- 2. frontier -----------------------------------------------------
    add(_sec("2. FRONTIER: bias reduction vs collateral (design.md §5.4)"))
    add(f"  budgets: strict = d_mmlu >= {BUDGETS['strict']['d_mmlu_min']}, "
        f"ppl_ratio <= {BUDGETS['strict']['ppl_ratio_max']};  "
        f"loose = d_mmlu >= {BUDGETS['loose']['d_mmlu_min']}, "
        f"ppl_ratio <= {BUDGETS['loose']['ppl_ratio_max']}")
    try:
        fr = frontier(df)
        if fr.get("status") != "ok":
            add("  insufficient data")
        else:
            add("\n  -- best achievable bias reduction per condition --")
            add(_fmt(fr["frontier"]))
            tr = fr["trace"]
            keep = tr[tr["variant"].isin([MAIN_VARIANT, FP_VARIANT])] \
                if "variant" in tr.columns else tr
            add(f"\n  -- ALPHA TRACE (never a single scalar); printed for "
                f"{MAIN_VARIANT} and {FP_VARIANT} only, "
                f"{len(tr) - len(keep)} further rows for the ablation "
                "variants are returned by frontier() --")
            add(_fmt(keep))
    except Exception as e:
        add(f"  frontier failed: {type(e).__name__}: {e}")

    # ---- 3. primary interaction -----------------------------------------
    add(_sec("3. PRIMARY INFERENTIAL QUANTITY -- origin x designer_family "
             "(design.md §6)"))
    def _emit_interaction(mm, label):
        add(f"\n  --- {label} ---")
        if mm.get("status") != "ok":
            add(f"  insufficient data: {mm.get('reason')}")
            if mm.get("cells"):
                add(f"  cells present: {mm['cells']}")
            return
        add(f"  model    : {mm['method']}")
        if mm.get("fallback_note"):
            add(f"  NOTE     : {mm['fallback_note']}")
        if mm.get("warning"):
            add(f"  WARNING  : {mm['warning']}")
        add(f"  formula  : {mm['formula']}")
        add(f"  subset   : signal={mm['signal']}, variant={mm['variant']}, "
            f"collateral budget={mm['budget']}, arm={mm['arm']}")
        add(f"  term     : {mm['term']}")
        add(f"  estimate : {mm['estimate']:+.4f}   "
            f"95% CI {_fmt_ci(mm['ci95'])}   p = {mm['pvalue']:.4g}")
        add(f"  n        : {mm['n']} units over {mm['n_seeds']} seeds "
            f"(random intercept | {mm.get('group_var')}) {mm['cells']}")
        add("  cell means (dBias):")
        for k, v in sorted(mm["cell_means"].items()):
            add(f"      {k:24s} {v:+.4f}")

    try:
        _emit_interaction(primary_interaction(df), "POOLED over arms")
        arms = sorted(df["arm"].astype(str).unique().tolist())
        if len(arms) > 1:
            for a in arms:
                _emit_interaction(primary_interaction(df, arm=a),
                                  f"ARM = {a}")
        add("\n  orientation: positive = same-family does relatively better "
            "on ACQUIRED than on INHERITED bias, vs cross-family (H2).")
    except Exception as e:
        add(f"  primary_interaction failed: {type(e).__name__}: {e}")

    # ---- 4. bootstrap ----------------------------------------------------
    add(_sec("4. BOOTSTRAP INTERACTION (resampling seeds AND designers)"))
    try:
        bs = bootstrap_interaction(df, n_boot=n_boot, seed=seed)
        if bs.get("status") != "ok":
            add(f"  insufficient data: {bs.get('reason')}")
            if bs.get("cells_present"):
                add(f"  cells present: {bs['cells_present']}")
        else:
            add("  contrast : [dBias(acq,same) - dBias(inh,same)] "
                "- [dBias(acq,cross) - dBias(inh,cross)]")
            add(f"  point    : {bs['point_estimate']:+.4f}")
            add(f"  boot mean: {bs['mean']:+.4f}   95% CI {_fmt_ci(bs['ci95'])}"
                f"   (n_boot={bs['n_boot_valid']}/{bs['n_boot']})")
            add(f"  P(boot > 0) = {bs['frac_gt0']:.3f}   "
                f"P(boot < 0) = {bs['frac_lt0']:.3f}")
            add(f"  units    : {bs['n_units']} ({bs['n_seeds']} seeds, "
                f"{bs['n_designers']} designers), budget={bs['budget']}")
            if bs.get("warning"):
                add(f"  WARNING  : {bs['warning']}")
    except Exception as e:
        add(f"  bootstrap_interaction failed: {type(e).__name__}: {e}")

    # ---- 4b. EXPERIMENT A (experiments_v2.md) ----------------------------
    add(_sec("4b. EXPERIMENT A -- ARM REPLICATION (make-or-break; "
             "experiments_v2.md)"))
    try:
        ea = experiment_a_report(df, n_boot=n_boot, seed=seed)
        if ea.get("status") != "ok":
            add(f"  insufficient data: {ea.get('reason')}")
        else:
            add(f"  arms: {ea['arms']}  (n={ea['n_arms']}, "
                f"A2/A3 need >= {ea['arms_needed']} arms)")
            add("\n  PER-ARM FOREST of the origin x designer_family "
                "interaction (pooled-only reporting is banned, §N):")
            for ln in _forest_lines(ea["forest"]):
                add(ln)
            add("\n  tidy table (also returned by per_arm_interactions() for "
                "the figures script):")
            cols = [c for c in ("scope", "arm", "status", "n", "n_seeds",
                                "estimate", "ci_lo", "ci_hi", "pvalue",
                                "positive", "ci_excludes_0")
                    if c in ea["forest"].columns]
            add(_fmt(ea["forest"][cols]))

            a1 = ea["A1"]
            add("\n  A1 pooled interaction (ARM AS RANDOM EFFECT):")
            if a1.get("status") == "ok":
                add(f"     estimate {a1['estimate']:+.4f}  95% CI "
                    f"{_fmt_ci(a1['ci95'])}  p={a1['pvalue']:.4g}  "
                    f"n={a1['n']}  [{a1['method']}]")
                add(f"     -> A1 {'PASS' if a1['passed'] else 'FAIL'}")
            else:
                add(f"     insufficient data ({a1.get('reason')})")

            a2 = ea["A2"]
            add(f"\n  A2 per-arm replication: positive in "
                f"{a2['n_positive']}/{a2['n_arms']} arms "
                f"(need >= {a2['needed_positive']}), CI excludes 0 in "
                f"{a2['n_ci_excludes_0']} (need >= {a2['needed_ci']})")
            add(f"     positive arms: {a2['arms_positive']}   "
                f"CI-significant arms: {a2['arms_ci']}   "
                f"computable: {a2['n_computable']}/{a2['n_arms']}")
            add(f"     -> A2 {'PASS' if a2['passed'] else ('FAIL' if a2['passed'] is False else 'N/A')}")

            a3 = ea["A3"]
            add("\n  A3 blind-spot null-equivalence -- (same - partition-null) "
                "on INHERITED, per arm:")
            for a, g in sorted(a3["per_arm"].items()):
                if g.get("status") == "insufficient data":
                    add(f"     {a:>8s}  insufficient data ({g.get('reason')})")
                else:
                    at = ("AT CHANCE" if (np.isfinite(g['ci95'][0])
                                          and g['ci95'][0] <= 0 <= g['ci95'][1])
                          else "above null")
                    add(f"     {a:>8s}  {g['point_estimate']:+.4f}  95% CI "
                        f"{_fmt_ci(g['ci95'])}  -> {at}")
            add(f"     at chance in {a3['n_at_chance']}/{a3['n_computable']} "
                f"computable arms (need >= {a3['needed']})")
            add(f"     -> A3 {'PASS' if a3['passed'] else ('FAIL' if a3['passed'] is False else 'N/A')}")

            a4 = ea["A4"]
            add("\n  A4 acquired non-specificity -- pooled (cross - same) on "
                "ACQUIRED:")
            if a4.get("status") == "ok":
                add(f"     {a4['point_estimate']:+.4f}  95% CI "
                    f"{_fmt_ci(a4['ci95'])}  P(>0)={a4['frac_gt0']:.3f}")
                add(f"     -> A4 {'PASS (covers 0, effect is inherited-specific)' if a4['passed'] else 'FAIL (cross beats same on acquired too)'}")
            else:
                add(f"     insufficient data ({a4.get('reason')})")

            dm = ea["demote"]
            add(f"\n  KILL / DEMOTE: triggered={dm['triggered']}")
            add(f"     {dm['verdict']}")
    except Exception as e:
        add(f"  experiment_a_report failed: {type(e).__name__}: {e}")

    # ---- 5. mechanism ----------------------------------------------------
    add(_sec("5. MECHANISM TEST -- endogenous vs exogenous signal "
             "(design.md §5.6★, H3)"))
    try:
        mt = mechanism_test(df, n_boot=n_boot, seed=seed)
        if mt.get("status") != "ok":
            add(f"  insufficient data: {mt.get('reason')}")
        else:
            add(f"  budget: {mt['budget']}")
            if mt.get("warning"):
                add(f"  WARNING: {mt['warning']}")
            add(_fmt(mt["table"]))
            add("\n  gaps (bootstrap over seeds and designers):")
            for k, g in sorted(mt["gaps"].items()):
                if g.get("status") == "insufficient data":
                    add(f"    {k:34s} insufficient data ({g.get('reason')})")
                else:
                    add(f"    {k:34s} {g['point_estimate']:+.4f}  "
                        f"95% CI {_fmt_ci(g['ci95'])}  "
                        f"P(>0)={g['frac_gt0']:.3f}")
            h3 = mt.get("h3", {})
            if h3.get("status") == "insufficient data":
                add(f"\n  H3 verdict: insufficient data ({h3.get('reason')})")
            else:
                add(f"\n  H3 verdict: blind spot present (endogenous) = "
                    f"{h3['blind_spot_present_endogenous']}; "
                    f"closed under exogenous = {h3['blind_spot_closed_exogenous']}"
                    f"  => H3 supported = {h3['supported']}")
    except Exception as e:
        add(f"  mechanism_test failed: {type(e).__name__}: {e}")

    # ---- 5b. random-partition null ---------------------------------------
    add(_sec("5b. RANDOM-PARTITION NULL (designer_role='random')"))
    add("  Same sentence pool, partition drawn at random, NO designer. If the "
        "null\n  matches the elicited designers, no designer-family claim is "
        "interpretable.")
    try:
        rn = random_null_test(df, n_boot=n_boot, seed=seed)
        if rn.get("table") is not None and len(rn.get("table", [])):
            add(f"\n  frontier dBias by origin x family factor "
                f"(budget={rn.get('budget')}):")
            add(_fmt(rn["table"]))
        if rn.get("status") != "ok":
            add(f"\n  insufficient data: {rn.get('reason')}")
        else:
            if rn.get("warning"):
                add(f"  WARNING: {rn['warning']}")
            add("\n  gaps vs the null (bootstrap over seeds and designers):")
            for k, g in sorted(rn["gaps"].items()):
                if g.get("status") == "insufficient data":
                    add(f"    {k:36s} insufficient data ({g.get('reason')})")
                else:
                    add(f"    {k:36s} {g['point_estimate']:+.4f}  "
                        f"95% CI {_fmt_ci(g['ci95'])}  "
                        f"P(>0)={g['frac_gt0']:.3f}")
    except Exception as e:
        add(f"  random_null_test failed: {type(e).__name__}: {e}")

    # ---- 5c. arm x designer_family ---------------------------------------
    add(_sec("5c. ARM x DESIGNER_FAMILY -- bidirectional control "
             "(design.md §5.1)"))
    try:
        aa = arm_asymmetry(df, n_boot=n_boot, seed=seed)
        if aa.get("status") != "ok":
            add(f"  insufficient data: {aa.get('reason')}")
        else:
            add(f"  inherited row, frontier dBias (budget={aa['budget']}), "
                f"arms present: {aa['arms']}")
            add(_fmt(aa["table"]))
            if aa.get("warning"):
                add(f"\n  WARNING: {aa['warning']}")
            add("\n  relation view -- (cross - same) per arm "
                "(must be POSITIVE IN EVERY ARM):")
            for a, g in sorted(aa["per_arm"].items()):
                if g.get("status") == "insufficient data":
                    add(f"    arm={a:10s} insufficient data ({g.get('reason')})")
                else:
                    add(f"    arm={a:10s} {g['point_estimate']:+.4f}  "
                        f"95% CI {_fmt_ci(g['ci95'])}  "
                        f"P(>0)={g['frac_gt0']:.3f}")
            if aa.get("lineage_table") is not None and len(aa["lineage_table"]):
                add("\n  lineage view -- mean dBias by arm x designer lineage:")
                add(_fmt(aa["lineage_table"]))
            add("\n  does each lineage's advantage FLIP when the arm flips?")
            for lin, v in sorted(aa["lineage"].items()):
                if v.get("status") != "ok":
                    add(f"    {lin:12s} not testable ({v.get('reason')})")
                else:
                    add(f"    {lin:12s} as-cross {v['dBias_as_cross']:+.4f} "
                        f"vs as-same {v['dBias_as_same']:+.4f}  "
                        f"flip={v['flip']:+.4f} "
                        f"({'FLIPS as predicted' if v['flip'] > 0 else 'NO FLIP'})")
            add(f"\n  verdict: relation-consistent across arms="
                f"{aa['relation_consistent']} "
                f"(CI-strict={aa['relation_consistent_ci']}), "
                f"lineage flips={aa['lineage_flips']} "
                f"over {aa['n_lineages_testable']} testable lineage(s)")
    except Exception as e:
        add(f"  arm_asymmetry failed: {type(e).__name__}: {e}")

    # ---- 5d. direction alignment -----------------------------------------
    add(_sec("5d. DIRECTION ALIGNMENT -- cosine(v_role, v_exogenous_gt) "
             "from the 20k-d sketches"))
    add("  Did the designer find the RIGHT direction, not merely a large one?")
    try:
        da = direction_alignment(df, sketch_dir=sketch_dir, n_boot=0)
        if da.get("status") != "ok":
            add(f"  insufficient data: {da.get('reason')}"
                f"  (dir={da.get('sketch_dir')}, files={da.get('n_files', 0)})")
        else:
            add(f"  sketches: {da['n_files']} files, {da['n_references']} "
                f"gt references, {da['n_aligned']} aligned, "
                f"{da['n_unreadable']} unreadable/mid-write")
            if len(da.get("by_factor", [])):
                add("\n  mean cosine to the ground-truth direction, by family "
                    "factor:")
                add(_fmt(da["by_factor"]))
            if len(da.get("by_origin_factor", [])):
                add("\n  ... by origin x family factor:")
                add(_fmt(da["by_origin_factor"]))
            corr = da.get("correlation")
            if isinstance(corr, dict) and corr.get("status") == "insufficient data":
                add(f"\n  alignment-vs-removal: insufficient data "
                    f"({corr.get('reason')})")
            elif isinstance(corr, dict):
                add(f"\n  alignment vs frontier bias reduction "
                    f"(n_merged={da.get('n_merged')}):")
                for scope, r in corr.items():
                    if r.get("status") != "ok":
                        add(f"    {scope:16s} insufficient data "
                            f"({r.get('reason')})")
                    else:
                        add(f"    {scope:16s} n={r['n']:3d}  "
                            f"pearson r={r['pearson_r']:+.3f} "
                            f"{_fmt_ci(r['pearson_ci95'])} "
                            f"p={r['pearson_p']:.3g}  |  spearman rho="
                            f"{r['spearman_r']:+.3f} "
                            f"{_fmt_ci(r['spearman_ci95'])} "
                            f"p={r['spearman_p']:.3g}")
    except Exception as e:
        add(f"  direction_alignment failed: {type(e).__name__}: {e}")

    # ---- 6. blind-spot correlation ---------------------------------------
    add(_sec("6. BLIND-SPOT CORRELATION -- does elicited contrast predict "
             "removal? (design.md §12/§8)"))
    try:
        bc = blind_spot_correlation(df)
        if bc.get("status") != "ok":
            add(f"  insufficient data: {bc.get('reason')}")
        else:
            if bc.get("note"):
                add(f"  NOTE: {bc['note']}")
            add(f"  units: {bc['n_units']}")
            for p, blocks in bc["correlations"].items():
                add(f"\n  predictor: {p}")
                if blocks.get("status") == "insufficient data":
                    add(f"    insufficient data ({blocks.get('reason')})")
                    continue
                for scope, r in blocks.items():
                    if r.get("status") != "ok":
                        add(f"    {scope:12s} insufficient data "
                            f"({r.get('reason')})")
                        continue
                    add(f"    {scope:12s} n={r['n']:3d}  "
                        f"pearson r={r['pearson_r']:+.3f} "
                        f"{_fmt_ci(r['pearson_ci95'])} p={r['pearson_p']:.3g}  |  "
                        f"spearman rho={r['spearman_r']:+.3f} "
                        f"{_fmt_ci(r['spearman_ci95'])} p={r['spearman_p']:.3g}")
    except Exception as e:
        add(f"  blind_spot_correlation failed: {type(e).__name__}: {e}")

    # ---- 7. kill switches ------------------------------------------------
    add(_sec("7. PRE-REGISTERED DECISION CRITERIA (design.md §7)"))
    try:
        ks = kill_switch_report(df, n_boot=n_boot, seed=seed)
        for name, c in ks.items():
            if name.startswith("_"):
                continue
            status = ("PASS" if c["passed"] is True
                      else "FAIL" if c["passed"] is False else "N/A ")
            val = _num(c["value"])
            add(f"  [{status}] {name}")
            add(f"          value     : {val}")
            add(f"          threshold : {c['threshold']}")
            if c.get("note"):
                add(f"          note      : {c['note']}")
        h1 = ks.get("_h1", {})
        if h1.get("status") == "ok":
            add(f"\n  H1 detail (matched COLLATERAL -- the §7 gate):")
            add(f"    frontier dBias  binary={h1['mean_bin']:+.4f} "
                f"(mean best alpha {h1['mean_alpha_bin']:.2f})  vs  "
                f"fp={h1['mean_fp']:+.4f} "
                f"(mean best alpha {h1['mean_alpha_fp']:.2f})")
            add(f"    retention       median={h1['median_ratio']:.3f}  "
                f"IQR=[{h1['iqr'][0]:.3f}, {h1['iqr'][1]:.3f}]  "
                f"range=[{h1['ratio_min']:.3f}, {h1['ratio_max']:.3f}]  "
                f"ratio-of-means={h1['ratio_of_means']:.3f}")
            add(f"    conditions      {h1['n_used']} used of "
                f"{h1['n_conditions']} "
                f"({h1['n_dropped_fp_nonpositive']} dropped: fp_best <= 0)"
                f"   budget={h1['budget']}")
            if np.isfinite(h1["bytes_median"]):
                add(f"    edit size       median "
                    f"{h1['bytes_median'] / 1024.0:.1f} KB")
            add(f"    arm coverage    with fp: {h1.get('arms_with_fp')}; "
                f"WITHOUT fp (skipped, H1 settled per experiments_v2.md §1): "
                f"{h1.get('arms_without_fp')}")
            for a, v in sorted(h1.get("per_arm", {}).items()):
                if v.get("status") == "ok":
                    add(f"      {a:>8s}  median ratio={v['median_ratio']:.3f} "
                        f"ratio-of-means={v['ratio_of_means']:.3f}  "
                        f"n={v['n_used']} (dropped {v['n_dropped_fp_nonpositive']})")
                else:
                    add(f"      {a:>8s}  {v['status']}")
            sec = h1.get("matched_alpha_secondary", {})
            if sec.get("status") == "ok":
                add(f"    [secondary diagnostic, CONFOUNDED -- matched alpha, "
                    f"n={sec['n_pairs']}]: ratio-of-means="
                    f"{sec['ratio_of_means']:.3f}, median pairwise="
                    f"{sec['median_pairwise_ratio']:.3f}")
    except Exception as e:
        add(f"  kill_switch_report failed: {type(e).__name__}: {e}")

    add("\n" + "=" * 78)
    add("end of report")
    add("=" * 78)
    text = "\n".join(L)
    if write:
        _write(text)
    return text


def _write(text):
    try:
        os.makedirs(RESULTS, exist_ok=True)
        with open(REPORT_PATH, "w") as f:
            f.write(text + "\n")
    except Exception as e:                                    # pragma: no cover
        print(f"(could not write {REPORT_PATH}: {e})")


if __name__ == "__main__":
    print(report())
    print(f"\n[written to {REPORT_PATH}]")
