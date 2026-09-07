"""Figures for the 2x2 blind-spot experiment (design.md §5.1, §5.4, §5.6, §7, §12).

Read-only over results/runs.jsonl and results/sketches/*.npy. Safe to run while
run_experiment.py is still appending: every figure is individually guarded, so a
cell that has not been run yet produces a "SKIP figN: <reason>" line instead of
an exception, and the remaining figures are still written.

    fig1_frontier.png          bias removed vs collateral cost, traced over alpha (§5.4)
    fig2_2x2.png               the 2x2 itself, frontier bias reduction (§5.1 / §6)
    fig3_bidirectional.png     same-vs-cross ordering per arm -- does it flip? (§5.1)
    fig4_blindspot_scatter.png elicited contrast vs removal (§12)
    fig5_bitbudget.png         bit-budget sweep (§5.6)
    fig6_alignment.png         cosine(v_designer, v_groundtruth) (§5.6 mechanism)

Loading, the collateral budgets and the frontier definition are imported from
analyze.py -- this module only draws.
"""
import os
import math
import textwrap

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyze import (load_runs, frontier, best_within_budget,
                     BUDGETS, MAIN_VARIANT, FP_VARIANT, SKETCH_DIR,
                     direction_alignment)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGDIR = os.path.join(RESULTS, "figures")

DPI = 150
PPL_BUDGET = BUDGETS["strict"]["ppl_ratio_max"]     # 1.10
MMLU_BUDGET = BUDGETS["strict"]["d_mmlu_min"]       # -0.02

# Okabe-Ito, colourblind-safe. One colour per designer_family_factor, used
# identically in every figure.
FAMILY_COLORS = {
    "same":      "#D55E00",   # vermillion
    "cross":     "#0072B2",   # blue
    "null":      "#999999",   # grey  (random-partition control)
    "exogenous": "#009E73",   # bluish green (ground-truth signal)
    "n/a":       "#CC79A7",
}
FAMILY_ORDER = ["same", "cross", "null", "exogenous"]
FAMILY_LABEL = {
    "same": "same family (self / sibling)",
    "cross": "cross family",
    "null": "random-partition null",
    "exogenous": "exogenous ground truth",
    "n/a": "unclassified",
}
ORIGIN_ORDER = ["inherited", "acquired"]
ORIGIN_MARKER = {"inherited": "o", "acquired": "^"}

_written = []


class Skip(Exception):
    """Raised inside a figure builder when the needed cells are absent."""


def _color(f):
    return FAMILY_COLORS.get(str(f), "#444444")


def _fams(values):
    """Family factors present, in the canonical order (extras appended)."""
    have = [str(v) for v in pd.unique(pd.Series(list(values)).dropna())]
    return ([f for f in FAMILY_ORDER if f in have]
            + [f for f in have if f not in FAMILY_ORDER])


def _wrap(text, width):
    """Wrap each explicit line so long titles never run off the canvas."""
    return "\n".join(textwrap.fill(ln, width) if ln.strip() else ln
                     for ln in text.split("\n"))


def _save(fig, name, summary, adjust=None):
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    if adjust:                      # must come AFTER tight_layout, not before
        fig.subplots_adjust(**adjust)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    _written.append(path)
    print(f"WROTE {name}: {summary}")


def _unit_frontier(df, variant=MAIN_VARIANT, budget="strict", signal=None,
                   factors=None, origins=None, arms=None):
    """One frontier point per (arm, seed, origin, designer_role, designer):
    the largest bias reduction that still fits the collateral budget (§5.4)."""
    d = df
    if variant is not None:
        d = d[d["variant"] == variant]
    if signal is not None:
        d = d[d["signal"] == signal]
    if factors is not None:
        d = d[d["designer_family_factor"].isin(list(factors))]
    if origins is not None:
        d = d[d["origin"].isin(list(origins))]
    if arms is not None:
        d = d[d["arm"].astype(str).isin([str(a) for a in arms])]
    if d.empty:
        return d
    cols = [c for c in ("arm", "seed", "origin", "designer_role", "designer")
            if c in d.columns]
    if not cols:
        return d.iloc[0:0]
    return best_within_budget(d, cols, budget)


def _mean_sem(sub, value="bias_reduction"):
    """Mean over units, SEM across seeds (units are averaged within a seed
    first, so a seed with many designers does not inflate the precision)."""
    s = sub[sub[value].notna()]
    if s.empty:
        return float("nan"), float("nan"), 0, 0
    if "seed" in s.columns and s["seed"].notna().any():
        per_seed = s.groupby("seed")[value].mean()
    else:
        per_seed = pd.Series([s[value].mean()])
    n_seeds = int(per_seed.size)
    m = float(per_seed.mean())
    sem = (float(per_seed.std(ddof=1) / math.sqrt(n_seeds))
           if n_seeds >= 2 else float("nan"))
    return m, sem, int(len(s)), n_seeds


def _bar_panel(ax, units, group_col, groups, fams, ylabel=True, single=False):
    """Grouped bars: x groups = `groups` of `group_col`, bars = family factor.
    With single=True the x axis IS the family factor (one bar per tick).
    Returns True if anything was drawn."""
    width = 0.8 / max(len(fams), 1)
    drew = False
    if single:
        for j, f in enumerate(groups):
            sub = units[units["designer_family_factor"].astype(str) == str(f)]
            m, sem, n, n_seeds = _mean_sem(sub)
            if not np.isfinite(m):
                continue
            drew = True
            e = sem if np.isfinite(sem) else 0.0
            ax.bar([j], [m], width=0.62, color=_color(f), edgecolor="white",
                   yerr=[e], capsize=3,
                   error_kw=dict(ecolor="#333333", lw=1.0), zorder=3)
            ax.annotate(f"n={n}/{n_seeds}s", (j, m + (e if m >= 0 else -e)),
                        textcoords="offset points",
                        xytext=(0, 4 if m >= 0 else -12), ha="center",
                        fontsize=7, color="#333333")
        ax.axhline(0, color="#333333", lw=0.8, zorder=2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.margins(y=0.14)          # room for the n= labels
        if ylabel:
            ax.set_ylabel("frontier bias reduction\n"
                          "|bias$_{pre}$| − |bias$_{post}$| (higher = more removed)")
        return drew
    for i, f in enumerate(fams):
        xs, ys, es, ns = [], [], [], []
        for j, g in enumerate(groups):
            sub = units[(units[group_col].astype(str) == str(g))
                        & (units["designer_family_factor"].astype(str) == f)]
            m, sem, n, n_seeds = _mean_sem(sub)
            xs.append(j - 0.4 + width * (i + 0.5))
            ys.append(m if np.isfinite(m) else np.nan)
            es.append(sem if np.isfinite(sem) else 0.0)
            ns.append((n, n_seeds))
        if not np.any(np.isfinite(ys)):
            continue
        drew = True
        ax.bar(xs, ys, width=width * 0.9, color=_color(f),
               edgecolor="white", linewidth=0.6,
               yerr=es, capsize=3,
               error_kw=dict(ecolor="#333333", lw=1.0), label=FAMILY_LABEL[f]
               if f in FAMILY_LABEL else f, zorder=3)
        for x, y, e, (n, n_seeds) in zip(xs, ys, es, ns):
            if not np.isfinite(y):
                continue
            ax.annotate(f"n={n}/{n_seeds}s", (x, y + (e if y >= 0 else -e)),
                        textcoords="offset points",
                        xytext=(0, 4 if y >= 0 else -12), ha="center",
                        fontsize=6.5, color="#333333")
    ax.axhline(0, color="#333333", lw=0.8, zorder=2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups])
    ax.grid(axis="y", color="#DDDDDD", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(y=0.14)              # room for the n= labels
    if ylabel:
        ax.set_ylabel("frontier bias reduction\n|bias$_{pre}$| - |bias$_{post}$| (higher = more removed)")
    return drew


def _empty_panel(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9,
            color="#777777", style="italic", transform=ax.transAxes, wrap=True)
    # tick_params only -- set_xticks([]) would propagate through a shared axis
    # and strip the tick labels off the populated panel as well.
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_color("#CCCCCC")


# --------------------------------------------------------------------------
# fig 1 -- the frontier (design.md §5.4: "a frontier, not a point")
# --------------------------------------------------------------------------
def fig1_frontier(df):
    d = df[df["variant"] == MAIN_VARIANT]
    d = d[d["bias_reduction"].notna() & d["ppl_ratio"].notna()]
    if d.empty:
        raise Skip(f"no rows for the main binary variant ({MAIN_VARIANT}) "
                   "with both bias_reduction and ppl_ratio")
    fr = frontier(d, group_cols=("origin", "designer_family_factor"))
    tr = fr.get("trace")
    if tr is None or len(tr) == 0:
        raise Skip("frontier() returned an empty alpha trace")

    origins_present = [o for o in ORIGIN_ORDER
                       if (tr["origin"].astype(str) == o).any()]
    if not origins_present:
        raise Skip("no rows with origin in {inherited, acquired}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    fams_all = []
    for ax, origin in zip(axes, ORIGIN_ORDER):
        sub = tr[tr["origin"].astype(str) == origin]
        ax.set_title(f"origin = {origin}", fontsize=11, fontweight="bold")
        ax.set_xlabel("collateral cost:  post/pre perplexity ratio  (1.0 = free)")
        if len(sub) == 0:
            _empty_panel(ax, f"no {origin}-bias runs yet\n(cell not populated)")
            continue
        fams = _fams(sub["designer_family_factor"])
        fams_all += fams
        for f in fams:
            s = sub[sub["designer_family_factor"].astype(str) == f
                    ].sort_values("alpha")
            ax.plot(s["ppl_ratio"], s["dBias"], marker="o", ms=4.5, lw=1.8,
                    color=_color(f), label=FAMILY_LABEL.get(f, f), zorder=3)
            for _, r in s.iterrows():
                if np.isfinite(r["ppl_ratio"]) and np.isfinite(r["dBias"]):
                    ax.annotate(f"α={r['alpha']:g}", (r["ppl_ratio"], r["dBias"]),
                                textcoords="offset points", xytext=(4, 3),
                                fontsize=6.5, color=_color(f))
        ax.axvline(PPL_BUDGET, color="#333333", ls="--", lw=1.2, zorder=2)
        ax.axhline(0, color="#888888", lw=0.8, zorder=1)
        ax.annotate(f"strict collateral budget\nppl ratio ≤ {PPL_BUDGET:g}",
                    (PPL_BUDGET, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                    color="#333333", rotation=90, va="bottom")
        ax.grid(color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("bias reduction  |bias$_{pre}$| - |bias$_{post}$|\n(higher = more bias removed)")

    handles = [Line2D([], [], color=_color(f), marker="o", ms=4.5, lw=1.8,
                      label=FAMILY_LABEL.get(f, f))
               for f in _fams(fams_all)]
    handles.append(Line2D([], [], color="#333333", ls="--", lw=1.2,
                          label=f"strict budget (ppl ≤ {PPL_BUDGET:g})"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               frameon=False, fontsize=8.5)
    fig.suptitle(_wrap(
        "Fig 1 — Debiasing is a frontier, not a point (design.md §5.4)\n"
        "Each point is one edit strength α; read up-and-left as better. H2 predicts "
        "cross-family (blue) sits above same-family (orange) on inherited bias, and "
        "the two lie on top of each other on acquired bias.", 105), fontsize=10)
    _save(fig, "fig1_frontier.png",
          f"alpha traces for origins={origins_present}, "
          f"families={_fams(tr['designer_family_factor'])}, "
          f"{tr['alpha'].nunique()} alpha levels",
          adjust=dict(bottom=0.20, top=0.82))


# --------------------------------------------------------------------------
# fig 2 -- the 2x2 itself
# --------------------------------------------------------------------------
def fig2_2x2(df):
    units = _unit_frontier(df, variant=MAIN_VARIANT, budget="strict")
    budget_name = "strict"
    if len(units) == 0:
        units = _unit_frontier(df, variant=MAIN_VARIANT, budget="loose")
        budget_name = "loose (strict budget was empty)"
    if len(units) == 0:
        raise Skip("no (arm, seed, origin, designer) unit meets any collateral "
                   f"budget for variant {MAIN_VARIANT}")
    fams = _fams(units["designer_family_factor"])
    origins = [o for o in ORIGIN_ORDER
               if (units["origin"].astype(str) == o).any()]
    if not origins:
        raise Skip("no unit with origin in {inherited, acquired}")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    drew = _bar_panel(ax, units, "origin", ORIGIN_ORDER, fams)
    if not drew:
        plt.close(fig)
        raise Skip("no populated origin x family cell")
    missing = [o for o in ORIGIN_ORDER if o not in origins]
    for j, o in enumerate(ORIGIN_ORDER):
        if o in missing:
            ax.annotate("not run yet", (j, 0), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=9,
                        color="#777777", style="italic")
    ax.set_xlabel("bias origin")
    fig.legend(loc="lower center", ncol=min(len(fams), 4), frameon=False,
               fontsize=8.5)
    n_cells = int(units.groupby(["origin", "designer_family_factor"]).ngroups)
    ax.set_title(_wrap(
        "Fig 2 — The 2×2: best bias reduction inside the collateral budget\n"
        f"budget = {budget_name} (ΔMMLU ≥ {MMLU_BUDGET:+.2f}, ppl ratio ≤ "
        f"{PPL_BUDGET:g}); variant {MAIN_VARIANT}; error bars = SEM across seeds\n"
        "H2: the orange 'same' bar collapses only on inherited bias; the null "
        "(grey) bar must stay near zero or nothing here is interpretable.", 88),
        fontsize=9.5)
    _save(fig, "fig2_2x2.png",
          f"{n_cells}/4+ origin×family cells populated, families={fams}, "
          f"origins={origins}, budget={budget_name}",
          adjust=dict(bottom=0.16))


# --------------------------------------------------------------------------
# fig 3 -- bidirectional control (design.md §5.1)
# --------------------------------------------------------------------------
def fig3_bidirectional(df):
    factors = ("same", "cross", "null")
    units = _unit_frontier(df, variant=MAIN_VARIANT, budget="strict",
                           factors=factors, origins=("inherited",))
    budget_name = "strict"
    if len(units) == 0:
        units = _unit_frontier(df, variant=MAIN_VARIANT, budget="loose",
                               factors=factors, origins=("inherited",))
        budget_name = "loose (strict budget was empty)"
    if len(units) == 0:
        raise Skip("no inherited-origin unit with a same/cross/null designer "
                   "meets any collateral budget")
    arms = sorted(units["arm"].astype(str).unique().tolist())
    fams = _fams(units["designer_family_factor"])

    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms) + 1.6, 4.9),
                             sharey=True, squeeze=False)
    axes = axes[0]
    gaps = {}
    for k, (ax, arm) in enumerate(zip(axes, arms)):
        u = units[units["arm"].astype(str) == arm]
        _bar_panel(ax, u, "designer_family_factor", fams, fams,
                   ylabel=(k == 0), single=True)
        # x is the family factor itself here, so use readable labels
        ax.set_xticklabels([FAMILY_LABEL.get(f, f).split(" (")[0]
                            for f in fams], fontsize=8.5)
        same = u[u["designer_family_factor"] == "same"]["bias_reduction"]
        cross = u[u["designer_family_factor"] == "cross"]["bias_reduction"]
        if len(same) and len(cross):
            g = float(cross.mean() - same.mean())
            gaps[arm] = g
            ax.set_title(f"arm = {arm}  (target lineage = {arm})\n"
                         f"cross − same = {g:+.3f}",
                         fontsize=10, fontweight="bold")
        else:
            gaps[arm] = float("nan")
            miss = "cross" if not len(cross) else "same"
            ax.set_title(f"arm = {arm}  (target lineage = {arm})\n"
                         f"no {miss}-family designer yet",
                         fontsize=10, fontweight="bold")
        ax.set_xlabel("designer family (relative to target)")

    handles = [Patch(facecolor=_color(f), label=FAMILY_LABEL.get(f, f))
               for f in fams]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=8.5)
    ok = {a: g for a, g in gaps.items() if np.isfinite(g)}
    if len(ok) >= 2:
        verdict = ("cross > same in BOTH arms → the effect follows the *relation*, "
                   "not the lineage (capability confound rejected)"
                   if all(g > 0 for g in ok.values()) else
                   "the ordering does NOT hold in both arms → consistent with a "
                   "capability confound, not a blind spot")
    else:
        verdict = (f"only {len(arms)} arm present ({', '.join(arms)}) — the flip "
                   "cannot be evaluated until the other arm runs")
    fig.suptitle(_wrap(
        "Fig 3 — Bidirectional control (design.md §5.1): inherited bias only\n"
        "Does the same-vs-cross ordering survive swapping which lineage is the "
        "target?\n" + verdict, 95), fontsize=10)
    _save(fig, "fig3_bidirectional.png",
          f"arms={arms}, families={fams}, cross−same gaps="
          + ", ".join(f"{a}:{g:+.3f}" for a, g in gaps.items())
          + f", budget={budget_name}",
          adjust=dict(bottom=0.20, top=0.80))


# --------------------------------------------------------------------------
# fig 4 -- does measured contrast predict removal? (design.md §12)
# --------------------------------------------------------------------------
def fig4_blindspot_scatter(df):
    if "elicit_contrast_gap" not in df.columns:
        raise Skip("column elicit_contrast_gap absent")
    units = _unit_frontier(df, variant=MAIN_VARIANT, budget="strict",
                           signal="endogenous")
    budget_name = "strict"
    if len(units) == 0:
        units = _unit_frontier(df, variant=MAIN_VARIANT, budget="loose",
                               signal="endogenous")
        budget_name = "loose (strict budget was empty)"
    if len(units) == 0:
        raise Skip("no endogenous unit meets any collateral budget")
    u = units[units["elicit_contrast_gap"].notna()
              & units["bias_reduction"].notna()]
    if len(u) < 3:
        raise Skip(f"only {len(u)} endogenous unit(s) with an elicitation "
                   "contrast — need ≥3 for a scatter/regression")
    # the random-partition null has no elicitation, so it cannot inform a
    # question about what the elicitation predicts -- shown, but not fitted
    fit = u[u["designer_family_factor"].astype(str) != "null"]
    x = fit["elicit_contrast_gap"].to_numpy(float)
    y = fit["bias_reduction"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    has_null = False
    for f in _fams(u["designer_family_factor"]):
        for o in ORIGIN_ORDER:
            s = u[(u["designer_family_factor"].astype(str) == f)
                  & (u["origin"].astype(str) == o)]
            if s.empty:
                continue
            null = (f == "null")
            has_null |= null
            ax.scatter(s["elicit_contrast_gap"], s["bias_reduction"],
                       s=52, marker=ORIGIN_MARKER[o],
                       facecolor="none" if null else _color(f),
                       edgecolor=_color(f) if null else "white",
                       linewidth=1.4 if null else 0.8, zorder=3)

    title_r = f"Pearson r undefined (n={len(fit)} elicited units, or a constant column)"
    if len(fit) >= 3 and np.ptp(x) > 1e-12 and np.ptp(y) > 1e-12:
        from scipy import stats as _st
        r, p = _st.pearsonr(x, y)
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, b0 + b1 * xs, color="#333333", lw=1.6, ls="-", zorder=2)
        title_r = (f"Pearson r = {r:+.3f} (p = {p:.3g}, n = {len(fit)} "
                   f"elicited units, slope {b1:+.3f})")
    ax.axhline(0, color="#888888", lw=0.8, zorder=1)
    ax.axvline(0, color="#888888", lw=0.8, zorder=1)
    ax.set_xlabel("elicited contrast gap  (biased − debiased congruence; "
                  "how much bias the designer can see in itself)")
    ax.set_ylabel("frontier bias reduction  |bias$_{pre}$| − |bias$_{post}$|")
    ax.grid(color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], ls="", marker="s", ms=8, color=_color(f),
                      label=FAMILY_LABEL.get(f, f)
                      + (" — shown, not fitted" if f == "null" else ""))
               for f in _fams(u["designer_family_factor"])]
    handles += [Line2D([], [], ls="", marker=ORIGIN_MARKER[o], ms=8,
                       color="#555555", label=f"origin = {o}")
                for o in ORIGIN_ORDER if (u["origin"].astype(str) == o).any()]
    handles += [Line2D([], [], color="#333333", lw=1.6,
                       label="OLS fit (elicited designers only)")]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="best")
    ax.set_title(_wrap(
        "Fig 4 — Does measured 'inherited-ness' predict removal? (design.md §12)\n"
        + title_r + "\n"
        "One point per (arm, seed, origin, designer), endogenous signal, budget = "
        f"{budget_name}. A positive slope means the designers that can *see* the "
        "bias in themselves are the ones that remove it — the blind spot as a "
        "continuous curve.", 88), fontsize=9)
    _save(fig, "fig4_blindspot_scatter.png",
          f"n={len(u)} endogenous units, {title_r}, budget={budget_name}")


# --------------------------------------------------------------------------
# fig 5 -- bit-budget sweep (design.md §5.6)
# --------------------------------------------------------------------------
_SPARSE_TAGS = ("-s0.5", "-s0.9", "-s0.99")


def fig5_bitbudget(df):
    if "variant" not in df.columns or df.empty:
        raise Skip("no records")
    variants = [str(v) for v in df["variant"].dropna().unique()]
    sparse = sorted(v for v in variants
                    if any(t in v for t in _SPARSE_TAGS))
    if not sparse:
        raise Skip("no sparsity-ablation variants present "
                   f"(looked for {', '.join(_SPARSE_TAGS)} in {variants})")

    keep = sparse + [MAIN_VARIANT]
    units = _unit_frontier(df[df["variant"].isin(keep)], variant=None,
                           budget="strict")
    budget_name = "strict"
    if len(units) == 0:
        units = _unit_frontier(df[df["variant"].isin(keep)], variant=None,
                               budget="loose")
        budget_name = "loose (strict budget was empty)"
    if len(units) == 0:
        raise Skip("no sparsity-variant unit meets any collateral budget")
    # _unit_frontier collapses over variant, so redo the frontier per variant
    d = df[df["variant"].isin(keep)]
    cols = [c for c in ("arm", "seed", "origin", "designer_role", "designer",
                        "variant") if c in d.columns]
    units = best_within_budget(d, cols, "strict")
    if len(units) == 0:
        units = best_within_budget(d, cols, "loose")
        budget_name = "loose (strict budget was empty)"
    if len(units) == 0:
        raise Skip("no sparsity-variant unit meets any collateral budget")

    sp_col = ("edit_effective_sparsity" if "edit_effective_sparsity" in units
              else None)
    by_col = "edit_bytes" if "edit_bytes" in units else None
    if sp_col is None and by_col is None:
        raise Skip("neither edit_effective_sparsity nor edit_bytes recorded")

    fams = _fams(units["designer_family_factor"])
    panels = [c for c in (sp_col, by_col) if c is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels) + 0.8,
                                                      4.9), squeeze=False)
    axes = axes[0]
    for ax, col in zip(axes, panels):
        for f in fams:
            rows = []
            for v in keep:
                s = units[(units["variant"].astype(str) == v)
                          & (units["designer_family_factor"].astype(str) == f)]
                if s.empty or not s[col].notna().any():
                    continue
                m, sem, n, _ = _mean_sem(s)
                rows.append((float(s[col].mean()), m,
                             sem if np.isfinite(sem) else 0.0))
            if not rows:
                continue
            rows.sort()
            xs, ys, es = zip(*rows)
            xs = np.array(xs, float)
            if col == "edit_bytes":
                xs = xs / 1024.0
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, lw=1.8, capsize=3,
                        color=_color(f), label=FAMILY_LABEL.get(f, f), zorder=3)
        if col == "edit_bytes":
            ax.set_xscale("log")
            ax.set_xlabel("edit size (KiB, log scale) — design.md §5.4 'bytes of E'")
        else:
            ax.set_xlabel("effective sparsity (fraction of weights left unflipped)")
        ax.grid(color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#888888", lw=0.8, zorder=1)

    # horizontal references: dense binary and full-precision edits
    refs = []
    for ri, (vname, style, lab) in enumerate(
            ((MAIN_VARIANT, "--", "dense binary (sign-only)"),
             (FP_VARIANT, ":", "full-precision task-vector negation"))):
        dv = df[df["variant"] == vname]
        if dv.empty:
            continue
        uv = _unit_frontier(dv, variant=None, budget="strict")
        if len(uv) == 0:
            uv = _unit_frontier(dv, variant=None, budget="loose")
        if len(uv) == 0:
            continue
        m, _, n, _ = _mean_sem(uv)
        if not np.isfinite(m):
            continue
        refs.append((lab, m))
        for ax in axes:
            ax.axhline(m, color="#333333", ls=style, lw=1.3, zorder=2)
            # stagger the two labels so close-together references stay legible
            ax.annotate(f"{lab}: {m:.3f}", (0.01, m), xycoords=("axes fraction",
                        "data"), xytext=(0, 4 if ri == 0 else -11),
                        textcoords="offset points", fontsize=7.5,
                        color="#333333")

    axes[0].set_ylabel("frontier bias reduction  |bias$_{pre}$| − |bias$_{post}$|")
    handles = [Line2D([], [], color=_color(f), marker="o", lw=1.8,
                      label=FAMILY_LABEL.get(f, f)) for f in fams]
    handles += [Line2D([], [], color="#333333", ls="--", lw=1.3,
                       label="dense binary reference"),
                Line2D([], [], color="#333333", ls=":", lw=1.3,
                       label="full-precision reference")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               frameon=False, fontsize=8.5)
    fig.suptitle(_wrap(
        "Fig 5 — Bit-budget sweep (design.md §5.6): how few sign flips still debias?\n"
        f"Frontier bias reduction inside the {budget_name} collateral budget vs "
        "edit cost. A curve that stays flat out to high sparsity means the debias "
        "signal lives in a few bits; one that falls off a cliff kills the "
        "'few-flip' framing.", 100), fontsize=10)
    _save(fig, "fig5_bitbudget.png",
          f"sparsity variants={sparse}, families={fams}, "
          f"references={[f'{l}={m:.3f}' for l, m in refs]}, budget={budget_name}",
          adjust=dict(bottom=0.22, top=0.80))


# --------------------------------------------------------------------------
# fig 6 -- direction alignment against the ground-truth sketch
# --------------------------------------------------------------------------
def fig6_alignment(df):
    if not os.path.isdir(SKETCH_DIR):
        raise Skip(f"sketch directory {SKETCH_DIR} does not exist")
    res = direction_alignment(df, sketch_dir=SKETCH_DIR, variant=MAIN_VARIANT,
                              budget="strict")
    if res.get("status") != "ok":
        raise Skip(res.get("reason", "direction_alignment reported "
                                     "insufficient data"))
    al = res.get("alignments")
    if al is None or len(al) == 0:
        raise Skip("no sketch aligned to an exogenous/gt reference")
    non_ref = al[~al["is_reference"].astype(bool)]
    if non_ref.empty:
        raise Skip("only reference (gt) sketches present — nothing to align")

    fams = _fams(non_ref["designer_family_factor"])
    origins = [o for o in ORIGIN_ORDER
               if (non_ref["origin"].astype(str) == o).any()]
    merged = res.get("merged")
    have_scatter = (merged is not None and len(merged)
                    and "cos_to_gt" in merged.columns)

    n_bar = max(len(origins), 1)
    fig, axes = plt.subplots(1, n_bar + 1,
                             figsize=(4.3 * (n_bar + 1) + 0.6, 4.8),
                             squeeze=False)
    axes = axes[0]
    for k, (ax, o) in enumerate(zip(axes, origins)):
        sub = non_ref[non_ref["origin"].astype(str) == o]
        xs, ys, cs, labs = [], [], [], []
        for i, f in enumerate(fams):
            s = sub[sub["designer_family_factor"].astype(str) == f]
            if s.empty:
                continue
            xs.append(len(xs))
            ys.append(float(s["cos_to_gt"].mean()))
            cs.append(_color(f))
            labs.append(FAMILY_LABEL.get(f, f).split(" (")[0])
            ax.annotate(f"n={len(s)}", (xs[-1], ys[-1]),
                        textcoords="offset points",
                        xytext=(0, 3 if ys[-1] >= 0 else -11), ha="center",
                        fontsize=7, color="#333333")
        if xs:
            ax.bar(xs, ys, width=0.62, color=cs, edgecolor="white", zorder=3)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, fontsize=8.5)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)
        ax.margins(y=0.14)
        ax.set_title(f"origin = {o}", fontsize=10, fontweight="bold")
        ax.set_xlabel("designer family")
        if k == 0:
            ax.set_ylabel("cosine(v$_{designer}$, v$_{ground\\ truth}$)\n"
                          "(1 = found the right direction)")

    ax = axes[-1]
    if have_scatter:
        m = merged[merged["cos_to_gt"].notna()
                   & merged["bias_reduction"].notna()]
        m = m[~m["is_reference"].astype(bool)] if "is_reference" in m else m
        if len(m) == 0:
            _empty_panel(ax, "no unit has both a sketch\nand a frontier point")
        else:
            for f in _fams(m["designer_family_factor"]):
                for o in ORIGIN_ORDER:
                    s = m[(m["designer_family_factor"].astype(str) == f)
                          & (m["origin"].astype(str) == o)]
                    if s.empty:
                        continue
                    ax.scatter(s["cos_to_gt"], s["bias_reduction"], s=52,
                               color=_color(f), marker=ORIGIN_MARKER[o],
                               edgecolor="white", linewidth=0.8, zorder=3)
            xv = m["cos_to_gt"].to_numpy(float)
            yv = m["bias_reduction"].to_numpy(float)
            sub_t = f"n = {len(m)}"
            if len(m) >= 3 and np.ptp(xv) > 1e-12 and np.ptp(yv) > 1e-12:
                from scipy import stats as _st
                r, p = _st.pearsonr(xv, yv)
                b1, b0 = np.polyfit(xv, yv, 1)
                gx = np.linspace(xv.min(), xv.max(), 100)
                ax.plot(gx, b0 + b1 * gx, color="#333333", lw=1.6, zorder=2)
                sub_t = f"Pearson r = {r:+.3f} (p = {p:.3g}, n = {len(m)})"
            ax.axhline(0, color="#888888", lw=0.8)
            ax.set_xlabel("cosine(v$_{designer}$, v$_{ground\\ truth}$)")
            ax.set_ylabel("frontier bias reduction")
            ax.grid(color="#EEEEEE", lw=0.6)
            ax.set_axisbelow(True)
            ax.set_title("alignment vs removal\n" + sub_t, fontsize=10,
                         fontweight="bold")
    else:
        _empty_panel(ax, "no frontier point merges with a sketch yet")

    handles = [Patch(facecolor=_color(f), label=FAMILY_LABEL.get(f, f))
               for f in fams]
    handles += [Line2D([], [], ls="", marker=ORIGIN_MARKER[o], ms=8,
                       color="#555555", label=f"origin = {o}")
                for o in origins]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               frameon=False, fontsize=8.5)
    fig.suptitle(_wrap(
        "Fig 6 — Did the designer find the *right* direction, or just a big one?\n"
        "Cosine between each designer's contrast-direction sketch and the exogenous "
        "ground-truth direction, per (arm, seed, origin). The blind-spot mechanism "
        "predicts same-family designers are less aligned with ground truth on "
        "inherited bias — and that alignment, not norm, predicts removal.", 100),
        fontsize=9.5)
    _save(fig, "fig6_alignment.png",
          f"{len(non_ref)} non-reference sketches, {res.get('n_references', 0)} "
          f"reference(s), origins={origins}, families={fams}",
          adjust=dict(bottom=0.22, top=0.78))


# --------------------------------------------------------------------------
FIGURES = [
    ("fig1", fig1_frontier),
    ("fig2", fig2_2x2),
    ("fig3", fig3_bidirectional),
    ("fig4", fig4_blindspot_scatter),
    ("fig5", fig5_bitbudget),
    ("fig6", fig6_alignment),
]


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = load_runs()
    print(f"loaded {len(df)} rows from {df.attrs.get('source', '?')} "
          f"({df.attrs.get('n_malformed', 0)} malformed lines skipped)")
    if not df.empty:
        print("  arms=%s  origins=%s  signals=%s  families=%s  variants=%s  seeds=%s"
              % (sorted(df['arm'].astype(str).unique()),
                 sorted(df['origin'].dropna().astype(str).unique()),
                 sorted(df['signal'].dropna().astype(str).unique()),
                 sorted(df['designer_family_factor'].dropna().astype(str).unique()),
                 sorted(df['variant'].dropna().astype(str).unique()),
                 sorted(pd.to_numeric(df['seed'], errors='coerce').dropna()
                        .astype(int).unique().tolist())))

    for name, fn in FIGURES:
        if df.empty:
            print(f"SKIP {name}: results/runs.jsonl has no usable records yet")
            continue
        try:
            fn(df)
        except Skip as e:
            print(f"SKIP {name}: {e}")
        except Exception as e:      # never let one figure kill the rest
            print(f"SKIP {name}: unexpected {type(e).__name__}: {e}")

    print(f"\n{len(_written)} figure(s) written to {FIGDIR}:")
    for p in _written:
        print("  " + p)
    if not _written:
        print("  (none)")
    return _written


if __name__ == "__main__":
    main()
