"""Fig 4 — support localization (left) + causal check (right).

LEFT  : layer x projection fold-enrichment over a hypergeometric null, from
        results/v10/sup1_matrix.json. The JOINT matrix is a v10 recomputation
        from the 27 non-fused support dumps — the published artifact persists
        only 1-D marginals (sup1_matrix.recon_check.verdict). phi's 9 dumps are
        excluded: a lone fused qkv_proj cannot be enriched against itself.
RIGHT : LOPO paired deltas (C-ref minus C-ref-no_<proj>) with cell-bootstrap
        CIs, from results/v10/lopo_v9.json, v9 arm only. Sign convention is the
        artifact's own: POSITIVE = dropping that projection COSTS removal =
        load-bearing.

A figure carries DATA; prose lives in figures/captions.md. The image holds
the heatmap, the bars, their CIs and the zero line, and nothing that is a
sentence. Removed from the canvas and moved to the caption: the two n / "phi
excluded (fused qkv_proj)" blocks above the axes, the depth-binning rule, the
"density is not importance" argument, and both footnote paragraphs (the
per-target gradient reversal, and the v9-vs-as-run sign flip). `_note_a` and
`_note_b` still COMPUTE those two sentences from the artifacts and print them
to stdout, so the caption's digits are generated, never typed.

Honesty encodings that stay in the figure, because they are encodings and not
prose:
  * bar fill = CI status (solid = excludes 0, open + hatched = covers 0);
  * all 7 cells are drawn as dots, so v_proj's 4/7 unanimity and the two cells
    carrying its mean are visible beside o_proj's 7/7;
  * x tick labels carry the density multiplier (left) and the unanimity count
    (right) — the two halves of "density is not importance", as numbers;
  * the right-hand axis is share of the SEVEN-cell C-ref mean and prints it;
    the 10-cell +0.3532 goes to the CSV only, so the two cannot be confused;
  * the sign convention is written into the y axis label.

No hardcoded data values: every number is read from results/v10/*.json at run
time, or derived from those values by arithmetic in this file.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
import v10_style as S

S.apply()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

SUP_FP = "results/v10/sup1_matrix.json"
LOPO_FP = "results/v10/lopo_v9.json"

# Presentation order: the artifacts store projections alphabetically; q,k,v,o
# is the order the manuscript's 5.4 table uses and the order that lets a reader
# compare the two panels column for column.
PROJ_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj")

CSV_HEADER = ["panel", "kind", "projection", "depth_bin", "cell", "target",
              "value", "ci_lo", "ci_hi", "n", "status", "note"]


def _short(p):
    return p.split("_")[0]


def _row(panel, kind, value, projection="", depth_bin="", cell="", target="",
         ci_lo="", ci_hi="", n="", status="", note=""):
    return [panel, kind, projection, depth_bin, cell, target, value,
            ci_lo, ci_hi, n, status, note]


def _status(lo, hi):
    """Project convention: 'excludes 0' iff the interval does not straddle 0."""
    return "excludes 0" if (lo > 0 or hi < 0) else "covers 0"


def _reading(lo, hi):
    """Derived from the interval rather than inherited from the artifact's
    `reading` field, which is CI-status-only and would label a NEGATIVE
    excludes-0 delta 'load-bearing'. Not triggered by today's data."""
    if lo > 0:
        return "load-bearing"
    if hi < 0:
        return "dropping it INCREASES removal"
    return "droppable"


def _per_target_depth(sup):
    """Per-target late/early gradient and per-target projection means, from the
    artifact's own native layer x projection matrices, using its own depth rule
    ((L + 0.5)/n_layers) at K = 3. Returns {target: dict}."""
    out = {}
    for t, d in (sup.get("per_target_native") or {}).items():
        a = np.asarray(d["values"], dtype=float)
        n = int(d["n_layers"])
        b = [min(2, int(3 * ((L + 0.5) / n))) for L in range(n)]
        thirds = [float(np.mean(a[[i for i in range(n) if b[i] == j]]))
                  for j in range(3)]
        out[t] = dict(
            n_layers=n,
            n_dumps=d.get("n_dumps", ""),
            thirds=thirds,
            late_over_early=(thirds[2] / thirds[0] if thirds[0]
                             else float("nan")),
            proj_mean={p: float(np.mean(a[:, i]))
                       for i, p in enumerate(d["projections"])},
        )
    return out


def _note_a(sup):
    """Footer (a): the pooled panel averages a per-target sign reversal away."""
    if not sup:
        return None
    per_t = _per_target_depth(sup)
    if not per_t:
        return None
    rev = sorted([t for t, v in per_t.items() if v["late_over_early"] < 1.0])
    if rev:
        t0 = rev[0]
        return (f"(a) the late-layer gradient is not universal: {t0} reverses "
                f"it (late/early {per_t[t0]['late_over_early']:.2f}x, "
                f"{len(rev)} of {len(per_t)} targets).")
    lo = min(v["late_over_early"] for v in per_t.values())
    hi = max(v["late_over_early"] for v in per_t.values())
    return (f"(a) all {len(per_t)} targets are late-concentrated "
            f"(late/early {lo:.2f}x-{hi:.2f}x).")


def _note_b(lopo):
    """Footer (b): vintage. These are v9 values, and two of them changed sign
    against the digits currently printed in the draft's 5.4 table."""
    if not lopo:
        return None
    cmp_ = (lopo.get("lopo") or {}).get("comparison") or {}
    flipped = [_short(p) for p in PROJ_ORDER
               if str(cmp_.get(p, {}).get("sign_change", "")
                      ).startswith("CHANGED")]
    if flipped:
        return ("(b) v9 values; the re-score flips the sign of "
                + " and ".join(flipped) + " vs the draft's as-run digits "
                "(both cover 0).")
    return "(b) v9 values throughout."



# ------------------------------------------------------------------ LEFT ----

def _draw_heatmap(fig, sub, sup, csv, blocked):
    """8 x 4 fold-enrichment heatmap + horizontal colorbar."""
    # hspace leaves room for the two-line x tick labels and the x label; with
    # the footnote block gone the panel is taller, so the old 0.95 opened a
    # visible hole between the heatmap and its colorbar.
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=sub,
                                 height_ratios=[1.0, 0.045], hspace=0.62)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])
    tick = plt.rcParams["xtick.labelsize"]

    if not sup or "matrix" not in sup:
        blocked.append(f"{SUP_FP} missing or carries no `matrix` block — the "
                       "left panel is not drawn (no substitute source used).")
        ax.set_axis_off()
        cax.set_axis_off()
        ax.text(.5, .5, f"{SUP_FP}\nnot available —\npanel not drawn",
                ha="center", va="center", transform=ax.transAxes, color=S.INK)
        return ax

    mat = sup["matrix"]
    rows = list(mat["row_labels"])
    vals = mat["values"]
    ndumps = mat.get("n_dumps")
    col_of = {p: i for i, p in enumerate(mat["projections"])}

    missing = [p for p in PROJ_ORDER if p not in col_of]
    if missing:
        blocked.append("projections absent from sup1_matrix.matrix ("
                       + ", ".join(missing) + ") — columns omitted, never "
                       "zero-filled.")
    cols = [p for p in PROJ_ORDER if p in col_of]
    M = np.array([[vals[r][col_of[p]] for p in cols] for r in range(len(rows))],
                 dtype=float)

    # Fold enrichment is a ratio, so the scale is logarithmic and the 1x null
    # gets its own colorbar tick. cividis is safe in all three CVD types.
    norm = mcolors.LogNorm(vmin=float(np.nanmin(M)), vmax=float(np.nanmax(M)))
    cmap = plt.get_cmap("cividis")
    im = ax.imshow(M, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.grid(False)

    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            rgba = cmap(norm(M[r, c]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center",
                    fontsize=tick - 1,
                    color=("#FFFFFF" if lum < 0.55 else S.INK))
            csv.append(_row("left(a) heatmap", "fold_enrichment_cell",
                            f"{M[r, c]:.6f}", projection=cols[c],
                            depth_bin=rows[r],
                            n=(ndumps[r][col_of[cols[c]]] if ndumps else ""),
                            note="mean over included dumps of that dump's own "
                                 "(k_cell/N_cell)/density_dump; hypergeometric "
                                 "null = 1.0; no zeroed non-measurements"))

    # x labels carry the pooled projection marginal — the multiplier the
    # "density is not importance" sentence quotes. It is NOT the column mean of
    # this panel: projections hold unequal parameter counts.
    marg = (sup.get("marginals") or {}).get("projection") or {}
    xlabs = []
    for p in cols:
        m = marg.get(p, {}).get("mean")
        xlabs.append(f"{_short(p)}\n{m:.2f}x" if m is not None else _short(p))
        if m is not None:
            csv.append(_row("left(a) heatmap", "projection_density_multiplier",
                            f"{m:.6f}", projection=p,
                            n=marg.get(p, {}).get("n_dumps", ""),
                            note="pooled marginal over support dumps, drawn as "
                                 "the x tick label; not a column mean of the "
                                 "panel"))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(xlabs)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("projection  (density multiplier)")
    ax.set_ylabel("fractional depth bin")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    inc = sup.get("inclusion") or {}
    # n, target composition and phi's structural exclusion are caption text,
    # not canvas text: they are printed by build() and live in captions.md.
    ax.set_title("(a) support density", pad=6)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("fold enrichment vs null")
    ticks = [t for t in (0.1, 0.5, 1.0, 2.0) if norm.vmin <= t <= norm.vmax]
    cb.set_ticks(ticks)
    cb.ax.set_xticklabels([("1x (null)" if abs(t - 1.0) < 1e-9 else f"{t:g}x")
                           for t in ticks])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)

    # ---- numbers behind footer (a) and the caption's heterogeneity clause ---
    for t, v in _per_target_depth(sup).items():
        lbl = f"{t} ({v['n_layers']}L)"
        csv.append(_row("left(a) heatmap", "per_target_late_over_early",
                        f"{v['late_over_early']:.6f}", target=lbl,
                        n=v["n_dumps"],
                        note="fractional-depth thirds (K=3) of the per-target "
                             "native matrix, mean over layers and the 4 "
                             "projections; footer (a), not a panel cell"))
        for p in ("q_proj", "v_proj"):
            if p in v["proj_mean"]:
                csv.append(_row("left(a) heatmap",
                                "per_target_projection_mean",
                                f"{v['proj_mean'][p]:.6f}", projection=p,
                                target=lbl, n=v["n_dumps"],
                                note="mean over that target's native layers; "
                                     "cited in the caption because the v<q "
                                     "ordering reverses for one target; not "
                                     "plotted"))
    grad = (sup.get("gradient") or {}).get("fracdepth_thirds_nonfused") or {}
    for k in ("early", "middle", "late", "late_over_early"):
        if k in grad:
            csv.append(_row("left(a) heatmap", f"pooled_gradient_{k}",
                            f"{grad[k]:.6f}", n=grad.get("population_n", ""),
                            note="K=3 view of the plotted (phi-excluded) "
                                 "population; caption number, not a panel "
                                 "cell. The draft's 0.41x->1.52x is the "
                                 "phi-INCLUDED 36-dump version"))
    csv.append(_row("left(a) heatmap", "n_support_dumps_included",
                    inc.get("n_included", ""), n=inc.get("n_included", ""),
                    note="4 targets x 3 seeds x axes; 18 of the 27 are one "
                         "axis (occ_gender) and 12 one designer (qwen_3b)"))
    csv.append(_row("left(a) heatmap", "n_support_dumps_excluded_fused",
                    inc.get("n_excluded", ""), target="phi",
                    note="phi fuses q/k/v into qkv_proj and never touches "
                         "o_proj; a lone fused module has enrichment 1.0 by "
                         "construction"))
    return ax


# ----------------------------------------------------------------- RIGHT ----

def _draw_lopo(fig, sub, lopo, csv, blocked):
    ax = fig.add_subplot(sub)

    if not lopo or "v9" not in (lopo.get("lopo") or {}):
        blocked.append(f"{LOPO_FP} missing or carries no v9 arm — the right "
                       "panel is not drawn. The as-run arm in the artifact is "
                       "superseded and is NOT substituted.")
        ax.set_axis_off()
        ax.text(.5, .5, f"{LOPO_FP}\nnot available —\npanel not drawn",
                ha="center", va="center", transform=ax.transAxes, color=S.INK)
        return ax

    v9 = lopo["lopo"]["v9"]
    deltas = v9["deltas"]
    per_cell = v9["per_cell"]
    cref7 = v9["condition_means"]["C-ref"]      # SEVEN-cell C-ref, not the 10

    have = [p for p in PROJ_ORDER if p in deltas]
    absent = [p for p in PROJ_ORDER if p not in deltas]
    if absent:
        blocked.append("v9 LOPO deltas absent for " + ", ".join(absent)
                       + " — bars omitted rather than zero-filled.")
    if not have:
        ax.set_axis_off()
        return ax

    S.zeroline(ax)
    dot_lo, dot_hi = 0.0, 0.0
    for i, p in enumerate(have):
        d = deltas[p]
        lo, hi, pt, n = d["lo"], d["hi"], d["point"], d["n"]
        st, rd = _status(lo, hi), _reading(lo, hi)
        col = S.PROJ[p]
        solid = st == "excludes 0"
        ax.bar(i, pt, width=0.62, zorder=2, color=(col if solid else "none"),
               edgecolor=col, linewidth=(0 if solid else 1.1),
               hatch=(None if solid else "////"))
        ax.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], fmt="none",
                    ecolor=S.DARKGREY, elinewidth=1.0, capsize=2, zorder=4)

        cells = per_cell.get(p, {})
        keys = sorted(cells)
        offs = np.linspace(-0.24, 0.24, len(keys)) if len(keys) > 1 else [0.0]
        for k, off in zip(keys, offs):
            v = cells[k]["delta"]
            dot_lo, dot_hi = min(dot_lo, v), max(dot_hi, v)
            ax.plot(i + off, v, marker="o", ms=2.6, mfc="#FFFFFF",
                    mec=S.DARKGREY, mew=0.6, ls="none", zorder=5)
            csv.append(_row("right(b) LOPO", "lopo_per_cell_delta",
                            f"{v:.6f}", projection=p, cell=k, n=1,
                            note="C-ref - LOPO on that (target,axis) cell, "
                                 "mean over its 3 seed rows; drawn as a dot"))

        npos = sum(1 for c in cells.values() if c["delta"] > 0)
        csv.append(_row("right(b) LOPO", "lopo_delta_v9", f"{pt:.6f}",
                        projection=p, ci_lo=f"{lo:.6f}", ci_hi=f"{hi:.6f}",
                        n=n, status=st,
                        note=f"{rd}; paired 10k percentile bootstrap over {n} "
                             "cells, seed 0, v9 integer-item gate; zero "
                             "budget-fail rows in this arm"))
        csv.append(_row("right(b) LOPO", "lopo_cells_with_positive_delta",
                        npos, projection=p, n=n, status=st,
                        note="unanimity count, drawn in the x tick label"))
        csv.append(_row("right(b) LOPO", "lopo_delta_pct_of_C_ref_7cell",
                        f"{100.0 * pt / cref7:.4f}", projection=p, n=n,
                        note="right-hand axis; denominator is the 7-cell C-ref "
                             "mean, never the 10-cell +0.3532"))

    labs = []
    for p in have:
        npos = sum(1 for c in per_cell.get(p, {}).values() if c["delta"] > 0)
        labs.append(f"{_short(p)}\n{npos}/{deltas[p]['n']}")
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels(labs)
    ax.set_xlim(-0.62, len(have) - 0.38)
    ax.set_xlabel("projection dropped  (cells with $\\Delta>0$)")
    ax.set_ylabel("$\\Delta$ removal:  C-ref $-$ LOPO\n"
                  "$+$ = dropping it HURTS removal")

    ylo = min(dot_lo, min(deltas[p]["lo"] for p in have)) - 0.05
    yhi = max(dot_hi, max(deltas[p]["hi"] for p in have)) + 0.09
    ax.set_ylim(ylo, yhi)

    n_cells = deltas[have[0]]["n"]
    n_dec = v9.get("n_cells_all_dec", "")
    # "LOPO, n=7 of 10 DEC cells" and phi's exclusion are caption sentences.
    ax.set_title("(b) causal check: drop one projection", pad=6)

    # share of the SEVEN-cell C-ref mean — the reference these bars belong to.
    ax2 = ax.twinx()
    ax2.set_ylim(100.0 * ylo / cref7, 100.0 * yhi / cref7)
    ax2.set_ylabel(f"% of {n_cells}-cell C-ref (+{cref7:.3f})")
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    csv.append(_row("right(b) LOPO", "C_ref_mean_7_cell", f"{cref7:.6f}",
                    n=n_cells,
                    note="right-axis denominator; the 10-cell C-ref mean would "
                         "inflate every bar"))
    if "c_ref_mean_all_dec_cells" in v9:
        csv.append(_row("right(b) LOPO", "C_ref_mean_all_dec_cells",
                        f"{v9['c_ref_mean_all_dec_cells']:.6f}", n=n_dec,
                        note="NOT the denominator used here; listed so the two "
                             "cannot be confused"))

    ax.legend(handles=[
        Patch(facecolor=S.DARKGREY, edgecolor=S.DARKGREY, label="excludes 0"),
        Patch(facecolor="none", edgecolor=S.DARKGREY, hatch="////",
              label="covers 0"),
        Line2D([], [], marker="o", ms=2.6, mfc="#FFFFFF", mec=S.DARKGREY,
               mew=0.6, ls="none", label="one cell"),
    ], loc="upper left", handlelength=1.0, borderpad=0.0, labelspacing=0.22,
        handletextpad=0.4, borderaxespad=0.25)
    return ax


# ------------------------------------------------------------------ main ----

def build():
    C.ensure_dirs()
    csv, blocked = [], []

    sup = C.jload(SUP_FP)
    lopo = C.jload(LOPO_FP)
    if sup is None:
        blocked.append(f"{SUP_FP} not found — left panel degraded.")
    if lopo is None:
        blocked.append(f"{LOPO_FP} not found — right panel degraded.")

    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.46,
                          left=0.085, right=0.905, top=0.905, bottom=0.155)
    _draw_heatmap(fig, gs[0, 0], sup, csv, blocked)
    _draw_lopo(fig, gs[0, 1], lopo, csv, blocked)

    # Cross-panel: the same density multipliers on the 21 dumps that actually
    # match the 7 LOPO cells (the bars' own denominator) beside the 27 the
    # heatmap draws and the manuscript quotes.
    if lopo:
        sd = lopo.get("support_density") or {}
        for key in ("four_projection", "lopo_matched"):
            blk = sd.get(key) or {}
            for p, v in (blk.get("enrichment") or {}).items():
                if p in PROJ_ORDER:
                    csv.append(_row("cross-panel",
                                    f"projection_density_multiplier_{key}",
                                    f"{v:.6f}", projection=p,
                                    n=blk.get("n_dumps", ""),
                                    note="four_projection = the 27 dumps the "
                                         "heatmap draws; lopo_matched = the 21 "
                                         "dumps behind the 7 LOPO cells"))
        for k, v in (lopo.get("phi_exclusion") or {}).items():
            if isinstance(v, int):
                csv.append(_row("cross-panel", f"count_{k}", v,
                                note="phi_exclusion block of lopo_v9.json"))

    pdf, png = S.save(fig, "fig4")
    csv_fp = C.write_csv(os.path.join(C.FIGDIR, "fig4_data.csv"),
                         CSV_HEADER, csv)
    print(f"wrote {pdf}\nwrote {png}\nwrote {csv_fp}  ({len(csv)} rows)")
    # The two sentences that used to be drawn under the axes. They are still
    # computed from the artifacts — they now belong to figures/captions.md.
    for n in (_note_a(sup), _note_b(lopo)):
        if n:
            print("CAPTION:", n)
    for b in blocked:
        print("BLOCKED:", b)
    return dict(pdf=pdf, png=png, csv=csv_fp, blocked=blocked, n_rows=len(csv))


if __name__ == "__main__":
    build()
