"""Fig 2 (headline) — the seven-condition DEC decomposition.

Horizontal dot-and-CI plot, conditions ordered by v9 mean descending
(C-ref at the top ... C-bottom at the bottom).

SOURCE: results/v10/dec_analysis_v9.json ONLY — the v9 re-scored panel
(results_v9/v8dec/removal.jsonl, integer-item gate). The as-run
results/v8/dec_analysis.json is superseded and is never read here.

Nothing in this module is a hand-typed result. Every plotted value and the one
direct label are read out of the artifact at build time; only axis limits,
marker glyphs and static label wording are literals.

THE IMAGE CARRIES DATA; THE PROSE LIVES IN figures/captions.md. The numeric
mean/CI column, the footnote block (v10-vs-v9 CI provenance, the cell-matched
C-ref reference for the 7-cell condition, the grid-conditional alpha counts,
the budget-fail count), the registered qualifiers and the C-tensorshuf
single-cell explanation are all in the caption. figures/fig2_data.csv is
unchanged and remains the machine-readable record of every one of them.

Honesty encodings that stay IN the figure, because they are encodings and not
sentences:

  * per-cell dots under every condition mean, so a mean carried by one cell
    cannot hide behind its bar (C-tensorshuf is exactly that case);
  * a dagger on the C-tensorshuf tick label and in the legend key: n = 7, the
    three phi cells carry no C-tensorshuf row at all (fused qkv_proj), they are
    ABSENT, never zero-filled;
  * dashed CI bars for intervals that COVER 0 (C-tensorshuf is the only one);
  * a zoom panel for the four collapsed conditions, because a linear axis that
    shows C-ref also hides whether "collapses to ~0" is a measurement or a
    rounding artifact;
  * carets marking values that run off the zoom panel;
  * one direct label, on the headline estimand Delta_selection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
import v10_style as S
S.apply()

import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ART = "results/v10/dec_analysis_v9.json"

# Marker glyph per perturbation class. The CLASSIFICATION comes from the
# artifact (perturb_classification, itself v10_common.DEC_PERTURBS); only the
# glyph/fill convention is chosen here.
MARKER = {"reference": "D", "support": "o", "sign": "s",
          "location": "^", "both": "X"}
# Filled  = the perturbation moves WHICH coordinates are edited, true signs kept
#           at those coordinates  (support-type).
# Open    = the perturbation attacks the sign field itself or where it sits
#           (sign-type / location-type / both).  Spelled out in the caption.
FILLED_CLASSES = ("reference", "support")
CLASS_LABEL = {
    "reference": "reference",
    "support":   "support moved",
    "sign":      "signs permuted",
    "location":  "pattern relocated",
    "both":      "support + signs random",
}


def _f(x, p=4):
    return "n/a" if x is None else f"{x:+.{p}f}"


def build():
    C.ensure_dirs()
    D = C.jload(ART)
    if D is None:
        raise SystemExit(f"missing artifact: {ART} (run src/v10_regen_dec.py)")

    order = list(D["conditions_ordered_by_v9_mean_desc"])
    per = D["per_condition"]
    klass = D["perturb_classification"]
    est = D["estimands"]
    hdr = D["per_cell_table"]["header"]
    ptrows = D["per_cell_table"]["rows"]
    ci_of = {c: per[c]["ci"] for c in order}

    # sanity: the artifact's stated order really is descending by mean
    means = [per[c]["mean"] for c in order]
    assert means == sorted(means, reverse=True), "artifact order is not mean-desc"

    ix = {name: i for i, name in enumerate(hdr)}
    cells = [
        dict(cell=r[ix["cell"]], target=r[ix["target"]], axis=r[ix["axis"]],
             family=r[ix["axis_family"]],
             vals={c: r[ix[c]] for c in order})
        for r in ptrows
    ]

    # --- cell-matched C-ref reference for the 7-cell condition -------------
    # (the artifact's share_of_C_ref for C-tensorshuf divides a 7-cell mean by
    #  the 10-cell C-ref mean; the honest denominator is C-ref on those 7 cells)
    # Not drawn any more — it is a sentence, so it lives in the caption and in
    # the CSV, both of which this block still feeds.
    short = [c for c in order if per[c]["n_cells"] != D["n_cells"]]
    matched = {}
    for c in short:
        sub = [d["vals"]["C-ref"] for d in cells if d["vals"][c] is not None]
        matched[c] = dict(cref_mean=sum(sub) / len(sub), n=len(sub))

    y_of = {c: len(order) - 1 - i for i, c in enumerate(order)}

    # ---------------------------------------------------------------- axes --
    fig = plt.figure(figsize=S.FULL)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.70, 1.00], wspace=0.06)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    fig.subplots_adjust(left=0.150, right=0.988, top=0.965, bottom=0.310)

    # zoom window: the four lowest-ranked conditions, derived (not typed)
    low = order[3:]
    zhi_src = [ci_of[c]["hi"] for c in low] + \
              [d["vals"][c] for d in cells for c in low if d["vals"][c] is not None]
    zlo_src = [ci_of[c]["lo"] for c in low] + \
              [d["vals"][c] for d in cells for c in low if d["vals"][c] is not None]
    zhi = max(zhi_src) * 1.22
    zlo = min(min(zlo_src), 0.0) - 0.10 * zhi

    allv = [d["vals"][c] for d in cells for c in order if d["vals"][c] is not None]
    xhi = max(max(allv), max(ci_of[c]["hi"] for c in order)) * 1.10
    xlo = min(min(allv), min(ci_of[c]["lo"] for c in order)) - 0.035 * xhi

    ax0.set_xlim(xlo, xhi)
    ax1.set_xlim(zlo, zhi)
    ax0.set_ylim(-0.58, len(order) - 0.52)
    for ax in (ax0, ax1):
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        S.zeroline(ax, x=True)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(3))

    # the zoom window, shown on the full-range panel
    ax0.axvspan(zlo, zhi, color=S.SKY, alpha=0.13, lw=0, zorder=0)
    for sp in ("bottom", "left"):
        ax1.spines[sp].set_color(S.SKY)

    rng = random.Random(0)
    csv_rows = []

    for cond in order:
        y = y_of[cond]
        col = S.COND[cond]
        kl = klass[cond]
        mk = MARKER[kl]
        filled = kl in FILLED_CLASSES
        ci = ci_of[cond]
        covers0 = ci["status"] != "excludes 0"
        ls = (0, (2.2, 1.3)) if covers0 else "-"

        for ax, lo_x, hi_x in ((ax0, xlo, xhi), (ax1, zlo, zhi)):
            # per-cell dots, on a strip below the mean marker
            n_off = n_lo = 0
            for d in cells:
                v = d["vals"][cond]
                if v is None:
                    continue
                if v > hi_x:
                    n_off += 1
                    continue
                if v < lo_x:
                    n_lo += 1
                    continue
                ax.plot(v, y - 0.22 + rng.uniform(-0.055, 0.055), "o",
                        ms=2.5, mfc=col, mec="white", mew=0.35, alpha=0.60,
                        zorder=2, clip_on=True)
            pad = 0.018 * (hi_x - lo_x)
            if n_off:
                ax.plot(hi_x - pad, y - 0.22, ">", ms=3.0,
                        color=col, alpha=0.75, zorder=3, clip_on=False)
            if n_lo:
                ax.plot(lo_x + pad, y - 0.22, "<", ms=3.0,
                        color=col, alpha=0.75, zorder=3, clip_on=False)
            # CI bar, drawn only where the interval actually meets this window
            if ci["lo"] < hi_x and ci["hi"] > lo_x:
                ax.plot([max(ci["lo"], lo_x), min(ci["hi"], hi_x)], [y, y],
                        ls=ls, color=col, lw=1.5, solid_capstyle="butt",
                        zorder=3)
                if ci["hi"] > hi_x:
                    ax.plot(hi_x - pad, y, ">", ms=3.4, color=col, zorder=4,
                            clip_on=False)
                if ci["lo"] < lo_x:
                    ax.plot(lo_x + pad, y, "<", ms=3.4, color=col, zorder=4,
                            clip_on=False)
            elif ci["lo"] >= hi_x:
                ax.plot(hi_x - pad, y, ">", ms=3.4, color=col, zorder=4,
                        clip_on=False)
            if lo_x <= per[cond]["mean"] <= hi_x:
                ax.plot(per[cond]["mean"], y, mk, ms=6.0, color=col,
                        mfc=col if filled else "white", mec=col, mew=1.15,
                        zorder=5)

        n = per[cond]["n_cells"]
        csv_rows.append([
            "condition_mean", cond, kl, mk, "filled" if filled else "open",
            "", "", "", "", f"{per[cond]['mean']:.6f}",
            f"{ci['lo']:.6f}", f"{ci['hi']:.6f}", n, ci["status"],
            ci.get("registered", False),
            ("n=7 of 10 cells: phi|occ_gender, phi|bbq_Age, phi|ss_intra carry no "
             "C-tensorshuf row (fused qkv_proj) - ABSENT, not zero-filled"
             if n != D["n_cells"] else ""),
        ])
        for d in cells:
            v = d["vals"][cond]
            csv_rows.append([
                "cell_value", cond, kl, mk, "filled" if filled else "open",
                d["cell"], d["target"], d["axis"], d["family"],
                "" if v is None else f"{v:.6f}", "", "", "", "", "",
                "structurally absent (fused qkv_proj)" if v is None else "",
            ])

    # the dagger is the whole n=7 caveat in-plot; the caption names the 3 cells
    ax0.set_yticks([y_of[c] for c in order])
    ax0.set_yticklabels([c + (" †" if per[c]["n_cells"] != D["n_cells"] else "")
                         for c in order])
    for lab in ax0.get_yticklabels():
        lab.set_fontweight("bold" if lab.get_text() == order[0] else "normal")
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.tick_params(left=False)

    ax0.set_xlabel("bias removal (v9-scored)")
    ax1.set_xlabel("zoom")

    # --------------------------------------------- the ONE direct label ------
    # Delta_selection: the headline estimand, spanned by an arrow between the
    # two means it is the difference of. Its CI, n and unanimity are caption.
    ds = est["Delta_selection"]
    un = D["unanimity"]
    ya = (y_of["C-ref"] + y_of["C-a"]) / 2.0
    ax0.annotate("", xy=(per["C-ref"]["mean"], ya), xytext=(per["C-a"]["mean"], ya),
                 arrowprops=dict(arrowstyle="<->", color=S.INK, lw=0.85,
                                 shrinkA=0, shrinkB=0))
    ax0.text((per["C-ref"]["mean"] + per["C-a"]["mean"]) / 2.0, ya,
             f"$\\Delta_{{selection}}$ = {_f(ds['point'], 3)}",
             ha="center", va="center", fontsize=7, color=S.INK,
             bbox=dict(fc="white", ec="none", pad=0.8))

    # ------------------------------------------------------------- legend ----
    handles = [
        Line2D([], [], ls="none", marker=MARKER[k], ms=5.5, color=S.DARKGREY,
               mfc=S.DARKGREY if k in FILLED_CLASSES else "white",
               mec=S.DARKGREY, mew=1.1, label=CLASS_LABEL[k])
        for k in ("reference", "support", "sign", "location", "both")
    ]
    handles += [
        Line2D([], [], ls="none", marker="o", ms=2.6, color=S.DARKGREY,
               alpha=0.6, label="per-cell value"),
        Line2D([], [], ls=(0, (2.2, 1.3)), color=S.DARKGREY, lw=1.5,
               label="CI covers 0"),
        Line2D([], [], ls="none", marker=">", ms=3.4, color=S.DARKGREY,
               label="off panel scale"),
        Line2D([], [], ls="none", marker="$\\dagger$", ms=5.0, color=S.INK,
               label=f"n = {per['C-tensorshuf']['n_cells']} cells"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=3,
               bbox_to_anchor=(0.022, 0.008), columnspacing=1.2,
               handletextpad=0.45, labelspacing=0.30)

    # ------------------------------------------------------------ CSV only ---
    # Everything below is written to fig2_data.csv (audit contract: rows and
    # columns are frozen) and is stated in the caption. None of it is drawn.
    ca = D["alpha_grid"]["by_condition"]["C-a"]
    cxr = est["C-a minus C-rand"]
    dt = est.get("Delta_tensor")
    ts = "C-tensorshuf"
    m7 = matched.get(ts)

    for name, d in (("Delta_selection", ds), ("C-a minus C-rand", cxr),
                    ("Delta_tensor", dt)):
        if d is None:
            continue
        csv_rows.append(["annotation", name, "", "", "", "", "", "", "",
                         f"{d['point']:.6f}", f"{d['lo']:.6f}", f"{d['hi']:.6f}",
                         d["n"], d["status"], d.get("registered", ""),
                         "paired cell bootstrap; registered v9 estimand"])
    csv_rows.append(["annotation", "unanimity", "", "", "", "", "", "", "",
                     un["count"], "", "", un["denominator"], "", True,
                     f"cells with {un['comparison']}"])
    csv_rows.append(["annotation", "C-a argmax at grid max", "", "", "", "", "",
                     "", "", ca["at_grid_max"], "", "",
                     ca["n_runs"], "", True,
                     f"argmax counts per run at alpha={D['alpha_grid']['grid_max']:g};"
                     " NOT budget-availability counts"])
    for k, lab in (("n_targets", "targets"), ("n_axes", "axes"),
                   ("n_benchmark_families", "benchmark families")):
        csv_rows.append(["annotation", f"coverage: {lab}", "", "", "", "", "", "",
                         "", D["coverage"][k], "", "", D["n_cells"], "", True,
                         "panel composition stated in the figure footnote; the "
                         "denominator column is the cell count"])
    csv_rows.append(["annotation", "alpha grid maximum", "", "", "", "", "", "",
                     "", D["alpha_grid"]["grid_max"], "", "", "", "", True,
                     "largest alpha in the frozen deployment grid"])
    csv_rows.append(["annotation", "budget_fail_rows_zeroed", "", "", "", "", "",
                     "", "", D["source"]["budget_fail_rows_zeroed"], "", "",
                     D["source"]["n_rows_after_dedupe"], "", True,
                     "nan/None removal -> 0.0 convention; zero such rows here"])
    if m7:
        csv_rows.append(["annotation", "C-ref mean on the C-tensorshuf cells",
                         "reference", "", "", "", "", "", "",
                         f"{m7['cref_mean']:.6f}", "", "", m7["n"], "", False,
                         "cell-matched denominator for the n=7 condition; the "
                         "artifact's share_of_C_ref for C-tensorshuf uses the "
                         "10-cell C-ref mean instead"])
        csv_rows.append(["annotation", "C-tensorshuf mean excluding max cell",
                         klass[ts], "", "", per[ts]["max_cell"], "", "", "",
                         f"{per[ts]['mean_excluding_max_cell']:.6f}", "", "",
                         per[ts]["n_cells"] - 1, "", False,
                         "mean over the cells other than the max cell; the cell column "
                         "names the EXCLUDED max cell"])

    pdf, png = S.save(fig, "fig2")
    csv = C.write_csv(
        os.path.join(C.FIGDIR, "fig2_data.csv"),
        ["kind", "condition", "perturb_class", "marker", "marker_fill", "cell",
         "target", "axis", "axis_family", "value", "ci_lo", "ci_hi", "n",
         "ci_status", "ci_registered_in_v9", "note"],
        csv_rows)
    print("wrote", pdf, png, csv)
    return pdf, png, csv


if __name__ == "__main__":
    build()
