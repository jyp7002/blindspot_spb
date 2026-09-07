"""v10 Fig 5 — the frontier (per-cell dots + pooled paired deltas).

Source of every plotted number: results/v10/frontier_v9.json (the verified v9
regen artifact), plus results/v10/patch_sizes.json for the single numeric cell
of Table B. Nothing here is typed by hand: axis limits, marker glyphs and label
words are chosen, values are read at run time.

WHAT THE FIGURE HAS TO CARRY (the honesty requirements are the point, not
decoration):

  * MIXED GATE, THREE PROVENANCES, NOT TWO. `true_v9` is a per-(cell, method)
    flag in the artifact and it is False for twelve of the thirty-two dots, for
    two different reasons:
      - the 4 llama/qwen EDIT cells come from results/x, which has no
        alpha_trace and no re-scorable twin -> as-run, never re-scorable;
      - ALL 8 DPO dots carry gate_mode='UNGATED': that panel could not be
        re-gated from removal.jsonl, so its collateral_ok is still the buggy
        as-run float gate.
    The manuscript's gate-mixing disclosure currently names only the first
    four. Encoding: filled = true-v9; open with a solid edge = as-run, never
    re-scorable; open with a dotted edge = UNGATED.

  * FOUR DOTS ARE NOT MEASUREMENTS. They are 0.0 manufactured by the project's
    nan->0 rule, and the two reasons differ, so they do not share a marker:
      - 'x'  SentenceDebias phi|crows: 12 configurations were measured and
             every one failed the collateral budget ("could not be done within
             budget", not "had no effect");
      - the empty-set glyph: DPO gemma|occ, gemma|crows, llama|crows were never
             run at all (<20 informative preference pairs).

  * TWO DOTS MOVED UNDER THE V9 RE-SCORE (steering at llama|occ and qwen|occ).
    The draft's Table A still prints the as-run digit at llama|occ, so the
    as-run position is drawn as a grey tick joined to the v9 dot.

  * EVERY POOLED ESTIMAND IS MIXED-GATE: all of them contain the edit arm's 4
    as-run cells, and edit - DPO also contains the wholly ungated DPO arm. So
    every pooled diamond carries the same dotted "not fully v9" edge.

  * THE RESTRICTED ESTIMATE IS PLOTTED FOR STEERING ONLY.
    restricted_estimate_availability.published marks only
    `edit - steering [re-scorable cells only]` as having a published v9
    counterpart. The SentenceDebias (n=3) and DPO (n=2) restricted values are
    v10-only, and an n=2 percentile bootstrap is not an inferential statement,
    so neither is drawn as an interval anywhere in this figure.

  * phi-3.5 fuses q/k/v and its edit never touches o_proj, so it is not
    parameterization-matched to gemma/llama/qwen -- and it supplies 2 of the 4
    cells behind the restricted estimate. Marked on the tick labels.

WHAT IS DRAWN AND WHAT IS WRITTEN. The image carries data and encodings only:
the dots, the pooled diamonds with their intervals, the marker states above, a
short title per panel, axis labels, and two compact legends (method colour;
marker encoding). Every sentence that explains methodology, gate provenance or
a caveat -- the true-v9 census, the argmax-selection wording, the
restricted-estimate discussion, the roster of as-run and ungated cells, the
meaning of the dagger -- lives in the caption that build() returns and that
figures/captions.md carries, which is what a reader actually has beside the
figure. Exactly one number is printed next to a point (the single pooled
interval that excludes 0): direct labels work because they are sparing.

Table B (deployment properties) stays a LaTeX table beside the figure and is
never drawn; its one numeric cell is read from patch_sizes.json in that
artifact's own unit (MiB = bytes / 2**20, which the manuscript prints as "MB").

Outputs: figures/fig5.pdf, figures/fig5.png, figures/fig5_data.csv,
figures/table_b.tex.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
import v10_style as S
S.apply()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FRONTIER = "results/v10/frontier_v9.json"
SIZES = "results/v10/patch_sizes.json"

# marker glyphs / line styles: presentation choices, not data
M_DOT = "o"
M_BUDGET_FAIL = "x"
M_NEVER_RUN = "$\\varnothing$"
DOTTED = (0, (1, 1.1))
S_FILLED, S_OPEN, S_GLYPH, S_DIAMOND = 24, 28, 46, 40
MINUS = "\u2212"            # the draft's minus sign
DAGGER = "\u2020"


# ------------------------------------------------------------ classifiers ---
def provenance(md):
    """Three-way gate provenance, derived from the artifact's own fields."""
    if md.get("true_v9"):
        return "true-v9"
    if str(md.get("gate_mode", "")).upper().startswith("UNGATED"):
        return "ungated"
    return "as-run"


def zero_kind(md):
    """Which of the two nan->0 reasons produced this 0.0 (artifact string)."""
    if not md.get("zeroed"):
        return None
    r = str(md.get("zeroed_reason", "")).lower()
    if "too_few_pairs" in r or "never run" in r:
        return "never_run"
    if "budget" in r or "in-budget" in r:
        return "budget_fail"
    return "unknown"


def fmt_ci(d, p=3):
    return (f"{d['point']:+.{p}f} [{d['lo']:+.{p}f}, {d['hi']:+.{p}f}]"
            .replace("-", MINUS))


def comparator(key):
    return key.split("[")[0].strip().split(" - ")[-1].strip()


# ------------------------------------------------------------------ TeX -----
def _md_cell_to_tex(cell):
    s, out, i = cell.strip(), [], 0
    while i < len(s):
        if s.startswith("**", i):
            j = s.find("**", i + 2)
            if j < 0:
                out.append(s[i:])
                break
            out.append("\\textbf{" + s[i + 2:j] + "}")
            i = j + 2
        else:
            out.append(s[i])
            i += 1
    s = "".join(out)
    s = s.replace("\u2212", "$-$").replace("\u2013", "--").replace("\u2014", "---")
    s = s.replace("\u00a7", "\\S").replace("&", "\\&").replace("%", "\\%")
    return s.replace("`", "")


def table_b_tex(PS, blocked):
    """Table B stays a LaTeX table, never a plot. The qualitative deployment
    properties are lifted verbatim from the manuscript's own Table B; the ONE
    numeric cell (artifact size) is replaced by the value read from
    patch_sizes.json, in the unit that artifact actually uses."""
    draft_fp = os.path.join(C.HERE, "binary_debiaser_draft_v3.md")
    if not os.path.exists(draft_fp):
        blocked.append("figures/table_b.tex not written: binary_debiaser_draft_v3.md "
                       "is absent, so the qualitative deployment properties have no "
                       "source and were not invented.")
        return None
    lines = open(draft_fp).read().splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().startswith("**Table B")), None)
    if start is None:
        blocked.append("figures/table_b.tex not written: no '**Table B' heading in "
                       "binary_debiaser_draft_v3.md.")
        return None
    tbl = []
    for ln in lines[start + 1:]:
        if ln.strip().startswith("|"):
            tbl.append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif tbl:
            break
    if len(tbl) < 3:
        blocked.append("figures/table_b.tex not written: the draft's Table B did not "
                       "parse into a header plus rows.")
        return None
    head, body = tbl[0], [r for r in tbl[2:] if r and r[0]]

    pr = (PS or {}).get("patch_range_1pct") or {}
    payload = pr.get("payload_only") or {}
    idx = pr.get("with_entropy_bound_index") or {}
    lo, hi = payload.get("min_mib"), payload.get("max_mib")
    ilo, ihi = idx.get("min_mib"), idx.get("max_mib")
    size_txt = foot = None
    if lo is not None and hi is not None:
        size_txt = "\\textbf{%.2f--%.2f\\,MiB}" % (lo, hi)
        n_t = pr.get("n_targets", len((PS or {}).get("per_target", {})))
        foot = ("Artifact size is the $s=0.99$ merged patch over %d targets. "
                "MiB $=$ bytes$/2^{20}$; the manuscript prints these as MB. "
                "Payload only: bytes $=$ retained$/8$ $+$ 2 bytes per tensor, so "
                "nothing is charged for recording \\emph{which} coordinates are "
                "retained" % n_t)
        if ilo is not None and ihi is not None:
            foot += ("; with an entropy-bound support index the same patches are "
                     "%.2f--%.2f\\,MiB" % (ilo, ihi))
        foot += (". phi-3.8B's patch covers only the fused \\texttt{qkv\\_proj} and is "
                 "not commensurable with the q/k/v/o patches. "
                 "Source: \\texttt{results/v10/patch\\_sizes.json}.")
    else:
        blocked.append("Table B's artifact-size cell kept the draft's own text: "
                       "patch_sizes.json carries no patch_range_1pct.payload_only.")

    rows = []
    for r in body:
        r = list(r) + [""] * (len(head) - len(r))
        if size_txt is not None and "artifact size" in r[0].lower():
            r[1] = size_txt
        rows.append(" & ".join(_md_cell_to_tex(c) for c in r[:len(head)]) + r" \\")

    tex = ["% figures/table_b.tex - generated by src/v10_fig5.py. Do not edit by hand.",
           "% Qualitative properties: binary_debiaser_draft_v3.md, Table B (verbatim).",
           "% Numeric cell (artifact size): results/v10/patch_sizes.json.",
           "% Requires \\usepackage{booktabs}.",
           r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Deployment properties. Fig.~5 shows a behavioral tie; this is "
           r"what decides it.}",
           r"\label{tab:deployment}",
           r"\begin{tabular}{@{}l" + "l" * (len(head) - 1) + r"@{}}", r"\toprule",
           " & ".join(_md_cell_to_tex(h) for h in head) + r" \\", r"\midrule"] + rows + \
          [r"\bottomrule", r"\end{tabular}"]
    if foot:
        tex += [r"\vspace{2pt}", r"{\footnotesize " + foot + "}"]
    tex += [r"\end{table}", ""]

    os.makedirs(C.FIGDIR, exist_ok=True)
    out = os.path.join(C.FIGDIR, "table_b.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(tex))
    return out, (lo, hi), (ilo, ihi)


# ------------------------------------------------------------------ build ---
def build():
    C.ensure_dirs()
    blocked = []
    F = C.jload(FRONTIER)
    if F is None:
        raise SystemExit("MISSING %s — Fig 5 has no source and must not be faked."
                         % FRONTIER)
    PS = C.jload(SIZES)
    if PS is None:
        blocked.append("results/v10/patch_sizes.json absent: Table B's artifact-size "
                       "cell has no source, so figures/table_b.tex was not written.")

    methods = list(F["methods"])                      # edit, steering, SD, DPO
    cells = list(F["cells"])
    gs = F.get("gate_summary", {})
    pooled = F["pooled_v9"]
    avail = (F.get("restricted_estimate_availability") or {}).get("published", {})

    # ---- ordering: axis groups in artifact order, edit value descending -----
    axis_order = []
    for c in cells:
        if c["axis"] not in axis_order:
            axis_order.append(c["axis"])
    ordered = []
    for a in axis_order:
        grp = [c for c in cells if c["axis"] == a]
        grp.sort(key=lambda c: -c["methods"][methods[0]]["v9"])
        ordered.extend(grp)
    grp_bounds, prev = [], None
    for i, c in enumerate(ordered):
        if prev is not None and c["axis"] != prev:
            grp_bounds.append(i - 0.5)
        prev = c["axis"]

    # ---- counts, all derived --------------------------------------------------
    n_dots = F.get("n_dots", len(cells) * len(methods))
    n_true = gs.get("n_true_v9_dots_total",
                    sum(1 for c in cells for m in methods
                        if c["methods"][m].get("true_v9")))
    n_edit_asrun = gs.get("n_edit_cells_as_run",
                          sum(1 for c in cells
                              if provenance(c["methods"][methods[0]]) == "as-run"))
    ungated = {}
    for m in methods:
        k = sum(1 for c in cells if provenance(c["methods"][m]) == "ungated")
        if k:
            ungated[m] = k
    n_zeroed = F.get("n_zeroed", sum(1 for c in cells for m in methods
                                     if c["methods"][m].get("zeroed")))
    changed = F.get("cells_changed_by_v9", [])
    changed_key = {(d["cell"], d["method"]): d for d in changed}
    seed_txt = "/".join(str(s) for s in sorted(
        {c["methods"][m]["n_seeds"] for c in cells for m in methods
         if c["methods"][m].get("is_measurement")}))
    asrun_targets = sorted({k.split("|")[0] for k in gs.get("edit_cells_as_run", [])})
    restricted_cells = gs.get("edit_cells_true_v9", [])
    restricted_targets = sorted({k.split("|")[0] for k in restricted_cells})
    fused_targets = sorted({c["target"] for c in cells if c["target"].startswith("phi")})
    other_targets = sorted({c["target"] for c in cells} - set(fused_targets))

    # ============================================================== figure ====
    fig = plt.figure(figsize=S.FULL_TALL)
    # the right panel is only slightly narrower than the left: its rows carry
    # estimand names, and a name that overruns its own panel is unreadable
    gsp = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.14,
                           left=0.150, right=0.985, top=0.900, bottom=0.235)
    axL = fig.add_subplot(gsp[0, 0])
    axR = fig.add_subplot(gsp[0, 1])

    step = 0.205
    offs = {m: (i - (len(methods) - 1) / 2.0) * step for i, m in enumerate(methods)}

    csv_rows, xs_all = [], []

    for yi, c in enumerate(ordered):
        for m in methods:
            d = c["methods"][m]
            col = S.METHOD.get(m, S.GREY)
            y = yi + offs[m]
            v = float(d["v9"])
            xs_all.append(v)
            prov, zk = provenance(d), zero_kind(d)

            if zk is not None:
                axL.scatter([v], [y],
                            marker=(M_NEVER_RUN if zk == "never_run" else M_BUDGET_FAIL),
                            s=S_GLYPH, color=col, linewidths=1.1, zorder=5)
                fill = "special glyph (not a measurement)"
            elif prov == "true-v9":
                axL.scatter([v], [y], marker=M_DOT, s=S_FILLED, color=col,
                            edgecolors=col, linewidths=0.8, zorder=4)
                fill = "filled"
            elif prov == "as-run":
                axL.scatter([v], [y], marker=M_DOT, s=S_OPEN, facecolors="none",
                            edgecolors=col, linewidths=1.15, zorder=4)
                fill = "open, solid edge"
            else:
                axL.scatter([v], [y], marker=M_DOT, s=S_OPEN, facecolors="none",
                            edgecolors=col, linewidths=1.15, linestyle=DOTTED, zorder=4)
                fill = "open, dotted edge"

            ch = changed_key.get((c["cell"], m))
            if ch is not None:
                a = float(ch["as_run"])
                xs_all.append(a)
                axL.plot([a, v], [y, y], color=S.GREY, lw=0.7, zorder=2)
                axL.scatter([a], [y], marker="|", s=24, color=S.GREY,
                            linewidths=0.9, zorder=3)

            csv_rows.append([
                "cell_dot", "left: per-cell removal", c["cell"], c["target"], c["axis"],
                m, "%.3f" % y, "%.6f" % v, "%.6f" % float(d["as_run"]),
                "%.6f" % float(d.get("delta_v9_minus_asrun", 0.0)),
                "", "", "", "", d.get("n_seeds", ""),
                prov, d.get("gate_mode", ""), d.get("true_v9", ""),
                d.get("is_measurement", ""), (d.get("zeroed_reason") or ""),
                (M_NEVER_RUN if zk == "never_run"
                 else M_BUDGET_FAIL if zk == "budget_fail" else M_DOT),
                fill, d.get("source", ""),
                ("v9 re-score moved this dot off its as-run value; the draft's Table A "
                 "still prints the as-run digit" if ch is not None else ""),
            ])

    for b in grp_bounds:
        axL.axhline(b, color=S.DARKGREY, lw=0.6, alpha=0.45, zorder=1)
    S.zeroline(axL, x=True)

    axL.set_yticks(range(len(ordered)))
    axL.set_yticklabels([(c.get("short") or c["cell"]) +
                         (" " + DAGGER if c["target"] in fused_targets else "")
                         for c in ordered])
    axL.set_ylim(len(ordered) - 0.5, -0.5)
    xlo, xhi = min(xs_all), max(xs_all)
    pad = 0.07 * (xhi - xlo)
    axL.set_xlim(xlo - pad, xhi + pad)
    axL.set_xlabel("removal under the v9 collateral budget")
    axL.set_title("Per-cell removal (%d cells $\\times$ %d methods)"
                  % (len(cells), len(methods)))
    axL.grid(axis="y", visible=False)

    # ------------------------------------------------ right: pooled deltas ----
    pooled_keys = sorted([k for k in pooled if "[" not in k],
                         key=lambda k: -pooled[k]["point"])
    restricted_keys = [k for k in pooled
                       if "re-scorable cells only" in k and avail.get(k) is True]
    plot_keys = pooled_keys + restricted_keys

    # ONE direct label: the single pooled interval that excludes 0. Direct
    # labels work because they are sparing, so the other point estimates are
    # left to the caption, which prints all of them with their intervals.
    ns = {pooled[k]["n"] for k in pooled_keys}
    common_n = ns.pop() if len(ns) == 1 else None

    labelled = [j for j, k in enumerate(plot_keys)
                if pooled[k]["status"] == "excludes 0"][:3]

    for j, k in enumerate(plot_keys):
        d = pooled[k]
        col = S.METHOD.get(comparator(k), S.GREY)
        restricted = "[" in k
        axR.plot([d["lo"], d["hi"]], [j, j], color=col, lw=1.6,
                 solid_capstyle="butt", zorder=3)
        axR.scatter([d["point"]], [j], marker="D", s=S_DIAMOND,
                    facecolors=("none" if restricted else col),
                    edgecolors=(col if restricted else S.DARKGREY),
                    linewidths=1.2, linestyle=DOTTED, zorder=4)
        lab = k.split("[")[0].strip().replace(" - ", " %s " % MINUS)
        if restricted:
            lab += "  (restricted, n = %d)" % d["n"]
        elif common_n is None:
            lab += "  (n = %d)" % d["n"]
        axR.text(0.015, j - 0.15, lab, transform=axR.get_yaxis_transform(),
                 ha="left", va="bottom", color=S.INK,
                 bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
        if j in labelled:
            axR.annotate(("%+.3f" % d["point"]).replace("-", MINUS), (d["hi"], j),
                         textcoords="offset points", xytext=(5, 0), ha="left",
                         va="center", color=S.INK, fontweight="bold")
        csv_rows.append([
            "pooled_delta", "right: pooled paired delta", k, "", "", comparator(k),
            "%.3f" % j, "%.6f" % d["point"], "", "",
            "%.6f" % d["lo"], "%.6f" % d["hi"], d["status"], d["n"], "",
            "mixed-gate (every estimand contains non-v9 cells)", "", "False", "True", "",
            "D", ("open diamond, dotted edge" if restricted
                  else "filled diamond, dotted edge"),
            FRONTIER,
            ("published v9 restricted estimate (steering is the only arm with one)"
             if restricted else "registered v9 pooled estimand"),
        ])

    axR.set_yticks(range(len(plot_keys)))
    axR.set_yticklabels([""] * len(plot_keys))
    axR.tick_params(axis="y", length=0)
    axR.set_ylim(len(plot_keys) - 0.42, -0.62)
    lo = min(pooled[k]["lo"] for k in plot_keys)
    hi = max(pooled[k]["hi"] for k in plot_keys)
    rng = hi - lo
    axR.set_xlim(lo - 0.10 * rng, hi + 0.42 * rng)
    S.zeroline(axR, x=True)
    axR.set_xlabel("paired $\\Delta$ (10k cell bootstrap)")
    axR.set_title("Pooled paired $\\Delta$"
                  + (" (n = %d cells)" % common_n if common_n else ""))
    axR.grid(axis="y", visible=False)

    # ------------------------------------------------------------- legends ---
    fig.legend(handles=[Line2D([], [], marker=M_DOT, ls="none", ms=4.6,
                               color=S.METHOD.get(m, S.GREY), label=m)
                        for m in methods],
               loc="upper center", bbox_to_anchor=(0.5, 1.035), ncol=len(methods),
               handletextpad=0.35, columnspacing=1.5)

    # proxies are real scatter artists, so the legend shows the exact edge style
    h_v9 = axL.scatter([], [], marker=M_DOT, s=S_FILLED, color=S.DARKGREY,
                       edgecolors=S.DARKGREY, linewidths=0.8)
    h_ar = axL.scatter([], [], marker=M_DOT, s=S_OPEN, facecolors="none",
                       edgecolors=S.DARKGREY, linewidths=1.15)
    h_un = axL.scatter([], [], marker=M_DOT, s=S_OPEN, facecolors="none",
                       edgecolors=S.DARKGREY, linewidths=1.15, linestyle=DOTTED)
    h_bf = axL.scatter([], [], marker=M_BUDGET_FAIL, s=S_GLYPH, color=S.DARKGREY,
                       linewidths=1.1)
    h_nr = axL.scatter([], [], marker=M_NEVER_RUN, s=S_GLYPH, color=S.DARKGREY)
    h_tk = Line2D([], [], marker="|", ls="-", ms=5.0, lw=0.8, color=S.GREY)
    # Each entry names an encoding and nothing else. Which cells are as-run,
    # which arm is ungated, how many dots moved and how many are true-v9 are
    # counts, not encodings: they are in the caption and in fig5_data.csv.
    enc = [(h_v9, "filled: true-v9 gate"),
           (h_ar, "open: as-run gate"),
           (h_un, "dotted: ungated gate"),
           (h_tk, "as-run value"),
           (h_bf, "zeroed: budget fail"),
           (h_nr, "zeroed: never run")]
    fig.legend([h for h, _ in enc], [l for _, l in enc], loc="upper center",
               bbox_to_anchor=(0.5, 0.128), ncol=3, handletextpad=0.5,
               columnspacing=1.9, labelspacing=0.4)

    pdf, png = S.save(fig, "fig5")

    # ------------------------------------------------------------- Table B ---
    tb = table_b_tex(PS, blocked) if PS else None

    # ----------------------------------------------------------------- CSV ---
    for name, val, what in [
        ("n_dots", n_dots, "cells x methods plotted in the left panel"),
        ("n_true_v9_dots", n_true,
         "dots drawn filled (gate_summary.n_true_v9_dots_total)"),
        ("n_not_true_v9_dots", n_dots - n_true, "dots drawn open (as-run or UNGATED)"),
        ("n_edit_cells_as_run", n_edit_asrun,
         "edit cells from results/x: as-run, no alpha_trace, no re-scorable twin ("
         + ", ".join(gs.get("edit_cells_as_run", [])) + ")"),
        ("n_ungated_dots", sum(ungated.values()),
         "dots with gate_mode=UNGATED ("
         + ", ".join("%s=%d" % (k, v) for k, v in ungated.items()) + ")"),
        ("n_zeroed_non_measurements", n_zeroed,
         "dots that are nan->0 rather than measured removals"),
        ("n_dots_changed_by_v9_rescore", len(changed),
         "; ".join("%s %s: %+.6f -> %+.6f"
                   % (d["cell"], d["method"], d["as_run"], d["v9"]) for d in changed)),
        ("n_restricted_cells", len(restricted_cells),
         "fully re-scorable edit cells: " + ", ".join(restricted_cells)),
    ]:
        csv_rows.append(["count", "figure annotation", name, "", "", "", "", str(val),
                         "", "", "", "", "", "", "", "", "", "", "", "", "", "",
                         FRONTIER, what])

    if tb:
        _, (plo, phi_), (ilo, ihi) = tb
        for nm, v, w in [
            ("table_b_artifact_size_min_MiB", plo,
             "s=0.99 merged patch, payload only (bytes = retained/8 + 2 B/tensor)"),
            ("table_b_artifact_size_max_MiB", phi_,
             "s=0.99 merged patch, payload only; the manuscript prints MiB as 'MB'"),
            ("table_b_artifact_size_min_with_index_MiB", ilo,
             "same patches once an entropy-bound support index is charged"),
            ("table_b_artifact_size_max_with_index_MiB", ihi,
             "same patches once an entropy-bound support index is charged"),
        ]:
            if v is not None:
                csv_rows.append(["table_b", "LaTeX table beside the figure (not plotted)",
                                 nm, "", "", "", "", "%.6f" % v, "", "", "", "", "", "",
                                 "", "v9 (patch bytes are gate-independent)", "", "",
                                 "True", "", "", "", SIZES, w])

    header = ["row_type", "panel", "estimand_or_cell", "target", "axis",
              "method_or_comparator", "y_position", "value", "value_as_run",
              "delta_v9_minus_as_run", "ci_lo", "ci_hi", "ci_status", "n_cells",
              "n_seeds", "gate_provenance", "gate_mode", "true_v9", "is_measurement",
              "zeroed_reason", "marker", "marker_fill", "source", "note"]
    csv = C.write_csv(os.path.join(C.FIGDIR, "fig5_data.csv"), header, csv_rows)

    # ------------------------------------------------------------- caption ---
    rk = restricted_keys[0] if restricted_keys else None
    cells_by = {c["cell"]: c for c in cells}
    n_targets = len({c["target"] for c in cells})

    def ci_txt(k):
        return "%s = %s (n=%d, %s)" % (
            k.split("[")[0].strip().replace(" - ", " %s " % MINUS),
            fmt_ci(pooled[k]), pooled[k]["n"], pooled[k]["status"])

    sig = [k for k in plot_keys if pooled[k]["status"] == "excludes 0"]
    # restricted values the artifact carries but the figure refuses to draw
    unpublished = [(comparator(k), pooled[k]["n"]) for k in sorted(pooled)
                   if "re-scorable cells only" in k and not avail.get(k)]
    # the two nan->0 reasons, kept apart in the caption exactly as their glyphs
    # are kept apart in the image
    zbr = {}
    for c in cells:
        for m in methods:
            d = c["methods"][m]
            if d.get("zeroed"):
                zbr.setdefault(zero_kind(d), []).append(
                    "%s %s (%s)" % (c["cell"], m, d.get("zeroed_reason", "")))
    # dots the re-score moved and whose as-run digit the draft's Table A still prints
    stale = [d["cell"] for d in changed
             if (cells_by[d["cell"]]["methods"][d["method"]].get("draft") or {})
             .get("matches_as_run")
             and not (cells_by[d["cell"]]["methods"][d["method"]].get("draft") or {})
             .get("matches_v9")]

    caption = " ".join(x for x in [

        "Figure 5: the frontier. LEFT: per-cell removal under the strict v9 "
        "collateral budget, for %d methods on the %d headline cells "
        "(%d targets $\\times$ %d axes). Methods are separated by colour and by "
        "a fixed vertical offset within each cell row; the two axis groups are "
        "divided by a horizontal rule. Each dot is the mean over %s seeds of a "
        "per-seed argmax over in-budget configurations, so every dot is a "
        "selection-biased maximum — equally so for every method, which is what "
        "keeps the cross-method comparison fair even though each absolute value is "
        "optimistic."
        % (len(methods), len(cells), n_targets, len(axis_order), seed_txt),

        "RIGHT: pooled paired deltas over the same %d cells, 10k percentile "
        "bootstrap resampling the (target, axis) CELL and never the seed, so "
        "seed-to-seed spread is invisible in these intervals: %s."
        % (len(cells), "; ".join(ci_txt(k) for k in pooled_keys)),

        ("Only %s excludes 0, and it is the one estimate whose point value is "
         "printed inside the panel; every other value in this figure is read from "
         "its position, and is given numerically here."
         % " and ".join(k.split("[")[0].strip().replace(" - ", " %s " % MINUS)
                        for k in sig)) if sig else "",

        ("RESTRICTED ESTIMATE (open diamond, bottom row). Restricted to the %d fully "
         "re-scorable edit cells (%s): %s. That restricted estimate is published for "
         "%s only. The %s restricted values are also in frontier_v9.json but are v10 "
         "additions with no published v9 counterpart, the two restriction rules differ "
         "from one another, and an n=2 percentile bootstrap is degenerate rather than "
         "inferential — so none of them is drawn as an interval anywhere in this "
         "figure."
         % (pooled[rk]["n"], "/".join(restricted_targets), ci_txt(rk),
            comparator(rk),
            " and ".join("%s (n=%d)" % (a, n) for a, n in unpublished)))
        if rk else "No published restricted estimate exists for any arm, so none is drawn.",

        "MIXED GATE — three provenances, not two, encoded by marker fill. Only "
        "%d of %d dots are true-v9 (filled). The %d edit cells on %s are as-run and can "
        "never be re-scored (open marker, solid edge): they come from results/x, which "
        "carries no alpha_trace and has no re-scorable twin, so their boundary exposure "
        "is unknown and is not assumed to be zero. All %d DPO dots carry "
        "gate_mode=UNGATED (open marker, dotted edge), because that panel could not be "
        "re-gated from removal.jsonl and its collateral check is still the buggy as-run "
        "float gate — a wider exposure than the manuscript's current gate-mixing "
        "disclosure, which names only the edit cells. Consequently no pooled estimand is "
        "fully v9: every one of them contains the edit arm's %d as-run cells, and edit "
        "%s DPO contains the wholly ungated DPO arm as well. That is why every pooled "
        "diamond in the right panel carries the same dotted edge, and it is the reason "
        "the panel is not titled as a clean v9 comparison."
        % (n_true, n_dots, n_edit_asrun, "/".join(asrun_targets),
           ungated.get("DPO", 0), n_edit_asrun, MINUS),

        "%d of %d dots ARE NOT MEASUREMENTS. They are 0.0 manufactured by the project's "
        "nan$\\to$0 rule, and because the two reasons differ they do not share a glyph. "
        "$\\times$ = measured, but no configuration was ever in budget: %s. "
        "$\\varnothing$ = never run at all: %s."
        % (n_zeroed, n_dots,
           "; ".join(sorted(zbr.get("budget_fail", []))) or "none",
           "; ".join(sorted(zbr.get("never_run", []))) or "none"),

        "%d dots MOVED UNDER THE V9 RE-SCORE (%s); the grey tick marks the as-run "
        "position and the grey segment joins it to the v9 dot.%s"
        % (len(changed),
           "; ".join("%s %s %+.3f$\\to$%+.3f"
                     % (d["cell"], d["method"], d["as_run"], d["v9"]) for d in changed),
           (" The draft's Table A still prints the as-run digit at %s, so the figure and "
            "that table will disagree until the table is updated."
            % ", ".join(stale)) if stale else ""),

        "%s A dagger on a row label marks %s, which fuses q/k/v into one qkv_proj and "
        "whose edit never touches o_proj: it is not parameterization-matched to %s, and "
        "it supplies 2 of the %d cells behind the restricted estimate."
        % (DAGGER, "/".join(fused_targets), "/".join(other_targets),
           len(restricted_cells)),

        "LEGEND IN FULL: filled = true-v9 gate; open with a solid edge = as-run gate; "
        "open with a dotted edge = ungated gate; $\\times$ = zeroed, budget fail; "
        "$\\varnothing$ = zeroed, never run; grey tick = the as-run value of a dot the "
        "v9 re-score moved. Table B, printed beside this figure, carries the deployment "
        "properties that decide the behavioural tie; its artifact-size cell is in MiB "
        "(bytes/2$^{20}$), which the manuscript prints as MB.",

    ] if x).replace("+-", MINUS)

    print("wrote:", pdf)
    print("wrote:", png)
    print("wrote:", csv)
    if tb:
        print("wrote:", tb[0])
    print("\nCAPTION:\n" + caption)
    if blocked:
        print("\nBLOCKED / DEGRADED:")
        for b in blocked:
            print(" -", b)
    return dict(pdf=pdf, png=png, csv=csv, table_b=(tb[0] if tb else None),
                caption=caption, blocked=blocked)


if __name__ == "__main__":
    build()
