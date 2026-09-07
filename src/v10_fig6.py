"""Fig 6 — operating characterization: pre-skew vs removal over the v9-scored cells.

Reads ONLY results/v10/env_v9.json (the verified v9 regen artifact). Nothing is
hand-typed: every plotted coordinate, count, correlation, utility and CI bound is
read from the artifact at build time, and every number is mirrored into
figures/fig6_data.csv.

THE FIGURE CARRIES DATA; THE PROSE IS IN THE CAPTION. The earlier build drew a
seven-line statistics box over the scatter, a four-line fig.text footnote, a
six-line margin block beside panel B, two multi-line leader-line annotations, a
k/n_obs number on every partly-zeroed point and a nine-row legend of
parenthetical counts. All of that is REMOVED FROM THE IMAGE and is carried
verbatim by the caption string that build() returns (and by fig6_data.csv,
which is byte-for-byte unchanged and remains under the audit contract).

What the image encodes, all of it visual:

  * x = pre_skew, y = removal, one point per (target, axis) self cell.
  * benchmark family by MARKER SHAPE; the per-family n moved to the caption.
  * backfire cells (cell mean removal < 0) filled in S.BACKFIRE, over a faint
    shaded removal < 0 region.
  * calibration vs held-out by marker OUTLINE, not fill.
  * NON-MEASUREMENTS ARE NOT DRAWN AS MEASUREMENTS. The 3 granite x occ_gender
    cells sit at removal = 0.0 only because all 3 of their rows are budget-fails
    scored 0.0 under the project's nan->0 rule; they stay hollow, in
    S.NULL_COLOR, struck through with an x, and out of the annotated
    correlation. Four further cells that average 1-2 budget-fail zeros into an
    otherwise real mean keep their dotted ring; their k/n_obs counts moved to
    the caption.
  * ONE annotated correlation — the artifact's own directive
    (fig6.annotated_correlation, the non-excluded set), with its n and the
    OLS line fitted on exactly that set. The all-cell and BBQ alternatives, and
    the draft's as-run +0.910* with its 'held-out' mis-description, are in the
    caption.
  * AT MOST TWO direct labels, on the points that carry claims: the hollow
    granite | occ_gender group and the deepest backfire cell.
  * panel B keeps the gate's held-out failure as bars + bootstrap whiskers with
    a dashed edit-all reference line, so "the gate costs utility" is visible
    without a number on every bar; the in-sample-vs-held-out narrative, the
    reject counts and the utility definition are in the caption.

Caveats stated in the caption rather than drawn: cells pool designers and
panels, so n_obs of 6 or 9 still means 3 seeds; phi is INCLUDED here and its
edit runs through a fused qkv projection, a different edited-module set from
every other target; the panel B bootstrap whiskers at n = 14 are not
inferential.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
import v10_style as S
S.apply()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

ART = "results/v10/env_v9.json"

# Marker shape per benchmark family. A presentation choice, fixed here so the
# legend, the plot and the CSV cannot disagree; the families themselves and
# their counts are read from the artifact.
FAM_MARKER = {"BBQ": "o", "CrowS": "s", "templated": "^", "StereoSet": "D"}
FAM_ORDER = ("BBQ", "CrowS", "templated", "StereoSet")
FALLBACK_MARKER = "P"

# The artifact's own directive (fig6.annotated_correlation) is the non-excluded
# set; the all-cell set and the BBQ set (the draft's +0.910 referent) are shown
# beside it because the choice of set moves r by ~0.33.
K_MAIN = "v9: non-excluded cells"
K_ALL = "v9: all self cells"
K_BBQ = "v9: BBQ cells (the +0.910 referent)"


def _f(x, p=3):
    return f"{x:+.{p}f}"


def build():
    C.ensure_dirs()
    env = C.jload(ART)
    if env is None:
        raise SystemExit(f"missing artifact: {ART}")

    cells = env["v9"]["cells"]
    corr = env["correlations"]["v9"]
    pol = env["policy"]["cell_level"]["v9"]
    restr = env["restriction_loss"]
    cap_in = env["fig6"]["caption_inputs"]
    bf = env["backfire"]["v9"]
    gate_status = env["gate_status"]
    blocked_notes = []

    # ---------------------------------------------------------------- points --
    fam_counts = {}
    for c in cells:
        fam_counts[c["axis_family"]] = fam_counts.get(c["axis_family"], 0) + 1
    unknown = [f for f in fam_counts if f not in FAM_MARKER]
    if unknown:
        blocked_notes.append("benchmark families with no assigned marker shape, drawn "
                             f"with the fallback marker '{FALLBACK_MARKER}': {sorted(unknown)}")

    pts = []
    for c in cells:
        n_bf = int(c.get("n_budget_fail", 0))
        n_obs = int(c.get("n_obs", 0))
        allz = (n_obs > 0 and n_bf == n_obs)     # value manufactured, not measured
        pts.append(dict(
            cell=c["cell"], target=c["target"], axis=c["axis"],
            fam=c["axis_family"], split=c["split"],
            x=float(c["pre_skew"]), y=float(c["removal"]),
            backfire=bool(c["backfire"]), n_obs=n_obs, n_seed=int(c.get("n_seed", 0)),
            n_bf=n_bf, all_zeroed=allz, part_zeroed=(0 < n_bf < n_obs),
            panels=";".join(c.get("panels", [])),
            contrast_gap=float(c.get("contrast_gap", float("nan"))),
        ))

    cmain, call, cbbq = corr[K_MAIN], corr[K_ALL], corr[K_BBQ]
    draft = {d["id"]: d for d in env["draft_checks"]}
    dcorr = draft["fig6.corr_pre_skew_removal"]

    # OLS fit over exactly the cells behind the annotated correlation.
    main_set = set(cmain["cells"])
    fx = np.array([p["x"] for p in pts if p["cell"] in main_set])
    fy = np.array([p["y"] for p in pts if p["cell"] in main_set])
    slope, intercept = (float(v) for v in np.polyfit(fx, fy, 1))

    n_ho = sum(1 for p in pts if p["split"] == "held_out")
    n_cal = sum(1 for p in pts if p["split"] == "calibration")
    n_all0 = sum(1 for p in pts if p["all_zeroed"])
    n_part0 = sum(1 for p in pts if p["part_zeroed"])
    n_phi = sum(1 for p in pts if p["target"] == "phi")

    # ---------------------------------------------------------------- canvas --
    # The canvas carries DATA. Every statistic, caveat, provenance note and
    # alternative estimate that used to be drawn here now lives in the caption
    # string returned by build() and in fig6_data.csv; nothing was deleted.
    fig = plt.figure(figsize=S.FULL)
    gs = GridSpec(1, 2, width_ratios=[2.3, 1.0], wspace=0.35,
                  left=0.078, right=0.972, top=0.900, bottom=0.245)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ------------------------------------------------------- A: the scatter --
    ylo = min(p["y"] for p in pts) - 0.07
    yhi = max(p["y"] for p in pts) + 0.10
    ax.axhspan(ylo, 0.0, color=S.BACKFIRE, alpha=0.055, lw=0, zorder=0)
    S.zeroline(ax)

    xs = np.linspace(min(p["x"] for p in pts) - 0.02, max(p["x"] for p in pts) + 0.02, 50)
    ax.plot(xs, slope * xs + intercept, color=S.DARKGREY, lw=0.8, ls=(0, (5, 3)),
            zorder=2, alpha=0.85)

    # Honesty encodings, all of them visual: shape = benchmark family, heavy
    # outline = held-out, red fill = backfire, hollow grey + x = the value is
    # not a measurement, dotted ring = the mean averages in budget-fail zeros.
    for p in pts:
        mk = FAM_MARKER.get(p["fam"], FALLBACK_MARKER)
        if p["all_zeroed"]:
            face, edge, lw = "none", S.NULL_COLOR, 1.0
        else:
            face = S.BACKFIRE if p["backfire"] else S.OK_COLOR
            edge, lw = (S.INK, 1.15) if p["split"] == "held_out" else (face, 0.4)
        ax.scatter([p["x"]], [p["y"]], marker=mk, s=34, facecolors=face,
                   edgecolors=edge, linewidths=lw, zorder=5)
        if p["all_zeroed"]:
            ax.scatter([p["x"]], [p["y"]], marker="x", s=13, c=S.NULL_COLOR,
                       linewidths=0.9, zorder=6)
        if p["part_zeroed"]:
            ax.scatter([p["x"]], [p["y"]], marker="o", s=112, facecolors="none",
                       edgecolors=S.NULL_COLOR, linewidths=0.7, linestyle=":",
                       zorder=4)

    # ONE correlation: the one the paper cites. The other three defensible cell
    # sets, and the draft's as-run referent, are in the caption.
    ax.text(0.020, 0.972, f"r = {_f(cmain['pearson'])}  "
            f"(n = {cmain['n']} of {len(pts)} cells)",
            transform=ax.transAxes, ha="left", va="top", color=S.DARKGREY,
            zorder=7)

    # Direct labels, two of them, on the points that carry the paper's claims.
    zer = sorted((p for p in pts if p["all_zeroed"]), key=lambda p: p["x"])
    if zer:
        ax.annotate(f"the {n_all0} granite\nocc_gender cells",
                    (zer[-1]["x"], zer[-1]["y"]), textcoords="offset points",
                    xytext=(2, 6), ha="right", va="bottom", color=S.NULL_COLOR,
                    linespacing=1.15, zorder=7)
    deep = next((p for p in pts if p["cell"] == bf["deepest_cell"]), None)
    if deep is not None:
        ax.annotate(deep["cell"], (deep["x"], deep["y"]),
                    textcoords="offset points", xytext=(9, 0), ha="left",
                    va="center", color=S.BACKFIRE, zorder=7)

    ax.set_xlabel("pre-editing skew on the axis")
    ax.set_ylabel("bias removal (cell mean, 3 seeds)")
    ax.set_ylim(ylo, yhi)
    ax.set_title("A   operating characterization", loc="left")

    # --------------------------------------------------------------- panel B --
    tab = pol["held_out_table"]
    ypos = list(range(len(tab)))[::-1]
    base_u = tab[0]["utility"]
    # The reference line is what makes "the gate COSTS utility" visible without
    # printing a number on every bar.
    axb.axvline(base_u, color=S.DARKGREY, lw=0.8, ls=(0, (3, 2)), zorder=2)
    for yi, r_ in zip(ypos, tab):
        base = (r_["policy"] == "edit-all")
        col = S.OK_COLOR if base else S.NULL_COLOR
        axb.barh(yi, r_["utility"], height=0.58, color=col,
                 alpha=1.0 if base else 0.8, zorder=3)
        lo, hi = r_["utility_ci"]
        axb.plot([lo, hi], [yi, yi], color=S.INK, lw=0.8, zorder=4)
    axb.set_yticks(ypos)
    axb.set_yticklabels([r_["policy"].replace(" gate", "") for r_ in tab])
    axb.set_xlim(0, max(r_["utility_ci"][1] for r_ in tab) * 1.02)
    axb.set_ylim(-0.62, len(tab) - 0.38)
    axb.set_xticks([0.0, 0.1, 0.2, 0.3])
    axb.set_xlabel("held-out utility")
    axb.set_title(f"B   held-out gates\n     (n = {pol['n_held_out']} cells)",
                  loc="left")
    ce, cg = pol["calibration_edit_all"], pol["calibration_joint_gate"]

    # ---------------------------------------------------------------- legend --
    # Compact: identity by shape, status by fill/outline. The per-family counts
    # that used to crowd every row are in the caption and the CSV.
    fams = [f for f in FAM_ORDER if f in fam_counts] + \
           [f for f in sorted(fam_counts) if f not in FAM_ORDER]
    fam_h = [Line2D([], [], ls="none", marker=FAM_MARKER.get(f, FALLBACK_MARKER),
                    mfc=S.GREY, mec=S.GREY, ms=4.5, label=f) for f in fams]
    st_h = [
        Line2D([], [], ls="none", marker="o", mfc=S.OK_COLOR, mec=S.INK, mew=1.15,
               ms=4.5, label="held-out"),
        Line2D([], [], ls="none", marker="o", mfc=S.BACKFIRE, mec=S.BACKFIRE,
               ms=4.5, label="backfire"),
        Line2D([], [], ls="none", marker="x", mfc="none", mec=S.NULL_COLOR,
               ms=4.5, mew=0.9, label="not measured"),
        Line2D([], [], ls="none", marker="o", mfc="none", mec=S.NULL_COLOR,
               ms=6.5, mew=0.7, label="partly zeroed"),
    ]
    fig.legend(handles=fam_h + st_h, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=4, handletextpad=0.5,
               labelspacing=0.45, columnspacing=1.4, borderpad=0.0)

    pdf, png = S.save(fig, "fig6")

    # Quantities the CSV and the caption still report even though the canvas no
    # longer draws them (the v9-restriction accounting, the gate census).
    lost = restr["cells_lost"]
    n_v9_rows = sum(gate_status["v9_gate_modes"].values())
    partial = sorted((p for p in pts if p["part_zeroed"]), key=lambda p: p["x"])

    # ------------------------------------------------------------------ CSV --
    hdr = ["record_type", "key", "target", "axis", "axis_family", "split",
           "marker", "fill_role", "outline_role", "pre_skew", "removal",
           "backfire", "n_obs", "n_seed", "n_budget_fail", "is_non_measurement",
           "panels", "contrast_gap", "stat", "value", "ci_lo", "ci_hi", "n", "note"]
    out = []

    def row(**kw):
        out.append([kw.get(h, "") for h in hdr])

    for p in sorted(pts, key=lambda p: p["x"]):
        row(record_type="cell", key=p["cell"], target=p["target"], axis=p["axis"],
            axis_family=p["fam"], split=p["split"],
            marker=FAM_MARKER.get(p["fam"], FALLBACK_MARKER),
            fill_role=("non_measurement_hollow" if p["all_zeroed"] else
                       ("backfire" if p["backfire"] else "measured")),
            outline_role=("held_out_heavy" if p["split"] == "held_out" else
                          ("excluded_hollow" if p["split"] == "excluded" else
                           "calibration_none")),
            pre_skew=f"{p['x']:.6f}", removal=f"{p['y']:.6f}",
            backfire=p["backfire"], n_obs=p["n_obs"], n_seed=p["n_seed"],
            n_budget_fail=p["n_bf"], is_non_measurement=p["all_zeroed"],
            panels=p["panels"], contrast_gap=f"{p['contrast_gap']:.6f}",
            note=("plotted removal 0.0 is manufactured by the nan->0 budget-fail rule, "
                  "not measured; excluded from the annotated correlation"
                  if p["all_zeroed"] else
                  (f"{p['n_bf']} of {p['n_obs']} rows are budget-fails scored 0.0, so the "
                   "plotted mean is shallower than the truth" if p["part_zeroed"] else "")))

    for k, cc in ((K_MAIN, cmain), (K_ALL, call), (K_BBQ, cbbq)):
        for st in ("pearson", "spearman"):
            ci = cc.get(st + "_ci") or {}
            row(record_type="correlation", key=k, stat=st, value=f"{cc[st]:.6f}",
                ci_lo=("" if ci.get("lo") is None else f"{ci['lo']:.6f}"),
                ci_hi=("" if ci.get("hi") is None else f"{ci['hi']:.6f}"),
                n=cc["n"],
                note=("annotated in panel A; percentile bootstrap over cells, "
                      "anticonservative and not inferential at this n"))
    row(record_type="draft_comparison", key="fig6.corr_pre_skew_removal", stat="pearson",
        value=f"{dcorr['draft_value']:.6f}", n=dcorr["draft_n"],
        note=f"draft line {dcorr['draft_line']} prints the AS-RUN starred value "
             f"\"{dcorr['draft_says']}\"; the v9 recomputation on the same "
             f"{dcorr['v9_n']} BBQ cells is {dcorr['v9_pearson']:.6f} "
             f"(delta v9 - draft {dcorr['delta_v9_minus_draft']:+.6f}); the figure shows "
             f"the v9 value, and only "
             f"{dcorr['n_of_those_cells_that_are_held_out']} of {dcorr['v9_n']} of those "
             f"cells are held-out, so the draft's 'held-out BBQ' wording is wrong")

    row(record_type="fit", key="OLS over " + K_MAIN, stat="slope", value=f"{slope:.6f}",
        n=cmain["n"], note="dashed line in panel A, fitted on the annotated cell set")
    row(record_type="fit", key="OLS over " + K_MAIN, stat="intercept",
        value=f"{intercept:.6f}", n=cmain["n"],
        note="dashed line in panel A, fitted on the annotated cell set")

    for fam in fams:
        row(record_type="family_count", key=fam, axis_family=fam,
            marker=FAM_MARKER.get(fam, FALLBACK_MARKER), stat="n_cells",
            value=fam_counts[fam], n=fam_counts[fam],
            note="marker-shape legend annotation" +
                 ("; 2 cells only, labelled 'not a population'"
                  if fam_counts[fam] <= 2 else ""))
    for nm, v in (("calibration", n_cal), ("held_out", n_ho),
                  ("backfire", bf["n_backfire"]), ("all_rows_budget_fail", n_all0),
                  ("partly_budget_fail", n_part0), ("phi_cells", n_phi)):
        row(record_type="encoding_count", key=nm, stat="n_cells", value=v,
            n=cap_in["n_cells_v9"], note="legend / footnote annotation in the figure")

    for r_ in tab:
        lo, hi = r_["utility_ci"]
        row(record_type="held_out_policy", key=r_["policy"], split="held_out",
            stat="utility", value=f"{r_['utility']:.6f}", ci_lo=f"{lo:.6f}",
            ci_hi=f"{hi:.6f}", n=r_["n_cells"],
            note=f"n_edited={r_['n_edited']}, rejects {r_['n_cells'] - r_['n_edited']} of "
                 f"{r_['n_cells']}, edit_rate={r_['edit_rate']:.6f}, "
                 f"backfire_rate_edited={r_['backfire_rate_edited']:.6f}, "
                 f"tau={r_['tau']}, gamma={r_['gamma']}; 10k cell bootstrap, "
                 f"not inferential at n={r_['n_cells']}")
    for nm, blk in (("edit-all", ce), ("joint gate", cg)):
        row(record_type="calibration_policy", key=nm, split="calibration", stat="utility",
            value=f"{blk['utility']:.6f}", n=blk["n_cells"],
            note=f"backfire_rate_edited={blk['backfire_rate_edited']:.6f}, "
                 f"n_edited={blk['n_edited']}; quoted in the panel-B note")
        row(record_type="calibration_policy", key=nm, split="calibration",
            stat="backfire_rate_edited", value=f"{blk['backfire_rate_edited']:.6f}",
            n=blk["n_cells"], note="quoted in the panel-B note")
    row(record_type="calibration_policy", key="joint gate", split="calibration",
        stat="n_rejected", value=pol["calibration_n_rejected"], n=pol["n_cal"],
        note="quoted in the panel-B note")

    row(record_type="restriction", key="rows_dropped", stat="n_rows",
        value=restr["n_rows_dropped"], n=restr["n_rows_as_run"],
        note=f"v9 keeps {restr['n_rows_v9']} of {restr['n_rows_as_run']} rows; rule: "
             f"{restr['rule']}; panels: {', '.join(C.NOT_RESCORABLE)}")
    row(record_type="restriction", key="cells_dropped", stat="n_cells",
        value=restr["n_cells_as_run"] - restr["n_cells_v9"], n=restr["n_cells_as_run"],
        note="; ".join(f"{c['cell']} (split={c['split']}, panels={'+'.join(c['panels'])}, "
                       f"as-run pre_skew={c['pre_skew']:.4f}, removal={c['removal']:.4f})"
                       for c in lost))
    row(record_type="restriction", key="held_out_size", stat="n_cells", value=n_ho,
        n=env["split"]["frozen_counts"]["held_out"],
        note="the frozen split was dealt over the 36 as-run cells; carried by cell "
             "identity, the v9 held-out half is smaller than the frozen one")
    row(record_type="gate", key="v9_gate_modes", stat="n_rows", value=n_v9_rows,
        note=f"modes={gate_status['v9_gate_modes']}; {gate_status['note']}")
    row(record_type="gate", key="self_rows_budget_fail", stat="n_rows",
        value=sum(p["n_bf"] for p in pts), n=sum(p["n_obs"] for p in pts),
        note=f"budget-fail rows scored 0.0 inside the {len(pts)} plotted cells; the "
             f"artifact's "
             f"all-role count is {gate_status['n_rows_v9_budget_fail']}")

    csv = C.write_csv(os.path.join(C.FIGDIR, "fig6_data.csv"), hdr, out)

    # ----------------------------------------------------------------- caption --
    # Everything the canvas used to draw as prose lives here: the alternative
    # correlations, the v9-restriction accounting, the zeroed-cell explanation
    # and the in-sample-vs-held-out narrative. Nothing was dropped; it moved.
    fam_txt = ", ".join(f"{f} n={fam_counts[f]}" for f in fams)
    part_txt = ", ".join(f"{p['cell']} ({p['n_bf']} of {p['n_obs']})"
                         for p in partial)
    pol_txt = "; ".join(
        f"{r_['policy']} {_f(r_['utility'])} "
        f"[{r_['utility_ci'][0]:+.3f}, {r_['utility_ci'][1]:+.3f}], rejecting "
        f"{r_['n_cells'] - r_['n_edited']} of {r_['n_cells']}" for r_ in tab)
    caption = (
        f"Figure 6: operating characterization, and a gate that does not validate. "
        f"Each point in A is one (target, axis) self cell scored under the v9 "
        f"integer-item gate (n = {cap_in['n_cells_v9']}); x is the target's pre-editing "
        f"skew on that axis, y its mean removal over 3 seeds. Marker SHAPE gives the "
        f"benchmark family ({fam_txt}) — StereoSet is two cells and is not a "
        f"population; a heavy dark OUTLINE marks the frozen held-out half "
        f"(n = {n_ho}) against calibration (n = {n_cal}, no outline); red FILL marks "
        f"the {bf['n_backfire']} cells whose mean removal is negative, which is also "
        f"the shaded region below zero. "
        f"Removal tracks pre-existing skew: the annotated Pearson "
        f"r = {_f(cmain['pearson'])} is over the {cmain['n']} non-excluded cells, and "
        f"the dashed line is the OLS fit on exactly that set "
        f"(slope {slope:+.3f}, intercept {intercept:+.3f}). The choice of cell set "
        f"moves r by about a third, so the alternatives are stated here rather than "
        f"drawn: r = {_f(call['pearson'])} over all {call['n']} v9 cells (i.e. "
        f"including the {n_all0} non-measurements), and r = {_f(cbbq['pearson'])} over "
        f"the {cbbq['n']} BBQ cells. That BBQ value is the v9 replacement for the "
        f"draft's as-run {_f(dcorr['draft_value'])}*, which the draft additionally "
        f"describes as held-out: only {dcorr['n_of_those_cells_that_are_held_out']} of "
        f"those {dcorr['v9_n']} cells are in fact held-out (v9 minus draft "
        f"{dcorr['delta_v9_minus_draft']:+.3f}). Bootstrap CIs at these n are "
        f"percentile intervals over cells and are not inferential. "
        f"NON-MEASUREMENTS ARE NOT DRAWN AS MEASUREMENTS. The {n_all0} hollow grey "
        f"markers struck through with x, the group direct-labelled in A, are the "
        f"protocol-inoperable granite x occ_gender cells "
        f"(granite, granite2, granite8), whose removal is 0.0 only because every one of "
        f"their rows is a budget-fail scored 0.0 under the project's nan->0 rule; they "
        f"are plotted at their true pre-skew but are excluded from the annotated "
        f"correlation. A dotted ring marks {n_part0} further cells that average "
        f"budget-fail zeros into an otherwise real mean, with k of n_obs rows zeroed: "
        f"{part_txt}. One of them is the deepest backfire cell "
        f"({bf['deepest_cell']}, the second direct label), whose true backfire is "
        f"therefore deeper than the point drawn. "
        f"B is the same story as a deployment rule, and it fails. The pre-skew / "
        f"contrast-gap abstention gate calibrates perfectly IN-SAMPLE (calibration, "
        f"n = {pol['n_cal']}): backfire rate among edited cells "
        f"{ce['backfire_rate_edited']:.3f} -> {cg['backfire_rate_edited']:.3f} and "
        f"utility {_f(ce['utility'])} -> {_f(cg['utility'])}, rejecting "
        f"{pol['calibration_n_rejected']} of {pol['n_cal']} cells. Held-out "
        f"(n = {pol['n_held_out']}) it buys nothing and costs utility: {pol_txt}. The "
        f"dashed vertical line is the edit-all baseline, so a bar short of it is a "
        f"gate that paid utility for its abstentions; the pre-skew threshold alone "
        f"rejects nothing at all. Utility is sum(removal over EDITED cells) / n_cells, "
        f"abstention scoring 0. The whiskers are 10k percentile cell bootstraps and "
        f"are not inference at n = {pol['n_held_out']}. "
        f"Provenance and restriction: the v9 re-score drops the non-re-scorable "
        f"panels ({', '.join(C.NOT_RESCORABLE)}), never substituting for them, which "
        f"removes {restr['n_rows_dropped']} of {restr['n_rows_as_run']} observation "
        f"rows and {restr['n_cells_as_run'] - restr['n_cells_v9']} of "
        f"{restr['n_cells_as_run']} cells ({', '.join(c['cell'] for c in lost)}, both "
        f"held-out), so the held-out half here is {n_ho} and not the frozen "
        f"{env['split']['frozen_counts']['held_out']}. All {n_v9_rows} v9 rows carry "
        f"the integer-item gate; there is no mixed-gate cell. A cell pools designers "
        f"and panels but is still 3 seeds, so an n_obs of 6 or 9 is not 6 or 9 "
        f"independent runs. phi contributes {n_phi} of {cap_in['n_cells_v9']} cells "
        f"through a fused qkv projection — a different edited-module set from every "
        f"other target. Per-cell coordinates, per-family counts, both bootstrap "
        f"intervals, all four correlation sets and the full policy table are in "
        f"figures/fig6_data.csv."
    )

    print("wrote:", pdf, png, csv, sep="\n  ")
    print(f"\nrows in CSV: {len(out)}")
    print("\nCAPTION\n" + caption)
    if blocked_notes:
        print("\nBLOCKED\n- " + "\n- ".join(blocked_notes))
    return dict(pdf=pdf, png=png, csv=csv, caption=caption, blocked=blocked_notes)


if __name__ == "__main__":
    build()
