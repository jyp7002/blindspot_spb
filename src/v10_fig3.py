"""Fig 3 — sparsity curves, both tiers (<=3.8B and 7-9B).

Reads ONLY verified v10 regen artifacts:
  * results/v10/spc_v9.json     — v9 (integer-item gate) SPC curves, both tiers,
                                  per-cell rows, budget-fail inventories,
                                  per-target patch sizes across the full sweep.
  * results/v10/patch_sizes.json — the 1% patch-size callout and its support-index
                                  cost (the payload-only defect).

Nothing is hand-typed. Every plotted value, every printed number in the
in-figure footnote, and every CSV row is read or derived from those two files at
run time.

WHAT THE IMAGE CARRIES, AND WHAT THE CAPTION CARRIES
---------------------------------------------------
The image carries DATA and the ENCODINGS that keep it honest, nothing else:
the two curves with their CI bands, the per-target deltas the bands resample,
filled/open markers for CI status, an open square for the dense reference
(a definition, not a measurement), red x markers for budget-fail rows (not
measurements either), open grey circles for the as-run fails the v9 gate
recovered, the drop-the-fails sensitivity curve, and the shaded cliff region.

Every sentence that used to be drawn on the canvas — the fail inventory, the
n = 2 degeneracy caveat, the draft adjudication, the zeroing sensitivity
numbers, the payload-only meaning of the MiB axis and its per-target range —
now lives in figures/captions.md. build() prints that prose to stdout, derived
from the same artifact values, so the caption cannot drift from the figure.
Nothing was deleted: every moved number is still a row in fig3_data.csv.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
import v10_style as S
S.apply()

import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SPC_PATH = "results/v10/spc_v9.json"
SIZE_PATH = "results/v10/patch_sizes.json"

TIER_ORDER = ("small", "big")
TIER_TITLE = {"small": "≤3.8B", "big": "7–9B"}
YLIM = (-0.70, 0.32)
WRAP = 96           # wrap width for the caption prose printed to stdout


# --------------------------------------------------------------- helpers ----

def _skey(sparsity):
    """The patch_sizes key inside spc_v9.json for a sparsity level."""
    return "s=%s" % sparsity


def _pct(x):
    return "%g%%" % (x * 100.0)


def _g(x, sig=3):
    return "%.*g" % (sig, x)


def _levels(spc, tier):
    """Levels ordered dense -> sparsest (decreasing retained fraction)."""
    T = spc["tiers"][tier]
    return T, sorted(T["levels"], key=lambda L: -L["retained"])


def _dense(levels):
    for L in levels:
        if L["is_dense_reference"]:
            return L
    raise RuntimeError("spc_v9.json: no dense reference level")


def _cell_mean_dropfail(cell):
    """Cell mean with budget-fail rows DROPPED instead of scored 0.0.
    The registered project rule is the zeroing one; this is the sensitivity."""
    v = [r["removal_raw"] for r in cell["rows"] if not r["budget_fail"]]
    return float(np.mean(v)) if v else None


def _paired(level, dense, valfn):
    """Paired delta (level - dense) over the cells, artifact key order — the
    same order v10_regen_spc used, which is what reproduces the published CIs."""
    pairs = []
    for k in level["cells"]:
        if k not in dense["cells"]:
            continue
        a, b = valfn(level["cells"][k]), valfn(dense["cells"][k])
        if a is None or b is None:
            continue
        pairs.append((a, b))
    return C.boot_paired(pairs)


def _target_label(sizes, tier, target):
    """Disambiguate the tier-colliding bare target names ('llama' is llama-3.2B
    in the small tier and llama-8B in the big one) using patch_sizes.json's
    (tier, target) -> draft_row_label mapping, which is keyed on n_params."""
    for label, rec in sizes["per_target"].items():
        if rec["tier"] == tier and rec["target"] == target:
            return label, bool(rec.get("fused_qkv_only"))
    return target, False


def _fail_key(r):
    return (r["target"], r["seed"], round(float(r["sparsity"]), 6))


# ------------------------------------------------------------------ build ---

def build():
    spc = C.jload(SPC_PATH)
    sizes = C.jload(SIZE_PATH)
    if spc is None:
        raise SystemExit("MISSING ARTIFACT: %s — Fig 3 cannot be drawn." % SPC_PATH)
    if sizes is None:
        raise SystemExit("MISSING ARTIFACT: %s — Fig 3 cannot be drawn." % SIZE_PATH)

    C.ensure_dirs()
    csv_rows = []

    # S.FULL, not FULL_TALL: with the footnote moved to the caption there is
    # nothing under the axes, so the panels get the whole canvas. Width is
    # S.TEXTWIDTH less a hair (the Fig 4 convention) so that the tick labels
    # bbox="tight" adds never push the render past the 5.5in ICLR column.
    fig, axes = plt.subplots(1, 2, figsize=(S.TEXTWIDTH - 0.06, S.FULL[1]),
                             sharey=True)
    fig.subplots_adjust(left=0.105, right=0.982, top=0.795, bottom=0.145, wspace=0.20)

    panel = {}

    for ax, tier in zip(axes, TIER_ORDER):
        col = S.TIER[tier]
        T, levels = _levels(spc, tier)
        dense = _dense(levels)
        tlabels = {t: _target_label(sizes, tier, t) for t in T["targets"]}

        # ---- registered v9 curve (budget-fail -> 0.0, the project rule) -----
        x, y, lo, hi, excl = [], [], [], [], []
        for L in levels:
            d = L["delta_vs_dense"]
            x.append(L["retained"])
            if L["is_dense_reference"]:
                y.append(0.0); lo.append(0.0); hi.append(0.0); excl.append(None)
            else:
                y.append(d["point"]); lo.append(d["lo"]); hi.append(d["hi"])
                excl.append(L["ci_status"] == "excludes 0")
            # self-check: the published delta must reproduce from the cells
            if not L["is_dense_reference"]:
                chk = _paired(L, dense, lambda c: c["mean"])
                assert abs(chk["point"] - d["point"]) < 1e-9, (tier, L["sparsity"])

        ax.fill_between(x, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, y, color=col, lw=1.3, zorder=4)
        S.zeroline(ax)

        for xi, yi, e in zip(x, y, excl):
            if e is None:                                    # the dense reference
                ax.plot([xi], [yi], marker="s", ms=5, mfc="white",
                        mec=col, mew=1.2, ls="none", zorder=6)
            else:
                ax.plot([xi], [yi], marker="o", ms=4.5,
                        mfc=col if e else "white", mec=col, mew=1.1,
                        ls="none", zorder=6)

        # ---- the per-target deltas the band is actually made of -------------
        for L in levels:
            for t, cell in L["cells"].items():
                dv = cell["mean"] - dense["cells"][t]["mean"]
                ax.plot([L["retained"]], [dv], marker="o", ms=1.9, ls="none",
                        color=col, alpha=0.75, zorder=5)
                csv_rows.append([tier, TIER_TITLE[tier], L["n_cells"],
                                 "per_target_delta_vs_dense", t, tlabels[t][0],
                                 L["sparsity"], L["retained"], "%.6f" % dv,
                                 "", "", "", 1, "removal delta", "v9",
                                 "one of the %d cells the band resamples" % L["n_cells"]])
                csv_rows.append([tier, TIER_TITLE[tier], L["n_cells"],
                                 "per_target_cell_mean", t, tlabels[t][0],
                                 L["sparsity"], L["retained"], "%.6f" % cell["mean"],
                                 "", "", "", cell["n_rows"], "removal", "v9",
                                 "mean over %d seed rows; %d budget-fail scored 0.0"
                                 % (cell["n_rows"], cell["n_budget_fail"])])

        for L, yi, l_, h_ in zip(levels, y, lo, hi):
            csv_rows.append([tier, TIER_TITLE[tier], L["n_cells"],
                             "delta_vs_dense_v9", "", "", L["sparsity"], L["retained"],
                             "%.6f" % yi, "%.6f" % l_, "%.6f" % h_,
                             L["ci_status"], L["n_cells"], "removal delta", "v9",
                             "REGISTERED v9 value; budget-fail scored 0.0"
                             + (" — dense reference, 0 by construction"
                                if L["is_dense_reference"] else "")])

        # ---- sensitivity: drop the budget-fails instead of zeroing them -----
        sx, sy, slo, shi, sfmt = [], [], [], [], {}
        for L in levels:
            if L["is_dense_reference"]:
                sx.append(L["retained"]); sy.append(0.0); slo.append(0.0); shi.append(0.0)
                continue
            d = _paired(L, dense, _cell_mean_dropfail)
            sx.append(L["retained"]); sy.append(d["point"])
            slo.append(d["lo"]); shi.append(d["hi"])
            sfmt[L["sparsity"]] = d
            csv_rows.append([tier, TIER_TITLE[tier], d["n"],
                             "delta_vs_dense_sensitivity_fails_dropped", "", "",
                             L["sparsity"], L["retained"], "%.6f" % d["point"],
                             "%.6f" % d["lo"], "%.6f" % d["hi"], C.status(d),
                             d["n"], "removal delta", "v9",
                             "NOT registered — v10 sensitivity to the nan->0 rule"])
        differs = max(abs(a - b) for a, b in zip(sy, y)) > 1e-9
        if differs:
            ax.plot(sx, sy, color=S.NULL_COLOR, lw=1.0, ls=(0, (4, 2)), zorder=3)

        # ---- budget-fail inventory ------------------------------------------
        bf = spc["budget_fail_encoding"][tier]
        v9_fails = bf["v9_fail_inventory"]
        v9_keys = {_fail_key(r) for r in v9_fails}
        recovered = [r for r in bf["as_run_fail_inventory"] if _fail_key(r) not in v9_keys]

        y_v9, y_rec = YLIM[0] + 0.100, YLIM[0] + 0.048
        for r in v9_fails:
            ax.plot([r["retained"]], [y_v9], marker="x", ms=4.5, mew=1.4,
                    color=S.BACKFIRE, ls="none", zorder=7)
            csv_rows.append([tier, TIER_TITLE[tier], "", "budget_fail_row_v9",
                             r["target"], tlabels.get(r["target"], (r["target"],))[0],
                             r["sparsity"], r["retained"], "%.1f" % r["scored_as"],
                             "", "", "", 1, "removal (scored)", "v9",
                             "seed %d — NOT a measurement; nan scored 0.0" % r["seed"]])
        for r in recovered:
            ax.plot([r["retained"]], [y_rec], marker="o", ms=3.6, mfc="white",
                    mec=S.NULL_COLOR, mew=0.9, ls="none", zorder=7)
            csv_rows.append([tier, TIER_TITLE[tier], "",
                             "budget_fail_row_as_run_recovered_by_v9",
                             r["target"], tlabels.get(r["target"], (r["target"],))[0],
                             r["sparsity"], r["retained"], "%.1f" % r["scored_as"],
                             "", "", "", 1, "removal (scored)", "as-run",
                             "seed %d — failed the as-run float gate, PASSES under v9"
                             % r["seed"]])
        csv_rows.append([tier, TIER_TITLE[tier], "", "annotation.n_budget_fail",
                         "", "", "", "", bf["n_fail_v9"], "", "", "",
                         T["n_rows"], "rows", "v9",
                         "%d of %d v9 rows are budget-fails (as-run: %d)"
                         % (bf["n_fail_v9"], T["n_rows"], bf["n_fail_as_run"])])

        # ---- cliff ----------------------------------------------------------
        cliff = [L for L in levels
                 if not L["is_dense_reference"]
                 and L["ci_status"] == "excludes 0" and L["delta_vs_dense"]["point"] < 0]
        if cliff:
            first = cliff[0]
            prev = [L for L in levels if L["retained"] > first["retained"]][-1]
            edge = float(np.sqrt(first["retained"] * prev["retained"]))
            ax.axvspan(edge, 1e-9, color=S.GREY, alpha=0.10, lw=0, zorder=0)
            # one word, naming the shaded region; the threshold, its CI and the
            # definition of "cliff" are in the caption.
            ax.text(float(np.sqrt(first["retained"] * cliff[-1]["retained"])),
                    YLIM[1] - 0.018, "cliff",
                    ha="center", va="top", color=S.DARKGREY,
                    fontsize=plt.rcParams["legend.fontsize"])
            for L in cliff:
                csv_rows.append([tier, TIER_TITLE[tier], L["n_cells"],
                                 "annotation.cliff_level", "", "", L["sparsity"],
                                 L["retained"], "%.6f" % L["delta_vs_dense"]["point"],
                                 "%.6f" % L["delta_vs_dense"]["lo"],
                                 "%.6f" % L["delta_vs_dense"]["hi"], L["ci_status"],
                                 L["n_cells"], "removal delta", "v9",
                                 "negative and excludes 0 -> shaded as cliff"])

        # ---- axes ------------------------------------------------------------
        ax.set_xscale("log")
        ax.set_xlim(1.55, 6.5e-4)              # inverted: dense on the left
        ax.set_ylim(*YLIM)
        decades = [L for L in levels
                   if abs(np.log10(L["retained"]) - round(np.log10(L["retained"]))) < 1e-6]
        ax.set_xticks([L["retained"] for L in decades])
        ax.set_xticklabels([_pct(L["retained"]) for L in decades])
        ax.set_xticks([L["retained"] for L in levels], minor=True)
        ax.set_xticklabels([], minor=True)
        ax.set_xlabel("retained coordinate fraction (log)")

        n_cells = len(dense["cells"])
        # n is data. The "band = the 2 targets" degeneracy caveat, the target
        # roster and the budget-fail census are prose: they are in the caption.
        ax.set_title("%s  (n = %d cells)" % (TIER_TITLE[tier], n_cells))

        # ---- secondary axis: patch payload MiB, RANGE over the tier ---------
        top = ax.twiny()
        top.set_xscale("log")
        top.set_xlim(ax.get_xlim())
        top.grid(False)
        top.set_xticks([L["retained"] for L in decades])
        tl = []
        for L in decades:
            vals = [T["patch_sizes"][t][_skey(L["sparsity"])]["mib_mean"]
                    for t in T["targets"]]
            tl.append("%s–%s" % (_g(min(vals)), _g(max(vals))))
            csv_rows.append([tier, TIER_TITLE[tier], "",
                             "patch_payload_mib_axis_range", "", "", L["sparsity"],
                             L["retained"], "%.4f–%.4f" % (min(vals), max(vals)),
                             "%.4f" % min(vals), "%.4f" % max(vals), "", len(vals),
                             "MiB (bytes/2**20)", "v9",
                             "printed on the secondary axis; payload only"])
        top.set_xticklabels(tl)
        top.set_xlabel("patch payload MiB (min–max)")
        top.set_xticks([L["retained"] for L in levels], minor=True)
        top.set_xticklabels([], minor=True)

        for t in T["targets"]:
            for L in levels:
                ps = T["patch_sizes"][t][_skey(L["sparsity"])]
                csv_rows.append([tier, TIER_TITLE[tier], "", "patch_payload_mib",
                                 t, tlabels[t][0], L["sparsity"], L["retained"],
                                 "%.6f" % ps["mib_mean"], "%.6f" % ps["mib_min"],
                                 "%.6f" % ps["mib_max"], "", ps["n"],
                                 "MiB (bytes/2**20)", "v9 (size is gate-invariant)",
                                 "payload only: bits + per-tensor scales, no support index"])

        panel[tier] = dict(ax=ax, levels=levels, dense=dense, bf=bf, sfmt=sfmt,
                           v9_fails=v9_fails, recovered=recovered, tlabels=tlabels,
                           differs=differs, T=T, cliff=cliff)

    axes[0].set_ylabel("Δ removal vs dense (paired)")

    # ------------------------------------------------------------ legends ---
    # Identity is never colour-alone: fill, marker shape and dash carry it.
    a0, cs = axes[0], S.TIER["small"]
    a0.legend(handles=[
        Line2D([], [], color=cs, lw=1.3, marker="o", ms=4.5, mfc=cs, mec=cs,
               label="CI excludes 0"),
        Line2D([], [], color=cs, lw=1.3, marker="o", ms=4.5, mfc="white", mec=cs,
               label="CI covers 0"),
        Line2D([], [], color=cs, lw=0, marker="s", ms=5, mfc="white", mec=cs,
               label="dense reference"),
        Line2D([], [], color=cs, lw=0, marker="o", ms=1.9, alpha=0.75,
               label="per-target Δ"),
        Patch(facecolor=cs, alpha=0.16, lw=0, label="95% CI"),
    ], loc="lower left", handlelength=1.5, handletextpad=0.6,
        borderaxespad=0.25, labelspacing=0.20, bbox_to_anchor=(0.0, 0.015))

    a1 = axes[1]
    h = [Line2D([], [], color=S.BACKFIRE, lw=0, marker="x", ms=4.5, mew=1.4,
                label="budget-fail (scored 0)"),
         Line2D([], [], color=S.NULL_COLOR, lw=0, marker="o", ms=3.6, mfc="white",
                label="recovered by v9 gate")]
    if panel["big"]["differs"]:
        h.append(Line2D([], [], color=S.NULL_COLOR, lw=1.0, ls=(0, (4, 2)),
                        label="fails dropped, not zeroed"))
    a1.legend(handles=h, loc="lower left", handlelength=1.5, handletextpad=0.6,
              borderaxespad=0.25, labelspacing=0.20, bbox_to_anchor=(0.0, 0.115))

    # ------------------------------------------ prose moved to the caption ---
    # These paragraphs used to be drawn under the axes. They are still computed
    # from the artifacts (their numbers are audited CSV rows below) but they are
    # PRINTED, not plotted: figures/captions.md is where a reader meets them.
    big = panel["big"]
    bbf = big["bf"]
    fails = big["v9_fails"]
    fail_rets = ", ".join(_pct(r["retained"]) for r in
                          sorted(fails, key=lambda r: -r["retained"]))
    fail_tgts = sorted({"%s seed %d" % (big["tlabels"][r["target"]][0], r["seed"])
                        for r in fails})
    dep = spc["deployability"]["big"]["v9"]["dense"]
    n_pass = sum(v["n_seeds_passing"] for v in dep.values())
    n_tot = sum(v["n_seeds"] for v in dep.values())
    env = spc["deployability"]["big"]
    v9_clean = env["v9"]["envelope"]["clean_from"]
    v9_maxs, asrun_maxs = env["v9"]["envelope"]["max_sparsity"], env["as_run"]["envelope"]["max_sparsity"]

    # Every level whose REGISTERED delta beats dense, and what the nan->0 rule
    # is worth there. This is the sensitivity the artifact does not carry.
    beats = [L for L in big["levels"]
             if not L["is_dense_reference"]
             and L["ci_status"] == "excludes 0" and L["delta_vs_dense"]["point"] > 0]

    pr = sizes["patch_range_1pct"]
    paras = [
        "%s v9 budget-fails: %d rows, all %s, at %s retained — one of them inside "
        "the dense reference, so every %s Δ is measured against a reference "
        "deflated by a non-measurement. %s: none."
        % (TIER_TITLE["big"], bbf["n_fail_v9"], "/".join(fail_tgts), fail_rets,
           TIER_TITLE["big"], TIER_TITLE["small"]),
        "The draft's \u201cdense through %s\u201d is the AS-RUN envelope: v9 recovers %d of %d as-run fails, "
        "shrinking it to dense through %s; and the dense %s edit passes the budget in %d of %d runs, "
        "so it is not undeployable."
        % (_pct(asrun_maxs), len(big["recovered"]), bbf["n_fail_as_run"], _pct(v9_maxs),
           TIER_TITLE["big"], n_pass, n_tot),
    ]
    if beats:
        paras.append(
            "Zeroing that one dense row is what makes sparse beat dense: at %s the registered "
            "%s become %s when the fails are dropped instead (grey dashed)."
            % ("/".join(_pct(L["sparsity"]) for L in beats),
               "/".join("%+.3f" % L["delta_vs_dense"]["point"] for L in beats),
               "/".join("%+.3f" % big["sfmt"][L["sparsity"]]["point"] for L in beats)))
    paras.append(
        "Patch MiB (bytes/2\u00b2\u2070) is payload only: %s\u2013%s at 1%% retained, %s\u2013%s once the support is named." %
        (_g(pr["payload_only"]["min_mib"]), _g(pr["payload_only"]["max_mib"]),
         _g(pr["with_entropy_bound_index"]["min_mib"]),
         _g(pr["with_entropy_bound_index"]["max_mib"])))

    caption_moves = [
        "%s targets: %s. %s targets: %s. phi-3.8B fuses q/k/v into qkv_proj, so "
        "its edit never touches o_proj and that cell comes from a different "
        "edited-module set." % (
            TIER_TITLE["small"],
            ", ".join(sorted(panel["small"]["tlabels"][t][0]
                             for t in panel["small"]["T"]["targets"])),
            TIER_TITLE["big"],
            ", ".join(sorted(big["tlabels"][t][0] for t in big["T"]["targets"]))),
        "Budget-fail census: %s has %d of %d rows, %s has %d of %d."
        % (TIER_TITLE["small"], panel["small"]["bf"]["n_fail_v9"],
           panel["small"]["T"]["n_rows"], TIER_TITLE["big"], bbf["n_fail_v9"],
           big["T"]["n_rows"]),
        "The %s bootstrap resamples n = %d target cells, so its band is literally "
        "[min, max] of the two per-target Δ and is not an inferential interval."
        % (TIER_TITLE["big"], len(big["dense"]["cells"])),
        "Cliff (shaded): the retained fractions whose Δ is negative with a CI "
        "excluding 0 — %s." % "; ".join(
            "%s from ≥%s sparsity (%+.3f [%+.3f, %+.3f] at %s)"
            % (TIER_TITLE[t],
               _pct(panel[t]["cliff"][0]["sparsity"]),
               panel[t]["cliff"][0]["delta_vs_dense"]["point"],
               panel[t]["cliff"][0]["delta_vs_dense"]["lo"],
               panel[t]["cliff"][0]["delta_vs_dense"]["hi"],
               _pct(panel[t]["cliff"][0]["sparsity"]))
            for t in TIER_ORDER if panel[t]["cliff"]),
    ] + paras + [
        "The secondary axis is a RANGE because patch size is per-target: no single "
        "MiB value describes a tier. Every per-target size is in fig3_data.csv.",
    ]

    print("\n  PROSE MOVED OUT OF THE IMAGE, INTO figures/captions.md")
    for para in caption_moves:
        for i, ln in enumerate(textwrap.wrap(para, WRAP)):
            print("    %s%s" % ("- " if i == 0 else "  ", ln))

    for L in beats:
        d = big["sfmt"][L["sparsity"]]
        csv_rows.append(["big", TIER_TITLE["big"], L["n_cells"],
                         "annotation.beats_dense_under_zeroing", "", "", L["sparsity"],
                         L["retained"], "%.6f" % L["delta_vs_dense"]["point"],
                         "%.6f" % L["delta_vs_dense"]["lo"], "%.6f" % L["delta_vs_dense"]["hi"],
                         L["ci_status"], L["n_cells"], "removal delta", "v9",
                         "sensitivity (fails dropped): %+.4f [%+.4f, %+.4f] (%s)"
                         % (d["point"], d["lo"], d["hi"], C.status(d))])

    # numbers printed in the footnote must be auditable from the CSV too
    csv_rows += [
        ["big", TIER_TITLE["big"], "", "annotation.dense_deployable_runs_v9", "", "",
         0.0, 1.0, n_pass, "", "", "", n_tot, "runs passing the collateral budget",
         "v9", "refutes the draft's 'too disruptive to deploy at all'"],
        ["big", TIER_TITLE["big"], "", "annotation.fail_envelope_max_sparsity_v9", "", "",
         v9_maxs, "", v9_maxs, "", "", "", bbf["n_fail_v9"], "sparsity", "v9",
         "deepest sparsity at which a v9 budget-fail occurs"],
        ["big", TIER_TITLE["big"], "", "annotation.fail_envelope_max_sparsity_as_run", "", "",
         asrun_maxs, "", asrun_maxs, "", "", "", bbf["n_fail_as_run"], "sparsity",
         "as-run", "the draft's 'dense through 97%' envelope — SUPERSEDED"],
        ["big", TIER_TITLE["big"], "", "annotation.fail_envelope_clean_from_v9", "", "",
         v9_clean, "", v9_clean, "", "", "", "", "sparsity", "v9",
         "every level at or beyond this sparsity is budget-clean under v9"],
        ["", "", "", "annotation.patch_1pct_payload_mib", "",
         "%s..%s" % (pr["payload_only"]["min_target"], pr["payload_only"]["max_target"]),
         0.99, 0.01, "%.4f–%.4f" % (pr["payload_only"]["min_mib"], pr["payload_only"]["max_mib"]),
         "%.4f" % pr["payload_only"]["min_mib"], "%.4f" % pr["payload_only"]["max_mib"],
         "", pr["n_targets"], "MiB (bytes/2**20)", "v9",
         "the manuscript's '0.4-1.6 MB' callout; payload only"],
        ["", "", "", "annotation.patch_1pct_with_support_index_mib", "",
         "%s..%s" % (pr["with_entropy_bound_index"]["min_target"],
                     pr["with_entropy_bound_index"]["max_target"]),
         0.99, 0.01, "%.4f–%.4f" % (pr["with_entropy_bound_index"]["min_mib"],
                                    pr["with_entropy_bound_index"]["max_mib"]),
         "%.4f" % pr["with_entropy_bound_index"]["min_mib"],
         "%.4f" % pr["with_entropy_bound_index"]["max_mib"], "", pr["n_targets"],
         "MiB (bytes/2**20)", "v9",
         "payload + information-theoretic floor for naming the support"],
    ]

    pdf, png = S.save(fig, "fig3")

    header = ["tier", "tier_label", "n_cells", "series", "target", "target_label",
              "sparsity", "retained_fraction", "value", "ci_lo", "ci_hi",
              "ci_status", "n", "unit", "gate", "note"]
    csv = C.write_csv(os.path.join(C.FIGDIR, "fig3_data.csv"), header, csv_rows)

    print("Fig 3 written:")
    for p in (pdf, png, csv):
        print("  %s (%d bytes)" % (p, os.path.getsize(p)))
    print("  rows in CSV: %d" % len(csv_rows))
    print("  v9 fails  small=%d big=%d | as-run fails small=%d big=%d"
          % (panel["small"]["bf"]["n_fail_v9"], bbf["n_fail_v9"],
             panel["small"]["bf"]["n_fail_as_run"], bbf["n_fail_as_run"]))
    print("  draft line 149 adjudication: 'dense through %s' is AS-RUN; v9 envelope is dense through %s."
          % (_pct(asrun_maxs), _pct(v9_maxs)))
    print("  dense 7-9B deploys in %d of %d runs (draft says 'not at all')." % (n_pass, n_tot))
    return dict(pdf=pdf, png=png, csv=csv)


if __name__ == "__main__":
    build()
