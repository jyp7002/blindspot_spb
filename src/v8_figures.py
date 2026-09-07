"""v8 figures. Read-only over results/v8/*.json.

Following src/figures.py's convention: matplotlib Agg, DPI 150, every figure
individually guarded so a not-yet-run experiment prints "SKIP figN: <reason>"
instead of raising and killing the remaining figures.

    fig_env_envelope.png    ENV: the operating envelope, calibration vs held-out
    fig_dec_decomposition.png  DEC: information decomposition (needs the panel)
    fig_spc_curve.png       SPC: sparsity curve to 99.9% (needs the panel)

Static PNGs for a paper, so light mode only -- these render on white pages, and
a dark variant would be a second artifact nobody prints. Palette is the first two
categorical slots (blue/orange), validated all-pairs for scatter:
CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the light surface.
"""
import os, json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
V8 = os.path.join(RESULTS, "v8")
FIGDIR = os.path.join(RESULTS, "figures")

DPI = 150
CAL, HELD = "#2a78d6", "#eb6834"          # categorical slots 1, 2
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#9a998f"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)


def _scatter(ax, cells, xkey, color, label):
    ax.scatter([c[xkey] for c in cells], [c["removal"] for c in cells],
               s=70, c=color, label=label, edgecolors=SURFACE, linewidths=1.5,
               zorder=3)


def fig_env_envelope():
    fp = os.path.join(V8, "env_split.json")
    pol = os.path.join(V8, "env_policy.json")
    if not os.path.exists(fp) or not os.path.exists(pol):
        print("SKIP fig_env_envelope: env_split.json / env_policy.json missing")
        return
    sp = json.load(open(fp))
    th = json.load(open(pol))["thresholds"]
    cal, held, excl = sp["calibration"], sp["held_out"], sp["excluded"]
    tau = th["joint_gate"]["tau"]
    gamma = th["joint_gate"]["gamma"]

    allc = cal + held + excl
    lo = min(c["removal"] for c in allc)
    hi = max(c["removal"] for c in allc)
    pad = 0.10 * (hi - lo)
    ylo, yhi = lo - pad * 2.2, hi + pad     # extra room below for the labels

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, xkey, thr, xlabel in (
            (axes[0], "abs_pre_skew", tau, "|pre_skew|  (available bias)"),
            (axes[1], "contrast_gap", gamma, "contrast_gap  (elicitation quality)")):
        _style(ax)
        ax.set_ylim(ylo, yhi)
        # backfire region: removal < 0
        ax.axhspan(ylo, 0, color=MUTED, alpha=0.10, zorder=0)
        ax.axhline(0, color=MUTED, linewidth=1.0, zorder=1)
        if thr is not None:
            ax.axvline(thr, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
            lab = f" τ={thr:.3f}" if xkey == "abs_pre_skew" else f" γ={thr:.3f}"
            # blended transform: x in data units, y in axes units, so the label
            # sits below the legend row rather than colliding with it at the top
            from matplotlib.transforms import blended_transform_factory
            ax.text(thr, 0.62, lab, color=INK2, fontsize=7.5, va="center",
                    ha="left", zorder=4,
                    transform=blended_transform_factory(ax.transData, ax.transAxes),
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        if excl:
            ax.scatter([c[xkey] for c in excl], [c["removal"] for c in excl],
                       s=70, marker="x", c=MUTED, linewidths=1.6, zorder=3,
                       label="excluded (protocol-inoperable)")
        _scatter(ax, cal, xkey, CAL, f"calibration (n={len(cal)})")
        _scatter(ax, held, xkey, HELD, f"held-out (n={len(held)})")
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)

        # Selective direct labels: ONLY the backfires -- they are the point of the
        # figure. They also cluster tightly, so labels are stacked in a reserved
        # band under the data with leader lines instead of being pinned to each
        # marker, which collided illegibly.
        back = sorted((c for c in cal + held if c["removal"] < 0),
                      key=lambda c: c[xkey])
        if back:
            x0, x1 = ax.get_xlim()
            band = ylo + 0.06 * (yhi - ylo)
            step = (yhi - ylo) * 0.052
            for i, c in enumerate(back):
                short = c["axis"].replace("bbq_", "").replace("crows_", "")
                ty = band + (len(back) - 1 - i) * step
                # keep the text inside the axes at both edges
                frac = (c[xkey] - x0) / (x1 - x0)
                ha = "left" if frac < 0.2 else ("right" if frac > 0.8 else "center")
                ax.annotate(f"{c['target']}|{short}", (c[xkey], c["removal"]),
                            xytext=(c[xkey], ty), fontsize=7, color=INK2,
                            ha=ha, va="center",
                            arrowprops=dict(arrowstyle="-", color=MUTED,
                                            linewidth=0.7, shrinkA=1, shrinkB=4))
    axes[0].set_ylabel("removal (in-budget, nan→0)", color=INK2, fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK2)

    fig.suptitle("The operating envelope: removal tracks available bias, but the gate "
                 "does not generalise", color=INK, fontsize=11, x=0.5, y=0.99)
    # Caption sits below the axes rather than inside a panel: in-panel it collided
    # with the rightmost backfire label.
    fig.text(0.5, 0.015,
             "Thresholds τ, γ were chosen on the calibration split only. No held-out "
             "cell falls below either, so every gated policy degenerates to edit-all; "
             "labelled points are backfires (removal < 0).",
             ha="center", va="bottom", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    out = os.path.join(FIGDIR, "fig_env_envelope.png")
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def fig_spc_curve():
    rows = []
    for tier in ("small", "big"):
        fp = os.path.join(RESULTS, "v8spc", tier, "removal.jsonl")
        if os.path.exists(fp):
            rows += [dict(json.loads(l), tier=tier) for l in open(fp) if l.strip()]
    if not rows:
        print("SKIP fig_spc_curve: no v8spc rows yet")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=SURFACE)
    _style(ax)
    for tier, color, lab in (("small", CAL, "≤3.8B"), ("big", HELD, "7–9B")):
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        by, mb, nfail = {}, {}, {}
        for r in sub:
            rem = r.get("removal")
            failed = rem is None or rem != rem
            by.setdefault(r["sparsity"], []).append(0.0 if failed else rem)
            nfail[r["sparsity"]] = nfail.get(r["sparsity"], 0) + int(failed)
            if r.get("bytes"):
                mb.setdefault(r["sparsity"], []).append(r["bytes"] / 1e6)
        xs = sorted(by)
        ys = [float(np.mean(by[x])) for x in xs]
        ax.plot([1 - x for x in xs], ys, color=color, linewidth=2.0,
                marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=lab, zorder=3)
        # A point whose mean includes nan->0 rows is depressed by BUDGET FAILURE,
        # not by loss of removal. Unmarked, the 7-9B dip at 0.1-0.5 retained reads
        # as a sparsity effect when it is the dense edit failing the collateral
        # budget. Ring those points and say so in the caption.
        fx = [1 - x for x in xs if nfail.get(x)]
        fy = [float(np.mean(by[x])) for x in xs if nfail.get(x)]
        if fx:
            ax.scatter(fx, fy, s=150, facecolors="none", edgecolors=color,
                       linewidths=1.8, zorder=4)
        # §SPC wants the x-axis in triplicate (retained fraction / nonzero count /
        # patch bytes). Fraction is the axis; patch size is carried as a selective
        # endpoint label at the two extremes, so the deployment number is on the
        # figure without putting a value on every point.
        for x, ha, dx in ((xs[0], "left", 4), (xs[-1], "right", -4)):
            if x in mb:
                ax.annotate(f"{np.mean(mb[x]):.2f} MB", (1 - x, float(np.mean(by[x]))),
                            textcoords="offset points", xytext=(dx, -14),
                            fontsize=7, color=INK2, ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("retained parameter fraction (log)", color=INK2, fontsize=9)
    ax.set_ylabel("removal (in-budget)", color=INK2, fontsize=9)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    ax.set_title("Sparsity curve to 99.9%", color=INK, fontsize=11)
    fig.text(0.5, 0.015,
             "Ringed points include budget-fail seeds (nan→0): the 7–9B dip at "
             "0.1–0.5 retained is the dense edit failing the\ncollateral budget, "
             "not loss of removal. No 7–9B failure occurs at 99% sparsity or beyond.",
             ha="center", va="bottom", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    out = os.path.join(FIGDIR, "fig_spc_curve.png")
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def fig_dec_decomposition():
    fp = os.path.join(V8, "dec_analysis.json")
    if not os.path.exists(fp):
        print("SKIP fig_dec_decomposition: dec_analysis.json missing")
        return
    d = json.load(open(fp))
    if d.get("adjudication", {}).get("verdict") == "NOT ADJUDICATED":
        print("SKIP fig_dec_decomposition: panel incomplete, not adjudicated")
        return
    means = d["per_condition_mean"]
    order = ["C-ref", "C-a", "C-bottom", "C-b", "C-layershuf", "C-tensorshuf", "C-rand"]
    order = [c for c in order if c in means]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    _style(ax)
    vals = [means[c] for c in order]
    cols = [CAL if c == "C-ref" else HELD for c in order]
    bars = ax.bar(range(len(order)), vals, color=cols, width=0.62,
                  edgecolor=SURFACE, linewidth=2.0, zorder=3)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.3f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=1)
    # Cell counts differ by condition: the architecture guard drops C-tensorshuf
    # (and every LOPO) for phi's fused-qkv cells, so those bars are means over 7
    # cells against 10 for the rest. Unlabelled, the bars would invite a direct
    # comparison that is not valid.
    ncell = {c: sum(1 for m in d["per_cell"].values() if c in m) for c in order}
    nmax = max(ncell.values())
    labels = [c if ncell[c] == nmax else f"{c}\n(n={ncell[c]})" for c in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8, color=INK2, rotation=20, ha="right")
    ax.set_ylabel(f"mean removal over cells (n={nmax} unless noted)",
                  color=INK2, fontsize=9)
    ax.set_title(f"DEC information decomposition — "
                 f"{d['adjudication']['verdict']}", color=INK, fontsize=11)
    sel = d["estimands"].get("Delta_selection (C-ref - C-a)")
    if sel:
        ax.text(0.98, 0.94,
                f"Δ_selection = {sel['point']:+.4f} "
                f"[{sel['lo']:+.3f}, {sel['hi']:+.3f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig_dec_decomposition.png")
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    for fn in (fig_env_envelope, fig_spc_curve, fig_dec_decomposition):
        try:
            fn()
        except Exception as e:
            print(f"SKIP {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
