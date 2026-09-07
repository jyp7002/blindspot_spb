"""Fig 1 — the information-removal pipeline (drawn schematic, artifact-driven callouts).

WS-B B1: "FP32 magnitudes -> one bit per weight -> 1% of coordinates -> effect
preserved -> merged patch", framed as WHAT IS REMOVED at each stage rather than
as a software pipeline. Nothing here is plotted data; every number on the panel
is read from a v10 artifact at build time (the R2-bug rule extends to figures).

Sources, both v9-gated:
  results/v10/patch_sizes.json     (v9 tree; panels v6trace/ablate{,_big})
  results/v10/dec_analysis_v9.json (results_v9/v8dec, 285 rows, integer-item gate)

The panel carries STAGES, QUANTITIES and HONESTY ENCODINGS. It carries no
prose: the footnote keys, the two provenance lines, the unit derivation and the
phi caveat that used to be drawn under the axes now live in figures/captions.md,
which is what a reader of the paper actually sees beside the figure. What
survives on the canvas, and why it is not prose:

  * The full-precision stage is drawn with a DASHED, HATCHED, grey box because it
    is the one stage that was never measured: no `bytes` field for the `fp`
    variant exists anywhere in results_v9/. Its range is DERIVED here as
    edited-tensor params x 2 bytes (bf16) and is labelled "computed, not
    measured" — three words, an encoding, not a sentence.
  * The 1%-sparse size is the PAYLOAD ONLY. The extra cost of shipping the
    support (information-theoretic floor, from the artifact's index_cost block)
    is drawn as a second, grey, smaller line inside the stage, so the headline
    range is visibly a payload-only number. § keys the byte formula in the caption.
  * The plan says "FP32 magnitudes". No artifact supports FP32: the edits live in
    bf16, and the `fp` variant was never sized at all. The stage is therefore
    labelled bf16 and its range is derived at 2 bytes/parameter — which reproduces
    the 648-2560 MiB the size verification independently states.
  * The unit is MiB (bytes / 2**20) and is printed on every quantity. The
    manuscript prints "MB"; the decimal-MB inflation the panel used to spell out
    is computed into the CSV and stated in the caption.
  * The effect number carries a dagger: the per-condition CI is a v10 addition
    (registered:false), not a re-scored v9 quantity — v9 bootstrapped paired
    differences only. ‡ / § / † are keyed in the caption, not on the canvas.
  * n survives as data ("n = 6 targets", "10/10 cells"); the fused-qkv caveat,
    the panel names and the rows-per-cell counts moved to the caption.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
import v10_style as S
S.apply()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

SIZE_ART = "results/v10/patch_sizes.json"
DEC_ART = "results/v10/dec_analysis_v9.json"
BF16_BYTES = 2          # bf16 = 2 bytes/parameter. The one arithmetic constant.
FIG = "fig1"

# marker keys used on the panel
DAG, DDAG, SEC = "†", "‡", "§"


# ------------------------------------------------------------------ inputs ---

def _inputs():
    """Read both artifacts. Returns (payload, blocked) — never fabricates."""
    blocked = []
    size = C.jload(SIZE_ART)
    dec = C.jload(DEC_ART)
    if size is None:
        blocked.append(f"{SIZE_ART} absent — the patch-size callout cannot be drawn.")
    if dec is None:
        blocked.append(f"{DEC_ART} absent — the effect-preserved callout cannot be drawn.")
    if blocked:
        raise SystemExit("Fig 1 blocked:\n  " + "\n  ".join(blocked))

    per = size["per_target"]
    rng = size["patch_range_1pct"]["payload_only"]
    idx = size["patch_range_1pct"]["with_entropy_bound_index"]

    # bf16 baseline: DERIVED, never measured. Formula carried into the CSV.
    bf16 = {t: v["edited_n_params"] * BF16_BYTES / C.MIB for t, v in per.items()}
    dense = {t: v["dense"]["mib"] for t, v in per.items()}
    sparse = {t: v["s0.99"]["mib"] for t, v in per.items()}
    ship = {t: v["s0.99"]["shippable_mib"]["payload_plus_entropy_bound"]
            for t, v in per.items()}
    retained = {t: 1.0 - v["s0.99"]["effective_sparsity"] for t, v in per.items()}

    # the representative target is the one the artifact itself names as the range
    # maximum — not a hand-picked row.
    rep = rng["max_target"]
    repd = per[rep]
    bits_per_coord = [b for b in repd["bytes_fit"]
                      if b["variant"] == "s0.99"][0]["bits_per_retained_coordinate"]
    bits_all = {t: [b for b in v["bytes_fit"] if b["variant"] == "s0.99"][0]
                ["bits_per_retained_coordinate"] for t, v in per.items()}

    # self-checks against the artifact's own summary block
    checks = {
        "range_min_matches_per_target": abs(min(sparse.values()) - rng["min_mib"]) < 1e-9,
        "range_max_matches_per_target": abs(max(sparse.values()) - rng["max_mib"]) < 1e-9,
        "callout_reproduces_under_mib":
            bool(size["manuscript_callout"]["reproduces_under_mib"]),
        "callout_reproduces_under_decimal_mb":
            bool(size["manuscript_callout"]["reproduces_under_decimal_mb"]),
        "dec_density_matches_size_retained":
            abs(dec["coverage"]["density"][0] - retained[rep]) < 1e-6,
    }

    cref = dec["per_condition"]["C-ref"]
    p = dict(
        size=size, dec=dec, per=per, rng=rng, idx=idx,
        bf16=bf16, dense=dense, sparse=sparse, ship=ship, retained=retained,
        rep=rep, rep_row=repd, bits_per_coord=bits_per_coord,
        bits_all=bits_all, bits_per_coord_max=max(bits_all.values()),
        n_targets=len(per),
        n_rows_per_size_cell=repd["dense"]["n_rows"],
        fused=[t for t, v in per.items() if v["fused_qkv_only"]],
        drop=dense[rep] / sparse[rep],
        mb_inflation=rng["max_mb_decimal"] / rng["max_mib"] - 1.0,
        density=dec["coverage"]["density"][0],
        cref=cref, cref_ci=cref["ci"],
        n_cells=dec["n_cells"],
        unanimity=dec["unanimity"]["annotation"],
        checks=checks,
        panels=size["panels"]["contributing"],
        dec_path=dec["source"]["path"],
        dec_gate=dec["gate"],
        size_gate=size["gate"],
    )
    return p


# ------------------------------------------------------------------- draw ----

def _box(ax, xc, y0, y1, w, facecolor, edgecolor, ls="-", hatch=None, lw=1.0,
         alpha=1.0, zorder=2):
    b = FancyBboxPatch((xc - w / 2, y0), w, y1 - y0,
                       boxstyle="round,pad=0.0,rounding_size=0.010",
                       linewidth=lw, linestyle=ls, edgecolor=edgecolor,
                       facecolor=facecolor, alpha=alpha, hatch=hatch,
                       zorder=zorder, mutation_aspect=0.5)
    ax.add_patch(b)
    return b


def _fit_check(fig, ax, texts):
    """Warn if any label runs outside the panel. Keeps the tight bbox honest:
    a label that overflows would silently widen the saved figure and break the
    ICLR column width."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for t in texts:
        bb = t.get_window_extent(fig.canvas.get_renderer())
        (x0, y0), (x1, y1) = inv.transform(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        if x0 < -0.004 or x1 > 1.004 or y0 < -0.004 or y1 > 1.004:
            bad.append((t.get_text().split("\n")[0][:40], round(x0, 3), round(x1, 3)))
    return bad


def _box_fit(fig, ax, pairs):
    """Warn if an in-box label is wider than the stage box holding it. Text that
    spills over its own border is the failure mode this panel had; the check is
    cheaper than re-reading the PNG."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for t, w in pairs:
        bb = t.get_window_extent(fig.canvas.get_renderer())
        (x0, _), (x1, _) = inv.transform(((bb.x0, bb.y0), (bb.x1, bb.y1)))
        if (x1 - x0) > w - 0.006:
            bad.append((t.get_text().split("\n")[0][:40], round(x1 - x0, 4), round(w, 4)))
    return bad


def _collisions(fig, texts, pad=1.5):
    """Warn on any two labels whose rendered boxes overlap."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = [(t, t.get_window_extent(r)) for t in texts]
    bad = []
    for i in range(len(bb)):
        for j in range(i + 1, len(bb)):
            a, b = bb[i][1], bb[j][1]
            if (a.x0 - pad < b.x1 and b.x0 - pad < a.x1
                    and a.y0 - pad < b.y1 and b.y0 - pad < a.y1):
                bad.append((bb[i][0].get_text().split("\n")[0][:28],
                            bb[j][0].get_text().split("\n")[0][:28]))
    return bad


def build():
    p = _inputs()
    C.ensure_dirs()
    fs_s = plt.rcParams["legend.fontsize"]   # 7 pt, set by S.apply()
    blue, grey, dgrey, ink = S.COND["C-ref"], S.NULL_COLOR, S.DARKGREY, S.INK

    # WIDE, not FULL: the four prose blocks that used to fill the lower third
    # are in the caption now, so the diagram gets the whole canvas.
    fig = plt.figure(figsize=S.WIDE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Texture, not a second line weight. v10_figures.py builds every figure in
    # ONE process, so this global is restored below rather than left set.
    hlw0 = plt.rcParams["hatch.linewidth"]
    plt.rcParams["hatch.linewidth"] = 0.6
    T = []                                    # every text, for the fit check
    INBOX = []                                # (text, box width) for the box fit

    def txt(x, y, s, ha="center", va="center", color=ink, small=False, z=4, **kw):
        t = ax.text(x, y, s, ha=ha, va=va, color=color, zorder=z,
                    **({"fontsize": fs_s} if small else {}), **kw)
        T.append(t)
        return t

    rep = p["rep"]
    ret_pct = p["retained"][rep] * 100.0
    rem_pct = 100.0 - ret_pct
    nbits = BF16_BYTES * 8

    txt(0.015, 0.985, "What is removed at each stage", ha="left", va="top",
        fontsize=plt.rcParams["axes.titlesize"], fontweight="bold")

    n_st = 5
    bw, pitch = 0.178, 0.2005
    xc = [0.0975 + i * pitch for i in range(n_st)]
    by0, by1 = 0.505, 0.785
    mid = (by0 + by1) / 2

    # ---- stage boxes -------------------------------------------------------
    # Measured stages carry the C-ref blue (this pipeline IS the C-ref arm).
    # The full-precision stage is the one that was never measured, so it gets the
    # palette's reserved grey, dashed and hatched.
    titles = ["Full-precision\nΔW (bf16)", "One bit\nper weight", f"{ret_pct:.0f}% of\ncoordinates",
              "Effect\npreserved", "Merged\npatch"]
    subs = ["never sized", "sign + scale", "magnitude top-k", "DEC C-ref", "no runtime hook"]
    for i in range(n_st):
        if i == 0:
            _box(ax, xc[i], by0, by1, bw, "white", grey, ls=(0, (3, 2)),
                 hatch="//", lw=0.9)
        else:
            _box(ax, xc[i], by0, by1, bw, blue, blue, lw=0.0, alpha=0.10)
            _box(ax, xc[i], by0, by1, bw, "none", blue, lw=0.9)
        col = dgrey if i == 0 else ink
        bb = dict(facecolor="white", edgecolor="none", pad=1.6) if i == 0 else None
        a = txt(xc[i], mid + 0.034, titles[i], color=col, linespacing=1.30, bbox=bb)
        b = txt(xc[i], mid - 0.068, subs[i], color=dgrey, small=True, bbox=bb)
        INBOX += [(a, bw), (b, bw)]

    # ---- transition arrows: WHAT IS REMOVED --------------------------------
    removed = [f"− magnitudes\n({nbits - 1} of {nbits} bits)",
               f"− {rem_pct:.0f}% of\ncoordinates",
               "− nothing:\nmeasure survivors",
               "− the runtime\nhook"]
    for i in range(n_st - 1):
        x0, x1 = xc[i] + bw / 2 + 0.003, xc[i + 1] - bw / 2 - 0.003
        ax.add_patch(FancyArrowPatch((x0, mid), (x1, mid), arrowstyle="-|>",
                                     mutation_scale=7, lw=1.0, color=dgrey,
                                     shrinkA=0, shrinkB=0, zorder=3))
        txt((x0 + x1) / 2, 0.915, removed[i], va="top", color=dgrey, small=True,
            linespacing=1.25)

    # ---- per-stage quantities, below each box ------------------------------
    # These are the figure's data. Symbol keys (‡ § †) are defined in the
    # caption; the panel prints the numbers, not the explanations.
    ty = by0 - 0.045
    def under(i, lines, color=ink, dy=0.0):
        txt(xc[i], ty - dy, lines, va="top", color=color, small=True,
            linespacing=1.45)

    under(0, f"{min(p['bf16'].values()):.0f} – {max(p['bf16'].values()):.0f} MiB{DDAG}\n"
             f"computed,\nnot measured", grey)
    under(1, f"{min(p['dense'].values()):.1f} – {max(p['dense'].values()):.1f} MiB\n"
             f"measured, n = {p['n_targets']}\n"
             f"≤ {p['bits_per_coord_max']:.4f} bits/coord")
    under(2, f"{p['rng']['min_mib']:.2f} – {p['rng']['max_mib']:.2f} MiB{SEC}\n"
             f"payload only")
    under(2, f"+{p['idx']['min_mib']:.2f} – {p['idx']['max_mib']:.2f} MiB\n"
             f"with support index", grey, dy=0.118)
    under(3, f"removal {p['cref_ci']['point']:+.3f}{DAG}\n"
             f"[{p['cref_ci']['lo']:+.3f}, {p['cref_ci']['hi']:+.3f}]\n"
             f"{p['unanimity']}")

    # ---- the one data-driven callout ---------------------------------------
    cx0, cx1 = xc[0] - bw / 2, xc[-1] + bw / 2
    cy0, cy1 = 0.035, 0.165
    ax.add_patch(Rectangle((cx0, cy0), cx1 - cx0, cy1 - cy0, facecolor=blue,
                           alpha=0.08, edgecolor=blue, lw=0.7, zorder=1))
    txt((cx0 + cx1) / 2, (cy0 + cy1) / 2,
        f"{rep}:  {p['dense'][rep]:.1f} MiB dense  →  "
        f"{p['sparse'][rep]:.2f} MiB at {ret_pct:.0f}% of coordinates  —  "
        f"{p['drop']:.0f}× smaller")

    for b in _fit_check(fig, ax, T):
        print(f"[{FIG}] WARNING label overflows the panel: {b}")
    for b in _box_fit(fig, ax, INBOX):
        print(f"[{FIG}] WARNING label wider than its stage box: {b}")
    for b in _collisions(fig, T):
        print(f"[{FIG}] WARNING labels collide: {b}")

    pdf, png = S.save(fig, FIG)
    plt.rcParams["hatch.linewidth"] = hlw0

    # ------------------------------------------------------------- the CSV --
    hdr = ["figure", "stage_order", "stage", "item", "label", "value", "unit",
           "measured", "n", "source_file", "source_key", "note"]
    rows = []

    def R(order, stage, item, label, value, unit, measured, n, src, key, note):
        rows.append([FIG, order, stage, item, label, value, unit, measured, n,
                     src, key, note])

    R(1, "full-precision dW", "size_min", f"bf16 baseline, {p['rng']['min_target']}",
      f"{min(p['bf16'].values()):.4f}", "MiB", "derived", p["n_targets"], SIZE_ART,
      "per_target.*.edited_n_params", f"edited-tensor params x {BF16_BYTES} B (bf16) / 2**20; "
      "full precision was NEVER sized (no fp bytes field in results_v9/)")
    R(1, "full-precision dW", "size_max", f"bf16 baseline, {p['rng']['max_target']}",
      f"{max(p['bf16'].values()):.4f}", "MiB", "derived", p["n_targets"], SIZE_ART,
      "per_target.*.edited_n_params", f"edited-tensor params x {BF16_BYTES} B (bf16) / 2**20")
    for t in sorted(p["per"], key=lambda k: p["bf16"][k]):
        R(1, "full-precision dW", "per_target_bf16", t, f"{p['bf16'][t]:.4f}", "MiB",
          "derived", 1, SIZE_ART, f"per_target.{t}.edited_n_params",
          f"n_params={p['per'][t]['edited_n_params']:.0f} x {BF16_BYTES} B / 2**20")

    R(2, "1 bit per weight", "size_min", "dense binary patch, min",
      f"{min(p['dense'].values()):.4f}", "MiB", "measured", p["n_targets"], SIZE_ART,
      "per_target.*.dense.mib", "dense column is binary-per_tensor-s0.0 (1 bit/weight), "
      "NOT the full-precision edit")
    R(2, "1 bit per weight", "size_max", "dense binary patch, max",
      f"{max(p['dense'].values()):.4f}", "MiB", "measured", p["n_targets"], SIZE_ART,
      "per_target.*.dense.mib", "dense column is binary-per_tensor-s0.0 (1 bit/weight)")
    for t in sorted(p["per"], key=lambda k: p["dense"][k]):
        R(2, "1 bit per weight", "per_target_dense", t, f"{p['dense'][t]:.4f}", "MiB",
          "measured", p["per"][t]["dense"]["n_rows"], SIZE_ART, f"per_target.{t}.dense.mib",
          "fused_qkv_only=%s" % p["per"][t]["fused_qkv_only"])
    for t in sorted(p["bits_all"], key=lambda k: p["bits_all"][k]):
        R(2, "1 bit per weight", "bits_per_retained_coordinate", t,
          f"{p['bits_all'][t]:.6f}", "bits/coord", "measured",
          p["per"][t]["s0.99"]["n_rows"], SIZE_ART,
          f"per_target.{t}.bytes_fit[s0.99].bits_per_retained_coordinate",
          "residual over exactly one bit each is the per-tensor scales; "
          "the panel prints the max over targets")

    R(3, "1% of coordinates", "retained_fraction", "retained coordinate fraction",
      f"{p['retained'][rep]:.4f}", "fraction", "measured", p["n_targets"], SIZE_ART,
      f"per_target.{rep}.s0.99.effective_sparsity",
      "1 - effective_sparsity; equals dec coverage.density = %.2f" % p["density"])
    R(3, "1% of coordinates", "payload_min",
      f"1%-sparse payload, {p['rng']['min_target']}", f"{p['rng']['min_mib']:.4f}", "MiB",
      "measured", p["n_targets"], SIZE_ART, "patch_range_1pct.payload_only.min_mib",
      "PAYLOAD ONLY: bytes = retained/8 + 2 B per tensor; no support index priced")
    R(3, "1% of coordinates", "payload_max",
      f"1%-sparse payload, {p['rng']['max_target']}", f"{p['rng']['max_mib']:.4f}", "MiB",
      "measured", p["n_targets"], SIZE_ART, "patch_range_1pct.payload_only.max_mib",
      "PAYLOAD ONLY: bytes = retained/8 + 2 B per tensor; no support index priced")
    for t in sorted(p["per"], key=lambda k: p["sparse"][k]):
        R(3, "1% of coordinates", "per_target_sparse_payload", t, f"{p['sparse'][t]:.4f}",
          "MiB", "measured", p["per"][t]["s0.99"]["n_rows"], SIZE_ART,
          f"per_target.{t}.s0.99.mib",
          "spread across rows = %.3f bytes" % p["per"][t]["s0.99"]["spread_bytes"])
        R(3, "1% of coordinates", "per_target_shippable_entropy_bound", t,
          f"{p['ship'][t]:.4f}", "MiB", "measured", p["per"][t]["s0.99"]["n_rows"],
          SIZE_ART, f"per_target.{t}.s0.99.shippable_mib.payload_plus_entropy_bound",
          "payload + information-theoretic floor for naming the support")
    R(3, "1% of coordinates", "shippable_min",
      f"payload + support index, {p['idx']['min_target']}", f"{p['idx']['min_mib']:.4f}",
      "MiB", "measured", p["n_targets"], SIZE_ART,
      "patch_range_1pct.with_entropy_bound_index.min_mib",
      "EXCLUDED from the manuscript's headline range")
    R(3, "1% of coordinates", "shippable_max",
      f"payload + support index, {p['idx']['max_target']}", f"{p['idx']['max_mib']:.4f}",
      "MiB", "measured", p["n_targets"], SIZE_ART,
      "patch_range_1pct.with_entropy_bound_index.max_mib",
      "EXCLUDED from the manuscript's headline range")

    R(4, "effect preserved", "C-ref_mean", "DEC C-ref mean removal",
      f"{p['cref_ci']['point']:.4f}", "removal", "measured", p["n_cells"], DEC_ART,
      "per_condition.C-ref.ci.point",
      "cell = (target,axis) mean over 3 rows; %s" % p["dec_gate"])
    R(4, "effect preserved", "C-ref_ci_lo", "DEC C-ref CI lower",
      f"{p['cref_ci']['lo']:.4f}", "removal", "measured", p["n_cells"], DEC_ART,
      "per_condition.C-ref.ci.lo",
      "NEW IN v10, registered=false: v9 bootstrapped paired differences only")
    R(4, "effect preserved", "C-ref_ci_hi", "DEC C-ref CI upper",
      f"{p['cref_ci']['hi']:.4f}", "removal", "measured", p["n_cells"], DEC_ART,
      "per_condition.C-ref.ci.hi",
      "NEW IN v10, registered=false: v9 bootstrapped paired differences only")
    R(4, "effect preserved", "unanimity", "cells with C-ref > C-a",
      p["unanimity"], "cells", "measured", p["n_cells"], DEC_ART,
      "unanimity.annotation", "comparison = C-ref > C-a")
    R(4, "effect preserved", "dec_density", "DEC panel density",
      f"{p['density']:.4f}", "fraction", "measured", p["n_cells"], DEC_ART,
      "coverage.density[0]", "the DEC edit is the same construction as stage 3: "
      "sign(dW) at top-|dW| coordinates, one scale per tensor")

    R(5, "merged patch", "callout_dense", f"{rep} dense", f"{p['dense'][rep]:.4f}",
      "MiB", "measured", p["per"][rep]["dense"]["n_rows"], SIZE_ART,
      f"per_target.{rep}.dense.mib", "representative target = artifact's own range max")
    R(5, "merged patch", "callout_sparse", f"{rep} at 1%", f"{p['sparse'][rep]:.4f}",
      "MiB", "measured", p["per"][rep]["s0.99"]["n_rows"], SIZE_ART,
      f"per_target.{rep}.s0.99.mib", "representative target = artifact's own range max")
    R(5, "merged patch", "callout_drop", f"{rep} dense / 1% size ratio",
      f"{p['drop']:.3f}", "x", "derived", p["per"][rep]["s0.99"]["n_rows"], SIZE_ART,
      f"per_target.{rep}.dense.mib / per_target.{rep}.s0.99.mib", "payload only")
    R(1, "full-precision dW", "bits_removed_per_weight", "magnitude bits removed",
      f"{BF16_BYTES * 8 - 1} of {BF16_BYTES * 8}", "bits", "derived", p["n_targets"],
      SIZE_ART, "conventions.dense_column_is",
      "bf16 carries 16 bits/coordinate; the patch keeps the sign bit "
      "(+ one scale per tensor)")
    R(3, "1% of coordinates", "removed_coordinate_fraction",
      "coordinates removed by magnitude top-k",
      f"{p['per'][rep]['s0.99']['effective_sparsity']:.4f}", "fraction", "measured",
      p["n_targets"], SIZE_ART, f"per_target.{rep}.s0.99.effective_sparsity",
      "the panel prints this as %.0f%%" % (p["per"][rep]["s0.99"]["effective_sparsity"] * 100))
    R(0, "unit", "mib_vs_decimal_mb", "decimal MB inflation over MiB",
      f"{p['mb_inflation']:.6f}", "fraction", "derived", p["n_targets"], SIZE_ART,
      "patch_range_1pct.payload_only.max_mb_decimal / .max_mib - 1",
      "the manuscript prints MB; the table only reproduces under MiB = bytes/2**20")
    for k, v in sorted(p["checks"].items()):
        R(0, "self_check", k, k, str(bool(v)), "bool", "check", "", SIZE_ART, "-",
          "build-time consistency check of the artifact against its own per-target rows")

    csv = C.write_csv(os.path.join(C.FIGDIR, f"{FIG}_data.csv"), hdr, rows)

    print(f"[{FIG}] {pdf}\n[{FIG}] {png}\n[{FIG}] {csv}  ({len(rows)} data rows)")
    for k, v in sorted(p["checks"].items()):
        print(f"[{FIG}] check {k}: {v}")
    return dict(pdf=pdf, png=png, csv=csv, checks=p["checks"])


if __name__ == "__main__":
    build()
