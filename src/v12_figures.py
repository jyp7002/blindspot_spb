#!/usr/bin/env python3
"""v12 figure panels — the scale-up, drawn. Reads artifacts only.

    python3 src/v12_figures.py            # every panel whose inputs exist

  fig2b  DEC over 20 cells: per-cell C-a -> C-ref dumbbells, sorted by C-ref,
         the 27-32B cells set apart. Companion panel to fig2 (the registered
         10-cell decomposition, unchanged). Source: results/v11/dec_big_v11.json.
  fig4c  Leave-one-projection-out at 27B / 32B: the drop in C-ref when each
         projection's surviving coordinates are removed. Source:
         results/v11/big_v11.json.
  fig3t  Sparsity by tier (<=3.8B / 7-9B / 27-32B): paired removal vs the dense
         one-bit edit, per retained fraction, cell bootstrap. Tiers 1-2 read the
         published re-scored SPC panel (results_v9/v8spc); tier 3 reads v12big
         and, if a frozen prediction exists, marks each cell's p*. Until v12big
         exists the third panel says so instead of being left out.

Same conventions as every v10 figure: v10_style sizes and Okabe-Ito colors
(C-ref blue, C-a orange, projections as v10_style.PROJ), the cell as the
resampling unit via v10_common.boot_paired, nan -> 0, and every plotted number
mirrored into figures/<name>_data.csv. Tier is encoded by position and label,
never by a new hue.
"""
import collections
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V      # noqa: E402
import v10_style as S       # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
BIG = {"gemma27b", "qwen32b"}
SIZE = {"gemma": "2.6B", "llama": "3B", "qwen": "3B", "phi": "3.8B",
        "qwen7b": "7B", "llama8b": "8B", "gemma27b": "27B", "qwen32b": "32B"}
AXIS_SHORT = {"occ_gender": "occ", "bbq_Age": "BBQ-age",
              "bbq_Race_ethnicity": "BBQ-race", "ss_intra": "SS",
              "crows_socioeconomic": "CrowS-SES"}


def _j(rel):
    fp = os.path.join(RESULTS, rel)
    return json.load(open(fp)) if os.path.exists(fp) else None


FAMILY = {"gemma": "gemma", "llama": "llama", "qwen": "qwen", "phi": "phi",
          "qwen7b": "qwen", "llama8b": "llama", "gemma27b": "gemma",
          "qwen32b": "qwen"}


def _label(cell):
    t, a = cell.split("|")
    return f"{FAMILY.get(t, t)} {SIZE.get(t, '')} · {AXIS_SHORT.get(a, a)}"


# ------------------------------------------------------------------ fig2b ----
def fig2b():
    d = _j("v11/dec_big_v11.json")
    if d is None:
        print("fig2b: results/v11/dec_big_v11.json missing -- skipped")
        return
    import matplotlib.pyplot as plt
    S.apply()
    cells = sorted(d["cells"].items(), key=lambda kv: kv[1]["C_ref"])
    big = [c for c, _ in cells if c.split("|")[0] in BIG]
    rest = [c for c, _ in cells if c.split("|")[0] not in BIG]
    order = rest + big                                   # big tier on top
    fig, ax = plt.subplots(figsize=(S.TEXTWIDTH, 4.2))
    rows = []
    for y, c in enumerate(order):
        v = d["cells"][c]
        is_big = c.split("|")[0] in BIG
        ax.plot([v["C_a"], v["C_ref"]], [y, y], color=S.GREY, lw=1.2, zorder=1)
        ax.scatter(v["C_a"], y, s=34, facecolor="white", edgecolor=S.COND["C-a"],
                   lw=1.6, zorder=3, label="C-a (random support)" if y == 0 else None)
        ax.scatter(v["C_ref"], y, s=40, color=S.COND["C-ref"], edgecolor="white",
                   lw=1.0, zorder=4, label="C-ref (top-|ΔW| support)" if y == 0 else None)
        rows.append(dict(cell=c, tier="27-32B" if is_big else "<=9B",
                         published=v.get("published"), C_ref=v["C_ref"], C_a=v["C_a"],
                         delta_selection=v["delta_selection"],
                         in_envelope=v.get("in_envelope")))
    if big:
        y0 = len(rest) - 0.5
        ax.axhspan(y0, len(order) - 0.5, color=S.GREY, alpha=0.12, lw=0, zorder=0)
        ax.axhline(y0, color=S.GREY, lw=0.6)
        ax.text(0.01, (y0 + len(order) - 0.5) / 2, "27–32B",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7, color=S.DARKGREY, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_label(c) + (" *" if d["cells"][c].get("published") else "")
                        for c in order], fontsize=7)
    for tl, c in zip(ax.get_yticklabels(), order):
        if c.split("|")[0] in BIG:
            tl.set_fontweight("bold")
    S.zeroline(ax, x=True)
    ax.set_xlabel("bias removal (cell mean over seeds, in-budget argmax)")
    g = d["groups"]
    gb = g.get("b_published_plus_in_envelope") or next(
        v for k, v in g.items() if k.startswith("b_"))
    ga = g["a_published"]
    ax.set_title(f"Δ_selection, {gb['n_cells']} cells: {gb['delta_selection']:+.3f} "
                 f"[{gb['ci_lo']:+.3f}, {gb['ci_hi']:+.3f}], {gb['unanimous']}/{gb['of']}"
                 f" positive · registered 10 (*): {ga['delta_selection']:+.3f}",
                 fontsize=8, loc="left")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    S.save(fig, "fig2b")
    V.write_csv(os.path.join(V.FIGDIR, "fig2b_data.csv"), list(rows[0]),
                [list(r.values()) for r in rows])
    print(f"fig2b: {len(order)} cells ({len(big)} at 27-32B)")


# ------------------------------------------------------------------ fig4c ----
def fig4c():
    d = _j("v11/big_v11.json")
    if d is None:
        print("fig4c: results/v11/big_v11.json missing -- skipped")
        return
    import matplotlib.pyplot as plt
    import numpy as np
    S.apply()
    projs = ["q_proj", "k_proj", "v_proj", "o_proj"]
    cells = [c for c in d["cells"] if c.split("|")[0] in BIG]
    fig, ax = plt.subplots(figsize=S.HALF)
    w, rows = 0.19, []
    for i, c in enumerate(cells):
        cond = d["cells"][c]["conditions"]
        ref = cond["C-ref"]
        drops = {p: ref - cond.get(f"C-ref-no_{p}", float("nan")) for p in projs}
        top = max(drops, key=lambda p: drops[p])
        for j, p in enumerate(projs):
            x = i + (j - 1.5) * w
            ax.bar(x, drops[p], width=w * 0.9, color=S.PROJ[p],
                   label=p.replace("_proj", "") if i == 0 else None)
            rows.append(dict(cell=c, projection=p, C_ref=ref,
                             C_ref_without=cond.get(f"C-ref-no_{p}"),
                             drop=drops[p], largest=(p == top)))
        ax.annotate(top.replace("_proj", ""), (i + (projs.index(top) - 1.5) * w,
                                               drops[top]),
                    xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=7, fontweight="bold")
    S.zeroline(ax)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([_label(c) for c in cells], fontsize=7)
    ax.set_ylabel("C-ref − C-ref without projection")
    ax.legend(ncol=4, fontsize=6.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    S.save(fig, "fig4c")
    V.write_csv(os.path.join(V.FIGDIR, "fig4c_data.csv"), list(rows[0]),
                [list(r.values()) for r in rows])
    print(f"fig4c: {len(cells)} cells")


# ------------------------------------------------------------------ fig3t ----
def _spc_rows():
    """Tier -> removal rows (target, axis, seed, variant, removal)."""
    out = collections.defaultdict(list)
    for tier, sub in (("<=3.8B", "small"), ("7-9B", "big")):
        fp = os.path.join(REPO, "results_v9", "v8spc", sub, "removal.jsonl")
        for ln in open(fp):
            r = json.loads(ln)
            # published big tier names its 7-9B targets qwen/llama
            t = {"qwen": "qwen7b", "llama": "llama8b"}[r["target"]] if sub == "big" \
                else r["target"]
            out[tier].append(dict(target=t, axis=r["axis"], seed=r["seed"],
                                  variant=r["variant"], removal=r["removal"]))
    fp = os.path.join(RESULTS, "v12big", "removal.jsonl")
    if os.path.exists(fp):
        last = {}
        for ln in open(fp):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            last[(r["target"], r["axis"], r["seed"], r["variant"])] = r
        out["27-32B"] = list(last.values())
    return out


def fig3t():
    import matplotlib.pyplot as plt
    S.apply()
    tiers = _spc_rows()
    pred = _j("v12/pstar_prediction.json")
    fig, axes = plt.subplots(1, 3, figsize=(S.TEXTWIDTH, 2.3), sharey=True,
                             sharex=True)
    csv = []
    for ax, tier in zip(axes, ("<=3.8B", "7-9B", "27-32B")):
        rs = tiers.get(tier, [])
        ax.set_title(tier, fontsize=8)
        S.zeroline(ax)
        ax.set_xscale("log")
        ax.set_xlim(0.7, 3.5e-4)            # dense-adjacent -> past 99.95%
        if not rs:
            ax.text(0.5, 0.5, "v12big not yet\nmeasured", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color=S.DARKGREY)
            continue
        cm = collections.defaultdict(list)
        for r in rs:
            v = r["removal"]
            cm[(r["target"], r["axis"], r["variant"])].append(
                0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v)
        cm = {k: sum(v) / len(v) for k, v in cm.items()}
        cells = sorted({(t, a) for t, a, _v in cm})
        levels = sorted({v for _t, _a, v in cm if v.startswith("s") and v != "s0.0"},
                        key=lambda v: float(v[1:]))
        xs, pts, lo, hi = [], [], [], []
        for vn in levels:
            pr = [(cm[(t, a, vn)], cm[(t, a, "s0.0")]) for t, a in cells
                  if (t, a, vn) in cm and (t, a, "s0.0") in cm]
            if not pr:
                continue
            b = V.boot_paired(pr)
            p = 1 - float(vn[1:])
            xs.append(p); pts.append(b["point"]); lo.append(b["lo"]); hi.append(b["hi"])
            csv.append([tier, vn, p, b["point"], b["lo"], b["hi"], b["n"]])
            for t, a in cells:
                if (t, a, vn) in cm:
                    ax.scatter(p, cm[(t, a, vn)] - cm[(t, a, "s0.0")], s=6,
                               color=S.GREY, alpha=0.6, lw=0, zorder=2)
        ax.fill_between(xs, lo, hi, color=S.COND["C-ref"], alpha=0.18, lw=0)
        ax.plot(xs, pts, color=S.COND["C-ref"], lw=2, marker="o", ms=4, zorder=3)
        ax.text(0.03, 0.04, f"n = {len(cells)} cells", transform=ax.transAxes,
                fontsize=6.5, color=S.DARKGREY)
        if tier == "27-32B" and pred:
            for c, pc in pred["cells"].items():
                ax.axvline(pc["p_star"], color=S.DARKGREY, lw=0.8, ls="--", zorder=1)
                csv.append([tier, f"p*:{c}", pc["p_star"], None, None, None, None])
    axes[0].set_ylabel("removal − dense one-bit")
    for ax in axes:
        ax.set_xlabel("retained fraction p", fontsize=7)
    S.save(fig, "fig3t")
    V.write_csv(os.path.join(V.FIGDIR, "fig3t_data.csv"),
                ["tier", "variant", "p_retained", "delta_vs_dense", "ci_lo", "ci_hi", "n"],
                csv)
    print(f"fig3t: tiers with data {[t for t in ('<=3.8B', '7-9B', '27-32B') if tiers.get(t)]}")


def main():
    V.ensure_dirs()
    fig2b()
    fig4c()
    fig3t()


if __name__ == "__main__":
    main()
