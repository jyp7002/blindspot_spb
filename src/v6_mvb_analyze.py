"""MV-B corpus transplant — cell-level analysis (experiments_v6 §2 MV-B).

THE UNIT MATTERS. Every provisional read during the run resampled the
designer-seed pool, which treats designers and seeds within a target as
independent replicates -- anti-conservative, and exactly the error the v5
methodological note records (SUMMARY_V5 §2: a "significant" crows self-advantage
that evaporated when the target cell was resampled instead). This module uses
the pre-registered unit: the target x axis CELL.

With 3 targets x 2 axes there are only 6 cells (4 for same_large, which exists
only for qwen and gemma -- phi has no larger sibling in the registry). Six cells
is a weak bootstrap and the CIs are correspondingly wide; that weakness is the
honest state of the probe, not something to average away.

Estimands
  MV-B1  same_large - self       does a same-family-large corpus CLOSE the penalty?
  MV-B3  cross_large - cross_small   the quality axis; and whether it tracks the
         penalty across cells (if the gradient is LARGEST where there is no
         penalty, it is designer strength, not the ingredient)

Usage:  python src/v6_mvb_analyze.py [--boot 10000] [--seed 0] [--no-write]
"""
import os, json, argparse, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(REPO, "results")

# x2/x reference penalty (cross - self) per cell, from the frozen v5 bridge panel.
# Used ONLY to order cells by penalty size; not an input to any estimate.
REF_PENALTY = {("phi", "occ_gender"): +0.288, ("phi", "crows_socioeconomic"): +0.002,
               ("qwen", "occ_gender"): +0.051, ("qwen", "crows_socioeconomic"): +0.227,
               ("gemma", "occ_gender"): -0.174, ("gemma", "crows_socioeconomic"): -0.082}
ROLES = ("self", "same_large", "cross_large", "cross_small")


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def boot_ci(vals, rng, n_boot):
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    d = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return float(np.mean(v)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(RES, "v6trace", "mvb", "removal.jsonl"))
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows = [json.loads(ln) for ln in open(a.panel) if ln.strip()]
    print(f"panel: {a.panel}  ({len(rows)} cells-rows)")

    # ---- collapse to one value per (cell, role): mean over designers AND seeds
    per = collections.defaultdict(list)
    for r in rows:
        per[((r["target"], r["axis"]), r["role"])].append(z(r["removal"]))
    cells = sorted({k[0] for k in per}, key=lambda c: -REF_PENALTY.get(c, 0))

    print("\nper-cell mean removal by source role (cells ordered by v5 penalty, high->low)")
    print(f"  {'cell':30s} {'v5 pen':>7s} " + "".join(f"{r:>13s}" for r in ROLES))
    table = []
    for c in cells:
        vals = {r: (float(np.mean(per[(c, r)])) if per[(c, r)] else None) for r in ROLES}
        table.append(dict(cell=f"{c[0]}|{c[1]}", ref_penalty=REF_PENALTY.get(c), **vals))
        line = f"  {c[0] + '|' + c[1]:30s} {REF_PENALTY.get(c, float('nan')):+7.3f} "
        line += "".join((f"{vals[r]:+13.3f}" if vals[r] is not None else f"{'--':>13s}") for r in ROLES)
        print(line)

    # ---- contrasts, bootstrapped over CELLS
    contrasts = [("same_large - self", "same_large", "self"),
                 ("cross_large - self", "cross_large", "self"),
                 ("cross_small - self", "cross_small", "self"),
                 ("cross_large - cross_small", "cross_large", "cross_small")]
    print("\nCELL-LEVEL bootstrap (the pre-registered unit; n = usable cells)")
    out = {}
    for lbl, A, B in contrasts:
        vals, used = [], []
        for c in cells:
            if per[(c, A)] and per[(c, B)]:
                vals.append(float(np.mean(per[(c, A)])) - float(np.mean(per[(c, B)])))
                used.append(f"{c[0]}|{c[1]}")
        m, lo, hi = boot_ci(vals, rng, a.boot)
        excl = "" if (lo < 0 < hi or np.isnan(lo)) else "  *excludes 0"
        print(f"  {lbl:28s} n={len(vals)}  {m:+.4f}  [{lo:+.3f}, {hi:+.3f}]{excl}")
        out[lbl] = dict(mean=m, ci=[lo, hi], n_cells=len(vals), cells=used,
                        per_cell=vals)
    if any(v["n_cells"] < 3 for v in out.values()):
        print("  !! some contrasts rest on <3 cells — those CIs are degenerate")

    # ---- does the quality gradient track the penalty?
    print("\nMV-B3 diagnostic: does the quality gradient (cross_large - cross_small)")
    print("track the v5 penalty across cells? If it is largest where the penalty is")
    print("ABSENT, the gradient is designer strength, not the blind-spot ingredient.")
    xs, ys, lbls = [], [], []
    for c in cells:
        if per[(c, "cross_large")] and per[(c, "cross_small")] and c in REF_PENALTY:
            xs.append(REF_PENALTY[c])
            ys.append(float(np.mean(per[(c, "cross_large")])) - float(np.mean(per[(c, "cross_small")])))
            lbls.append(f"{c[0]}|{c[1]}")
    print(f"  {'cell':30s} {'v5 penalty':>11s} {'quality grad':>13s}")
    for l, x, y in sorted(zip(lbls, xs, ys), key=lambda t: -t[1]):
        print(f"  {l:30s} {x:+11.3f} {y:+13.3f}")
    if len(xs) >= 3:
        r = float(np.corrcoef(xs, ys)[0, 1])
        print(f"\n  Pearson r(penalty, quality gradient) = {r:+.3f}  over n={len(xs)} cells")
        print("  (descriptive only; n is far too small for an inferential claim)")
        out["gradient_vs_penalty_r"] = r

    if not a.no_write:
        os.makedirs(os.path.join(RES, "v6"), exist_ok=True)
        fp = os.path.join(RES, "v6", "mvb_analysis.json")
        json.dump(dict(panel=a.panel, n_rows=len(rows), unit="target x axis cell",
                       note=("cell-level bootstrap; provisional in-run reads used the "
                             "designer-seed pool and are superseded by this file"),
                       per_cell=table, contrasts=out,
                       boot=a.boot, boot_seed=a.seed), open(fp, "w"), indent=2)
        print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
