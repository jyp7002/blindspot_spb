#!/usr/bin/env python3
"""R1 E2 -- rho_k and the first-order account of the alpha* scale ratio.

    python3 src/r1_e2.py            -> results/r1/E2/e2.json

REGISTERED (R1_REBUTTAL_EXPERIMENTS.md, E2; commit 0fe9043). Per (cell, seed),
from the dense contrast vector dW (computed by run_r1.py in every unit, a
function of dW alone):

    A_ref   = sum_t s_t * sum_{i in top-k of t} |dW_i|      (C-ref as built)
    A_a     = sum_t s_t * sum_{i in C-a's support} |dW_i|   (same per-tensor scales)
    rho_eff = A_ref / A_a          first-order removal per unit alpha, C-ref over C-a
    r_alpha = alpha*(C-a) / alpha*(C-ref)                    (registered v12astar values)

Reading, on the ten registered cells (cell value = seed median, as the plan says
for r_alpha; the median is then taken over cells):
    E2-P  median(rho_eff / r_alpha) in [0.75, 1.33]: the alpha* tie is what the
          first-order account predicts.
    E2-N  outside: first-order failure (saturation, interaction between
          coordinates); App. H's "does not claim" paragraph says so.

Reported next to it: the plain concentration ratio rho_k at k = 1% (3.62 for a
Gaussian), kurtosis and top-1% mass share of dW; the predicted removal ratio
rho_eff / r_alpha against the observed R_ref(alpha*) / R_a(alpha*); and
rho_eff^2, r_alpha^2 next to E1's curvature ratio.
"""
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V    # noqa: E402
import v12_astar as A     # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E2")
LO, HI = 0.75, 1.33
GAUSS_RHO_1PCT = 3.62


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def main():
    os.makedirs(OUT, exist_ok=True)
    units = {}
    for src in ("E4", "E5"):                       # E4 first; E5 only fills gaps
        for u in jl(os.path.join(RESULTS, "r1", src, "unit.jsonl")):
            k = (u["target"], u["axis"], u["seed"])
            if u.get("e2") and (src == "E4" or k not in units):
                units[k] = u
    need = [(t, a, s) for t, a in A.CELLS for s in A.SEEDS]
    miss = [k for k in need if k not in units]
    if miss:
        print(f"E2 statistics on disk for {len(need) - len(miss)}/{len(need)} units; "
              "nothing is read before all exist")
        return 1
    bad = [k for k in need if units[k].get("geom_match") is not True]
    if bad:
        raise SystemExit(f"dW fingerprint not reproduced for {bad}: the v12astar alpha* "
                         "values do not belong to these contrast vectors")
    ref = A.load("v12astar")

    per_unit, cells = [], collections.OrderedDict()
    for t, a in A.CELLS:
        rows = []
        for s in A.SEEDS:
            e = units[(t, a, s)]["e2"]
            ad_r = ref[(t, a, s, "s0.99")]["adaptive"]
            ad_a = ref[(t, a, s, "C-a@0.01")]["adaptive"]
            r_alpha = ad_a["alpha_star"] / ad_r["alpha_star"]
            row = dict(cell=f"{t}|{a}", seed=s, rho_eff=e["rho_eff"], rho_k=e["rho_k"],
                       kurtosis=e["kurtosis"], top_mass_share=e["top_mass_share"],
                       alpha_star_ref=ad_r["alpha_star"], alpha_star_a=ad_a["alpha_star"],
                       r_alpha=r_alpha, ratio=e["rho_eff"] / r_alpha,
                       R_ref=A.z(ad_r.get("removal")), R_a=A.z(ad_a.get("removal")))
            rows.append(row)
            per_unit.append(row)
        med = lambda f: float(np.median([r[f] for r in rows]))
        R_ref, R_a = float(np.mean([r["R_ref"] for r in rows])), float(np.mean([r["R_a"] for r in rows]))
        cells[f"{t}|{a}"] = dict(
            rho_eff=med("rho_eff"), rho_k=med("rho_k"), kurtosis=med("kurtosis"),
            top_mass_share=med("top_mass_share"), r_alpha=med("r_alpha"),
            ratio=med("rho_eff") / med("r_alpha"),
            predicted_removal_ratio=med("rho_eff") / med("r_alpha"),
            observed_removal_ratio=(R_ref / R_a if R_a > 0 else None),
            R_ref_at_alpha_star=R_ref, R_a_at_alpha_star=R_a)

    ratios = [c["ratio"] for c in cells.values()]
    m = float(np.median(ratios))
    reading = "E2-P" if LO <= m <= HI else "E2-N"
    obs = [c["observed_removal_ratio"] for c in cells.values()
           if c["observed_removal_ratio"] is not None]
    out = dict(
        script="src/r1_e2.py", registered="R1_REBUTTAL_EXPERIMENTS.md E2 (0fe9043)",
        reading=reading, band=[LO, HI], median_ratio=m,
        ratio_range=[float(min(ratios)), float(max(ratios))],
        cells_in_band=sum(LO <= r <= HI for r in ratios),
        median_ratio_over_30_units=float(np.median([r["ratio"] for r in per_unit])),
        rho_eff=dict(median=float(np.median([c["rho_eff"] for c in cells.values()])),
                     range=[float(min(c["rho_eff"] for c in cells.values())),
                            float(max(c["rho_eff"] for c in cells.values()))]),
        rho_k=dict(median=float(np.median([c["rho_k"] for c in cells.values()])),
                   range=[float(min(c["rho_k"] for c in cells.values())),
                          float(max(c["rho_k"] for c in cells.values()))],
                   gaussian=GAUSS_RHO_1PCT),
        kurtosis_range=[float(min(c["kurtosis"] for c in cells.values())),
                        float(max(c["kurtosis"] for c in cells.values()))],
        top_mass_share_range=[float(min(c["top_mass_share"] for c in cells.values())),
                              float(max(c["top_mass_share"] for c in cells.values()))],
        r_alpha=dict(median=float(np.median([c["r_alpha"] for c in cells.values()])),
                     range=[float(min(c["r_alpha"] for c in cells.values())),
                            float(max(c["r_alpha"] for c in cells.values()))]),
        squares=dict(rho_eff_sq_median=float(np.median([c["rho_eff"] ** 2 for c in cells.values()])),
                     r_alpha_sq_median=float(np.median([c["r_alpha"] ** 2 for c in cells.values()]))),
        observed_removal_ratio_median=(float(np.median(obs)) if obs else None),
        cells=cells, per_unit=per_unit)
    e1 = os.path.join(RESULTS, "r1", "E1", "e1.json")
    if os.path.exists(e1):
        out["e1_json"] = os.path.relpath(e1, REPO)
    with open(os.path.join(OUT, "e2.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("cell                       rho_k  rho_eff  r_alpha  rho_eff/r_alpha  "
          "R_ref/R_a at alpha*   kurtosis  top-1% mass")
    for k, c in cells.items():
        o = c["observed_removal_ratio"]
        print(f"  {k:26s} {c['rho_k']:5.2f}  {c['rho_eff']:6.2f}  {c['r_alpha']:6.2f}  "
              f"{c['ratio']:9.2f}        {('%8.2f' % o) if o is not None else '     n/a'}"
              f"            {c['kurtosis']:6.1f}   {c['top_mass_share']:.3f}")
    print(f"\nrho_k at 1%: median {out['rho_k']['median']:.2f} "
          f"(range {out['rho_k']['range'][0]:.2f}-{out['rho_k']['range'][1]:.2f}; "
          f"Gaussian {GAUSS_RHO_1PCT})")
    print(f"rho_eff: median {out['rho_eff']['median']:.2f}   r_alpha: median "
          f"{out['r_alpha']['median']:.2f}   squares: {out['squares']['rho_eff_sq_median']:.1f} "
          f"vs {out['squares']['r_alpha_sq_median']:.1f}")
    print(f"median(rho_eff / r_alpha) over 10 cells = {m:.3f}  "
          f"(range {min(ratios):.2f}-{max(ratios):.2f}; {out['cells_in_band']}/10 cells in "
          f"[{LO}, {HI}]; over 30 units {out['median_ratio_over_30_units']:.3f})")
    print(f"observed R_ref/R_a at alpha*: median {out['observed_removal_ratio_median']:.2f}")
    print(f"REGISTERED READING: {reading}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'e2.json'), REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
