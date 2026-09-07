"""Is any bias axis actually FAMILY-SPECIFIC? (tests design.md §3's premise)

design.md §3 defines cross-family as "disjoint pretraining lineage" and the
whole of H2 rests on the implication that a disjoint lineage carries a
*different* bias profile -- that is what would let a cross-family designer see
what a same-family designer cannot. The premise is never tested in the design.

This script tests it directly on the 8-model, 22-axis inventory by decomposing
each axis's variance into a between-family and a within-family (sibling)
component. A genuinely family-specific axis should show
between-family SD >> within-family SD.

Output: results/family_divergence.json + a ranked table.
"""
import os, json
import numpy as np
from common import save_json

INV = os.path.join(os.path.dirname(__file__), "..", "results", "inventory.json")
METRICS = [("bbq", "s_AMB"), ("bbq", "s_DIS"), ("stereoset", "SS"),
           ("stereoset", "ICAT"), ("crows", "crows_pct")]


def main():
    inv = json.load(open(INV))
    fam = {k: v["family"] for k, v in inv.items()}
    rows = []
    for kind, metric in METRICS:
        axes = {}
        for m, rec in inv.items():
            for axis, sc in rec.get(kind, {}).items():
                if sc.get(metric) is not None:
                    axes.setdefault(axis, {})[m] = sc[metric]
        for axis, vals in axes.items():
            if len(vals) < 6:
                continue
            by = {}
            for m, v in vals.items():
                by.setdefault(fam[m], []).append(v)
            fam_means = [float(np.mean(v)) for v in by.values()]
            within = [float(np.std(v, ddof=1)) for v in by.values() if len(v) > 1]
            bsd = float(np.std(fam_means, ddof=1))
            wsd = float(np.mean(within)) if within else None
            rows.append(dict(benchmark=kind, axis=axis, metric=metric,
                             between_family_sd=bsd, within_family_sd=wsd,
                             ratio=(bsd / wsd) if wsd else None,
                             lo=float(min(vals.values())),
                             hi=float(max(vals.values())),
                             family_means={k: float(np.mean(v)) for k, v in by.items()}))
    rows.sort(key=lambda r: -(r["ratio"] if r["ratio"] else -1))
    print(f"{'benchmark/axis(metric)':38s} {'between':>9s} {'within':>9s} {'ratio':>7s}")
    print("-" * 68)
    for r in rows[:15]:
        print(f"{r['benchmark']}/{r['axis']}({r['metric']})"[:38].ljust(38)
              + f" {r['between_family_sd']:9.4f} "
                f"{(r['within_family_sd'] or float('nan')):9.4f} "
                f"{(r['ratio'] or float('nan')):7.2f}")
    save_json(rows, os.path.join(os.path.dirname(__file__), "..", "results",
                                 "family_divergence.json"))
    print("\nwrote results/family_divergence.json")


if __name__ == "__main__":
    main()
