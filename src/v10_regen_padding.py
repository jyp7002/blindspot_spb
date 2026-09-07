"""v10 — the padding-bug magnitudes, recomputed from the quarantined panel.

§5.9 quotes two numbers from the `padding_side` defect: how much it inflated
ΔMMLU under steering, and the false edit-over-steering advantage it produced.
Both were printed as prose with no artifact behind them, and neither reproduced
as written (V10_FINDINGS: the printed 0.215 matches no summary of the panel;
the printed +0.292 was computed with as-run edit values).

The rewritten §5.9 quotes the RECOMPUTED values instead. Those have to come
from somewhere a manifest entry can resolve — registering them as literals
would repeat exactly the defect that same paragraph retracts (hardcoded values
in `method_evidence.py`). This module computes them.

Both arms are paired configuration-for-configuration against the clean steer
panel on (target, axis, seed, layer, strength, kind, grid); the quarantined
panel is `v6trace/steer_paddingbug_1785685027`.

Run: python3 src/v10_regen_padding.py
"""
import os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C

BUG = "v6trace/steer_paddingbug_1785685027"
CLEAN = "v6trace/steer"
KEY = ("target", "axis", "seed", "layer", "strength", "kind", "grid")


def _index(panel):
    rs = C.rows(C.V9, panel) or C.rows(C.RES, panel)
    return {tuple(r.get(k) for k in KEY): r for r in rs}


def dmmlu_inflation():
    bug, clean = _index(BUG), _index(CLEAN)
    ks = sorted(set(bug) & set(clean), key=str)
    out = {"_what": "paired ΔMMLU inflation, buggy panel minus clean panel",
           "_pairing": "matched on " + ", ".join(KEY),
           "n_matched_configs": len(ks)}
    for arm in ("real", "all"):
        sub = [k for k in ks if arm == "all" or k[5] == arm]
        d = [bug[k]["dmmlu"] - clean[k]["dmmlu"] for k in sub
             if bug[k].get("dmmlu") is not None and clean[k].get("dmmlu") is not None]
        if not d:
            continue
        out["real_arm" if arm == "real" else "all_arms"] = {
            "mean": float(np.mean(d)), "median": float(np.median(d)),
            "max": float(np.max(d)), "min": float(np.min(d)), "n_pairs": len(d)}
    return out


def false_claim():
    """edit − steering computed against the QUARANTINED steering panel: the
    comparison that produced the retracted advantage."""
    front = C.jload("results/v10/frontier_v9.json") or {}
    edit = {c["cell"]: c["methods"]["edit"]["v9"] for c in (front.get("cells") or [])}
    bug = _index(BUG)
    g = collections.defaultdict(list)
    for k, r in bug.items():
        if k[5] != "real" or not r.get("collateral_ok"):
            continue
        g[(k[0], k[1], k[2])].append(r.get("bias_reduction"))
    best = {k: (max(v) if v else 0.0) for k, v in g.items()}
    cells = collections.defaultdict(list)
    for (t, a, _s), v in best.items():
        cells[f"{t}|{a}"].append(v)
    # nan -> 0 over the frontier's own cell set, as the original comparison did
    steer = {k: (float(np.mean(cells[k])) if k in cells else 0.0) for k in edit}
    pairs = [(edit[k], steer[k]) for k in sorted(edit)]
    d = C.boot_paired(pairs)
    return {
        "_what": ("edit − steering against the quarantined padding-bug panel, "
                  "the comparison that produced the retracted advantage"),
        "_note": ("the edit arm here is the CURRENT v9 arm, so this is the "
                  "closest reconstruction rather than the original figure, "
                  "which used as-run edit values"),
        "edit_minus_steering": (dict(d, text=C.fmt(d), status=C.status(d))
                                if d else None),
        "n_cells": len(pairs),
        "steering_cells_with_any_in_budget_config": len(cells),
    }


def main():
    C.ensure_dirs()
    rep = {"artifact": "v10_regen_padding", "module": "src/v10_regen_padding.py",
           "panel": BUG, "clean_panel": CLEAN,
           "purpose": ("give §5.9's recomputed padding-bug magnitudes a "
                       "resolvable key, so the paragraph that retracts hardcoded "
                       "values does not itself print hardcoded values"),
           "dmmlu_inflation": dmmlu_inflation(),
           "false_claim": false_claim()}
    fp = C.jdump(rep, "results/v10/padding_recompute.json")
    ra = rep["dmmlu_inflation"].get("real_arm", {})
    print("[v10 padding]")
    print(f"  dmmlu inflation (real arm, n={ra.get('n_pairs')}): "
          f"mean {ra.get('mean',0):+.4f}  median {ra.get('median',0):+.4f}  "
          f"max {ra.get('max',0):+.4f}")
    fc = rep["false_claim"]["edit_minus_steering"]
    print(f"  edit − steering vs buggy panel: {fc['text'] if fc else 'n/a'}")
    print(f"wrote {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
