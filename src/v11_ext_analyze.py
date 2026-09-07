#!/usr/bin/env python3
"""v11 ALPHAEXT — does the edit-scale extension saturate, and where?

PREREGISTRATION.md §v11.C fixed the criterion BEFORE the run:

  SATURATED  <=>  C-a's in-budget count reaches 0, matching C-ref, which was
                  already 0/30 at alpha=128.

and committed to reporting three things:
  1. the alpha at which C-a's in-budget count reaches 0;
  2. Delta_selection at the deepest alpha where BOTH arms still have an
     in-budget run;
  3. whether the depth series has flattened, and by how much per doubling.

WHY IT MATTERS. Section 5.3 currently has to say the measured gap is an UPPER
BOUND on the unbounded-extension gap, because the series was still falling at
the top of the probed grid. If the extension terminates -- if no configuration
above some alpha is admissible under the collateral budget -- then the value at
the deepest admissible alpha is not an upper bound that might keep falling. It
is the answer.

THE PANELS ARE JOINED, NOT RE-RUN. run_alpha_ext refuses any alpha inside the
frozen grid, so v11ext probes only its own new alphas. The depth series is
assembled across three panels:

    v8dec   alpha in {2,4,8,16}    the frozen, registered grid
    v10ext  alpha in {32,64,128}   the v10 post-hoc extension
    v11ext  alpha in {256,512}     this run

The chain is anchored by v11dec having reproduced v8dec bit-identically, so the
frozen arm these extensions are measured against is verified, not assumed.

  python3 src/v11_ext_analyze.py      # -> results/v11/ext_v11.json
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v11_panel as P  # noqa: E402
import v9_gate  # noqa: E402

PANELS = ["v8dec", "v10ext", "v11ext"]
ARMS = ("C-ref", "C-a")
N_BOOT = 10_000


def regate(rows):
    """Re-decide the collateral gate from the stored measurements.

    THE STORED FLAG CANNOT BE TRUSTED ACROSS PANELS. results/<panel> holds the
    AS-RUN tree, whose `collateral_ok` was written by the old float gate --
    the one that rejected configurations sitting exactly on budget. Reading it
    mixes two gates in one series: on the published DEC cells the as-run flags
    give Delta_selection +0.2537 where the corrected gate gives +0.2835, and
    that difference is the v9 correction, not a measurement.

    v9_gate.collateral_ok_row re-decides from pre/post MMLU (integer items) and
    the perplexity ratio, so every alpha in the series is gated the same way
    regardless of which tree its row came from. Rows the gate cannot decide are
    dropped and counted rather than assumed.
    """
    kept, undecidable, modes = [], 0, collections.Counter()
    for r in rows:
        ok, how = v9_gate.collateral_ok_row(r, r.get("n_items", 200))
        if ok is None:
            undecidable += 1
            continue
        r = dict(r)
        r["collateral_ok"] = bool(ok)
        modes[how] += 1
        kept.append(r)
    return kept, undecidable, modes


def trace(panel):
    """Deduped alpha_trace rows.

    alpha_trace.jsonl is append-only and a resumed cell is re-traced, so
    (target, axis, seed, designer, condition, alpha) is NOT unique. The file's
    own docstring says to keep the LAST occurrence -- the completed attempt.
    """
    seen = {}
    for r in P._rows(panel, "alpha_trace.jsonl"):
        if r.get("condition") not in ARMS:
            continue
        key = (r.get("target"), r.get("axis"), r.get("seed"),
               r.get("designer"), r["condition"], float(r["alpha"]))
        seen[key] = r
    return list(seen.values())


def boot(vals, n=N_BOOT, seed=0):
    if not vals:
        return None, None, None
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    m = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(n)])
    return float(a.mean()), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    found = {}
    for p in PANELS:
        t = trace(p)
        found[p] = len(t)
        for r in t:
            r["_panel"] = p
        rows += t
    if not rows:
        raise SystemExit("no alpha_trace rows found in " + ", ".join(PANELS))
    rows, undecidable, modes = regate(rows)
    print(f"re-gated {len(rows)} rows with v9_gate "
          f"({dict(modes)}); {undecidable} undecidable and dropped\n")

    out = {"panels": found, "n_boot": N_BOOT, "by_alpha": {},
           "gate": "v9_gate.collateral_ok_row, recomputed (stored flags are as-run)",
           "criterion": "saturated when C-a in-budget count reaches 0 "
                        "(PREREGISTRATION.md v11.C)"}

    alphas = sorted({float(r["alpha"]) for r in rows})
    for al in alphas:
        e = {}
        for arm in ARMS:
            s = [r for r in rows if float(r["alpha"]) == al
                 and r["condition"] == arm]
            ok = [r for r in s if r.get("collateral_ok")]
            e[arm] = {
                "probed": len(s), "in_budget": len(ok),
                "mean_removal_in_budget":
                    (sum(r["bias_reduction"] for r in ok) / len(ok))
                    if ok else None,
                "median_mmlu_items_dropped":
                    float(np.median([abs(r["dmmlu"]) * r.get("n_items", 200)
                                     for r in s])) if s else None,
                "median_ppl_ratio":
                    float(np.median([r["ppl_ratio"] for r in s])) if s else None,
            }
        out["by_alpha"][str(int(al))] = e

    # (1) where does C-a stop clearing the budget?
    sat = None
    for al in alphas:
        e = out["by_alpha"][str(int(al))]["C-a"]
        if e["probed"] and e["in_budget"] == 0:
            sat = int(al)
            break
    out["saturating_alpha"] = sat

    # (2) deepest alpha where BOTH arms still have an in-budget run
    deepest = None
    for al in alphas:
        e = out["by_alpha"][str(int(al))]
        if e["C-ref"]["in_budget"] > 0 and e["C-a"]["in_budget"] > 0:
            deepest = int(al)
    out["deepest_both_in_budget"] = deepest

    # (3) the depth series: best-in-budget per cell up to each alpha cap.
    # This is the estimand run_dec uses -- max removal over admissible alphas,
    # nan when none is admissible -- recomputed at successive caps.
    # RESAMPLE AT THE CELL, NOT THE SEED. Seeds within a target x axis cell are
    # not independent replicates; pooling them shrinks intervals
    # anti-conservatively, and doing so anyway is exactly the defect v10 caught
    # in method_evidence.py (paper §5.9 defect vi). Best-in-budget is computed
    # per SEED -- that is the estimand run_dec uses -- and the per-seed values
    # are then averaged into a cell before the bootstrap resamples cells.
    seeds = sorted({(r["target"], r["axis"], r["seed"]) for r in rows})
    series = {}
    for cap in alphas:
        per_cell = collections.defaultdict(list)
        for c in seeds:
            best = {}
            for arm in ARMS:
                v = [r["bias_reduction"] for r in rows
                     if (r["target"], r["axis"], r["seed"]) == c
                     and r["condition"] == arm and float(r["alpha"]) <= cap
                     and r.get("collateral_ok")]
                best[arm] = max(v) if v else float("nan")
            if best["C-ref"] == best["C-ref"] and best["C-a"] == best["C-a"]:
                per_cell[c[:2]].append(best["C-ref"] - best["C-a"])
        d = [sum(v) / len(v) for v in per_cell.values()]
        if d:
            m, lo, hi = boot(d)
            series[str(int(cap))] = {"n_cells": len(d), "delta": m,
                                     "ci_lo": lo, "ci_hi": hi,
                                     "excludes_zero": lo > 0,
                                     "unit": "cell"}
    out["depth_series"] = series

    dest = a.out or os.path.join(P.RESULTS, "v11", "ext_v11.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------------------------------------- report --
    print("v11 ALPHAEXT — saturation of the edit-scale extension\n")
    print("panels joined: " + ", ".join(f"{k}({v} rows)" for k, v in found.items()))
    print(f"\n{'alpha':>6s} {'C-ref in-budget':>16s} {'C-a in-budget':>14s} "
          f"{'median MMLU drop':>17s} {'median ppl':>11s}")
    for al in alphas:
        e = out["by_alpha"][str(int(al))]
        print(f"{int(al):>6d} {e['C-ref']['in_budget']:>7d}/"
              f"{e['C-ref']['probed']:<8d} {e['C-a']['in_budget']:>6d}/"
              f"{e['C-a']['probed']:<7d} "
              f"{e['C-a']['median_mmlu_items_dropped'] or 0:>13.0f} items "
              f"{e['C-a']['median_ppl_ratio'] or 0:>10.2f}")
    print("   (budget: <=4 MMLU items of 200, ppl ratio <= 1.10)")

    print(f"\ndepth series — Delta_selection at each alpha cap:")
    for cap, s in series.items():
        print(f"   alpha <= {cap:>4s}   n={s['n_cells']:2d} cells  {s['delta']:+.4f} "
              f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  "
              f"{'excludes 0' if s['excludes_zero'] else 'covers 0'}")

    print(f"\nC-a stops clearing the budget at alpha = {sat}")
    print(f"deepest alpha with BOTH arms in budget = {deepest}")
    if sat is not None:
        print("\nSATURATED per the registered criterion. The extension "
              "terminates: beyond the deepest admissible alpha no configuration "
              "is in budget, so the gap there is not an upper bound that might "
              "keep falling.")
    else:
        print("\nNOT saturated at the probed depth — report the value as an "
              "upper bound and do NOT extrapolate (v11.C).")
    print(f"\nwrote {os.path.relpath(dest, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
