#!/usr/bin/env python3
"""v11 DEC — the three registered reportings of Delta_selection.

PREREGISTRATION.md §v11.F commits to publishing Delta_selection over three
populations, fixed before 53 of 54 units had run:

  (a) the 10 PUBLISHED cells        the registered estimand, unchanged
  (b) published + in-envelope new   the scale-up result
  (c) ALL cells                     the conservative reading

and to reporting every cell's in/out status with its C-ref value, so a reader
can recompute any of the three. Nothing is chosen after the fact; nothing is
dropped from the record.

THE SCREEN READS C-ref ONLY. A new cell is in-envelope iff its C-ref, pooled
over seeds under the nan->0 rule, is positive with at least one seed clearing
the collateral budget. Delta_selection never enters the screen, so the screen
cannot select on the quantity being estimated.

Resampling is at the CELL unit with a 10,000-sample percentile bootstrap, per
the project's standing rule (App. "Why the resampling unit is the cell"); seeds
within a cell are not independent replicates and pooling them is
anti-conservative.

  python3 src/v11_dec_analyze.py                 # -> results/v11/dec_v11.json
  python3 src/v11_dec_analyze.py --panel v11dec
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

N_BOOT = 10_000


def nan_to_zero(x):
    """The registered scoring rule: a budget FAILURE scores 0.

    Not a clip on a measurement -- v10 caught a bug that seeded
    best-in-budget at 0.0 and thereby floored legitimately negative removals.
    Only nan (no configuration in budget) becomes 0.
    """
    return 0.0 if (x is None or x != x) else float(x)


def load_cells(panel):
    """-> {(target, axis): {condition: cell-mean over seeds}}, and seed counts."""
    rows = P._rows(panel, "removal.jsonl")
    per_seed = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        key = (r["target"], r["axis"])
        per_seed[key][r["condition"]].append(nan_to_zero(r.get("removal")))
    cells, nseeds = {}, {}
    for key, conds in per_seed.items():
        cells[key] = {c: sum(v) / len(v) for c, v in conds.items()}
        nseeds[key] = max(len(v) for v in conds.values())
    return cells, nseeds


def boot(vals, n=N_BOOT, seed=0):
    if not vals:
        return None, None, None
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, dtype=float)
    means = np.array([rng.choice(a, a.size, replace=True).mean()
                      for _ in range(n)])
    return float(a.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v11dec")
    ap.add_argument("--config", default="configs/v11/dec_v11.yaml")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = P.load(os.path.join(REPO, a.config))
    published = {(c["target"], c["axis"]) for c in cfg["cells"]
                 if c.get("published")}
    cells, nseeds = load_cells(a.panel)
    keys = [k for k in cells if "C-a" in cells[k] and "C-ref" in cells[k]]

    def delta(k):
        return cells[k]["C-ref"] - cells[k]["C-a"]

    # ---- §v11.F screen: C-ref only ----
    screen = {k: (cells[k]["C-ref"] > 0) for k in keys}

    groups = {
        "a_published": [k for k in keys if k in published],
        "b_published_plus_in_envelope":
            [k for k in keys if k in published or screen[k]],
        "c_all": list(keys),
    }
    out = {"panel": a.panel, "config": a.config, "n_boot": N_BOOT,
           "resampling_unit": "cell", "groups": {}, "cells": {}}

    for name, ks in groups.items():
        d = [delta(k) for k in ks]
        m, lo, hi = boot(d)
        out["groups"][name] = {
            "n_cells": len(ks), "delta_selection": m, "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": bool(lo is not None and lo > 0),
            "unanimous": sum(1 for x in d if x > 0), "of": len(d),
            "cells": [f"{t}|{ax}" for t, ax in sorted(ks)],
        }

    for k in sorted(keys, key=lambda x: -cells[x]["C-ref"]):
        out["cells"][f"{k[0]}|{k[1]}"] = {
            "published": k in published,
            "in_envelope": bool(screen[k]),
            "seeds": nseeds[k],
            "C_ref": cells[k]["C-ref"], "C_a": cells[k]["C-a"],
            "delta_selection": delta(k),
        }

    # Descriptive, NOT a selection rule: how much of the pooled shift is simply
    # weak cells having little effect to decompose?
    cref = np.array([cells[k]["C-ref"] for k in keys])
    dd = np.array([delta(k) for k in keys])
    out["corr_Cref_delta"] = float(np.corrcoef(cref, dd)[0, 1]) \
        if len(keys) > 2 else None

    fam = {"occ_gender": "templated", "crows_socioeconomic": "CrowS",
           "ss_intra": "StereoSet"}
    out["families"] = dict(collections.Counter(
        fam.get(ax, "BBQ") for _, ax in keys))

    dest = a.out or os.path.join(P.RESULTS, "v11", "dec_v11.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)

    # ---------------------------------------------------------------- print --
    print(f"v11 DEC — Delta_selection, cell unit, {N_BOOT:,} bootstrap\n")
    label = {"a_published": "(a) published 10 cells  [registered estimand]",
             "b_published_plus_in_envelope": "(b) published + in-envelope new",
             "c_all": "(c) ALL cells"}
    for name, g in out["groups"].items():
        print(f"{label[name]}")
        print(f"    n={g['n_cells']:2d}  {g['delta_selection']:+.4f} "
              f"[{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]  "
              f"{'EXCLUDES 0' if g['excludes_zero'] else 'covers 0'}  "
              f"unanimity {g['unanimous']}/{g['of']}")
    print(f"\ncorr(C-ref, Delta) = {out['corr_Cref_delta']:+.3f}  "
          "(descriptive: weak cells have little effect to decompose)")
    print(f"benchmark families: {out['families']} -> {len(out['families'])}\n")
    print(f"{'cell':30s} {'C-ref':>9s} {'C-a':>9s} {'Delta':>9s}  new? env?")
    for name, c in out["cells"].items():
        print(f"{name:30s} {c['C_ref']:>+9.4f} {c['C_a']:>+9.4f} "
              f"{c['delta_selection']:>+9.4f}  "
              f"{'   ' if c['published'] else 'NEW'}  "
              f"{'in ' if c['in_envelope'] else 'OUT'}")
    print(f"\nwrote {os.path.relpath(dest, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
