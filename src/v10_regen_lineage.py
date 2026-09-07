"""v10 - the H2 lineage interaction, the one number sec 1 cites from the
retired science line.

RESULTS_METHOD.md sec 10 defines it as

    I = [dBias(acquired, same) - dBias(acquired, cross)]
      - [dBias(inherited, same) - dBias(inherited, cross)]

on the FROZEN protocol only (attn q,k,v,o r16 -> the `Tq+k+v+o` tag inside
`key`; method_evidence.py's module-set discipline), endogenous signal, main
binary per-tensor variant, best-in-budget per (cell, factor, seed, designer)
with the project-wide nan->0 rule, and the experiments_v7 resampling unit:
the target x axis CELL.

A cell enters only if it carries BOTH a same-family and a cross-family
designer, which is what makes the within-cell contrast paired. That
restriction is not a choice made here: it is the only filter set that
reproduces the design sec 10 reports, 33 inherited cells and 5 acquired.

Gate: the cell SET is identical under both gates; only which alpha is in
budget moves (56 of 1524 frozen-protocol rows flip `collateral_ok` between
results/runs.jsonl and results_v9/runs.jsonl), and it moves the estimate
across zero. Both are emitted; the manuscript cites the v9 one.

Usage:  python3 src/v10_lineage.py
"""
import os, sys, json, random, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v10_common as C

FROZEN = "Tq+k+v+o"
MAIN_VARIANT = "binary-per_tensor-s0.0"


def _mset(key):
    """method_evidence.mset - the frozen protocol is tagged inside `key`."""
    t = [p for p in str(key).split("|") if p.startswith("T")]
    return t[0] if t else "untagged"


def _factor(r):
    """analyze._family_factor, restricted to the two levels sec 10 contrasts."""
    role = str(r.get("designer_role"))
    if role == "random":
        return "null"
    if str(r.get("signal")) == "exogenous" or role == "gt":
        return "exogenous"
    if role in ("self", "sibling"):
        return "same"
    if role in ("cross", "cross2"):
        return "cross"
    rel = str(r.get("family_relation"))
    return rel if rel in ("same", "cross") else "n/a"


def _load(path):
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("signal") != "endogenous":
                continue
            if r.get("variant") != MAIN_VARIANT:
                continue
            if _mset(r.get("key")) != FROZEN:
                continue
            f = _factor(r)
            if f not in ("same", "cross"):
                continue
            r["_f"] = f
            out.append(r)
    return out


def _cells(rows):
    """(target, axis) x factor -> mean over (seed, designer) of the best
    in-budget removal, budget-fail/nan -> 0.0 (v10_common.z)."""
    sweep = collections.defaultdict(list)
    for r in rows:
        sweep[(r["target"], r["axis"], r["_f"], r["seed"], r["designer"])].append(r)
    best = {}
    for k, v in sweep.items():
        ok = [C.z(x.get("bias_reduction")) for x in v if x.get("collateral_ok")]
        best[k] = max(ok) if ok else 0.0
    agg = collections.defaultdict(list)
    for (t, a, f, _s, _d), val in best.items():
        agg[((t, a), f)].append(val)
    cf = {k: float(np.mean(v)) for k, v in agg.items()}
    origin = {}
    for r in rows:
        origin.setdefault((r["target"], r["axis"]), r["origin"])
    both = sorted(c for c in {c for (c, _f) in cf}
                  if (c, "same") in cf and (c, "cross") in cf)
    return cf, origin, both


def boot_two_group(a, b, n=C.BOOT_N, seed=C.BOOT_SEED):
    """mean(a) - mean(b), resampling each group's CELLS independently.

    The interaction spans two DISJOINT cell sets (5 acquired, 33 inherited),
    so it is not a paired contrast and boot_paired does not apply to it. The
    RNG discipline is boot_paired's, verbatim: random.Random(seed),
    rng.randrange, 10k draws, the same percentile indices. boot_paired itself
    IS used, unchanged, for the two within-origin same-vs-cross gaps, which
    are paired by cell."""
    if not a or not b:
        return None
    rng = random.Random(seed)
    bs = []
    for _ in range(n):
        ma = np.mean([a[rng.randrange(len(a))] for _ in a])
        mb = np.mean([b[rng.randrange(len(b))] for _ in b])
        bs.append(float(ma - mb))
    bs.sort()
    return dict(point=float(np.mean(a) - np.mean(b)),
                lo=bs[int(.025 * n)], hi=bs[int(.975 * n) - 1],
                n=len(a) + len(b), n_acquired=len(a), n_inherited=len(b))


def analyse(path):
    rows = _load(path)
    cf, origin, both = _cells(rows)
    by = {o: [c for c in both if origin[c] == o]
          for o in ("inherited", "acquired")}
    pairs = {o: [(cf[(c, "same")], cf[(c, "cross")]) for c in by[o]] for o in by}
    gaps = {o: [s - x for s, x in pairs[o]] for o in by}
    return {
        "interaction": boot_two_group(gaps["acquired"], gaps["inherited"]),
        "same_minus_cross": {o: C.boot_paired(pairs[o]) for o in by},
        "cell_means": {o: {"same": float(np.mean([p[0] for p in pairs[o]])),
                           "cross": float(np.mean([p[1] for p in pairs[o]]))}
                       for o in by},
        "n_cells": {o: len(by[o]) for o in by},
        "cells": {o: ["|".join(c) for c in by[o]] for o in by},
        "n_rows": len(rows),
        "n_designers": len({r["designer"] for r in rows}),
        "n_seeds": len({r["seed"] for r in rows}),
    }


def main():
    C.ensure_dirs()
    out = {
        "_estimand": "[same-cross | acquired] - [same-cross | inherited]; "
                     "H2 predicts > 0",
        "_population": "frozen attn q,k,v,o r16 protocol, endogenous signal, "
                       "binary-per_tensor-s0.0, integer-item collateral "
                       "budget, 3 seeds, 18 designers, 9 targets, 25 axes",
        "_resampling_unit": "target x axis cell (33 inherited, 5 acquired), "
                            "10k bootstrap, seed 0",
        "v9": analyse(os.path.join(C.V9, "runs.jsonl")),
        "as_run": analyse(os.path.join(C.RES, "runs.jsonl")),
    }
    assert out["v9"]["n_cells"] == {"inherited": 33, "acquired": 5}, \
        "design drifted from RESULTS_METHOD sec 10 (33 inherited / 5 acquired)"
    fp = C.jdump(out, os.path.join(C.OUT_V10, "lineage_h2.json"))
    for g in ("v9", "as_run"):
        print(g, C.fmt(out[g]["interaction"], 3), out[g]["n_cells"])
    print("wrote", fp)


if __name__ == "__main__":
    main()
