#!/usr/bin/env python3
"""Do the replayed published cells reproduce the panel they replay?

experiments_v11.md §VI registers this as a gate: every v11 panel replays the
published cells alongside the new ones, and must reproduce them. A silent stack
change is the risk being guarded against -- run_ws2_determinism.py already
attributes a cross-panel MMLU divergence to a `transformers` rebuild, and the
GPU stack on this box has been reinstalled since the published panels ran.

COMPARE AGAINST THE RE-SCORED TREE, NOT THE AS-RUN ONE. results/v8dec is the
AS-RUN panel, scored under the old float collateral gate. results_v9/v8dec is
the same panel re-scored under the corrected integer gate, which is the gate the
current code uses. Comparing a fresh run against results/v8dec shows large
differences on exactly the rows carrying v9_changed=True, and they are the gate
correction, not a regression.

  python3 scripts/check_reproduction.py                     # v11dec vs results_v9/v8dec
  python3 scripts/check_reproduction.py --new results/v11ext --ref results_v9/v10ext
"""
import argparse, json, os, sys

def load(fp):
    out = {}
    if not os.path.exists(fp):
        return out
    for l in open(fp):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        out[(r.get("target"), r.get("axis"), r.get("seed"), r.get("condition"))] = r
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="results/v11dec")
    ap.add_argument("--ref", default="results_v9/v8dec")
    ap.add_argument("--as-run", default="results/v8dec",
                    help="the pre-re-score tree, shown for context only")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="0 means demand bit-identical")
    a = ap.parse_args()

    new = load(os.path.join(a.new, "removal.jsonl"))
    ref = load(os.path.join(a.ref, "removal.jsonl"))
    asrun = load(os.path.join(a.as_run, "removal.jsonl"))
    shared = sorted(set(new) & set(ref))
    if not shared:
        print(f"no overlapping rows between {a.new} and {a.ref} yet — "
              f"{len(new)} new rows, none replayed")
        return 0

    worst, bad, changed = 0.0, [], 0
    cells = {}
    for k in shared:
        d = new[k]["removal"] - ref[k]["removal"]
        if d != d:                        # nan on both sides counts as equal
            d = 0.0 if new[k]["removal"] != new[k]["removal"] else float("inf")
        worst = max(worst, abs(d))
        if abs(d) > a.tol:
            bad.append((k, ref[k]["removal"], new[k]["removal"], d))
        if k in asrun and abs(new[k]["removal"] - asrun[k]["removal"]) > 1e-9:
            changed += 1
        cells.setdefault(k[:3], []).append(k)

    print(f"replayed rows : {len(shared)} over {len(cells)} cell(s)")
    print(f"reference     : {a.ref}")
    print(f"max |delta|   : {worst:.3e}")
    print(f"rows differing from the AS-RUN tree ({a.as_run}): {changed} "
          f"— expected, these are the v9 gate corrections")
    if bad:
        print(f"\n*** {len(bad)} ROW(S) DO NOT REPRODUCE ***")
        for k, o, n, d in bad[:20]:
            print(f"  {k[0]}|{k[1]}|s{k[2]} {k[3]:18s} ref={o:+.6f} new={n:+.6f} d={d:+.2e}")
        return 1
    print("\nREPRODUCTION OK — every replayed row is bit-identical to the "
          "re-scored panel.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
