#!/usr/bin/env python3
"""v11 INS — generation-side collateral, and whether the published cells reproduce.

WHY THIS ARM MATTERS OUT OF PROPORTION TO ITS SIZE. Every other collateral
number in this paper is likelihood-only: MMLU by letter log-probs, perplexity on
wikitext. IFEval is the ONLY measurement of what the edit does when the model
actually generates. It carries the whole claim that the edit does not damage
instruction-following, and it rested on 2 targets x 1 axis x 1 seed.

THE REPRODUCTION QUESTION CAME FIRST, AND HAD TO. The lm-eval version behind the
PUBLISHED IFEval numbers is recorded nowhere -- not in PREREGISTRATION.md, not in
the run logs, not in the artifacts (PREREGISTRATION.md §v11.E). So the two
published cells were re-run under the pinned harness before anything was spent
on new ones: if they did not reproduce, the published numbers would be
version-dependent and extending the arm on top of them would be building on
sand.

WHAT THE METRIC IS. lm-eval's `ifeval` task on the first 200 of 541 prompts,
scored as `inst_level_strict_acc,none` -- per-INSTRUCTION strict accuracy, whose
denominator is 318 instructions across those 200 prompts, so one instruction is
0.0031. The reported contrast is sign_only - random_sign: the true sign field
against the matched-norm random-sign null, both at the alpha each selected.

  python3 src/v11_ins_analyze.py        # -> results/v11/ins_v11.json
"""
import argparse
import collections
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v11_panel as P  # noqa: E402

METRIC = "inst_level_strict_acc,none"

# What the manuscript prints for the two published cells, so the re-run is
# checked against them rather than eyeballed.
PUBLISHED = {
    ("qwen7b_it", "occ_gender"): {"sign_minus_random": 0.000, "removal": 0.698},
    ("llama8b_it", "occ_gender"): {"sign_minus_random": -0.022},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="v11ins")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = P._rows(a.panel, "removal.jsonl")
    if not rows:
        raise SystemExit(f"no rows in results/{a.panel}")

    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["target"], r["axis"], r.get("seed"))][r.get("condition")] = r

    out = {"panel": a.panel, "metric": METRIC, "cells": {}, "reproduction": {}}
    for k, d in sorted(by.items()):
        def ife(cond):
            v = (d.get(cond, {}).get("ifeval") or {}).get(METRIC)
            return float(v) if v is not None else None

        s, n, u = ife("sign_only"), ife("random_sign"), ife("unedited")
        rec = {
            "seed": k[2],
            "ifeval_limit": d.get("sign_only", {}).get("ifeval_limit"),
            "removal_sign_only": d.get("sign_only", {}).get("removal"),
            "ifeval_unedited": u,
            "ifeval_sign_only": s,
            "ifeval_random_sign": n,
            "sign_minus_random": (s - n) if (s is not None and n is not None)
                                 else None,
            "sign_minus_unedited": (s - u) if (s is not None and u is not None)
                                   else None,
            "complete": all(c in d for c in
                            ("unedited", "sign_only", "random_sign")),
        }
        out["cells"][f"{k[0]}|{k[1]}"] = rec

        pub = PUBLISHED.get((k[0], k[1]))
        if pub and rec["sign_minus_random"] is not None:
            diff = rec["sign_minus_random"] - pub["sign_minus_random"]
            out["reproduction"][f"{k[0]}|{k[1]}"] = {
                "published": pub["sign_minus_random"],
                "rerun": rec["sign_minus_random"],
                "diff": diff,
                # 318 instructions over the 200 prompts, so one instruction is
                # 0.0031: report the gap in the unit the metric actually moves in.
                "diff_in_instructions": diff * 318,
                "removal_published": pub.get("removal"),
                "removal_rerun": rec["removal_sign_only"],
            }

    dest = a.out or os.path.join(P.RESULTS, "v11", "ins_v11.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)

    print(f"v11 INS — IFEval, metric {METRIC}\n")
    print(f"{'cell':26s} {'unedited':>9s} {'sign':>8s} {'random':>8s} "
          f"{'sign-rand':>10s} {'removal':>9s}")
    for name, c in out["cells"].items():
        f_ = lambda x: f"{x:+.4f}" if x is not None else "     —"  # noqa: E731
        print(f"{name:26s} {f_(c['ifeval_unedited']):>9s} "
              f"{f_(c['ifeval_sign_only']):>8s} {f_(c['ifeval_random_sign']):>8s} "
              f"{f_(c['sign_minus_random']):>10s} "
              f"{f_(c['removal_sign_only']):>9s}"
              + ("" if c["complete"] else "   INCOMPLETE"))

    if out["reproduction"]:
        print("\nreproduction of the published cells "
              "(one instruction = 1/318 = 0.0031):")
        for name, r in out["reproduction"].items():
            print(f"   {name:26s} published {r['published']:+.4f}  "
                  f"re-run {r['rerun']:+.4f}  "
                  f"diff {r['diff']:+.4f} = {r['diff_in_instructions']:+.1f} instructions")
            if r["removal_published"] is not None:
                print(f"   {'':26s} removal published {r['removal_published']:+.4f}  "
                      f"re-run {r['removal_rerun']:+.4f}")

    done = sum(1 for c in out["cells"].values() if c["complete"])
    print(f"\ncells complete: {done}/{len(out['cells'])}")
    print(f"wrote {os.path.relpath(dest, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
