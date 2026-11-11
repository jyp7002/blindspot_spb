#!/usr/bin/env python3
"""R1 -- what the occupation probe's removal consists of (post hoc, descriptive).

    python3 src/r1_pronoun.py        # -> results/r1/E5/pronoun_check.json

The occupation skew is a gap: P(she) on female-stereotyped minus P(she) on
male-stereotyped occupations, per template. A method can close that gap by
making the pronoun independent of the occupation at an unchanged base rate, or
by pushing every occupation to one pronoun (collapse). E7b showed that DPO's
large occupation removals are the second kind. This script asks the same
question of every 1% edit of the E5 panel, from the per-item log-likelihoods
stored with each rung (probe_ll/*.npz), at each arm's own alpha*, and for the
two arms that have those rungs also at alpha*/2 and at the scale the fixed grid
{2,4,8,16} selects (the operating point of the paper's main tables).

Reported per arm, over the five occupation cells x three seeds:
  f, m      mean P(she) on female- / male-stereotyped probe occupations
  gap       f - m          (what the skew measures)
  base      (f + m) / 2    (the pronoun base rate; 0.5 = balanced)
  collapsed runs: both f and m above 0.95, or both below 0.05

Not a registered reading. It reads a complete panel and changes no verdict.
"""
import ast
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v12_astar as A     # noqa: E402
import r1_e5 as E5        # noqa: E402

OCC_CELLS = [c for c in A.CELLS if c[1] == "occ_gender"]


def probe_occ():
    src = open(os.path.join(REPO, "colab_t2t4.py"), encoding="utf-8").read()
    occ = ast.literal_eval(re.search(r"^PROBE_OCC = (.*)$", src, re.M).group(1))
    tpl = ast.literal_eval(re.search(r"^PROBE_TEMPLATES = (.*)$", src, re.M).group(1))
    return list(occ.items()), len(tpl)


def pshe(ll, occ, n_tpl):
    d = ll[1::2] - ll[0::2]                        # she - he, per (occupation, template)
    p = 1.0 / (1.0 + np.exp(-d.astype(np.float64)))
    p = p.reshape(len(occ), n_tpl).mean(1)
    g = np.array([x[1] for x in occ])
    return float(p[g == "f"].mean()), float(p[g == "m"].mean())


def at(zf, alpha, occ, n_tpl):
    i = [j for j, a in enumerate(zf["alphas"]) if float(a) == float(alpha)]
    if len(i) != 1:
        raise SystemExit(f"alpha {alpha} not stored exactly once")
    return pshe(zf["ll"][i[0]], occ, n_tpl)


def main():
    rows, _ = E5.load()
    if any(k not in rows for k in E5.need()):
        print("incomplete: the E5 panel is not read before every unit-arm exists")
        return 1
    occ, n_tpl = probe_occ()
    out = dict(script="src/r1_pronoun.py", status="post hoc, descriptive",
               n_female=sum(g == "f" for _, g in occ), n_male=sum(g == "m" for _, g in occ),
               n_templates=n_tpl, arms={}, cells={})
    for arm in E5.ARMS:
        runs = []
        for t, a in OCC_CELLS:
            for s in A.SEEDS:
                r = rows[(t, a, s, arm)]
                a_star = r["adaptive"]["alpha_star"]
                if a_star is None:
                    continue
                fp = os.path.join(E5.OUT, "probe_ll",
                                  f"{t}__{a}__s{s}__{r['designer']}__{arm}.npz")
                zf = np.load(fp)
                pre = pshe(zf["pre_ll"], occ, n_tpl)
                post = at(zf, a_star, occ, n_tpl)
                d = dict(target=t, seed=s, alpha_star=a_star, pre_f=pre[0], pre_m=pre[1],
                         post_f=post[0], post_m=post[1],
                         collapsed=bool((post[0] > 0.95 and post[1] > 0.95)
                                        or (post[0] < 0.05 and post[1] < 0.05)))
                if r.get("backoff"):
                    half = [b["alpha"] for b in r["backoff"] if abs(b["factor"] - 0.5) < 1e-9]
                    h = at(zf, half[0], occ, n_tpl)
                    d.update(half_f=h[0], half_m=h[1])
                fr = r.get("frozen") or {}
                if fr.get("alpha") is not None:
                    gq = at(zf, fr["alpha"], occ, n_tpl)
                    d.update(grid_f=gq[0], grid_m=gq[1], grid_alpha=fr["alpha"],
                             grid_collapsed=bool((gq[0] > 0.95 and gq[1] > 0.95)
                                                 or (gq[0] < 0.05 and gq[1] < 0.05)))
                runs.append(d)

        def m(k):
            v = [x[k] for x in runs if k in x]
            return float(np.mean(v)) if v else None
        summ = dict(n_runs=len(runs), pre_f=m("pre_f"), pre_m=m("pre_m"),
                    post_f=m("post_f"), post_m=m("post_m"),
                    half_f=m("half_f"), half_m=m("half_m"),
                    grid_f=m("grid_f"), grid_m=m("grid_m"),
                    grid_n=sum("grid_f" in x for x in runs),
                    grid_collapsed_runs=sum(x.get("grid_collapsed", False) for x in runs),
                    collapsed_runs=sum(x["collapsed"] for x in runs))
        summ["pre_gap"] = summ["pre_f"] - summ["pre_m"]
        summ["post_gap"] = summ["post_f"] - summ["post_m"]
        summ["pre_base"] = (summ["pre_f"] + summ["pre_m"]) / 2
        summ["post_base"] = (summ["post_f"] + summ["post_m"]) / 2
        if summ["half_f"] is not None:
            summ["half_gap"] = summ["half_f"] - summ["half_m"]
            summ["half_base"] = (summ["half_f"] + summ["half_m"]) / 2
        out["arms"][arm] = summ
        by_cell = {}
        for t, _a in OCC_CELLS:
            rs = [x for x in runs if x["target"] == t]
            by_cell[t] = {k: float(np.mean([x[k] for x in rs]))
                          for k in ("pre_f", "pre_m", "post_f", "post_m", "half_f", "half_m",
                                    "grid_f", "grid_m", "grid_alpha")
                          if all(k in x for x in rs)}
            by_cell[t]["collapsed_runs"] = sum(x["collapsed"] for x in rs)
        out["cells"][arm] = by_cell
    with open(os.path.join(E5.OUT, "pronoun_check.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"occupation probe: {out['n_female']} female- and {out['n_male']} male-stereotyped "
          f"occupations x {n_tpl} templates; five cells x three seeds; each arm at its own alpha*")
    print(f"  {'arm':10s} {'P(she) f / m  before':22s} {'at alpha*':14s} "
          f"{'gap':16s} {'base rate':16s} collapsed")
    for arm, s in out["arms"].items():
        print(f"  {arm:10s} {s['pre_f']:.2f} / {s['pre_m']:.2f}            "
              f"{s['post_f']:.2f} / {s['post_m']:.2f}    "
              f"{s['pre_gap']:.2f} -> {s['post_gap']:.2f}    "
              f"{s['pre_base']:.2f} -> {s['post_base']:.2f}    "
              f"{s['collapsed_runs']}/{s['n_runs']}")
        if s["half_f"] is not None:
            print(f"  {'  at alpha*/2':10s}                        "
                  f"{s['half_f']:.2f} / {s['half_m']:.2f}    "
                  f"{s['pre_gap']:.2f} -> {s['half_gap']:.2f}    "
                  f"{s['pre_base']:.2f} -> {s['half_base']:.2f}")
        if s["grid_f"] is not None:
            print(f"  {'  fixed grid':10s}                        "
                  f"{s['grid_f']:.2f} / {s['grid_m']:.2f}    "
                  f"{s['pre_gap']:.2f} -> {s['grid_f'] - s['grid_m']:.2f}    "
                  f"{s['pre_base']:.2f} -> {(s['grid_f'] + s['grid_m']) / 2:.2f}    "
                  f"{s['grid_collapsed_runs']}/{s['grid_n']}")
    print("\n  by cell (seed mean), P(she) f / m: before -> at alpha*  [-> at alpha*/2]")
    for arm, cells in out["cells"].items():
        for t, c in cells.items():
            extra = (f"   [alpha*/2: {c['half_f']:.2f} / {c['half_m']:.2f}]"
                     if "half_f" in c else "")
            if "grid_f" in c:
                extra += (f"   [grid, mean alpha {c['grid_alpha']:.1f}: "
                          f"{c['grid_f']:.2f} / {c['grid_m']:.2f}]")
            print(f"    {arm:10s} {t:7s} {c['pre_f']:.2f} / {c['pre_m']:.2f} -> "
                  f"{c['post_f']:.2f} / {c['post_m']:.2f}{extra}   collapsed "
                  f"{c['collapsed_runs']}/3")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
