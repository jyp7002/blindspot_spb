#!/usr/bin/env python3
"""R1 E3b -- the fixed-grid results under other capability budgets (zero GPU).

    python3 src/r1_e3b.py            the table
    python3 src/r1_e3b.py selftest

PLAN (R1_REBUTTAL_EXPERIMENTS.md, E3b; itTC Q2). Every configuration of every
fixed-grid experiment was stored with the MMLU items it loses (of 200) and its
perplexity ratio. The budget is therefore re-applied here with an MMLU
allowance b in {2, 4, 6, 8} items and a perplexity bound c in {1.05, 1.10,
1.20}, and with the perplexity bound alone; the registered budget is b = 4,
c = 1.10. For each budget every run re-selects its best admissible
configuration (largest removal; none admissible -> 0) and the estimands are
recomputed. The plan asks for a table and for which estimands change whether
their interval excludes zero; it fixes no verdict.

Scope fixed in R1_NOTES.md section 3 (entry of 2026-11-13):
  A  ten main cells, grid {2,4,8,16}: C-ref, C-a, C-ref - C-a (the selection
     effect), and C-ref minus each perturbed variant; from results/r1/E5 and
     results/r1/E4, whose fixed-grid values replay the published ones exactly.
  B  eight comparison cells, four configurations per method: dense edit,
     steering, SentenceDebias (results/v12frontier), DPO at lower learning rates
     (results/r1/E7b), the 1% edit (results/r1/E5, results/r1/FRONT8); the
     differences of both edits with each other method; share of configurations
     within the budget.
Not covered: the extended grid and alpha* (their budget sensitivity is App. B.1
and B.7), and the sparsity panels.

Sanity gate: at the registered budget the recomputed values equal the stored
selections of every run.
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V    # noqa: E402
import v12_astar as A     # noqa: E402
import r1_e5 as E5        # noqa: E402
import r1_front8 as F8    # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E3")
GRID = (2.0, 4.0, 8.0, 16.0)
BUDGETS = ([(b, c) for c in (1.05, 1.10, 1.20) for b in (2, 4, 6, 8)]
           + [(None, c) for c in (1.05, 1.10, 1.20)])
REG = (4, 1.10)
VARIANTS = ["C-ref", "C-a", "C-rand", "C-b", "C-layershuf", "C-tensorshuf", "C-bottom"]
PANEL = {"C-ref": "E5", "C-a": "E5"}
METHODS = ["edit", "edit_1pct", "steering", "sentdebias", "dpo"]
NAME = dict(edit="dense edit", edit_1pct="1% edit", steering="steering", sentdebias="SentenceDebias",
            dpo="DPO")


def admissible(p, b, c):
    """p: dict(dmmlu_items, ppl_ratio). b None = perplexity bound alone."""
    return (b is None or p["dmmlu_items"] <= b) and p["ppl_ratio"] <= c + 1e-9


def best(points, b, c):
    """points: [dict(removal, dmmlu_items, ppl_ratio)] in configuration order -> removal (0 if none)."""
    out = None
    for p in points:
        if admissible(p, b, c) and (out is None or p["removal"] > out):
            out = p["removal"]
    return 0.0 if out is None else float(out)


def share(points, b, c):
    return sum(admissible(p, b, c) for p in points), len(points)


def load_main():
    """(t, a, s, variant) -> four grid points."""
    rows = {p: E5.load(os.path.join(RESULTS, "r1", p))[0] for p in ("E4", "E5")}
    out = {}
    for v in VARIANTS:
        src = rows[PANEL.get(v, "E4")]
        for t, a in A.CELLS:
            if v == "C-tensorshuf" and t == "phi":
                continue
            for s in A.SEEDS:
                r = src[(t, a, s, v)]
                rung = {float(x["alpha"]): x for x in r["rungs"]}
                pts = [dict(removal=A.z(rung[g]["eval"]["removal"]), dmmlu_items=rung[g]["eval"]["dmmlu_items"],
                            ppl_ratio=rung[g]["eval"]["ppl_ratio"]) for g in GRID]
                if abs(best(pts, *REG) - A.z((r.get("frozen") or {}).get("removal"))) > 1e-9:
                    raise SystemExit(f"registered budget does not reproduce the frozen row: {t}|{a}|s{s}|{v}")
                out[(t, a, s, v)] = pts
    return out


def load_comparison():
    """(t, a, s, method) -> configurations (empty list: method not run in that cell)."""
    out = {}
    stored = {}
    for r in E5.jl(os.path.join(RESULTS, "v12frontier", "removal.jsonl")):
        if r.get("row") == "select":
            stored[(r["target"], r["axis"], r["seed"], r["method"])] = r
    tr = {}
    for r in E5.jl(os.path.join(RESULTS, "v12frontier", "alpha_trace.jsonl")):
        tr.setdefault((r["target"], r["axis"], r["seed"], r["method"]), {})[r["config"]] = dict(
            removal=float(r["bias_reduction"]), dmmlu_items=r["dmmlu_items"], ppl_ratio=r["ppl_ratio"])
    for m in ("edit", "steering", "sentdebias"):
        for t, a in F8.CELLS:
            for s in F8.SEEDS:
                pts = list(tr.get((t, a, s, m), {}).values())
                if abs(best(pts, *REG) - A.z(stored[(t, a, s, m)].get("removal"))) > 1e-9:
                    raise SystemExit(f"registered budget does not reproduce the stored selection: {t}|{a}|s{s}|{m}")
                out[(t, a, s, m)] = pts
    e7 = {}
    for r in E5.jl(os.path.join(RESULTS, "r1", "E7b", "removal.jsonl")):
        e7[(r["target"], r["axis"], r["seed"], r["arm"])] = r
    one = F8.load_runs()
    for t, a in F8.CELLS:
        for s in F8.SEEDS:
            pts = []
            for arm in ("lr1e-5", "lr5e-5"):
                r = e7[(t, a, s, arm)]
                if not r.get("skipped"):
                    pts += [dict(removal=float(p["removal"]), dmmlu_items=p["dmmlu_items"],
                                 ppl_ratio=p["ppl_ratio"]) for p in r["points"]]
            out[(t, a, s, "dpo")] = pts
            r = one[(t, a, s)]
            rung = {float(x["alpha"]): x for x in r["rungs"]}
            out[(t, a, s, "edit_1pct")] = [
                dict(removal=A.z(rung[g]["eval"]["removal"]), dmmlu_items=rung[g]["eval"]["dmmlu_items"],
                     ppl_ratio=rung[g]["eval"]["ppl_ratio"]) for g in GRID]
    return out


def table(main, comp):
    def mcells(v, b, c):
        cs = [x for x in A.CELLS if not (v == "C-tensorshuf" and x[0] == "phi")]
        return [float(np.mean([best(main[(t, a, s, v)], b, c) for s in A.SEEDS])) for t, a in cs]

    def ccells(m, b, c):
        return [float(np.mean([best(comp[(t, a, s, m)], b, c) for s in F8.SEEDS])) for t, a in F8.CELLS]

    out = {}
    for b, c in BUDGETS:
        key = f"mmlu<={b},ppl<={c:.2f}" if b is not None else f"ppl<={c:.2f} only"
        rec = dict(mmlu_items=b, ppl_bound=c, registered=(b, c) == REG, main={}, comparison={})
        ref, ca = mcells("C-ref", b, c), mcells("C-a", b, c)
        rec["main"]["C-ref"] = V.boot_mean(ref)
        rec["main"]["C-a"] = V.boot_mean(ca)
        rec["main"]["C-ref - C-a"] = V.boot_paired(list(zip(ref, ca)))
        for v in ("C-rand", "C-b", "C-layershuf", "C-bottom"):
            rec["main"][f"C-ref - {v}"] = V.boot_paired(list(zip(ref, mcells(v, b, c))))
        for v in ("C-ref", "C-a"):
            n, d = map(sum, zip(*[share(main[(t, a, s, v)], b, c) for t, a in A.CELLS for s in A.SEEDS]))
            rec["main"][f"{v} in budget"] = n / d
        cc = {m: ccells(m, b, c) for m in METHODS}
        for m in METHODS:
            rec["comparison"][NAME[m]] = V.boot_mean(cc[m])
            n, d = map(sum, zip(*[share(comp[(t, a, s, m)], b, c) for t, a in F8.CELLS for s in F8.SEEDS]))
            rec["comparison"][f"{NAME[m]} in budget"] = n / d if d else None
        for x in ("edit", "edit_1pct"):
            for y in ("steering", "sentdebias", "dpo"):
                rec["comparison"][f"{NAME[x]} - {NAME[y]}"] = V.boot_paired(list(zip(cc[x], cc[y])))
        out[key] = rec
    return out


def excl(d):
    return not E5.covers(d)


def fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def main():
    tab = table(load_main(), load_comparison())
    reg = next(v for v in tab.values() if v["registered"])
    contrasts = [("main", k) for k in reg["main"] if " - " in k] + \
                [("comparison", k) for k in reg["comparison"] if " - " in k]
    changes = []
    for key, rec in tab.items():
        for sec, k in contrasts:
            if excl(rec[sec][k]) != excl(reg[sec][k]):
                changes.append((key, k, fmt(rec[sec][k]), fmt(reg[sec][k])))
    out = dict(script="src/r1_e3b.py", registered_budget="mmlu<=4,ppl<=1.10", budgets=tab,
               status_changes=[dict(budget=a, contrast=b, value=c, registered=d) for a, b, c, d in changes])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e3b_budgets.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("fixed-grid results under other budgets (best admissible configuration per run)")
    print(f"  {'budget':22s} {'C-ref':8s} {'C-a':8s} {'C-ref - C-a':26s} in budget C-ref/C-a | "
          f"{'dense':7s} {'1%':7s} {'steer':7s} {'SD':7s} {'DPO':7s} {'dense - steer':26s} {'dense - SD':26s}")
    for key, r in tab.items():
        m, c = r["main"], r["comparison"]
        print(f"  {key:22s} {m['C-ref']['point']:+.3f}   {m['C-a']['point']:+.3f}   {fmt(m['C-ref - C-a']):26s} "
              f"{100 * m['C-ref in budget']:3.0f}%/{100 * m['C-a in budget']:3.0f}%          | "
              + " ".join(f"{c[NAME[x]]['point']:+.3f} " for x in METHODS)
              + f"{fmt(c['dense edit - steering']):26s} {fmt(c['dense edit - SentenceDebias']):26s}"
              + ("   <- registered" if r["registered"] else ""))
    print("  in-budget share of configurations (comparison): " + "; ".join(
        f"{key}: " + "/".join(f"{100 * r['comparison'][NAME[x] + ' in budget']:.0f}" for x in METHODS)
        for key, r in tab.items()))
    print(f"\ncontrasts whose interval status differs from the registered budget: {len(changes)}")
    for a, b, c, d in changes:
        print(f"  {a:22s} {b:32s} {c}   (registered {d})")
    print("wrote", os.path.relpath(os.path.join(OUT, "e3b_budgets.json"), REPO))
    return 0


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    pts = [dict(removal=0.1, dmmlu_items=0, ppl_ratio=1.01), dict(removal=0.2, dmmlu_items=3, ppl_ratio=1.04),
           dict(removal=0.4, dmmlu_items=5, ppl_ratio=1.08), dict(removal=0.6, dmmlu_items=9, ppl_ratio=1.15)]
    check("registered budget (4 items, 1.10) picks 0.2", best(pts, 4, 1.10) == 0.2)
    check("6 items picks 0.4; 2 items picks 0.1", best(pts, 6, 1.10) == 0.4 and best(pts, 2, 1.10) == 0.1)
    check("perplexity alone at 1.10 picks 0.4, at 1.20 picks 0.6",
          best(pts, None, 1.10) == 0.4 and best(pts, None, 1.20) == 0.6)
    check("tighter perplexity bound 1.05 with 8 items picks 0.2", best(pts, 8, 1.05) == 0.2)
    check("nothing admissible -> 0", best(pts[2:], 2, 1.05) == 0.0 and best([], 4, 1.10) == 0.0)
    check("share counts admissible configurations", share(pts, 4, 1.10) == (2, 4))
    check("15 budgets, the registered one among them", len(BUDGETS) == 15 and REG in BUDGETS)
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "selftest": selftest}[cmd]())
