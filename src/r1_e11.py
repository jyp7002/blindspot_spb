#!/usr/bin/env python3
"""R1 E11 -- the comparison with other interventions at budget-limited scales: reading.

    python3 src/r1_e11.py            reading (refuses an incomplete panel)
    python3 src/r1_e11.py progress
    python3 src/r1_e11.py selftest

REGISTERED (R1_REBUTTAL_EXPERIMENTS.md, E11; details in R1_NOTES.md section 3, entry
of 2026-11-13, before the first unit). Eight comparison cells, seeds 0-2. Every
method is scaled to its own capability budget by the bias-blind rule and its
removal is read once, on the evaluation split, at the selected value:

    edit        C-ref at alpha*  (results/r1/E5 for occupation, results/r1/FRONT8
                for CrowS socioeconomic; not run again)
    steering    strength s* at the pre-fixed layer        (results/r1/E11)
    sentdebias  rank k* at the pre-fixed layer            (results/r1/E11)
    dpo         merge scale m* at learning rate 1e-5      (results/r1/E11)

A run in which nothing passes the budget, or a method that could not be built
(too few differing pairs), scores 0. Cell = mean over seeds; paired cell bootstrap.

  D = steering(s*) - edit(alpha*)
  E11-steering   the interval of D lies above 0: the comparison claim becomes
                 "at budget-limited scales steering removes more than the edit"
  E11-tie        the interval of D includes 0: the tie stands
  An interval entirely below 0 is not a registered outcome; it is reported as
  "the edit removes more than steering at budget-limited scales".

GATE. The unedited skew of every E11 run must equal the one stored with the edit's
run of the same (cell, seed) to 1e-6.

REPORTED, never the verdict: SentenceDebias - edit and DPO - edit; the two nulls
(random direction, random subspace) scaled by the same rule; selected values,
how often a ladder was exhausted, and how often the selected value passes the
budget on the evaluation items.
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
OUT = os.path.join(RESULTS, "r1", "E11")
CELLS, SEEDS = F8.CELLS, F8.SEEDS
METHODS = ["steering", "steering_null", "sentdebias", "sentdebias_null", "dpo"]
NAME = dict(steering="steering", steering_null="random direction", sentdebias="SentenceDebias",
            sentdebias_null="random subspace", dpo="DPO (lr 1e-5)")


class Partial(Exception):
    pass


def load():
    rows = {}
    for r in E5.jl(os.path.join(OUT, "removal.jsonl")):
        rows[(r["target"], r["axis"], r["seed"], r["method"])] = r
    return rows


def reading(D):
    if D["lo"] > 0:
        return "E11-steering"
    if E5.covers(D):
        return "E11-tie"
    return "neither registered outcome: the edit removes more than steering at budget-limited scales"


def analyse(rows, edit):
    """Pure. rows: (t, a, s, method) -> E11 row; edit: (t, a, s) -> C-ref run_r1 row."""
    need = [(t, a, s, m) for t, a in CELLS for s in SEEDS for m in METHODS]
    have = sum(k in rows for k in need)
    if have < len(need) or any((t, a, s) not in edit for t, a in CELLS for s in SEEDS):
        raise Partial(f"{have}/{len(need)} unit-methods on disk")
    bad = [k for k in need if abs(rows[k]["pre_skew"] - edit[k[:3]]["pre_skew"]) > 1e-6]
    if bad:
        raise SystemExit(f"gate: unedited skew differs from the edit's run in {bad[:6]}")

    def val(r):
        return A.z(r.get("removal")) if r.get("selected") is not None else 0.0

    def cells(m):
        return [float(np.mean([val(rows[(t, a, s, m)]) for s in SEEDS])) for t, a in CELLS]

    ec = [float(np.mean([E5.adaptive(edit[(t, a, s)]) for s in SEEDS])) for t, a in CELLS]
    out = dict(script="src/r1_e11.py", registered="R1_REBUTTAL_EXPERIMENTS.md E11; R1_NOTES.md section 3 (2026-11-13)",
               cells_order=[F8.ck(c) for c in CELLS],
               edit=dict(removal=V.boot_mean(ec), cells={F8.ck(c): ec[i] for i, c in enumerate(CELLS)},
                         eval_pass=sum(bool((edit[(t, a, s)].get("adaptive") or {}).get("eval_ok"))
                                       for t, a in CELLS for s in SEEDS),
                         runs=len(CELLS) * len(SEEDS)),
               methods={})
    for m in METHODS:
        mc = cells(m)
        rs = [rows[(t, a, s, m)] for t, a in CELLS for s in SEEDS]
        sel = [r["selected"] for r in rs if r.get("selected") is not None]
        out["methods"][m] = dict(
            removal=V.boot_mean(mc), cells={F8.ck(c): mc[i] for i, c in enumerate(CELLS)},
            minus_edit=V.boot_paired(list(zip(mc, ec))),
            selected_median=(float(np.median(sel)) if sel else None),
            status={st: sum(r.get("status") == st for r in rs) for st in ("ok", "exhausted", "none")},
            skipped=sum(bool(r.get("skipped")) for r in rs),
            eval_pass=sum(bool(r.get("eval_ok")) for r in rs), runs=len(rs))
    out["verdict"] = reading(out["methods"]["steering"]["minus_edit"])
    for real, null in (("steering", "steering_null"), ("sentdebias", "sentdebias_null")):
        out["methods"][null]["real_minus_null"] = V.boot_paired(list(zip(cells(real), cells(null))))
    return out


def edit_rows():
    return F8.load_runs()


def fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def progress():
    rows = load()
    for m in METHODS:
        n = sum((t, a, s, m) in rows for t, a in CELLS for s in SEEDS)
        print(f"E11 {m:15s} {n}/{len(CELLS) * len(SEEDS)} runs on disk")
    return 0


def main():
    try:
        out = analyse(load(), edit_rows())
    except Partial as e:
        progress()
        print(f"NO PARTIAL READS: {e}")
        return 1
    with open(os.path.join(OUT, "e11_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    e = out["edit"]
    print(f"budget-limited scales, eight comparison cells")
    print(f"  edit (C-ref at alpha*)  {fmt(e['removal'])}   eval budget passed {e['eval_pass']}/{e['runs']}")
    for m in METHODS:
        r = out["methods"][m]
        extra = f"   real - null {fmt(r['real_minus_null'])}" if "real_minus_null" in r else ""
        print(f"  {NAME[m]:18s} {fmt(r['removal'])}   minus edit {fmt(r['minus_edit'])}   selected (median) "
              f"{r['selected_median']}   status {r['status']} skipped {r['skipped']}   eval budget passed "
              f"{r['eval_pass']}/{r['runs']}{extra}")
    print("  per cell: edit | steering | SentenceDebias | DPO")
    for k in out["cells_order"]:
        print(f"    {k:26s} {e['cells'][k]:+.3f} | " + " | ".join(
            f"{out['methods'][m]['cells'][k]:+.3f}" for m in ("steering", "sentdebias", "dpo")))
    print(f"\nREGISTERED READING: {out['verdict']}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'e11_verdict.json'), REPO)}")
    return 0


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    def panel(steer, edit_v=0.30, none=()):
        rows, edit = {}, {}
        for i, (t, a) in enumerate(CELLS):
            for s in SEEDS:
                edit[(t, a, s)] = dict(pre_skew=0.5, adaptive=dict(alpha_star=20.0, removal=edit_v + 0.01 * i,
                                                                    eval_ok=True))
                for m in METHODS:
                    v = steer(i) if m == "steering" else 0.1
                    r = dict(pre_skew=0.5, selected=1.0, removal=v, status="ok", eval_ok=True)
                    if (t, a, s, m) in none:
                        r = dict(pre_skew=0.5, selected=None, removal=None, status="none", eval_ok=None)
                    rows[(t, a, s, m)] = r
        return rows, edit

    o = analyse(*panel(lambda i: 0.45 + 0.012 * i))
    check("steering above the edit in every cell -> E11-steering", o["verdict"] == "E11-steering")
    o = analyse(*panel(lambda i: 0.30 + 0.01 * i + (0.05 if i % 2 else -0.05)))
    check("differences of mixed sign -> E11-tie", o["verdict"] == "E11-tie")
    o = analyse(*panel(lambda i: 0.10))
    check("steering below the edit -> named as not registered", o["verdict"].startswith("neither"))
    k0 = (CELLS[0][0], CELLS[0][1], 0, "steering")
    o = analyse(*panel(lambda i: 0.45, none={k0}))
    check("a run in which nothing passes the budget scores 0",
          abs(o["methods"]["steering"]["cells"][F8.ck(CELLS[0])] - 0.30) < 1e-12
          and o["methods"]["steering"]["status"]["none"] == 1)
    rows, edit = panel(lambda i: 0.45)
    del rows[k0]
    try:
        analyse(rows, edit)
        check("an incomplete panel is refused", False)
    except Partial:
        check("an incomplete panel is refused", True)
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]())
