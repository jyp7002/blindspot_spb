#!/usr/bin/env python3
"""R1 E9 -- held-out capability battery at alpha*: the registered reading.

    python3 src/r1_e9.py            reading (refuses an incomplete panel)
    python3 src/r1_e9.py progress
    python3 src/r1_e9.py selftest

REGISTERED (R1_REBUTTAL_EXPERIMENTS.md, E9; details in R1_NOTES.md section 3,
entry of 2026-11-12). Four model states per run: unedited, and C-ref, C-a,
C-rand each at its own alpha*. Five likelihood tasks that no selection reads
(arc_challenge, hellaswag, truthfulqa_mc1, winogrande, mmlu_heldout).

  primary metric   per run and edited state: the mean over the five tasks of
                   (accuracy of the state - accuracy of the unedited model);
                   a run whose arm was not deployed has difference 0
  cell             mean over seeds 0-2; ten main cells; paired cell bootstrap
  D                macro(C-ref) - macro(C-a)

  E9-H1   D > 0 and its interval excludes 0: at the same budget the deployed
          edit loses less held-out capability than the random-coordinate edit
          ("magnitude selection buys held-out collateral efficiency")
  E9-H0   the interval of D includes 0: no hidden advantage
  An interval entirely below 0 is not one of the two registered outcomes; it is
  reported as "E9-H1 fails, the deployed edit loses more".

GATES. Every edited row must carry geom_match = True, an unedited skew equal to
the stored one, and (if deployed) a skew at alpha* equal to the stored one, all
to 1e-6. Otherwise the script stops and lists the rows.

REPORTED, never the verdict: each state's macro difference to the unedited
model with its interval, the per-task differences, C-ref - C-rand and
C-a - C-rand, and the generation tasks of the seeds that ran them.
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

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E9")
TASKS = ["arc_challenge", "hellaswag", "truthfulqa_mc1", "winogrande", "mmlu_heldout"]
ARMS = ["C-ref", "C-a", "C-rand"]
STATES = ["unedited"] + ARMS
TOL = 1e-6


class Partial(Exception):
    pass


def load(fname="battery.jsonl"):
    rows = {}
    for r in E5.jl(os.path.join(OUT, fname)):
        rows[(r["target"], r["axis"], r["seed"], r["state"])] = r
    return rows


def gate(rows):
    bad = []
    for (t, a, s, st), r in rows.items():
        if st == "unedited":
            continue
        if r.get("geom_match") is not True:
            bad.append(((t, a, s, st), f"geom_match={r.get('geom_match')}"))
        elif r.get("pre_skew_abs_diff") is None or r["pre_skew_abs_diff"] > TOL:
            bad.append(((t, a, s, st), f"unedited skew differs by {r.get('pre_skew_abs_diff')}"))
        elif r.get("deployed") and (r.get("skew_abs_diff") is None or r["skew_abs_diff"] > TOL):
            bad.append(((t, a, s, st), f"skew at alpha* differs by {r.get('skew_abs_diff')}"))
    return bad


def analyse(rows):
    """Pure. rows: (target, axis, seed, state) -> battery row."""
    need = [(t, a, s, st) for t, a in A.CELLS for s in A.SEEDS for st in STATES]
    have = sum(k in rows for k in need)
    if have < len(need):
        raise Partial(f"{have}/{len(need)} unit-states on disk")

    def delta(t, a, s, arm, task=None):
        r, u = rows[(t, a, s, arm)], rows[(t, a, s, "unedited")]
        if not r.get("deployed", True):
            return 0.0
        if task is None:
            return float(np.mean([r["acc"][k] - u["acc"][k] for k in TASKS]))
        return float(r["acc"][task] - u["acc"][task])

    def cells(arm, task=None):
        return [float(np.mean([delta(t, a, s, arm, task) for s in A.SEEDS])) for t, a in A.CELLS]

    macro = {arm: cells(arm) for arm in ARMS}
    D = V.boot_paired(list(zip(macro["C-ref"], macro["C-a"])))
    if D["lo"] > 0:
        verdict = "E9-H1"
    elif E5.covers(D):
        verdict = "E9-H0"
    else:
        verdict = "neither registered outcome: E9-H1 fails, the deployed edit loses more"
    out = dict(
        script="src/r1_e9.py", registered="R1_REBUTTAL_EXPERIMENTS.md E9; R1_NOTES.md section 3 (2026-11-12)",
        verdict=verdict, C_ref_minus_C_a=D,
        unedited_accuracy={k: float(np.mean([rows[(t, a, s, "unedited")]["acc"][k]
                                             for t, a in A.CELLS for s in A.SEEDS])) for k in TASKS},
        n_items={k: rows[need[0]]["n"][k] for k in TASKS},
        macro={arm: dict(delta=V.boot_mean(macro[arm]),
                         cells={f"{t}|{a}": macro[arm][i] for i, (t, a) in enumerate(A.CELLS)},
                         runs_deployed=sum(bool(rows[(t, a, s, arm)].get("deployed", True))
                                           for t, a in A.CELLS for s in A.SEEDS))
               for arm in ARMS},
        per_task={k: {arm: V.boot_mean(cells(arm, k)) for arm in ARMS} for k in TASKS},
        per_task_C_ref_minus_C_a={k: V.boot_paired(list(zip(cells("C-ref", k), cells("C-a", k))))
                                  for k in TASKS},
        C_ref_minus_C_rand=V.boot_paired(list(zip(macro["C-ref"], macro["C-rand"]))),
        C_a_minus_C_rand=V.boot_paired(list(zip(macro["C-a"], macro["C-rand"]))))
    return out


def fmt(d, scale=100.0):
    return f"{scale * d['point']:+.2f} [{scale * d['lo']:+.2f}, {scale * d['hi']:+.2f}]"


def progress():
    rows = load()
    for st in STATES:
        n = sum((t, a, s, st) in rows for t, a in A.CELLS for s in A.SEEDS)
        print(f"E9 {st:9s} {n}/{len(A.CELLS) * len(A.SEEDS)} runs on disk")
    g = load("gen.jsonl")
    print(f"E9 generation tasks: {len(g)} unit-states on disk")
    return 0


def main():
    rows = load()
    try:
        out = analyse(rows)
    except Partial as e:
        progress()
        print(f"NO PARTIAL READS: {e}")
        return 1
    bad = gate(rows)
    if bad:
        print(f"REPLAY GATE: {len(bad)} rows do not reproduce the stored panels:")
        for k, why in bad[:40]:
            print("   ", k, why)
        return 1
    gen = load("gen.jsonl")
    if gen:
        out["generation_descriptive"] = {
            f"{t}|{a}|s{s}|{st}": r.get("gen") for (t, a, s, st), r in sorted(gen.items())}
    with open(os.path.join(OUT, "e9_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("held-out battery at alpha*, ten main cells; accuracy differences to the unedited model, "
          "in percentage points")
    print("  unedited accuracy: " + ", ".join(f"{k} {100 * v:.1f} (n={out['n_items'][k]})"
                                             for k, v in out["unedited_accuracy"].items()))
    for arm in ARMS:
        m = out["macro"][arm]
        print(f"  {arm:7s} macro {fmt(m['delta'])}   deployed {m['runs_deployed']}/30   "
              + "  ".join(f"{k} {fmt(out['per_task'][k][arm])}" for k in TASKS))
    print(f"  C-ref - C-a   macro {fmt(out['C_ref_minus_C_a'])}   "
          + "  ".join(f"{k} {fmt(v)}" for k, v in out["per_task_C_ref_minus_C_a"].items()))
    print(f"  C-ref - C-rand macro {fmt(out['C_ref_minus_C_rand'])}   C-a - C-rand macro "
          f"{fmt(out['C_a_minus_C_rand'])}")
    print("  per cell, macro: " + "; ".join(
        f"{k} {100 * out['macro']['C-ref']['cells'][k]:+.2f}/{100 * out['macro']['C-a']['cells'][k]:+.2f}/"
        f"{100 * out['macro']['C-rand']['cells'][k]:+.2f}" for k in out["macro"]["C-ref"]["cells"]))
    if gen:
        print(f"  generation tasks (descriptive): {len(gen)} unit-states in e9_verdict.json")
    print(f"\nREGISTERED READING: {out['verdict']}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'e9_verdict.json'), REPO)}")
    return 0


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    def panel(d_ref, d_a, d_rand=-0.05, undeployed=()):
        rows = {}
        for i, (t, a) in enumerate(A.CELLS):
            for s in A.SEEDS:
                base = {k: 0.5 + 0.01 * i for k in TASKS}
                rows[(t, a, s, "unedited")] = dict(acc=dict(base), n={k: 100 for k in TASKS})
                for arm, d in (("C-ref", d_ref), ("C-a", d_a), ("C-rand", d_rand)):
                    dep = (t, a, s, arm) not in undeployed
                    rows[(t, a, s, arm)] = dict(
                        acc={k: base[k] + d + (0.002 * i if arm == "C-ref" else 0.0) for k in TASKS},
                        n={k: 100 for k in TASKS}, deployed=dep, geom_match=True,
                        pre_skew_abs_diff=0.0, skew_abs_diff=0.0)
        return rows

    o = analyse(panel(-0.01, -0.03))
    check("deployed edit loses less -> E9-H1", o["verdict"] == "E9-H1"
          and abs(o["macro"]["C-a"]["delta"]["point"] + 0.03) < 1e-12)
    o = analyse(panel(-0.03, -0.01))
    check("deployed edit loses more -> not a registered outcome, named as such",
          o["verdict"].startswith("neither"))
    rows = panel(-0.02, -0.02)
    for i, (t, a) in enumerate(A.CELLS):          # make the difference straddle zero
        for s in A.SEEDS:
            for k in TASKS:
                rows[(t, a, s, "C-ref")]["acc"][k] = rows[(t, a, s, "unedited")]["acc"][k] - 0.02 + (0.01 if i % 2 else -0.01)
    check("mixed signs -> E9-H0", analyse(rows)["verdict"] == "E9-H0")
    k0 = (A.CELLS[0][0], A.CELLS[0][1], 0, "C-a")
    o = analyse(panel(-0.01, -0.03, undeployed={k0}))
    check("a run that was not deployed counts as difference 0",
          abs(o["macro"]["C-a"]["cells"][f"{A.CELLS[0][0]}|{A.CELLS[0][1]}"] + 0.02) < 1e-12
          and o["macro"]["C-a"]["runs_deployed"] == 29)
    rows = panel(-0.01, -0.03)
    del rows[k0]
    try:
        analyse(rows)
        check("an incomplete panel is refused", False)
    except Partial:
        check("an incomplete panel is refused", True)
    rows = panel(-0.01, -0.03)
    rows[k0]["skew_abs_diff"] = 1e-3
    check("a state whose skew does not replay is caught by the gate", len(gate(rows)) == 1)
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]())
