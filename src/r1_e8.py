#!/usr/bin/env python3
"""R1 E8 -- alpha* backoff and budget sensitivity: the registered reading.

    python3 src/r1_e8.py            # -> results/r1/E5/e8_verdict.json
    python3 src/r1_e8.py progress   # unit counts only; reads no removal
    python3 src/r1_e8.py selftest

REGISTERED (R1_REBUTTAL_EXPERIMENTS.md, E8; commit 0fe9043; measured inside the
E5 panel on its two replay arms, R1_NOTES.md section 3, first entry). For C-ref
and C-a, each at its own alpha*, the edit is measured on the evaluation split
and on 1,000 MMLU items at alpha*, alpha*/sqrt(2) and alpha*/2, and alpha* is
re-selected with a 2-item and an 8-item MMLU allowance.

  Deployment rule: alpha*/sqrt(2).  Target: evaluation-budget pass >= 90%.
  Prediction (homogeneous collateral): Delta_selection = C-ref - C-a does not
  change under backoff.

HOW THE TWO SENTENCES ARE READ (fixed in R1_NOTES.md section 3 before the panel
was complete, since the plan does not spell them out):
  - the pass rate is over the units in which the edit is deployed (alpha* exists),
    per arm; the target is met only if BOTH arms reach 0.90 at alpha*/sqrt(2).
    The pooled rate is printed next to it.
  - "does not change": the paired cell bootstrap interval of
    Delta_selection(alpha* x f) - Delta_selection(alpha*) includes 0, for
    f = 1/sqrt(2) (the deployment rule; the reading) and f = 1/2 (reported).

Removal is the registered |pre skew| - |post skew|; an undeployed edit scores 0
at every backoff; cell = seed mean; ten registered cells; paired cell bootstrap.
The signed and capped definitions of E3a are printed next to it, labelled.

NO PARTIAL READS: nothing is printed until all 60 unit-arms (C-ref and C-a) are
on disk.
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

OUT = E5.OUT
ARMS = ["C-ref", "C-a"]
FACTORS = [1.0, 0.7071067811865476, 0.5]
FNAME = {1.0: "alpha*", 0.7071067811865476: "alpha*/sqrt2", 0.5: "alpha*/2"}
DEPLOY = 0.7071067811865476
TARGET = 0.90
BUDGETS = ["2", "4", "8"]          # MMLU items allowed on 200; 4 is the registered alpha*


def at_factor(r, f):
    """The rung record of this unit at alpha* x f; None if the edit is not deployed."""
    a_star = r["adaptive"].get("alpha_star")
    if a_star is None:
        return None
    alpha = [b["alpha"] for b in r["backoff"] if abs(b["factor"] - f) < 1e-9]
    if len(alpha) != 1:
        raise SystemExit(f"backoff factor {f} not recorded once: {r['target']} {r['axis']} "
                         f"s{r['seed']} {r['arm']}")
    recs = [x for x in r["rungs"] if x["alpha"] == alpha[0]]
    if len(recs) != 1:
        raise SystemExit(f"rung alpha={alpha[0]} not recorded once: {r['target']} "
                         f"{r['axis']} s{r['seed']} {r['arm']}")
    return recs[0]


def val(r, f, defn):
    rec = at_factor(r, f)
    if rec is None:
        return 0.0
    if defn == "ii_abs":
        return A.z(rec["eval"]["removal"])
    return E5.removal(defn, r["pre_skew"], rec["eval"]["post_skew"])


def val_budget(r, b, defn):
    d = r["adaptive"] if b == "4" else r["budgets"][b]
    if d.get("alpha_star") is None:
        return 0.0
    if defn == "ii_abs":
        return A.z(d.get("removal"))
    return E5.removal(defn, r["pre_skew"], d["post_skew"])


def cells(rows, arm, fn):
    return [float(np.mean([fn(rows[(t, a, s, arm)]) for s in A.SEEDS])) for t, a in A.CELLS]


def analyse(rows):
    miss = [k for k in E5.need(ARMS) if k not in rows]
    if miss:
        raise E5.Partial(f"{len(E5.need(ARMS)) - len(miss)}/{len(E5.need(ARMS))} "
                         "unit-arms on disk")
    # sanity: the factor-1 rung is the alpha* measurement itself
    for k in E5.need(ARMS):
        r = rows[k]
        rec = at_factor(r, 1.0)
        if rec is not None and rec["eval"]["removal"] != r["adaptive"]["removal"]:
            raise SystemExit(f"factor-1 rung differs from the alpha* row: {k}")

    out = dict(script="src/r1_e8.py", registered="R1_REBUTTAL_EXPERIMENTS.md E8 (0fe9043)",
               backoff={}, budgets={})
    # ---- pass rates and collateral at each backoff ----
    for f in FACTORS:
        d = dict(arms={})
        pooled = []
        for arm in ARMS:
            recs = [at_factor(rows[k], f) for k in E5.need([arm])]
            dep = [x for x in recs if x is not None]
            ok = [bool(x["eval"]["ok"]) for x in dep]
            pooled += ok
            has1k = [x for x in dep if "mmlu1k" in x]
            d["arms"][arm] = dict(
                n_deployed=len(dep), n_not_deployed=len(recs) - len(dep),
                eval_pass=sum(ok), eval_pass_rate=(sum(ok) / len(ok) if ok else None),
                eval_dmmlu_items_mean=float(np.mean([x["eval"]["dmmlu_items"] for x in dep])),
                eval_ppl_ratio_median=float(np.median([x["eval"]["ppl_ratio"] for x in dep])),
                mmlu1k_n=len(has1k),
                mmlu1k_drop_items_mean=(float(np.mean([x["mmlu1k"]["dmmlu_items"]
                                                       for x in has1k])) if has1k else None),
                mmlu1k_pass_rate=(float(np.mean([bool(x["mmlu1k"]["ok"]) for x in has1k]))
                                  if has1k else None))
        d["eval_pass_rate_pooled"] = sum(pooled) / len(pooled) if pooled else None
        d["by_definition"] = {}
        for defn in E5.DEFNS:
            cr = cells(rows, "C-ref", lambda r: val(r, f, defn))
            ca = cells(rows, "C-a", lambda r: val(r, f, defn))
            c1r = cells(rows, "C-ref", lambda r: val(r, 1.0, defn))
            c1a = cells(rows, "C-a", lambda r: val(r, 1.0, defn))
            dsel = [x - y for x, y in zip(cr, ca)]
            dsel1 = [x - y for x, y in zip(c1r, c1a)]
            ch = V.boot_paired(list(zip(dsel, dsel1)))
            d["by_definition"][defn] = dict(
                C_ref=V.boot_mean(cr), C_a=V.boot_mean(ca),
                delta_selection=V.boot_paired(list(zip(cr, ca))),
                change_vs_alpha_star=ch, change_includes_zero=E5.covers(ch),
                retained_C_ref=float(np.mean(cr) / np.mean(c1r)) if np.mean(c1r) else None,
                retained_C_a=float(np.mean(ca) / np.mean(c1a)) if np.mean(c1a) else None)
        out["backoff"][FNAME[f]] = d

    # ---- alpha* re-selected under a 2-item / 8-item allowance ----
    for b in BUDGETS:
        d = dict(arms={}, by_definition={})
        for arm in ARMS:
            ds = [(rows[k]["adaptive"] if b == "4" else rows[k]["budgets"][b])
                  for k in E5.need([arm])]
            al = [x["alpha_star"] for x in ds if x.get("alpha_star") is not None]
            okk = "eval_ok" if b == "4" else "eval_ok_same_budget"
            ok = [bool(x[okk]) for x in ds if x.get("alpha_star") is not None]
            d["arms"][arm] = dict(n_deployed=len(al),
                                  alpha_star_median=(float(np.median(al)) if al else None),
                                  eval_pass_same_budget=(sum(ok) / len(ok) if ok else None))
        for defn in E5.DEFNS:
            cr = cells(rows, "C-ref", lambda r: val_budget(r, b, defn))
            ca = cells(rows, "C-a", lambda r: val_budget(r, b, defn))
            d["by_definition"][defn] = dict(C_ref=V.boot_mean(cr), C_a=V.boot_mean(ca),
                                            delta_selection=V.boot_paired(list(zip(cr, ca))))
        out["budgets"][b] = d

    dep = out["backoff"][FNAME[DEPLOY]]
    rates = {arm: dep["arms"][arm]["eval_pass_rate"] for arm in ARMS}
    out["verdict"] = dict(
        deployment_rule="alpha*/sqrt2", target=TARGET, eval_pass_rate=rates,
        eval_pass_rate_pooled=dep["eval_pass_rate_pooled"],
        target_met=bool(all(v is not None and v >= TARGET for v in rates.values())),
        delta_selection_unchanged_under_backoff=bool(
            dep["by_definition"]["ii_abs"]["change_includes_zero"]),
        change=dep["by_definition"]["ii_abs"]["change_vs_alpha_star"])
    return out


def report(out):
    print("evaluation budget (4 of 200 MMLU items, perplexity x1.10) and MMLU-1000, by backoff")
    for name, d in out["backoff"].items():
        for arm, x in d["arms"].items():
            print(f"  {name:13s} {arm:6s} eval pass {x['eval_pass']}/{x['n_deployed']} = "
                  f"{x['eval_pass_rate']:.3f}  dMMLU(200) {x['eval_dmmlu_items_mean']:+.2f}  "
                  f"ppl x{x['eval_ppl_ratio_median']:.3f}  dMMLU(1000) "
                  f"{x['mmlu1k_drop_items_mean']}  1k pass {x['mmlu1k_pass_rate']}  "
                  f"(not deployed: {x['n_not_deployed']})")
        print(f"  {name:13s} pooled eval pass {d['eval_pass_rate_pooled']:.3f}")
    for defn, label in E5.DEFNS.items():
        print(f"\n[{label}]")
        for name, d in out["backoff"].items():
            x = d["by_definition"][defn]
            print(f"  {name:13s} C-ref {V.fmt(x['C_ref'], 3)}  C-a {V.fmt(x['C_a'], 3)}  "
                  f"D_sel {V.fmt(x['delta_selection'], 3)}  change vs alpha* "
                  f"{V.fmt(x['change_vs_alpha_star'], 3)}  retained "
                  f"{x['retained_C_ref']}/{x['retained_C_a']}")
        for b, d in out["budgets"].items():
            x = d["by_definition"][defn]
            print(f"  budget {b} items  alpha* median "
                  f"{d['arms']['C-ref']['alpha_star_median']}/"
                  f"{d['arms']['C-a']['alpha_star_median']}  C-ref {V.fmt(x['C_ref'], 3)}  "
                  f"C-a {V.fmt(x['C_a'], 3)}  D_sel {V.fmt(x['delta_selection'], 3)}  "
                  f"eval pass at that budget {d['arms']['C-ref']['eval_pass_same_budget']}/"
                  f"{d['arms']['C-a']['eval_pass_same_budget']}")
    v = out["verdict"]
    print(f"\nREGISTERED READING  deployment rule alpha*/sqrt2: evaluation pass "
          + ", ".join(f"{a} {r:.3f}" for a, r in v["eval_pass_rate"].items())
          + f" (pooled {v['eval_pass_rate_pooled']:.3f}); target {v['target']} -> "
          + ("MET" if v["target_met"] else "NOT MET"))
    print(f"                    Delta_selection change under backoff {V.fmt(v['change'], 3)} -> "
          + ("unchanged (interval includes 0)" if v["delta_selection_unchanged_under_backoff"]
             else "changes"))


def progress():
    rows, _ = E5.load()
    for arm in ARMS:
        n = E5.need([arm])
        print(f"{E5.PANEL} {arm:6s} {sum(k in rows for k in n)}/{len(n)} units on disk")
    return 0


def main():
    rows, _ = E5.load()
    try:
        out = analyse(rows)
    except E5.Partial as e:
        progress()
        print(f"incomplete ({e}): E8 is not read before every unit exists")
        return 1
    with open(os.path.join(OUT, "e8_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    report(out)
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'e8_verdict.json'), REPO)}")
    return 0


# ------------------------------------------------------------------ selftest ----
def _rung(alpha, pre, rem, ok, k1=True):
    r = dict(alpha=alpha, eval=dict(removal=rem, post_skew=pre - rem, ok=ok,
                                    dmmlu_items=1, ppl_ratio=1.02))
    if k1:
        r["mmlu1k"] = dict(dmmlu_items=5, ok=True)
    return r


def _unit(t, a, s, arm, pre, rem, fail_at_star, fail_at_backoff=False, shrink=1.0):
    """removal at alpha* x f is rem x f**shrink-ish: here simply rem * (f ** shrink)."""
    a_star = 8.0
    rungs = [_rung(a_star * f, pre, rem * (f ** shrink),
                   ok=not ((f == 1.0 and fail_at_star) or (f != 1.0 and fail_at_backoff)))
             for f in FACTORS]
    ad = dict(alpha_star=a_star, status="ok", removal=rem, post_skew=pre - rem,
              eval_ok=not fail_at_star)
    bud = {b: dict(alpha_star=a_star * m, status="ok", removal=rem * m, post_skew=pre - rem * m,
                   eval_ok_same_budget=True) for b, m in (("2", 0.5), ("8", 1.0))}
    return dict(target=t, axis=a, seed=s, arm=arm, pre_skew=pre, adaptive=ad,
                backoff=[dict(factor=f, alpha=a_star * f) for f in FACTORS],
                budgets=bud, rungs=sorted(rungs, key=lambda x: x["alpha"]))


def selftest():
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(f"  {'ok ' if cond else 'FAIL'} {name}")

    def panel(shrink_ref=1.0, shrink_a=1.0, fail_backoff_every=0):
        rows, i = {}, 0
        for ci, (t, a) in enumerate(A.CELLS):
            for s in A.SEEDS:
                for arm, rem, sh in (("C-ref", 0.30 + 0.01 * ci, shrink_ref),
                                     ("C-a", 0.20 + 0.01 * ci, shrink_a)):
                    i += 1
                    rows[(t, a, s, arm)] = _unit(
                        t, a, s, arm, 0.6, rem, fail_at_star=(i % 2 == 0),
                        fail_at_backoff=bool(fail_backoff_every and i % fail_backoff_every == 0),
                        shrink=sh)
        return rows

    o = analyse(panel())
    v = o["verdict"]
    check("alpha* pooled pass rate 0.5; every backoff unit passes -> target met",
          o["backoff"]["alpha*"]["eval_pass_rate_pooled"] == 0.5 and v["target_met"]
          and v["eval_pass_rate"]["C-ref"] == 1.0)
    d = o["backoff"]["alpha*/sqrt2"]["by_definition"]["ii_abs"]
    check("Delta_selection scales with f when both arms shrink alike (0.1 -> 0.0707)",
          abs(d["delta_selection"]["point"] - 0.1 * DEPLOY) < 1e-9
          and abs(d["retained_C_ref"] - DEPLOY) < 1e-9)
    o2 = analyse(panel(shrink_ref=0.0, shrink_a=0.0))
    check("removal flat under backoff -> Delta_selection unchanged",
          o2["verdict"]["delta_selection_unchanged_under_backoff"]
          and abs(o2["verdict"]["change"]["point"]) < 1e-12)
    o3 = analyse(panel(shrink_ref=0.0, shrink_a=2.0))
    check("only C-a shrinks under backoff -> Delta_selection changes",
          not o3["verdict"]["delta_selection_unchanged_under_backoff"]
          and o3["verdict"]["change"]["point"] > 0)
    o4 = analyse(panel(fail_backoff_every=5))
    check("backoff pass rate 0.8 -> target not met", not o4["verdict"]["target_met"]
          and abs(o4["backoff"]["alpha*/sqrt2"]["eval_pass_rate_pooled"] - 0.8) < 1e-9)
    check("budget 2 halves removal in the synthetic panel",
          abs(o["budgets"]["2"]["by_definition"]["ii_abs"]["delta_selection"]["point"]
              - 0.05) < 1e-9
          and abs(o["budgets"]["4"]["by_definition"]["ii_abs"]["delta_selection"]["point"]
                  - 0.1) < 1e-9)
    rows = panel()
    k = (A.CELLS[0][0], A.CELLS[0][1], 0, "C-a")
    rows[k] = dict(rows[k], adaptive=dict(alpha_star=None, status="none"), backoff=None,
                   budgets={"2": dict(alpha_star=None, status="none"),
                            "8": dict(alpha_star=None, status="none")})
    o5 = analyse(rows)
    check("an undeployed unit scores 0 and leaves the pass-rate denominator",
          o5["backoff"]["alpha*"]["arms"]["C-a"]["n_deployed"] == 29
          and o5["backoff"]["alpha*"]["arms"]["C-a"]["n_not_deployed"] == 1)
    del rows[k]
    try:
        analyse(rows)
        check("one missing unit -> refuses", False)
    except E5.Partial:
        check("one missing unit -> refuses", True)
    print(f"selftest: {sum(ok)}/{len(ok)}")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]() or 0)
