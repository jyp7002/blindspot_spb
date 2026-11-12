#!/usr/bin/env python3
"""R1 E6 -- collateral-aware support selection (hypothesis H-act): registered readings.

    python3 src/r1_e6.py            readings (refuses an incomplete panel)
    python3 src/r1_e6.py progress
    python3 src/r1_e6.py selftest

REGISTERED (R1_REBUTTAL_EXPERIMENTS.md, E6; details in R1_NOTES.md section 3, entry
of 2026-11-12, before the first unit).

HYPOTHESIS. In a linear layer |dW_ij| grows with the size n_j of input channel j,
and so does the capability curvature (as n_j^2). Selecting by |dW| would then
select high-curvature coordinates, which would explain why the deployed edit
reaches its budget at a quarter of the scale of the random-coordinate edit.

E6-M (mechanism) holds iff BOTH
  (a) the n_j^2 enrichment of C-ref's support -- mean of n_j^2 over its kept
      entries divided by the mean over all edited entries, per run; cell = seed
      mean -- has a ten-cell mean of at least 1.5 with an interval above 1; and
  (b) the median over cells of h_proxy(C-ref) / h_proxy(C-a), with
      h_proxy(S) = sum_t s_t^2 sum_{(i,j) in S_t} n_j^2, lies within [0.25, 4]
      times the median over cells of r_alpha^2, r_alpha = alpha*(C-a) /
      alpha*(C-ref) of the same run (cell value = median over seeds).
E6-W (method) holds iff the intervals of
      S-act1 - C-ref,  S-act1 - C-a,  S-act1 - C-rand      (removal at alpha*)
  all lie above 0.
CONTROL. S-wanda(alpha*) <= C-ref(alpha*) on the ten-cell means; if not, the
  mechanism reading is withheld whatever (a) and (b) say.
S-fisher is secondary: if S-act1 fails E6-W and S-fisher passes the same three
  tests, that is reported as "secondary, needs confirmation" and decides nothing.

Reference arms at alpha*: C-ref and C-a from results/r1/E5 (exact replays of the
registered v12astar values), C-rand from results/r1/E4. Removal as deployed at
alpha*, not deployed -> 0, cell = seed mean, paired cell bootstrap (v10_common).

GATES. Every E6 unit must reproduce the registered contrast-vector fingerprint
and carry an E6a row saying that the baseline was unchanged after the
statistics pass. NO PARTIAL READS.
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
OUT = os.path.join(RESULTS, "r1", "E6")
ARMS = ["S-act1", "S-wanda", "S-fisher"]
ENRICH_MIN, H_BAND = 1.5, (0.25, 4.0)


class Partial(Exception):
    pass


def e6m(enrich_cells, h_cells, r2_cells):
    """Pure. Lists over the ten cells -> the mechanism reading."""
    e = V.boot_mean(enrich_cells)
    hm, rm = float(np.median(h_cells)), float(np.median(r2_cells))
    a = e["point"] >= ENRICH_MIN and e["lo"] > 1.0
    b = H_BAND[0] * rm <= hm <= H_BAND[1] * rm
    return dict(enrichment=e, a_holds=bool(a), h_proxy_ratio_median=hm, r_alpha_sq_median=rm,
                h_over_r_alpha_sq=hm / rm, b_holds=bool(b), holds=bool(a and b))


def e6w(arm_cells, ref, a, rand):
    """Pure. Cell lists of removal at alpha* -> the three differences and the test."""
    d = dict(minus_C_ref=V.boot_paired(list(zip(arm_cells, ref))),
             minus_C_a=V.boot_paired(list(zip(arm_cells, a))),
             minus_C_rand=V.boot_paired(list(zip(arm_cells, rand))))
    d["holds"] = bool(all(d[k]["lo"] > 0 for k in ("minus_C_ref", "minus_C_a", "minus_C_rand")))
    return d


def analyse(rows, units, e6a, e5, e4):
    need = [(t, a, s, arm) for t, a in A.CELLS for s in A.SEEDS for arm in ARMS]
    have = sum(k in rows for k in need)
    ukeys = [(t, a, s) for t, a in A.CELLS for s in A.SEEDS]
    if have < len(need) or any(k not in e6a for k in ukeys):
        raise Partial(f"{have}/{len(need)} unit-arms and {sum(k in e6a for k in ukeys)}/{len(ukeys)} "
                      "diagnostic rows on disk")
    geom_bad = sorted(k for k in ukeys if (units.get(k) or {}).get("geom_match") is not True)
    base_bad = sorted(k for k in ukeys if e6a[k].get("baseline_unchanged_after_stats") is not True)
    if geom_bad or base_bad:
        raise SystemExit(f"gate: fingerprint not reproduced in {geom_bad}; baseline changed in {base_bad}")

    def cells(table, arm):
        return [float(np.mean([E5.adaptive(table[(t, a, s, arm)]) for s in A.SEEDS])) for t, a in A.CELLS]

    ref, ca, rand = cells(e5, "C-ref"), cells(e5, "C-a"), cells(e4, "C-rand")
    enrich = [float(np.mean([e6a[(t, a, s)]["enrich_C_ref"] for s in A.SEEDS])) for t, a in A.CELLS]
    enrich_a = [float(np.mean([e6a[(t, a, s)]["enrich_C_a"] for s in A.SEEDS])) for t, a in A.CELLS]
    within = [float(np.mean([e6a[(t, a, s)]["enrich_C_ref_within_tensor"] for s in A.SEEDS]))
              for t, a in A.CELLS]
    h = [float(np.median([e6a[(t, a, s)]["h_proxy_ratio"] for s in A.SEEDS])) for t, a in A.CELLS]
    r2 = [float(np.median([(e5[(t, a, s, "C-a")]["adaptive"]["alpha_star"]
                            / e5[(t, a, s, "C-ref")]["adaptive"]["alpha_star"]) ** 2 for s in A.SEEDS]))
          for t, a in A.CELLS]
    M = e6m(enrich, h, r2)
    M.update(enrichment_C_a=V.boot_mean(enrich_a), enrichment_within_tensor=V.boot_mean(within),
             cells={f"{t}|{a}": dict(enrich_C_ref=enrich[i], enrich_C_a=enrich_a[i], h_proxy_ratio=h[i],
                                     r_alpha_sq=r2[i]) for i, (t, a) in enumerate(A.CELLS)})
    proj = {}
    for k in ukeys:
        for name, q in e6a[k]["by_projection"].items():
            proj.setdefault(name, []).append(q)
    M["by_projection"] = {name: dict(
        mean_n2_over_global=float(np.mean([q["mean_n2_over_global"] for q in qs])),
        share_of_entries=float(np.mean([q["share_of_entries"] for q in qs])),
        share_of_C_ref_support=float(np.mean([q["share_of_C_ref_support"] for q in qs])),
        units=len(qs)) for name, qs in sorted(proj.items())}

    arms = {}
    for arm in ARMS:
        ac = cells(rows, arm)
        rs = [rows[(t, a, s, arm)] for t, a in A.CELLS for s in A.SEEDS]
        st = [((r.get("adaptive") or {}).get("alpha_star")) for r in rs]
        arms[arm] = dict(
            removal=V.boot_mean(ac), cells={f"{t}|{a}": ac[i] for i, (t, a) in enumerate(A.CELLS)},
            test=e6w(ac, ref, ca, rand),
            alpha_star_median=float(np.median([x for x in st if x is not None])) if any(st) else None,
            runs_deployed=sum(x is not None for x in st),
            eval_pass_rate=float(np.mean([bool((r.get("adaptive") or {}).get("eval_ok")) for r in rs])),
            overlap_with_C_ref=float(np.mean([r["meta"]["overlap_with_C_ref"] for r in rs])),
            fixed_grid_removal=V.boot_mean([float(np.mean(
                [A.z((rows[(t, a, s, arm)].get("frozen") or {}).get("removal")) for s in A.SEEDS]))
                for t, a in A.CELLS]))
    control = float(np.mean(cells(rows, "S-wanda"))) <= float(np.mean(ref))
    W = arms["S-act1"]["test"]["holds"]
    if W:
        w_text = "E6-W holds"
    elif arms["S-fisher"]["test"]["holds"]:
        w_text = "E6-W fails; S-fisher passes the same tests (secondary, needs confirmation)"
    else:
        w_text = "E6-W fails"
    m_text = ("E6-M holds" if M["holds"] else "E6-M fails") + (
        "" if control else " (CONTROL VIOLATED: S-wanda > C-ref; mechanism reading withheld)")
    return dict(script="src/r1_e6.py", registered="R1_REBUTTAL_EXPERIMENTS.md E6; R1_NOTES.md section 3 (2026-11-12)",
                E6_M=M, arms=arms, control_S_wanda_le_C_ref=bool(control),
                references=dict(C_ref=float(np.mean(ref)), C_a=float(np.mean(ca)), C_rand=float(np.mean(rand))),
                verdict=dict(E6_M=m_text, E6_W=w_text))


def fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def load_all():
    rows, units = E5.load(OUT)
    e6a = {}
    for r in E5.jl(os.path.join(OUT, "e6a.jsonl")):
        e6a[(r["target"], r["axis"], r["seed"])] = r
    e5, _ = E5.load()
    e4, _ = E5.load(os.path.join(RESULTS, "r1", "E4"))
    return rows, units, e6a, e5, e4


def progress():
    rows, _units, e6a, _e5, _e4 = load_all()
    n = len(A.CELLS) * len(A.SEEDS)
    print(f"E6a diagnostics {sum((t, a, s) in e6a for t, a in A.CELLS for s in A.SEEDS)}/{n} units on disk")
    for arm in ARMS:
        print(f"E6b {arm:9s} {sum((t, a, s, arm) in rows for t, a in A.CELLS for s in A.SEEDS)}/{n} runs on disk")
    return 0


def main():
    try:
        out = analyse(*load_all())
    except Partial as e:
        progress()
        print(f"NO PARTIAL READS: {e}")
        return 1
    with open(os.path.join(OUT, "e6_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    M = out["E6_M"]
    print("E6a  n_j^2 enrichment of the support (ten cells): C-ref "
          f"{M['enrichment']['point']:.2f} [{M['enrichment']['lo']:.2f}, {M['enrichment']['hi']:.2f}]"
          f", C-a {M['enrichment_C_a']['point']:.2f}; within tensors, C-ref "
          f"{M['enrichment_within_tensor']['point']:.2f}  -> (a) {'holds' if M['a_holds'] else 'fails'}")
    print(f"     h_proxy(C-ref)/h_proxy(C-a) median {M['h_proxy_ratio_median']:.2f}; r_alpha^2 median "
          f"{M['r_alpha_sq_median']:.2f}; ratio {M['h_over_r_alpha_sq']:.2f} (band 0.25-4)  -> (b) "
          f"{'holds' if M['b_holds'] else 'fails'}")
    for name, q in M["by_projection"].items():
        print(f"     {name:9s} mean n_j^2 / global {q['mean_n2_over_global']:.2f}; share of entries "
              f"{q['share_of_entries']:.2f}; share of C-ref's support {q['share_of_C_ref_support']:.2f}")
    r = out["references"]
    print(f"E6b  references at alpha*: C-ref {r['C_ref']:+.3f}, C-a {r['C_a']:+.3f}, C-rand {r['C_rand']:+.3f}")
    for arm in ARMS:
        a = out["arms"][arm]
        t = a["test"]
        print(f"     {arm:9s} {fmt(a['removal'])}  - C-ref {fmt(t['minus_C_ref'])}  - C-a {fmt(t['minus_C_a'])}  "
              f"- C-rand {fmt(t['minus_C_rand'])}  alpha* median {a['alpha_star_median']}  eval pass "
              f"{100 * a['eval_pass_rate']:.0f}%  overlap with C-ref {a['overlap_with_C_ref']:.2f}  "
              f"fixed grid {fmt(a['fixed_grid_removal'])}")
    print(f"     control S-wanda <= C-ref: {out['control_S_wanda_le_C_ref']}")
    print(f"\nREGISTERED READINGS: {out['verdict']['E6_M']}; {out['verdict']['E6_W']}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'e6_verdict.json'), REPO)}")
    return 0


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    ten = lambda x, d=0.0: [x + d * i for i in range(10)]   # noqa: E731
    m = e6m(ten(2.0, 0.05), ten(20.0), ten(19.0))
    check("enrichment 2 and h ratio near r_alpha^2 -> E6-M holds", m["holds"] and m["a_holds"] and m["b_holds"])
    m = e6m(ten(1.2, 0.01), ten(20.0), ten(19.0))
    check("enrichment 1.2 -> (a) fails", not m["a_holds"] and not m["holds"])
    m = e6m(ten(2.0, 0.05), ten(2.0), ten(19.0))
    check("h ratio a tenth of r_alpha^2 -> (b) fails", m["a_holds"] and not m["b_holds"])
    ref, ca, rand = ten(0.38, 0.01), ten(0.30, 0.01), ten(0.01)
    check("S-act1 above all three references -> E6-W holds",
          e6w(ten(0.45, 0.012), ref, ca, rand)["holds"])
    check("S-act1 equal to C-ref -> E6-W fails", not e6w(list(ref), ref, ca, rand)["holds"])
    check("S-act1 above C-a and C-rand but below C-ref -> E6-W fails",
          not e6w(ten(0.34, 0.01), ref, ca, rand)["holds"])
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]())
