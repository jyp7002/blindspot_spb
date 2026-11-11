#!/usr/bin/env python3
"""R1 E13 -- balanced-contrast control on the occupation axis: the reading.

    python3 src/r1_e13.py            # -> results/r1/E13/e13_verdict.json
    python3 src/r1_e13.py progress   # unit counts only; reads no result
    python3 src/r1_e13.py selftest   # the verdict logic on synthetic numbers

REGISTERED in R1_NOTES.md section 3 (entry of 2026-11-11) before the first E13
unit. E13 is not in the plan of 0fe9043; it was added after the pronoun check
showed that the deployed edit closes the occupation skew by raising P("she") on
male-stereotyped occupations, and after we attributed that to the elicited
contrast being one-sided (76-100% of its differing items replace "he" by "she").

THE QUESTION. Does the pronoun base rate still move when the edit is built from
a contrast in which the two directions of replacement are equally frequent?

ARMS, all the deployed edit C-ref (top 1%, signs, per-matrix scales), five
occupation cells x three seeds:
    elicited   the C-ref rows of results/r1/E5 (the published contrast)
    balanced   results/r1/E13, corpus <designer>_bal  (elicited + mirrored swaps)
    exo        results/r1/E13, corpus exo              (every item swapped; secondary)

QUANTITIES, from the per-item likelihoods stored with each rung. f, m = mean
P("she") on the 12 female- / 12 male-stereotyped probe occupations;
base = (f + m) / 2; gap = f - m (equal to the occupation skew). For a run,
    shift = base(after) - base(before)
at the scale the fixed grid {2,4,8,16} selects (the operating point of the
paper's main tables; a run with no in-budget grid scale is not edited: shift 0,
removal 0). Cell = seed mean; paired bootstrap over the five cells (v10_common).

READING (primary control: balanced).
    ratio = |mean shift(balanced)| / mean shift(elicited)
    D     = shift(elicited) - shift(balanced), paired over cells
  E13-confirmed  ratio <= 0.25 and D's interval excludes 0
                 -> the base-rate shift comes from the one-sided contrast
  E13-partial    0.25 < ratio < 0.75 and D's interval excludes 0
                 -> the one-sided contrast explains part of it
  E13-refuted    ratio >= 0.75, or D's interval includes 0
                 -> the explanation is withdrawn; the shift is reported as unexplained
  Qualifier: if the balanced edit's removal under the grid rule has an interval
  that includes 0, a confirmed or partial verdict is labelled VACUOUS (an edit
  that does nothing cannot shift anything).

ALSO REPORTED, never the verdict: the same quantities at alpha* and alpha*/2;
the f-side and m-side movements separately; removal under both rules; runs in
which both f and m exceed 0.95 or fall below 0.05; the exo arm read by the same
rule, labelled secondary.

NO PARTIAL READS: an arm is not read before all of its 15 units are on disk.
"""
import ast
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V    # noqa: E402
import v12_astar as A     # noqa: E402
import r1_e5 as E5        # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E13")
CELLS = [c for c in A.CELLS if c[1] == "occ_gender"]
ARM = "C-ref"
R_CONF, R_REFUTE = 0.25, 0.75
POINTS = ("grid", "half", "star")
PNAME = {"grid": "fixed grid", "half": "alpha*/2", "star": "alpha*"}


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


def load13():
    rows = {}
    for r in E5.jl(os.path.join(OUT, "removal.jsonl")):
        if r["arm"] == ARM:
            rows[(r["target"], r["axis"], r["seed"], r["designer"])] = r      # last wins
    return rows


def arm_rows():
    """arm -> {(target, seed): (row, panel dir)}; an arm is absent until complete."""
    e5, _ = E5.load()
    r13 = load13()
    out, counts = {}, {}
    el = {(t, s): e5.get((t, a, s, ARM)) for t, a in CELLS for s in A.SEEDS}
    counts["elicited"] = sum(v is not None for v in el.values())
    if all(v is not None for v in el.values()):
        out["elicited"] = {k: (v, E5.OUT) for k, v in el.items()}
    for arm in ("balanced", "exo"):
        d = {}
        for t, a in CELLS:
            for s in A.SEEDS:
                base = el[(t, s)]["designer"] if el[(t, s)] else None
                dn = "exo" if arm == "exo" else (f"{base}_bal" if base else None)
                d[(t, s)] = r13.get((t, a, s, dn))
        counts[arm] = sum(v is not None for v in d.values())
        if all(v is not None for v in d.values()):
            out[arm] = {k: (v, OUT) for k, v in d.items()}
    return out, counts


def unit_stats(r, panel_dir, occ, n_tpl):
    fp = os.path.join(panel_dir, "probe_ll", f"{r['target']}__{r['axis']}__s{r['seed']}__"
                                             f"{r['designer']}__{ARM}.npz")
    zf = np.load(fp)
    alphas = [float(x) for x in zf["alphas"]]

    def at(alpha):
        i = [j for j, x in enumerate(alphas) if x == float(alpha)]
        if len(i) != 1:
            raise SystemExit(f"alpha {alpha} not stored exactly once in {fp}")
        return pshe(zf["ll"][i[0]], occ, n_tpl)

    pre = pshe(zf["pre_ll"], occ, n_tpl)
    if abs((pre[0] - pre[1]) - r["pre_skew"]) > 1e-5:
        raise SystemExit(f"gap of the stored likelihoods != recorded skew: {fp}")
    rung = {x["alpha"]: x for x in r["rungs"]}
    fr = r.get("frozen") or {}
    a_star = r["adaptive"].get("alpha_star")
    half = None
    if a_star is not None and r.get("backoff"):
        h = [b["alpha"] for b in r["backoff"] if abs(b["factor"] - 0.5) < 1e-9]
        half = h[0] if h else None
    pts = dict(grid=fr.get("alpha"), half=half, star=a_star)
    out = dict(pre_f=pre[0], pre_m=pre[1], pre_skew=r["pre_skew"])
    for name, alpha in pts.items():
        f, m = at(alpha) if alpha is not None else pre          # not edited at this point
        rem = 0.0 if alpha is None else A.z(rung[float(alpha)]["eval"]["removal"])
        out[name] = dict(alpha=alpha, f=f, m=m, removal=rem,
                         shift=(f + m) / 2 - (pre[0] + pre[1]) / 2,
                         d_f=f - pre[0], d_m=m - pre[1],
                         collapsed=bool((f > 0.95 and m > 0.95) or (f < 0.05 and m < 0.05)))
    if fr.get("alpha") is not None and abs(out["grid"]["removal"] - A.z(fr.get("removal"))) > 1e-12:
        raise SystemExit(f"grid removal differs from the frozen row: {fp}")
    return out


def cells_of(stats, point, key):
    return [float(np.mean([stats[(t, s)][point][key] for s in A.SEEDS])) for t, _ in CELLS]


def verdict(shift_el, shift_ctl, removal_ctl):
    """Pure. Cell lists (five each) -> the registered reading."""
    D = V.boot_paired(list(zip(shift_el, shift_ctl)))
    ratio = abs(float(np.mean(shift_ctl))) / float(np.mean(shift_el))
    rem = V.boot_mean(removal_ctl)
    if E5.covers(D) or ratio >= R_REFUTE:
        name = "E13-refuted"
    elif ratio <= R_CONF:
        name = "E13-confirmed"
    else:
        name = "E13-partial"
    vacuous = name != "E13-refuted" and E5.covers(rem)
    return dict(verdict=name + (" (VACUOUS)" if vacuous else ""), ratio=ratio,
                shift_elicited=float(np.mean(shift_el)), shift_control=float(np.mean(shift_ctl)),
                elicited_minus_control=D, control_removal_grid=rem, vacuous=bool(vacuous))


def progress():
    _, counts = arm_rows()
    for arm, n in counts.items():
        print(f"E13 {arm:9s} {n}/{len(CELLS) * len(A.SEEDS)} units on disk")
    return 0


def main():
    arms, counts = arm_rows()
    if "elicited" not in arms or "balanced" not in arms:
        progress()
        print("incomplete: E13 is not read before the elicited and the balanced arm are complete")
        return 1
    occ, n_tpl = probe_occ()
    stats = {arm: {k: unit_stats(r, d, occ, n_tpl) for k, (r, d) in rows.items()}
             for arm, rows in arms.items()}
    # the three arms edit the same unedited model: baselines must agree
    for arm in stats:
        for k in stats[arm]:
            a, b = stats[arm][k], stats["elicited"][k]
            if abs(a["pre_skew"] - b["pre_skew"]) > 1e-9 or abs(a["pre_f"] - b["pre_f"]) > 1e-6:
                raise SystemExit(f"baseline of {arm} {k} differs from the elicited run's")

    out = dict(script="src/r1_e13.py", registered="R1_NOTES.md section 3 (2026-11-11)",
               arms_complete=sorted(stats), units=counts, by_arm={}, readings={})
    for arm, st in stats.items():
        d = dict(pre=dict(f=float(np.mean([x["pre_f"] for x in st.values()])),
                          m=float(np.mean([x["pre_m"] for x in st.values()]))), points={})
        for p in POINTS:
            xs = [x[p] for x in st.values()]
            d["points"][p] = dict(
                f=float(np.mean([x["f"] for x in xs])), m=float(np.mean([x["m"] for x in xs])),
                shift=V.boot_mean(cells_of(st, p, "shift")),
                d_f=V.boot_mean(cells_of(st, p, "d_f")), d_m=V.boot_mean(cells_of(st, p, "d_m")),
                removal=V.boot_mean(cells_of(st, p, "removal")),
                not_edited=sum(x["alpha"] is None for x in xs),
                collapsed_runs=sum(x["collapsed"] for x in xs),
                alpha_median=(float(np.median([x["alpha"] for x in xs if x["alpha"] is not None]))
                              if any(x["alpha"] is not None for x in xs) else None),
                cells={t: dict(f=float(np.mean([st[(t, s)][p]["f"] for s in A.SEEDS])),
                               m=float(np.mean([st[(t, s)][p]["m"] for s in A.SEEDS])),
                               removal=float(np.mean([st[(t, s)][p]["removal"] for s in A.SEEDS])))
                       for t, _ in CELLS})
        d["cells_pre"] = {t: dict(f=float(np.mean([st[(t, s)]["pre_f"] for s in A.SEEDS])),
                                  m=float(np.mean([st[(t, s)]["pre_m"] for s in A.SEEDS])))
                          for t, _ in CELLS}
        out["by_arm"][arm] = d
    for ctl in ("balanced", "exo"):
        if ctl not in stats:
            continue
        out["readings"][ctl] = {
            p: verdict(cells_of(stats["elicited"], p, "shift"), cells_of(stats[ctl], p, "shift"),
                       cells_of(stats[ctl], "grid", "removal")) for p in POINTS}
    out["verdict"] = out["readings"]["balanced"]["grid"]["verdict"]
    with open(os.path.join(OUT, "e13_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)

    print("mean P(she) on female- / male-stereotyped probe occupations, five occupation cells")
    for arm, d in out["by_arm"].items():
        print(f"\n[{arm}]   unedited {d['pre']['f']:.2f} / {d['pre']['m']:.2f}")
        for p in POINTS:
            x = d["points"][p]
            print(f"  {PNAME[p]:10s} {x['f']:.2f} / {x['m']:.2f}   base-rate shift "
                  f"{V.fmt(x['shift'], 3)}   f-side {x['d_f']['point']:+.2f}  m-side "
                  f"{x['d_m']['point']:+.2f}   removal {V.fmt(x['removal'], 3)}   "
                  f"median alpha {x['alpha_median']}   not edited {x['not_edited']}   "
                  f"collapsed {x['collapsed_runs']}/15")
        g = d["points"]["grid"]["cells"]
        print("  per cell at the grid scale: " + "   ".join(
            f"{t} {d['cells_pre'][t]['f']:.2f}/{d['cells_pre'][t]['m']:.2f} -> "
            f"{g[t]['f']:.2f}/{g[t]['m']:.2f}" for t, _ in CELLS))
    for ctl, rd in out["readings"].items():
        tag = "PRIMARY" if ctl == "balanced" else "secondary"
        for p in POINTS:
            v = rd[p]
            print(f"\n{ctl} ({tag}), {PNAME[p]}: shift elicited {v['shift_elicited']:+.3f}, "
                  f"control {v['shift_control']:+.3f}, ratio {v['ratio']:.2f}, elicited - "
                  f"control {V.fmt(v['elicited_minus_control'], 3)} -> {v['verdict']}"
                  + ("" if p == "grid" else "   (reported, not the verdict)"))
    print(f"\nREGISTERED READING (balanced, fixed grid): {out['verdict']}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'e13_verdict.json'), REPO)}")
    return 0


def selftest():
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(f"  {'ok ' if cond else 'FAIL'} {name}")

    el = [0.30, 0.35, 0.40, 0.25, 0.33]
    rem = [0.30, 0.25, 0.20, 0.10, 0.35]
    v = verdict(el, [0.02, -0.01, 0.03, 0.00, 0.01], rem)
    check("shift gone under balance -> confirmed", v["verdict"] == "E13-confirmed")
    v = verdict(el, [0.15, 0.17, 0.20, 0.12, 0.16], rem)
    check("half the shift remains -> partial", v["verdict"] == "E13-partial")
    v = verdict(el, [0.29, 0.33, 0.37, 0.22, 0.31], rem)
    check("shift unchanged -> refuted", v["verdict"] == "E13-refuted")
    v = verdict(el, [0.02, 0.70, -0.40, 0.00, 0.01], rem)
    check("inconsistent cells (interval includes 0) -> refuted", v["verdict"] == "E13-refuted")
    v = verdict(el, [-0.30, -0.33, -0.36, -0.25, -0.31], rem)
    check("equal shift the other way is not 'confirmed' (ratio uses |shift|)",
          v["verdict"] == "E13-refuted" and v["ratio"] > 0.9)
    v = verdict(el, [0.02, -0.01, 0.03, 0.00, 0.01], [0.01, -0.02, 0.00, 0.02, -0.01])
    check("no shift but no removal either -> VACUOUS", v["verdict"] == "E13-confirmed (VACUOUS)")
    print(f"selftest: {sum(ok)}/{len(ok)}")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]() or 0)
