#!/usr/bin/env python3
"""R1 E3a -- the selection effect under alternative definitions of removal (zero GPU).

    python3 src/r1_e3a.py            -> results/r1/E3/e3a_removal_definitions.json

WHY. The paper never states its removal formula. The code uses
    (ii)  R = |S_pre| - |S_post|                      (colab_t2t4.bias_reduction)
so an edit that pushes the skew through zero is CHARGED for the overshoot. 5PJm
asks how opposite-direction skew is handled. E3a (registered as a robustness
report, no fork) re-reads the same stored measurements under
    (i)   R = s * (S_pre - S_post),  s = sign(S_pre)   signed movement toward and
                                                       past zero
    (iii) R = (|S_pre| - |S_post|) / |S_pre|           relative
and, labelled POST-HOC because the plan does not list it,
    (iv)  R = |S_pre| - max(0, s * S_post)             overshoot neither rewarded
                                                       nor charged (capped at |S_pre|)
On the BBQ axes a negative skew means the model prefers the "unknown" answer in
an ambiguous context, which is the answer BBQ scores as correct there; (iv) is
the reading under which that is not counted as bias.

SCOPE OF THIS FILE. Only panels that stored signed pre and post skew can be
re-read: the alpha* panel v12astar (C-ref = variant s0.99, C-a = C-a@0.01; the
frozen grid {2,4,8,16} and the alpha* point, evaluation split). The grid
extension (v10ext/v11ext) stored |.|-based removal only, and the other five DEC
conditions stored no post skew at all; both are covered once the R1 panels with
per-rung signed skew are complete (results/r1/E4, E5).

RULES, unchanged from the published analysis: alpha* never reads bias, so the
alpha* point is simply re-scored; under the frozen grid the in-budget argmax is
taken over the definition being reported (the rule selects on the reported
quantity). Integer gate (v9_gate), nan/none -> 0, cell = seed mean, paired cell
bootstrap in the registered cell order (v10_common).

SANITY GATE. Definition (ii) must reproduce the published v12astar numbers:
frozen +0.285 [+0.170, +0.411], alpha* +0.073 [-0.036, +0.213].
"""
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v9_gate            # noqa: E402
import v10_common as V    # noqa: E402
import v12_astar as A     # noqa: E402   (CELLS, SEEDS, ARMS, registered order)

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E3")
PANEL = "v12astar"
DEFS = collections.OrderedDict([
    ("ii_abs", "|pre| - |post|  (the code's definition; published)"),
    ("i_signed", "sign(pre) * (pre - post)  (overshoot counted as removal)"),
    ("iii_relative", "(|pre| - |post|) / |pre|"),
    ("iv_capped", "|pre| - max(0, sign(pre) * post)  (POST-HOC; overshoot neutral)"),
])


def removal(defn, pre, post):
    s = 1.0 if pre >= 0 else -1.0
    if defn == "ii_abs":
        return abs(pre) - abs(post)
    if defn == "i_signed":
        return s * (pre - post)
    if defn == "iii_relative":
        return (abs(pre) - abs(post)) / abs(pre) if pre != 0 else float("nan")
    if defn == "iv_capped":
        return abs(pre) - max(0.0, s * post)
    raise ValueError(defn)


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


def load():
    rows = A.load(PANEL)                               # (t, a, s, variant) -> row, last wins
    trace = {}
    for r in jl(os.path.join(RESULTS, PANEL, "alpha_trace.jsonl")):
        if r.get("split") == "eval" and r.get("rule") == "frozen":
            trace[(r["target"], r["axis"], r["seed"], r["variant"], float(r["alpha"]))] = r
    return rows, trace


def per_run(rows, trace, defn):
    """-> {(t,a,s,variant): dict(frozen=..., adaptive=...)} under one definition."""
    out = {}
    for (t, a, s, vn), r in rows.items():
        pre = r["pre_skew"]
        best = float("nan")
        for al in (2.0, 4.0, 8.0, 16.0):
            tr = trace[(t, a, s, vn, al)]
            ok, _mode = v9_gate.collateral_ok_row(tr)
            if ok:
                v = removal(defn, tr["pre_skew"], tr["post_skew"])
                best = v if best != best else max(best, v)
        ad = r.get("adaptive") or {}
        if ad.get("alpha_star") is None:
            adv = 0.0                                   # status 'none': not deployed
        else:
            adv = removal(defn, pre, ad["post_skew"])
        out[(t, a, s, vn)] = dict(frozen=best, adaptive=adv,
                                  flipped=(ad.get("post_skew") is not None
                                           and pre * ad["post_skew"] < 0))
    return out


def cell_pairs(pr, field):
    pairs = []
    for t, a in A.CELLS:
        m = {vn: [A.z(pr[(t, a, s, vn)][field]) for s in A.SEEDS] for vn in A.ARMS}
        pairs.append((float(np.mean(m["s0.99"])), float(np.mean(m["C-a@0.01"]))))
    return pairs


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, trace = load()
    miss = A.missing(rows)
    if miss:
        raise SystemExit(f"{PANEL} incomplete ({len(miss)} rows missing)")
    out = dict(script="src/r1_e3a.py", panel=PANEL, definitions=dict(DEFS),
               note="(iv) is post-hoc; (i) and (iii) are the registered E3a variants; "
                    "(ii) is the published definition", by_definition={})
    for defn in DEFS:
        pr = per_run(rows, trace, defn)
        F = V.boot_paired(cell_pairs(pr, "frozen"))
        E = V.boot_paired(cell_pairs(pr, "adaptive"))
        cells = {}
        for (t, a), (fr, fa), (er, ea) in zip(A.CELLS, cell_pairs(pr, "frozen"),
                                              cell_pairs(pr, "adaptive")):
            cells[f"{t}|{a}"] = dict(
                frozen_ref=fr, frozen_a=fa, frozen_gap=fr - fa,
                astar_ref=er, astar_a=ea, astar_gap=er - ea,
                astar_sign_flips=dict(
                    ref=sum(pr[(t, a, s, "s0.99")]["flipped"] for s in A.SEEDS),
                    a=sum(pr[(t, a, s, "C-a@0.01")]["flipped"] for s in A.SEEDS)))
        out["by_definition"][defn] = dict(
            frozen=F, astar=E, branch_v12E=A.branch(E, F),
            ratio_astar_over_frozen=(E["point"] / F["point"] if F["point"] else None),
            arm_means=dict(
                frozen_ref=float(np.mean([c["frozen_ref"] for c in cells.values()])),
                frozen_a=float(np.mean([c["frozen_a"] for c in cells.values()])),
                astar_ref=float(np.mean([c["astar_ref"] for c in cells.values()])),
                astar_a=float(np.mean([c["astar_a"] for c in cells.values()]))),
            cells_positive_astar=sum(c["astar_gap"] > 0 for c in cells.values()),
            cells=cells)

    ref = out["by_definition"]["ii_abs"]
    pub = json.load(open(os.path.join(RESULTS, "v12", "astar_verdict.json")))
    gate = all(abs(ref[k][f] - pub[p][f]) < 1e-12
               for k, p in (("frozen", "frozen"), ("astar", "adaptive"))
               for f in ("point", "lo", "hi"))
    out["sanity_gate"] = dict(reproduced=gate, published_frozen=pub["frozen"],
                              published_astar=pub["adaptive"])
    print(f"sanity gate (definition ii reproduces results/v12/astar_verdict.json): {gate}")
    if not gate:
        raise SystemExit("cannot reproduce the published alpha* numbers; not proceeding")

    flips = collections.Counter()
    pr = per_run(rows, trace, "ii_abs")
    for (t, a, s, vn), d in pr.items():
        if d["flipped"]:
            flips[(f"{t}|{a}", vn)] += 1
    out["sign_flips_at_astar"] = {f"{c} {vn}": n for (c, vn), n in sorted(flips.items())}
    print(f"runs whose skew changes sign at alpha*: "
          f"C-ref {sum(n for (c, vn), n in flips.items() if vn == 's0.99')}/30, "
          f"C-a {sum(n for (c, vn), n in flips.items() if vn == 'C-a@0.01')}/30")
    print("\nDelta_selection = C-ref - C-a, 10 registered cells, cell bootstrap")
    for defn, d in out["by_definition"].items():
        print(f"\n  {defn:13s} {DEFS[defn]}")
        print(f"    frozen grid  {V.fmt(d['frozen'], 3)}")
        print(f"    alpha*       {V.fmt(d['astar'], 3)}   positive in "
              f"{d['cells_positive_astar']}/10 cells   -> v12.E branch {d['branch_v12E']}")
        m = d["arm_means"]
        print(f"    arm means    frozen: C-ref {m['frozen_ref']:+.3f}, C-a {m['frozen_a']:+.3f}"
              f"   alpha*: C-ref {m['astar_ref']:+.3f}, C-a {m['astar_a']:+.3f}")
    print("\n  per cell at alpha*, C-ref - C-a:   (ii) published | (i) signed | (iv) capped"
          "   [sign flips C-ref/C-a of 3]")
    for t, a in A.CELLS:
        k = f"{t}|{a}"
        c2, c1, c4 = (out["by_definition"][x]["cells"][k] for x in
                      ("ii_abs", "i_signed", "iv_capped"))
        f = c2["astar_sign_flips"]
        print(f"    {k:26s} {c2['astar_gap']:+.3f} | {c1['astar_gap']:+.3f} | "
              f"{c4['astar_gap']:+.3f}   [{f['ref']}/{f['a']}]")
    with open(os.path.join(OUT, "e3a_removal_definitions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.relpath(os.path.join(OUT, 'e3a_removal_definitions.json'), REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
