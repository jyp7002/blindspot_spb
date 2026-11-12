#!/usr/bin/env python3
"""R1 FRONT8 -- the 1% edit on the eight cells of the comparison table.

    python3 src/r1_front8.py              reading (refuses an incomplete panel)
    python3 src/r1_front8.py progress
    python3 src/r1_front8.py selftest

Registered in R1_NOTES.md section 3 (entry of 2026-11-12) before the first unit of
configs/r1/front8.yaml. The comparison table of the paper (edit / steering /
SentenceDebias / DPO on eight cells) was run with the DENSE sign-only edit; the
abstract, the deployment table and the conclusion speak of the 1% patch. This
script puts the deployed edit (C-ref) on the same eight cells and under the same
selection space, and decides which sentence the paper may keep.

QUANTITY. Per run: removal of C-ref at the best in-budget of the four fixed
scales {2, 4, 8, 16} (none in budget -> not edited -> 0), i.e. the `frozen` field
of the run_r1.py row. This is the selection space of every method in that table.
Cell = mean over seeds 0-2; eight cells in the order of results/v12/frontier_v12.json;
10,000-resample percentile bootstrap over cells (v10_common), paired for differences.

SOURCES. 1% edit: occupation cells from the C-ref rows of results/r1/E5 (replays
of v12astar), CrowS-socioeconomic cells from results/r1/FRONT8. Comparators are
the stored published cell values, not re-run: dense edit, steering and
SentenceDebias from results/v12/frontier_v12.json (table_a); DPO from the
corrected measurement, results/r1/E7b/e7b.json (published selection space).

READING (fixed before the FRONT8 units; see the notes for what was already known):
    d_steer = 1% edit - steering,   d_sent = 1% edit - SentenceDebias
    FRONT8-tie    both intervals include 0
                  -> "the 1% patch removes as much as steering and projection
                     within the uncertainty" may be said of the patch itself
    FRONT8-below  at least one interval lies entirely below 0, none above
                  -> the tie is stated for the dense sign-only edit only; the
                     paper says the 1% patch removes less, with the numbers, and
                     the conclusion sentence is rewritten
    FRONT8-above  at least one interval entirely above 0, none below -> stated so
    FRONT8-mixed  one below and one above -> both stated, no summary claim
Reported, never the verdict: 1% - dense (what sparsity costs on these cells),
1% - DPO, the same at alpha*, medians of the chosen configurations, the share of
the 96 grid configurations in budget, baseline agreement with the stored panel.
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
OUT = os.path.join(RESULTS, "r1", "FRONT8")
ARM = "C-ref"
TARGETS = ("gemma", "llama", "qwen", "phi")
AXES = ("occ_gender", "crows_socioeconomic")
CELLS = [(t, a) for t in TARGETS for a in AXES]
SEEDS = (0, 1, 2)
GRID = (2.0, 4.0, 8.0, 16.0)
PUB_DPO = "published selection (merge 0.25-1, <20 pairs -> 0)"
# fixed-grid C-ref cell values printed in the scale-up table of the submitted paper
# (another runner, panel v11); shown beside the new values, not used.
SUBMITTED_SCALEUP = {"gemma|crows_socioeconomic": 0.084, "llama|crows_socioeconomic": 0.011,
                     "phi|crows_socioeconomic": 0.029}


class Partial(Exception):
    pass


def ck(c):
    return f"{c[0]}|{c[1]}"


def load_runs():
    """(target, axis, seed) -> C-ref row; occupation from E5, CrowS from FRONT8."""
    e5, _ = E5.load()
    f8, _ = E5.load(OUT)
    runs = {}
    for t, a in CELLS:
        src = e5 if a == "occ_gender" else f8
        for s in SEEDS:
            r = src.get((t, a, s, ARM))
            if r is not None:
                runs[(t, a, s)] = r
    return runs


def run_stats(r):
    """Pure: one run_r1.py row -> what the comparison table needs."""
    rung = {float(x["alpha"]): x for x in r["rungs"]}
    miss = [a for a in GRID if a not in rung]
    if miss:
        raise SystemExit(f"fixed-grid rung(s) {miss} missing in {r['target']}|{r['axis']}|s{r['seed']}")
    fr = r.get("frozen") or {}
    a = fr.get("alpha")
    ok = [bool(rung[g]["eval"]["ok"]) for g in GRID]
    out = dict(alpha=a, removal=0.0, dmmlu_items=None, ppl_ratio=None, n_in_budget=int(sum(ok)),
               pre_skew=float(r["pre_skew"]), star=E5.adaptive(r),
               star_deployed=(r.get("adaptive") or {}).get("alpha_star") is not None,
               star_eval_ok=bool((r.get("adaptive") or {}).get("eval_ok")))
    if a is not None:
        ev = rung[float(a)]["eval"]
        if not ev["ok"]:
            raise SystemExit(f"frozen scale is out of budget in {r['target']}|{r['axis']}|s{r['seed']}")
        best = max(A.z(rung[g]["eval"]["removal"]) for g, o in zip(GRID, ok) if o)
        if abs(best - A.z(fr.get("removal"))) > 1e-12 or abs(A.z(ev["removal"]) - best) > 1e-12:
            raise SystemExit(f"frozen row is not the in-budget argmax in {r['target']}|{r['axis']}|s{r['seed']}")
        out.update(removal=A.z(fr.get("removal")), dmmlu_items=int(ev["dmmlu_items"]),
                   ppl_ratio=float(ev["ppl_ratio"]))
    elif any(ok):
        raise SystemExit(f"a grid scale is in budget but frozen is empty: {r['target']}|{r['axis']}|s{r['seed']}")
    return out


def reading(d_steer, d_sent):
    below = d_steer["hi"] < 0 or d_sent["hi"] < 0
    above = d_steer["lo"] > 0 or d_sent["lo"] > 0
    if below and above:
        return "FRONT8-mixed"
    if below:
        return "FRONT8-below"
    if above:
        return "FRONT8-above"
    return "FRONT8-tie"


def analyse(runs, comp):
    """Pure. runs: (t, a, s) -> row. comp: name -> {"t|a": cell value}."""
    need = [(t, a, s) for t, a in CELLS for s in SEEDS]
    have = sum(k in runs for k in need)
    if have < len(need):
        raise Partial(f"{have}/{len(need)} runs on disk")
    st = {k: run_stats(runs[k]) for k in need}
    cell = {c: float(np.mean([st[(c[0], c[1], s)]["removal"] for s in SEEDS])) for c in CELLS}
    star = {c: float(np.mean([st[(c[0], c[1], s)]["star"] for s in SEEDS])) for c in CELLS}
    one = [cell[c] for c in CELLS]
    diffs = {}
    for name in ("steering", "sentdebias", "dense_edit", "dpo"):
        diffs[name] = V.boot_paired([(cell[c], A.z(comp[name][ck(c)])) for c in CELLS])
    chosen = [v for v in st.values() if v["alpha"] is not None]
    out = dict(
        script="src/r1_front8.py", arm=ARM, cells_order=[ck(c) for c in CELLS],
        reading=reading(diffs["steering"], diffs["sentdebias"]),
        one_percent_edit=dict(
            removal=V.boot_mean(one), cells={ck(c): cell[c] for c in CELLS},
            dmmlu_items_median=(float(np.median([v["dmmlu_items"] for v in chosen])) if chosen else None),
            ppl_ratio_median=(float(np.median([v["ppl_ratio"] for v in chosen])) if chosen else None),
            grid_configs_in_budget=f"{sum(v['n_in_budget'] for v in st.values())}/{len(GRID) * len(st)}",
            budget_pass_rate=sum(v["n_in_budget"] for v in st.values()) / (len(GRID) * len(st)),
            runs_not_edited=sum(v["alpha"] is None for v in st.values()),
            chosen_scales={str(g): sum(v["alpha"] == g for v in st.values()) for g in GRID}),
        comparators={name: dict(removal=V.boot_mean([A.z(comp[name][ck(c)]) for c in CELLS]),
                                cells={ck(c): A.z(comp[name][ck(c)]) for c in CELLS})
                     for name in ("dense_edit", "steering", "sentdebias", "dpo")},
        one_percent_minus=diffs,
        at_alpha_star=dict(
            removal=V.boot_mean([star[c] for c in CELLS]), cells={ck(c): star[c] for c in CELLS},
            runs_deployed=sum(v["star_deployed"] for v in st.values()),
            runs_eval_ok=sum(v["star_eval_ok"] for v in st.values()), runs=len(st)),
        submitted_scaleup_fixed_grid={k: dict(submitted=v, here=cell[tuple(k.split("|"))])
                                      for k, v in SUBMITTED_SCALEUP.items()},
        pre_skew={f"{t}|{a}|s{s}": st[(t, a, s)]["pre_skew"] for t, a, s in need})
    return out


def comparators():
    fr = json.load(open(os.path.join(RESULTS, "v12", "frontier_v12.json")))
    ta = fr["table_a"]
    if list(ta.keys()) != [ck(c) for c in CELLS]:
        raise SystemExit("cell order differs from results/v12/frontier_v12.json")
    comp = dict(dense_edit={k: v["edit"] for k, v in ta.items()},
                steering={k: v["steering"] for k, v in ta.items()},
                sentdebias={k: v["sentdebias"] for k, v in ta.items()})
    # gate: this script's bootstrap reproduces the published pooled differences
    for name, key in (("steering", "edit - steering"), ("sentdebias", "edit - SentenceDebias")):
        d = V.boot_paired([(A.z(comp["dense_edit"][ck(c)]), A.z(comp[name][ck(c)])) for c in CELLS])
        p = fr["pooled"][key]
        if max(abs(d[x] - p[x]) for x in ("point", "lo", "hi")) > 1e-9:
            raise SystemExit(f"published pooled value not reproduced: {key}")
    e7 = json.load(open(os.path.join(RESULTS, "r1", "E7b", "e7b.json")))
    comp["dpo"] = dict(e7["pub"]["selection"][PUB_DPO]["cells"])
    return comp


def stored_baselines():
    """(t, a, s) -> pre_skew recorded by the stored comparison panel (v12frontier)."""
    out = {}
    for r in E5.jl(os.path.join(RESULTS, "v12frontier", "alpha_trace.jsonl")):
        if r.get("pre_skew") is not None:
            out.setdefault((r["target"], r["axis"], r["seed"]), float(r["pre_skew"]))
    return out


def fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def progress():
    runs = load_runs()
    for a in AXES:
        n = sum((t, a, s) in runs for t in TARGETS for s in SEEDS)
        src = "results/r1/E5" if a == "occ_gender" else "results/r1/FRONT8"
        print(f"FRONT8 {a:20s} {n}/{len(TARGETS) * len(SEEDS)} runs on disk ({src})")
    return 0


def main():
    try:
        out = analyse(load_runs(), comparators())
    except Partial as e:
        progress()
        print(f"NO PARTIAL READS: {e}")
        return 1
    sb = stored_baselines()
    dif = [abs(v - sb[tuple(k.rsplit("|s", 1)[0].split("|")) + (int(k.rsplit("|s", 1)[1]),)])
           for k, v in out["pre_skew"].items()
           if tuple(k.rsplit("|s", 1)[0].split("|")) + (int(k.rsplit("|s", 1)[1]),) in sb]
    out["baseline_vs_stored_panel"] = dict(runs_compared=len(dif),
                                           max_abs_diff=(max(dif) if dif else None))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "front8.json"), "w") as f:
        json.dump(out, f, indent=1)
    o = out["one_percent_edit"]
    print(f"1% edit (C-ref), fixed grid, 8 cells: {fmt(o['removal'])}")
    print(f"  median dMMLU {o['dmmlu_items_median']}, median ppl ratio {o['ppl_ratio_median']:.3f}, "
          f"grid configurations in budget {o['grid_configs_in_budget']} "
          f"({100 * o['budget_pass_rate']:.1f}%), runs not edited {o['runs_not_edited']}, "
          f"chosen scales {o['chosen_scales']}")
    print("  per cell (seed means):   1% edit | dense edit | steering | SentenceDebias | DPO")
    for c in CELLS:
        k = ck(c)
        cc = out["comparators"]
        print(f"    {k:26s} {o['cells'][k]:+.3f} | {cc['dense_edit']['cells'][k]:+.3f} | "
              f"{cc['steering']['cells'][k]:+.3f} | {cc['sentdebias']['cells'][k]:+.3f} | "
              f"{cc['dpo']['cells'][k]:+.3f}")
    for name, label in (("steering", "steering"), ("sentdebias", "SentenceDebias"),
                        ("dense_edit", "dense edit"), ("dpo", "DPO")):
        print(f"  1% edit - {label:15s} {fmt(out['one_percent_minus'][name])}   "
              f"(comparator {fmt(out['comparators'][name]['removal'])})")
    s = out["at_alpha_star"]
    print(f"  at alpha*: {fmt(s['removal'])}; deployed {s['runs_deployed']}/{s['runs']}, "
          f"eval budget passed {s['runs_eval_ok']}/{s['runs']}")
    for k, v in out["submitted_scaleup_fixed_grid"].items():
        print(f"  submitted scale-up table {k}: {v['submitted']:+.3f}; here {v['here']:+.3f}")
    b = out["baseline_vs_stored_panel"]
    print(f"  pre-skew vs the stored comparison panel: {b['runs_compared']} runs compared, "
          f"max |diff| {b['max_abs_diff']}")
    print(f"\nREADING: {out['reading']}")
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'front8.json'), REPO)}")
    return 0


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    def row(t, a, s, rem, in_budget=(True, True, True, True), star=0.3):
        rungs = [dict(alpha=g, eval=dict(ok=o, removal=r, dmmlu_items=1, ppl_ratio=1.02,
                                         post_skew=0.1))
                 for g, o, r in zip(GRID, in_budget, rem)]
        okr = [r for r, o in zip(rem, in_budget) if o]
        fr = dict(alpha=None, removal=None)
        if okr:
            b = max(okr)
            fr = dict(alpha=GRID[[i for i, (r, o) in enumerate(zip(rem, in_budget)) if o and r == b][0]],
                      removal=b)
        return dict(target=t, axis=a, seed=s, arm=ARM, pre_skew=0.5, rungs=rungs, frozen=fr,
                    adaptive=dict(alpha_star=20.0, removal=star, eval_ok=True))

    def panel(val):
        return {(t, a, s): row(t, a, s, (0.0, val(t, a) / 2, val(t, a), val(t, a) + 0.5),
                               in_budget=(True, True, True, False))
                for t, a in CELLS for s in SEEDS}

    base = {ck(c): 0.30 + 0.05 * i for i, c in enumerate(CELLS)}
    comp = dict(dense_edit=dict(base), steering=dict(base), sentdebias=dict(base),
                dpo={k: 0.0 for k in base})
    runs = panel(lambda t, a: base[f"{t}|{a}"])
    o = analyse(runs, comp)
    check("equal cells -> FRONT8-tie, differences are exactly 0",
          o["reading"] == "FRONT8-tie" and abs(o["one_percent_minus"]["steering"]["point"]) < 1e-12)
    check("out-of-budget scale is never chosen (16 has the largest removal)",
          o["one_percent_edit"]["chosen_scales"] == {"2.0": 0, "4.0": 0, "8.0": 24, "16.0": 0})
    check("share in budget counts all 96 grid configurations",
          o["one_percent_edit"]["grid_configs_in_budget"] == "72/96")
    o = analyse(panel(lambda t, a: base[f"{t}|{a}"] - 0.2), comp)
    check("uniformly lower -> FRONT8-below", o["reading"] == "FRONT8-below")
    o = analyse(panel(lambda t, a: base[f"{t}|{a}"] + 0.2), comp)
    check("uniformly higher -> FRONT8-above", o["reading"] == "FRONT8-above")
    comp2 = dict(comp, steering={k: v + 0.2 for k, v in base.items()},
                 sentdebias={k: v - 0.2 for k, v in base.items()})
    check("below one, above the other -> FRONT8-mixed", analyse(runs, comp2)["reading"] == "FRONT8-mixed")
    r2 = dict(runs)
    k0 = (CELLS[0][0], CELLS[0][1], 0)
    r2[k0] = row(*k0, (0.1, 0.2, 0.3, 0.4), in_budget=(False, False, False, False))
    o = analyse(r2, comp)
    check("a run with nothing in budget is not edited and scores 0",
          o["one_percent_edit"]["runs_not_edited"] == 1
          and abs(o["one_percent_edit"]["cells"][ck(CELLS[0])] - 2 * base[ck(CELLS[0])] / 3) < 1e-12)
    r3 = dict(runs)
    del r3[k0]
    try:
        analyse(r3, comp)
        check("an incomplete panel is refused", False)
    except Partial:
        check("an incomplete panel is refused", True)
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]())
