#!/usr/bin/env python3
"""R1 E3c on the eight-cell comparison -- select on one half, report on the other.

    python3 src/r1_e3c_front.py            reading (refuses an incomplete replay)
    python3 src/r1_e3c_front.py progress
    python3 src/r1_e3c_front.py selftest

Same rule as src/r1_e3c.py (R1_NOTES.md section 3, entries of 2026-11-12), applied
to the comparison of the edit with steering, SentenceDebias and DPO: every method
chooses the best in-budget of four configurations on the probe items it reports
(pe22 Q3). Here the configuration is chosen on one half of the probe items
(sha256(item_id) mod 2, audit/r1_e3c_split.json) and the removal read on the
other half, in both directions, averaged.

SOURCES of per-item likelihoods
  steering, SentenceDebias, dense edit   results/r1/E3c  (run_r1_front_ll.py: the
        published configurations rebuilt and the probe evaluated again; the budget
        decision of each configuration is the stored one of results/v12frontier)
  DPO, lower learning rates              results/r1/E7b  (arms lr1e-5, lr5e-5;
        four configurations = two learning rates x merge 1, 2)
  1% edit (C-ref)                        results/r1/E5 (occupation), results/r1/FRONT8
        (CrowS socioeconomic); grid {2, 4, 8, 16}

GATES. A replayed run is used only if its unedited skew and the skew of each of
its configurations equal the stored values to 1e-6; otherwise this script stops
and lists the runs (deciding what to do then is an amendment). For every method
the full-set selection recomputed here must equal the stored selection.

AMENDMENT (R1_NOTES.md section 3, 2026-11-12 ~12:05Z; after the first replay unit,
before any split-half value was computed). SentenceDebias fits its subspace with
an unseeded randomized PCA (scripts/sd_determinism.py), so a configuration can
differ between processes: in the first unit one of four did, by 0.002 in skew.
For SentenceDebias the tolerance is therefore 0.01 in skew per configuration and
0.01 in the full-set removal; within it the replayed likelihoods are used as they
are, and the removal recomputed from the replay is printed beside the stored one.
Steering and the dense edit keep 1e-6.

REPORTED (no verdict is registered): per method, removal with the configuration
chosen on all items (published), on the other half, on the same half, and the
optimism; for the paired differences of the dense edit and of the 1% edit with
each other method, the interval under both rules and whether its status changes.
Cell = seed mean; eight cells in the order of results/v12/frontier_v12.json;
bootstrap over cells (v10_common). A method that was not run in a cell (too few
pairs) or has no configuration in budget scores 0 there, as published.
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
import r1_e13 as E13      # noqa: E402
import r1_e3c as S3       # noqa: E402
import r1_front8 as F8    # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E3c")
CELLS, SEEDS = F8.CELLS, F8.SEEDS
REPLAYED = ("steering", "sentdebias", "edit")
METHODS = ("edit", "edit_1pct", "steering", "sentdebias", "dpo")
NAME = dict(edit="dense edit", edit_1pct="1% edit", steering="steering",
            sentdebias="SentenceDebias", dpo="DPO (lower lr)")
DPO_ARMS = ("lr1e-5", "lr5e-5")
TOL = dict(steering=1e-6, edit=1e-6, sentdebias=1e-2)      # skew, per configuration
TOL_FULL = dict(steering=1e-5, edit=1e-5, sentdebias=1e-2)  # full-set removal


class Partial(Exception):
    pass


def select_report(pre, post, ok):
    """Pure. pre: {None,0,1: skew}; post: [ {None,0,1: skew} per configuration ];
    ok: [bool]. -> dict(full, split, same, i_full, i_h)."""
    def rem(which, i):
        return abs(pre[which]) - abs(post[i][which])

    def best(which):
        b, bi = None, None
        for i in range(len(post)):
            if ok[i] and (b is None or rem(which, i) > b):
                b, bi = rem(which, i), i
        return bi
    if not any(ok):
        return dict(full=0.0, split=0.0, same=0.0, i_full=None, i_h=(None, None))
    i_f, i0, i1 = best(None), best(0), best(1)
    return dict(full=rem(None, i_f), split=0.5 * (rem(1, i0) + rem(0, i1)),
                same=0.5 * (rem(0, i0) + rem(1, i1)), i_full=i_f, i_h=(i0, i1))


ZERO = dict(full=0.0, split=0.0, same=0.0, i_full=None, i_h=(None, None), stored=0.0)


def load_all():
    split = json.load(open(S3.SPLIT_FP))
    occ, n_tpl = E13.probe_occ()
    fns = {a: S3.skew_fn(a, split[a]["half"], occ, n_tpl) for a in F8.AXES}
    three = lambda fn, ll: {w: fn(ll, w) for w in (None, 0, 1)}   # noqa: E731
    runs, missing, bad = {}, [], []

    # ---- replayed methods
    rep = {}
    for r in E5.jl(os.path.join(OUT, "replay.jsonl")):
        rep[(r["target"], r["axis"], r["seed"], r["method"])] = r
    stored = {}
    for r in E5.jl(os.path.join(RESULTS, "v12frontier", "removal.jsonl")):
        if r.get("row") == "select":
            stored[(r["target"], r["axis"], r["seed"], r["method"])] = r
    for m in REPLAYED:
        for t, a in CELLS:
            for s in SEEDS:
                k = (t, a, s, m)
                r = rep.get(k)
                if r is None:
                    missing.append(k)
                    continue
                st = stored[k]
                if r.get("skipped"):
                    if st.get("removal") is not None:
                        bad.append((k, "skipped here but run in the stored panel"))
                    runs[k] = dict(ZERO)
                    continue
                if (not r["all_configs_stored"] or r["max_abs_diff"] > TOL[m]
                        or r["pre_abs_diff"] > 1e-6):
                    bad.append((k, f"max diff {r['max_abs_diff']}, pre diff {r['pre_abs_diff']}"))
                    continue
                zf = np.load(os.path.join(OUT, "probe_ll", f"{t}__{a}__s{s}__{m}.npz"))
                pre = three(fns[a], zf["pre_ll"])
                post = [three(fns[a], zf["ll"][i]) for i in range(len(r["configs"]))]
                ok = [bool(c["collateral_ok"]) for c in r["configs"]]
                v = select_report(pre, post, ok)
                if abs(v["full"] - A.z(st.get("removal"))) > TOL_FULL[m]:
                    bad.append((k, f"full-set selection {v['full']:+.6f} != stored {st.get('removal')}"))
                v["stored"] = A.z(st.get("removal"))
                v["replay_max_abs_diff"] = float(r["max_abs_diff"])
                runs[k] = v

    # ---- DPO, lower learning rates
    e7 = {}
    for r in E5.jl(os.path.join(RESULTS, "r1", "E7b", "removal.jsonl")):
        e7[(r["target"], r["axis"], r["seed"], r["arm"])] = r
    for t, a in CELLS:
        for s in SEEDS:
            k = (t, a, s, "dpo")
            pre, post, ok = None, [], []
            for arm in DPO_ARMS:
                r = e7.get((t, a, s, arm))
                if r is None:
                    missing.append((t, a, s, arm))
                    continue
                if r.get("skipped"):
                    continue
                zf = np.load(os.path.join(RESULTS, "r1", "E7b", "probe_ll",
                                          f"{t}__{a}__s{s}__{r['designer']}__{arm}.npz"))
                pre = three(fns[a], zf["pre_ll"])
                merges = [float(x) for x in zf["merge"]]
                for p in r["points"]:
                    i = [j for j, x in enumerate(merges) if x == float(p["merge"])]
                    if len(i) != 1:
                        raise SystemExit(f"merge {p['merge']} not stored exactly once: {k} {arm}")
                    q = three(fns[a], zf["ll"][i[0]])
                    if abs(q[None] - p["post_skew"]) > 1e-5:
                        raise SystemExit(f"recomputed DPO skew != recorded: {k} {arm} {p['merge']}")
                    post.append(q)
                    ok.append(bool(p["ok"]))
            runs[k] = select_report(pre, post, ok) if post else dict(ZERO)
            runs[k]["stored"] = runs[k]["full"]

    # ---- 1% edit
    one = F8.load_runs()
    for t, a in CELLS:
        for s in SEEDS:
            k = (t, a, s, "edit_1pct")
            r = one.get((t, a, s))
            if r is None:
                missing.append(k)
                continue
            pdir = E5.OUT if a == "occ_gender" else F8.OUT
            zf = np.load(os.path.join(pdir, "probe_ll", f"{t}__{a}__s{s}__{r['designer']}__C-ref.npz"))
            alphas = [float(x) for x in zf["alphas"]]
            rung = {float(x["alpha"]): x for x in r["rungs"]}
            pre = three(fns[a], zf["pre_ll"])
            post, ok = [], []
            for g in F8.GRID:
                i = [j for j, x in enumerate(alphas) if x == g]
                if len(i) != 1:
                    raise SystemExit(f"scale {g} not stored exactly once: {k}")
                post.append(three(fns[a], zf["ll"][i[0]]))
                ok.append(bool(rung[g]["eval"]["ok"]))
            v = select_report(pre, post, ok)
            if abs(v["full"] - A.z((r.get("frozen") or {}).get("removal"))) > 1e-5:
                bad.append((k, "full-set selection != frozen row"))
            v["stored"] = v["full"]
            runs[k] = v
    return runs, missing, bad


def cell_vals(runs, m, key):
    return [float(np.mean([runs[(t, a, s, m)][key] for s in SEEDS])) for t, a in CELLS]


def analyse(runs):
    """Pure given complete `runs`."""
    out = dict(script="src/r1_e3c_front.py", methods={}, contrasts={})
    for m in METHODS:
        full, split, same = (cell_vals(runs, m, k) for k in ("full", "split", "same"))
        ks = [k for k in runs if k[3] == m and runs[k]["i_full"] is not None]
        stored = cell_vals(runs, m, "stored")
        out["methods"][m] = dict(
            stored=V.boot_mean(stored),
            max_abs_cell_diff_replay_vs_stored=float(max(abs(a - b) for a, b in zip(full, stored))),
            runs_not_bit_exact=sum(runs[k].get("replay_max_abs_diff", 0.0) > 1e-6
                                   for k in runs if k[3] == m),
            full=V.boot_mean(full), split=V.boot_mean(split), same=V.boot_mean(same),
            optimism=V.boot_paired(list(zip(same, split))),
            halves_disagree=f"{sum(runs[k]['i_h'][0] != runs[k]['i_h'][1] for k in ks)}/{len(ks)}")
    for x in ("edit", "edit_1pct"):
        for y in ("steering", "sentdebias", "dpo"):
            rec = {}
            for key in ("full", "split"):
                rec[key] = V.boot_paired(list(zip(cell_vals(runs, x, key), cell_vals(runs, y, key))))
                rec[key]["excludes_zero"] = not E5.covers(rec[key])
            rec["status_changes"] = rec["full"]["excludes_zero"] != rec["split"]["excludes_zero"]
            out["contrasts"][f"{NAME[x]} - {NAME[y]}"] = rec
    return out


def fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def progress():
    rep = {(r["target"], r["axis"], r["seed"], r["method"])
           for r in E5.jl(os.path.join(OUT, "replay.jsonl"))}
    need = [(t, a, s, m) for t, a in CELLS for s in SEEDS for m in REPLAYED]
    print(f"E3c comparison replay: {sum(k in rep for k in need)}/{len(need)} unit-methods on disk")
    return 0


def main():
    runs, missing, bad = load_all()
    if missing:
        progress()
        print(f"NO PARTIAL READS: {len(missing)} runs missing, e.g. {missing[:3]}")
        return 1
    if bad:
        print(f"REPLAY GATE: {len(bad)} runs do not reproduce the stored panel:")
        for k, why in bad:
            print("   ", k, why)
        return 1
    out = analyse(runs)
    print("eight comparison cells: removal with the configuration chosen on ...")
    print(f"  {'method':16s} {'all items (published)':26s} {'the other half':26s} "
          f"{'the same half':26s} optimism")
    for m in METHODS:
        r = out["methods"][m]
        print(f"  {NAME[m]:16s} {fmt(r['full']):26s} {fmt(r['split']):26s} {fmt(r['same']):26s} "
              f"{fmt(r['optimism'])}   halves disagree in {r['halves_disagree']} runs")
        if r["runs_not_bit_exact"]:
            print(f"      replay not bit-exact in {r['runs_not_bit_exact']} runs; stored value "
                  f"{fmt(r['stored'])}; largest cell difference replay vs stored "
                  f"{r['max_abs_cell_diff_replay_vs_stored']:.4f}")
    for name, r in out["contrasts"].items():
        print(f"  {name:32s} published {fmt(r['full'])}   split-half {fmt(r['split'])}   "
              f"interval status {'CHANGES' if r['status_changes'] else 'unchanged'}")
    with open(os.path.join(OUT, "e3c_front.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.relpath(os.path.join(OUT, "e3c_front.json"), REPO))
    return 0


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    pre = {None: 0.5, 0: 0.5, 1: 0.5}
    same = lambda x: {None: x, 0: x, 1: x}   # noqa: E731
    v = select_report(pre, [same(0.4), same(0.2), same(0.3)], [True, True, True])
    check("identical halves: split == same == full, best configuration chosen",
          v["i_full"] == 1 and abs(v["full"] - 0.3) < 1e-12 and abs(v["split"] - 0.3) < 1e-12
          and abs(v["same"] - 0.3) < 1e-12)
    v = select_report(pre, [same(0.4), same(0.2), same(0.3)], [True, False, True])
    check("a configuration out of budget is never chosen", v["i_full"] == 2 and v["i_h"] == (2, 2))
    v = select_report(pre, [same(0.4)], [False])
    check("nothing in budget -> 0", v["full"] == 0.0 and v["split"] == 0.0 and v["i_full"] is None)
    post = [{None: 0.45, 0: 0.40, 1: 0.50}, {None: 0.45, 0: 0.50, 1: 0.40}]
    v = select_report(pre, post, [True, True])
    check("halves that disagree: same-half 0.10, split-half 0.00",
          abs(v["same"] - 0.10) < 1e-12 and abs(v["split"]) < 1e-12 and v["i_h"] == (0, 1))
    runs = {}
    for t, a in CELLS:
        for s in SEEDS:
            for m in METHODS:
                x = 0.3 if m.startswith("edit") else 0.2
                runs[(t, a, s, m)] = dict(full=x, split=x - 0.01, same=x, i_full=0, i_h=(0, 0),
                                          stored=x)
    o = analyse(runs)
    check("analyse: optimism 0.01 for every method, differences unchanged in status",
          all(abs(o["methods"][m]["optimism"]["point"] - 0.01) < 1e-12 for m in METHODS)
          and not any(c["status_changes"] for c in o["contrasts"].values())
          and abs(o["contrasts"]["dense edit - steering"]["split"]["point"] - 0.1) < 1e-12)
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]())
