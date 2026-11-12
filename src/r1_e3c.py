#!/usr/bin/env python3
"""R1 E3c -- select on one half of the probe items, report on the other (fixed grid).

    python3 src/r1_e3c.py split      write audit/r1_e3c_split.json (loads the probe sets)
    python3 src/r1_e3c.py            the analysis, from stored per-item likelihoods
    python3 src/r1_e3c.py selftest

PLAN (R1_REBUTTAL_EXPERIMENTS.md, E3c; 5PJm Q3, pe22 Q3). Under the grid rule the
scale is chosen on the same probe items the removal is reported on. Split the
probe items into two halves by sha256(item_id) mod 2, choose the scale on one
half, report on the other, do the same in the other direction, and average.
The plan fixes no verdict: it asks for the absolute change and for whether the
intervals of the main contrasts change status. Details fixed in R1_NOTES.md
section 3 (entry of 2026-11-12) before this script was run on any data:

  item_id   occupation axis: the occupation string of PROBE_OCC (the eight
            templates of an occupation stay together); pair axes (BBQ,
            StereoSet): the key of the probe pair as built by colab_t2t4.
  skew      recomputed per half from the stored per-item log-likelihoods with
            the formulas of colab_t2t4 (occ_skew; mean pair difference).
  budget    not split: a scale is admissible if the run's recorded evaluation
            budget check passed at that scale, as in the published rule.
  select    on half A: the admissible grid scale with the largest removal on A
            (first maximum, as v12_opsel.frozen_argmax); none admissible -> 0.
  report    removal on half B at that scale; then A and B swapped; the
            split-half value of the run is the mean of the two.
  same-half for comparison: select and report on the same half, mean of the two
            halves. same-half minus split-half is the optimism of selecting on
            the reported items.
  scope     the seven variants on the ten main cells, three seeds: C-ref and
            C-a from results/r1/E5, the other five from results/r1/E4 (their
            fixed-grid values replay the published panel exactly). The eight-
            cell comparison is not covered: per-item likelihoods of steering
            and SentenceDebias were not stored.
  cell = seed mean; bootstrap over the ten cells (v10_common).

Sanity gate: for every run and scale the skew recomputed from all items equals
the recorded one, and the full-set selection equals the run's `frozen` row.
"""
import hashlib
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

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E3")
SPLIT_FP = os.path.join(REPO, "audit", "r1_e3c_split.json")
PANEL = {"C-ref": "E5", "C-a": "E5", "C-rand": "E4", "C-b": "E4", "C-layershuf": "E4",
         "C-tensorshuf": "E4", "C-bottom": "E4"}
ARMS = list(PANEL)
GRID = (2.0, 4.0, 8.0, 16.0)


def half_of(item_id):
    return int(hashlib.sha256(item_id.encode("utf-8")).hexdigest(), 16) % 2


def build_split():
    """Needs the probe sets (datasets cache); writes the half of every probe item."""
    sys.path.insert(0, REPO)
    import colab_t2t4 as C
    out = {}
    # the axes of the ten main cells, and CrowS socioeconomic for the eight-cell
    # comparison (src/r1_e3c_front.py); the rule is the same for every axis
    for axis in sorted({a for _, a in A.CELLS} | {"crows_socioeconomic"}):
        if axis == "occ_gender":
            ids = list(C.PROBE_OCC.keys())
        else:
            probe, _ = C.load_axis(axis)
            ids = [k for _c, _i, k in probe]
            if len(set(ids)) != len(ids):
                raise SystemExit(f"probe keys of {axis} are not unique")
        h = [half_of(i) for i in ids]
        out[axis] = dict(n=len(ids), n_half1=int(sum(h)), half=h,
                         ids_sha256=hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest())
        print(f"{axis}: {len(ids)} items, half 0 / 1 = {len(ids) - sum(h)} / {sum(h)}")
    os.makedirs(os.path.dirname(SPLIT_FP), exist_ok=True)
    with open(SPLIT_FP, "w") as f:
        json.dump(out, f)
    print("wrote", os.path.relpath(SPLIT_FP, REPO))
    return 0


def skew_fn(axis, half, occ=None, n_tpl=None):
    """-> f(ll, mask_or_None) = skew on the items selected by the 0/1 mask."""
    half = np.asarray(half)
    if axis == "occ_gender":
        g = np.array([x[1] for x in occ])
        if len(half) != len(occ):
            raise SystemExit("occupation split does not match PROBE_OCC")

        def f(ll, which=None):
            d = ll[1::2].astype(np.float64) - ll[0::2].astype(np.float64)
            p = (1.0 / (1.0 + np.exp(-d))).reshape(len(occ), n_tpl).mean(1)
            congr = np.where(g == "f", p, 1.0 - p)
            sel = np.ones(len(occ), bool) if which is None else (half == which)
            return 2.0 * float(congr[sel].mean()) - 1.0
        return f

    def f(ll, which=None):
        d = ll[0::2].astype(np.float64) - ll[1::2].astype(np.float64)
        if len(d) != len(half):
            raise SystemExit(f"{axis}: {len(d)} stored pairs, {len(half)} in the split")
        sel = np.ones(len(d), bool) if which is None else (half == which)
        return float(d[sel].mean())
    return f


def argmax(cands):
    """cands: [(alpha, removal)] admissible, in grid order -> alpha or None."""
    best, best_a = None, None
    for a, r in cands:
        if best is None or r > best:
            best, best_a = r, a
    return best_a


def run_values(pre, post, ok):
    """Pure. pre: {None,0,1: skew}; post: {alpha: {None,0,1: skew}}; ok: {alpha: bool}.
    -> dict(full, split, same) removals of one run."""
    def rem(which, a):
        return abs(pre[which]) - abs(post[a][which])
    adm = [a for a in GRID if ok[a]]
    if not adm:
        return dict(full=0.0, split=0.0, same=0.0, alpha_full=None, alpha_h=(None, None))
    a_full = argmax([(a, rem(None, a)) for a in adm])
    a0 = argmax([(a, rem(0, a)) for a in adm])
    a1 = argmax([(a, rem(1, a)) for a in adm])
    return dict(full=rem(None, a_full),
                split=0.5 * (rem(1, a0) + rem(0, a1)),
                same=0.5 * (rem(0, a0) + rem(1, a1)),
                alpha_full=a_full, alpha_h=(a0, a1))


def load_runs():
    split = json.load(open(SPLIT_FP))
    occ, n_tpl = E13.probe_occ()
    rows = {p: E5.load(os.path.join(RESULTS, "r1", p))[0] for p in ("E4", "E5")}
    out, missing = {}, []
    for arm in ARMS:
        pdir = os.path.join(RESULTS, "r1", PANEL[arm])
        for t, a in A.CELLS:
            if arm == "C-tensorshuf" and t == "phi":
                continue
            fn = skew_fn(a, split[a]["half"], occ, n_tpl)
            for s in A.SEEDS:
                r = rows[PANEL[arm]].get((t, a, s, arm))
                if r is None:
                    missing.append((t, a, s, arm))
                    continue
                zf = np.load(os.path.join(pdir, "probe_ll", f"{t}__{a}__s{s}__{r['designer']}__{arm}.npz"))
                alphas = [float(x) for x in zf["alphas"]]
                rung = {float(x["alpha"]): x for x in r["rungs"]}
                pre = {w: fn(zf["pre_ll"], w) for w in (None, 0, 1)}
                if abs(pre[None] - r["pre_skew"]) > 1e-5:
                    raise SystemExit(f"recomputed pre-skew != recorded: {t}|{a}|s{s}|{arm}")
                post, ok = {}, {}
                for g in GRID:
                    i = [j for j, x in enumerate(alphas) if x == g]
                    if len(i) != 1:
                        raise SystemExit(f"scale {g} not stored exactly once: {t}|{a}|s{s}|{arm}")
                    post[g] = {w: fn(zf["ll"][i[0]], w) for w in (None, 0, 1)}
                    if abs(post[g][None] - rung[g]["eval"]["post_skew"]) > 1e-5:
                        raise SystemExit(f"recomputed post-skew != recorded: {t}|{a}|s{s}|{arm}|{g}")
                    ok[g] = bool(rung[g]["eval"]["ok"])
                v = run_values(pre, post, ok)
                fr = r.get("frozen") or {}
                if abs(v["full"] - A.z(fr.get("removal"))) > 1e-5 or v["alpha_full"] != fr.get("alpha"):
                    raise SystemExit(f"full-set selection != frozen row: {t}|{a}|s{s}|{arm}")
                out[(t, a, s, arm)] = v
    return out, missing


def cells(runs, arm, key):
    cs = [c for c in A.CELLS if not (arm == "C-tensorshuf" and c[0] == "phi")]
    return cs, [float(np.mean([runs[(t, a, s, arm)][key] for s in A.SEEDS])) for t, a in cs]


def fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def main():
    if not os.path.exists(SPLIT_FP):
        print("run `python3 src/r1_e3c.py split` first")
        return 1
    runs, missing = load_runs()
    if missing:
        print(f"NO PARTIAL READS: {len(missing)} runs missing, e.g. {missing[:3]}")
        return 1
    out = dict(script="src/r1_e3c.py", registered="R1_REBUTTAL_EXPERIMENTS.md E3c; R1_NOTES.md section 3 (2026-11-12)",
               arms={}, contrasts={})
    print("fixed grid, ten main cells (C-tensorshuf: seven): removal with the scale chosen on ...")
    print(f"  {'variant':13s} {'all items (published)':26s} {'the other half (split)':26s} "
          f"{'the same half':26s} optimism (same - split)")
    for arm in ARMS:
        _, full = cells(runs, arm, "full")
        _, split = cells(runs, arm, "split")
        _, same = cells(runs, arm, "same")
        n_diff = sum(runs[k]["alpha_h"][0] != runs[k]["alpha_h"][1] for k in runs if k[3] == arm)
        n_runs = sum(1 for k in runs if k[3] == arm)
        rec = dict(full=V.boot_mean(full), split=V.boot_mean(split), same=V.boot_mean(same),
                   optimism=V.boot_paired(list(zip(same, split))),
                   split_minus_full=V.boot_paired(list(zip(split, full))),
                   runs_where_halves_choose_different_scales=f"{n_diff}/{n_runs}")
        out["arms"][arm] = rec
        print(f"  {arm:13s} {fmt(rec['full']):26s} {fmt(rec['split']):26s} {fmt(rec['same']):26s} "
              f"{fmt(rec['optimism'])}   halves disagree on the scale in {n_diff}/{n_runs} runs")
    for name, (x, y) in {"C-ref - C-a (selection effect)": ("C-ref", "C-a"),
                         "C-a - C-rand": ("C-a", "C-rand"), "C-ref - C-rand": ("C-ref", "C-rand"),
                         "C-ref - C-b": ("C-ref", "C-b"), "C-ref - C-layershuf": ("C-ref", "C-layershuf"),
                         "C-ref - C-bottom": ("C-ref", "C-bottom")}.items():
        rec = {}
        for key in ("full", "split"):
            _, vx = cells(runs, x, key)
            _, vy = cells(runs, y, key)
            rec[key] = V.boot_paired(list(zip(vx, vy)))
            rec[key]["excludes_zero"] = not E5.covers(rec[key])
        rec["status_changes"] = rec["full"]["excludes_zero"] != rec["split"]["excludes_zero"]
        out["contrasts"][name] = rec
        print(f"  {name:32s} published {fmt(rec['full'])}   split-half {fmt(rec['split'])}"
              f"   interval status {'CHANGES' if rec['status_changes'] else 'unchanged'}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e3c_split_half.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.relpath(os.path.join(OUT, "e3c_split_half.json"), REPO))
    return 0


def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name)

    check("the hash rule is deterministic and binary",
          half_of("midwife") == half_of("midwife") and half_of("midwife") in (0, 1))
    # a run whose two halves agree exactly: split == same == full
    pre = {None: 0.6, 0: 0.6, 1: 0.6}
    post = {g: {None: 0.6 - 0.02 * g, 0: 0.6 - 0.02 * g, 1: 0.6 - 0.02 * g} for g in GRID}
    v = run_values(pre, post, {g: True for g in GRID})
    check("identical halves: split == same == full, largest scale chosen",
          abs(v["split"] - v["full"]) < 1e-12 and abs(v["same"] - v["full"]) < 1e-12 and v["alpha_full"] == 16.0)
    v = run_values(pre, post, {2.0: True, 4.0: True, 8.0: False, 16.0: False})
    check("out-of-budget scales are never chosen", v["alpha_full"] == 4.0 and v["alpha_h"] == (4.0, 4.0))
    v = run_values(pre, post, {g: False for g in GRID})
    check("nothing in budget -> 0 everywhere", v["full"] == 0.0 and v["split"] == 0.0 and v["same"] == 0.0)
    # pure noise: each half prefers a different scale -> same > split (optimism)
    pre = {None: 0.5, 0: 0.5, 1: 0.5}
    post = {2.0: {None: 0.45, 0: 0.40, 1: 0.50}, 4.0: {None: 0.45, 0: 0.50, 1: 0.40},
            8.0: {None: 0.5, 0: 0.5, 1: 0.5}, 16.0: {None: 0.5, 0: 0.5, 1: 0.5}}
    v = run_values(pre, post, {g: True for g in GRID})
    check("halves that disagree: same-half 0.10, split-half 0.00",
          abs(v["same"] - 0.10) < 1e-12 and abs(v["split"]) < 1e-12 and v["alpha_h"] == (2.0, 4.0))
    # skew functions against direct formulas
    occ = [("a", "f"), ("b", "m"), ("c", "f"), ("d", "m")]
    rng = np.random.default_rng(0)
    ll = rng.normal(size=4 * 2 * 2).astype(np.float32)
    f = skew_fn("occ_gender", [0, 0, 1, 1], occ, 2)
    d = ll[1::2].astype(np.float64) - ll[0::2].astype(np.float64)
    p = (1 / (1 + np.exp(-d))).reshape(4, 2).mean(1)
    congr = np.array([p[0], 1 - p[1], p[2], 1 - p[3]])
    check("occupation skew, all items and one half",
          abs(f(ll) - (2 * congr.mean() - 1)) < 1e-12 and abs(f(ll, 1) - (2 * congr[2:].mean() - 1)) < 1e-12)
    g = skew_fn("bbq_Age", [0, 1, 1])
    l2 = np.array([1.0, 0.0, 2.0, 0.0, 3.0, 5.0], np.float32)
    check("pair skew, all items and one half", abs(g(l2) - (1 + 2 - 2) / 3) < 1e-12 and abs(g(l2, 1) - 0.0) < 1e-12)
    print("selftest", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "split": build_split, "selftest": selftest}[cmd]())
