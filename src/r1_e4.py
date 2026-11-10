#!/usr/bin/env python3
"""R1 E4 -- budget-matched null and necessity under alpha*: the registered readings.

    python3 src/r1_e4.py            # verdicts -> results/r1/E4/e4_verdict.json
    python3 src/r1_e4.py progress   # unit counts only; reads no removal

REGISTERED (R1_REBUTTAL_EXPERIMENTS.md, E4; commit 0fe9043). All arms are 1%
edits built from the same contrast vector with C-ref's per-tensor scales, each
scaled to its own budget by the bias-blind rule alpha* (calibration split,
first failure on {1..512}, three bisections); removal as deployed at alpha*;
status 'none' -> 0; cell = seed mean over {0,1,2}; ten registered cells; paired
cell bootstrap (v10_common).

  G1, the validity gate.  q = C-rand(alpha*) / C-a(alpha*)   (ratio of the
  ten-cell means), D = C-a(alpha*) - C-rand(alpha*).
      G1-valid         q <= 0.25 and D's interval excludes 0
      G1-partial       0.25 < q < 0.5, or q <= 0.25 with D's interval including 0
      G1-degradation   q >= 0.5
  N-hold (per necessity arm X in C-b, C-layershuf, C-bottom):
      X(alpha*) <= 0.25 * C-ref(alpha*)    (ten-cell means)
  C-tensorshuf is reported on its 7 applicable cells, outside N-hold as registered.

REFERENCE ARMS. C-ref(alpha*) and C-a(alpha*) are the registered v12astar values
(R1_NOTES.md section 2.5), provided every R1 unit reproduced v12astar's dW
fingerprint; otherwise this script stops and says so.

NO PARTIAL READS. Nothing about removal is printed for an arm until all of its
units are on disk (C-tensorshuf: all applicable units).

ALSO REPORTED, labelled, never the verdict: the same contrasts under the
alternative removal definitions of E3a (signed; capped at zero, post hoc), the
alpha* of each arm, the evaluation-budget pass rate, and the frozen-grid replay
of each arm against the published DEC panel.
"""
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V    # noqa: E402
import v12_astar as A     # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
PANEL = os.path.join("r1", "E4")
OUT = os.path.join(RESULTS, PANEL)
ARMS = ["C-rand", "C-b", "C-layershuf", "C-tensorshuf", "C-bottom"]
NECESSITY = ["C-b", "C-layershuf", "C-bottom"]
Q_VALID, Q_DEGR, N_FRAC = 0.25, 0.5, 0.25


def jl(fp):
    out = []
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def load():
    rows, na = {}, set()
    for r in jl(os.path.join(OUT, "removal.jsonl")):
        rows[(r["target"], r["axis"], r["seed"], r["arm"])] = r          # last wins
    for r in jl(os.path.join(OUT, "not_applicable.jsonl")):
        na.add((r["target"], r["axis"], r["seed"], r["arm"]))
    return rows, na


def need(arm, na):
    return [(t, a, s, arm) for t, a in A.CELLS for s in A.SEEDS if (t, a, s, arm) not in na]


def removal(defn, pre, post):
    s = 1.0 if pre >= 0 else -1.0
    if defn == "ii_abs":
        return abs(pre) - abs(post)
    if defn == "i_signed":
        return s * (pre - post)
    if defn == "iv_capped":
        return abs(pre) - max(0.0, s * post)
    raise ValueError(defn)


def adaptive(r, defn="ii_abs"):
    ad = r.get("adaptive") or {}
    if ad.get("alpha_star") is None:
        return 0.0                                   # not deployed
    if defn == "ii_abs":
        return A.z(ad.get("removal"))
    return removal(defn, r["pre_skew"], ad["post_skew"])


def cell_means(rows, arm, cells, key, defn="ii_abs"):
    out = []
    for t, a in cells:
        vals = [adaptive(rows[(t, a, s, key(arm))], defn) for s in A.SEEDS]
        out.append(float(np.mean(vals)))
    return out


def progress():
    rows, na = load()
    for arm in ARMS:
        n = need(arm, na)
        print(f"{PANEL} {arm:13s} {sum(k in rows for k in n)}/{len(n)} units on disk")


def main():
    rows, na = load()
    ref = A.load("v12astar")                          # (t, a, s, variant)
    if A.missing(ref):
        raise SystemExit("v12astar (the reference arms) is incomplete")
    done = [arm for arm in ARMS if all(k in rows for k in need(arm, na)) and need(arm, na)]
    if "C-rand" not in done:
        progress()
        print("incomplete: C-rand is not read before all 30 units exist")
        return 1

    # ---- replay guards ----
    units = {}
    for u in jl(os.path.join(OUT, "unit.jsonl")):
        units[(u["target"], u["axis"], u["seed"])] = u
    bad_geom = sorted(k for k, u in units.items() if u.get("geom_match") is not True)
    missing_units = [(t, a, s) for t, a in A.CELLS for s in A.SEEDS if (t, a, s) not in units]
    if bad_geom or missing_units:
        raise SystemExit(f"dW fingerprint not reproduced or unit rows missing: "
                         f"{bad_geom or missing_units}. The registered reference arms do "
                         "not apply (R1_NOTES 2.5); use the E5 replays instead.")
    base_ok = sum(bool(u.get("pre_eval_match")) and bool(u.get("pre_calib_match"))
                  for u in units.values())
    dec = {}
    for r in jl(os.path.join(REPO, "results_v9", "v8dec", "removal.jsonl")):
        dec[(r["target"], r["axis"], r["seed"], r.get("condition"))] = r
    replay = {}
    for arm in done:
        pairs = [(A.z((rows[k].get("frozen") or {}).get("removal")), A.z(dec[k]["removal"]))
                 for k in need(arm, na) if k in dec]
        replay[arm] = dict(n=len(pairs), n_diff=sum(x != y for x, y in pairs),
                           max_abs=max((abs(x - y) for x, y in pairs), default=0.0))

    def ref_key(arm):
        return {"C-ref": "s0.99", "C-a": "C-a@0.01"}[arm]

    out = dict(script="src/r1_e4.py", registered="R1_REBUTTAL_EXPERIMENTS.md E4 (0fe9043)",
               arms_complete=done, geom_sha_reproduced=f"{len(units)}/30",
               baselines_reproduced=f"{base_ok}/30", frozen_replay_vs_v8dec=replay,
               by_definition={})
    for defn in ("ii_abs", "i_signed", "iv_capped"):
        c_ref = cell_means(ref, "C-ref", A.CELLS, ref_key, defn)
        c_a = cell_means(ref, "C-a", A.CELLS, ref_key, defn)
        c_rand = cell_means(rows, "C-rand", A.CELLS, lambda x: x, defn)
        q = float(np.mean(c_rand) / np.mean(c_a)) if np.mean(c_a) else float("nan")
        D = V.boot_paired(list(zip(c_a, c_rand)))
        Dref = V.boot_paired(list(zip(c_ref, c_rand)))
        covers = not (D["lo"] > 0 or D["hi"] < 0)
        if q >= Q_DEGR:
            g1 = "G1-degradation"
        elif q <= Q_VALID and not covers:
            g1 = "G1-valid"
        else:
            g1 = "G1-partial"
        d = dict(means=dict(C_ref=float(np.mean(c_ref)), C_a=float(np.mean(c_a)),
                            C_rand=float(np.mean(c_rand))),
                 C_rand=V.boot_mean(c_rand), q=q, C_a_minus_C_rand=D,
                 C_ref_minus_C_rand=Dref, G1=g1,
                 cells={f"{t}|{a}": dict(C_ref=x, C_a=y, C_rand=z_)
                        for (t, a), x, y, z_ in zip(A.CELLS, c_ref, c_a, c_rand)})
        nec = {}
        for arm in NECESSITY + ["C-tensorshuf"]:
            if arm not in done:
                continue
            cells = [c for c in A.CELLS
                     if all((c[0], c[1], s, arm) not in na for s in A.SEEDS)]
            x = cell_means(rows, arm, cells, lambda v: v, defn)
            rr = cell_means(ref, "C-ref", cells, ref_key, defn)
            nec[arm] = dict(n_cells=len(cells), mean=float(np.mean(x)), ci=V.boot_mean(x),
                            C_ref_same_cells=float(np.mean(rr)),
                            ratio_to_C_ref=float(np.mean(x) / np.mean(rr)),
                            holds=bool(np.mean(x) <= N_FRAC * np.mean(rr)),
                            C_ref_minus_arm=V.boot_paired(list(zip(rr, x))))
        d["necessity"] = nec
        if all(a_ in nec for a_ in NECESSITY):
            d["N_hold"] = all(nec[a_]["holds"] for a_ in NECESSITY)
        out["by_definition"][defn] = d

    # ---- descriptive: alpha*, budget pass rate, status ----
    desc = {}
    for arm in done:
        rs = [rows[k] for k in need(arm, na)]
        al = sorted(r["adaptive"]["alpha_star"] for r in rs
                    if r["adaptive"].get("alpha_star") is not None)
        ev = [r["adaptive"]["eval_ok"] for r in rs if r["adaptive"].get("alpha_star") is not None]
        desc[arm] = dict(status=dict(collections.Counter(r["adaptive"]["status"] for r in rs)),
                         alpha_star_median=(al[len(al) // 2] if al else None),
                         alpha_star_range=([al[0], al[-1]] if al else None),
                         eval_pass_rate=(sum(ev) / len(ev) if ev else None),
                         sign_flips_at_alpha_star=sum(
                             r["adaptive"].get("post_skew") is not None
                             and r["pre_skew"] * r["adaptive"]["post_skew"] < 0 for r in rs))
    same_alpha = sum(rows[(t, a, s, "C-rand")]["adaptive"]["alpha_star"]
                     == ref[(t, a, s, "C-a@0.01")]["adaptive"]["alpha_star"]
                     for t, a in A.CELLS for s in A.SEEDS)
    out["descriptive"] = dict(arms=desc, C_rand_alpha_star_equal_to_C_a=f"{same_alpha}/30")

    main_d = out["by_definition"]["ii_abs"]
    out["G1"] = main_d["G1"]
    if "N_hold" in main_d:
        out["N_hold"] = main_d["N_hold"]
    with open(os.path.join(OUT, "e4_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"replay guards: dW fingerprint {out['geom_sha_reproduced']}, baselines "
          f"{out['baselines_reproduced']}; frozen-grid replay vs v8dec: "
          + ", ".join(f"{a_} {r['n'] - r['n_diff']}/{r['n']}" for a_, r in replay.items()))
    print(f"C-rand alpha* equals C-a's (same support, same scales) in {same_alpha}/30 units")
    for arm, d in desc.items():
        print(f"  alpha* {arm:13s} median {d['alpha_star_median']}  status {d['status']}  "
              f"eval pass {d['eval_pass_rate']}")
    labels = {"ii_abs": "registered definition |pre| - |post|",
              "i_signed": "signed (E3a variant; not the verdict)",
              "iv_capped": "capped at zero (post hoc; not the verdict)"}
    for defn, d in out["by_definition"].items():
        m = d["means"]
        print(f"\n[{labels[defn]}]")
        print(f"  at alpha*: C-ref {m['C_ref']:+.3f}   C-a {m['C_a']:+.3f}   "
              f"C-rand {V.fmt(d['C_rand'], 3)}")
        print(f"  q = C-rand / C-a = {d['q']:.3f}    C-a - C-rand {V.fmt(d['C_a_minus_C_rand'], 3)}"
              f"    C-ref - C-rand {V.fmt(d['C_ref_minus_C_rand'], 3)}")
        print(f"  -> {d['G1']}")
        for arm, n in d["necessity"].items():
            print(f"  {arm:13s} {V.fmt(n['ci'], 3)}  = {n['ratio_to_C_ref']:.3f} x C-ref "
                  f"({n['n_cells']} cells)  holds={n['holds']}")
        if "N_hold" in d:
            print(f"  -> N-hold: {d['N_hold']}")
    print(f"\nREGISTERED VERDICT: {out['G1']}" + (f", N-hold={out['N_hold']}"
                                                  if "N_hold" in out else ""))
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'e4_verdict.json'), REPO)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress}[cmd]() or 0)
