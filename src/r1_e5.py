#!/usr/bin/env python3
"""R1 E5 -- operator 2x2 + ComPEFT under alpha*: the registered readings.

    python3 src/r1_e5.py            # verdicts -> results/r1/E5/e5_verdict.json
    python3 src/r1_e5.py progress   # unit counts only; reads no removal
    python3 src/r1_e5.py selftest   # the reading logic on synthetic rows

REGISTERED (R1_REBUTTAL_EXPERIMENTS.md, E5; commit 0fe9043). Five 1% edits of
the same contrast vector, merged W <- W - alpha*E, each scaled to its own
budget by the bias-blind rule alpha* (calibration split, first failure on its
ladder, three bisections):

    arm        support              values          scale
    O-trim     top-1% (magnitude)   fp dW           alpha
    O-dare     random 1% (= C-a's)  fp dW x 100     alpha
    O-compeft  top-1% (magnitude)   sign            sigma(dW) x alpha (one scalar)
    C-ref      top-1% (magnitude)   sign            s_t x alpha
    C-a        random 1%            sign            s_t x alpha

Removal as deployed at alpha* (|pre skew| - |post skew|); status 'none' -> 0;
cell = seed mean over {0,1,2}; ten registered cells; paired cell bootstrap
(v10_common).

  E5-1  C-ref - O-trim     interval includes 0 -> one bit is free at matched
                           collateral as well
  E5-2  O-trim - O-dare    interval includes 0 -> the support does not matter at
                           matched collateral in full precision either
  E5-3  C-ref - O-compeft  interval includes 0 -> the per-tensor scale is not a
                           point of difference from ComPEFT
  Reported with them, as the plan lists: C-a - O-dare (values vs sign on the
  random support) and C-ref - C-a (support with sign values; the existing
  Delta_selection under alpha*).

REFERENCE ARMS (R1_NOTES.md section 2.5). C-ref(alpha*) and C-a(alpha*) are the
registered v12astar values, provided every unit of this panel reproduced
v12astar's dW fingerprint; if any unit did not, the same-process replays of
this panel replace them in all units. The other source is printed next to the
verdict, and a reading that changes with the source is labelled FRAGILE.

NO PARTIAL READS. Nothing about removal is printed until all 150 unit-arms are
on disk.

ALSO REPORTED, labelled, never the verdict: the same contrasts under the
alternative removal definitions of E3a (signed; capped at zero, post hoc); each
arm's alpha*, realized collateral at alpha* and evaluation-budget pass rate;
how exactly the two replays reproduce v12astar.
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
PANEL = os.path.join("r1", "E5")
OUT = os.path.join(RESULTS, PANEL)
ARMS = ["O-compeft", "O-trim", "O-dare", "C-ref", "C-a"]
REF_KEY = {"C-ref": "s0.99", "C-a": "C-a@0.01"}
REGISTERED = [
    ("E5-1", "C-ref", "O-trim", "values vs sign on the top-1% support",
     "one bit is free at matched collateral as well"),
    ("E5-2", "O-trim", "O-dare", "support, full-precision values",
     "the support does not matter at matched collateral in full precision either"),
    ("E5-3", "C-ref", "O-compeft", "per-tensor scale s_t vs one global sigma",
     "the per-tensor scale is not a point of difference from ComPEFT"),
]
OTHER = [
    ("C-a", "O-dare", "values vs sign on the random support"),
    ("C-ref", "C-a", "support, sign values (Delta_selection under alpha*)"),
]
DEFNS = {"ii_abs": "registered definition |pre| - |post|",
         "i_signed": "signed (E3a variant; not the verdict)",
         "iv_capped": "capped at zero (post hoc; not the verdict)"}


class Partial(Exception):
    pass


def jl(fp):
    out = []
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def load(out_dir=OUT):
    rows, units = {}, {}
    for r in jl(os.path.join(out_dir, "removal.jsonl")):
        rows[(r["target"], r["axis"], r["seed"], r["arm"])] = r          # last wins
    for u in jl(os.path.join(out_dir, "unit.jsonl")):
        units[(u["target"], u["axis"], u["seed"])] = u
    return rows, units


def need(arms=ARMS):
    return [(t, a, s, arm) for t, a in A.CELLS for s in A.SEEDS for arm in arms]


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


def cell_means(table, key, defn):
    return [float(np.mean([adaptive(table[(t, a, s, key)], defn) for s in A.SEEDS]))
            for t, a in A.CELLS]


def covers(d):
    return not (d["lo"] > 0 or d["hi"] < 0)


def analyse(rows, units, ref):
    """Pure: rows of this panel, its unit rows, v12astar rows -> verdict dict."""
    miss = [k for k in need() if k not in rows]
    if miss:
        raise Partial(f"{len(need()) - len(miss)}/{len(need())} unit-arms on disk")
    if A.missing(ref):
        raise SystemExit("v12astar (the registered reference arms) is incomplete")
    ukeys = [(t, a, s) for t, a in A.CELLS for s in A.SEEDS]
    geom_bad = sorted(k for k in ukeys if (units.get(k) or {}).get("geom_match") is not True)
    source = "replay" if geom_bad else "registered"
    base_ok = sum(bool((units.get(k) or {}).get("pre_eval_match"))
                  and bool((units.get(k) or {}).get("pre_calib_match")) for k in ukeys)

    def series(arm, src, defn):
        if arm in REF_KEY and src == "registered":
            return cell_means(ref, REF_KEY[arm], defn)
        return cell_means(rows, arm, defn)

    out = dict(script="src/r1_e5.py", registered="R1_REBUTTAL_EXPERIMENTS.md E5 (0fe9043)",
               reference_source=source, geom_sha_not_reproduced=[list(k) for k in geom_bad],
               geom_sha_reproduced=f"{len(ukeys) - len(geom_bad)}/{len(ukeys)}",
               baselines_reproduced=f"{base_ok}/{len(ukeys)}", by_source={})
    for src in ("registered", "replay"):
        by_def = {}
        for defn in DEFNS:
            m = {arm: series(arm, src, defn) for arm in ARMS}
            d = dict(means={arm: float(np.mean(m[arm])) for arm in ARMS},
                     ci={arm: V.boot_mean(m[arm]) for arm in ARMS},
                     cells={f"{t}|{a}": {arm: m[arm][i] for arm in ARMS}
                            for i, (t, a) in enumerate(A.CELLS)},
                     contrasts={})
            for tag, x, y, what, _ in REGISTERED:
                c = V.boot_paired(list(zip(m[x], m[y])))
                d["contrasts"][f"{x} - {y}"] = dict(c, tag=tag, what=what,
                                                    includes_zero=covers(c))
            for x, y, what in OTHER:
                c = V.boot_paired(list(zip(m[x], m[y])))
                d["contrasts"][f"{x} - {y}"] = dict(c, tag=None, what=what,
                                                    includes_zero=covers(c))
            by_def[defn] = d
        out["by_source"][src] = by_def

    # ---- the registered readings: registered definition, the applicable source ----
    other = "replay" if source == "registered" else "registered"
    verdict = {}
    for tag, x, y, what, meaning in REGISTERED:
        k = f"{x} - {y}"
        c = out["by_source"][source]["ii_abs"]["contrasts"][k]
        c2 = out["by_source"][other]["ii_abs"]["contrasts"][k]
        verdict[tag] = dict(contrast=k, point=c["point"], lo=c["lo"], hi=c["hi"],
                            includes_zero=c["includes_zero"],
                            reading=(meaning if c["includes_zero"] else
                                     f"does not hold: {x} {'>' if c['point'] > 0 else '<'} {y}"),
                            fragile_to_reference_source=bool(
                                c["includes_zero"] != c2["includes_zero"]),
                            by_definition={dn: out["by_source"][source][dn]["contrasts"][k]
                                           ["includes_zero"] for dn in DEFNS})
    out["verdict"] = verdict

    # ---- how exactly the replays reproduce v12astar ----
    rep = {}
    for arm, vk in REF_KEY.items():
        same_a = same_r = 0
        mx = 0.0
        for t, a, s in ukeys:
            x, y = rows[(t, a, s, arm)], ref[(t, a, s, vk)]
            same_a += x["adaptive"].get("alpha_star") == y["adaptive"].get("alpha_star")
            dx = abs(adaptive(x) - adaptive(y))
            same_r += dx == 0.0
            mx = max(mx, dx)
        rep[arm] = dict(n=len(ukeys), alpha_star_equal=int(same_a),
                        removal_equal=int(same_r), max_abs_removal_diff=mx)
    out["replay_vs_v12astar"] = rep

    # ---- descriptive ----
    desc = {}
    for arm in ARMS:
        rs = [rows[k] for k in need([arm])]
        dep = [r["adaptive"] for r in rs if r["adaptive"].get("alpha_star") is not None]
        al = sorted(d["alpha_star"] for d in dep)
        desc[arm] = dict(
            ladder=rs[0].get("ladder"),
            status=dict(collections.Counter(r["adaptive"]["status"] for r in rs)),
            alpha_star_median=(float(np.median(al)) if al else None),
            alpha_star_range=([al[0], al[-1]] if al else None),
            eval_pass_rate=(sum(bool(d["eval_ok"]) for d in dep) / len(dep) if dep else None),
            eval_dmmlu_items_mean=(float(np.mean([d["dmmlu_items"] for d in dep]))
                                   if dep else None),
            eval_ppl_ratio_median=(float(np.median([d["ppl_ratio"] for d in dep]))
                                   if dep else None),
            realized_l2_median=(float(np.median([d["realized_l2"] for d in dep]))
                                if dep else None),
            sign_flips_at_alpha_star=sum(
                r["adaptive"].get("post_skew") is not None
                and r["pre_skew"] * r["adaptive"]["post_skew"] < 0 for r in rs))
    out["descriptive"] = desc
    return out


def report(out):
    src = out["reference_source"]
    print(f"replay guards: dW fingerprint {out['geom_sha_reproduced']}, baselines "
          f"{out['baselines_reproduced']}")
    print(f"reference arms (C-ref, C-a): {src}"
          + ("  [v12astar values]" if src == "registered" else
             "  [same-process replays: fingerprint not reproduced in "
             f"{len(out['geom_sha_not_reproduced'])} units]"))
    for arm, r in out["replay_vs_v12astar"].items():
        print(f"  replay {arm:6s} alpha* equal {r['alpha_star_equal']}/{r['n']}, removal "
              f"equal {r['removal_equal']}/{r['n']}, max |diff| {r['max_abs_removal_diff']:.2e}")
    print("\nat alpha* (each arm at its own):")
    for arm, d in out["descriptive"].items():
        print(f"  {arm:10s} alpha* median {d['alpha_star_median']}  range "
              f"{d['alpha_star_range']}  status {d['status']}  eval pass "
              f"{d['eval_pass_rate']}  dMMLU {d['eval_dmmlu_items_mean']}  ppl x"
              f"{d['eval_ppl_ratio_median']}  sign flips {d['sign_flips_at_alpha_star']}")
    for s in (src, "replay" if src == "registered" else "registered"):
        for defn, label in DEFNS.items():
            d = out["by_source"][s][defn]
            print(f"\n[reference arms: {s}{' (the verdict source)' if s == src else ''}; "
                  f"{label}]")
            print("  removal: " + "   ".join(f"{arm} {V.fmt(d['ci'][arm], 3)}"
                                             for arm in ARMS))
            for k, c in d["contrasts"].items():
                print(f"  {(c['tag'] or ''):5s} {k:20s} {V.fmt(c, 3)}  "
                      f"{'includes 0' if c['includes_zero'] else 'excludes 0'}   ({c['what']})")
    print("\nREGISTERED READINGS")
    for tag, v in out["verdict"].items():
        print(f"  {tag}  {v['contrast']:20s} {v['point']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
              f"  -> {v['reading']}"
              + ("   FRAGILE to the reference source" if v["fragile_to_reference_source"]
                 else ""))


def progress():
    rows, _ = load()
    for arm in ARMS:
        n = need([arm])
        print(f"{PANEL} {arm:10s} {sum(k in rows for k in n)}/{len(n)} units on disk")
    return 0


def main():
    rows, units = load()
    try:
        out = analyse(rows, units, A.load("v12astar"))
    except Partial as e:
        progress()
        print(f"incomplete ({e}): E5 is not read before every unit-arm exists")
        return 1
    with open(os.path.join(OUT, "e5_verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    report(out)
    print(f"wrote {os.path.relpath(os.path.join(OUT, 'e5_verdict.json'), REPO)}")
    return 0


# ------------------------------------------------------------------ selftest ----
def _row(pre, post, alpha=8.0, status="ok"):
    ad = dict(alpha_star=alpha, status=status)
    if alpha is not None:
        ad.update(removal=abs(pre) - abs(post), post_skew=post, eval_ok=True,
                  dmmlu_items=1, ppl_ratio=1.05, realized_l2=1.0)
    return dict(pre_skew=pre, adaptive=ad, ladder="default")


def _synthetic():
    """C-ref == O-trim up to symmetric noise; O-dare 0.2 below; O-compeft 0.1 below;
    C-a 0.05 below C-ref."""
    rows, units, ref = {}, {}, {}
    for ci, (t, a) in enumerate(A.CELLS):
        for s in A.SEEDS:
            pre = 0.5 + 0.02 * ci
            base = 0.30 + 0.01 * ci
            eps = 0.01 * (1 if (ci + s) % 2 else -1)
            rem = {"C-ref": base, "O-trim": base + eps, "O-dare": base - 0.2 + 0.5 * eps,
                   "O-compeft": base - 0.1 + 0.3 * eps, "C-a": base - 0.05}
            for arm, r in rem.items():
                rows[(t, a, s, arm)] = _row(pre, pre - r)
            for arm, vk in REF_KEY.items():
                ref[(t, a, s, vk)] = _row(pre, pre - rem[arm])
            units[(t, a, s)] = dict(geom_match=True, pre_eval_match=True,
                                    pre_calib_match=True)
    return rows, units, ref


def selftest():
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(f"  {'ok ' if cond else 'FAIL'} {name}")

    rows, units, ref = _synthetic()
    out = analyse(rows, units, ref)
    v = out["verdict"]
    check("complete panel, fingerprints reproduced -> registered reference arms",
          out["reference_source"] == "registered")
    check("E5-1 includes 0 when C-ref and O-trim differ by symmetric noise",
          v["E5-1"]["includes_zero"] and abs(v["E5-1"]["point"]) < 0.01)
    check("E5-2 excludes 0, O-trim > O-dare by 0.2",
          not v["E5-2"]["includes_zero"] and abs(v["E5-2"]["point"] - 0.2) < 0.02
          and ">" in v["E5-2"]["reading"])
    check("E5-3 excludes 0, C-ref > O-compeft by 0.1",
          not v["E5-3"]["includes_zero"] and abs(v["E5-3"]["point"] - 0.1) < 0.02)
    c = out["by_source"]["registered"]["ii_abs"]["contrasts"]["C-ref - C-a"]
    check("C-ref - C-a = 0.05 from the reference rows", abs(c["point"] - 0.05) < 1e-9)
    check("replays identical to the reference -> 30/30 equal",
          all(r["alpha_star_equal"] == 30 and r["removal_equal"] == 30
              for r in out["replay_vs_v12astar"].values()))
    check("no reading is fragile when replays equal the reference",
          not any(x["fragile_to_reference_source"] for x in v.values()))

    r2 = dict(rows)
    del r2[(A.CELLS[3][0], A.CELLS[3][1], 1, "O-dare")]
    try:
        analyse(r2, units, ref)
        check("one missing unit-arm -> refuses", False)
    except Partial:
        check("one missing unit-arm -> refuses", True)

    u2 = dict(units)
    u2[(A.CELLS[0][0], A.CELLS[0][1], 2)] = dict(geom_match=False)
    ref2 = {k: _row(r["pre_skew"], r["pre_skew"]) for k, r in ref.items()}   # ref removal 0
    o2 = analyse(rows, u2, ref2)
    check("one fingerprint mismatch -> replays replace the reference in all units",
          o2["reference_source"] == "replay"
          and abs(o2["verdict"]["E5-3"]["point"] - 0.1) < 0.02)
    check("... and a reading that differs under the other source is labelled FRAGILE",
          o2["verdict"]["E5-1"]["fragile_to_reference_source"])

    r3 = dict(rows)
    for s in A.SEEDS:
        k = (A.CELLS[0][0], A.CELLS[0][1], s, "O-trim")
        r3[k] = _row(rows[k]["pre_skew"], 0.0, alpha=None, status="none")
    o3 = analyse(r3, units, ref)
    check("status none scores 0 (the edit is not deployed)",
          o3["by_source"]["registered"]["ii_abs"]["cells"]
          [f"{A.CELLS[0][0]}|{A.CELLS[0][1]}"]["O-trim"] == 0.0
          and o3["descriptive"]["O-trim"]["status"].get("none") == 3)

    check("removal definitions on an overshoot (pre +0.4, post -0.1)",
          abs(removal("ii_abs", 0.4, -0.1) - 0.3) < 1e-12
          and abs(removal("i_signed", 0.4, -0.1) - 0.5) < 1e-12
          and abs(removal("iv_capped", 0.4, -0.1) - 0.4) < 1e-12
          and abs(removal("i_signed", -0.4, 0.1) - 0.5) < 1e-12)
    print(f"selftest: {sum(ok)}/{len(ok)}")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]() or 0)
