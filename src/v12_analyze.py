#!/usr/bin/env python3
"""v12 — replay checks and the registered readings of the OPSEL panels.

    python3 src/v12_analyze.py --check-replay   # gate: do replayed cells reproduce?
    python3 src/v12_analyze.py                  # -> results/v12/opsel_summary.json

REPLAY GATE (run first, report nothing until it passes):
    v12cal   s<sp>      vs results_v9/v8spc/{small,big}   (same key, same variant)
    v12dec1k s0.99      vs results_v9/v8dec  C-ref
    v12dec1k C-a@0.01   vs results_v9/v8dec  C-a
    v12big   s0.99, s0  vs results/v11big    C-ref  (occ_gender only)
Compared against the RE-SCORED trees (integer gate), never the as-run ones.

READINGS (all at the CELL unit, nan -> 0, 10k percentile bootstrap, seed 0,
via v10_common -- the project's standing conventions):

  1. sparsity by tier       s<sp> - s0.0, paired per cell, three tiers
                            (<=3.8B / 7-9B / 27-32B); plus binarization
                            retention s0.0 vs fp
  2. adaptive vs frozen     per variant: removal under α* minus removal under
                            the frozen argmax; the held-out budget-pass rate of
                            α* on the EVALUATION split; α* status counts
                            ('exhausted' = ladder never failed, α* is a bound)
  3. Delta_selection        s0.99 - C-a@0.01 under the frozen rule and under α*
  4. joint operating point  (p*, α*(p*)) vs frozen s0.99 and vs the best frozen
                            grid point (the oracle), held-out cells only
  5. MMLU-1000              for every selected point with a 1000-item reading:
                            does the 200-item decision survive at 1000? and,
                            where every α was read at 1000, the argmax the
                            1000-item gate itself selects
"""
import argparse
import collections
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V   # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))

TIERS = collections.OrderedDict([
    ("<=3.8B", {"gemma", "llama", "qwen", "phi"}),
    ("7-9B", {"qwen7b", "llama8b"}),
    ("27-32B", {"gemma27b", "qwen32b"}),
])
PANELS = ("v12cal", "v12big", "v12dec1k")


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def rows(panel, fname="removal.jsonl", root=None):
    fp = os.path.join(root or RESULTS, panel, fname)
    out = {}
    if not os.path.exists(fp):
        return []
    for ln in open(fp):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        k = (r.get("target"), r.get("axis"), r.get("seed"),
             r.get("variant", r.get("condition")))
        out[k] = r                              # last occurrence wins
    return list(out.values())


def cellmean(rs, field=lambda r: r["removal"]):
    """{(target, axis, variant): mean over seeds of field, nan -> 0}."""
    acc = collections.defaultdict(list)
    for r in rs:
        acc[(r["target"], r["axis"], r["variant"])].append(z(field(r)))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def tier_of(t):
    for name, ts in TIERS.items():
        if t in ts:
            return name
    return "other"


# --------------------------------------------------------------- replay ----
def check_replay(tol=0.0):
    pairs = []

    def cmp(new_panel, ref_root, ref_panel, vmap, only=None):
        new = {(r["target"], r["axis"], r["seed"], r["variant"]): r
               for r in rows(new_panel)}
        ref = {(r["target"], r["axis"], r["seed"],
                r.get("variant", r.get("condition"))): r
               for r in rows(ref_panel, root=ref_root)}
        for (t, a, s, vn), r in new.items():
            rv = vmap.get(vn, vn if vn.startswith("s") else None)
            if rv is None or (only and not only(t, a)):
                continue
            k = (t, a, s, rv)
            if k in ref:
                pairs.append((f"{new_panel}:{t}|{a}|s{s}|{vn}", z(r["removal"]),
                              z(ref[k]["removal"])))

    v9 = os.path.join(REPO, "results_v9")
    cmp("v12cal", os.path.join(v9, "v8spc"), "small", {})
    cmp("v12cal", os.path.join(v9, "v8spc"), "big", {})
    cmp("v12dec1k", v9, "v8dec", {"s0.99": "C-ref", "C-a@0.01": "C-a"})
    cmp("v12big", RESULTS, "v11big", {"s0.99": "C-ref"},
        only=lambda t, a: a == "occ_gender")
    if not pairs:
        print("no replayed rows yet")
        return True
    bad = [(k, n, r) for k, n, r in pairs if abs(n - r) > tol]
    worst = max(abs(n - r) for _k, n, r in pairs)
    print(f"replayed rows: {len(pairs)}   max |delta| = {worst:.3g}   "
          f"over tol {tol}: {len(bad)}")
    for k, n, r in bad[:20]:
        print(f"  {k:50s} new={n:+.6f} ref={r:+.6f} d={n - r:+.2e}")
    print("REPLAY " + ("PASS" if not bad else "FAIL — do not report the new "
                       "panels until this is explained"))
    return not bad


# ------------------------------------------------------------- readings ----
def reading_sparsity(allrows):
    cm = cellmean(allrows)
    cells = sorted({(t, a) for t, a, _v in cm})
    variants = sorted({v for _t, _a, v in cm if v.startswith("s")},
                      key=lambda v: float(v[1:]))
    out = {}
    for tier in TIERS:
        tc = [c for c in cells if tier_of(c[0]) == tier]
        if not tc:
            continue
        d = dict(cells=["|".join(c) for c in tc], vs_dense={}, retention={})
        for vn in variants:
            pr = [(cm[(t, a, vn)], cm[(t, a, "s0.0")]) for t, a in tc
                  if (t, a, vn) in cm and (t, a, "s0.0") in cm]
            if pr:
                d["vs_dense"][vn] = V.boot_paired(pr)
                dense = sum(b for _a, b in pr)
                d["retention"][vn] = (sum(a for a, _b in pr) / dense) if dense else None
        pr = [(cm[(t, a, "s0.0")], cm[(t, a, "fp")]) for t, a in tc
              if (t, a, "fp") in cm and (t, a, "s0.0") in cm]
        if pr:
            fp_sum = sum(b for _a, b in pr)
            d["binarization"] = dict(deficit=V.boot_paired(pr),
                                     retention=(sum(a for a, _b in pr) / fp_sum)
                                     if fp_sum else None)
        out[tier] = d
    return out


def reading_adaptive(allrows):
    per = collections.defaultdict(list)
    for r in allrows:
        ad = r.get("adaptive")
        if ad is None:
            continue
        per[r["variant"]].append(r)
    out = {}
    for vn, rs in sorted(per.items()):
        fz = cellmean(rs)
        ad = cellmean(rs, lambda r: (r["adaptive"] or {}).get("removal"))
        pr = [(ad[k], fz[k]) for k in fz if k in ad]
        evals = [r["adaptive"]["eval_ok"] for r in rs
                 if r["adaptive"].get("alpha_star") is not None]
        stat = collections.Counter(r["adaptive"]["status"] for r in rs)
        alphas = sorted(r["adaptive"]["alpha_star"] for r in rs
                        if r["adaptive"].get("alpha_star") is not None)
        out[vn] = dict(
            adaptive_minus_frozen=V.boot_paired(pr),
            eval_budget_pass_rate=(sum(evals) / len(evals)) if evals else None,
            n_units=len(rs), status=dict(stat),
            alpha_star_median=alphas[len(alphas) // 2] if alphas else None,
            alpha_star_range=[alphas[0], alphas[-1]] if alphas else None)
    return out


def reading_selection(allrows):
    out = {}
    for rule, field in (("frozen", lambda r: r["removal"]),
                        ("adaptive", lambda r: (r.get("adaptive") or {}).get("removal"))):
        cm = cellmean(allrows, field)
        pr = [(cm[(t, a, "s0.99")], cm[(t, a, "C-a@0.01")])
              for t, a, v in cm if v == "s0.99" and (t, a, "C-a@0.01") in cm]
        out[rule] = V.boot_paired(pr)
    return out


def reading_joint(big):
    fz = cellmean(big)
    ad = cellmean(big, lambda r: (r.get("adaptive") or {}).get("removal"))
    out = {}
    for t, a in sorted({(t, a) for t, a, _v in fz}):
        if (t, a, "pstar") not in ad:
            continue
        grid = {v: fz[(t, a, v)] for tt, aa, v in fz
                if (tt, aa) == (t, a) and (v.startswith("s") or v == "fp")}
        best_v = max(grid, key=grid.get) if grid else None
        out[f"{t}|{a}"] = dict(
            joint=ad[(t, a, "pstar")],
            frozen_1pct=fz.get((t, a, "s0.99")),
            oracle_grid=grid.get(best_v), oracle_variant=best_v,
            regret_vs_oracle=(grid.get(best_v) - ad[(t, a, "pstar")])
            if best_v else None)
    return out


def reading_mmlu1k(allrows):
    flips, n, argmax_changes, n_all = [], 0, [], 0
    for r in allrows:
        k1 = r.get("mmlu1k_frozen")
        if k1 and "ok" in k1 and r.get("alpha_frozen") is not None:
            n += 1
            if not k1["ok"]:
                flips.append(f"{r['target']}|{r['axis']}|s{r['seed']}|{r['variant']}"
                             f"@{r['alpha_frozen']}: {k1['dmmlu_items']}/1000 items")
        if k1 and "argmax_1k_alpha" in k1:
            n_all += 1
            if k1["argmax_1k_alpha"] != r.get("alpha_frozen"):
                argmax_changes.append(
                    f"{r['target']}|{r['axis']}|s{r['seed']}|{r['variant']}: "
                    f"α {r.get('alpha_frozen')} -> {k1['argmax_1k_alpha']}")
        ad = r.get("adaptive") or {}
        if "mmlu1k" in ad:
            n += 1
            if not ad["mmlu1k"]["ok"]:
                flips.append(f"{r['target']}|{r['axis']}|s{r['seed']}|{r['variant']}"
                             f"@α*={ad['alpha_star']:.1f}: {ad['mmlu1k']['dmmlu_items']}/1000")
    # Delta_selection with the 1000-item argmax, where every α was read at 1000
    k1rows = [dict(r, removal=r["mmlu1k_frozen"]["argmax_1k_removal"])
              for r in allrows
              if "argmax_1k_removal" in (r.get("mmlu1k_frozen") or {})]
    sel = reading_selection([dict(r, adaptive=None) for r in k1rows])["frozen"] \
        if k1rows else None
    return dict(points_read=n, fail_at_1000=len(flips), failures=flips,
                units_with_full_1k_trace=n_all,
                argmax_changes_at_1000=argmax_changes,
                delta_selection_under_1000_gate=sel)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-replay", action="store_true")
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.check_replay:
        return 0 if check_replay(a.tol) else 1

    cal, big, d1k = (rows(p) for p in PANELS)
    out_fp = os.path.join(RESULTS, "v12", "opsel_summary.json")
    allrows = cal + big
    out = dict(
        panels={p: len(rows(p)) for p in PANELS},
        sparsity_by_tier=reading_sparsity(allrows),
        adaptive_vs_frozen=dict(calibration=reading_adaptive(cal),
                                heldout=reading_adaptive(big)),
        delta_selection=dict(calibration=reading_selection(cal),
                             heldout=reading_selection(big)),
        joint_operating_point=reading_joint(big),
        mmlu1k=dict(dec=reading_mmlu1k(d1k), calibration=reading_mmlu1k(cal),
                    heldout=reading_mmlu1k(big)),
    )
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    json.dump(out, open(out_fp, "w"), indent=1)

    print("v12 OPSEL — readings (cell unit, 10k bootstrap)\n")
    for tier, d in out["sparsity_by_tier"].items():
        print(f"[{tier}] {len(d['cells'])} cells")
        if "binarization" in d:
            b = d["binarization"]
            print(f"   binarization  retention {b['retention']:.3f}   "
                  f"s0.0 - fp {V.fmt(b['deficit'], 3)}")
        for vn, ci in d["vs_dense"].items():
            if vn != "s0.0":
                print(f"   {vn:8s} - dense  {V.fmt(ci, 3)}   retention "
                      f"{d['retention'][vn]:.3f}" if d['retention'][vn] is not None
                      else f"   {vn:8s} - dense  {V.fmt(ci, 3)}")
    for pop in ("calibration", "heldout"):
        print(f"\nadaptive α* vs frozen argmax ({pop}):")
        for vn, d in out["adaptive_vs_frozen"][pop].items():
            pr = d["eval_budget_pass_rate"]
            print(f"   {vn:9s} {V.fmt(d['adaptive_minus_frozen'], 3)}  eval pass "
                  f"{'-' if pr is None else f'{pr:.2f}'}  status {d['status']}")
        ds = out["delta_selection"][pop]
        print(f"   Δ_selection frozen   {V.fmt(ds['frozen'], 3)}")
        print(f"   Δ_selection adaptive {V.fmt(ds['adaptive'], 3)}")
    for c, d in out["joint_operating_point"].items():
        print(f"\njoint (p*, α*) {c}: {d['joint']:+.3f}  frozen 1% "
              f"{z(d['frozen_1pct']):+.3f}  oracle {z(d['oracle_grid']):+.3f} "
              f"({d['oracle_variant']})")
    for pop, d in out["mmlu1k"].items():
        print(f"\nMMLU-1000 [{pop}]: {d['fail_at_1000']}/{d['points_read']} selected "
              f"points fail at 1000; argmax changes {len(d['argmax_changes_at_1000'])}"
              f"/{d['units_with_full_1k_trace']}")
        if d["delta_selection_under_1000_gate"]:
            print(f"   Δ_selection under the 1000-item gate "
                  f"{V.fmt(d['delta_selection_under_1000_gate'], 3)}")
    print(f"\nwrote {out_fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
