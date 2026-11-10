#!/usr/bin/env python3
"""R1 E7b -- DPO as measured by the corrected harness (run_r1_dpo.py).

    python3 src/r1_e7b.py            -> results/r1/E7b/e7b.json
    python3 src/r1_e7b.py progress

TWO THINGS ARE READ, AND THEY ARE KEPT APART.

  (1) Arm `pub`: the as-published recipe (lr 1e-4, 250 steps), measured with the
      baseline taken before training and the update added to a clean model. It
      replaces the invalid published column and checks the E7a reconstruction.
      Reported under the published selection space (merge scales
      {0.25, 0.5, 0.75, 1}, cells with < 20 differing pairs scored 0) and, next
      to it, with every trained cell (>= 8 pairs) and with the finer merge
      ladder. Not a registered reading.

  (2) Arms `lr1e-5`, `lr5e-5`: the registered E7b design. Selection space
      {lr 1e-5, 5e-5} x merge {1, 2}, skip threshold 8 pairs. Registered reading
      (R1_REBUTTAL_EXPERIMENTS.md, E7):
          E7-1  edit - DPO covers 0   -> "DPO removes none" is withdrawn in full
                                         and replaced by "DPO is comparable"
          E7-2  edit - DPO excludes 0 and DPO ~ 0 -> the claim stays, out of
                                         the abstract
      Read only when all 24 units of both arms are on disk.

Conventions as everywhere: best in-budget removal over the selection space,
none in budget -> 0, cell = seed mean, bootstrap over the 8 cells in
v12_frontier order; the edit's cell values are the published Table A ones
(results/v12/frontier_v12.json).

DEGENERACY CHECK (occupation axis). The occupation skew is 0 for any model whose
pronoun preference does not depend on the occupation, including one that always
prefers the same pronoun. From the stored per-item log-likelihoods this script
reports the mean P("she") on female- and on male-stereotyped probe occupations
before and after DPO, and flags a run as collapsed when both are below 0.05 or
both above 0.95 after the update.
"""
import ast
import collections
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V    # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E7b")
CELLS = [(t, a) for t in ("gemma", "llama", "qwen", "phi")
         for a in ("occ_gender", "crows_socioeconomic")]
DESIGNER = dict(gemma="gemma_3b", llama="llama_3b", qwen="qwen_3b", phi="phi_3b")
SEEDS = (0, 1, 2)
PUB_MERGE = (0.25, 0.5, 0.75, 1.0)
PUB_MIN_PAIRS = 20


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and x != x) else float(x)


def load():
    rows = {}
    for r in jl(os.path.join(OUT, "removal.jsonl")):
        rows[(r["target"], r["axis"], r["seed"], r["arm"])] = r
    return rows


def best(points, merges=None):
    ok = [p for p in points if p["ok"] and (merges is None or p["merge"] in merges)]
    return max(ok, key=lambda p: p["removal"]) if ok else None


def cell_mean(per_seed):
    acc = collections.defaultdict(list)
    for (t, a, _s), v in per_seed.items():
        acc[(t, a)].append(z(v))
    return {c: (float(np.mean(acc[c])) if acc.get(c) else 0.0) for c in CELLS}


def probe_occ():
    src = open(os.path.join(REPO, "colab_t2t4.py"), encoding="utf-8").read()
    occ = ast.literal_eval(re.search(r"^PROBE_OCC = (.*)$", src, re.M).group(1))
    tpl = ast.literal_eval(re.search(r"^PROBE_TEMPLATES = (.*)$", src, re.M).group(1))
    return list(occ.items()), len(tpl)


def pshe(ll, occ, n_tpl):
    """Mean P('she') over templates, separately for f- and m-stereotyped occupations."""
    d = ll[1::2] - ll[0::2]                        # she - he, per (occupation, template)
    p = 1.0 / (1.0 + np.exp(-d.astype(np.float64)))
    p = p.reshape(len(occ), n_tpl).mean(1)
    g = np.array([x[1] for x in occ])
    return float(p[g == "f"].mean()), float(p[g == "m"].mean())


def degeneracy(rows, arm):
    occ, n_tpl = probe_occ()
    out = {}
    for (t, a, s, an), r in rows.items():
        if an != arm or a != "occ_gender" or r.get("skipped"):
            continue
        fp = os.path.join(OUT, "probe_ll", f"{t}__{a}__s{s}__{DESIGNER[t]}__{arm}.npz")
        if not os.path.exists(fp):
            continue
        zf = np.load(fp)
        pre_f, pre_m = pshe(zf["pre_ll"], occ, n_tpl)
        per = []
        for m, ll in zip(zf["merge"], zf["ll"]):
            f, mm = pshe(ll, occ, n_tpl)
            per.append(dict(merge=float(m), pshe_f=f, pshe_m=mm,
                            collapsed=bool((f < 0.05 and mm < 0.05) or (f > 0.95 and mm > 0.95))))
        out[f"{t}|{a}|s{s}"] = dict(pre_pshe_f=pre_f, pre_pshe_m=pre_m, by_merge=per)
    return out


def progress():
    rows = load()
    for arm in ("pub", "lr1e-5", "lr5e-5"):
        n = sum((t, a, s, arm) in rows for t, a in CELLS for s in SEEDS)
        print(f"r1/E7b {arm:7s} {n}/24 units on disk")


def read_pub(rows, edit):
    arm = "pub"
    need = [(t, a, s, arm) for t, a in CELLS for s in SEEDS]
    if any(k not in rows for k in need):
        return None
    restored = all(rows[k].get("baseline_restored") for k in need if not rows[k].get("skipped"))
    variants = collections.OrderedDict()
    spaces = [("published selection (merge 0.25-1, <20 pairs -> 0)", PUB_MERGE, PUB_MIN_PAIRS),
              ("every trained cell (>= 8 pairs), merge 0.25-1", PUB_MERGE, 8),
              ("every trained cell, finer ladder 0.0625-1", None, 8)]
    for label, merges, min_pairs in spaces:
        per = {}
        for t, a, s, _ in need:
            r = rows[(t, a, s, arm)]
            if r.get("skipped") or r["n_informative"] < min_pairs:
                per[(t, a, s)] = 0.0
                continue
            b = best(r["points"], merges)
            per[(t, a, s)] = b["removal"] if b else float("nan")
        cm = cell_mean(per)
        variants[label] = dict(
            cells={f"{t}|{a}": cm[(t, a)] for t, a in CELLS},
            dpo=V.boot_mean([cm[c] for c in CELLS]),
            edit_minus_dpo=V.boot_paired([(z(edit[c]), cm[c]) for c in CELLS]),
            runs_with_nothing_in_budget=sum(1 for v in per.values() if v != v))
    full, pts_in, pts_all = {}, 0, 0
    per_cell_full = collections.defaultdict(list)
    for t, a, s, _ in need:
        r = rows[(t, a, s, arm)]
        if r.get("skipped"):
            continue
        p1 = [p for p in r["points"] if p["merge"] == 1.0][0]
        per_cell_full[(t, a)].append(p1)
        pub_pts = [p for p in r["points"] if p["merge"] in PUB_MERGE]
        pts_in += sum(p["ok"] for p in pub_pts)
        pts_all += len(pub_pts)
    for c, ps in per_cell_full.items():
        full[f"{c[0]}|{c[1]}"] = dict(
            removal=float(np.mean([p["removal"] for p in ps])),
            ppl_ratio=float(np.mean([p["ppl_ratio"] for p in ps])),
            dmmlu_items=float(np.mean([p["dmmlu_items"] for p in ps])),
            in_budget=sum(p["ok"] for p in ps), runs=len(ps),
            abs_post_skew=float(np.mean([abs(p["post_skew"]) for p in ps])))
    n_pairs = {f"{t}|{a}": sorted({rows[(t, a, s, arm)]["n_informative"] for s in SEEDS})
               for t, a in CELLS}
    return dict(baseline_restored_in_every_run=bool(restored), n_differing_pairs=n_pairs,
                selection=variants, at_trained_scale=full,
                published_merge_configs_in_budget=f"{pts_in}/{pts_all}",
                degeneracy=degeneracy(rows, arm))


def read_registered(rows, edit):
    arms = ("lr1e-5", "lr5e-5")
    need = [(t, a, s, an) for t, a in CELLS for s in SEEDS for an in arms]
    if any(k not in rows for k in need):
        return None
    per = {}
    for t, a in CELLS:
        for s in SEEDS:
            pts = []
            for an in arms:
                r = rows[(t, a, s, an)]
                if not r.get("skipped"):
                    pts += [dict(p, arm=an) for p in r["points"]]
            b = best(pts) if pts else None
            per[(t, a, s)] = (0.0 if not pts else (b["removal"] if b else float("nan")))
    cm = cell_mean(per)
    D = V.boot_paired([(z(edit[c]), cm[c]) for c in CELLS])
    M = V.boot_mean([cm[c] for c in CELLS])
    covers = not (D["lo"] > 0 or D["hi"] < 0)
    dpo_zero = not (M["lo"] > 0 or M["hi"] < 0)
    reading = "E7-1" if covers else ("E7-2" if dpo_zero else
                                     "neither (edit - DPO excludes 0, DPO excludes 0)")
    steps = sorted({rows[k]["train"]["global_step"] for k in need if not rows[k].get("skipped")})
    return dict(cells={f"{t}|{a}": cm[(t, a)] for t, a in CELLS}, dpo=M, edit_minus_dpo=D,
                reading=reading, optimizer_steps_range=[steps[0], steps[-1]] if steps else None)


def main():
    rows = load()
    fr = json.load(open(os.path.join(RESULTS, "v12", "frontier_v12.json")))
    edit = {tuple(k.split("|")): v["edit"] for k, v in fr["table_a"].items()}
    pub = read_pub(rows, edit)
    if pub is None:
        progress()
        print("incomplete: arm `pub` is not read before all 24 units exist")
        return 1
    reg = read_registered(rows, edit)
    e7a = None
    fp = os.path.join(RESULTS, "r1", "E7", "e7a_dpo.json")
    if os.path.exists(fp):
        a = json.load(open(fp))
        e7a = {k: v["dpo_best_in_budget"] for k, v in a["per_cell"].items()}
    out = dict(script="src/r1_e7b.py", edit_cells={f"{t}|{a}": z(edit[(t, a)]) for t, a in CELLS},
               pub=pub, registered=reg, e7a_reconstruction_best_in_budget=e7a)
    with open(os.path.join(OUT, "e7b.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"arm pub: baseline restored after unloading in every run: "
          f"{pub['baseline_restored_in_every_run']}; published merge configurations in "
          f"budget: {pub['published_merge_configs_in_budget']}")
    for label, v in pub["selection"].items():
        print(f"\n  [{label}]")
        print(f"    DPO        {V.fmt(v['dpo'], 3)}")
        print(f"    edit - DPO {V.fmt(v['edit_minus_dpo'], 3)}")
    sel = list(pub["selection"].values())
    print("\n  per cell (seed means): pairs | edit | DPO pub-sel | DPO all cells | DPO finer | "
          "E7a reconstr. || trained scale: removal, ppl x, in budget")
    for t, a in CELLS:
        k = f"{t}|{a}"
        f_ = pub["at_trained_scale"].get(k)
        tail = (f"{f_['removal']:+.3f}, x{f_['ppl_ratio']:.2f}, {f_['in_budget']}/{f_['runs']}"
                if f_ else "n/a")
        rec = "   n/a" if e7a is None else f"{e7a[k]:+.3f}"
        print(f"    {k:26s} {str(pub['n_differing_pairs'][k]):10s} {out['edit_cells'][k]:+.3f} | "
              f"{sel[0]['cells'][k]:+.3f} | {sel[1]['cells'][k]:+.3f} | {sel[2]['cells'][k]:+.3f} | "
              f"{rec} || {tail}")
    print("\n  occupation probe, mean P(she) on female- / male-stereotyped occupations "
          "(before -> at trained scale):")
    for k, d in pub["degeneracy"].items():
        last = d["by_merge"][-1]
        first_c = next((x["merge"] for x in d["by_merge"] if x["collapsed"]), None)
        print(f"    {k:22s} {d['pre_pshe_f']:.2f} / {d['pre_pshe_m']:.2f}  ->  "
              f"{last['pshe_f']:.2f} / {last['pshe_m']:.2f}"
              + (f"   COLLAPSED from merge {first_c}" if first_c is not None else ""))
    if reg is None:
        print("\nregistered arms (lr1e-5, lr5e-5): not complete; not read")
    else:
        print(f"\nregistered E7b design ({{lr 1e-5, 5e-5}} x merge {{1, 2}}, >= 8 pairs; "
              f"optimizer steps {reg['optimizer_steps_range']}):")
        print(f"    DPO        {V.fmt(reg['dpo'], 3)}")
        print(f"    edit - DPO {V.fmt(reg['edit_minus_dpo'], 3)}")
        print(f"    REGISTERED READING: {reg['reading']}")
    print(f"\nwrote {os.path.relpath(os.path.join(OUT, 'e7b.json'), REPO)}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "read"
    sys.exit({"read": main, "progress": progress}[cmd]() or 0)
