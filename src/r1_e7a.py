#!/usr/bin/env python3
"""R1 E7a -- the published DPO column, re-read from its own artifacts (zero GPU).

    python3 src/r1_e7a.py            -> results/r1/E7/e7a_dpo.json (+ a per-cell CSV)

WHAT THIS FOUND. run_dpo_baseline.py (the script behind Table A's DPO row)
measures the wrong contrast, and the persisted panel shows it without a rerun:

  * `dpo_delta` hands the model to TRL's DPOTrainer with a peft_config. peft
    injects the LoRA layers into that model object in place, so after training
    the object the script calls `model` still carries the trained adapter.
  * `pre = fast_eval(model, ...)` is therefore the skew, MMLU and perplexity of
    the model WITH the full DPO adapter, not of the unedited model.
  * `apply_edit(model, dw, alpha=a, sign=-1.0)` then SUBTRACTS a*dW from the
    base weights underneath the still-active adapter, so the "post" model is
    base + (1 - a) * dW: at a = 1 it is (up to bf16 rounding) the unedited model.

So the recorded "removal at alpha = a" is |S(base + dW)| - |S(base + (1-a) dW)|:
the published number is the bias REGAINED when the DPO update is undone, with
the sign of a removal. Three things in results/v6trace/dpo confirm it:
  (1) removal.jsonl's pre_skew differs from the unedited skew of the same model
      and axis measured by every other panel (llama|occ: 0.0000 vs 0.7605);
  (2) pre_skew - removal(alpha=1) reproduces that unedited skew to ~1e-3;
  (3) the trace's ppl_ratio is below 1 and falls with alpha (the "edit" makes
      the model MORE fluent, because it is removing the DPO update).

WHAT THIS RECONSTRUCTS. Every quantity needed for the intended comparison is in
the persisted rows. With S_dpo = pre_skew, rec(a) = recorded removal and the
trace's dmmlu(a), ppl_ratio(a), for merge scale m of the DPO delta:

    m in {0.25, 0.5, 0.75}:  |S(m)| = |S_dpo| - rec(1 - m)
                             MMLU drop = dmmlu(1 - m) - dmmlu(1)       (items/200)
                             ppl ratio = ppl_ratio(1 - m) / ppl_ratio(1)
    m = 1 (the trained adapter):  |S(1)| = |S_dpo|
                             MMLU drop = -dmmlu(1),  ppl ratio = 1 / ppl_ratio(1)
    removal(m) = |S_base| - |S(m)|

The four merge scales {0.25, 0.5, 0.75, 1.0} are the selection space the script
documents, so the published rule (best in-budget of four configurations, integer
gate, nan -> 0, skipped cells 0, cell = seed mean, cell bootstrap) applies as is.
S_base is the unedited skew measured in the same run (pre_skew - rec(1)); the
clean value from results/v12frontier is reported next to it as a check.

THIS IS A RECONSTRUCTION, NOT A RE-MEASUREMENT: the in-run "unedited" model is
adapter-plus-subtraction in bf16, and collateral is known only relative to it.
E7b re-runs DPO with the baseline taken before training and the delta added.

SANITY GATE. Before any corrected number, the loader must reproduce the
published values from the same rows: DPO +0.006 [-0.064, +0.095] and
edit - DPO +0.276 [+0.027, +0.513] (results/v12/frontier_v12.json).
"""
import collections
import csv
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v9_gate            # noqa: E402
import v10_common as V    # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "r1", "E7")
CELLS = [(t, a) for t in ("gemma", "llama", "qwen", "phi")
         for a in ("occ_gender", "crows_socioeconomic")]          # v12_frontier.CELLS
SEEDS = (0, 1, 2)
MERGE = (0.25, 0.5, 0.75, 1.0)
PPL_BUDGET, PPL_TOL, ITEMS = 1.10, 1e-9, 4


def jl(fp):
    return [json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and x != x) else float(x)


def load():
    rem = jl(os.path.join(RESULTS, "v6trace", "dpo", "removal.jsonl"))
    tr = jl(os.path.join(RESULTS, "v6trace", "dpo", "alpha_trace.jsonl"))
    rows, skipped, trace = {}, {}, {}
    for r in rem:                                   # last occurrence wins
        k = (r["target"], r["axis"], r["seed"])
        if r.get("skipped"):
            skipped[k] = dict(n_informative=r.get("n_informative"), n_total=r.get("n_total"))
        else:
            rows[k + (float(r["alpha"]),)] = r
    for r in tr:
        trace[(r["target"], r["axis"], r["seed"], float(r["alpha"]))] = r
    return rows, skipped, trace


def clean_baselines():
    """Unedited skew per cell from the frontier re-measurement (same eval batch)."""
    out = collections.defaultdict(set)
    for r in jl(os.path.join(RESULTS, "v12frontier", "alpha_trace.jsonl")):
        if r.get("pre_skew") is not None:
            out[(r["target"], r["axis"])].add(r["pre_skew"])
    return {k: sorted(v) for k, v in out.items()}


def edit_cells():
    fr = json.load(open(os.path.join(RESULTS, "v12", "frontier_v12.json")))
    return ({tuple(k.split("|")): v["edit"] for k, v in fr["table_a"].items()},
            {tuple(k.split("|")): v["dpo"] for k, v in fr["table_a"].items()}, fr["pooled"])


def as_published(rows, trace):
    """The published reading: best in-budget recorded removal over the 4 alphas."""
    per = {}
    for (t, a, s, al), r in rows.items():
        ok, _mode = v9_gate.collateral_ok_row(trace[(t, a, s, al)])
        k = (t, a, s)
        per.setdefault(k, float("nan"))
        if ok:
            br = trace[(t, a, s, al)]["bias_reduction"]
            per[k] = br if per[k] != per[k] else max(per[k], br)
    return per


def reconstruct(rows, trace, base_clean):
    """Per (cell, seed): the intended contrast at each merge scale."""
    out = {}
    for (t, a, s) in sorted({k[:3] for k in rows}):
        r1, t1 = rows[(t, a, s, 1.0)], trace[(t, a, s, 1.0)]
        s_dpo = abs(r1["pre_skew"])
        s_base_run = s_dpo - r1["removal"]                    # |S| of the in-run "unedited"
        clean = base_clean.get((t, a), [])
        pts = []
        for m in MERGE:
            if m == 1.0:
                s_m = s_dpo
                drop = -t1["dmmlu"] * 200
                ppl = 1.0 / t1["ppl_ratio"]
            else:
                al = round(1.0 - m, 2)
                s_m = s_dpo - rows[(t, a, s, al)]["removal"]
                drop = (trace[(t, a, s, al)]["dmmlu"] - t1["dmmlu"]) * 200
                ppl = trace[(t, a, s, al)]["ppl_ratio"] / t1["ppl_ratio"]
            if abs(drop - round(drop)) > 1e-6:
                raise RuntimeError(f"non-integral MMLU drop {drop} at {(t, a, s, m)}")
            drop = int(round(drop))
            ok = bool(drop <= ITEMS and ppl <= PPL_BUDGET + PPL_TOL)
            pts.append(dict(merge=m, abs_skew=s_m, removal=s_base_run - s_m,
                            removal_vs_clean_base=(abs(clean[0]) - s_m if len(clean) == 1
                                                   else None),
                            dmmlu_items=drop, ppl_ratio=ppl, in_budget=ok))
        inb = [p for p in pts if p["in_budget"]]
        best = max(inb, key=lambda p: p["removal"]) if inb else None
        out[(t, a, s)] = dict(
            abs_skew_with_full_dpo=s_dpo, abs_skew_unedited_in_run=s_base_run,
            abs_skew_unedited_clean=(abs(clean[0]) if len(clean) == 1 else clean),
            points=pts, best_in_budget=(best["removal"] if best else float("nan")),
            best_merge=(best["merge"] if best else None),
            removal_full_adapter=pts[-1]["removal"],
            full_adapter_in_budget=pts[-1]["in_budget"],
            best_any_budget=max(p["removal"] for p in pts))
    return out


def cell_mean(per_seed, cells):
    acc = collections.defaultdict(list)
    for (t, a, _s), v in per_seed.items():
        acc[(t, a)].append(z(v))
    return {c: (float(np.mean(acc[c])) if acc.get(c) else 0.0) for c in cells}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, skipped, trace = load()
    edit, dpo_pub_cells, pooled_pub = edit_cells()
    trainable = sorted({k[:2] for k in rows}, key=CELLS.index)
    skipped_cells = sorted({k[:2] for k in skipped}, key=CELLS.index)

    # ---- sanity gate: the published numbers, from the same rows ----
    pub = cell_mean(as_published(rows, trace), CELLS)
    pub_mean = V.boot_mean([pub[c] for c in CELLS])
    pub_diff = V.boot_paired([(z(edit[c]), pub[c]) for c in CELLS])
    ref = pooled_pub["edit - DPO"]
    # The published artifact stores DPO cell values rounded to 6 decimals, so the
    # comparison is to 2e-6, not to machine precision.
    tol = 2e-6
    gate_ok = (abs(pub_diff["point"] - ref["point"]) < tol
               and abs(pub_diff["lo"] - ref["lo"]) < tol
               and abs(pub_diff["hi"] - ref["hi"]) < tol
               and all(abs(pub[c] - z(dpo_pub_cells[c])) < tol for c in CELLS))
    print(f"sanity gate (published DPO reproduced from the rows): {gate_ok}")
    print(f"  as published: DPO {V.fmt(pub_mean, 3)};  edit - DPO {V.fmt(pub_diff, 3)}")
    if not gate_ok:
        raise SystemExit("cannot reproduce the published DPO numbers; not proceeding")

    # ---- the three confirmations that 'pre' carried the adapter ----
    base_clean = clean_baselines()
    rec = reconstruct(rows, trace, base_clean)
    conf = []
    for (t, a, s), d in rec.items():
        c = d["abs_skew_unedited_clean"]
        tr1 = trace[(t, a, s, 1.0)]
        conf.append(dict(
            cell=f"{t}|{a}", seed=s, skew_recorded_as_pre=d["abs_skew_with_full_dpo"],
            unedited_skew_other_panels=c,
            pre_minus_removal_at_alpha1=d["abs_skew_unedited_in_run"],
            gap_to_unedited=(d["abs_skew_unedited_in_run"] - c if isinstance(c, float)
                             else None),
            ppl_ratio_by_alpha=[trace[(t, a, s, al)]["ppl_ratio"]
                                for al in (0.25, 0.5, 0.75, 1.0)],
            ppl_of_dpo_model_over_unedited=1.0 / tr1["ppl_ratio"]))
    gaps = [abs(x["gap_to_unedited"]) for x in conf if x["gap_to_unedited"] is not None]
    pre_diff = [abs(x["skew_recorded_as_pre"] - x["unedited_skew_other_panels"])
                for x in conf if isinstance(x["unedited_skew_other_panels"], float)]
    print(f"\n'pre' was not the unedited model: |recorded pre-skew - unedited skew| "
          f"median {np.median(pre_diff):.3f}, max {max(pre_diff):.3f} over {len(pre_diff)} runs")
    print(f"pre-skew - removal(alpha=1) recovers the unedited skew: max |gap| {max(gaps):.4f}")
    mono = sum(all(x["ppl_ratio_by_alpha"][i] >= x["ppl_ratio_by_alpha"][i + 1] - 1e-12
                   for i in range(3)) for x in conf)
    print(f"recorded ppl ratio is non-increasing in alpha in {mono}/{len(conf)} runs; "
          f"ppl(DPO model)/ppl(unedited) median "
          f"{np.median([x['ppl_of_dpo_model_over_unedited'] for x in conf]):.3f}")

    # ---- the intended contrast ----
    best = cell_mean({k: d["best_in_budget"] for k, d in rec.items()}, CELLS)
    full = cell_mean({k: d["removal_full_adapter"] for k, d in rec.items()}, CELLS)
    anyb = cell_mean({k: d["best_any_budget"] for k, d in rec.items()}, CELLS)
    out = dict(
        script="src/r1_e7a.py",
        label="post-hoc reconstruction from persisted rows; E7b re-measures",
        sanity_gate=dict(reproduced=gate_ok, dpo_as_published=pub_mean,
                         edit_minus_dpo_as_published=pub_diff),
        cells=[f"{t}|{a}" for t, a in CELLS],
        trainable_cells=[f"{t}|{a}" for t, a in trainable],
        skipped_cells={f"{t}|{a}": [skipped[(t, a, s)] for s in SEEDS if (t, a, s) in skipped]
                       for t, a in skipped_cells},
        confirmations=conf,
        per_run={f"{t}|{a}|s{s}": d for (t, a, s), d in rec.items()},
        per_cell={f"{t}|{a}": dict(edit=z(edit[(t, a)]), dpo_as_published=pub[(t, a)],
                                   dpo_best_in_budget=best[(t, a)],
                                   dpo_full_adapter_ignoring_budget=full[(t, a)],
                                   dpo_best_ignoring_budget=anyb[(t, a)],
                                   skipped=(t, a) in skipped_cells)
                  for t, a in CELLS},
        corrected=dict(
            all8=dict(dpo=V.boot_mean([best[c] for c in CELLS]),
                      edit_minus_dpo=V.boot_paired([(z(edit[c]), best[c]) for c in CELLS])),
            trainable5=dict(dpo=V.boot_mean([best[c] for c in trainable]),
                            edit=V.boot_mean([z(edit[c]) for c in trainable]),
                            edit_minus_dpo=V.boot_paired([(z(edit[c]), best[c])
                                                          for c in trainable])),
            ignoring_budget_all8=dict(
                dpo_full_adapter=V.boot_mean([full[c] for c in CELLS]),
                dpo_best_of_four=V.boot_mean([anyb[c] for c in CELLS])),
            budget=dict(
                runs=len(rec),
                full_adapter_in_budget=sum(d["full_adapter_in_budget"] for d in rec.values()),
                any_merge_scale_in_budget=sum(d["best_merge"] is not None
                                              for d in rec.values()),
                configs_in_budget=sum(p["in_budget"] for d in rec.values()
                                      for p in d["points"]),
                configs=4 * len(rec))))
    with open(os.path.join(OUT, "e7a_dpo.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT, "e7a_dpo_per_run.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "seed", "merge_scale", "abs_skew", "removal", "dmmlu_items",
                    "ppl_ratio", "in_budget", "abs_skew_unedited_in_run",
                    "abs_skew_unedited_clean"])
        for (t, a, s), d in rec.items():
            for p in d["points"]:
                w.writerow([f"{t}|{a}", s, p["merge"], f"{p['abs_skew']:.6f}",
                            f"{p['removal']:.6f}", p["dmmlu_items"], f"{p['ppl_ratio']:.6f}",
                            int(p["in_budget"]), f"{d['abs_skew_unedited_in_run']:.6f}",
                            d["abs_skew_unedited_clean"]])

    c = out["corrected"]
    print("\nreconstructed, intended contrast (best in-budget of merge scales "
          "{0.25,0.5,0.75,1}; skipped cells 0):")
    print(f"  DPO, 8 cells            {V.fmt(c['all8']['dpo'], 3)}")
    print(f"  edit - DPO, 8 cells     {V.fmt(c['all8']['edit_minus_dpo'], 3)}")
    print(f"  DPO, 5 trainable cells  {V.fmt(c['trainable5']['dpo'], 3)}")
    print(f"  edit, same 5 cells      {V.fmt(c['trainable5']['edit'], 3)}")
    print(f"  edit - DPO, 5 cells     {V.fmt(c['trainable5']['edit_minus_dpo'], 3)}")
    b = c["budget"]
    print(f"  budget: full adapter in budget in {b['full_adapter_in_budget']}/{b['runs']} "
          f"runs; some merge scale in budget in {b['any_merge_scale_in_budget']}/{b['runs']}; "
          f"{b['configs_in_budget']}/{b['configs']} configurations in budget")
    print(f"  ignoring the budget: full adapter "
          f"{V.fmt(c['ignoring_budget_all8']['dpo_full_adapter'], 3)}")
    print("\nper cell (seed means):  edit | DPO as published | DPO best in budget | "
          "DPO full adapter, any budget")
    for t, a in CELLS:
        pc = out["per_cell"][f"{t}|{a}"]
        print(f"  {t + '|' + a:28s} {pc['edit']:+.3f} | {pc['dpo_as_published']:+.3f} | "
              f"{pc['dpo_best_in_budget']:+.3f} | {pc['dpo_full_adapter_ignoring_budget']:+.3f}"
              + ("   (skipped: <20 differing pairs)" if pc["skipped"] else ""))
    print(f"\nwrote {os.path.relpath(os.path.join(OUT, 'e7a_dpo.json'), REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
