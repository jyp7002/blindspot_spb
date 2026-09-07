"""v10 — emit the remaining starred values as resolvable JSON keys.

Nine provenance claims were checked against the artifacts the author named.
Six produced values that reproduce; three did not. The six all failed to reach
the manifest for the same mechanical reason: they live in raw `.jsonl` panels
or in derived quantities, and the manifest resolves dotted JSON keys, not row
scans. This module computes them and writes them where an entry can point.

Gate is recorded per value, never assumed:
  INLP    v9 (results_v9/v6trace/inlp is re-scored; not in NOT_RESCORABLE)
  INS-A   v9 (results_v9/v8ins/A)
  INS-B   v9 (results_v9/v8ins/BP; the base rows are bit-identical to as-run)
  STE     AS-RUN, permanently — no alpha_trace, no re-scorable twin
  PROMPT  AS-RUN tree, but gate-invariant: the 16/18 count is identical under
          the old float gate and the v9 integer gate, so no cell sits in the
          boundary-defect region
  DPO     NOT-RESCORABLE (t2x corpora)

Nothing here is hardcoded: every value is recomputed from the panels, and the
printed manuscript values are carried alongside only as `*_printed` fields so
the audit can diff them.

Run: python3 src/v10_regen_starred.py
"""
import os, sys, glob, json, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
import v9_gate

IFEVAL_METRIC = "inst_level_strict_acc,none"


def _fmt(d, p=4):
    return None if d is None else dict(
        d, text=C.fmt(d, p), status=C.status(d))


# ------------------------------------------------------------------ INLP ----

def inlp():
    """real vs its random-subspace null, paired on the target x axis cell."""
    out = {}
    for root, lab in ((C.V9, "v9"), (C.RES, "as_run")):
        rs = C.rows(root, "v6trace/inlp")
        if not rs:
            continue
        best = collections.defaultdict(list)
        for r in rs:
            if r.get("collateral_ok"):
                best[(r["target"], r["axis"], r["seed"], r["kind"])].append(
                    r.get("bias_reduction"))
        val = {k: (max(v) if v else 0.0) for k, v in best.items()}
        cells = collections.defaultdict(dict)
        for (t, a, s, kind) in {(k[0], k[1], k[2], k[3])
                                for k in set(val) | {
                                    (r["target"], r["axis"], r["seed"], r["kind"])
                                    for r in rs}}:
            cells[(t, a, kind)].setdefault("v", []).append(val.get((t, a, s, kind), 0.0))
        cm = {k: float(np.mean(v["v"])) for k, v in cells.items()}
        pairs, per_cell = [], {}
        for (t, a, kind) in sorted(cm):
            if kind != "real":
                continue
            nk = (t, a, "null_random")
            if nk not in cm:
                continue
            pairs.append((cm[(t, a, kind)], cm[nk]))
            per_cell[f"{t}|{a}"] = {"real": cm[(t, a, kind)], "null_random": cm[nk],
                                    "diff": cm[(t, a, kind)] - cm[nk]}
        ok = [r for r in rs if r.get("collateral_ok")]
        real_ok = [r for r in ok if r.get("kind") == "real"]
        out[lab] = {
            "delta_cell": _fmt(C.boot_paired(pairs)),
            "per_cell": per_cell,
            "n_rows": len(rs), "n_in_budget": len(ok),
            "n_in_budget_real": len(real_ok),
            "n_real_rows": sum(1 for r in rs if r.get("kind") == "real"),
            "mean_ppl_ratio_in_budget_real": (
                float(np.mean([r["ppl_ratio"] for r in real_ok
                               if r.get("ppl_ratio") is not None])) if real_ok else None),
            "median_ppl_ratio_all_rows": float(np.median(
                [r["ppl_ratio"] for r in rs if r.get("ppl_ratio") is not None])),
        }
    out["_gate"] = "v9 (v6trace/inlp is re-scored and is NOT in NOT_RESCORABLE)"
    out["_printed"] = "+0.025* [-0.070, +0.150]"
    return out


# ------------------------------------------------------------------- STE ----

def ste():
    """STE-trained signs vs sign(dW_fp). Permanently as-run."""
    rs = C.rows(C.RES, "v6trace/ste")
    g = collections.defaultdict(dict)
    for r in rs:
        g[(r["target"], r["axis"], r["seed"])][r["variant"]] = r.get("removal")
    sweep, collapse = [], []
    for k, d in sorted(g.items()):
        if "ste" not in d or "sign_fp" not in d:
            continue
        a, b = d["ste"], d["sign_fp"]
        if a is None or (isinstance(a, float) and a != a):
            collapse.append(f"{k[0]}|{k[1]}|s{k[2]}")
        sweep.append((C.z(a), C.z(b)))
    cellg = collections.defaultdict(list)
    for k, d in g.items():
        if "ste" in d and "sign_fp" in d:
            cellg[(k[0], k[1])].append((C.z(d["ste"]), C.z(d["sign_fp"])))
    cell = [(float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
            for v in cellg.values()]
    dropped = [(C.z(d["ste"]), C.z(d["sign_fp"])) for k, d in sorted(g.items())
               if "ste" in d and "sign_fp" in d
               and not (d["ste"] is None or (isinstance(d["ste"], float) and d["ste"] != d["ste"]))]
    return {
        "_gate": ("AS-RUN, permanently: results/v6trace/ste has no alpha_trace and "
                  "no re-scorable twin in results_v9/"),
        "_printed": "-0.007* [-0.023, +0.001], n = 12",
        "delta_sweep": _fmt(C.boot_paired(sweep)),
        "delta_cell": _fmt(C.boot_paired(cell)),
        "delta_sweep_collapse_dropped": _fmt(C.boot_paired(dropped)),
        "n_pairs_sweep": len(sweep), "n_pairs_cell": len(cell),
        "collapse_cells": collapse,
        "_sensitivity_note": ("dropping the collapse cell instead of applying the "
                              "nan->0 rule flips the sign of the point estimate"),
    }


# ----------------------------------------------------------------- INS-A ----

def ins_a():
    """IFEval instruction-following cost, sign_only vs random_sign."""
    rs = C.rows(C.V9, "v8ins/A") or C.rows(C.RES, "v8ins/A")
    by = {(r["target"], r["condition"]): r for r in rs}
    out, targets = {}, sorted({r["target"] for r in rs})
    for t in targets:
        s, n = by.get((t, "sign_only")), by.get((t, "random_sign"))
        if not s or not n:
            continue
        si = (s.get("ifeval") or {}).get(IFEVAL_METRIC)
        ni = (n.get("ifeval") or {}).get(IFEVAL_METRIC)
        out[t] = {
            "sign_only_ifeval": si, "random_sign_ifeval": ni,
            "sign_minus_random": (si - ni) if (si is not None and ni is not None) else None,
            "sign_only_removal": s.get("removal"),
            "random_sign_removal": n.get("removal"),
            "ifeval_sample_len": (s.get("ifeval") or {}).get("sample_len"),
            "ifeval_limit": s.get("ifeval_limit"),
        }
    return {
        "_gate": "v9 (results_v9/v8ins/A)",
        "_metric": IFEVAL_METRIC,
        "_metric_note": ("the draft sentence does not name the metric; only this "
                         "one of the four IFEval fields yields both printed values"),
        "_printed": "qwen +0.000* at +0.698* removal; llama -0.022*",
        "_denominator_note": ("the artifacts carry sample_len 200 / ifeval_limit 200; "
                              "the '541' in the draft is a code constant "
                              "(run_ins.py), not an artifact value"),
        "targets": out,
    }


# ----------------------------------------------------------------- INS-B ----

def ins_b():
    """Base-model editability, and the prompt-format MMLU shift."""
    rs = C.rows(C.V9, "v8ins/BP") or C.rows(C.RES, "v8ins/BP")
    g = collections.defaultdict(list)
    for r in rs:
        g[(r["target"], r["axis"], r["condition"])].append(C.z(r.get("removal")))
    means = {f"{k[0]}|{k[1]}|{k[2]}": float(np.mean(v)) for k, v in g.items()}
    ob = {a: C.jload(f"results/v8ins/{a}/onboard.json", {}) for a in ("B", "BP")}
    cell = "gemma2b_it|occ_gender"
    shift = None
    if cell in ob["B"] and cell in ob["BP"]:
        shift = ob["BP"][cell]["mmlu"] - ob["B"][cell]["mmlu"]
    base_key = "gemma2b_base|occ_gender"
    return {
        "_gate": ("v9 (results_v9/v8ins/BP); the gemma2b_base rows are "
                  "bit-identical to as-run, v9_changed=false on every one"),
        "_printed": "+0.605* / null +0.000* / format -0.175*",
        "base_sign_only": means.get("gemma2b_base|occ_gender|sign_only"),
        "base_random_sign": means.get("gemma2b_base|occ_gender|random_sign"),
        "base_matched_norm_contrast": (
            means.get("gemma2b_base|occ_gender|sign_only", 0.0)
            - means.get("gemma2b_base|occ_gender|random_sign", 0.0)),
        "prompt_format_mmlu_shift": shift,
        "onboard_mmlu": {"B_chat_template": ob["B"].get(cell, {}).get("mmlu"),
                         "BP_plain_text": ob["BP"].get(cell, {}).get("mmlu")},
        "base_mmlu_plain_text": ob["BP"].get(base_key, {}).get("mmlu"),
        "_below_chance_note": ("the base MMLU 0.19 < 0.25 four-way chance is "
                               "measured in PLAIN TEXT (arm BP), not chat format; "
                               "gemma-2-2b ships no chat template, so arm B "
                               "produced no base rows at all"),
        "all_means": means,
    }


# ---------------------------------------------------------- prompt baseline --

def prompt_baseline():
    d = C.jload("results/v6trace/prompt_baseline.json", {})
    recs = {k: v for k, v in d.items()}
    base = {k: v for k, v in recs.items() if k.endswith("|none")}
    conds, fails, passes, ratios = [], 0, [], {}
    for k, v in sorted(recs.items()):
        tgt, ax, pr = k.split("|")
        if pr == "none":
            continue
        b = base.get(f"{tgt}|{ax}|none")
        if not b:
            continue
        ok = (v9_gate.mmlu_ok(b["mmlu_acc"], v["mmlu_acc"])
              and v9_gate.ppl_ok(b["ppl"], v["ppl"]))
        ratio = v["ppl"] / b["ppl"]
        removal = abs(b["skew"]) - abs(v["skew"])
        ratios[k] = ratio
        conds.append({"cell": k, "in_budget": bool(ok), "ppl_ratio": ratio,
                      "removal_abs_skew": removal})
        if ok:
            passes.append(k)
        else:
            fails += 1
    best = max((c for c in conds if c["cell"].endswith("|strong")),
               key=lambda c: c["removal_abs_skew"], default=None)
    mild = {}
    for tgt in sorted({k.split("|")[0] for k in recs}):
        b = recs.get(f"{tgt}|occ_gender|none"); m = recs.get(f"{tgt}|occ_gender|mild")
        if b and m:
            mild[tgt] = {"none_abs_skew": abs(b["skew"]), "mild_abs_skew": abs(m["skew"]),
                         "mild_increases_skew": abs(m["skew"]) > abs(b["skew"])}
    return {
        "_gate": ("as-run tree, but GATE-INVARIANT: the pass/fail count is identical "
                  "under the old float gate and the v9 integer gate, so no condition "
                  "sits in the boundary-defect region"),
        "_printed": "16 of 18 conditions*; ppl ratio 1.28*; three of four targets*",
        "n_conditions": len(conds), "n_fail_budget": fails,
        "n_pass_budget": len(passes), "cells_in_budget": passes,
        "max_removal_strong_cell": best,
        "ppl_ratio_max_removal_strong": (best or {}).get("ppl_ratio"),
        "mild_prompt_occ_gender": mild,
        "n_targets_mild_increases_skew": sum(
            1 for v in mild.values() if v["mild_increases_skew"]),
        "n_targets": len(mild),
        "_target_count_defect": (
            "the draft says the mild prompt raises skew on 'three of four "
            "targets', but prompt_baseline.json contains only THREE targets "
            "(qwen, phi, gemma). The denominator does not exist."),
        "conditions": conds,
    }


# -------------------------------------------------------------- DPO pairs ----

def dpo_pairs():
    """Identical-string fraction of the elicited pairs, and its correlation
    with contrast_gap. Source is the t2x corpora, which are not re-scorable."""
    rows = []
    for fp in sorted(glob.glob(os.path.join(C.RES, "t2x", "corpora", "*|*|s*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:                                   # noqa: BLE001
            continue
        b, dd = d.get("biased") or [], d.get("debiased") or []
        n = min(len(b), len(dd))
        if not n:
            continue
        diff = sum(1 for i in range(n) if b[i] != dd[i])
        rows.append({"designer": d.get("designer"), "axis": d.get("axis"),
                     "seed": d.get("seed"), "n": n,
                     "frac_differing": diff / n, "frac_identical": 1 - diff / n,
                     "contrast_gap": (d.get("diag") or {}).get("contrast_gap")})
    AX = {"occ_gender", "crows_socioeconomic"}
    # The printed 46-92% is population-dependent, exactly as contrast_gap's
    # +0.507 is. Both the broad population and the restricted one the range
    # actually reproduces on are emitted, labelled.
    LARGE = {"qwen", "llama", "gemma", "olmo", "granite"}
    def stats(sub):
        if len(sub) < 3:
            return None
        fd = [r["frac_differing"] for r in sub]
        gp = [r["contrast_gap"] for r in sub]
        return {"n_cells": len(sub),
                "frac_identical_min": min(r["frac_identical"] for r in sub),
                "frac_identical_max": max(r["frac_identical"] for r in sub),
                "frac_differing_min": min(fd), "frac_differing_max": max(fd),
                "pearson_differing_vs_contrast_gap": float(np.corrcoef(fd, gp)[0, 1]),
                "designers": sorted({r["designer"] for r in sub})}
    two_ax = [r for r in rows if r["axis"] in AX and r["contrast_gap"] is not None]
    restricted = [r for r in two_ax
                  if r["designer"] in LARGE or str(r["designer"]).endswith("_3b")]
    pops = {"all_designers_two_axes": stats(two_ax),
            "large_plus_3b_siblings": stats(restricted)}
    sub = two_ax
    fd = [r["frac_differing"] for r in sub]
    gp = [r["contrast_gap"] for r in sub]
    pear = float(np.corrcoef(fd, gp)[0, 1]) if len(sub) > 2 else None
    return {
        "populations": pops,
        "_population_note": ("the printed 46-92% does not reproduce on every "
                             "population; it is a property of a designer subset, "
                             "and the manuscript names none"),
        "_gate": "NOT-RESCORABLE (t2x corpora)",
        "_printed": "identical on 46-92%* of pairs; differing fraction correlates +0.79*",
        "_direction": ("identical is the HIGH number and differing the LOW one; the "
                       "max differing fraction over every cached corpus is well "
                       "under 92%, so a '92% differing' reading is impossible"),
        "n_corpora_all": len(rows), "n_corpora_two_axes": len(sub),
        "frac_identical_min": min((r["frac_identical"] for r in sub), default=None),
        "frac_identical_max": max((r["frac_identical"] for r in sub), default=None),
        "frac_differing_min": min(fd, default=None),
        "frac_differing_max": max(fd, default=None),
        "pearson_differing_vs_contrast_gap": pear,
    }


def env_sensitivity():
    """The registered n=87 sensitivity: what each gate buys over editing all.

    The draft prints the DIFFERENCE ("buys +0.0009 utility"), which no artifact
    stores -- results/v8/env_sensitivity.json holds the policy utilities and the
    edit-all baseline separately. The subtraction is done here so the printed
    number is computed rather than eyeballed."""
    d = C.jload("results/v8/env_sensitivity.json", {})
    tbl = {r["policy"]: r for r in (d.get("held_out_table") or [])}
    base = tbl.get("edit-all", {}).get("utility")
    out = {}
    for pol, rec in tbl.items():
        if pol == "edit-all" or base is None:
            continue
        out[pol] = {"utility": rec["utility"], "edit_all_utility": base,
                    "gain_over_edit_all": rec["utility"] - base,
                    "n_cells": rec.get("n_cells"),
                    "backfire_rate_edited": rec.get("backfire_rate_edited")}
    return {
        "_gate": ("AS-RUN (results/v8/env_sensitivity.json). The draft calls this "
                  "'the registered n = 87 sensitivity', i.e. it cites the "
                  "registered as-run analysis by name, so an as-run source is the "
                  "correct referent here."),
        "_printed": "the pre-skew gate buys +0.0009 utility; the contrast-gap gate costs utility",
        "unit": d.get("unit"), "n_cells": d.get("n_held_out"),
        "edit_all_utility": base, "policies": out,
    }


def main():
    C.ensure_dirs()
    rep = {"artifact": "v10_regen_starred", "module": "src/v10_regen_starred.py",
           "purpose": ("emit the remaining starred values as resolvable JSON keys; "
                       "they were derivable all along but lived in raw panels"),
           "inlp": inlp(), "ste": ste(), "ins_a": ins_a(), "ins_b": ins_b(),
           "prompt_baseline": prompt_baseline(), "dpo_pairs": dpo_pairs(),
           "env_sensitivity": env_sensitivity()}
    fp = C.jdump(rep, "results/v10/starred_sources.json")

    i = rep["inlp"].get("v9", {})
    print("[v10 starred]")
    print(f"  INLP   v9 cell  {i.get('delta_cell', {}).get('text')}  "
          f"(in-budget {i.get('n_in_budget')}/{i.get('n_rows')}, "
          f"real arm {i.get('n_in_budget_real')}/{i.get('n_real_rows')})")
    s = rep["ste"]
    print(f"  STE    sweep    {s['delta_sweep']['text']}   cell {s['delta_cell']['text']}"
          f"   collapse cells {s['collapse_cells']}")
    for t, v in rep["ins_a"]["targets"].items():
        print(f"  INS-A  {t:12s} sign-random {v['sign_minus_random']:+.4f}   "
              f"removal {v['sign_only_removal']:+.4f}")
    b = rep["ins_b"]
    print(f"  INS-B  base sign_only {b['base_sign_only']:+.4f}  null "
          f"{b['base_random_sign']:+.4f}  format shift {b['prompt_format_mmlu_shift']:+.3f}")
    p = rep["prompt_baseline"]
    print(f"  PROMPT {p['n_fail_budget']} of {p['n_conditions']} fail; "
          f"ppl@max-removal-strong {p['ppl_ratio_max_removal_strong']:.4f}; "
          f"mild raises skew on {p['n_targets_mild_increases_skew']}/{p['n_targets']}")
    dp = rep["dpo_pairs"]
    print(f"  DPO    identical {100*dp['frac_identical_min']:.1f}-"
          f"{100*dp['frac_identical_max']:.1f}%  r(differing,gap)="
          f"{dp['pearson_differing_vs_contrast_gap']:+.4f}  n={dp['n_corpora_two_axes']}")
    es = rep["env_sensitivity"]
    for pol, v in (es.get("policies") or {}).items():
        print(f"  ENV    {pol:18s} utility {v['utility']:+.4f}  "
              f"gain over edit-all {v['gain_over_edit_all']:+.4f}")
    print(f"wrote {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
