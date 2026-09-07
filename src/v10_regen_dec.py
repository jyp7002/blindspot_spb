"""v10 / WS-B B2 — the Fig 2 source: the seven-condition decomposition, v9-gated.

WHY THIS FILE EXISTS. `experiments_v10.md` B2 requires Fig 2 to be built from
"the v9-re-scored DEC analysis (`results_v9/…/dec`, re-emitted as
`dec_analysis_v9.json` if absent — never the as-run file)". It IS absent: the
only DEC analysis object on disk is `results/v8/dec_analysis.json`, which was
computed from `results/v8dec/` (AS-RUN, float collateral gate) by
`src/v8_dec_analyze.py`. This module is the v9 replacement. It reads
`results_v9/v8dec/removal.jsonl` and nothing else for its numbers.

WHAT IS REPRODUCED AND WHAT IS NEW.
  * Reproduced (must match `results/v9/v9_diff_report.json` and
    `RESULTS_METHOD_v9.md` §1/§2 to the printed precision): the seven
    per-condition means, the five registered paired deltas plus
    "C-a minus C-rand", and the unanimity count. Every one of them is computed
    through `v10_common` (whose `dec_cells` / `boot_paired` are the verbatim
    v9_regen.dec conventions), and then CHECKED against the published artifacts
    parsed at run time — no published value is typed here.
  * NEW IN v10, NOT part of the registered v9 set: **per-condition bootstrap
    CIs**. v9 bootstrapped only PAIRED differences; a CI on a single condition
    mean exists nowhere in the v9 artifacts. They are marked `registered: false`
    in the artifact and must be described in the Fig 2 caption as a v10
    addition, not as a re-scored v9 quantity.

CONVENTIONS (all inherited via v10_common; see its docstring):
  budget-fail -> 0.0; the resampling unit is the (target, axis) CELL; a cell
  value is the MEAN over its rows; 10k resamples, seed 0; percentile CIs.

A CONDITION THAT WAS NEVER RUN IN A CELL IS `null`, NOT 0.0. The nan->0 rule
covers a run that failed the collateral budget. It does not cover a cell where
the condition is structurally undefined (phi's fused qkv_proj: no C-tensorshuf,
no LOPO) — zero-filling those would fabricate seven-cell evidence into a
ten-cell table. That is exactly why Delta_tensor is n=7.

Run:  python3 src/v10_regen_dec.py
Emits: results/v10/dec_analysis_v9.json   (the ONLY output of this module)
"""
import os, sys, re, json, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
from v8_env import axis_family            # the project's benchmark-family rule

PANEL = "v8dec"
OUT_NAME = "dec_analysis_v9.json"

# Same names, same (a, b) order, same sequence as v9_regen.dec. Do not reorder:
# the bootstrap is seeded per call, but the pair list order is part of the draw.
ESTIMANDS = (("Delta_selection", "C-ref", "C-a"),
             ("Delta_sign",      "C-ref", "C-b"),
             ("Delta_null",      "C-ref", "C-rand"),
             ("Delta_location",  "C-ref", "C-layershuf"),
             ("Delta_tensor",    "C-ref", "C-tensorshuf"),
             ("C-a minus C-rand", "C-a",  "C-rand"))

UNANIMITY_PAIR = ("C-ref", "C-a")        # the registered unanimity comparison


# ----------------------------------------------------------- published set ---
# Everything below PARSES a published artifact at run time. Nothing is typed.

def parse_results_method_v9(path):
    """Pull the citable DEC numbers out of RESULTS_METHOD_v9.md §1 and §2."""
    out = {"deltas": {}, "means": {}, "unanimity": None, "qualifier": None}
    if not os.path.exists(path):
        return out
    txt = open(path).read()
    val = r"([+-]\d+\.\d+)"
    row1 = re.compile(r"^\|\s*DEC\s*::\s*(?P<name>[^|]+?)\s*\|[^|]*\|\s*\*\*"
                      r"(?P<v9>[^*]+?)\*\*\s*\|", re.M)
    dec1 = re.compile(val + r"\s*\[\s*" + val + r"\s*,\s*" + val +
                      r"\s*\]\s*\(n=(\d+),\s*([a-z ]*0)\)")
    for m in row1.finditer(txt):
        d = dec1.search(m.group("v9"))
        if d:
            out["deltas"][m.group("name")] = dict(
                point=float(d.group(1)), lo=float(d.group(2)), hi=float(d.group(3)),
                n=int(d.group(4)), status=d.group(5))
    for m in re.finditer(r"^\|\s*(C-[A-Za-z]+)\s*\|\s*" + val + r"\s*\|\s*\*\*"
                         + val + r"\*\*\s*\|", txt, re.M):
        out["means"][m.group(1)] = float(m.group(3))
    u = re.search(r"Unanimity \(C-ref > C-a\):.*?\*\*(\d+)/(\d+)\*\*", txt)
    if u:
        out["unanimity"] = dict(count=int(u.group(1)), denominator=int(u.group(2)))
    q = re.search(r"^(Magnitude remains .*?)$", txt, re.M)
    if q:
        out["qualifier"] = q.group(1).strip()
    return out


def parse_diff_report(obj):
    """DEC entries of results/v9/v9_diff_report.json -> {name: {...}}."""
    out = {"deltas": {}, "means": {}, "unanimity": None}
    if not obj:
        return out
    val = r"([+-]\d+\.\d+)"
    dec1 = re.compile(val + r"\s*\[\s*" + val + r"\s*,\s*" + val +
                      r"\s*\]\s*\(n=(\d+),\s*([a-z ]*0)\)")
    for k, v in (obj.get("diff") or {}).items():
        if not k.startswith("DEC :: "):
            continue
        name = k.split("DEC :: ", 1)[1]
        d = dec1.search(v.get("v9", ""))
        rec = dict(point_full=v.get("v9_point"))
        if d:
            rec.update(point=float(d.group(1)), lo=float(d.group(2)),
                       hi=float(d.group(3)), n=int(d.group(4)), status=d.group(5))
        out["deltas"][name] = rec
    out["means"] = dict(obj.get("DEC_means_v9") or {})
    un = obj.get("DEC_unanimity") or {}
    if "v9" in un:
        out["unanimity"] = un["v9"]
    return out


def close(a, b, places):
    """Agreement at the precision the source actually prints."""
    if a is None or b is None:
        return False
    return round(float(a), places) == round(float(b), places)


# ------------------------------------------------------------- draft check ---

NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10}   # parse table, not data


def draft_lines(path):
    return open(path).read().splitlines() if os.path.exists(path) else []


def check_draft(lines, cond_means, family_count, alpha_counts, grid_max):
    """Compare the manuscript's DEC claims to what this module computed.

    Every 'draft_says' below is READ OUT OF THE DRAFT at run time; the
    'computed' side is the artifact value. Nothing is typed on either side.
    """
    conflicts = []
    num = re.compile(r"[+\-−]?\d+\.\d+")
    for i, ln in enumerate(lines, 1):
        if not ln.lstrip().startswith("|"):
            continue
        m = re.search(r"\|\s*\*{0,2}(C-[A-Za-z]+)\*{0,2}\s*\|", ln)
        if not m or m.group(1) not in cond_means:
            continue
        nums = num.findall(ln.replace("−", "-"))
        if not nums:
            continue
        claimed = float(nums[-1])
        computed = cond_means[m.group(1)]
        places = len(nums[-1].split(".")[1])
        if not close(claimed, computed, places):
            conflicts.append(dict(
                draft_line=i, kind="per_condition_mean", condition=m.group(1),
                draft_says=f"{m.group(1)} = {nums[-1]}",
                computed=round(computed, 6),
                computed_at_draft_precision=f"{computed:+.{places}f}"))
    for i, ln in enumerate(lines, 1):
        f = re.search(r"(?:across|spanning)\s+([A-Za-z]+)\s+benchmark famil", ln)
        if f:
            claimed = NUMWORDS.get(f.group(1).lower())
            if claimed is None and f.group(1).isdigit():
                claimed = int(f.group(1))
            if claimed is not None and claimed != family_count:
                conflicts.append(dict(
                    draft_line=i, kind="benchmark_family_count",
                    draft_says=f"{f.group(0)}", computed=family_count))
        a = re.search(r"argmax at the largest\s+\S+\s+in\s+(\d+)\s*/\s*(\d+)\s*runs", ln)
        if a:
            got = alpha_counts.get("C-a", {})
            if (int(a.group(1)), int(a.group(2))) != (got.get("at_grid_max"), got.get("n_runs")):
                conflicts.append(dict(
                    draft_line=i, kind="C-a_argmax_at_grid_max",
                    draft_says=a.group(0),
                    computed=f"{got.get('at_grid_max')}/{got.get('n_runs')} "
                             f"runs at alpha={grid_max}"))
    return conflicts


# -------------------------------------------------------------------- main ---

def main():
    C.ensure_dirs()
    raw = C.rows(C.V9, PANEL)
    if not raw:
        raise SystemExit(f"no rows at {os.path.join(C.V9, PANEL, 'removal.jsonl')}")

    # Cell means: the ONE definition, from v10_common (verbatim v9_regen.dec).
    cells = C.dec_cells(C.V9)
    keys_v9order = list(cells)                       # insertion order == v9_regen's
    keys_sorted = sorted(cells)
    name = lambda k: f"{k[0]}|{k[1]}"

    # Row bookkeeping (counts only — cell VALUES come from dec_cells, never here).
    seen = {}
    for r in raw:
        seen[(r["target"], r["axis"], r["seed"], r["designer"], r["condition"])] = r
    dedup = list(seen.values())
    n_rows_by_cell_cond = collections.Counter(
        ((r["target"], r["axis"]), r["condition"]) for r in dedup)
    budget_fail = sum(1 for r in dedup
                      if r.get("removal") is None
                      or (isinstance(r.get("removal"), float)
                          and r["removal"] != r["removal"]))
    rescore_changed = sum(1 for r in dedup if r.get("v9_changed"))

    # ---- per-condition means (reproduce v9) + bootstrap CIs (NEW in v10) ----
    per_condition = {}
    for c in C.DEC_CONDITIONS:
        vals_sorted = [cells[k][c] for k in keys_sorted if c in cells[k]]
        if not vals_sorted:
            continue
        b = C.boot_mean(vals_sorted)
        cell_vals = {name(k): cells[k][c] for k in keys_sorted if c in cells[k]}
        top = max(cell_vals, key=cell_vals.get)
        rest = [v for k, v in cell_vals.items() if k != top]
        per_condition[c] = dict(
            mean=float(np.mean(vals_sorted)),
            n_cells=len(vals_sorted),
            perturbs=C.DEC_PERTURBS[c],
            ci=dict(lo=b["lo"], hi=b["hi"], point=b["point"], n=b["n"],
                    status=C.status(b), method="percentile bootstrap of the mean, "
                    "cell = resampling unit, 10k, seed 0",
                    registered=False,
                    note="NEW IN v10 — no per-condition CI exists in any v9 "
                         "artifact; v9 bootstrapped paired differences only."),
            formatted=C.fmt(b),
            cell_min=min(cell_vals.values()), cell_max=max(cell_vals.values()),
            max_cell=top,
            mean_excluding_max_cell=(float(np.mean(rest)) if rest else None),
            share_of_C_ref=None)
    ref_mean = per_condition["C-ref"]["mean"]
    for c, d in per_condition.items():
        d["share_of_C_ref"] = d["mean"] / ref_mean if ref_mean else None

    order = [c for c, _ in sorted(per_condition.items(),
                                  key=lambda kv: -kv[1]["mean"])]

    # ------------------------ registered paired estimands (reproduce v9) ----
    estimands = {}
    for nm, a, b in ESTIMANDS:
        pairs = [(cells[k][a], cells[k][b]) for k in keys_v9order
                 if a in cells[k] and b in cells[k]]
        d = C.boot_paired(pairs)
        used = [name(k) for k in keys_v9order if a in cells[k] and b in cells[k]]
        estimands[nm] = dict(minuend=a, subtrahend=b, point=d["point"], lo=d["lo"],
                             hi=d["hi"], n=d["n"], status=C.status(d),
                             formatted=C.fmt(d), cells_used=used,
                             cells_absent=[name(k) for k in keys_sorted
                                           if name(k) not in used],
                             registered=True)

    # ------------------------------------------------------- unanimity ------
    ua, ub = UNANIMITY_PAIR
    both = [k for k in keys_sorted if ua in cells[k] and ub in cells[k]]
    wins = [name(k) for k in both if cells[k][ua] > cells[k][ub]]
    unanimity = dict(comparison=f"{ua} > {ub}", count=len(wins),
                     denominator=len(both), cells_supporting=wins,
                     cells_not_supporting=[name(k) for k in both
                                           if name(k) not in wins],
                     annotation=f"{len(wins)}/{len(both)} cells")

    # ------------------------------------------- the full per-cell table ----
    per_cell, table = {}, []
    for k in keys_sorted:
        m = cells[k]
        row = {c: (m[c] if c in m else None) for c in order}
        d_sel = (m[ua] - m[ub]) if (ua in m and ub in m) else None
        per_cell[name(k)] = dict(
            target=k[0], axis=k[1], axis_family=axis_family(k[1]),
            values=row,
            n_rows={c: n_rows_by_cell_cond[(k, c)] for c in order
                    if n_rows_by_cell_cond[(k, c)]},
            delta_selection=d_sel,
            delta_selection_pct_of_C_ref=(100.0 * d_sel / m[ua]
                                          if d_sel is not None and m.get(ua) else None),
            conditions_absent=[c for c in order if c not in m])
        table.append([name(k), k[0], k[1], axis_family(k[1])] +
                     [row[c] for c in order])

    # -------------------------------------------------------- coverage -----
    targets = sorted({k[0] for k in keys_sorted})
    axes = sorted({k[1] for k in keys_sorted})
    fam = collections.Counter(axis_family(a) for a in axes)
    fam_cells = collections.Counter(axis_family(k[1]) for k in keys_sorted)
    coverage = dict(
        n_cells=len(cells), cells=[name(k) for k in keys_sorted],
        targets=targets, n_targets=len(targets),
        axes=axes, n_axes=len(axes),
        designers=sorted({r["designer"] for r in dedup}),
        seeds=sorted({r["seed"] for r in dedup}),
        roles=sorted({str(r.get("role")) for r in dedup}),
        density=sorted({r.get("density") for r in dedup}),
        scale_mode=sorted({str(r.get("scale_mode")) for r in dedup}),
        benchmark_families=dict(sorted(fam.items())),
        n_benchmark_families=len(fam),
        cells_per_benchmark_family=dict(sorted(fam_cells.items())),
        family_rule="src/v8_env.axis_family (bbq_->BBQ, crows_->CrowS, "
                    "ss_->StereoSet, else templated)")

    # ------------------------------ alpha grid (the grid-conditional claim) --
    alphas = sorted({r["v9_selected_alpha"] for r in dedup
                     if r.get("v9_selected_alpha") is not None})
    grid_max = max(alphas) if alphas else None
    alpha_counts = {}
    for c in C.DEC_CONDITIONS:
        rs = [r for r in dedup if r["condition"] == c
              and r.get("v9_selected_alpha") is not None]
        if rs:
            alpha_counts[c] = dict(
                n_runs=len(rs),
                at_grid_max=sum(1 for r in rs if r["v9_selected_alpha"] == grid_max),
                by_alpha={str(a): sum(1 for r in rs if r["v9_selected_alpha"] == a)
                          for a in alphas})

    # --------------------------------------------- why Delta_tensor is n=7 --
    absent = collections.defaultdict(list)
    for k in keys_sorted:
        for c in C.DEC_CONDITIONS:
            if c not in cells[k]:
                absent[c].append(name(k))
    lopo_conds = sorted({r["condition"] for r in dedup
                         if r["condition"].startswith("C-ref-no_")})
    cells_with_lopo = {(r["target"], r["axis"]) for r in dedup
                       if r["condition"].startswith("C-ref-no_")}
    tensor_note = dict(
        n=estimands["Delta_tensor"]["n"],
        n_cells_in_panel=len(cells),
        cells_without_C_tensorshuf=absent.get("C-tensorshuf", []),
        targets_without_C_tensorshuf=sorted({c.split("|")[0]
                                             for c in absent.get("C-tensorshuf", [])}),
        lopo_conditions_in_panel=lopo_conds,
        cells_without_any_LOPO=[name(k) for k in keys_sorted
                                if k not in cells_with_lopo],
        reason="The three phi cells carry no C-tensorshuf rows and no LOPO rows. "
               "Phi-3.5-mini fuses q/k/v into a single qkv_proj, so ATTN resolves "
               "to ['qkv_proj'] alone: there is no shape-compatible sibling "
               "projection to permute the sign pattern with, and no separate "
               "q/k/v/o module to leave out (run_dec.py, the structural "
               "exclusion). Those cells are ABSENT, not zero — zero-filling them "
               "would put fabricated evidence into a paired contrast.",
        evidence="results_v9/v8dec/removal.jsonl contains no condition="
                 "'C-tensorshuf' row for target='phi'")

    # -------------------------------------------------- verification --------
    pub_md = parse_results_method_v9(os.path.join(C.HERE, "RESULTS_METHOD_v9.md"))
    pub_dr = parse_diff_report(C.jload(os.path.join("results", "v9",
                                                    "v9_diff_report.json")))
    checks = []
    for c, d in per_condition.items():
        for src, book in (("RESULTS_METHOD_v9.md §2", pub_md["means"]),
                          ("results/v9/v9_diff_report.json DEC_means_v9", pub_dr["means"])):
            if c in book:
                checks.append(dict(kind="per_condition_mean", estimand=c, source=src,
                                   published=book[c], computed=round(d["mean"], 6),
                                   places=4, matches=close(book[c], d["mean"], 4)))
    for nm, d in estimands.items():
        for src, book in (("RESULTS_METHOD_v9.md §1", pub_md["deltas"]),
                          ("results/v9/v9_diff_report.json diff[DEC :: *].v9", pub_dr["deltas"])):
            p = book.get(nm)
            if not p:
                continue
            ok = all(close(p.get(f), d[f], 4) for f in ("point", "lo", "hi")) \
                 and p.get("n") == d["n"] and p.get("status", d["status"]) == d["status"]
            checks.append(dict(kind="paired_delta", estimand=nm, source=src,
                               published=f"{p['point']:+.4f} [{p['lo']:+.4f}, "
                                         f"{p['hi']:+.4f}] (n={p['n']}, {p['status']})",
                               computed=C.fmt(d and dict(point=d["point"], lo=d["lo"],
                                                         hi=d["hi"], n=d["n"])),
                               places=4, matches=bool(ok)))
    if pub_md["unanimity"]:
        checks.append(dict(kind="unanimity", estimand="C-ref > C-a",
                           source="RESULTS_METHOD_v9.md §2",
                           published=f"{pub_md['unanimity']['count']}/"
                                     f"{pub_md['unanimity']['denominator']}",
                           computed=f"{unanimity['count']}/{unanimity['denominator']}",
                           matches=(pub_md["unanimity"]["count"] == unanimity["count"]
                                    and pub_md["unanimity"]["denominator"]
                                    == unanimity["denominator"])))
    if pub_dr["unanimity"] is not None:
        checks.append(dict(kind="unanimity", estimand="C-ref > C-a",
                           source="results/v9/v9_diff_report.json DEC_unanimity.v9",
                           published=pub_dr["unanimity"], computed=unanimity["count"],
                           matches=(pub_dr["unanimity"] == unanimity["count"])))

    conflicts = check_draft(draft_lines(os.path.join(C.HERE,
                                                     "binary_debiaser_draft_v3.md")),
                            {c: d["mean"] for c, d in per_condition.items()},
                            coverage["n_benchmark_families"], alpha_counts, grid_max)

    art = {
        "artifact": "v10_dec_analysis_v9",
        "module": "src/v10_regen_dec.py",
        "figure": "fig2 (headline) — seven-condition decomposition",
        "gate": "v9 integer-item collateral gate (<=4 of 200 MMLU items)",
        "source": {
            "panel": PANEL,
            "path": os.path.relpath(os.path.join(C.V9, PANEL, "removal.jsonl"), C.HERE),
            "tree": "results_v9 (RE-SCORED, citable)",
            "as_run_file_deliberately_not_used": "results/v8/dec_analysis.json",
            "n_rows_in_file": len(raw), "n_rows_after_dedupe": len(dedup),
            "dedupe_key": "(target, axis, seed, designer, condition) — last wins",
            "rows_changed_by_v9_rescore": rescore_changed,
            "budget_fail_rows_zeroed": budget_fail,
            "v9_gate_mode": sorted({str(r.get("v9_gate_mode")) for r in dedup}),
            "conditions_in_panel": sorted({r["condition"] for r in dedup}),
            "conditions_used_by_fig2": list(C.DEC_CONDITIONS),
            "conditions_present_but_out_of_scope":
                sorted({r["condition"] for r in dedup}
                       - set(C.DEC_CONDITIONS))},
        "conventions": {
            "cell": "(target, axis); value = mean over its rows (seeds/designers)",
            "budget_fail": "nan/None removal -> 0.0",
            "absent_condition": "null (NOT 0.0) — structurally not run in that cell",
            "bootstrap": "percentile, 10k resamples, seed 0, CELL is the unit",
            "paired_pair_order": "cells in panel insertion order (v9_regen.dec)",
            "per_condition_ci_cell_order": "cells sorted by (target, axis)"},
        "n_cells": len(cells),
        "conditions_ordered_by_v9_mean_desc": order,
        "per_condition": per_condition,
        "per_condition_ci_provenance":
            "PER-CONDITION CIs ARE NEW IN v10. They are not in the registered v9 "
            "set: v9 (src/v9_regen.py) bootstrapped paired differences only, so "
            "no per-condition interval exists in RESULTS_METHOD_v9.md or "
            "results/v9/v9_diff_report.json. Same estimator family and same "
            "resampling unit as the registered deltas (v10_common.boot_mean), "
            "but they are a v10 addition and the Fig 2 caption must say so.",
        "perturb_classification": dict(C.DEC_PERTURBS),
        "perturb_classification_provenance":
            "v10_common.DEC_PERTURBS — a v10 presentation choice (marker "
            "encoding), not a registered v9 quantity.",
        "estimands": estimands,
        "unanimity": unanimity,
        "per_cell": per_cell,
        "per_cell_table": {
            "header": ["cell", "target", "axis", "axis_family"] + order,
            "rows": table,
            "note": "null = condition structurally absent in that cell "
                    "(see delta_tensor_n_explained)."},
        "coverage": coverage,
        "alpha_grid": {
            "alphas_observed": alphas, "grid_max": grid_max,
            "by_condition": alpha_counts,
            "reading": "C-a is GRID-limited: its in-budget argmax piles up at the "
                       "largest alpha in the grid. C-ref's argmax sits below the "
                       "grid max in the rest of its runs. Counts are per RUN "
                       "(cell x seed) and are ARGMAX counts, not "
                       "budget-availability counts — the two are different "
                       "quantities and must not be quoted interchangeably.",
            "field": "v9_selected_alpha (present only in the re-scored tree; the "
                     "as-run results/v8dec rows carry no selected alpha, so any "
                     "alpha-selection count quoted from v8 is an as-run number)",
            "not_computed_here": "The alpha-EXTENSION counterfactual (extrapolating "
                       "grid-capped C-a runs past the grid) is NOT computed by this "
                       "module. It is re-computable under the v9 gate — "
                       "results/v8dec/alpha_trace.jsonl carries per-alpha "
                       "bias_reduction, dmmlu and ppl_ratio for all four alphas, "
                       "which is exactly what src/v9_gate.py re-gates — but it "
                       "needs the per-alpha curve, not this panel's selected "
                       "value. Any extension number still in the manuscript is an "
                       "AS-RUN v8 quantity until that is done."},
        "delta_tensor_n_explained": tensor_note,
        "derived": {
            "concentration_ratio_C_ref_over_C_a":
                per_condition["C-ref"]["mean"] / per_condition["C-a"]["mean"],
            "C_a_share_of_C_ref": per_condition["C-a"]["mean"] / per_condition["C-ref"]["mean"],
            "delta_selection_pct_of_C_ref_range": [
                min(v["delta_selection_pct_of_C_ref"] for v in per_cell.values()),
                max(v["delta_selection_pct_of_C_ref"] for v in per_cell.values())]},
        "registered_qualifiers": {
            "grid_conditional_verbatim_from_RESULTS_METHOD_v9": pub_md["qualifier"],
            "concentrated_not_exclusive_estimand": "C-a minus C-rand",
            "concentrated_not_exclusive_value": estimands["C-a minus C-rand"]["formatted"]},
        "verification_vs_published_v9": {
            "n_checks": len(checks),
            "n_failed": sum(1 for c in checks if not c["matches"]),
            "checks": checks},
        "draft_conflicts_binary_debiaser_draft_v3": conflicts,
        "blocked": [],
    }

    fp = C.jdump(art, os.path.join(C.OUT_V10, OUT_NAME))

    # ------------------------------------------------------------ console ---
    print("=" * 78)
    print("v10 B2 — DEC under the v9 gate (Fig 2 source)")
    print("=" * 78)
    print(f"panel={PANEL} tree=results_v9 rows={len(raw)} dedup={len(dedup)} "
          f"cells={len(cells)} rescored_rows={rescore_changed} "
          f"budget_fail={budget_fail}")
    print("\n  condition       mean      per-condition CI (NEW in v10)        n  perturbs")
    for c in order:
        d = per_condition[c]
        print(f"    {c:14s} {d['mean']:+.4f}  {d['formatted']:42s} {d['n_cells']:2d}  "
              f"{d['perturbs']}")
    print("\n  registered paired estimands")
    for nm, d in estimands.items():
        print(f"    {nm:18s} {d['formatted']}")
    print(f"\n  unanimity ({unanimity['comparison']}): {unanimity['annotation']}")
    print(f"  coverage: {coverage['n_targets']} targets, {coverage['n_axes']} axes, "
          f"{coverage['n_benchmark_families']} benchmark families "
          f"{coverage['benchmark_families']}")
    print(f"  Delta_tensor n={tensor_note['n']}: no C-tensorshuf for "
          f"{tensor_note['cells_without_C_tensorshuf']}")
    print(f"\n  verification vs published v9: {len(checks)} checks, "
          f"{art['verification_vs_published_v9']['n_failed']} FAILED")
    for ch in checks:
        if not ch["matches"]:
            print(f"    MISMATCH {ch['estimand']} [{ch['source']}]: "
                  f"published={ch['published']} computed={ch['computed']}")
    print("\n  draft conflicts (binary_debiaser_draft_v3.md):")
    if not conflicts:
        print("    none")
    for c in conflicts:
        print(f"    L{c['draft_line']} {c['kind']}: draft={c['draft_says']!r} "
              f"computed={c.get('computed_at_draft_precision', c['computed'])}")
    print(f"\nwrote {fp}")
    return art


if __name__ == "__main__":
    main()
