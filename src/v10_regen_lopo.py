"""v10 — Fig 4 RIGHT panel source: leave-one-projection-out (LOPO), under the v9 gate.

WHERE THE DATA ACTUALLY IS
--------------------------
LOPO was never a separate panel. `run_dec.py` appends four extra conditions to
the DEC factorial — `C-ref-no_{q,k,v,o}_proj` (see `run_dec.LOPO` /
`run_dec.lopo_edit`, which zeroes one projection's surviving coordinates in the
*same* C-ref edit) — so the rows live in the **v8dec** panel, next to C-ref.
`results_v9/v8dec/removal.jsonl` carries them with `v9_selected_alpha` /
`v9_changed`, i.e. the LOPO arm IS re-scorable and IS true-v9.

  * `results_v9/v6trace/ablate*` holds ONLY binarization/sparsity variants
    (`fp`, `binary-per_tensor-s*`, `binary-per_channel-s0.0`, `binary-scalar-s0.0`).
    It has no per-projection ablation. That reconnaissance claim is correct but
    it is not where LOPO lives.

CELL / BOOTSTRAP CONVENTIONS are inherited wholesale from `v10_common`
(= `v9_regen`): dedupe on (target, axis, seed, designer, condition), a cell is
the MEAN over its rows, budget-fail -> 0.0, 10k paired percentile bootstrap over
CELLS at seed 0.

PHI IS STRUCTURALLY ABSENT, NOT DROPPED HERE. `run_dec.applicable_conditions`
refuses to emit LOPO for a one-projection-per-layer architecture: phi-3.5 fuses
q/k/v into `qkv_proj` and its ATTN fallback never resolves `o_proj`, so every
`C-ref-no_<p>` would match no module and return C-ref unchanged — a structural
null, not a measurement. That is why the LOPO arm has 7 cells where the rest of
DEC has 10. This module re-derives that 7 from the data instead of asserting it.

DENSITY-IS-NOT-IMPORTANCE (Fig 4's caption pairs the bars with the heatmap).
Per-projection support enrichment comes from `results/v8/sup1_analysis.json`.
Those values are **gate-invariant**: `run_dec.py` dumps the support from
`build_edit(..., "C-ref", ...)` BEFORE the alpha loop, and the v9 re-score only
re-selects alpha (`v9_rescore.py`), so no re-scored twin of the support dumps
can exist or is needed. Enrichment is reported over the phi-excluded dump set
and, separately, over the exact dumps that also carry LOPO rows.

Emits: results/v10/lopo_v9.json          (the ONLY output of this module)
"""
import os
import re
import sys
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10_common import (HERE, RES, V9, OUT_V10, z, rows, jload, jdump,
                        boot_paired, status, fmt, ensure_dirs)

PANEL = "v8dec"
REF = "C-ref"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
LOPO_COND = {p: f"{REF}-no_{p}" for p in PROJECTIONS}
SUP1 = os.path.join("results", "v8", "sup1_analysis.json")
DIFF_REPORT = os.path.join("results", "v9", "v9_diff_report.json")

# Documents that carry the currently-printed LOPO numbers. Parsed at run time so
# the comparison contains no typed literals (the R2-bug rule).
DRAFT = "binary_debiaser_draft_v3.md"
V8_FINDINGS = "V8_FINDINGS.md"


# ------------------------------------------------------------ cell machinery --

def dec_cells_with_ids(root, panel=PANEL):
    """Same dedupe + cell definition as v10_common.dec_cells, but it also keeps
    the row identities behind every cell so the support dumps can be matched to
    exactly the runs the LOPO bars are built from."""
    seen = {}
    for r in rows(root, panel):
        seen[(r["target"], r["axis"], r["seed"], r["designer"], r["condition"])] = r
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    ids = collections.defaultdict(lambda: collections.defaultdict(list))
    for (t, a, s, d, c), r in seen.items():
        cells[(t, a)][c].append(z(r.get("removal")))
        ids[(t, a)][c].append(f"{t}|{a}|s{s}|{d}")
    means = {k: {c: float(np.mean(v)) for c, v in d.items()} for k, d in cells.items()}
    return means, {k: {c: sorted(v) for c, v in d.items()} for k, d in ids.items()}


def lopo_arm(root):
    """The four paired deltas C-ref - C-ref-no_<p>, plus everything the figure
    needs to draw and caption itself."""
    cells, ids = dec_cells_with_ids(root)
    out = {"deltas": {}, "condition_means": {}, "per_cell": {}, "cells": {}}

    # ORDER MATTERS. boot_paired resamples indices into the paired list with a
    # FIXED seed, so a different iteration order over the cells yields a
    # different (equally valid) interval from the same data. v9_regen builds its
    # pair list from `C.values()` — dict INSERTION order, i.e. first appearance
    # in removal.jsonl — so this module does too. Sorting the keys instead moves
    # every published CI in the 3rd-4th decimal. Do not "tidy" this into sorted().
    for p, cond in LOPO_COND.items():
        keys = [k for k, m in cells.items() if REF in m and cond in m]
        pairs = [(cells[k][REF], cells[k][cond]) for k in keys]
        d = boot_paired(pairs)
        out["deltas"][p] = None if d is None else dict(
            point=d["point"], lo=d["lo"], hi=d["hi"], n=d["n"], status=status(d),
            reading="load-bearing" if status(d) == "excludes 0" else "droppable",
            pretty=fmt(d))
        out["cells"][p] = [f"{t}|{a}" for (t, a) in keys]
        out["per_cell"][p] = {f"{t}|{a}": dict(c_ref=cells[(t, a)][REF],
                                               lopo=cells[(t, a)][cond],
                                               delta=cells[(t, a)][REF] - cells[(t, a)][cond])
                              for (t, a) in keys}

    # Condition means restricted to the LOPO cell set, so the bar chart's
    # baseline is the same 7 cells the deltas are computed on (the all-10-cell
    # C-ref mean is a DIFFERENT number and must not be mixed in).
    lopo_keys = [k for k, m in cells.items()
                 if REF in m and all(c in m for c in LOPO_COND.values())]
    out["lopo_cell_set"] = [f"{t}|{a}" for (t, a) in lopo_keys]
    out["n_cells"] = len(lopo_keys)
    for c in (REF,) + tuple(LOPO_COND[p] for p in PROJECTIONS):
        out["condition_means"][c] = float(np.mean([cells[k][c] for k in lopo_keys]))
    # The same C-ref averaged over ALL DEC cells — recorded only so the figure
    # can prove it did not use it.
    out["c_ref_mean_all_dec_cells"] = float(
        np.mean([m[REF] for m in cells.values() if REF in m]))
    out["n_cells_all_dec"] = sum(1 for m in cells.values() if REF in m)

    # Which cells DEC has but LOPO does not (the structural phi exclusion).
    out["cells_without_lopo"] = [f"{t}|{a}" for (t, a), m in cells.items()
                                 if REF in m and (t, a) not in set(lopo_keys)]
    out["cell_order_note"] = ("bootstrap pair order = removal.jsonl first-appearance "
                              "order, verbatim from v9_regen's C.values(); the "
                              "fixed-seed CI is not order-invariant")
    out["_ids"] = {f"{t}|{a}": ids[(t, a)] for (t, a) in lopo_keys}
    return out


def gate_provenance(root):
    """v9 re-selection evidence for exactly the rows this arm consumes."""
    conds = {REF} | set(LOPO_COND.values())
    acc = collections.defaultdict(lambda: dict(rows=0, v9_changed=0, gate_modes=set(),
                                               has_v9_fields=False))
    for r in rows(root, PANEL):
        if r["condition"] not in conds:
            continue
        a = acc[r["condition"]]
        a["rows"] += 1
        if "v9_selected_alpha" in r:
            a["has_v9_fields"] = True
        if r.get("v9_changed"):
            a["v9_changed"] += 1
        if r.get("v9_gate_mode"):
            a["gate_modes"].add(r["v9_gate_mode"])
    return {c: dict(rows=v["rows"], v9_changed=v["v9_changed"],
                    gate_modes=sorted(v["gate_modes"]),
                    has_v9_fields=v["has_v9_fields"]) for c, v in acc.items()}


# ------------------------------------------------- density is not importance --

def support_density(lopo_ids):
    """Per-projection support enrichment from the SUP1 structural analysis.

    `enrichment` = (share of surviving coordinates in projection p) / (share of
    parameters in p) — 1.0x means "exactly its proportional share".

    Three views, because they are three different denominators and the draft
    quotes only the middle one:
      all_dumps        — every support dump, phi's fused qkv_proj included
      four_projection  — dumps whose architecture exposes q/k/v/o (phi excluded)
      lopo_matched     — only the dumps that also produced a LOPO removal row
    """
    sup = jload(SUP1)
    if sup is None:
        return {"available": False,
                "reason": f"{SUP1} not found under {HERE}"}
    st = sup.get("structure", {})
    four = set(PROJECTIONS)

    def view(pred):
        acc = collections.defaultdict(list)
        dumps, targets = [], collections.Counter()
        for k, v in st.items():
            pr = v.get("projection", {})
            if not pred(k, v, pr):
                continue
            dumps.append(k)
            targets[v.get("meta", {}).get("target")] += 1
            for p, x in pr.items():
                acc[p].append(float(x["enrichment"]))
        return dict(n_dumps=len(dumps),
                    targets=dict(sorted(targets.items())),
                    enrichment={p: float(np.mean(v)) for p, v in sorted(acc.items())},
                    share_of_support={p: float(np.mean(
                        [float(st[k]["projection"][p]["share_of_support"])
                         for k in dumps if p in st[k]["projection"]]))
                        for p in sorted(acc)},
                    dumps=sorted(dumps))

    lopo_id_set = set()
    for cell in lopo_ids.values():
        for c in cell.values():
            lopo_id_set.update(c)

    views = {
        "all_dumps": view(lambda k, v, pr: True),
        "four_projection": view(lambda k, v, pr: set(pr) == four),
        "lopo_matched": view(lambda k, v, pr: set(pr) == four and k in lopo_id_set),
    }
    fused = view(lambda k, v, pr: set(pr) != four)
    views["fused_qkv_excluded"] = dict(
        n_dumps=fused["n_dumps"], targets=fused["targets"],
        enrichment=fused["enrichment"],
        note="a single fused projection cannot be enriched relative to itself; "
             "its enrichment is 1.0 by construction, which is why phi is excluded "
             "from the projection comparison")

    # Depth thirds — the §5.6 companion number, recomputed on both denominators
    # because the layer axis of the Fig 4 heatmap and its projection axis do NOT
    # rest on the same dump set.
    def thirds(pred):
        acc = collections.defaultdict(list)
        for k, v in st.items():
            if not pred(k, v, v.get("projection", {})):
                continue
            for t in ("early", "middle", "late"):
                acc[t].append(float(v["thirds"][t]["enrichment"]))
        return {t: float(np.mean(x)) for t, x in acc.items()}

    views["_depth_thirds"] = {
        "all_dumps": thirds(lambda k, v, pr: True),
        "four_projection": thirds(lambda k, v, pr: set(pr) == set(PROJECTIONS)),
    }
    views["available"] = True
    views["n_dumps_total"] = sup.get("n_dumps")
    views["source"] = SUP1
    views["gate"] = ("GATE-INVARIANT: supports are dumped from build_edit(C-ref) "
                     "before the alpha loop (run_dec.py); the v9 re-score only "
                     "re-selects alpha, so there is no v9 twin and none is needed")
    return views


def support_share_from_dec(root):
    """Independent cross-check on the density story, from the DEC rows alone.

    Every LOPO row records `n_flips` (surviving coordinates in the edit it
    applied), so n_flips(C-ref) - n_flips(C-ref-no_p) is the coordinate count
    that lived in projection p — computed with no reference to the .npz dumps.
    Its four shares must sum to 1.
    """
    seen = {}
    for r in rows(root, PANEL):
        seen[(r["target"], r["axis"], r["seed"], r["designer"], r["condition"])] = r
    by = collections.defaultdict(dict)
    for (t, a, s, d, c), r in seen.items():
        by[(t, a, s, d)][c] = r
    acc = collections.defaultdict(list)
    used = 0
    for ident, d in by.items():
        if REF not in d or not all(c in d for c in LOPO_COND.values()):
            continue
        ref = d[REF].get("n_flips")
        if not ref:
            continue
        used += 1
        for p, cond in LOPO_COND.items():
            lp = d[cond].get("n_flips")
            if lp is None:
                continue
            acc[p].append((ref - lp) / float(ref))
    share = {p: float(np.mean(v)) for p, v in sorted(acc.items())}
    return dict(n_dumps=used, share_of_support=share,
                sums_to=float(sum(share.values())),
                method="(n_flips[C-ref] - n_flips[C-ref-no_p]) / n_flips[C-ref]")


# --------------------------------------------------------- printed-doc parse --

_NUM = r"[-+−]?\d*\.?\d+"


def _f(tok):
    return float(tok.replace("−", "-").replace("+", ""))


def parse_printed_lopo(relpath):
    """Pull whatever LOPO table a markdown document prints, so the comparison
    against the manuscript contains no hand-typed numbers."""
    fp = os.path.join(HERE, relpath)
    if not os.path.exists(fp):
        return {}
    out = {}
    pat = re.compile(
        r"^\|\s*\**(" + "|".join(PROJECTIONS) + r")\**\s*\|\s*\**\s*(" + _NUM +
        r")\**\*?\s*\[\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\]")
    for i, line in enumerate(open(fp), start=1):
        m = pat.match(line.strip())
        if m:
            out.setdefault(m.group(1), dict(
                line=i, point=_f(m.group(2)), lo=_f(m.group(3)), hi=_f(m.group(4)),
                text=line.strip()))
    return out


def parse_printed_density(relpath):
    """`v_proj holds fewer surviving coordinates (0.64x) than q_proj (0.83x)` and
    the RESULTS_V8 table row form `| q_proj | 0.83x |`."""
    fp = os.path.join(HERE, relpath)
    if not os.path.exists(fp):
        return {}
    out = {}
    inline = re.compile(r"(" + "|".join(PROJECTIONS) + r")\b[^|\n]{0,80}?\((" +
                        _NUM + r")×\)")
    tablerow = re.compile(r"^\|\s*\**(" + "|".join(PROJECTIONS) +
                          r")\**\s*\|\s*\**(" + _NUM + r")×\**\s*\|")
    for i, line in enumerate(open(fp), start=1):
        for m in inline.finditer(line):
            out.setdefault(m.group(1), dict(line=i, value=_f(m.group(2)),
                                            text=line.strip()[:200]))
        m = tablerow.match(line.strip())
        if m:
            out.setdefault(m.group(1), dict(line=i, value=_f(m.group(2)),
                                            text=line.strip()[:200]))
    return out


# ------------------------------------------------------------ reproductions --

def reproduce_published(cells_v9):
    """Re-derive the DEC estimands that DO exist in the citable v9 set, using
    this module's own cell machinery, and compare. If these match, the LOPO
    numbers below were produced by the same pipeline that produced the published
    ones; if they do not, nothing here should be trusted."""
    pub = jload(DIFF_REPORT)
    checks = {}
    if pub is None:
        return {"available": False, "reason": f"{DIFF_REPORT} missing"}
    pairs = (("Delta_selection", "C-ref", "C-a"),
             ("Delta_sign", "C-ref", "C-b"),
             ("Delta_null", "C-ref", "C-rand"),
             ("Delta_location", "C-ref", "C-layershuf"),
             ("Delta_tensor", "C-ref", "C-tensorshuf"),
             ("C-a minus C-rand", "C-a", "C-rand"))
    for name, a, b in pairs:
        d = boot_paired([(m[a], m[b]) for m in cells_v9.values() if a in m and b in m])
        entry = pub["diff"].get(f"DEC :: {name}", {})
        checks[name] = dict(computed=fmt(d), published=entry.get("v9"),
                            published_point=entry.get("v9_point"),
                            computed_point=None if d is None else round(d["point"], 6),
                            matches=(entry.get("v9") == fmt(d)))
    means = pub.get("DEC_means_v9", {})
    mine = {c: round(float(np.mean([m[c] for m in cells_v9.values() if c in m])), 4)
            for c in means}
    checks["_DEC_means_v9"] = dict(computed=mine, published=means,
                                   matches=(mine == means))
    checks["available"] = True
    return checks


# ------------------------------------------------------------------- main ----

def main():
    ensure_dirs()

    v9 = lopo_arm(V9)
    asrun = lopo_arm(RES)
    ids = v9.pop("_ids")
    asrun.pop("_ids", None)

    cells_v9, _ = dec_cells_with_ids(V9)
    recon = reproduce_published(cells_v9)

    dens = support_density(ids)
    dens_dec = support_share_from_dec(V9)

    printed = {
        "draft_v3_5.4": parse_printed_lopo(DRAFT),
        "V8_FINDINGS_2": parse_printed_lopo(V8_FINDINGS),
        "draft_v3_density": parse_printed_density(DRAFT),
    }

    # v9 vs as-run, and v9 vs whatever the manuscript currently prints.
    delta_table = {}
    for p in PROJECTIONS:
        a, b = asrun["deltas"][p], v9["deltas"][p]
        pr = printed["draft_v3_5.4"].get(p)
        delta_table[p] = dict(
            as_run=a, v9=b,
            ci_status_change=("none" if a["status"] == b["status"]
                              else f"CHANGED: {a['status']} -> {b['status']}"),
            sign_change=("none" if (a["point"] > 0) == (b["point"] > 0)
                         else f"CHANGED: {a['point']:+.4f} -> {b['point']:+.4f}"),
            reading_change=("none" if a["reading"] == b["reading"]
                            else f"CHANGED: {a['reading']} -> {b['reading']}"),
            draft_prints=pr,
            draft_matches_as_run=(pr is not None and
                                  abs(pr["point"] - a["point"]) < 5e-4),
            draft_matches_v9=(pr is not None and
                              abs(pr["point"] - b["point"]) < 5e-4))

    # Does the as-run arm reproduce the LOPO table V8_FINDINGS printed? If it
    # does, the only thing separating the draft from this artifact is the gate.
    asrun_recon = {}
    for p in PROJECTIONS:
        pub = printed["V8_FINDINGS_2"].get(p)
        mine = asrun["deltas"][p]
        asrun_recon[p] = dict(
            published=None if pub is None else
            f"{pub['point']:+.4f} [{pub['lo']:+.4f}, {pub['hi']:+.4f}]",
            published_line=None if pub is None else pub["line"],
            computed=f"{mine['point']:+.4f} [{mine['lo']:+.4f}, {mine['hi']:+.4f}]",
            matches=(pub is not None and
                     abs(pub["point"] - mine["point"]) < 5e-5 and
                     abs(pub["lo"] - mine["lo"]) < 5e-5 and
                     abs(pub["hi"] - mine["hi"]) < 5e-5))

    art = {
        "artifact": "v10_lopo",
        "generated_by": "src/v10_regen_lopo.py",
        "figure": "Fig 4 RIGHT — LOPO paired bars with CIs (q, k, v, o)",
        "gate": {
            "status": "TRUE-V9",
            "panel": f"results_v9/{PANEL}/removal.jsonl",
            "why": "LOPO conditions are DEC conditions (run_dec.py appends "
                   "C-ref-no_<proj> to the factorial); the v8dec panel has an "
                   "alpha_trace and was re-scored, so the LOPO arm is re-scorable. "
                   "No as-run substitution anywhere in this artifact.",
            "not_rescorable_panels_touched": [],
            "recon_note": "results_v9/v6trace/ablate and ablate_big carry ONLY "
                          "binarization/sparsity variants (fp, binary-per_tensor-s*, "
                          "binary-per_channel-s0.0, binary-scalar-s0.0) — no "
                          "per-projection ablation. Verified; LOPO is not there.",
            "row_provenance_v9": gate_provenance(V9),
        },
        "conventions": {
            "cell": "(target, axis); value = MEAN over its rows (seeds x designers)",
            "dedupe": "(target, axis, seed, designer, condition) — last row wins",
            "budget_fail": "nan removal -> 0.0",
            "bootstrap": "paired percentile over CELLS, 10k, seed 0 (v10_common.boot_paired)",
            "pair_order": "dict insertion order (first appearance in removal.jsonl), "
                          "verbatim from v9_regen — a fixed-seed bootstrap CI is "
                          "NOT invariant to this order",
            "estimand": "C-ref minus C-ref-no_<proj>; positive = dropping that "
                        "projection COSTS removal = load-bearing",
        },
        "lopo": {
            "v9": v9,
            "as_run": asrun,
            "comparison": delta_table,
        },
        "phi_exclusion": {
            "excluded": True,
            "mechanism": "structural, upstream of this module: "
                         "run_dec.applicable_conditions() emits no C-ref-no_<p> "
                         "when the architecture exposes <2 attention projections",
            "reason": "phi-3.5 fuses q/k/v into qkv_proj and its ATTN fallback "
                      "never resolves o_proj, so every LOPO edit would match no "
                      "module and return C-ref unchanged (a structural null)",
            "dec_cells_total": v9["n_cells_all_dec"],
            "lopo_cells": v9["n_cells"],
            "cells_dropped": v9["cells_without_lopo"],
            "support_dumps_total": dens.get("n_dumps_total"),
            "support_dumps_after_exclusion": dens.get("four_projection", {}).get("n_dumps"),
            "support_dumps_fused": dens.get("fused_qkv_excluded", {}).get("n_dumps"),
            "support_dumps_matched_to_lopo": dens.get("lopo_matched", {}).get("n_dumps"),
        },
        "support_density": dens,
        "support_share_cross_check_from_dec_rows": dens_dec,
        "printed_values_parsed": printed,
        "reproduced_published_v9_dec": recon,
        "reproduced_published_as_run_lopo": dict(
            source=V8_FINDINGS + " section 2 (as-run / v8; NOT citable under v9)",
            per_projection=asrun_recon,
            all_match=all(v["matches"] for v in asrun_recon.values())),
    }

    fp = jdump(art, os.path.join(OUT_V10, "lopo_v9.json"))

    # ------------------------------------------------------------- console --
    print("=" * 78)
    print("v10 LOPO — Fig 4 RIGHT panel source")
    print("=" * 78)
    print(f"panel: results_v9/{PANEL}  (TRUE-V9, re-scorable)")
    print(f"cells: {v9['n_cells']} of {v9['n_cells_all_dec']} DEC cells; "
          f"dropped (structural, phi): {', '.join(v9['cells_without_lopo'])}")
    print(f"\nC-ref mean over the {v9['n_cells']} LOPO cells: "
          f"{v9['condition_means'][REF]:+.4f}"
          f"   (over all {v9['n_cells_all_dec']} DEC cells: "
          f"{v9['c_ref_mean_all_dec_cells']:+.4f})")
    print("\ndropped     as-run (v8, what the draft prints)      v9 (citable)")
    for p in PROJECTIONS:
        a, b = asrun["deltas"][p], v9["deltas"][p]
        print(f"  {p:8s}  {a['pretty']}")
        print(f"  {'':8s}  {b['pretty']}   <- v9")
        for tag in ("ci_status_change", "sign_change", "reading_change"):
            if delta_table[p][tag] != "none":
                print(f"  {'':8s}  !! {tag}: {delta_table[p][tag]}")
    print("\nreproduction of published v9 DEC estimands (same cell machinery):")
    for k, v in recon.items():
        if k.startswith("_") or k == "available":
            continue
        print(f"  {k:20s} {'OK ' if v['matches'] else 'MISMATCH'} "
              f"computed={v['computed']}")
    mm = recon.get("_DEC_means_v9", {})
    print(f"  {'DEC means':20s} {'OK ' if mm.get('matches') else 'MISMATCH'}")
    print(f"\nas-run arm vs the LOPO table printed in {V8_FINDINGS}: "
          f"{'ALL MATCH' if art['reproduced_published_as_run_lopo']['all_match'] else 'MISMATCH'}")
    for p, v in asrun_recon.items():
        print(f"  {p:8s} published(L{v['published_line']}) {v['published']}  "
              f"computed {v['computed']}  {'OK' if v['matches'] else 'MISMATCH'}")
    if dens.get("available"):
        print("\ndensity is not importance — support enrichment "
              f"(phi excluded, n={dens['four_projection']['n_dumps']} dumps):")
        for p, v in sorted(dens["four_projection"]["enrichment"].items(),
                           key=lambda kv: -kv[1]):
            d = v9["deltas"][p]
            print(f"  {p:8s} {v:.3f}x   LOPO {d['point']:+.4f} ({d['reading']})")
        print(f"  fused qkv_proj (phi, excluded): "
              f"{dens['fused_qkv_excluded']['enrichment']} over "
              f"{dens['fused_qkv_excluded']['n_dumps']} dumps")
        print(f"  cross-check from DEC n_flips (n={dens_dec['n_dumps']} dumps), "
              f"share of support: "
              + ", ".join(f"{p}={s:.3f}" for p, s in dens_dec["share_of_support"].items())
              + f"  (sums to {dens_dec['sums_to']:.4f})")
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
