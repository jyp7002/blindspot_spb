"""v10 — reclaim the manuscript numbers whose only source was as-run prose.

The audit registers six ship-blockers: values the paper prints that no artifact
could produce. Three of them turn out to be derivable from data already on disk,
under the v9 gate, with no new compute. This module derives them. The fourth is
shown here NOT to be derivable, with the evidence, so it stops being an open
question.

  C1  (sec 5.2)  the two-nulls table and its pooled contrasts. The edit's real
                 arm and its sign-shuffle / partition nulls are all present in
                 re-scorable v6trace panels. The published values came from
                 HARDCODED print() statements in src/method_evidence.py
                 (the C1 block prints literals, it computes nothing).
  C2  (sec 1)    sign structure, real vs random-sign at matched scale and
                 sparsity, on the frozen Tq+k+v+o protocol. results_v9/runs.jsonl
                 IS re-scored (it carries v9_gate_mode), so C2 is a v9 quantity;
                 method_evidence.py was simply never re-run against it.
  C3  (sec 5.5)  binarization retention by tier. Same source, untagged protocol.
  ALPHA-EXTENSION (sec 5.3, +0.096) is NOT derivable: the deployment grid is
                 {2,4,8,16} and NO alpha above 16 was ever probed, in any trace,
                 anywhere in the repo. The "aggressive continuation" was an
                 extrapolation, not a measurement. It needs GPU time or it needs
                 to leave the paper.

Both resampling units are reported everywhere. src/method_evidence.py bootstraps
the SWEEP (target, axis, seed, designer); the project convention, used for every
registered v9 estimand, is the target x axis CELL. v5 banned per-seed pooling as
anti-conservative, so the cell unit is the one to cite and the sweep unit is
shown only to reconcile with the published figures.

Run: python3 src/v10_regen_reclaim.py
"""
import os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C

RUNS_V9 = os.path.join(C.V9, "runs.jsonl")

# real-arm panels and the null panels that pair with them
NULL_TRIPLES = [
    ("v6trace/mvb", "v6trace/mvb_null_sign_shuffle", "v6trace/mvb_null_partition"),
    ("v6trace/fxg", "v6trace/fxnull_sign_shuffle", "v6trace/fxnull_partition"),
    ("v6trace/ax", "v6trace/ax_null_sign_shuffle", "v6trace/ax_null_partition"),
    ("v6trace/ss", "v6trace/ss_null_sign_shuffle", "v6trace/ss_null_partition"),
]
KEY = ("target", "axis", "seed", "designer", "role")
SMALL_TIER = {"gemma", "qwen", "llama", "phi"}      # <=3.8B panel names
BIG_TIER = {"qwen7b", "llama8b", "qwen_7b", "llama_8b"}


def jsonl(fp):
    import json
    if not os.path.exists(fp):
        return []
    with open(fp) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def mset(r):
    """Module-set tag, from the pipe-delimited `key`. Verbatim from
    src/method_evidence.mset -- runs.jsonl holds two protocols under the same
    variant string and they are separable only here."""
    t = [p for p in r.get("key", "").split("|") if p.startswith("T")]
    return t[0] if t else "untagged"


def best_in_budget(rs):
    ok = [x["bias_reduction"] for x in rs if x.get("collateral_ok")]
    return max(ok) if ok else 0.0                      # nan -> 0


def by_sweep(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["target"], r["axis"], r["seed"], r["designer"])].append(r)
    return {k: best_in_budget(v) for k, v in g.items()}


def by_cell_from_sweeps(sw):
    g = collections.defaultdict(list)
    for (t, a, _s, _d), v in sw.items():
        g[(t, a)].append(v)
    return {k: float(np.mean(v)) for k, v in g.items()}


def both_units(a_sw, b_sw):
    """Paired bootstrap under BOTH resampling units."""
    ks = sorted(set(a_sw) & set(b_sw))
    sweep = C.boot_paired([(a_sw[k], b_sw[k]) for k in ks]) if ks else None
    ac, bc = by_cell_from_sweeps(a_sw), by_cell_from_sweeps(b_sw)
    kc = sorted(set(ac) & set(bc))
    cell = C.boot_paired([(ac[k], bc[k]) for k in kc]) if kc else None
    return {
        "sweep_unit": (dict(sweep, text=C.fmt(sweep), status=C.status(sweep))
                       if sweep else None),
        "cell_unit": (dict(cell, text=C.fmt(cell), status=C.status(cell))
                      if cell else None),
        "unit_changes_status": (bool(sweep and cell
                                     and C.status(sweep) != C.status(cell))),
    }


# ------------------------------------------------------------------- C1 -----

def c1_nulls():
    """Real vs sign-shuffle and real vs partition, from the re-scorable v6trace
    panels, matched row-for-row on (target, axis, seed, designer, role)."""
    real, shuf, part, prov = {}, {}, {}, collections.defaultdict(set)
    for rp, sp, pp in NULL_TRIPLES:
        for src, dst in ((rp, real), (sp, shuf), (pp, part)):
            for r in C.rows(C.V9, src):
                k = tuple(r.get(x) for x in KEY)
                dst[k] = C.z(r.get("removal"))
                prov[k[:2]].add(src)

    # The draft's sec 5.2 table names four (target, axis, role) cells. Pooling
    # over EVERY matched cell answers a different question -- it drags in the
    # low-signal bbq cells -- so both populations are reported and labelled.
    DRAFT_CELLS = {("phi", "occ_gender", "self"),
                   ("phi", "occ_gender", "cross_large"),
                   ("qwen", "crows_socioeconomic", "cross_large"),
                   ("gemma", "occ_gender", "same_large")}

    rows, pooled = [], {}
    for label, null in (("sign_shuffle", shuf), ("partition", part)):
        ks_all = sorted(set(real) & set(null), key=lambda x: tuple(map(str, x)))
        if not ks_all:
            pooled[f"real_minus_{label}"] = None
            continue
        for pop, ks in (("all_matched_cells", ks_all),
                        ("draft_table_cells", [k for k in ks_all
                                               if (k[0], k[1], k[4]) in DRAFT_CELLS])):
            if not ks:
                pooled[f"real_minus_{label} [{pop}]"] = None
                continue
            sw_r = {k[:4]: real[k] for k in ks}
            sw_n = {k[:4]: null[k] for k in ks}
            pooled[f"real_minus_{label} [{pop}]"] = both_units(sw_r, sw_n)
        ks = ks_all
        cells = collections.defaultdict(lambda: collections.defaultdict(list))
        for k in ks:
            cells[(k[0], k[1], k[4])]["real"].append(real[k])
            cells[(k[0], k[1], k[4])][label].append(null[k])
        for (t, a, role), d in cells.items():
            rows.append({"target": t, "axis": a, "role": role, "null": label,
                         "real": float(np.mean(d["real"])),
                         "null_value": float(np.mean(d[label])),
                         "n_seeds": len(d["real"]),
                         "panels": sorted(prov[(t, a)])})
    return {
        "_what": ("sec 5.2 two-nulls table, re-derived under the v9 gate from "
                  "re-scorable v6trace panels"),
        "_published_source": ("src/method_evidence.py prints the sec 5.2 cells and "
                              "the pooled +0.577 / +0.550 as HARDCODED literals; "
                              "nothing computed them"),
        "_panels": [list(t) for t in NULL_TRIPLES],
        "_match_key": list(KEY),
        "per_cell": sorted(rows, key=lambda r: (r["null"], r["target"], r["axis"])),
        "pooled": pooled,
    }


# ---------------------------------------------------------------- C2 / C3 ---

def _runs_pool():
    rs = jsonl(RUNS_V9)
    return [r for r in rs if r.get("signal") == "endogenous"], rs


def c2_sign_structure():
    endo, allrows = _runs_pool()
    if not endo:
        return {"_blocked": f"{RUNS_V9} absent or empty"}
    sel = lambda v, tag: [r for r in endo
                          if r.get("variant") == v and mset(r) == tag]
    real = by_sweep(sel("binary-per_tensor-s0.0", "Tq+k+v+o"))
    rand = by_sweep(sel("binary-per_tensor-s0.0-rand", "Tq+k+v+o"))
    ks = sorted(set(real) & set(rand))
    res = both_units(real, rand)
    res.update({
        "_what": "C2: real vs random-sign at matched scale and sparsity",
        "_protocol": "Tq+k+v+o (frozen)",
        "_gate": "v9 (results_v9/runs.jsonl carries v9_gate_mode)",
        "real_mean": float(np.mean([real[k] for k in ks])) if ks else None,
        "random_mean": float(np.mean([rand[k] for k in ks])) if ks else None,
        "n_sweeps": len(ks),
        "n_cells": len(by_cell_from_sweeps({k: real[k] for k in ks})),
    })
    return res


def _tier(target):
    return "big" if target in BIG_TIER else ("small" if target in SMALL_TIER else None)


def c3_binarization():
    """RESULTS_METHOD.md sec 3 reports n=48 at <=3.8B (4 targets) and n=24 at
    7-9B (2 targets). Those are exactly the row counts of results_v9/v6trace/
    ablate and ablate_big per variant -- NOT runs.jsonl, whose untagged pool is
    the v2-era layer-placement ablation (qwen1.5b/llama1b/smol1.7b) and carries
    fp--s0 for only two of those. The ablate panels are re-scored, so C3 is a v9
    quantity."""
    out = {"_what": "C3: binarization vs full precision, paired on identical dW",
           "_source": ["results_v9/v6trace/ablate (<=3.8B, 4 targets)",
                       "results_v9/v6trace/ablate_big (7-9B, 2 targets)"],
           "_pairing": "same (target, axis, seed, designer) row, variant is the "
                       "only difference",
           "_gate": "v9", "tiers": {}}
    for panel, tier in (("v6trace/ablate", "small"), ("v6trace/ablate_big", "big")):
        rs = C.rows(C.V9, panel)
        if not rs:
            continue
        grab = lambda v: {(r["target"], r["axis"], r["seed"], r["designer"]):
                          C.z(r.get("removal"))
                          for r in rs if r.get("variant") == v}
        b, f = grab("binary-per_tensor-s0.0"), grab("fp")
        ks = sorted(set(b) & set(f))
        if not ks:
            continue
        rec = both_units({k: b[k] for k in ks}, {k: f[k] for k in ks})  # binary - fp
        fm = float(np.mean([f[k] for k in ks]))
        bm = float(np.mean([b[k] for k in ks]))
        rec.update({"panel": panel, "fp_mean": fm, "binary_mean": bm,
                    "retention_pct": (100.0 * bm / fm) if fm else None,
                    "n_sweeps": len(ks),
                    "n_cells": len(by_cell_from_sweeps({k: b[k] for k in ks})),
                    "targets": sorted({k[0] for k in ks})})
        out["tiers"][tier] = rec
    return out


# ------------------------------------------------- alpha extension: absent ---

def alpha_extension_status():
    import glob, json as _j
    alphas, per_file = set(), {}
    for fp in glob.glob(os.path.join(C.RES, "**", "alpha_trace.jsonl"),
                        recursive=True):
        a = set()
        for l in open(fp):
            try:
                a.add(_j.loads(l)["alpha"])
            except Exception:                              # noqa: BLE001
                pass
        if a:
            per_file[os.path.relpath(os.path.dirname(fp), C.RES)] = sorted(a)
            alphas |= a
    return {
        "_what": ("sec 5.3's 'under aggressive alpha extension the gap shrinks to "
                  "+0.096 and covers 0'"),
        "derivable": False,
        "alphas_ever_probed": sorted(alphas),
        "max_alpha": max(alphas) if alphas else None,
        "deployment_grid": [2.0, 4.0, 8.0, 16.0],
        "n_trace_files_scanned": len(per_file),
        "why_blocked": ("no alpha above the grid maximum was ever probed, in any "
                        "trace, anywhere in results/. The 'continuation' of "
                        "grid-capped runs is therefore an extrapolation, not a "
                        "measurement, and cannot be re-derived at any gate. It "
                        "needs new GPU runs or it must leave the manuscript."),
        "what_IS_derivable": ("the count of grid-capped C-a runs: 27 of 30 under "
                              "v9 vs 26 of 30 as-run (see "
                              "results/v10/dec_analysis_v9.json alpha_grid)"),
    }


def main():
    C.ensure_dirs()
    rep = {
        "artifact": "v10_regen_reclaim",
        "module": "src/v10_regen_reclaim.py",
        "purpose": ("derive the audit's as-run-only blockers from data already on "
                    "disk, under the v9 gate, with no new compute"),
        "resampling_unit_note": (
            "both units reported. method_evidence.py bootstraps the SWEEP; the "
            "project convention for every registered v9 estimand is the target x "
            "axis CELL. v5 banned per-seed pooling as anti-conservative, so cite "
            "the cell unit."),
        "c1_nulls": c1_nulls(),
        "c2_sign_structure": c2_sign_structure(),
        "c3_binarization": c3_binarization(),
        "alpha_extension": alpha_extension_status(),
    }
    fp = C.jdump(rep, "results/v10/reclaimed.json")

    print("[v10 reclaim]")
    c1 = rep["c1_nulls"]
    print(f"  C1 sec 5.2 nulls: {len(c1['per_cell'])} cell-rows re-derived")
    for name, d in c1["pooled"].items():
        if not d:
            print(f"    {name}: NO MATCHED ROWS")
            continue
        print(f"    {name:26s} cell  {d['cell_unit']['text'] if d['cell_unit'] else 'n/a'}")
        print(f"    {'':26s} sweep {d['sweep_unit']['text'] if d['sweep_unit'] else 'n/a'}"
              + ("   <- UNIT CHANGES STATUS" if d["unit_changes_status"] else ""))
    c2 = rep["c2_sign_structure"]
    if "_blocked" not in c2:
        print(f"  C2 sign structure: real {c2['real_mean']:+.4f} vs random "
              f"{c2['random_mean']:+.4f}, n={c2['n_sweeps']} sweeps / "
              f"{c2['n_cells']} cells")
        print(f"    cell  {c2['cell_unit']['text'] if c2['cell_unit'] else 'n/a'}")
        print(f"    sweep {c2['sweep_unit']['text'] if c2['sweep_unit'] else 'n/a'}"
              + ("   <- UNIT CHANGES STATUS" if c2["unit_changes_status"] else ""))
    c3 = rep["c3_binarization"]
    if "_blocked" not in c3:
        print("  C3 binarization retention:")
        for tier, d in sorted(c3["tiers"].items()):
            print(f"    {tier:6s} n={d['n_sweeps']:3d} sweeps / {d.get('n_cells','?')} cells  "
                  f"fp {d['fp_mean']:+.4f}  binary {d['binary_mean']:+.4f}  "
                  f"retention {d['retention_pct']:6.1f}%")
            print(f"    {'':6s}   cell  {d['cell_unit']['text'] if d['cell_unit'] else 'n/a'}")
            print(f"    {'':6s}   sweep {d['sweep_unit']['text'] if d['sweep_unit'] else 'n/a'}"
                  + ("   <- UNIT CHANGES STATUS" if d["unit_changes_status"] else ""))
    ax = rep["alpha_extension"]
    print(f"  alpha extension: derivable={ax['derivable']} "
          f"(max alpha ever probed = {ax['max_alpha']})")
    print(f"wrote {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
