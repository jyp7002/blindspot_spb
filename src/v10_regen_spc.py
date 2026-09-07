"""v10 — Fig 3 source: the full v9 sparsity curves, both tiers.

`v9_regen.spc` computes only the four sparsity levels that the manuscript
quotes (0.9, 0.99, 0.995, 0.999).  Fig 3 draws a *curve*, so this module covers
every level actually present in `results_v9/v8spc/{small,big}/removal.jsonl`
and adds the three things a deployability figure needs and the v9 regen never
recorded:

  1. the per-target cell means at every level (the dots behind the CI band),
  2. the patch size per (target, sparsity) in bytes and MiB (the second x axis),
  3. the **budget-fail inventory** under the v9 gate, at (target, seed,
     sparsity) resolution, contrasted against the as-run fail set.

Conventions are inherited wholesale from `v10_common` / `v9_regen.spc`:
a cell is a TARGET (the panel has one axis, occ_gender), a cell's value is the
MEAN over its seed rows, budget-fail -> 0.0, and the paired bootstrap resamples
cells at 10k with seed 0.  The pair list is built in target-first-appearance
order so the RNG draw sequence is bit-identical to `v9_regen.spc`; the four
published levels are re-checked against `results/v9/v9_diff_report.json` at run
time and any disagreement is reported, not tuned away.

HOW A BUDGET-FAIL IS ENCODED (established from the artifacts, see
`budget_fail_encoding` in the output).  These panels are re-scored by
`v9_rescore.rescore_trace_panel` (path A), which re-applies the corrected gate
to every probed alpha and re-picks the in-budget argmax.  When NO probed alpha
clears the gate for a cell there is no row to drop -- the row survives with
`removal = NaN` and `v9_selected_alpha = null`.  So a budget-fail is a NaN
removal, never a missing row and never a `collateral_ok` flag (the spc removal
rows carry no such flag; the flag lives one level down, in alpha_trace.jsonl).
The module verifies the NaN <-> null-alpha equivalence, and independently
re-derives the same fail set by re-applying `v9_gate` to the raw alpha traces.

Emits: results/v10/spc_v9.json          (the only file this module writes)
"""
import os
import sys
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v10_common import (HERE, RES, V9, OUT_V10, z, rows, jload, jdump,
                        boot_paired, status, fmt, mib, ensure_dirs)
import v9_gate

PANEL = "v8spc"
TIERS = ("small", "big")
OUT_REL = os.path.join("results", "v10", "spc_v9.json")


# ------------------------------------------------------------------ atoms ---

def is_fail(r):
    """A budget-fail row: the re-scorer found no in-budget alpha for the cell.

    NaN survives json.loads as a float NaN, so the test is the canonical
    self-inequality.  `z()` is what turns this into the 0.0 that enters every
    mean -- see v10_common.
    """
    v = r.get("removal")
    return v is None or (isinstance(v, float) and v != v)


def target_order(rs):
    """Targets in first-appearance order -- the order `v9_regen.spc` builds its
    pair list in, and therefore the order the bootstrap RNG consumes."""
    seen = []
    for r in rs:
        if r["target"] not in seen:
            seen.append(r["target"])
    return seen


def levels_present(rs):
    return sorted({float(r["sparsity"]) for r in rs})


def cell_table(rs):
    """target -> sparsity -> list of z(removal), in file order (v9_regen.spc)."""
    c = collections.OrderedDict()
    for r in rs:
        c.setdefault(r["target"], collections.OrderedDict()) \
            .setdefault(float(r["sparsity"]), []).append(z(r["removal"]))
    return c


# ------------------------------------------------------------ inventories ---

def fail_inventory(rs, gate_label):
    """Every (target, seed, sparsity) row that fails the collateral budget."""
    out = []
    for r in rs:
        if is_fail(r):
            out.append(dict(target=r["target"], axis=r["axis"], seed=r["seed"],
                            sparsity=float(r["sparsity"]),
                            retained=1.0 - float(r["sparsity"]),
                            designer=r.get("designer"),
                            selected_alpha=r.get("v9_selected_alpha"),
                            scored_as=z(r.get("removal")), gate=gate_label))
    return sorted(out, key=lambda d: (d["target"], d["seed"], d["sparsity"]))


def encoding_evidence(rs):
    """Prove which mechanism encodes a budget-fail, rather than assuming one."""
    n = len(rs)
    nan_rows = [r for r in rs if is_fail(r)]
    has_alpha_field = any("v9_selected_alpha" in r for r in rs)
    has_flag = [r for r in rs if "collateral_ok" in r]
    ident = {(r["target"], r["axis"], r["seed"], float(r["sparsity"])) for r in rs}
    null_alpha = ([r for r in rs if r.get("v9_selected_alpha") is None]
                  if has_alpha_field else None)
    return dict(
        n_rows=n,
        n_distinct_identities=len(ident),
        rows_missing=len(ident) - n,
        n_nan_removal=len(nan_rows),
        has_v9_selected_alpha_field=has_alpha_field,
        n_null_selected_alpha=(None if null_alpha is None else len(null_alpha)),
        # only meaningful where the field exists (i.e. on the v9 panel)
        nan_iff_null_alpha=(None if null_alpha is None else (
            {(r["target"], r["seed"], float(r["sparsity"])) for r in nan_rows} ==
            {(r["target"], r["seed"], float(r["sparsity"])) for r in null_alpha})),
        n_rows_carrying_collateral_ok=len(has_flag),
        fields=sorted(rs[0].keys()) if rs else [],
    )


def gate_recheck(tier, v9_fails):
    """Independently re-derive the fail set from the raw alpha traces.

    Re-applies `v9_gate.collateral_ok_row` to every probed alpha of every cell
    in results/<panel>/<tier>/alpha_trace.jsonl -- the same computation
    v9_rescore did -- so the artifact's NaNs are corroborated rather than taken
    on faith.  Also yields the per-alpha evidence behind the dense-8B claim.
    """
    fp = os.path.join(RES, PANEL, tier, "alpha_trace.jsonl")
    if not os.path.exists(fp):
        return dict(available=False, reason=f"no alpha_trace.jsonl at {fp}")
    import json as _json
    tr = []
    for line in open(fp):
        line = line.strip()
        if line:
            try:
                tr.append(_json.loads(line))
            except ValueError:
                continue
    by = collections.OrderedDict()
    for r in tr:
        by.setdefault((r["target"], r["seed"], float(r["sparsity"])), []).append(r)
    recomputed_fail, recomputed_pass = set(), {}
    for k, probes in by.items():
        best, best_a = None, None
        for p in probes:
            ok, _mode = v9_gate.collateral_ok_row(p)
            br = p.get("bias_reduction")
            if ok and br is not None and (best is None or br > best):
                best, best_a = float(br), p.get("alpha")
        if best is None:
            recomputed_fail.add(k)
        else:
            recomputed_pass[k] = dict(best_bias_reduction=best, alpha=best_a)
    artifact_fail = {(f["target"], f["seed"], f["sparsity"]) for f in v9_fails}
    return dict(
        available=True,
        trace_rows=len(tr),
        cells=len(by),
        recomputed_fail=sorted(f"{t}|seed{s}|s={sp}" for t, s, sp in recomputed_fail),
        agrees_with_artifact=(recomputed_fail == artifact_fail),
        only_in_artifact=sorted(f"{t}|seed{s}|s={sp}"
                                for t, s, sp in artifact_fail - recomputed_fail),
        only_in_recheck=sorted(f"{t}|seed{s}|s={sp}"
                               for t, s, sp in recomputed_fail - artifact_fail),
        passing_cells={f"{t}|seed{s}|s={sp}": v
                       for (t, s, sp), v in sorted(recomputed_pass.items())},
    )


def patch_sizes(rs):
    """(target, sparsity) -> patch bytes / MiB. Bytes differ in the last digits
    across seeds (the packed sign block is per-seed), so mean/min/max are all
    reported; the manuscript's §5.5 table quotes the mean to 2 s.f."""
    acc = collections.OrderedDict()
    for r in rs:
        b = r.get("bytes")
        if b is None:
            continue
        acc.setdefault(r["target"], collections.OrderedDict()) \
            .setdefault(float(r["sparsity"]), []).append(float(b))
    out = {}
    for t, d in acc.items():
        out[t] = {}
        for sp, vals in sorted(d.items()):
            out[t][f"s={sp}"] = dict(
                sparsity=sp, retained=1.0 - sp, n=len(vals),
                bytes_mean=float(np.mean(vals)),
                bytes_min=float(min(vals)), bytes_max=float(max(vals)),
                mib_mean=mib(float(np.mean(vals))),
                mib_min=mib(float(min(vals))), mib_max=mib(float(max(vals))))
    return out


# -------------------------------------------------------------- the curve ---

def curve(root, tier):
    """Every sparsity level present, delta-vs-dense with a paired cell bootstrap.

    Identical cell/pairing/bootstrap semantics to `v9_regen.spc`, widened from
    four levels to all of them.
    """
    rs = rows(root, f"{PANEL}/{tier}")
    if not rs:
        return None
    order = target_order(rs)
    c = cell_table(rs)
    lv = levels_present(rs)
    dense = min(lv)                      # the reference level, read from the data
    means = {t: {sp: float(np.mean(v)) for sp, v in d.items()} for t, d in c.items()}
    nfail = collections.Counter()
    nrow = collections.Counter()
    for r in rs:
        nrow[(r["target"], float(r["sparsity"]))] += 1
        if is_fail(r):
            nfail[(r["target"], float(r["sparsity"]))] += 1

    out_levels = []
    for sp in lv:
        pairs = [(means[t][sp], means[t][dense]) for t in order
                 if sp in means.get(t, {}) and dense in means.get(t, {})]
        d = None if sp == dense else boot_paired(pairs)
        out_levels.append(dict(
            sparsity=sp,
            retained=1.0 - sp,
            is_dense_reference=(sp == dense),
            n_cells=len(pairs),
            cells={t: dict(mean=means[t][sp],
                           n_rows=nrow[(t, sp)],
                           n_budget_fail=nfail[(t, sp)],
                           rows=[dict(seed=r["seed"], removal_raw=(
                               None if is_fail(r) else float(r["removal"])),
                               scored=z(r.get("removal")),
                               budget_fail=is_fail(r),
                               selected_alpha=r.get("v9_selected_alpha"))
                               for r in sorted(rs, key=lambda x: x["seed"])
                               if r["target"] == t and float(r["sparsity"]) == sp])
                   for t in order if sp in means.get(t, {})},
            delta_vs_dense=d,
            delta_vs_dense_fmt=fmt(d),
            ci_status=status(d),
        ))
    return dict(
        panel=os.path.relpath(os.path.join(root, PANEL, tier), HERE),
        n_rows=len(rs),
        targets=order,
        axes=sorted({r["axis"] for r in rs}),
        seeds=sorted({r["seed"] for r in rs}),
        designers=sorted({r.get("designer") for r in rs}),
        sparsity_levels=lv,
        dense_reference=dense,
        levels=out_levels,
        patch_sizes=patch_sizes(rs),
    )


# ------------------------------------------------------- published crosscheck

def published_spc(tier):
    """The citable v9 strings, read (not typed) from the v9 diff report."""
    rep = jload(os.path.join("results", "v9", "v9_diff_report.json"), default={})
    pre = f"SPC {tier} :: "
    return {k[len(pre):]: v for k, v in rep.get("diff", {}).items()
            if k.startswith(pre)}


def crosscheck(tier, cur):
    pub = published_spc(tier)
    out = {}
    got = {f"s={l['sparsity']}": l for l in cur["levels"]}
    for key, entry in sorted(pub.items()):
        mine = got.get(key)
        computed = mine["delta_vs_dense_fmt"] if mine else "MISSING LEVEL"
        out[key] = dict(published_v9=entry["v9"], computed=computed,
                        matches=(computed == entry["v9"]),
                        published_point=entry.get("v9_point"),
                        computed_point=(None if not mine or not mine["delta_vs_dense"]
                                        else round(mine["delta_vs_dense"]["point"], 6)),
                        source="results/v9/v9_diff_report.json :: "
                               f"SPC {tier} :: {key}")
    out["_all_match"] = all(v["matches"] for k, v in out.items()
                            if not k.startswith("_"))
    out["_n_published_levels"] = len([k for k in out if not k.startswith("_")])
    out["_n_levels_computed"] = len(cur["levels"])
    return out


# ------------------------------------------------------------ draft checks ---

def deployability(tier, v9_fails, asrun_fails, cur):
    """The two §5.5 sentences the draft makes about budget failures, resolved
    against the artifacts under BOTH gates so the difference is visible."""
    lv = cur["sparsity_levels"]
    dense = cur["dense_reference"]

    def envelope(fails):
        if not fails:
            return dict(any=False, max_sparsity=None, min_sparsity=None,
                        clean_from=None, targets=[], seeds=[], n=0)
        sps = sorted({f["sparsity"] for f in fails})
        above = [s for s in lv if s > max(sps)]
        return dict(any=True, n=len(fails), max_sparsity=max(sps),
                    min_sparsity=min(sps),
                    fail_levels=sps,
                    clean_from=(min(above) if above else None),
                    targets=sorted({f["target"] for f in fails}),
                    seeds=sorted({f["seed"] for f in fails}),
                    rows=[f"{f['target']}|seed{f['seed']}|s={f['sparsity']}"
                          for f in fails])

    def dense_split(fails):
        d = collections.OrderedDict()
        for lvl in cur["levels"]:
            if lvl["sparsity"] != dense:
                continue
            for t, cell in lvl["cells"].items():
                failed = {f["seed"] for f in fails
                          if f["target"] == t and f["sparsity"] == dense}
                d[t] = dict(
                    n_seeds=cell["n_rows"],
                    n_seeds_failing=len(failed),
                    n_seeds_passing=cell["n_rows"] - len(failed),
                    seeds_failing=sorted(failed),
                    seeds_passing=sorted(r["seed"] for r in cell["rows"]
                                         if r["seed"] not in failed),
                    passing_removals=[r["removal_raw"] for r in cell["rows"]
                                      if r["seed"] not in failed],
                    deployable_in_some_seed=(cell["n_rows"] - len(failed)) > 0)
        return d

    return dict(tier=tier, dense_reference=dense, sparsity_levels=lv,
                v9=dict(envelope=envelope(v9_fails), dense=dense_split(v9_fails)),
                as_run=dict(envelope=envelope(asrun_fails),
                            dense=dense_split(asrun_fails)))


# -------------------------------------------------------------------- main ---

def main():
    ensure_dirs()
    art = dict(artifact="v10_spc_v9",
               produced_by=os.path.relpath(os.path.abspath(__file__), HERE),
               figure="Fig 3 (sparsity curves, both tiers)",
               tier_definition=dict(small="<=3.8B targets", big="7-9B targets"),
               gate="v9 (corrected integer-item collateral gate, src/v9_gate.py)",
               conventions=dict(
                   cell="one TARGET (this panel has a single axis, occ_gender)",
                   cell_value="MEAN over the cell's seed rows",
                   budget_fail="scored 0.0 (v10_common.z)",
                   bootstrap="paired, resamples CELLS, 10k, seed 0",
                   patch_size="MiB = bytes / 2**20 (manuscript prints 'MB')"),
               tiers={}, crosscheck={}, budget_fail_encoding={},
               deployability={}, blocked=[])

    for tier in TIERS:
        v9rows = rows(V9, f"{PANEL}/{tier}")
        asrows = rows(RES, f"{PANEL}/{tier}")
        if not v9rows:
            art["blocked"].append(
                f"results_v9/{PANEL}/{tier}/removal.jsonl absent — tier BLOCKED")
            continue
        cur = curve(V9, tier)
        art["tiers"][tier] = cur
        art["crosscheck"][tier] = crosscheck(tier, cur)

        v9_fails = fail_inventory(v9rows, "v9")
        as_fails = fail_inventory(asrows, "as-run") if asrows else []
        v9set = {(f["target"], f["seed"], f["sparsity"]) for f in v9_fails}
        asset = {(f["target"], f["seed"], f["sparsity"]) for f in as_fails}
        art["budget_fail_encoding"][tier] = dict(
            mechanism="NaN `removal` + null `v9_selected_alpha` on a row that IS "
                      "present; NOT a collateral_ok flag (absent from these rows) "
                      "and NOT a missing row",
            why="v9_rescore.rescore_trace_panel re-picks the in-budget argmax over "
                "alpha_trace; a cell with no in-budget alpha keeps its row and gets "
                "removal=NaN, v9_selected_alpha=None",
            v9_rows=encoding_evidence(v9rows),
            as_run_rows=(encoding_evidence(asrows) if asrows else
                         dict(note="as-run panel absent")),
            v9_fail_inventory=v9_fails,
            as_run_fail_inventory=as_fails,
            n_fail_v9=len(v9_fails), n_fail_as_run=len(as_fails),
            recovered_by_v9_gate=sorted(f"{t}|seed{s}|s={sp}"
                                        for t, s, sp in asset - v9set),
            newly_failing_under_v9=sorted(f"{t}|seed{s}|s={sp}"
                                          for t, s, sp in v9set - asset),
            as_run_note="as-run shown ONLY for the recovered/newly-failing contrast; "
                        "every quoted value in this artifact is v9",
            independent_gate_recheck=gate_recheck(tier, v9_fails),
        )
        art["deployability"][tier] = deployability(tier, v9_fails, as_fails, cur)

    fp = jdump(art, OUT_REL)

    # ------------------------------------------------------------- console --
    print("=" * 78)
    print("v10 SPC — full v9 sparsity curves (Fig 3 source)")
    print("=" * 78)
    for tier, cur in art["tiers"].items():
        print(f"\n--- tier {tier}  ({cur['n_rows']} rows, targets "
              f"{', '.join(cur['targets'])}, seeds {cur['seeds']}) ---")
        for lvl in cur["levels"]:
            tag = "  [dense reference]" if lvl["is_dense_reference"] else ""
            print(f"  s={lvl['sparsity']:<6} retained={lvl['retained']:<6.4g} "
                  f"n_cells={lvl['n_cells']}  {lvl['delta_vs_dense_fmt']}{tag}")
        cc = art["crosscheck"][tier]
        print(f"  crosscheck vs v9_diff_report: "
              f"{'ALL MATCH' if cc['_all_match'] else 'MISMATCH'} "
              f"({cc['_n_published_levels']} published of "
              f"{cc['_n_levels_computed']} computed)")
        for k, v in sorted(cc.items()):
            if k.startswith("_") or v["matches"]:
                continue
            print(f"    MISMATCH {k}: published {v['published_v9']} "
                  f"!= computed {v['computed']}")
        enc = art["budget_fail_encoding"][tier]
        print(f"  budget-fails: v9 {enc['n_fail_v9']}  as-run {enc['n_fail_as_run']}"
              f"  recovered by v9 gate: {enc['recovered_by_v9_gate'] or 'none'}")
        for f in enc["v9_fail_inventory"]:
            print(f"    v9 FAIL  {f['target']}|seed{f['seed']}|s={f['sparsity']}")
        rc = enc["independent_gate_recheck"]
        if rc.get("available"):
            print(f"  independent re-gate of alpha_trace agrees: "
                  f"{rc['agrees_with_artifact']} ({rc['cells']} cells)")
        dep = art["deployability"][tier]
        for gate in ("v9", "as_run"):
            env = dep[gate]["envelope"]
            print(f"  [{gate}] fail envelope: "
                  f"{'none' if not env['any'] else env['fail_levels']}"
                  f"  clean from s={env['clean_from']}")
            for t, d in dep[gate]["dense"].items():
                print(f"    [{gate}] dense {t}: {d['n_seeds_passing']}/"
                      f"{d['n_seeds']} seeds pass the budget "
                      f"(passing seeds {d['seeds_passing']})")
    print(f"\nwrote {fp}")
    return art


if __name__ == "__main__":
    main()
