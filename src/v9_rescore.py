"""WS1.2 — re-score every panel under the corrected gate. Analysis-only, no GPU.

Three re-scoring paths, because the program's panels store collateral three ways:

  A. alpha_trace.jsonl        27 panels. Re-apply the gate to every probed alpha
                              and re-pick the in-budget argmax. This is the real
                              re-selection: a cell whose best alpha was rejected
                              at the boundary can now select a *different* alpha.
  B. per-config removal rows   5 panels (steer, sentdebias, inlp, and the two
                              quarantined steer_*). Each row IS one config and
                              carries dmmlu/ppl_ratio, so the gate is re-applied
                              row-wise and downstream best-in-budget selection
                              picks up the change.
  C. results/runs.jsonl        the v2-era sweep behind C2. Carries nested
                              pre/post dicts, so it re-scores like (A).

  D. NOT RE-SCORABLE           5 panels: t2x, v1, x, x2 (v5-era) and ste. They
                              predate the v6 alpha-trace fix and store only a
                              scalar removal, so there is no record of which
                              alphas were probed or what collateral they cost.
                              Their boundary exposure is UNKNOWN and is reported
                              as such, never assumed to be zero.

Outputs mirror the original tree under results_v9/ so nothing is overwritten.
Each v9 removal row is the ORIGINAL row with `removal` replaced, plus
v9_selected_alpha / v9_changed / v9_gate_mode, so it is a drop-in replacement for
every downstream analysis.
"""
import os, sys, json, glob, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v9_gate import (collateral_ok_row, N_ITEMS_DEFAULT, mmlu_ok,
                     mmlu_ok_delta, NonIntegralAccuracy, PPL_BUDGET, PPL_TOL)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
OUT_ROOT = os.path.join(HERE, "results_v9")

# fields that describe the probe/outcome rather than the cell identity
OUTCOME = {"alpha", "dmmlu", "ppl_ratio", "collateral_ok", "bias_reduction",
           "pre_mmlu", "post_mmlu", "pre_ppl", "post_ppl", "pre_skew",
           "post_skew", "removal"}

NOT_RESCORABLE = ["t2x", "v1", "x", "x2", "v6trace/ste"]


def _ident_keys(row):
    return {k for k, v in row.items()
            if k not in OUTCOME and not isinstance(v, (dict, list))}


def _ident(row, keys=None):
    """Identity tuple. `keys` restricts to a per-panel agreed key set, because
    trace rows and removal rows do not always carry the same fields (3b/small's
    trace adds size_b/source; v8ins/A's removal adds ifeval/panel). Joining on
    the intersection is what makes the two files line up."""
    it = row.items() if keys is None else ((k, row.get(k)) for k in keys)
    return tuple(sorted((k, v) for k, v in it
                        if (keys is not None or k not in OUTCOME)
                        and not isinstance(v, (dict, list))))


def rescore_trace_panel(panel_dir):
    """Path A. -> (v9 rows, stats)"""
    tr_fp = os.path.join(panel_dir, "alpha_trace.jsonl")
    rm_fp = os.path.join(panel_dir, "removal.jsonl")
    traw = []
    for l in open(tr_fp):
        l = l.strip()
        if not l:
            continue
        try:
            traw.append(json.loads(l))
        except Exception:
            continue
    rraw = [json.loads(l) for l in open(rm_fp) if l.strip()]
    if not traw or not rraw:
        return [], collections.Counter(empty=1)
    # join on the intersection of identity keys present in BOTH files
    jkeys = sorted(_ident_keys(traw[0]) & _ident_keys(rraw[0]))
    trace = {}
    for r in traw:
        # append-only + resumable: (identity, alpha) is not unique; keep LAST
        trace[(_ident(r, jkeys), r.get("alpha"))] = r
    by_cell = collections.defaultdict(list)
    for (ident, _a), r in trace.items():
        by_cell[ident].append(r)

    stats = collections.Counter()
    modes = collections.Counter()
    sel = {}
    for ident, rows in by_cell.items():
        best_old = best_new = float("nan")
        a_old = a_new = None
        for r in rows:
            ok_new, mode = collateral_ok_row(r, N_ITEMS_DEFAULT)
            modes[mode] += 1
            if ok_new is None:
                continue
            br = r.get("bias_reduction")
            if br is None:
                continue
            if r.get("collateral_ok"):
                if best_old != best_old or br > best_old:
                    best_old, a_old = br, r.get("alpha")
            if ok_new:
                if best_new != best_new or br > best_new:
                    best_new, a_new = br, r.get("alpha")
            if ok_new and not r.get("collateral_ok"):
                stats["probes_recovered"] += 1
        sel[ident] = dict(old=best_old, new=best_new, a_old=a_old, a_new=a_new)
        if (best_old != best_old) != (best_new != best_new) or \
           (best_old == best_old and best_new == best_new and
                abs(best_old - best_new) > 1e-12):
            stats["cells_changed"] += 1
        stats["cells"] += 1

    # attach to the original removal rows by the agreed join key
    v9_rows, unmatched = [], 0
    stats["join_keys"] = jkeys
    for row in rraw:
        s = sel.get(_ident(row, jkeys))
        if s is None:
            unmatched += 1
            v9_rows.append(dict(row, v9_selected_alpha=None, v9_changed=None,
                                v9_gate_mode="UNMATCHED"))
            continue
        new = s["new"]
        v9_rows.append(dict(row, removal=(float("nan") if new != new else new),
                            v9_selected_alpha=s["a_new"],
                            v9_changed=bool((s["old"] != s["old"]) != (new != new) or
                                            (s["old"] == s["old"] and new == new and
                                             abs(s["old"] - new) > 1e-12)),
                            v9_gate_mode="integer"))
    stats["rows_unmatched"] = unmatched
    stats["gate_modes"] = dict(modes)
    return v9_rows, stats


def rescore_row_panel(panel_dir):
    """Path B. Each removal row is one config carrying its own collateral."""
    rm_fp = os.path.join(panel_dir, "removal.jsonl")
    out, stats = [], collections.Counter()
    for l in open(rm_fp):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        d, ratio = r.get("dmmlu"), r.get("ppl_ratio")
        if d is None or ratio is None:
            out.append(dict(r, v9_gate_mode="UNGATED"))
            stats["ungated"] += 1
            continue
        # integer gate on the stored difference (all dmmlu are on the 1/200 grid)
        try:
            ok = mmlu_ok_delta(d) and (ratio <= PPL_BUDGET + PPL_TOL)
            mode = "integer-delta"
        except NonIntegralAccuracy:
            ok = (d <= 0.02 + 1e-9) and (ratio <= PPL_BUDGET + PPL_TOL)
            mode = "float-fallback (dmmlu off-grid)"
        if ok and not r.get("collateral_ok"):
            stats["rows_recovered"] += 1
        stats[f"mode:{mode}"] += 1
        out.append(dict(r, collateral_ok=bool(ok), v9_gate_mode=mode))
        stats["rows"] += 1
    return out, stats


def rescore_runs_jsonl():
    """Path C. results/runs.jsonl, nested pre/post."""
    fp = os.path.join(RESULTS, "runs.jsonl")
    out, stats = [], collections.Counter()
    for l in open(fp):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        p, q = r.get("pre"), r.get("post")
        if not (isinstance(p, dict) and isinstance(q, dict)
                and "mmlu_acc" in p and "mmlu_acc" in q and p.get("ppl")):
            stats["ungated"] += 1
            continue
        ok = mmlu_ok(p["mmlu_acc"], q["mmlu_acc"]) and \
            (q["ppl"] / p["ppl"] <= PPL_BUDGET + PPL_TOL)
        if ok and not r.get("collateral_ok"):
            stats["rows_recovered"] += 1
        stats["rows"] += 1
        out.append(dict(r, collateral_ok=bool(ok), v9_gate_mode="integer"))
    return out, stats


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    report = dict(artifact="v9_rescore", panels={}, not_rescorable={})

    panels = sorted({os.path.dirname(p) for p in
                     glob.glob(os.path.join(RESULTS, "**", "removal.jsonl"),
                               recursive=True)})
    tot_recovered = tot_changed = 0
    for d in panels:
        rel = os.path.relpath(d, RESULTS)
        if rel in NOT_RESCORABLE:
            n = sum(1 for _ in open(os.path.join(d, "removal.jsonl")))
            report["not_rescorable"][rel] = dict(
                rows=n, reason="no alpha_trace and no per-config collateral; "
                               "predates the v6 trace fix. Boundary exposure UNKNOWN.")
            continue
        # Route by the STRUCTURE of removal.jsonl, not by the mere presence of a
        # trace: v6trace/dpo has a trace but its removal rows are per-config
        # (they carry `alpha`), so re-selecting an argmax over them would be
        # wrong -- downstream analysis does that selection itself.
        first = None
        with open(os.path.join(d, "removal.jsonl")) as fh:
            for ln in fh:
                if ln.strip():
                    first = json.loads(ln); break
        per_config = bool(first and "alpha" in first)
        has_trace = os.path.exists(os.path.join(d, "alpha_trace.jsonl")) and not per_config
        try:
            if has_trace:
                rows, st = rescore_trace_panel(d)
            else:
                rows, st = rescore_row_panel(d)
        except Exception as e:
            report["panels"][rel] = dict(error=f"{type(e).__name__}: {e}")
            print(f"[v9] FAIL {rel}: {type(e).__name__}: {e}", flush=True)
            continue
        od = os.path.join(OUT_ROOT, rel)
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "removal.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        report["panels"][rel] = dict(path="A-trace" if has_trace else "B-rows",
                                     n_rows=len(rows), **{k: v for k, v in st.items()})
        tot_recovered += st.get("probes_recovered", 0) + st.get("rows_recovered", 0)
        tot_changed += st.get("cells_changed", 0)
        print(f"[v9] {rel:42s} {'trace' if has_trace else 'rows ':>5s} "
              f"n={len(rows):5d} recovered={st.get('probes_recovered', 0) + st.get('rows_recovered', 0):4d} "
              f"cells_changed={st.get('cells_changed', 0):4d}", flush=True)

    rows, st = rescore_runs_jsonl()
    with open(os.path.join(OUT_ROOT, "runs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report["runs_jsonl"] = dict(st)
    tot_recovered += st.get("rows_recovered", 0)
    print(f"[v9] {'runs.jsonl':42s} {'runs':>5s} n={len(rows):5d} "
          f"recovered={st.get('rows_recovered', 0):4d}")

    report["totals"] = dict(probes_or_rows_recovered=tot_recovered,
                            cells_changed=tot_changed,
                            panels_rescored=len(report["panels"]),
                            panels_not_rescorable=len(report["not_rescorable"]))
    fp = os.path.join(OUT_ROOT, "rescore_report.json")
    json.dump(report, open(fp, "w"), indent=1)
    print(f"\n[v9] recovered {tot_recovered} probes/rows; {tot_changed} cells changed value")
    print(f"[v9] NOT re-scorable: {list(report['not_rescorable'])}")
    print(f"[v9] wrote {fp}")


if __name__ == "__main__":
    main()
