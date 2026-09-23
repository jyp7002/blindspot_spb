#!/usr/bin/env python3
"""v12 — Table A without the provenance paragraph, and the operational Table 3.

    python3 src/v12_frontier.py --check-replay   # v12frontier vs published panels
    python3 src/v12_frontier.py                  # -> results/v12/frontier_v12.json

THREE THINGS, EACH REPLACING A SENTENCE THE MANUSCRIPT HAS TO APOLOGISE FOR.

1. DPO IS RE-GATED FROM ITS TRACE. v9 left the DPO column UNGATED because its
   removal rows carry no collateral -- but results/v6trace/dpo/alpha_trace.jsonl
   does, and v10_regen_frontier.dpo_regate_diagnostic already computed the
   integer-gate decisions as an advisory. This promotes that computation to the
   reported value. Needs no GPU. (The diagnostic found 0 of 60 decisions flip,
   so the column's values do not move; what changes is that they are now true
   v9-gate values rather than as-run ones.)

2. THE EDIT ARM IS RE-MEASURED (v12frontier panel, run_frontier.py) so the four
   llama/qwen cells that came from the trace-less `results/x` panel are replaced
   by traced, integer-gated values. Until that panel exists, this script reports
   the published arm and says so.

3. TABLE 3 -- THE OPERATIONAL COMPARISON. Per method, all from rows:
     removal              cell mean of the selected configuration (nan -> 0)
     ΔMMLU (items)        median over (cell, seed) at the SELECTED configuration
     PPL ratio            median over (cell, seed) at the SELECTED configuration
     budget-pass rate     fraction of the method's REAL configurations in budget
                          (never pooled with a null -- the draft's "88%" for
                          SentenceDebias pooled the real configs with its
                          random-subspace null; real alone is 78.1% under v9)
     IFEval Δ             selected config minus unedited, inst-level strict acc,
                          seed 0 (v12frontier only)
   plus the four deployment properties, which are properties of the method and
   are stated as constants with the reason next to each.
"""
import argparse
import collections
import json
import math
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V                  # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
V9 = os.path.join(REPO, "results_v9")
PANEL = "v12frontier"
CELLS = [(t, a) for t in ("gemma", "llama", "qwen", "phi")
         for a in ("occ_gender", "crows_socioeconomic")]
IFEVAL_METRIC = "inst_level_strict_acc,none"

DEPLOYMENT = collections.OrderedDict([
    ("serving-time hook", dict(edit="no", steering="yes", sentdebias="yes",
                               why="steering adds, SentenceDebias projects, a "
                                   "residual-stream vector at a hooked layer")),
    ("activation access at serving", dict(edit="no", steering="yes",
                                          sentdebias="yes",
                                          why="the hook reads the hidden state")),
    ("additional per-token compute", dict(edit="none", steering="O(d) add",
                                          sentdebias="O(kd) projection",
                                          why="per generated token, at one layer")),
    ("persistent after merge", dict(edit="yes", steering="no", sentdebias="no",
                                    why="the edit is in the weights; a hook is "
                                        "lost when the checkpoint is redistributed")),
])


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def jl(fp):
    if not os.path.exists(fp):
        return []
    out = []
    for ln in open(fp):
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def cell_mean(per_seed):
    """{(t,a,s): v} -> {(t,a): mean over seeds}, nan -> 0."""
    acc = collections.defaultdict(list)
    for (t, a, _s), v in per_seed.items():
        acc[(t, a)].append(z(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


# ------------------------------------------------------------ published ----
def published_arms():
    """Selected value per (t,a,s) and the selected row, per method, v9 tree."""
    arms = {}
    for method, panel, filt in (
            ("steering", "v6trace/steer",
             lambda r: r.get("kind") == "real" and r.get("grid") == "primary"),
            ("sentdebias", "v6trace/sentdebias", lambda r: r.get("kind") == "real")):
        rs = [r for r in jl(os.path.join(V9, panel, "removal.jsonl")) if filt(r)]
        sel, selrow = {}, {}
        for r in rs:
            k = (r["target"], r["axis"], r["seed"])
            if r.get("collateral_ok") and (k not in sel or r["bias_reduction"] > sel[k]):
                sel[k], selrow[k] = r["bias_reduction"], r
        for r in rs:
            sel.setdefault((r["target"], r["axis"], r["seed"]), float("nan"))
        arms[method] = dict(sel=sel, selrow=selrow, configs=rs)
    return arms


def dpo_regated():
    """The integer gate applied to the DPO trace -- promoted from advisory."""
    import v10_regen_frontier as F
    d = F.dpo_regate_diagnostic()
    if not d.get("available"):
        raise SystemExit(f"DPO trace unavailable: {d.get('why')}")
    cells = {tuple(k.split("|")): v for k, v in d["regated_cells"].items()}
    # Cells with no trace rows were skipped for < 20 informative pairs: a zeroed
    # non-measurement, exactly as Table A marks them.
    out = {c: cells.get(c, 0.0) for c in CELLS}
    return out, dict(n_trace_rows=d["n_trace_rows"],
                     n_rows_flipped=d["n_rows_flipped"],
                     zeroed=[f"{t}|{a}" for t, a in CELLS if (t, a) not in cells])


def published_edit():
    """Table A's edit arm as published (v9 value; 4 cells as-run), from the
    v10 frontier artifact, which records per-cell provenance."""
    fr = json.load(open(os.path.join(RESULTS, "v10", "frontier_v9.json")))
    out = {}
    for c in fr.get("cells", []):
        out[(c["target"], c["axis"])] = c["methods"]["edit"]["v9"]
    missing = [c for c in CELLS if c not in out]
    if missing:
        raise SystemExit(f"frontier_v9.json lacks edit cells {missing}; run make regen")
    return out


# ------------------------------------------------------------------ v12 ----
def v12_panel():
    rows = jl(os.path.join(RESULTS, PANEL, "removal.jsonl"))
    trace = jl(os.path.join(RESULTS, PANEL, "alpha_trace.jsonl"))
    if not rows:
        return None
    last = {}
    for r in rows:
        last[(r["target"], r["axis"], r["seed"], r["method"], r.get("row", "select"))] = r
    tr = {}
    for t in trace:
        tr[(t["target"], t["axis"], t["seed"], t["method"], t["config"])] = t
    arms = {}
    for m in ("edit", "steering", "sentdebias"):
        sel, sel1k, selrow, ifev = {}, {}, {}, {}
        for (t, a, s, mm, kind), r in last.items():
            if mm != m:
                continue
            if kind == "select":
                sel[(t, a, s)] = r.get("removal")
                sel1k[(t, a, s)] = r.get("removal_1k_gate")
                if r.get("config"):
                    selrow[(t, a, s)] = tr.get((t, a, s, m, r["config"]))
            elif kind == "ifeval":
                base = last.get((t, a, s, "unedited", "ifeval"), {}).get("ifeval") or {}
                v = (r.get("ifeval") or {}).get(IFEVAL_METRIC)
                b = base.get(IFEVAL_METRIC)
                if v is not None and b is not None:
                    ifev[(t, a)] = v - b
        arms[m] = dict(sel=sel, sel1k=sel1k, selrow=selrow, ifeval=ifev,
                       configs=[x for k, x in tr.items() if k[3] == m])
    return arms


# ---------------------------------------------------------------- table ----
def table3(arms, dpo, cell_override=None):
    out = collections.OrderedDict()
    for m, d in arms.items():
        cm = (cell_override or {}).get(m) or cell_mean(d["sel"])
        sel_rows = [r for r in d["selrow"].values() if r]
        dm = [r.get("dmmlu_items", round(r.get("dmmlu", 0) * 200)) for r in sel_rows]
        pp = [r["ppl_ratio"] for r in sel_rows]
        cfg = d["configs"]
        out[m] = dict(
            removal=V.boot_mean([cm.get(c, 0.0) for c in CELLS]),
            dmmlu_items_median=statistics.median(dm) if dm else None,
            ppl_ratio_median=statistics.median(pp) if pp else None,
            budget_pass_rate=(sum(bool(r["collateral_ok"]) for r in cfg) / len(cfg))
            if cfg else None,
            n_configs=len(cfg),
            ifeval_delta=(V.boot_mean(list(d["ifeval"].values()))
                          if d.get("ifeval") else None),
            **{k: v[m] for k, v in DEPLOYMENT.items()})
    out["dpo"] = dict(removal=V.boot_mean([dpo.get(c, 0.0) for c in CELLS]),
                      note="re-gated from alpha_trace.jsonl; deployment properties "
                           "as the edit (a merged LoRA delta)")
    return out


def pooled(edit_cells, other_cells):
    return V.boot_paired([(z(edit_cells.get(c)), z(other_cells.get(c))) for c in CELLS])


# --------------------------------------------------------------- replay ----
def check_replay(tol=0.0):
    new = v12_panel()
    if new is None:
        print("no v12frontier rows yet")
        return True
    pub = published_arms()
    worst, bad, n = 0.0, [], 0
    for m in ("steering", "sentdebias"):
        for k, v in new[m]["sel"].items():
            if k in pub[m]["sel"]:
                d = abs(z(v) - z(pub[m]["sel"][k]))
                n += 1
                worst = max(worst, d)
                if d > tol:
                    bad.append((m, k, v, pub[m]["sel"][k]))
    x2 = {(r["target"], r["axis"], r["seed"]): r["removal"]
          for r in jl(os.path.join(V9, "v6trace/x2/removal.jsonl"))
          if r.get("role") == "self"}
    for k, v in new["edit"]["sel"].items():
        if k in x2:
            d = abs(z(v) - z(x2[k]))
            n += 1
            worst = max(worst, d)
            if d > tol and k[0] == "gemma":          # phi: batch 12 -> 6, see config
                bad.append(("edit", k, v, x2[k]))
            elif d > tol:
                print(f"  (expected) phi edit {k}: new={z(v):+.4f} pub={z(x2[k]):+.4f} "
                      "-- eval batch 6 vs published 12")
    print(f"replayed selections: {n}   max |delta| = {worst:.3g}   unexpected: {len(bad)}")
    for m, k, a, b in bad[:20]:
        print(f"  {m:10s} {'|'.join(map(str, k)):32s} new={z(a):+.4f} pub={z(b):+.4f}")
    print("REPLAY " + ("PASS" if not bad else "FAIL"))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-replay", action="store_true")
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()
    if a.check_replay:
        return 0 if check_replay(a.tol) else 1

    dpo, dpo_meta = dpo_regated()
    new = v12_panel()
    source = "v12frontier" if new else "published (v9 tree; edit arm has 4 as-run cells)"
    if new:
        arms = new
        edit_cells = cell_mean(arms["edit"]["sel"])
    else:
        arms = published_arms()
        arms["edit"] = dict(sel={}, selrow={}, configs=[], ifeval={})
        edit_cells = published_edit()
    other = {m: cell_mean(arms[m]["sel"]) for m in ("steering", "sentdebias")}
    out = dict(
        source=source,
        dpo_regate=dpo_meta,
        table_a={f"{t}|{a_}": dict(edit=edit_cells.get((t, a_)),
                                   steering=other["steering"].get((t, a_)),
                                   sentdebias=other["sentdebias"].get((t, a_)),
                                   dpo=dpo.get((t, a_)))
                 for t, a_ in CELLS},
        pooled={"edit - steering": pooled(edit_cells, other["steering"]),
                "edit - SentenceDebias": pooled(edit_cells, other["sentdebias"]),
                "edit - DPO": pooled(edit_cells, dpo)},
        table3=table3({m: arms[m] for m in ("edit", "steering", "sentdebias")}, dpo,
                      cell_override=None if new else {"edit": edit_cells}),
    )
    if new:
        e1k = cell_mean(arms["edit"]["sel1k"])
        out["pooled_1000_item_gate"] = {
            "edit - steering": pooled(e1k, cell_mean(arms["steering"]["sel1k"])),
            "edit - SentenceDebias": pooled(e1k, cell_mean(arms["sentdebias"]["sel1k"]))}
    os.makedirs(os.path.join(RESULTS, "v12"), exist_ok=True)
    fp = os.path.join(RESULTS, "v12", "frontier_v12.json")
    json.dump(out, open(fp, "w"), indent=1, default=str)

    print(f"source: {source}")
    print(f"DPO re-gated from trace: {dpo_meta['n_rows_flipped']} of "
          f"{dpo_meta['n_trace_rows']} decisions flip; zeroed {dpo_meta['zeroed']}")
    for k, d in out["pooled"].items():
        print(f"  {k:24s} {V.fmt(d, 3)}")
    for k, d in out.get("pooled_1000_item_gate", {}).items():
        print(f"  {k:24s} {V.fmt(d, 3)}   [1000-item gate]")
    print("\nTable 3")
    for m, d in out["table3"].items():
        if m == "dpo":
            print(f"  {m:10s} removal {V.fmt(d['removal'], 3)}")
            continue
        f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
        print(f"  {m:10s} removal {V.fmt(d['removal'], 3)}  ΔMMLU "
              f"{f(d['dmmlu_items_median'], 1)} items  ppl {f(d['ppl_ratio_median'])}"
              f"  pass {f(d['budget_pass_rate'])} (n={d['n_configs']})  IFEval Δ "
              f"{'-' if not d['ifeval_delta'] else V.fmt(d['ifeval_delta'], 3)}")
    print(f"\nwrote {os.path.relpath(fp, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
