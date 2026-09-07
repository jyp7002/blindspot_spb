"""v10 — regenerate the frontier (Fig 5 source + the manuscript's Table A).

WHY THIS EXISTS. `results/v9/v9_diff_report.json` stores only the four POOLED
frontier lines. The manuscript's Table A carries 32 per-cell numbers that exist
in no artifact at all — they were hand-typed from an as-run run and only
partially refreshed after the v9 re-score. The v10 plan (A5, checklist item 4)
requires them regenerated, with "steering qwen|occ is known to rise under v9"
visible in the diff log. This module is that regeneration.

EVERY convention is inherited, not re-invented:
  * `best_in_budget`, `edit_arm`, `to_cells` and the HEADLINE cell list are
    copied semantics-for-semantics from `src/v9_regen.py` (the script that
    produced the citable numbers). The pooled deltas this module emits are
    therefore the SAME numbers, recomputed, and are checked against both
    `results/v9/v9_diff_report.json` and `RESULTS_METHOD_v9.md` §3 at run time.
  * budget-fail / missing -> 0.0, bootstrap resamples the CELL at 10k seed 0,
    a cell is the MEAN over its seeds — all via `v10_common`.

WHAT THIS MODULE ADDS (none of it exists anywhere else):
  1. the 8 x 4 per-cell table under v9 AND its as-run counterpart, so the
     re-selection diff is auditable cell by cell;
  2. per-(cell, method) gate provenance as a boolean the figure can bind to
     marker fill — gemma/phi edit cells are true v9 (from the re-scorable twin
     `v6trace/x2`), llama/qwen edit cells are as-run (`results/x`, no trace,
     no twin);
  3. zeroed-cell marking: dots that are NOT measurements but 0.0 produced by
     the nan->0 rule, each with the reason read out of the panel rows;
  4. the DPO arm's honest gate status (v9 marks every DPO row UNGATED), plus a
     non-authoritative diagnostic showing whether re-gating would move it.

NOTHING here is hardcoded. Published values, the draft's Table A and the
§3 pooled lines are all PARSED at run time and compared.

Reads : results/, results_v9/, results/v9/v9_diff_report.json,
        RESULTS_METHOD_v9.md, binary_debiaser_draft_v3.md
Writes: results/v10/frontier_v9.json   (one artifact, nothing else)
"""
import os, sys, re, json, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10_common import (HERE, RES, V9, OUT_V10, ensure_dirs,
                        z, rows, jload, jdump, boot_paired, status, fmt)

DRAFT = os.path.join(HERE, "binary_debiaser_draft_v3.md")
METHOD_V9 = os.path.join(HERE, "RESULTS_METHOD_v9.md")
DIFF_V9 = os.path.join(HERE, "results", "v9", "v9_diff_report.json")
OUT = os.path.join(OUT_V10, "frontier_v9.json")

# The frontier's 8 headline cells — the list is verbatim from v9_regen.frontier.
HEADLINE = [(t, a) for t in ("gemma", "llama", "qwen", "phi")
            for a in ("occ_gender", "crows_socioeconomic")]

# Which arm each method's rows come from, and the row filter — verbatim from
# v9_regen.frontier. Kept as data so provenance and computation cannot drift.
ARMS = (
    ("steering",       "v6trace/steer",      lambda r: r.get("kind") == "real"
                                                       and r.get("grid") == "primary"),
    ("SentenceDebias", "v6trace/sentdebias", lambda r: r.get("kind") == "real"),
    ("DPO",            "v6trace/dpo",        lambda r: r.get("method") == "dpo"),
)
POOLED_NAMES = {"steering": "edit - steering",
                "SentenceDebias": "edit - SentenceDebias",
                "DPO": "edit - DPO"}
RESTRICTED_TARGETS = ("gemma", "phi")     # the re-scorable half of the edit arm


# ------------------------------------------------- v9_regen logic, verbatim --

def best_in_budget(rs, keyf, filt=lambda r: True):
    """Best in-budget score per key. Panels name the score `bias_reduction`
    (steer/sentdebias/inlp) or `removal` (dpo); accept either.
    VERBATIM from v9_regen.best_in_budget."""
    out = collections.defaultdict(lambda: float("nan"))
    for r in rs:
        if not filt(r) or not r.get("collateral_ok"):
            continue
        br = r.get("bias_reduction", r.get("removal"))
        if br is None:
            continue
        k = keyf(r)
        cur = out[k]
        out[k] = br if cur != cur else max(cur, br)
    return dict(out)


def edit_arm(root):
    """Edit self-removal per (target, axis, seed). VERBATIM from
    v9_regen.edit_arm — including the fact that `results/x` is read from the
    AS-RUN tree regardless of `root`, because it has no v9 twin."""
    out, provenance = {}, {}
    for r in rows(root, "v6trace/x2"):
        if r.get("role") == "self":
            out[(r["target"], r["axis"], r["seed"])] = z(r["removal"])
            provenance[r["target"]] = "v6trace/x2 (re-scorable)"
    for r in rows(RES, "x"):                       # as-run only; no v9 twin
        if r.get("role") == "self":
            out[(r["target"], r["axis"], r["seed"])] = z(r["removal"])
            provenance[r["target"]] = "results/x (NOT re-scorable — as-run)"
    return out, provenance


def to_cells(d):
    """(target,axis,seed) -> (target,axis), mean over seeds. VERBATIM from the
    inner `to_cells` of v9_regen.frontier. The project bootstraps the CELL,
    never the seed."""
    acc = collections.defaultdict(list)
    for (t, a, _s), v in d.items():
        acc[(t, a)].append(z(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ------------------------------------------------------------ provenance ----

def edit_cell_provenance(root, target):
    """(true_v9, source, gate_mode) for one target's EDIT cells."""
    if target in ("gemma", "phi"):
        rel = os.path.relpath(os.path.join(root, "v6trace", "x2"), HERE)
        modes = sorted({r.get("v9_gate_mode", "<as-run, ungated field>")
                        for r in rows(root, "v6trace/x2")
                        if r.get("target") == target and r.get("role") == "self"})
        return (root == V9), rel + " (role=self)", "/".join(modes) or "n/a"
    rel = os.path.relpath(os.path.join(RES, "x"), HERE)
    modes = sorted({r.get("v9_gate_mode", "<as-run, ungated field>")
                    for r in rows(RES, "x")
                    if r.get("target") == target and r.get("role") == "self"})
    return False, rel + " (role=self; NOT re-scorable — no trace, no twin)", \
        "/".join(modes) or "n/a"


def panel_gate_modes(root, panel):
    return sorted({r.get("v9_gate_mode", "<as-run, no gate field>")
                   for r in rows(root, panel)})


def zero_reason(root, panel, filt, target, axis):
    """Why a cell scored 0.0 by the nan->0 rule, read out of the panel rows."""
    cand = [r for r in rows(root, panel)
            if r.get("target") == target and r.get("axis") == axis and filt(r)]
    if not cand:
        return f"no rows for this cell in {panel}"
    skipped = sorted({r["skipped"] for r in cand if r.get("skipped")})
    if skipped and all(r.get("skipped") for r in cand):
        n_inf = sorted({r.get("n_informative") for r in cand})
        return (f"all {len(cand)} rows skipped ({','.join(skipped)}; "
                f"n_informative={n_inf}) — no configuration was ever run")
    n_ok = sum(1 for r in cand if r.get("collateral_ok"))
    if n_ok == 0:
        return (f"budget-fail: {len(cand)} candidate rows, 0 in-budget "
                f"(collateral_ok never true)")
    return f"{len(cand)} rows, {n_ok} in-budget — cell should not be zeroed"


# ------------------------------------------------------------- the table ----

def build(root):
    """Per-cell values for all four methods under one gate tree."""
    ed, edit_prov = edit_arm(root)
    edc = to_cells(ed)
    per = {"edit": edc}
    seeds = {"edit": ed}
    for name, panel, filt in ARMS:
        d = best_in_budget(rows(root, panel),
                           lambda r: (r["target"], r["axis"], r["seed"]), filt)
        per[name] = to_cells(d)
        seeds[name] = d
    return per, seeds, edit_prov


def pooled(per):
    """The three pooled paired deltas + the restricted estimates.
    The `edit - X` pairing (HEADLINE, .get(k, 0.0), keep zeroed cells) and the
    restricted rule are verbatim from v9_regen.frontier."""
    edc = per["edit"]
    out = {}
    for m in ("steering", "SentenceDebias", "DPO"):
        oc = per[m]
        out[POOLED_NAMES[m]] = boot_paired(
            [(edc[k], oc.get(k, 0.0)) for k in HEADLINE if k in edc])
        # v9_regen's rule: intersect on cells the OTHER arm actually has.
        out[f"{POOLED_NAMES[m]} [re-scorable cells only]"] = boot_paired(
            [(edc[k], oc[k]) for k in edc
             if k in oc and k[0] in RESTRICTED_TARGETS])
        # the same restriction under the pooled convention (zeroed cells KEPT),
        # so SD/DPO restricted estimates are comparable to the steering one.
        out[f"{POOLED_NAMES[m]} [re-scorable, zeroed cells kept]"] = boot_paired(
            [(edc[k], oc.get(k, 0.0)) for k in HEADLINE
             if k in edc and k[0] in RESTRICTED_TARGETS])
    return out


# ----------------------------------------------------- published-value IO ----

NUM = r"([+\-−]?\d+\.\d+)"
LINE_RE = re.compile(NUM + r"\s*\[\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\]")


def f2(s):
    return float(s.replace("−", "-"))


def method_v9_section3():
    """Parse the pooled frontier lines out of RESULTS_METHOD_v9.md §3.
    Each bullet is `- <name>: <as-run> -> **<v9>**`; we take the LAST triple on
    the line, which is the v9 one."""
    if not os.path.exists(METHOD_V9):
        return {}
    txt = open(METHOD_V9).read()
    sec = txt.split("## 3. D2")[-1].split("\n## ")[0]
    out = {}
    for ln in sec.splitlines():
        ln = ln.strip()
        if not ln.startswith(("-", ">")) and "Restricted" not in ln:
            continue
        hits = LINE_RE.findall(ln)
        if not hits:
            continue
        if ln.startswith("-"):
            name = ln[1:].split(":")[0].strip()
        elif "Restricted to the four" in ln:
            name = "edit - steering [re-scorable cells only]"
        else:
            continue
        p, lo, hi = hits[-1]
        out[name] = dict(point=f2(p), lo=f2(lo), hi=f2(hi), raw=ln)
    return out


def diff_report_frontier():
    """The four published pooled lines from results/v9/v9_diff_report.json."""
    rep = jload(DIFF_V9, default={}) or {}
    out = {}
    for k, v in (rep.get("diff") or {}).items():
        if not k.startswith("frontier :: "):
            continue
        name = k[len("frontier :: "):]
        out[name] = dict(v9_point=v.get("v9_point"), v8_point=v.get("v8_point"),
                         v9=v.get("v9"), v8=v.get("v8"))
    return out


def draft_table_a():
    """Parse Table A out of the manuscript, with line numbers, at run time."""
    if not os.path.exists(DRAFT):
        return {}, {}
    lines = open(DRAFT).read().splitlines()
    axes = sorted({a for _t, a in HEADLINE})
    cells, pooled_lines = {}, {}
    header_cols = None
    for i, raw in enumerate(lines, start=1):
        ln = raw.strip()
        if ln.startswith("| cell |"):
            header_cols = [c.strip() for c in ln.strip("|").split("|")][1:]
            continue
        if header_cols and ln.startswith("|") and "\\|" in ln:
            parts = [c.strip() for c in ln.replace("\\|", "\x00").strip("|").split("|")]
            tgt, _, ax_short = parts[0].partition("\x00")
            tgt, ax_short = tgt.strip(), ax_short.strip()
            ax = [a for a in axes if a.startswith(ax_short)]
            if len(ax) != 1 or tgt not in {t for t, _a in HEADLINE}:
                continue
            rec = {}
            for col, val in zip(header_cols, parts[1:]):
                m = re.search(NUM, val)
                if not m:
                    continue
                dec = len(m.group(1).split(".")[1])
                rec[col] = dict(value=f2(m.group(1)), decimals=dec,
                                marker=val.replace(m.group(1), "").strip(),
                                text=val)
            cells[(tgt, ax[0])] = dict(line=i, cols=rec, text=ln)
            continue
        # pooled sentences: grab every "<label> = point [lo, hi]" on the line
        for label in ("edit − steering", "edit − SentenceDebias",
                      "edit − DPO"):
            for m in re.finditer(re.escape(label) + r"\s*=\s*" + NUM +
                                 r"\s*\[\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\]", ln):
                key = label.replace("−", "-")
                if "Restricted to the four" in ln:
                    key += " [re-scorable cells only]"
                pooled_lines.setdefault(key, []).append(
                    dict(line=i, point=f2(m.group(1)), lo=f2(m.group(2)),
                         hi=f2(m.group(3)),
                         decimals=len(m.group(1).split(".")[1]), text=ln))
    return cells, pooled_lines


def near(a, b, decimals):
    """Equal once rounded to `decimals` places (half-unit tolerance)."""
    return abs(a - b) <= 0.5 * 10 ** (-decimals) + 1e-12


# ---------------------------------------- DPO re-gate diagnostic (advisory) --

def dpo_regate_diagnostic():
    """v9 marks every DPO row UNGATED because `removal.jsonl` carries no
    dmmlu/ppl_ratio — but `results/v6trace/dpo/alpha_trace.jsonl` DOES. This
    computes what the integer gate would do, purely as a disclosure. It never
    feeds the published numbers."""
    tp = os.path.join(RES, "v6trace", "dpo", "alpha_trace.jsonl")
    if not os.path.exists(tp):
        return dict(available=False, why=f"no {os.path.relpath(tp, HERE)}")
    try:
        from v9_gate import collateral_ok_row
    except Exception as e:                                   # pragma: no cover
        return dict(available=False, why=f"{type(e).__name__}: {e}")
    tr = [json.loads(l) for l in open(tp) if l.strip()]
    flips, modes = [], collections.Counter()
    regated = collections.defaultdict(lambda: float("nan"))
    for r in tr:
        ok, mode = collateral_ok_row(r)
        modes[mode] += 1
        if ok is not None and bool(ok) != bool(r.get("collateral_ok")):
            flips.append(dict(target=r["target"], axis=r["axis"], seed=r["seed"],
                              alpha=r.get("alpha"),
                              as_run=bool(r.get("collateral_ok")), v9=bool(ok)))
        if ok:
            k = (r["target"], r["axis"], r["seed"])
            cur = regated[k]
            br = r.get("bias_reduction")
            if br is not None:
                regated[k] = br if cur != cur else max(cur, br)
    rc = to_cells(dict(regated))
    return dict(available=True, n_trace_rows=len(tr), gate_modes=dict(modes),
                n_rows_flipped=len(flips), flips=flips,
                regated_cells={f"{t}|{a}": round(v, 6) for (t, a), v in
                               sorted(rc.items())},
                note="ADVISORY ONLY. The published v9 frontier keeps the DPO "
                     "arm UNGATED (v9_rescore routed it down the B-rows path, "
                     "where the removal rows carry no collateral). This shows "
                     "what re-gating from the trace would do; it is NOT "
                     "substituted for any published value.")


# ------------------------------------------------------------------- main ----

def main():
    ensure_dirs()
    per9, seeds9, prov9 = build(V9)
    per8, seeds8, prov8 = build(RES)
    methods = ("edit", "steering", "SentenceDebias", "DPO")
    panel_of = {"edit": None}
    filt_of = {}
    for name, panel, filt in ARMS:
        panel_of[name] = panel
        filt_of[name] = filt

    draft_cells, draft_pooled = draft_table_a()

    cells_out, zeroed = [], {m: [] for m in methods}
    changed = []
    for (t, a) in HEADLINE:
        key = (t, a)
        rec = dict(cell=f"{t}|{a}", target=t, axis=a,
                   short=f"{t} | {a.split('_')[0]}", methods={})
        e_true, e_src, e_mode = edit_cell_provenance(V9, t)
        rec["edit_true_v9"] = e_true
        for m in methods:
            v9v = per9[m].get(key)
            v8v = per8[m].get(key)
            is_zero = v9v is None
            val9 = 0.0 if is_zero else float(v9v)
            val8 = 0.0 if v8v is None else float(v8v)
            if m == "edit":
                tv9, src, gm = e_true, e_src, e_mode
            else:
                pan = panel_of[m]
                gmodes = panel_gate_modes(V9, pan)
                gm = "/".join(gmodes)
                src = os.path.relpath(os.path.join(V9, pan), HERE)
                # a panel whose every row is UNGATED was NOT re-gated by v9
                tv9 = bool(gmodes) and not all(g == "UNGATED" for g in gmodes)
            entry = dict(v9=round(val9, 6), as_run=round(val8, 6),
                         delta_v9_minus_asrun=round(val9 - val8, 6),
                         changed_by_v9=abs(val9 - val8) > 1e-12,
                         true_v9=bool(tv9), gate_mode=gm, source=src,
                         zeroed=bool(is_zero),
                         n_seeds=len([1 for (tt, aa, _s) in seeds9[m]
                                      if (tt, aa) == key]))
            if is_zero:
                entry["zeroed_reason"] = (
                    "edit arm has no rows for this cell"
                    if m == "edit" else
                    zero_reason(V9, panel_of[m], filt_of[m], t, a))
                entry["is_measurement"] = False
                zeroed[m].append(f"{t}|{a}")
            else:
                entry["is_measurement"] = True
            if entry["changed_by_v9"]:
                changed.append(dict(cell=f"{t}|{a}", method=m,
                                    as_run=entry["as_run"], v9=entry["v9"],
                                    delta=entry["delta_v9_minus_asrun"]))
            # draft cross-check
            d = draft_cells.get(key, {}).get("cols", {}).get(m)
            if d is not None:
                entry["draft"] = dict(
                    line=draft_cells[key]["line"], text=d["text"],
                    value=d["value"], marker=d["marker"],
                    matches_v9=near(d["value"], val9, d["decimals"]),
                    matches_as_run=near(d["value"], val8, d["decimals"]),
                    marks_zeroed=bool(d["marker"]) and d["marker"] != "*")
            rec["methods"][m] = entry
        cells_out.append(rec)

    pool9, pool8 = pooled(per9), pooled(per8)

    # ---- reproduction against the published artifacts
    pub_json = diff_report_frontier()
    pub_md = method_v9_section3()
    repro = []
    for name in sorted(set(pub_json) | set(pub_md)):
        mine = pool9.get(name)
        r = dict(estimand=name, computed=None if mine is None else fmt(mine))
        if name in pub_json:
            r["published_diff_report_v9_point"] = pub_json[name]["v9_point"]
            r["published_diff_report_v9"] = pub_json[name]["v9"]
            r["matches_diff_report"] = (
                mine is not None and pub_json[name]["v9_point"] is not None and
                abs(mine["point"] - pub_json[name]["v9_point"]) < 5e-7)
        if name in pub_md:
            p = pub_md[name]
            r["published_method_v9"] = [p["point"], p["lo"], p["hi"]]
            r["matches_method_v9"] = (
                mine is not None and near(p["point"], mine["point"], 4) and
                near(p["lo"], mine["lo"], 4) and near(p["hi"], mine["hi"], 4))
        repro.append(r)
    # the as-run side of the published diff, too
    for name, v in pub_json.items():
        mine = pool8.get(name)
        repro.append(dict(estimand=name + " [as-run]",
                          computed=None if mine is None else fmt(mine),
                          published_diff_report_v8_point=v["v8_point"],
                          published_diff_report_v8=v["v8"],
                          matches_diff_report=(mine is not None and
                                               v["v8_point"] is not None and
                                               abs(mine["point"] - v["v8_point"]) < 5e-7)))

    # ---- draft pooled sentences
    pooled_draft = []
    for name, hits in sorted(draft_pooled.items()):
        mine = pool9.get(name)
        for h in hits:
            pooled_draft.append(dict(
                estimand=name, draft_line=h["line"], draft_text=h["text"],
                draft=[h["point"], h["lo"], h["hi"]],
                computed_v9=None if mine is None else
                [round(mine["point"], 6), round(mine["lo"], 6), round(mine["hi"], 6)],
                matches_v9=bool(mine is not None and
                                near(h["point"], mine["point"], h["decimals"]) and
                                near(h["lo"], mine["lo"], h["decimals"]) and
                                near(h["hi"], mine["hi"], h["decimals"]))))

    art = dict(
        artifact="v10_frontier",
        generated_by="src/v10_regen_frontier.py",
        purpose="Fig 5 source + manuscript Table A, regenerated per cell.",
        conventions=dict(
            budget_fail="nan/missing -> 0.0 (kept in the pooled set, never dropped)",
            bootstrap="paired, resamples the (target,axis) CELL, 10k, seed 0",
            cell="mean over seeds",
            selection="argmax bias_reduction (or removal) over in-budget configs",
            logic_source="src/v9_regen.py :: frontier/edit_arm/best_in_budget"),
        headline_cells=[f"{t}|{a}" for t, a in HEADLINE],
        methods=list(methods),
        cells=cells_out,
        gate_summary=dict(
            edit_cells_true_v9=sorted(r["cell"] for r in cells_out
                                      if r["edit_true_v9"]),
            edit_cells_as_run=sorted(r["cell"] for r in cells_out
                                     if not r["edit_true_v9"]),
            n_edit_cells_as_run=sum(1 for r in cells_out
                                    if not r["edit_true_v9"]),
            n_true_v9_dots={m: sum(1 for r in cells_out
                                   if r["methods"][m]["true_v9"])
                            for m in methods},
            n_true_v9_dots_total=sum(1 for r in cells_out for m in methods
                                     if r["methods"][m]["true_v9"]),
            note="`n_edit_cells_as_run` is the count the manuscript's Gate-"
                 "mixing limitation must quote. `true_v9` is per (cell, "
                 "method): the DPO column is false everywhere because that "
                 "panel is UNGATED."),
        zeroed_cells=zeroed,
        n_zeroed=sum(len(v) for v in zeroed.values()),
        n_dots=len(HEADLINE) * len(methods),
        cells_changed_by_v9=changed,
        edit_provenance=dict(v9=prov9, as_run=prov8),
        pooled_v9={k: (None if v is None else
                       dict(v, point=round(v["point"], 6),
                            lo=round(v["lo"], 6), hi=round(v["hi"], 6),
                            status=status(v), text=fmt(v)))
                   for k, v in pool9.items()},
        pooled_as_run={k: (None if v is None else
                           dict(v, point=round(v["point"], 6),
                                lo=round(v["lo"], 6), hi=round(v["hi"], 6),
                                status=status(v), text=fmt(v)))
                       for k, v in pool8.items()},
        restricted_estimate_availability=dict(
            published={k: (k in pub_json or k in pub_md)
                       for k in sorted(pool9) if "re-scorable" in k},
            note="Only `edit - steering [re-scorable cells only]` exists in a "
                 "published v9 artifact. The SentenceDebias and DPO restricted "
                 "estimates emitted here are NEW; they have no published "
                 "counterpart and must be labelled as v10 additions if cited. "
                 "Note the two rules differ for those arms: v9_regen's rule "
                 "intersects on cells the other arm HAS (dropping zeroed "
                 "cells, so n<4), while the pooled convention keeps them "
                 "(n=4). Both are emitted."),
        dpo_gate=dict(
            v9_gate_mode=panel_gate_modes(V9, "v6trace/dpo"),
            as_run_gate_mode=panel_gate_modes(RES, "v6trace/dpo"),
            rescore_path=(jload("results_v9/rescore_report.json", default={}) or {})
                .get("panels", {}).get("v6trace/dpo"),
            re_gated_under_v9=False,
            statement="Every DPO row carries v9_gate_mode=UNGATED. The v9 "
                      "re-score could not re-gate this panel from "
                      "removal.jsonl (no dmmlu / ppl_ratio on the rows), so "
                      "the DPO arm's collateral_ok is the AS-RUN float gate. "
                      "The DPO column of Table A is therefore not true-v9 "
                      "either — a gate-mixing exposure the manuscript's "
                      "disclosure (which names only the llama/qwen EDIT "
                      "cells) does not cover.",
            diagnostic=dpo_regate_diagnostic()),
        reproduction=repro,
        draft_table_a=pooled_draft,
        blocked=dict(**{
            "frontier edit arm: llama, qwen":
                "results/x has no alpha_trace and no re-scorable twin; those 4 "
                "cells stay as-run -> the 8-cell frontier is MIXED-gate.",
            "frontier DPO arm (all 8 cells)":
                "results_v9/v6trace/dpo rows are UNGATED; the arm was not "
                "re-gated. MIXED-gate.",
        }),
    )
    fp = jdump(art, OUT)

    # ------------------------------------------------------------- console --
    print("=" * 78)
    print("v10 FRONTIER — per-cell regeneration (Fig 5 / Table A)")
    print("=" * 78)
    hdr = f"{'cell':28s}" + "".join(f"{m:>18s}" for m in methods)
    print(hdr)
    for rec in cells_out:
        line = f"{rec['short']:28s}"
        for m in methods:
            e = rec["methods"][m]
            tag = "Z" if e["zeroed"] else (" " if e["true_v9"] else "~")
            d = f"{e['v9']:+.4f}{tag}"
            if e["changed_by_v9"]:
                d += f"<-{e['as_run']:+.4f}"
            line += f"{d:>18s}"
        print(line)
    print("\n  legend:  (blank)=true-v9   ~=as-run/ungated   Z=zeroed by nan->0"
          "   <-x = as-run value it replaced")

    print("\nDIFF LOG — cells that moved under v9:")
    if not changed:
        print("   (none)")
    for c in changed:
        print(f"   {c['method']:15s} {c['cell']:28s} "
              f"{c['as_run']:+.4f} -> {c['v9']:+.4f}  ({c['delta']:+.4f})")

    print(f"\nZEROED DOTS ({art['n_zeroed']} of {art['n_dots']}) — not measurements:")
    for m in methods:
        for c in zeroed[m]:
            rr = next(r for r in cells_out if r["cell"] == c)
            print(f"   {m:15s} {c:28s} {rr['methods'][m]['zeroed_reason']}")

    print("\nPOOLED (v9) vs published:")
    for r in repro:
        ok = [v for k, v in r.items() if k.startswith("matches")]
        mark = "OK " if all(ok) and ok else "!! "
        print(f"   {mark}{r['estimand']:52s} {r['computed']}")
    print("\nNEW restricted estimates (no published counterpart):")
    for k, v in sorted(art["pooled_v9"].items()):
        if "re-scorable" in k and k not in pub_json and k not in pub_md:
            print(f"   {k:60s} {v['text'] if v else 'n/a'}")

    print("\nDRAFT Table A cross-check:")
    bad = 0
    for rec in cells_out:
        for m in methods:
            d = rec["methods"][m].get("draft")
            if d and not d["matches_v9"]:
                bad += 1
                print(f"   line {d['line']}  {rec['cell']:28s} {m:15s} "
                      f"draft {d['text']:>10s}  v9 {rec['methods'][m]['v9']:+.4f}"
                      f"  (as-run {rec['methods'][m]['as_run']:+.4f}"
                      f"{', = as-run' if d['matches_as_run'] else ''})")
            if d and rec["methods"][m]["zeroed"] and not d["marks_zeroed"]:
                bad += 1
                print(f"   line {d['line']}  {rec['cell']:28s} {m:15s} "
                      f"draft {d['text']:>10s} presented as a measurement, but "
                      f"the value is 0.0 from the nan->0 rule")
    for p in pooled_draft:
        if not p["matches_v9"]:
            bad += 1
            print(f"   line {p['draft_line']}  POOLED {p['estimand']}: "
                  f"draft {p['draft']} vs v9 {p['computed_v9']}")
    print(f"   -> {bad} disagreement(s)")

    print("\nDPO gate:", art["dpo_gate"]["v9_gate_mode"],
          "| re-gated under v9:", art["dpo_gate"]["re_gated_under_v9"])
    dg = art["dpo_gate"]["diagnostic"]
    if dg.get("available"):
        print(f"   advisory re-gate from alpha_trace: "
              f"{dg['n_rows_flipped']} of {dg['n_trace_rows']} rows would flip")
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
