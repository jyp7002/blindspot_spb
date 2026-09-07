"""v10 Fig 6 source — the ENV operating characterization, re-derived under v9.

WHY THIS MODULE EXISTS
----------------------
`results/v8/env_{inventory,split,policy,sensitivity}.json` are AS-RUN: they were
built before the collateral-gate float fix, and `RESULTS_METHOD_v9.md` §5 records
ENV as "partially blocked" precisely because its inputs mix re-scorable and
non-re-scorable panels. No v9 ENV artifact exists. Fig 6 must nonetheless be
drawn over v9-scored removals, and the +0.910 pre_skew/removal correlation the
draft stars is a function of gated removals, so `experiments_v10.md` A5 orders it
recomputed. This module does that and emits `results/v10/env_v9.json`.

DEFINITIONS ARE IMPORTED, NOT RETYPED
-------------------------------------
Every ENV definition is taken from `src/v8_env.py` itself, by import:

  * cell               `v8_env.cell_table`  -> (target, axis) under role="self",
                       mean over rows, budget-fail already mapped to 0.0
  * benchmark family   `v8_env.axis_family`
  * backfire           a cell with `removal < 0`. This is the definition the code
                       uses in `v8_env.score` ("backfire_rate_edited") and the one
                       `v8_figures.py` draws (lines 76 and 128, "backfire region:
                       removal < 0"). It is applied to the CELL value, i.e. to the
                       nan->0 mean over the cell's rows, not to individual rows.
  * utility            `v8_env.score` -> sum(removal over EDITED cells) / n_cells,
                       i.e. abstention scores 0, not "excluded from the average"
  * split              `v8_env.make_split` -> stratified by axis_family, seeded
                       alternate deal, granite{,2,8} x occ_gender excluded
  * calibration search `v8_env.calibrate`  (utility max s.t. backfire rate <= 0.10)

THE v9 RESTRICTION, AND WHAT IT COSTS
-------------------------------------
`v8_env.EDIT_PANELS` lists 12 panels. Four of them — t2x, v1, x, x2 — are in
`v10_common.NOT_RESCORABLE`: v5-era panels with no alpha_trace, so there is no
record of which alphas were probed or what collateral they cost. They cannot be
re-scored and must never be silently substituted, so the v9 set simply drops
them. Two consequences are computed and reported here rather than glossed:

  1. t2x and x2 have exact re-scorable twins already in the panel list
     (v6trace/t2x, v6trace/x2), so dropping them removes DUPLICATE observations;
     v1 and x have no twin, so dropping them removes cells outright.
  2. A cell's pre_skew is a MEAN OVER ITS ROWS. Shrinking the row set therefore
     moves pre_skew as well as removal — the x axis of Fig 6 is not gate-invariant.
     Every per-cell shift is emitted so the caption can state the restriction.

Both sets (v9-restricted and full as-run) are emitted, labelled, so the figure can
be drawn on v9 while the caption quantifies what v9 excludes.

Bootstrap discipline is the project's: resample the CELL, 10k draws, seed 0.
"""
import os, sys, json, hashlib, re, random, collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10_common import (HERE, RES, V9, OUT_V10, NOT_RESCORABLE, BOOT_N, BOOT_SEED,
                        z, jload, jdump, ensure_dirs, boot_mean, status)
import v8_env as E

OUT_JSON = os.path.join(OUT_V10, "env_v9.json")
DRAFT = os.path.join(HERE, "binary_debiaser_draft_v3.md")
SPLIT_FP = os.path.join(RES, "v8", "env_split.json")
POLICY_FP = os.path.join(RES, "v8", "env_policy.json")
SENS_FP = os.path.join(RES, "v8", "env_sensitivity.json")
INV_FP = os.path.join(RES, "v8", "env_inventory.json")


# ------------------------------------------------------------- utilities ----

def _clean(o):
    """numpy scalars -> python; nan/inf -> None, so the artifact is strict JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _sha256(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _panel_dir(rel):
    """('v6trace/ax/removal.jsonl') -> 'v6trace/ax'."""
    return rel[:-len("/removal.jsonl")] if rel.endswith("/removal.jsonl") else rel


def rescorable_panels():
    """v8_env.EDIT_PANELS minus v10_common.NOT_RESCORABLE, minus anything that
    has no file under results_v9/. Derived — nothing is listed by hand."""
    keep, dropped = [], []
    for rel, label in E.EDIT_PANELS:
        d = _panel_dir(rel)
        if d in NOT_RESCORABLE:
            dropped.append((rel, label, "in v10_common.NOT_RESCORABLE"))
        elif not os.path.exists(os.path.join(V9, rel)):
            dropped.append((rel, label, "no results_v9 file"))
        else:
            keep.append((rel, label))
    return keep, dropped


# --------------------------------------------------------------- rows ------

def build_rows(root, panels, gap):
    """Root-parameterised twin of v8_env.build_inventory's row loop.

    Identical join (designer, axis, seed) -> corpus contrast_gap, identical
    budget-fail rule, plus the three v9 provenance fields when the row carries
    them (results_v9 rows do; results/ rows do not)."""
    out, unjoined = [], collections.Counter()
    for rel, label in panels:
        fp = os.path.join(root, rel)
        if not os.path.exists(fp):
            continue
        with open(fp) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                g = gap.get((r.get("designer"), r.get("axis"), r.get("seed")))
                if g is None:
                    unjoined[(label, r.get("designer"), r.get("axis"))] += 1
                    continue
                rem = r.get("removal")
                bf = rem is None or (isinstance(rem, float) and rem != rem)
                out.append(dict(
                    panel=label, target=r["target"], axis=r["axis"], seed=r["seed"],
                    role=r.get("role"), designer=r["designer"],
                    axis_family=E.axis_family(r["axis"]),
                    removal=z(None if bf else rem), budget_fail=bool(bf),
                    pre_skew=float(r["pre_skew"]),
                    contrast_gap=g["contrast_gap"], corpus_md5=g["corpus_md5"],
                    n_items=g["n_items"],
                    v9_changed=r.get("v9_changed"),
                    v9_gate_mode=r.get("v9_gate_mode", "as-run (not v9-gated)"),
                    v9_selected_alpha=r.get("v9_selected_alpha")))
    return out, unjoined


def core(r):
    """The field set v8_env.build_inventory emits, for the fidelity self-check."""
    return tuple(sorted((k, v) for k, v in r.items()
                        if k in ("panel", "target", "axis", "seed", "role",
                                 "designer", "removal", "budget_fail", "pre_skew",
                                 "contrast_gap", "corpus_md5", "n_items")))


# ------------------------------------------------------------- correlation --

def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _ranks(v):
    """Average ranks, so ties are handled the way Spearman requires."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def _spearman(x, y):
    if len(x) < 3:
        return None
    return _pearson(_ranks(x), _ranks(y))


def _pval(r, n):
    """Two-sided p for a correlation, via scipy when it is importable. It is a
    convenience only — nothing in the artifact depends on it."""
    if r is None or n < 3 or abs(r) >= 1.0:
        return None
    try:
        from scipy import stats as _st
    except Exception:
        return None
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * _st.t.sf(abs(t), n - 2))


def boot_corr(x, y, fn, n=BOOT_N, seed=BOOT_SEED):
    """Percentile bootstrap of a correlation, resampling the CELL — same unit,
    count and RNG discipline as v10_common.boot_mean."""
    m = len(x)
    if m < 3:
        return None
    rng = random.Random(seed)
    bs = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        v = fn([x[i] for i in idx], [y[i] for i in idx])
        if v is not None and v == v:
            bs.append(v)
    if not bs:
        return None
    bs.sort()
    return dict(lo=bs[int(.025 * len(bs))], hi=bs[int(.975 * len(bs)) - 1],
                n_valid=len(bs))


def corr_block(cells, label, xkey="pre_skew"):
    x = [c[xkey] for c in cells]
    y = [c["removal"] for c in cells]
    p = _pearson(x, y)
    s = _spearman(x, y)
    return dict(label=label, n=len(cells), x=xkey,
                pearson=p, pearson_p=_pval(p, len(x)),
                pearson_ci=boot_corr(x, y, _pearson),
                spearman=s, spearman_p=_pval(s, len(x)),
                spearman_ci=boot_corr(x, y, _spearman),
                cells=[f"{c['target']}|{c['axis']}" for c in cells])


# ------------------------------------------------------------------ split ---

def frozen_assignment():
    """(target, axis) -> 'calibration' | 'held_out' | 'excluded', read off the
    frozen, hashed split artifact. The freeze rule (§ENV.1) makes the file
    authoritative: a cell keeps its assignment even when the population it was
    dealt from shrinks."""
    sp = jload(SPLIT_FP)
    if sp is None:
        return None, None
    a = {}
    for key in ("calibration", "held_out", "excluded"):
        for c in sp[key]:
            a[(c["target"], c["axis"])] = key
    return a, sp


def cell_records(cells, rows, assign, tag):
    """Fig-6-ready cell records: geometry, family, split, backfire, gate status."""
    by = collections.defaultdict(list)
    for r in rows:
        if r["role"] == "self":
            by[(r["target"], r["axis"])].append(r)
    out = []
    for c in cells:
        k = (c["target"], c["axis"])
        rs = by[k]
        rec = dict(c)
        rec["cell"] = f"{k[0]}|{k[1]}"
        rec["set"] = tag
        rec["split"] = assign.get(k, "UNASSIGNED (not in frozen split)")
        rec["backfire"] = bool(c["removal"] < 0)      # v8_env.score's definition
        rec["v9_gate_modes"] = sorted({str(r["v9_gate_mode"]) for r in rs})
        rec["n_v9_changed"] = sum(1 for r in rs if r.get("v9_changed") is True)
        rec["seeds"] = sorted({r["seed"] for r in rs})
        out.append(rec)
    return sorted(out, key=lambda r: (-r["pre_skew"], r["cell"]))


# ----------------------------------------------------------------- policy ---

def _in_order(cells, order, key):
    """Reorder `cells` to follow `order` (a list of identity tuples), dropping
    anything absent. v8_env._boot_ci resamples the list POSITIONALLY with a fixed
    RNG, so the CI depends on the order the split artifact recorded — this is what
    makes the as-run policy table reproduce env_policy.json bit for bit."""
    idx = {k: i for i, k in enumerate(order)}
    return [c for c in sorted(cells, key=lambda c: idx.get(key(c), 1 << 30))
            if key(c) in idx]


def policy_block(cells, assign, label, cal_order, hold_order):
    """Calibrate on the frozen calibration cells, score the frozen held-out
    cells. Thresholds come from v8_env.calibrate; the table from
    v8_env.policy_table — both unmodified."""
    k = lambda c: (c["target"], c["axis"])
    cal = _in_order(cells, cal_order, k)
    hold = _in_order(cells, hold_order, k)
    if not cal or not hold:
        return dict(label=label, blocked="empty calibration or held-out set")
    th = E.calibrate(cal)
    tab = E.policy_table(hold, th)
    edit_all_cal = E.score(cal)
    joint = dict(E.score(cal, th["joint_gate"]["tau"], th["joint_gate"]["gamma"]))
    return dict(label=label, n_cal=len(cal), n_held_out=len(hold),
                cal_cells=[f"{c['target']}|{c['axis']}" for c in cal],
                held_out_cells=[f"{c['target']}|{c['axis']}" for c in hold],
                backfire_cap=E.BACKFIRE_CAP, split_seed=E.SPLIT_SEED,
                thresholds=th,
                calibration_edit_all=edit_all_cal,
                calibration_joint_gate=joint,
                calibration_n_rejected=edit_all_cal["n_edited"] - joint["n_edited"],
                held_out_table=tab)


# ------------------------------------------------------------ draft parse ---

_NUM = r"[+\-−]?\d*\.?\d+"


def _f(s):
    return float(s.replace("−", "-"))


def parse_draft():
    """Pull the claims Fig 6 is responsible for straight out of the manuscript,
    with line numbers. Nothing about the draft is typed into this file."""
    if not os.path.exists(DRAFT):
        return {}
    lines = open(DRAFT).read().split("\n")
    got = {}

    def scan(key, pattern, cast):
        rx = re.compile(pattern)
        for i, ln in enumerate(lines, 1):
            m = rx.search(ln)
            if m:
                got[key] = dict(line=i, raw=m.group(0), value=cast(m),
                                sentence=ln.strip()[:400])
                return

    scan("corr_claim",
         r"corr\s*=\s*(" + _NUM + r")\*?\s*over\s+(\d+)\s+([A-Za-z \-]*?)cells",
         lambda m: dict(corr=_f(m.group(1)), n=int(m.group(2)),
                        described_as=m.group(3).strip()))
    scan("backfire_magnitude_claim",
         r"increasing bias by\s*[≈~]?\s*(" + _NUM + r")",
         lambda m: _f(m.group(1)))
    scan("calibration_backfire_claim",
         r"backfire\s+(" + _NUM + r")\s*(?:->|→)\s*(" + _NUM + r")",
         lambda m: [_f(m.group(1)), _f(m.group(2))])
    scan("calibration_utility_claim",
         r"utility\s+(" + _NUM + r")\s*(?:->|→)\s*(" + _NUM + r")",
         lambda m: [_f(m.group(1)), _f(m.group(2))])
    scan("rejection_claim",
         r"rejecting\s+(\d+)\s+of\s+(\d+)\s+cells",
         lambda m: [int(m.group(1)), int(m.group(2))])
    scan("sensitivity_n_claim",
         r"n\s*=\s*(\d+)\s+sensitivity",
         lambda m: int(m.group(1)))
    scan("sensitivity_gain_claim",
         r"pre-skew gate buys\s*(" + _NUM + r")\s*utility",
         lambda m: _f(m.group(1)))
    return got


# ------------------------------------------------------------------- main ---

def main():
    ensure_dirs()

    # ---- inputs, all read at run time -------------------------------------
    gap = E.load_gap_index()
    checked, disagree = E.crosscheck_frozen(gap)
    if disagree:
        raise SystemExit(f"[ENV-v9] FATAL: contrast_gap disagreements: {disagree[:3]}")

    keep, dropped = rescorable_panels()
    all_panels = list(E.EDIT_PANELS)

    rows_a_auth, _g, unj_a, _c = E.build_inventory()          # v8_env's own output
    rows_a, unj_mine = build_rows(RES, all_panels, gap)       # my builder, as-run
    rows_9, unj_9 = build_rows(V9, keep, gap)                 # my builder, v9

    fidelity = (sorted(map(core, rows_a_auth)) == sorted(map(core, rows_a)))

    cells_a = E.cell_table(rows_a, role="self")
    cells_9 = E.cell_table(rows_9, role="self")

    assign, sp = frozen_assignment()
    if assign is None:
        raise SystemExit("[ENV-v9] FATAL: results/v8/env_split.json missing")

    # ---- the frozen split, reproduced from the code -----------------------
    cal_r, hold_r, excl_r = E.make_split(cells_a)
    split_ok = all(
        [c["target"] for c in a] == [c["target"] for c in b] and
        [c["axis"] for c in a] == [c["axis"] for c in b]
        for a, b in ((cal_r, sp["calibration"]), (hold_r, sp["held_out"]),
                     (excl_r, sp["excluded"])))
    # what a NAIVE re-deal on the shrunken v9 population would give — reported
    # to show why the frozen assignment is carried instead of re-dealt
    cal_9r, hold_9r, excl_9r = E.make_split(cells_9)

    rec_a = cell_records(cells_a, rows_a, assign, "as_run")
    rec_9 = cell_records(cells_9, rows_9, assign, "v9")

    da = {c["cell"]: c for c in rec_a}
    d9 = {c["cell"]: c for c in rec_9}

    # ---- accounting of the v9 restriction ---------------------------------
    rows_by_panel_a = collections.Counter(r["panel"] for r in rows_a)
    rows_by_panel_9 = collections.Counter(r["panel"] for r in rows_9)
    shift = []
    for k in sorted(set(da) & set(d9)):
        a, b = da[k], d9[k]
        shift.append(dict(
            cell=k, split=a["split"], axis_family=a["axis_family"],
            n_obs_as_run=a["n_obs"], n_obs_v9=b["n_obs"],
            pre_skew_as_run=a["pre_skew"], pre_skew_v9=b["pre_skew"],
            d_pre_skew=b["pre_skew"] - a["pre_skew"],
            removal_as_run=a["removal"], removal_v9=b["removal"],
            d_removal=b["removal"] - a["removal"],
            contrast_gap_as_run=a["contrast_gap"], contrast_gap_v9=b["contrast_gap"]))
    lost_cells = [da[k] for k in sorted(set(da) - set(d9))]

    restriction = dict(
        rule="drop every EDIT_PANEL whose directory is in v10_common.NOT_RESCORABLE",
        panels_kept=[l for _r, l in keep],
        panels_dropped=[dict(panel=l, path=r, why=w) for r, l, w in dropped],
        n_rows_as_run=len(rows_a), n_rows_v9=len(rows_9),
        n_rows_dropped=len(rows_a) - len(rows_9),
        rows_dropped_by_panel={p: rows_by_panel_a[p] - rows_by_panel_9.get(p, 0)
                               for p in sorted(rows_by_panel_a)
                               if rows_by_panel_a[p] - rows_by_panel_9.get(p, 0)},
        n_self_rows_as_run=sum(1 for r in rows_a if r["role"] == "self"),
        n_self_rows_v9=sum(1 for r in rows_9 if r["role"] == "self"),
        n_cells_as_run=len(cells_a), n_cells_v9=len(cells_9),
        cells_lost=[dict(cell=c["cell"], split=c["split"],
                         axis_family=c["axis_family"], panels=c["panels"],
                         pre_skew=c["pre_skew"], removal=c["removal"])
                    for c in lost_cells],
        family_counts_as_run=dict(collections.Counter(c["axis_family"] for c in cells_a)),
        family_counts_v9=dict(collections.Counter(c["axis_family"] for c in cells_9)),
        duplicate_note=("t2x and x2 have exact re-scorable twins already in the "
                        "panel list (v6trace_t2x, v6trace_x2), so dropping them "
                        "removes duplicated observations; v1 and x have no twin, "
                        "so dropping them removes cells outright."),
        pre_skew_is_not_gate_invariant=(
            "a cell's pre_skew is the mean over its rows, so the v9 row set moves "
            "the Fig 6 x axis as well as the y axis"),
        max_abs_d_pre_skew=max((abs(s["d_pre_skew"]) for s in shift), default=None),
        max_abs_d_removal=max((abs(s["d_removal"]) for s in shift), default=None),
        n_cells_with_pre_skew_shift=sum(1 for s in shift if abs(s["d_pre_skew"]) > 0),
        n_cells_with_removal_shift=sum(1 for s in shift if abs(s["d_removal"]) > 0),
        per_cell_shift=shift)

    # ---- backfire ----------------------------------------------------------
    def backfire_block(recs, tag):
        bf = sorted([c for c in recs if c["backfire"]], key=lambda c: c["pre_skew"])
        mags = [-c["removal"] for c in bf]
        lows = sorted(recs, key=lambda c: c["abs_pre_skew"])[:len(bf)] if bf else []
        return dict(
            set=tag,
            definition="cell removal < 0 (v8_env.score / v8_figures.py:76,128)",
            n_cells=len(recs), n_backfire=len(bf),
            backfire_fraction=len(bf) / len(recs) if recs else None,
            cells=[dict(cell=c["cell"], split=c["split"], axis_family=c["axis_family"],
                        pre_skew=c["pre_skew"], removal=c["removal"]) for c in bf],
            mean_magnitude=float(np.mean(mags)) if mags else None,
            median_magnitude=float(np.median(mags)) if mags else None,
            max_magnitude=float(max(mags)) if mags else None,
            min_magnitude=float(min(mags)) if mags else None,
            deepest_cell=bf[int(np.argmax(mags))]["cell"] if bf else None,
            lowest_pre_skew_cells=[dict(cell=c["cell"], pre_skew=c["pre_skew"],
                                        removal=c["removal"]) for c in lows],
            mean_removal_over_lowest_k=(float(np.mean([c["removal"] for c in lows]))
                                        if lows else None))

    back_9, back_a = backfire_block(rec_9, "v9"), backfire_block(rec_a, "as_run")

    # ---- correlations ------------------------------------------------------
    def corrs(recs, tag):
        sel = lambda f: [c for c in recs if f(c)]
        blocks = [
            corr_block(recs, f"{tag}: all self cells"),
            corr_block(sel(lambda c: c["split"] != "excluded"),
                       f"{tag}: non-excluded cells"),
            corr_block(sel(lambda c: c["split"] == "calibration"),
                       f"{tag}: calibration cells"),
            corr_block(sel(lambda c: c["split"] == "held_out"),
                       f"{tag}: held-out cells"),
            corr_block(sel(lambda c: c["axis_family"] == "BBQ"),
                       f"{tag}: BBQ cells (the +0.910 referent)"),
            corr_block(sel(lambda c: c["axis_family"] == "BBQ"
                           and c["split"] == "held_out"),
                       f"{tag}: BBQ cells that are ALSO held-out under the frozen split"),
            corr_block(sel(lambda c: c["axis_family"] in ("BBQ", "StereoSet")),
                       f"{tag}: BBQ + StereoSet ('lands on the same line')"),
            corr_block(recs, f"{tag}: all self cells, x=|pre_skew|", xkey="abs_pre_skew"),
            corr_block(recs, f"{tag}: all self cells, x=contrast_gap", xkey="contrast_gap"),
        ]
        return {b["label"]: b for b in blocks}

    # row-level (self rows, no cell aggregation) — reported for completeness so
    # "points" cannot be ambiguous; the figure and the headline use cells.
    def row_corr(rows, tag):
        rs = [r for r in rows if r["role"] == "self"]
        x = [r["pre_skew"] for r in rs]
        y = [r["removal"] for r in rs]
        p, s = _pearson(x, y), _spearman(x, y)
        return dict(label=f"{tag}: self ROWS (not cells)", n=len(rs),
                    pearson=p, pearson_p=_pval(p, len(rs)),
                    spearman=s, spearman_p=_pval(s, len(rs)))

    corr_9, corr_a = corrs(rec_9, "v9"), corrs(rec_a, "as_run")

    bbq9 = corr_9["v9: BBQ cells (the +0.910 referent)"]
    bbqa = corr_a["as_run: BBQ cells (the +0.910 referent)"]

    # ---- policy (supports the caption's "held-out failure" clause) ---------
    cal_order = [(c["target"], c["axis"]) for c in sp["calibration"]]
    hold_order = [(c["target"], c["axis"]) for c in sp["held_out"]]
    pol_a = policy_block(cells_a, assign, "as_run, unit=(target,axis) self",
                         cal_order, hold_order)
    pol_9 = policy_block(cells_9, assign, "v9, unit=(target,axis) self",
                         cal_order, hold_order)
    pub = jload(POLICY_FP) or {}
    pol_a["reproduces_env_policy_json"] = dict(
        thresholds=(pol_a["thresholds"] == pub.get("thresholds")),
        held_out_table=(pol_a["held_out_table"] ==
                        [{k: v for k, v in r.items()} for r in pub.get("held_out_table", [])]))

    # registered SENSITIVITY unit — the n = 87 the draft cites. env_sensitivity.json
    # stores only counts, so the assignment is re-derived from v8_env.make_split on
    # the as-run designer cells (which reproduces 87/87) and then CARRIED onto the
    # v9 designer cells, exactly as the cell-level frozen split is carried.
    def _sens_eval(cal, hold, tag, how):
        if not cal or not hold:
            return dict(label=tag, how=how, blocked="empty calibration or held-out")
        th = E.calibrate(cal)
        tab = E.policy_table(hold, th)
        ea = [t for t in tab if t["policy"] == "edit-all"][0]
        ps = [t for t in tab if t["policy"] == "pre-skew gate"][0]
        cg = [t for t in tab if t["policy"] == "contrast-gap gate"][0]
        return dict(label=tag, how=how, unit="(target,axis,designer), all roles",
                    n_cal=len(cal), n_held_out=len(hold), thresholds=th,
                    held_out_table=tab,
                    edit_all_utility=ea["utility"],
                    pre_skew_gate_utility_gain=ps["utility"] - ea["utility"],
                    contrast_gap_gate_utility_gain=cg["utility"] - ea["utility"])

    dc_a = E.designer_cells(rows_a)
    dc_9 = E.designer_cells(rows_9)
    cal_da, hold_da, excl_da = E.make_split(dc_a)
    dassign = {}
    for grp, nm in ((cal_da, "calibration"), (hold_da, "held_out"),
                    (excl_da, "excluded")):
        for c in grp:
            dassign[(c["target"], c["axis"], c["designer"])] = nm

    dk = lambda c: (c["target"], c["axis"], c["designer"])
    dcal_order = [dk(c) for c in cal_da]
    dhold_order = [dk(c) for c in hold_da]

    def _pick(dc, order):
        return _in_order(dc, order, dk)

    sens_a = _sens_eval(cal_da, hold_da, "as_run",
                        "v8_env.make_split re-derived on the as-run designer cells")
    sens_a["n_designer_cells"] = len(dc_a)
    _psens = jload(SENS_FP) or {}
    sens_a["reproduces_env_sensitivity_json"] = dict(
        n_held_out=(_psens.get("n_held_out") == len(hold_da)),
        thresholds=(sens_a["thresholds"] == _psens.get("thresholds")),
        held_out_table=(sens_a["held_out_table"] == _psens.get("held_out_table")))
    sens_9 = _sens_eval(_pick(dc_9, dcal_order), _pick(dc_9, dhold_order), "v9",
                        "as-run designer-split assignment CARRIED to the v9 "
                        "designer cells (primary)")
    sens_9["n_designer_cells"] = len(dc_9)
    cal_d9, hold_d9, excl_d9 = E.make_split(dc_9)
    sens_9["naive_redeal"] = dict(
        n_cal=len(cal_d9), n_held_out=len(hold_d9), n_excluded=len(excl_d9),
        warning="NOT used: v8_env.make_split deals alternately after a seeded "
                "shuffle, so a shrunken population re-deals differently and the "
                "units would no longer be the registered ones.",
        **{k: v for k, v in _sens_eval(cal_d9, hold_d9, "v9-redealt",
                                       "re-dealt").items()
           if k in ("thresholds", "held_out_table", "edit_all_utility",
                    "pre_skew_gate_utility_gain")})

    # ---- draft cross-check -------------------------------------------------
    draft = parse_draft()
    checks = []
    cc = draft.get("corr_claim")
    if cc:
        checks.append(dict(
            id="fig6.corr_pre_skew_removal", draft_line=cc["line"],
            draft_says=cc["raw"], draft_value=cc["value"]["corr"],
            draft_n=cc["value"]["n"], draft_describes=cc["value"]["described_as"],
            as_run_pearson=bbqa["pearson"], as_run_n=bbqa["n"],
            v9_pearson=bbq9["pearson"], v9_spearman=bbq9["spearman"], v9_n=bbq9["n"],
            v9_pearson_ci=bbq9["pearson_ci"],
            delta_v9_minus_draft=(bbq9["pearson"] - cc["value"]["corr"]),
            as_run_reproduces_draft=abs(bbqa["pearson"] - cc["value"]["corr"]) < 5e-4,
            n_of_those_cells_that_are_held_out=corr_9[
                "v9: BBQ cells that are ALSO held-out under the frozen split"]["n"]))
    bm = draft.get("backfire_magnitude_claim")
    if bm:
        checks.append(dict(
            id="fig6.backfire_magnitude", draft_line=bm["line"], draft_says=bm["raw"],
            draft_value=bm["value"],
            v9_mean_backfire_magnitude=back_9["mean_magnitude"],
            v9_max_backfire_magnitude=back_9["max_magnitude"],
            v9_min_backfire_magnitude=back_9["min_magnitude"],
            v9_deepest_cell=back_9["deepest_cell"],
            v9_lowest_pre_skew_cells=back_9["lowest_pre_skew_cells"],
            as_run_mean_backfire_magnitude=back_a["mean_magnitude"],
            as_run_max_backfire_magnitude=back_a["max_magnitude"]))
    for key, got, exp in (
            ("fig6.calibration_backfire", draft.get("calibration_backfire_claim"),
             [pol_9.get("calibration_edit_all", {}).get("backfire_rate_edited"),
              pol_9.get("calibration_joint_gate", {}).get("backfire_rate_edited")]),
            ("fig6.calibration_utility", draft.get("calibration_utility_claim"),
             [pol_9.get("calibration_edit_all", {}).get("utility"),
              pol_9.get("calibration_joint_gate", {}).get("utility")]),
            ("fig6.rejection_count", draft.get("rejection_claim"),
             [pol_9.get("calibration_n_rejected"),
              pol_9.get("calibration_edit_all", {}).get("n_cells")]),
            ("fig6.sensitivity_n", draft.get("sensitivity_n_claim"),
             sens_9["n_held_out"]),
            ("fig6.sensitivity_gain", draft.get("sensitivity_gain_claim"),
             sens_9["pre_skew_gate_utility_gain"])):
        if got:
            checks.append(dict(id=key, draft_line=got["line"], draft_says=got["raw"],
                               draft_value=got["value"], v9_value=exp,
                               as_run_value=(
                                   [pol_a.get("calibration_edit_all", {}).get(
                                       "backfire_rate_edited"),
                                    pol_a.get("calibration_joint_gate", {}).get(
                                        "backfire_rate_edited")]
                                   if key == "fig6.calibration_backfire" else
                                   [pol_a.get("calibration_edit_all", {}).get("utility"),
                                    pol_a.get("calibration_joint_gate", {}).get("utility")]
                                   if key == "fig6.calibration_utility" else
                                   [pol_a.get("calibration_n_rejected"),
                                    pol_a.get("calibration_edit_all", {}).get("n_cells")]
                                   if key == "fig6.rejection_count" else
                                   sens_a["n_held_out"] if key == "fig6.sensitivity_n" else
                                   sens_a["pre_skew_gate_utility_gain"])))

    # ---- payload -----------------------------------------------------------
    payload = dict(
        artifact="v10_env_v9",
        generated_by="src/v10_regen_env.py",
        figure="Fig 6 — operating characterization (pre-skew vs removal)",
        gate="v9 (integer-item MMLU budget), removals read from results_v9/",
        definitions=dict(
            cell="v8_env.cell_table: (target, axis), role='self', mean over rows",
            budget_fail="removal nan/absent -> 0.0 (v10_common.z), row is kept",
            backfire="cell removal < 0 — v8_env.score's backfire_rate_edited and "
                     "v8_figures.py's 'backfire region: removal < 0' (lines 76, 128); "
                     "applied to the CELL mean, not to individual rows",
            utility="v8_env.score: sum(removal over EDITED cells) / n_cells "
                    "(abstention scores 0)",
            benchmark_family="v8_env.axis_family: bbq_* -> BBQ, crows_* -> CrowS, "
                             "ss_* -> StereoSet, else templated",
            split="v8_env.make_split: stratified by axis_family, seeded alternate "
                  "deal, granite/granite2/granite8 x occ_gender excluded",
            bootstrap=f"resample the CELL, {BOOT_N} draws, seed {BOOT_SEED}"),
        provenance=dict(
            edit_panels_source="v8_env.EDIT_PANELS",
            not_rescorable_source="v10_common.NOT_RESCORABLE",
            contrast_gap_source="results/t2x/corpora/{designer}|{axis}|s{seed}.json "
                                "-> diag.contrast_gap (v8_env.load_gap_index)",
            n_corpora_with_contrast_gap=len(gap),
            frozen_crosscheck=dict(n_checked=checked, n_disagree=len(disagree)),
            env_split_sha256=_sha256(SPLIT_FP),
            env_inventory_sha256=_sha256(INV_FP) if os.path.exists(INV_FP) else None,
            as_run_builder_matches_v8_env_build_inventory=fidelity,
            unjoined_as_run={f"{p}|{d}|{a}": c for (p, d, a), c in unj_mine.items()},
            unjoined_v9={f"{p}|{d}|{a}": c for (p, d, a), c in unj_9.items()}),
        gate_status=dict(
            v9_gate_modes=dict(collections.Counter(str(r["v9_gate_mode"]) for r in rows_9)),
            n_rows_v9_changed=sum(1 for r in rows_9 if r.get("v9_changed") is True),
            n_rows_v9_budget_fail=sum(1 for r in rows_9 if r["budget_fail"]),
            n_rows_as_run_budget_fail=sum(1 for r in rows_a if r["budget_fail"]),
            n_self_rows_v9_changed=sum(1 for r in rows_9
                                       if r["role"] == "self" and r.get("v9_changed") is True),
            note="every re-scorable ENV panel re-scored in 'integer' mode; no "
                 "UNMATCHED or UNGATED rows, so the v9 set has no partial coverage"),
        split=dict(
            frozen_artifact=os.path.relpath(SPLIT_FP, HERE),
            frozen_counts=dict(calibration=sp["n_calibration"],
                               held_out=sp["n_held_out"], excluded=len(sp["excluded"])),
            reproduced_from_v8_env_make_split=split_ok,
            reproduced_counts=dict(calibration=len(cal_r), held_out=len(hold_r),
                                   excluded=len(excl_r)),
            carried_to_v9=dict(collections.Counter(c["split"] for c in rec_9)),
            carried_to_as_run=dict(collections.Counter(c["split"] for c in rec_a)),
            naive_redeal_on_v9_population=dict(
                calibration=len(cal_9r), held_out=len(hold_9r), excluded=len(excl_9r),
                warning="NOT used: the split is frozen and hashed, so assignments "
                        "are carried by cell identity. Re-dealing the shrunken v9 "
                        "population changes assignments and would break the freeze."),
            excluded_rule=sp["excluded_rule"]),
        v9=dict(n_rows=len(rows_9), n_cells=len(cells_9), cells=rec_9, rows=rows_9),
        as_run=dict(n_rows=len(rows_a), n_cells=len(cells_a), cells=rec_a, rows=rows_a,
                    warning="AS-RUN, superseded by v9 wherever a panel is re-scorable; "
                            "emitted only so the caption can quantify the restriction"),
        restriction_loss=restriction,
        backfire=dict(v9=back_9, as_run=back_a),
        correlations=dict(v9=corr_9, as_run=corr_a,
                          row_level=dict(v9=row_corr(rows_9, "v9"),
                                         as_run=row_corr(rows_a, "as_run"))),
        policy=dict(cell_level=dict(v9=pol_9, as_run=pol_a),
                    designer_level=dict(v9=sens_9, as_run=sens_a)),
        fig6=dict(
            point_set="v9.cells (34 v9-scored (target,axis) self cells)",
            x="pre_skew", y="removal",
            marker_shape_by="axis_family",
            outline_by="split (calibration | held_out | excluded)",
            highlight="backfire == true",
            annotated_correlation=(
                "correlations.v9['v9: non-excluded cells'] — the 3 excluded "
                "granite x occ_gender cells are protocol-inoperable (100% "
                "budget-fail -> removal 0.0 at pre_skew ~ 0.85) and are plotted "
                "but are not part of the characterization; the all-cell value is "
                "also emitted so the caption can state either"),
            annotated_correlation_all_cells="correlations.v9['v9: all self cells']",
            caption_inputs=dict(
                n_cells_v9=len(cells_9), n_cells_as_run=len(cells_a),
                n_rows_v9=len(rows_9), n_rows_as_run=len(rows_a),
                n_backfire=back_9["n_backfire"],
                held_out_gate_degenerates_to_edit_all=all(
                    t["edit_rate"] == 1.0 for t in pol_9.get("held_out_table", [])),
                restriction_sentence=(
                    "Fig 6 is drawn over the %d cells whose removals survive the v9 "
                    "re-score; the %d as-run cells that depend on t2x/v1/x/x2 "
                    "cannot be re-scored, which drops %d of %d observation rows "
                    "and %d cells (both falcon)." % (
                        len(cells_9), len(cells_a),
                        len(rows_a) - len(rows_9), len(rows_a),
                        len(cells_a) - len(cells_9)))),
        ),
        draft_claims=draft,
        draft_checks=checks,
        blocked=[
            "results/t2x, results/v1, results/x, results/x2 have no alpha_trace and "
            "no re-scorable twin content beyond v6trace/{t2x,x2}; their boundary "
            "exposure is UNKNOWN, so they are dropped from the v9 set, never "
            "re-scored and never substituted.",
            "falcon|occ_gender and falcon|crows_socioeconomic exist ONLY in the v1 "
            "panel, so they have no v9 value at all — 2 of 36 cells, both held-out.",
            "The frozen calibration/held-out split was dealt over the 36 as-run "
            "cells; it cannot be re-derived from the 34 v9 cells, so assignments "
            "are carried by cell identity and the v9 held-out half is 14, not 16."])

    fp = jdump(_clean(payload), OUT_JSON)

    # ---- console -----------------------------------------------------------
    print("=" * 78)
    print("v10 ENV — Fig 6 source, v9-scored")
    print("=" * 78)
    print(f"panels kept (v9)   : {[l for _r, l in keep]}")
    print(f"panels dropped     : {[l for _r, l, _w in dropped]}")
    print(f"as-run builder == v8_env.build_inventory : {fidelity}")
    print(f"frozen split reproduced from v8_env.make_split : {split_ok} "
          f"({len(cal_r)}/{len(hold_r)}/{len(excl_r)})")
    print(f"env_split.json sha256 : {_sha256(SPLIT_FP)[:8]}…{_sha256(SPLIT_FP)[-6:]}")
    print(f"contrast_gap crosscheck: {checked} observations, {len(disagree)} disagreements")
    print()
    print(f"rows   as-run {len(rows_a):5d}  ->  v9 {len(rows_9):5d}   "
          f"(dropped {len(rows_a) - len(rows_9)})")
    print(f"cells  as-run {len(cells_a):5d}  ->  v9 {len(cells_9):5d}   "
          f"(lost {[c['cell'] for c in lost_cells]})")
    print(f"family as-run {restriction['family_counts_as_run']}")
    print(f"family v9     {restriction['family_counts_v9']}")
    print(f"split  carried to v9: {payload['split']['carried_to_v9']}")
    print(f"v9 rows changed by the gate: {payload['gate_status']['n_rows_v9_changed']}"
          f" (self rows: {payload['gate_status']['n_self_rows_v9_changed']})")
    print(f"max |Δpre_skew| under the restriction: "
          f"{restriction['max_abs_d_pre_skew']:+.4f}   "
          f"max |Δremoval|: {restriction['max_abs_d_removal']:+.4f}")
    print()
    print("BACKFIRE cells (removal < 0), v9:")
    for c in back_9["cells"]:
        print(f"   {c['cell']:34s} {c['split']:12s} pre={c['pre_skew']:+.4f} "
              f"rem={c['removal']:+.4f}")
    print(f"   n={back_9['n_backfire']}/{back_9['n_cells']}  "
          f"mean |backfire|={back_9['mean_magnitude']:.4f}  "
          f"max={back_9['max_magnitude']:.4f}")
    print()
    print("CORRELATIONS (pre_skew vs removal), v9 cells:")
    for lab, b in corr_9.items():
        if b["pearson"] is None:
            continue
        ci = b["pearson_ci"]
        cis = f" [{ci['lo']:+.3f}, {ci['hi']:+.3f}]" if ci else ""
        print(f"   {lab:62s} n={b['n']:3d} r={b['pearson']:+.4f}{cis} "
              f"rho={b['spearman']:+.4f}")
    print()
    print("AS-RUN comparison for the starred claim:")
    print(f"   BBQ cells as-run r={bbqa['pearson']:+.4f} (n={bbqa['n']})  ->  "
          f"v9 r={bbq9['pearson']:+.4f} (n={bbq9['n']})")
    print()
    print("DRAFT CHECKS:")
    for c in checks:
        print(f"   [{c['id']}] line {c['draft_line']}: {c['draft_says']}")
    print()
    print(f"wrote {fp}")
    return payload


if __name__ == "__main__":
    main()
