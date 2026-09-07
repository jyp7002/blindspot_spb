"""ENV (experiments_v8) — operating-envelope inventory.

STAGE 1 ONLY: build the inventory and the frozen calibration/held-out split.
Threshold search lives in `calibrate()` and MUST NOT be run before the split
file exists on disk (§ENV: "assignment frozen before any threshold is chosen").

Eligibility (frozen protocol, verified in colab_t2t4.py): every panel below is
produced by panel_run / t2x_run / _run_t2_seed, all of which call
train_task_vector(..., rank=16, targets=ATTN, steps=250, lr=1e-4) and
binarize(..., "per_tensor", 0.0, seed) over alphas (2,4,8,16). Baseline panels
(steer, sentdebias, inlp, dpo, ste), null panels and the quarantined
steer_* panels are NOT edit panels and are excluded by construction.

contrast_gap: read from the corpus file the run actually consumed
(results/t2x/corpora/{designer}|{axis}|s{seed}.json -> diag.contrast_gap), with
the file's md5 recorded. contrast_gap_frozen.json covers only occ_gender and
crows_socioeconomic (frozen 2026-07-29, before the BBQ/StereoSet runs); values
for those two axes are cross-checked against it and any disagreement is a hard
error, never a silent preference.
"""
import os, json, glob, hashlib, collections

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
CORPORA = os.path.join(RESULTS, "t2x", "corpora")
OUT_DIR = os.path.join(RESULTS, "v8")

# (path relative to results/, panel label) — real-edit, frozen-protocol panels only
EDIT_PANELS = [
    ("t2x/removal.jsonl",            "t2x"),
    ("v1/removal.jsonl",             "v1"),
    ("x/removal.jsonl",              "x"),
    ("x2/removal.jsonl",             "x2"),
    ("v6trace/t2x/removal.jsonl",    "v6trace_t2x"),
    ("v6trace/x2/removal.jsonl",     "v6trace_x2"),
    ("v6trace/mvb/removal.jsonl",    "mvb"),
    ("v6trace/fxg/removal.jsonl",    "fxg"),
    ("v6trace/ax/removal.jsonl",     "ax"),
    ("v6trace/axes/removal.jsonl",   "axes"),
    ("v6trace/ss/removal.jsonl",     "ss"),
    ("v6trace_3b/small/removal.jsonl", "v6trace_3b_small"),
]


def corpus_fp(designer, axis, seed):
    return os.path.join(CORPORA, f"{designer}|{axis}|s{seed}.json")


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_gap_index():
    """(designer, axis, seed) -> {contrast_gap, n_items, md5} from on-disk corpora."""
    idx = {}
    for fp in glob.glob(os.path.join(CORPORA, "*|*|s*.json")):
        base = os.path.basename(fp)[:-len(".json")]
        try:
            designer, axis, sseed = base.split("|")
            seed = int(sseed[1:])
        except ValueError:
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        diag = d.get("diag") or {}
        if "contrast_gap" not in diag:
            continue
        idx[(designer, axis, seed)] = dict(
            contrast_gap=float(diag["contrast_gap"]),
            n_items=int(diag.get("n_items", 0)),
            corpus_md5=_md5(fp), corpus_file=base + ".json")
    return idx


def crosscheck_frozen(idx):
    """The two frozen axes must agree with contrast_gap_frozen.json exactly."""
    fp = os.path.join(RESULTS, "v6", "contrast_gap_frozen.json")
    frozen = json.load(open(fp))["observations"]
    checked, disagree = 0, []
    for o in frozen:
        if not o.get("matches_loader_corpus"):
            continue
        k = (o["designer"], o["axis"], o["seed"])
        if k not in idx:
            continue
        checked += 1
        # The frozen artifact stores contrast_gap rounded to 3dp, so the widest
        # legitimate difference is half an ulp of that grid (5e-4), attained
        # exactly by half-way values such as qwen_large|occ_gender (0.3125 ->
        # 0.312 under round-half-to-even). The 1e-9 slack admits that boundary
        # case without admitting a real disagreement, which would be >= 1e-3.
        if abs(idx[k]["contrast_gap"] - o["contrast_gap"]) > 5e-4 + 1e-9:
            disagree.append((k, o["contrast_gap"], idx[k]["contrast_gap"]))
    return checked, disagree


def build_inventory():
    gap = load_gap_index()
    checked, disagree = crosscheck_frozen(gap)
    if disagree:
        raise SystemExit(
            f"[ENV] FATAL: {len(disagree)} contrast_gap disagreements vs frozen artifact: "
            + json.dumps(disagree[:5]))

    rows, unjoined = [], collections.Counter()
    for rel, panel in EDIT_PANELS:
        fp = os.path.join(RESULTS, rel)
        if not os.path.exists(fp):
            continue
        for ln in open(fp):
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            key = (r.get("designer"), r.get("axis"), r.get("seed"))
            g = gap.get(key)
            if g is None:
                unjoined[(panel, r.get("designer"), r.get("axis"))] += 1
                continue
            rem = r.get("removal")
            budget_fail = rem is None or (isinstance(rem, float) and rem != rem)
            rows.append(dict(
                panel=panel, target=r["target"], axis=r["axis"], seed=r["seed"],
                role=r.get("role"), designer=r["designer"],
                removal=0.0 if budget_fail else float(rem),
                budget_fail=bool(budget_fail),
                pre_skew=float(r["pre_skew"]),
                contrast_gap=g["contrast_gap"], corpus_md5=g["corpus_md5"],
                n_items=g["n_items"]))

    return rows, gap, unjoined, checked


def axis_family(axis):
    if axis.startswith("bbq_"):
        return "BBQ"
    if axis.startswith("crows_"):
        return "CrowS"
    if axis.startswith("ss_"):
        return "StereoSet"
    return "templated"


def cell_table(rows, role="self"):
    """Aggregate to the ENV decision unit: (target, axis) under the given role.

    Mean over seeds; nan->0 already applied per-row. A cell is emitted only if it
    has at least one seed. n_budget_fail is carried, never imputed away.
    """
    by = collections.defaultdict(list)
    for r in rows:
        if role is not None and r["role"] != role:
            continue
        by[(r["target"], r["axis"])].append(r)
    out = []
    for (tgt, ax), rs in sorted(by.items()):
        n = len(rs)
        out.append(dict(
            target=tgt, axis=ax, axis_family=axis_family(ax),
            n_obs=n, n_seed=len({r["seed"] for r in rs}),
            n_budget_fail=sum(r["budget_fail"] for r in rs),
            panels=sorted({r["panel"] for r in rs}),
            designers=sorted({r["designer"] for r in rs}),
            removal=sum(r["removal"] for r in rs) / n,
            pre_skew=sum(r["pre_skew"] for r in rs) / n,
            abs_pre_skew=sum(abs(r["pre_skew"]) for r in rs) / n,
            contrast_gap=sum(r["contrast_gap"] for r in rs) / n))
    return out


# ---------------------------------------------------------------- split ----
# Frozen exclusions, carried from PREREGISTRATION.md v8 §ENV.
#   granite x occ_gender: 100% budget-fail at EVERY alpha for EVERY designer
#   (prereg V3: low baseline ppl makes the ppl-ratio budget unreachable; MMLU
#   -0.080 = 4x the limit at the minimum alpha). This is target-side collateral
#   fragility, and it is NOT predictable from pre_skew or contrast_gap -- the two
#   features the gate is allowed to use. Including it would train the gate to
#   reject the highest-bias, highest-gap cells in the inventory for a reason the
#   gate cannot see. Excluded from calibration AND held-out, reported as its own
#   "protocol-inoperable" row so nothing is hidden.
EXCLUDE_CELLS = [("granite", "occ_gender"), ("granite2", "occ_gender"),
                 ("granite8", "occ_gender")]

SPLIT_SEED = 0
BACKFIRE_CAP = 0.10          # pre-set, §ENV.2


def make_split(cells, seed=SPLIT_SEED):
    """Deterministic ~50/50 split stratified by benchmark family.

    Assignment is a pure function of (cell identity, seed) and is computed and
    written to disk BEFORE any threshold is searched. It cannot be influenced by
    having seen the outcome column: within each stratum the cells are sorted by
    (target, axis) and dealt alternately after a seeded shuffle.
    """
    import random
    keep = [c for c in cells if (c["target"], c["axis"]) not in EXCLUDE_CELLS]
    excluded = [c for c in cells if (c["target"], c["axis"]) in EXCLUDE_CELLS]
    by_fam = collections.defaultdict(list)
    for c in keep:
        by_fam[c["axis_family"]].append(c)
    cal, hold = [], []
    for fam in sorted(by_fam):
        grp = sorted(by_fam[fam], key=lambda c: (c["target"], c["axis"]))
        random.Random(f"{seed}|{fam}").shuffle(grp)
        for i, c in enumerate(grp):
            (cal if i % 2 == 0 else hold).append(c)
    return cal, hold, excluded


# ------------------------------------------------------- policy + search ----

def apply_policy(cells, tau=None, gamma=None):
    """Edit a cell iff |pre_skew| >= tau (if set) AND contrast_gap >= gamma."""
    edited = []
    for c in cells:
        ok = True
        if tau is not None and c["abs_pre_skew"] < tau:
            ok = False
        if gamma is not None and c["contrast_gap"] < gamma:
            ok = False
        if ok:
            edited.append(c)
    return edited


def score(cells, tau=None, gamma=None):
    edited = apply_policy(cells, tau, gamma)
    n = len(cells)
    util = sum(c["removal"] for c in edited) / n if n else float("nan")
    back = (sum(1 for c in edited if c["removal"] < 0) / len(edited)
            if edited else 0.0)
    return dict(
        n_cells=n, n_edited=len(edited), edit_rate=len(edited) / n if n else float("nan"),
        mean_removal_edited=(sum(c["removal"] for c in edited) / len(edited)
                             if edited else float("nan")),
        backfire_rate_edited=back, utility=util)


def calibrate(cal, cap=BACKFIRE_CAP):
    """Maximise utility (abstain = 0) subject to backfire rate <= cap.

    Objective and cap are pre-registered; this function must not be run before
    the split file exists on disk.
    """
    taus = sorted({round(c["abs_pre_skew"], 6) for c in cal}) + [None]
    gammas = sorted({round(c["contrast_gap"], 6) for c in cal}) + [None]

    def best_over(grid):
        feas = [g for g in grid if g["backfire_rate_edited"] <= cap and g["n_edited"] > 0]
        pool = feas or grid
        # tie-break: higher utility, then fewer edits (prefer the more abstemious
        # gate), then the larger threshold -- fully determined, no judgement.
        return max(pool, key=lambda g: (g["utility"], -g["n_edited"],
                                        g["tau"] if g["tau"] is not None else -1,
                                        g["gamma"] if g["gamma"] is not None else -1))

    pre_grid = [dict(score(cal, tau=t), tau=t, gamma=None) for t in taus]
    gap_grid = [dict(score(cal, gamma=g), tau=None, gamma=g) for g in gammas]
    joint_grid = [dict(score(cal, tau=t, gamma=g), tau=t, gamma=g)
                  for t in taus for g in gammas]
    return dict(pre_skew_gate=best_over(pre_grid),
                contrast_gap_gate=best_over(gap_grid),
                joint_gate=best_over(joint_grid))


def _boot_ci(cells, stat, n=10000, seed=0):
    import random
    if not cells:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    m = len(cells)
    for _ in range(n):
        samp = [cells[rng.randrange(m)] for _ in range(m)]
        v = stat(samp)
        if v == v:
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def policy_table(cells, thresholds):
    """§ENV.3 held-out policy table, with bootstrap CIs over cells."""
    rows = []
    specs = [("edit-all", None, None),
             ("pre-skew gate", thresholds["pre_skew_gate"]["tau"], None),
             ("contrast-gap gate", None, thresholds["contrast_gap_gate"]["gamma"]),
             ("joint gate", thresholds["joint_gate"]["tau"],
              thresholds["joint_gate"]["gamma"])]
    for name, tau, gamma in specs:
        s = score(cells, tau, gamma)
        s.update(policy=name, tau=tau, gamma=gamma)
        s["utility_ci"] = _boot_ci(cells, lambda cc: score(cc, tau, gamma)["utility"])
        s["mean_removal_edited_ci"] = _boot_ci(
            cells, lambda cc: score(cc, tau, gamma)["mean_removal_edited"])
        rows.append(s)
    return rows


def run_env():
    """Stage 2. Requires the split file written by main()."""
    split_fp = os.path.join(OUT_DIR, "env_split.json")
    if not os.path.exists(split_fp):
        raise SystemExit("[ENV] refusing to calibrate: env_split.json does not exist. "
                         "Run the inventory stage first (§ENV.1 freeze rule).")
    sp = json.load(open(split_fp))
    cal, hold, excl = sp["calibration"], sp["held_out"], sp["excluded"]
    th = calibrate(cal)
    table = policy_table(hold, th)

    print(f"[ENV] calibration n={len(cal)}  held-out n={len(hold)}  excluded n={len(excl)}")
    print(f"[ENV] thresholds (chosen on calibration only):")
    for k, v in th.items():
        print(f"        {k:20s} tau={v['tau']}  gamma={v['gamma']}  "
              f"util={v['utility']:+.4f} backfire={v['backfire_rate_edited']:.2f}")
    print("\n  HELD-OUT POLICY TABLE")
    print(f"  {'policy':20s} {'mean rem (edited)':>19s} {'backfire':>9s} "
          f"{'edit rate':>10s} {'utility (abstain=0)':>22s}")
    for r in table:
        mr = r["mean_removal_edited"]
        ci = r["utility_ci"]
        print(f"  {r['policy']:20s} {mr:+19.4f} {r['backfire_rate_edited']:9.2f} "
              f"{r['edit_rate']:10.2f} "
              f"{r['utility']:+9.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]")

    out = dict(artifact="v8_env_policy", split_seed=SPLIT_SEED,
               backfire_cap=BACKFIRE_CAP, thresholds=th,
               held_out_table=table,
               excluded_cells=excl,
               n_cal=len(cal), n_held_out=len(hold))
    fp = os.path.join(OUT_DIR, "env_policy.json")
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\n[ENV] wrote {fp}")
    return out


def designer_cells(rows):
    """Registered SENSITIVITY unit: (target, axis, designer).

    The primary unit is (target, axis) under the self designer. This one asks the
    other deployment question -- "should I build the edit from THIS corpus for
    THIS target" -- and contrast_gap varies across designers, so it has
    discriminative power the self-only unit cannot have. Reported as sensitivity,
    never swapped in as primary.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["target"], r["axis"], r["designer"])].append(r)
    out = []
    for (tgt, ax, dn), rs in sorted(by.items()):
        n = len(rs)
        out.append(dict(
            target=tgt, axis=ax, designer=dn, axis_family=axis_family(ax),
            n_obs=n, n_budget_fail=sum(r["budget_fail"] for r in rs),
            removal=sum(r["removal"] for r in rs) / n,
            pre_skew=sum(r["pre_skew"] for r in rs) / n,
            abs_pre_skew=sum(abs(r["pre_skew"]) for r in rs) / n,
            contrast_gap=sum(r["contrast_gap"] for r in rs) / n))
    return out


def run_sensitivity():
    """All-roles (target, axis, designer) sensitivity, per the registered plan."""
    rows, _gap, _unj, _chk = build_inventory()
    cells = designer_cells(rows)
    cal, hold, excl = make_split(cells)
    th = calibrate(cal)
    table = policy_table(hold, th)
    print(f"\n[ENV-sensitivity] unit=(target,axis,designer)  "
          f"cal n={len(cal)}  held-out n={len(hold)}  excluded n={len(excl)}")
    for k, v in th.items():
        print(f"        {k:20s} tau={v['tau']}  gamma={v['gamma']}")
    print(f"  {'policy':20s} {'mean rem (edited)':>19s} {'backfire':>9s} "
          f"{'edit rate':>10s} {'utility':>26s}")
    for r in table:
        ci = r["utility_ci"]
        print(f"  {r['policy']:20s} {r['mean_removal_edited']:+19.4f} "
              f"{r['backfire_rate_edited']:9.2f} {r['edit_rate']:10.2f} "
              f"{r['utility']:+9.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    fp = os.path.join(OUT_DIR, "env_sensitivity.json")
    json.dump(dict(artifact="v8_env_sensitivity",
                   unit="(target,axis,designer)", thresholds=th,
                   n_cal=len(cal), n_held_out=len(hold),
                   held_out_table=table), open(fp, "w"), indent=1)
    print(f"[ENV-sensitivity] wrote {fp}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, gap, unjoined, checked = build_inventory()
    cells = cell_table(rows, role="self")

    print(f"[ENV] corpora with contrast_gap : {len(gap)}")
    print(f"[ENV] cross-checked vs frozen   : {checked} observations, 0 disagreements")
    print(f"[ENV] joined observation rows   : {len(rows)}")
    if unjoined:
        print(f"[ENV] UNJOINED (no corpus file) : {sum(unjoined.values())} rows")
        for (p, d, a), c in unjoined.most_common(12):
            print(f"        {p:18s} {str(d):14s} {a:22s} n={c}")
    print(f"[ENV] self-role cells           : {len(cells)}")
    fam = collections.Counter(c["axis_family"] for c in cells)
    print(f"[ENV] by benchmark family       : {dict(fam)}")

    print("\n  target        axis                    fam         n  pre_skew  gap      removal")
    for c in sorted(cells, key=lambda c: (-abs(c["pre_skew"]), c["target"])):
        print(f"  {c['target']:12s}  {c['axis']:22s}  {c['axis_family']:10s} "
              f"{c['n_obs']:2d}  {c['pre_skew']:+.3f}   {c['contrast_gap']:+.3f}   "
              f"{c['removal']:+.4f}")

    payload = dict(
        artifact="v8_env_inventory", stage="inventory-only-no-thresholds",
        n_rows=len(rows), n_cells=len(cells),
        frozen_crosscheck=dict(n_checked=checked, n_disagree=0),
        unjoined={f"{p}|{d}|{a}": c for (p, d, a), c in unjoined.items()},
        cells=cells, rows=rows)
    fp = os.path.join(OUT_DIR, "env_inventory.json")
    json.dump(payload, open(fp, "w"), indent=1)
    print(f"\n[ENV] wrote {fp}")

    # Freeze the split NOW, before any threshold is searched (§ENV.1).
    split_fp = os.path.join(OUT_DIR, "env_split.json")
    if os.path.exists(split_fp):
        print(f"[ENV] split already frozen at {split_fp} — left untouched")
        return
    cal, hold, excl = make_split(cells)
    json.dump(dict(artifact="v8_env_split", seed=SPLIT_SEED,
                   rule="stratified by axis_family, seeded alternate deal, "
                        "assignment is a pure function of (cell, seed)",
                   excluded_rule="granite x occ_gender: 100% budget-fail, "
                                 "prereg V3 collateral fragility, not visible to "
                                 "the (pre_skew, contrast_gap) gate",
                   n_calibration=len(cal), n_held_out=len(hold),
                   calibration=cal, held_out=hold, excluded=excl),
              open(split_fp, "w"), indent=1)
    fam_c = collections.Counter(c["axis_family"] for c in cal)
    fam_h = collections.Counter(c["axis_family"] for c in hold)
    print(f"[ENV] FROZE split -> {split_fp}")
    print(f"        calibration n={len(cal)} {dict(fam_c)}")
    print(f"        held-out    n={len(hold)} {dict(fam_h)}")
    print(f"        excluded    n={len(excl)} (protocol-inoperable)")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inventory"
    if cmd == "calibrate":
        run_env()
    elif cmd == "sensitivity":
        run_sensitivity()
    else:
        main()
