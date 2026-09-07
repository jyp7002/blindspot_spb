"""DEC analysis (experiments_v8) — the estimands and the adjudication.

Bootstrap resamples the target x axis CELL, never the seed: per-seed pooling is
anti-conservative and was the source of the v4 "self-advantage" artifact that v5
had to retract. 10k resamples, RNG seed 0, percentile CIs.

nan removal (no alpha passed the budget) -> 0.0, with the count reported. That
is the project's standing rule and it matters here: a condition that never
clears the budget must score 0, not be dropped, or the comparison silently
becomes a comparison over different cell sets.
"""
import os, sys, json, random, collections
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

CONDITIONS = ["C-ref", "C-a", "C-bottom", "C-b", "C-rand",
              "C-layershuf", "C-tensorshuf"]
LOPO = ["C-ref-no_q_proj", "C-ref-no_k_proj", "C-ref-no_v_proj", "C-ref-no_o_proj"]


def load(panel="v8dec"):
    fp = os.path.join(RESULTS, panel, "removal.jsonl")
    rows = []
    seen = {}
    for ln in open(fp):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        # resume can re-run a cell; keep the LAST occurrence
        seen[(r["target"], r["axis"], r["seed"], r["designer"], r["condition"])] = r
    for r in seen.values():
        rem = r.get("removal")
        fail = rem is None or (isinstance(rem, float) and rem != rem)
        rows.append(dict(r, removal=0.0 if fail else float(rem), budget_fail=fail))
    return rows


def cell_means(rows):
    """(target, axis) -> {condition: mean removal over seeds}, + fail counts."""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    fails = collections.Counter()
    for r in rows:
        acc[(r["target"], r["axis"])][r["condition"]].append(r["removal"])
        if r["budget_fail"]:
            fails[r["condition"]] += 1
    out = {}
    for cell, byc in acc.items():
        out[cell] = {c: float(np.mean(v)) for c, v in byc.items()}
        out[cell]["_n_seeds"] = max(len(v) for v in byc.values())
    return out, fails


# The registered DEC design is 10 in-envelope cells (v8 says 8-10). Adjudicating
# on fewer is not a weaker version of the experiment, it is a different one: with
# n=1 every bootstrap resample is the same cell, so the CI has zero width and
# "excludes 0" is guaranteed for any nonzero point estimate. A partially complete
# panel must therefore refuse to return a verdict rather than return a confident
# one.
MIN_CELLS_ADJUDICATE = 8
MIN_CELLS_CI = 3


def boot_paired(cells, a, b, n=10000, seed=0):
    """Paired (a - b) with the CELL as the resampling unit."""
    pairs = [(m[a], m[b]) for m in cells.values() if a in m and b in m]
    if not pairs:
        return None
    diffs = [x - y for x, y in pairs]
    point = float(np.mean(diffs))
    rng = random.Random(seed)
    k = len(diffs)
    if k < MIN_CELLS_CI:
        # A CI here would be an artifact of resampling one or two values.
        return dict(n_cells=k, point=point, lo=float("nan"), hi=float("nan"),
                    ci_valid=False)
    boots = []
    for _ in range(n):
        boots.append(np.mean([diffs[rng.randrange(k)] for _ in range(k)]))
    boots.sort()
    return dict(n_cells=k, point=point,
                lo=float(boots[int(0.025 * n)]), hi=float(boots[int(0.975 * n) - 1]),
                ci_valid=True)


def fmt(d):
    if d is None:
        return "n/a"
    if not d.get("ci_valid", True):
        return f"{d['point']:+.4f} [CI SUPPRESSED: n={d['n_cells']} < {MIN_CELLS_CI}]"
    star = "excludes 0" if (d["lo"] > 0 or d["hi"] < 0) else "covers 0"
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] (n={d['n_cells']}, {star})"


def main(panel="v8dec"):
    rows = load(panel)
    cells, fails = cell_means(rows)
    present = sorted({r["condition"] for r in rows})

    print(f"[dec] panel={panel}  rows={len(rows)}  cells={len(cells)}")
    print(f"[dec] budget-fail rows by condition: {dict(fails) or 'none'}")

    print("\n  PER-CONDITION MEAN REMOVAL (over cells)")
    means = {}
    for c in CONDITIONS + LOPO:
        vals = [m[c] for m in cells.values() if c in m]
        if vals:
            means[c] = float(np.mean(vals))
            print(f"    {c:18s} {means[c]:+.4f}   (n_cells={len(vals)})")

    print("\n  ESTIMANDS (paired, cell bootstrap)")
    est = {}
    for name, a, b in [("Delta_sign      (C-ref - C-b)", "C-ref", "C-b"),
                       ("Delta_selection (C-ref - C-a)", "C-ref", "C-a"),
                       ("Delta_bottom    (C-ref - C-bottom)", "C-ref", "C-bottom"),
                       ("Delta_null      (C-ref - C-rand)", "C-ref", "C-rand"),
                       ("Delta_location  (C-ref - C-layershuf)", "C-ref", "C-layershuf"),
                       ("Delta_tensor    (C-ref - C-tensorshuf)", "C-ref", "C-tensorshuf"),
                       ("C-a - C-rand  (does the true sign field help at random coords?)",
                        "C-a", "C-rand"),
                       ("C-a - C-bottom", "C-a", "C-bottom")]:
        d = boot_paired(cells, a, b)
        est[name] = d
        print(f"    {name:62s} {fmt(d)}")

    print("\n  LEAVE-ONE-PROJECTION-OUT (SUP1 2)")
    for c in LOPO:
        d = boot_paired(cells, "C-ref", c)
        if d:
            print(f"    C-ref - {c:20s} {fmt(d)}")
            est[f"lopo_{c}"] = d

    # ---- the adjudication ----
    sel = est.get("Delta_selection (C-ref - C-a)")
    verdict, rationale = "UNDETERMINED", "Delta_selection unavailable"
    if sel and (not sel.get("ci_valid", True)
                or sel["n_cells"] < MIN_CELLS_ADJUDICATE):
        verdict = "NOT ADJUDICATED"
        rationale = (f"panel incomplete: {sel['n_cells']} of the registered 10 cells "
                     f"present (minimum {MIN_CELLS_ADJUDICATE}). Delta_selection point "
                     f"estimate is {sel['point']:+.4f}, but no verdict is issued and "
                     f"no CI is quoted: with this few cells the bootstrap resamples "
                     f"the same cells and would report a spuriously tight interval.")
        print(f"\n  ADJUDICATION: {verdict}")
        print(f"    {rationale}")
    elif sel:
        covers0 = sel["lo"] <= 0 <= sel["hi"]
        ref = means.get("C-ref", float("nan"))
        frac = sel["point"] / ref if ref else float("nan")
        if covers0:
            verdict = "OUTCOME A (DARE-consistent)"
            rationale = ("Delta_selection CI covers 0: a random 1% support with the true "
                         "sign field matches the magnitude-selected 1%. The sign field is "
                         "distributed; magnitude only certifies coordinates.")
        else:
            verdict = "OUTCOME B (DARE-divergent)"
            rationale = (f"Delta_selection CI excludes 0 at {frac:.0%} of R(C-ref): the "
                         "behavioural edit concentrates in a magnitude-identifiable "
                         "sparse signed substructure.")
        print(f"\n  ADJUDICATION: {verdict}")
        print(f"    {rationale}")
        print(f"    Delta_selection = {fmt(sel)}; R(C-ref) = {ref:+.4f}")

    out = dict(artifact="v8_dec_analysis", panel=panel,
               n_rows=len(rows), n_cells=len(cells),
               budget_fail_by_condition=dict(fails),
               per_condition_mean=means,
               estimands={k: v for k, v in est.items()},
               per_cell={f"{t}|{a}": m for (t, a), m in cells.items()},
               adjudication=dict(verdict=verdict, rationale=rationale))
    fp = os.path.join(RESULTS, "v8", "dec_analysis.json")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    json.dump(out, open(fp, "w"), indent=1)
    print(f"\n[dec] wrote {fp}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v8dec")
