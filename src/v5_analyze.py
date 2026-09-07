"""Analyze the v5 closure panels — the abstract gate (V1), the T2 headline
(t2x, seed-filled), and the 3B->7B bridge arc (X).

One place, three measured artifact files, no hardcoded baseline constants
(experiments_v5 §1, R2-bug lesson). Every number CLOSE cites should come from
here so the paper text and the JSON agree.

Conventions carried from the existing pipeline:
  * budget-fail (removal == nan, i.e. no alpha stayed in the collateral budget)
    is counted as 0.0 achievable in-budget removal (experiments_v5 §1;
    t2x_analyze.py L66-68). n_nan is always reported so the zeros are visible.
  * per (target, axis) each role is aggregated as the MEAN over the seeds and
    designers present *right now* -- the script is correct while V1 seeds are
    still filling; it prints n_seed so partial fills are never mistaken for
    complete ones.
  * bootstrap CIs use a fixed RNG seed (default 0) so reruns are identical.

Estimands (design.md §5-6, experiments_v5 §V/§X, ABSTRACTS_PARKED selection rule):
  V1 gate : paired Δ = same_family_large - cross_family_large across the
            target×axis cells. CI covers 0 -> V1-a -> Abstract A;
            CI excludes 0 and negative -> V1-b -> Abstract B.
  t2x     : cross - self per axis (the "own best debiaser" quantity in the
            abstracts) and cross - same{self,sibling} (the family-level
            self-penalty), each with a bootstrap CI over the cells.
  X       : cross - self per (target, axis) with a CI over seeds, then stitched
            against the 7B point (t2x/v1) into a per-cell tier arc.

Usage:  python src/v5_analyze.py [--boot 10000] [--seed 0] [--no-write]
"""
import os, json, argparse
import numpy as np

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
AXES = ["occ_gender", "crows_socioeconomic"]


# ----------------------------- io helpers ---------------------------------

def load_panel(name):
    """Rows of results/<name>/removal.jsonl (empty list if absent)."""
    fp = os.path.join(RES, name, "removal.jsonl")
    if not os.path.exists(fp):
        return []
    return [json.loads(ln) for ln in open(fp) if ln.strip()]


def _z(x):
    """Budget-fail (nan) -> 0.0 achievable removal (experiments_v5 §1)."""
    return 0.0 if x is None or np.isnan(x) else float(x)


def cell_values(rows, target, axis, role):
    """The removal values for one (target, axis, role), nan->0, plus nan count.
    Averages nothing here -- returns the raw per (seed, designer) list so the
    caller controls aggregation."""
    vs = [r["removal"] for r in rows
          if r["target"] == target and r["axis"] == axis and r["role"] == role]
    n_nan = int(sum(1 for v in vs if v is None or np.isnan(v)))
    return [_z(v) for v in vs], n_nan


def seeds_for(rows, target, axis, role):
    return sorted({r["seed"] for r in rows
                   if r["target"] == target and r["axis"] == axis and r["role"] == role})


# ----------------------------- statistics ---------------------------------

def boot_ci(vals, rng, n_boot, stat=np.mean):
    """Percentile bootstrap CI of `stat` over `vals` (resampling the cells)."""
    v = np.asarray(vals, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    draws = [stat(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return float(stat(v)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fmt_ci(m, lo, hi):
    return f"{m:+.4f}  95%CI=[{lo:+.3f}, {hi:+.3f}]"


# ------------------------------- V1 gate ----------------------------------

def analyze_v1(rng, n_boot):
    rows = load_panel("v1")
    print("=" * 72)
    print("V1 — ABSTRACT GATE   Δ = same_family_large - cross_family_large")
    print("=" * 72)
    if not rows:
        print("  results/v1/removal.jsonl not found — skipping.\n")
        return None

    targets = sorted({r["target"] for r in rows})
    cells, out_cells = [], []
    selfsl = []                       # V1-c: self - same_large
    print(f"  {'target':7s} {'axis':20s} {'self':>7s} {'sameL':>7s} "
          f"{'crossμ':>7s} {'Δ(sL-cr)':>9s}  seeds")
    for tgt in targets:
        for ax in AXES:
            sl, sl_nan = cell_values(rows, tgt, ax, "same_large")
            se, se_nan = cell_values(rows, tgt, ax, "self")
            cr, cr_nan = cell_values(rows, tgt, ax, "cross")
            if not sl or not se or not cr:
                continue
            slm, sem, crm = np.mean(sl), np.mean(se), np.mean(cr)
            delta = slm - crm
            cells.append(delta)
            selfsl.append(sem - slm)
            ns = len(seeds_for(rows, tgt, ax, "self"))
            out_cells.append(dict(target=tgt, axis=ax, self=sem, same_large=slm,
                                  cross_mean=crm, delta_sL_minus_cross=delta,
                                  n_seed=ns, n_nan=sl_nan + se_nan + cr_nan))
            print(f"  {tgt:7s} {ax:20s} {sem:+7.3f} {slm:+7.3f} {crm:+7.3f} "
                  f"{delta:+9.3f}  {ns}")

    if not cells:
        print("  (no complete target×axis cells yet)\n")
        return None

    max_seeds = max(c["n_seed"] for c in out_cells)
    m, lo, hi = boot_ci(cells, rng, n_boot)
    covers0 = lo < 0 < hi
    verdict = "V1-a -> ABSTRACT A" if covers0 else \
              ("V1-b -> ABSTRACT B" if hi < 0 else "AMBIGUOUS (Δ CI wholly positive — same-family BETTER)")
    ms, ls, hs = boot_ci(selfsl, rng, n_boot)
    print(f"\n  paired Δ(same_large - cross_mean): {fmt_ci(m, lo, hi)}")
    print(f"  -> CI covers 0? {covers0}   VERDICT: {verdict}")
    print(f"  V1-c self - same_large (self identity-specific?): {fmt_ci(ms, ls, hs)}")
    if max_seeds < 3:
        print(f"  ⚠  gate is under-powered: max {max_seeds}/3 seeds present per cell — "
              f"do not freeze the abstract until 3 seeds land (experiments_v5 §V1).")
    print()
    return dict(cells=out_cells, delta_mean=m, delta_ci=[lo, hi], covers_zero=covers0,
                verdict=verdict, selfsl_mean=ms, selfsl_ci=[ls, hs], max_seeds=max_seeds)


# ------------------------------- t2x headline -----------------------------

def analyze_t2x(rng, n_boot):
    rows = load_panel("t2x")
    print("=" * 72)
    print("t2x — T2 HEADLINE (seed-filled)   cross - self  and  cross - same{self,sibling}")
    print("=" * 72)
    if not rows:
        print("  results/t2x/removal.jsonl not found — skipping.\n")
        return None

    targets = sorted({r["target"] for r in rows})
    out = {}
    for ax in AXES:
        cs_cells, cm_cells, rec = [], [], []
        for tgt in targets:
            se, se_nan = cell_values(rows, tgt, ax, "self")
            sib, sib_nan = cell_values(rows, tgt, ax, "sibling")
            cr, cr_nan = cell_values(rows, tgt, ax, "cross")
            if not se or not cr:
                continue
            sem, crm = np.mean(se), np.mean(cr)
            same = np.mean(se + sib) if sib else sem      # family = self (+ sibling if present)
            cs_cells.append(crm - sem)                    # >0 self-penalty; <0 self-advantage
            cm_cells.append(crm - same)
            rec.append((tgt, sem, same, crm, crm - sem, crm - same, se_nan + sib_nan + cr_nan))
        if not rec:
            print(f"  [{ax}] (no complete targets yet)\n"); continue
        print(f"  [{ax}]")
        print(f"    {'target':7s} {'self':>7s} {'same':>7s} {'cross':>7s} "
              f"{'cr-self':>8s} {'cr-same':>8s}  nan")
        for tgt, sem, same, crm, csv, cmv, nn in rec:
            print(f"    {tgt:7s} {sem:+7.3f} {same:+7.3f} {crm:+7.3f} "
                  f"{csv:+8.3f} {cmv:+8.3f}  {nn}")
        cm_cs, lo_cs, hi_cs = boot_ci(cs_cells, rng, n_boot)
        cm_cm, lo_cm, hi_cm = boot_ci(cm_cells, rng, n_boot)
        adv = hi_cs < 0
        print(f"    cross - self : {fmt_ci(cm_cs, lo_cs, hi_cs)}  "
              f"-> {'self-ADVANTAGE (excl 0)' if adv else 'tie (CI covers 0)' if lo_cs < 0 < hi_cs else 'self-PENALTY'}")
        print(f"    cross - same : {fmt_ci(cm_cm, lo_cm, hi_cm)}\n")
        out[ax] = dict(cross_minus_self=[cm_cs, lo_cs, hi_cs],
                       cross_minus_same=[cm_cm, lo_cm, hi_cm],
                       per_target=[dict(target=t, self=s, same=sa, cross=c,
                                        cross_minus_self=cs, cross_minus_same=cm, n_nan=nn)
                                   for (t, s, sa, c, cs, cm, nn) in rec])
    return out


# ------------------------------- X bridge arc -----------------------------

def _cross_minus_self_by_seed(rows, tgt, ax):
    """Per-seed (cross_mean - self) for one cell, nan->0."""
    out = []
    for s in seeds_for(rows, tgt, ax, "self"):
        se = [r["removal"] for r in rows
              if r["target"] == tgt and r["axis"] == ax and r["role"] == "self" and r["seed"] == s]
        cr = [r["removal"] for r in rows
              if r["target"] == tgt and r["axis"] == ax and r["role"] == "cross" and r["seed"] == s]
        if se and cr:
            out.append(np.mean([_z(x) for x in cr]) - _z(se[0]))
    return out


def _self_nan(rows, tgt, ax):
    vs = [r["removal"] for r in rows
          if r["target"] == tgt and r["axis"] == ax and r["role"] == "self"]
    return int(sum(1 for v in vs if v is None or np.isnan(v))), len(vs)


def analyze_x(rng, n_boot, t2x_rows, v1_rows):
    rows = load_panel("x") + load_panel("x2")   # x = qwen/llama/falcon 3B; x2 = gemma/phi/granite bridge
    print("=" * 72)
    print("X — 3B/BRIDGE TIER (x + x2)   cross - self (per seed), stitched to the 7B point")
    print("=" * 72)
    if not rows:
        print("  results/x/removal.jsonl not found — skipping.\n")
        return None

    def sevenB(tgt, ax):
        """cross - self at 7B for the same cell: prefer t2x, fall back to v1."""
        for src in (t2x_rows, v1_rows):
            se, _ = cell_values(src, tgt, ax, "self")
            cr, _ = cell_values(src, tgt, ax, "cross")
            if se and cr:
                return np.mean(cr) - np.mean(se)
        return float("nan")

    # size (B) per bridge family, for arc ordering
    SIZE = {"olmo": 1.5, "granite": 2.5, "gemma": 2.6, "qwen": 3.1, "llama": 3.2,
            "falcon": 3.2, "phi": 3.8}
    targets = sorted({r["target"] for r in rows}, key=lambda t: SIZE.get(t, 3.0))
    out = []
    print(f"  {'target':7s} {'~B':>4s} {'axis':20s} {'3B cr-self':>12s} {'95%CI':>18s} "
          f"{'7B cr-self':>11s}  {'arc':>13s}  self_nan")
    for tgt in targets:
        for ax in AXES:
            cs = _cross_minus_self_by_seed(rows, tgt, ax)
            if not cs:
                continue
            m, lo, hi = boot_ci(cs, rng, n_boot)
            seven = sevenB(tgt, ax)
            snan, sn = _self_nan(rows, tgt, ax)
            reg3 = "penalty" if lo > 0 else "self-adv" if hi < 0 else "tie"
            # a budget-fragile cell (self mostly nan->0) is not a real self-penalty
            frag = " FRAGILE" if snan >= max(1, sn - 1) else ""
            arc = f"{reg3}->{'?' if np.isnan(seven) else ('pen' if seven > 0.02 else 'adv' if seven < -0.02 else 'tie')}"
            print(f"  {tgt:7s} {SIZE.get(tgt, 3.0):>4.1f} {ax:20s} {m:>+12.3f} "
                  f"[{lo:+.3f},{hi:+.3f}] {seven:>+11.3f}  {arc:>13s}  {snan}/{sn}{frag}")
            out.append(dict(target=tgt, size_b=SIZE.get(tgt, 3.0), axis=ax,
                            x3b_cross_minus_self=m, x3b_ci=[lo, hi], n_seed=len(cs),
                            self_nan=snan, self_n=sn, fragile=bool(frag),
                            sevenB_cross_minus_self=None if np.isnan(seven) else seven,
                            regime_3b=reg3))
    print("\n  (positive => small-model self-penalty; negative => self-advantage.\n"
          "   Per experiments_v5 X2, report as bracketing — the crossover is target×axis-specific,\n"
          "   not a single threshold.)\n")
    return out


# --------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    v1 = analyze_v1(rng, a.boot)
    t2x = analyze_t2x(rng, a.boot)
    t2x_rows, v1_rows = load_panel("t2x"), load_panel("v1")
    x = analyze_x(rng, a.boot, t2x_rows, v1_rows)

    if not a.no_write:
        summary = dict(v1_gate=v1, t2x_headline=t2x, x_bridge=x,
                       boot=a.boot, boot_seed=a.seed)
        fp = os.path.join(RES, "v5_analysis.json")
        json.dump(summary, open(fp, "w"), indent=2)
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
