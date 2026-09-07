"""MV-A analysis — the budget/collateral-curve probe (experiments_v6 §2 MV-A).

Consumes the per-alpha trace written by colab_t2t4.alpha_trace() (added
2026-07-29). Before that patch the alpha sweep kept only `best = max in-budget
removal` and discarded every alpha's collateral, so none of this was computable
without a full re-run.

MV-A's hypothesis: same-family elicited directions overlap the family-shared
*capability* subspace more than cross-family ones, so negating them costs more
MMLU/ppl per unit of bias removed, and the penalty is borne by the BUDGET GATE
rather than by removal capacity.

Estimands
  MV-A1  removal, and the penalty (cross - self / cross - same), recomputed at
         each budget level: strict -> x1.5 -> x2 -> unconstrained. Deliverable is
         the penalty-vs-budget CURVE per designer class, not a single point.
  MV-A2  collateral-per-unit-removal, as the EFFICIENT FRONTIER: the cheapest
         ppl_ratio among alphas that actually achieve a given removal. Plus the
         CEILING: the highest removal reachable in-budget vs at any
         non-degenerate alpha. (A designer can match cross-family on cost per
         unit removal yet still be unable to buy the last increment at any
         price — a ceiling effect, not a rate effect. The two are distinguished
         here because the qwen/occ slice showed exactly that pattern.)
  MV-A3  ORD is deliberately NOT computed here. experiments_v6 §7 requires the
         ingredient measure to be frozen before its bridge correlation is run;
         this module writes the ingredient and stops. See --write-ingredient.

Conventions carried from src/v5_analyze.py and experiments_v6 §1:
  * budget-fail (no alpha in budget) -> 0.0 achievable removal; n_fail reported.
  * bootstrap unit = the target x axis CELL. Never per-(target, seed) pooling
    (the v5 methodological note). Fixed RNG seed so reruns are identical.
  * no hardcoded baseline constants: the strict budget is read out of
    colab_t2t4.collateral_ok's own defaults, not retyped here.
  * pooled-only reporting is banned (§7): every estimand prints per cell.
  * alpha_trace.jsonl is append-only and a cell that died mid-sweep is re-traced
    on resume, so (target, axis, seed, role, designer, alpha) is NOT unique --
    rows are deduped keeping the LAST occurrence.

Usage:
  python src/v6_mva_analyze.py [--root results/v6trace] [--panel t2x]
                               [--boot 10000] [--seed 0] [--degen 1.5]
                               [--ref-removal 0.5] [--no-write]
"""
import os, re, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(REPO, "results")
AXES = ["occ_gender", "crows_socioeconomic"]
SAME_ROLES = ("self", "sibling")


# --------------------------- protocol constants ---------------------------

def strict_budget(src=os.path.join(REPO, "colab_t2t4.py")):
    """Read the frozen strict budget off collateral_ok's own signature.

    Anchoring on the runner rather than retyping 0.02/1.10 here means the
    analysis cannot silently drift from the protocol it is analyzing (R2-bug
    rule). Falls back to the documented values, loudly, if the signature moves.
    """
    try:
        m = re.search(r"def collateral_ok\(pre, post, dmmlu=([0-9.]+), pplr=([0-9.]+)\)",
                      open(src).read())
        if m:
            return float(m.group(1)), float(m.group(2)), "colab_t2t4.collateral_ok"
    except OSError:
        pass
    print("  WARN could not read collateral_ok defaults — falling back to 0.02/1.10")
    return 0.02, 1.10, "fallback (signature not found)"


def budget_levels(dmmlu, pplr):
    """strict -> x1.5 -> x2 -> unconstrained, scaling the TOLERANCE.

    The ppl budget is a ratio, so the thing scaled is the permitted excess over
    1.0 (1.10 -> 1.15 -> 1.20), not the ratio itself. The MMLU budget is a
    permitted drop and scales directly. This mapping is a frozen choice: record
    it in PREREGISTRATION.md before running MV-A1 for record.
    """
    return [("strict", dmmlu, pplr),
            ("x1.5", dmmlu * 1.5, 1.0 + (pplr - 1.0) * 1.5),
            ("x2", dmmlu * 2.0, 1.0 + (pplr - 1.0) * 2.0),
            ("unconstrained", float("inf"), float("inf"))]


# ------------------------------- io helpers -------------------------------

def load_trace(root, panel):
    """Rows of <root>/<panel>/alpha_trace.jsonl, deduped keeping the LAST
    occurrence of each (target, axis, seed, role, designer, alpha)."""
    fp = os.path.join(root, panel, "alpha_trace.jsonl")
    if not os.path.exists(fp):
        return [], fp, 0
    raw = [json.loads(ln) for ln in open(fp) if ln.strip()]
    keep = {}
    for r in raw:                     # later rows overwrite earlier ones
        keep[(r["target"], r["axis"], r["seed"], r["role"], r["designer"], r["alpha"])] = r
    return list(keep.values()), fp, len(raw) - len(keep)


def load_removal(root, panel):
    fp = os.path.join(root, panel, "removal.jsonl")
    if not os.path.exists(fp):
        return []
    return [json.loads(ln) for ln in open(fp) if ln.strip()]


def sweeps(rows):
    """Group the trace into individual alpha-sweeps, keyed by the unit that
    produced one curve: (target, axis, seed, role, designer)."""
    g = {}
    for r in rows:
        g.setdefault((r["target"], r["axis"], r["seed"], r["role"], r["designer"]), []).append(r)
    for k in g:
        g[k].sort(key=lambda x: x["alpha"])
    return g


# ------------------------------- statistics -------------------------------

def boot_ci(vals, rng, n_boot, stat=np.mean):
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    draws = [stat(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return float(stat(v)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def fmt_ci(m, lo, hi):
    return f"{m:+.4f}  95%CI=[{lo:+.3f}, {hi:+.3f}]"


def verdict(lo, hi):
    return "tie (covers 0)" if lo < 0 < hi else ("penalty" if lo > 0 else "self-advantage")


# --------------------------- per-sweep quantities -------------------------

def in_budget(r, dm, pr):
    return r["dmmlu"] <= dm and r["ppl_ratio"] <= pr


def removal_at(sweep, dm, pr):
    """Max achievable removal under a budget; None if no alpha qualifies
    (the caller applies the nan->0 policy and counts the failure)."""
    ok = [r["bias_reduction"] for r in sweep if in_budget(r, dm, pr)]
    return max(ok) if ok else None


def frontier(sweep, target_removal, degen):
    """Cheapest ppl_ratio among alphas that ACHIEVE >= target_removal.

    Not an interpolation: the alpha->removal curve is NOT monotone (removal
    peaks then degrades), so interpolating between removal-ordered neighbours
    mixes in failing high-alpha points and overstates cost. Degenerate alphas
    (ppl_ratio >= degen, i.e. a wrecked model rather than a debiased one) are
    excluded per §7.
    """
    ok = [r["ppl_ratio"] for r in sweep
          if r["bias_reduction"] >= target_removal and r["ppl_ratio"] < degen]
    return min(ok) if ok else None


def ceiling(sweep, dm, pr, degen):
    """(max removal in budget, max removal at any non-degenerate alpha)."""
    strict = removal_at(sweep, dm, pr)
    free = [r["bias_reduction"] for r in sweep if r["ppl_ratio"] < degen]
    return strict, (max(free) if free else None)


# ------------------------------ integrity ---------------------------------

def check_faithful(sw, removal_rows, dm, pr):
    """The trace must reproduce the removal.jsonl the runner wrote. If it does
    not, the trace is not a faithful record of the search and nothing below can
    be trusted."""
    rec = {(r["target"], r["axis"], r["seed"], r["role"], r["designer"]): r["removal"]
           for r in removal_rows}
    checked, bad = 0, []
    for k, s in sw.items():
        if k not in rec:
            continue
        got = removal_at(s, dm, pr)
        got = 0.0 if got is None else got
        want = rec[k]
        want = 0.0 if want is None or (isinstance(want, float) and np.isnan(want)) else want
        checked += 1
        if abs(got - want) > 1e-9:
            bad.append((k, want, got))
    return checked, bad


# -------------------------------- MV-A1 -----------------------------------

def mv_a1(sw, levels, rng, n_boot):
    print("=" * 78)
    print("MV-A1 — penalty vs BUDGET LEVEL   (cross - self, cross - same{self,sibling})")
    print("=" * 78)
    cells = sorted({(k[0], k[1]) for k in sw})
    out = {"levels": [], "cells": [c[0] + "|" + c[1] for c in cells]}

    for name, dm, pr in levels:
        per_cell_cs, per_cell_cm, rec = [], [], []
        for tgt, ax in cells:
            role_vals, fails, ns = {}, {}, {}
            for role in ("self", "sibling", "cross"):
                vs, nf = [], 0
                for k, s in sw.items():
                    if k[0] == tgt and k[1] == ax and k[3] == role:
                        v = removal_at(s, dm, pr)
                        if v is None:
                            nf += 1
                        vs.append(0.0 if v is None else v)      # nan -> 0
                role_vals[role] = float(np.mean(vs)) if vs else None
                fails[role], ns[role] = nf, len(vs)
            # Budget-fails are attributed PER ROLE, not summed. nan->0 pulls a
            # role's mean down, so a cell where only the cross sweeps fail looks
            # like a self-advantage that is really a floor artifact on cross.
            # Whether the failures sit on self or on cross is the whole question
            # MV-A asks, so the aggregate count would hide the answer.
            n_fail = sum(fails.values())
            se, sib, cr = role_vals["self"], role_vals["sibling"], role_vals["cross"]
            if se is None or cr is None:
                continue
            same = float(np.mean([x for x in (se, sib) if x is not None]))
            per_cell_cs.append(cr - se)
            per_cell_cm.append(cr - same)
            rec.append(dict(cell=f"{tgt}|{ax}", self=se, sibling=sib, same=same, cross=cr,
                            cross_minus_self=cr - se, cross_minus_same=cr - same,
                            n_budget_fail=n_fail, fails=fails, n_sweeps=ns))
        if not rec:
            continue
        m_cs, lo_cs, hi_cs = boot_ci(per_cell_cs, rng, n_boot)
        m_cm, lo_cm, hi_cm = boot_ci(per_cell_cm, rng, n_boot)
        bl = f"dmmlu<={dm:.3g} pplr<={pr:.3g}" if np.isfinite(dm) else "no gate"
        print(f"\n  [{name}]  {bl}")
        print(f"    {'cell':28s} {'self':>7s} {'sib':>7s} {'cross':>7s} "
              f"{'cr-self':>8s} {'cr-same':>8s}   budget-fails self/sib/cross")
        for r in rec:
            sib_s = f"{r['sibling']:+7.3f}" if r["sibling"] is not None else f"{'--':>7s}"
            f_ = r["fails"]; n_ = r["n_sweeps"]
            fs = f"{f_['self']}/{n_['self']} {f_['sibling']}/{n_['sibling']} {f_['cross']}/{n_['cross']}"
            print(f"    {r['cell']:28s} {r['self']:+7.3f} {sib_s} {r['cross']:+7.3f} "
                  f"{r['cross_minus_self']:+8.3f} {r['cross_minus_same']:+8.3f}   {fs}")
        print(f"    cross - self : {fmt_ci(m_cs, lo_cs, hi_cs)}  -> {verdict(lo_cs, hi_cs)}")
        print(f"    cross - same : {fmt_ci(m_cm, lo_cm, hi_cm)}  -> {verdict(lo_cm, hi_cm)}")
        out["levels"].append(dict(level=name, dmmlu=None if not np.isfinite(dm) else dm,
                                  pplr=None if not np.isfinite(pr) else pr,
                                  cross_minus_self=[m_cs, lo_cs, hi_cs],
                                  cross_minus_same=[m_cm, lo_cm, hi_cm],
                                  per_cell=rec))

    seq = [l["cross_minus_self"][0] for l in out["levels"]]
    if len(seq) > 1:
        mono = all(abs(seq[i + 1]) <= abs(seq[i]) + 1e-12 for i in range(len(seq) - 1))
        print(f"\n  |penalty| across budget levels: "
              f"{' -> '.join(f'{abs(x):.3f}' for x in seq)}")
        print(f"  MV-A1 'shrinks monotonically as budget relaxes': {'HOLDS' if mono else 'FAILS'}")
        if not mono:
            print("    NOTE the alpha->removal curve is itself non-monotone (removal peaks then\n"
                  "    degrades), so a relaxed budget can admit a WORSE best alpha. MV-A1's\n"
                  "    criterion as written in experiments_v6 assumes monotonicity; if this\n"
                  "    persists on the full panel the criterion needs restating before it is\n"
                  "    used as a pass/fail gate.")
        out["monotone"] = bool(mono)
    print()
    return out


# -------------------------------- MV-A2 -----------------------------------

def mv_a2(sw, dm, pr, degen, ref_levels, rng, n_boot):
    print("=" * 78)
    print("MV-A2 — collateral per unit removal (EFFICIENT FRONTIER) and the CEILING")
    print("=" * 78)
    cells = sorted({(k[0], k[1]) for k in sw})
    out = {"frontier": [], "ceiling": None}

    for r_ref in ref_levels:
        same_c, cross_c, self_c, rec = [], [], [], []
        for tgt, ax in cells:
            vals = {}
            for role in ("self", "sibling", "cross"):
                fs = [frontier(s, r_ref, degen) for k, s in sw.items()
                      if k[0] == tgt and k[1] == ax and k[3] == role]
                fs = [f for f in fs if f is not None]
                vals[role] = float(np.mean(fs)) if fs else None
            se, sib, cr = vals["self"], vals["sibling"], vals["cross"]
            if se is None or cr is None:
                continue
            same = float(np.mean([x for x in (se, sib) if x is not None]))
            self_c.append(se - cr); same_c.append(same - cr)
            rec.append(dict(cell=f"{tgt}|{ax}", self=se, sibling=sib, same=same, cross=cr,
                            self_minus_cross=se - cr, same_minus_cross=same - cr))
        if not rec:
            print(f"\n  [removal >= {r_ref}]  no cell reaches this level — skipped")
            continue
        m_s, lo_s, hi_s = boot_ci(self_c, rng, n_boot)
        m_m, lo_m, hi_m = boot_ci(same_c, rng, n_boot)
        print(f"\n  [removal >= {r_ref}]  cheapest ppl_ratio that achieves it "
              f"(degenerate pplr>={degen} excluded)")
        print(f"    {'cell':28s} {'self':>7s} {'sib':>7s} {'cross':>7s} "
              f"{'self-cr':>8s} {'same-cr':>8s}")
        for r in rec:
            sib_s = f"{r['sibling']:7.3f}" if r["sibling"] is not None else f"{'--':>7s}"
            print(f"    {r['cell']:28s} {r['self']:7.3f} {sib_s} {r['cross']:7.3f} "
                  f"{r['self_minus_cross']:+8.3f} {r['same_minus_cross']:+8.3f}")
        print(f"    self - cross : {fmt_ci(m_s, lo_s, hi_s)}  "
              f"(>0 = self pays more per unit removal)")
        print(f"    same - cross : {fmt_ci(m_m, lo_m, hi_m)}   <- MV-A2's stated quantity")
        out["frontier"].append(dict(ref_removal=r_ref,
                                    self_minus_cross=[m_s, lo_s, hi_s],
                                    same_minus_cross=[m_m, lo_m, hi_m], per_cell=rec))

    # ---- ceiling: reachable in budget vs reachable at any non-degenerate alpha
    print(f"\n  CEILING  (max removal in strict budget  vs  at any alpha with pplr<{degen})")
    print(f"    {'cell':28s} {'role':8s} {'in-budget':>10s} {'free':>8s} {'deficit':>8s}")
    defi, rec2 = {"self": [], "sibling": [], "cross": []}, []
    for tgt, ax in cells:
        for role in ("self", "sibling", "cross"):
            pairs = [ceiling(s, dm, pr, degen) for k, s in sw.items()
                     if k[0] == tgt and k[1] == ax and k[3] == role]
            pairs = [(0.0 if a is None else a, b) for a, b in pairs if b is not None]
            if not pairs:
                continue
            ib = float(np.mean([a for a, _ in pairs]))
            fr = float(np.mean([b for _, b in pairs]))
            defi[role].append(fr - ib)
            rec2.append(dict(cell=f"{tgt}|{ax}", role=role, in_budget=ib, free=fr, deficit=fr - ib))
            print(f"    {tgt + '|' + ax:28s} {role:8s} {ib:10.3f} {fr:8.3f} {fr - ib:8.3f}")
    ce = {}
    for role in ("self", "sibling", "cross"):
        if defi[role]:
            m, lo, hi = boot_ci(defi[role], rng, n_boot)
            ce[role] = [m, lo, hi]
            print(f"    mean ceiling deficit [{role:8s}]: {fmt_ci(m, lo, hi)}")
    if "self" in ce and "cross" in ce:
        pair = [a - b for a, b in zip(defi["self"], defi["cross"])]
        m, lo, hi = boot_ci(pair, rng, n_boot)
        print(f"    self - cross ceiling deficit : {fmt_ci(m, lo, hi)}  -> {verdict(lo, hi)}")
        print("      (>0 = self leaves more removal on the table that the budget forbids —\n"
              "       a CEILING effect. Read together with the frontier above: matching cost\n"
              "       per unit removal + a positive ceiling deficit = self cannot buy the last\n"
              "       increment at any price, rather than paying a higher rate throughout.)")
        ce["self_minus_cross"] = [m, lo, hi]
    out["ceiling"] = dict(per_cell=rec2, summary=ce)
    print()
    return out


# ------------------------- MV-A3 / ORD ingredient -------------------------

def ingredient(sw, dm, pr, degen, r_ref):
    """Per-cell collateral-entanglement score, for ORD — NOT correlated here.

    experiments_v6 §7: ingredient measures are computed and frozen BEFORE their
    bridge correlation is run. This returns the measure; running ORD is a
    separate, later step against the frozen file.
    """
    cells = sorted({(k[0], k[1]) for k in sw})
    out = []
    for tgt, ax in cells:
        row = dict(target=tgt, axis=ax)
        for role in ("self", "sibling", "cross"):
            fs = [frontier(s, r_ref, degen) for k, s in sw.items()
                  if k[0] == tgt and k[1] == ax and k[3] == role]
            fs = [f for f in fs if f is not None]
            cs = [ceiling(s, dm, pr, degen) for k, s in sw.items()
                  if k[0] == tgt and k[1] == ax and k[3] == role]
            cs = [(0.0 if a is None else a, b) for a, b in cs if b is not None]
            row[f"{role}_frontier_excess"] = (float(np.mean(fs)) - 1.0) if fs else None
            row[f"{role}_ceiling_deficit"] = (float(np.mean([b - a for a, b in cs]))
                                              if cs else None)
        out.append(row)
    return out


# --------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(RES, "v6trace"),
                    help="output root holding <panel>/alpha_trace.jsonl")
    ap.add_argument("--panel", default="t2x")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--degen", type=float, default=1.5,
                    help="ppl_ratio at/above which an alpha is a wrecked model, not a debiased "
                         "one; excluded from frontier/unconstrained (§7)")
    ap.add_argument("--ref-removal", type=float, nargs="*", default=[0.3, 0.4, 0.5, 0.6],
                    help="removal levels at which the frontier is evaluated")
    ap.add_argument("--write-ingredient", action="store_true",
                    help="also write the frozen ORD ingredient (does NOT run ORD)")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    dm, pr, src = strict_budget()
    rows, fp, ndup = load_trace(a.root, a.panel)
    print(f"trace   : {fp}")
    if not rows:
        print("  no alpha_trace.jsonl — run colab_t2t4 with the alpha_trace patch first.")
        return
    sw = sweeps(rows)
    n_cells = len({(k[0], k[1]) for k in sw})
    print(f"          {len(rows)} alpha evals in {len(sw)} sweeps over {n_cells} "
          f"target×axis cells" + (f"  ({ndup} duplicate rows dropped, kept last)" if ndup else ""))
    print(f"budget  : strict dmmlu<={dm} pplr<={pr}   (read from {src})")
    print(f"degen   : ppl_ratio >= {a.degen} excluded from frontier/unconstrained")

    rem = load_removal(a.root, a.panel)
    checked, bad = check_faithful(sw, rem, dm, pr)
    if bad:
        print(f"  ⚠ FAITHFULNESS: {len(bad)}/{checked} sweeps disagree with removal.jsonl:")
        for k, want, got in bad[:5]:
            print(f"      {k}  recorded={want:+.4f} recomputed={got:+.4f}")
        print("    the trace does not reproduce the runner's search — investigate before "
              "trusting anything below.")
    else:
        print(f"faithful: {checked}/{checked} traced sweeps reproduce removal.jsonl exactly")
    n_degen = sum(1 for r in rows if r["ppl_ratio"] >= a.degen)
    if n_degen:
        print(f"degen   : {n_degen}/{len(rows)} alpha evals are degenerate "
              f"(max pplr={max(r['ppl_ratio'] for r in rows):.2f})")

    # The bootstrap unit is the target x axis cell. With very few cells the
    # resample is degenerate -- every draw is nearly the same set, so the CI
    # collapses to a hairline around the point estimate and LOOKS significant.
    # That failure mode (a CI that is tight because the unit is wrong/too few,
    # not because the effect is precise) is exactly the v5 methodological bug.
    underpowered = n_cells < 3
    if underpowered:
        print()
        print("  " + "!" * 70)
        print(f"  !! ONLY {n_cells} target×axis CELL(S) — the cell bootstrap is DEGENERATE.")
        print("  !! Point estimates below are readable; the CIs are NOT. A hairline CI here")
        print("  !! means 'one cell resampled to itself', not a precise effect. Do not quote")
        print("  !! any interval until the panel has enough cells (the full t2x panel has 8).")
        print("  " + "!" * 70)
    print()

    a1 = mv_a1(sw, budget_levels(dm, pr), rng, a.boot)
    a2 = mv_a2(sw, dm, pr, a.degen, a.ref_removal, rng, a.boot)
    ing = ingredient(sw, dm, pr, a.degen, max(a.ref_removal))

    print("MV-A3 / ORD: ingredient computed but NOT correlated against the bridge cells.\n"
          "  experiments_v6 §7 requires the measure to be frozen first. Run ORD as a\n"
          "  separate step against the written ingredient file.\n")

    if not a.no_write:
        os.makedirs(os.path.join(RES, "v6"), exist_ok=True)
        summary = dict(source_trace=fp, panel=a.panel,
                       strict_budget=dict(dmmlu=dm, pplr=pr, source=src),
                       degen_pplr=a.degen, ref_removal=a.ref_removal,
                       n_alpha_evals=len(rows), n_sweeps=len(sw), n_cells=n_cells,
                       n_duplicate_rows_dropped=ndup,
                       faithful=dict(checked=checked, mismatches=len(bad)),
                       underpowered=underpowered,
                       underpowered_note=("fewer than 3 target×axis cells: the cell bootstrap is "
                                          "degenerate and the CIs in this file must not be quoted"
                                          if underpowered else None),
                       mv_a1=a1, mv_a2=a2, boot=a.boot, boot_seed=a.seed)
        tag = f"{os.path.basename(a.root.rstrip(os.sep))}_{a.panel}"
        out = os.path.join(RES, "v6", f"mva_analysis_{tag}.json")
        json.dump(summary, open(out, "w"), indent=2)
        print(f"wrote {out}")
        if a.write_ingredient:
            ip = os.path.join(RES, "v6", f"mva_ord_ingredient_{tag}.json")
            json.dump(dict(note="ORD ingredient — frozen before any bridge correlation "
                                "(experiments_v6 §7 no-peeking).",
                           ref_removal=max(a.ref_removal), degen_pplr=a.degen,
                           cells=ing), open(ip, "w"), indent=2)
            print(f"wrote {ip}")


if __name__ == "__main__":
    main()
