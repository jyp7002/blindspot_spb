"""MV-C — residual decomposition (experiments_v6 §2 MV-C).

Hypothesis (shared-residual noise): an elicited direction decomposes into a
GT-aligned component plus a residual. At small scale, same-family residuals are
*mutually correlated* (shared systematic error — co-blindness living in the
residual rather than in the bias profile); at 7-9B they shrink or decorrelate.
If true this resolves the G/O4 paradox: the effect is family-categorical yet
invisible to pairwise bias-profile correlation, because it lives in residual
space.

    v_role = <v_role, ĝ> ĝ  +  r_role          ĝ = gt / ||gt||

Estimands
  MV-C1  corr(r_self, r_sibling)  >  corr(r_self, r_cross)   at <=3B, CI excludes 0
  MV-C2  that same-vs-cross residual gap shrinks to CI-covers-0 at 7-9B
  MV-C3  ORD — NOT computed here (experiments_v6 §7 freezes the ingredient first)

Data: results/sketches/*.npy, filtered to the FROZEN protocol module set
(Tq+k+v+o). As with runs.jsonl, untagged sketches are a v2-era layer-placement
ablation and must not be pooled with the frozen protocol.

MEASUREMENT CAVEAT (state this wherever the numbers are used): the sketches are
20000-dim random projections of the weight delta, not the delta itself. Inner
products — and therefore cosines, projections and residuals — are preserved only
up to Johnson-Lindenstrauss error. The `random` (sign-shuffle) role is carried
through every statistic as the chance floor, so a residual correlation is only
meaningful relative to it.

SCOPE: the frozen-protocol sketches cover llama and qwen at the <=3B tier ONLY.
There are no 7-9B sketches, so MV-C2 (the attenuates-with-scale half, and the F2
leg of the three-way filter) is NOT computable from cache — it needs a direction
-extraction run on the t2x targets. The alpha_trace records collateral, not
directions, so the completed t2x panel does not supply it.

Usage:  python src/v6_mvc_analyze.py [--boot 10000] [--seed 0] [--no-write]
"""
import os, json, glob, argparse, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(REPO, "results")
MODULE_TAG = "Tq+k+v+o"


def load_sketches():
    """Frozen-protocol sketches keyed (target, axis, seed) -> {role: vector}."""
    out = collections.defaultdict(dict)
    for fp in glob.glob(os.path.join(RES, "sketches", "*.npy")):
        p = os.path.basename(fp)[:-4].split("|")
        if len(p) != 7 or p[4] != MODULE_TAG:
            continue                      # v2-era ablation, different edit
        target, seed, origin, axis, _, signal, role = p
        out[(target, axis, int(seed))][role] = fp
    return out


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def residual(v, g):
    """v minus its component along the GT direction."""
    gh = unit(g)
    return v - float(v @ gh) * gh


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def boot_ci(vals, rng, n_boot):
    v = np.asarray([x for x in vals if x is not None], float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    d = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return float(np.mean(v)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt(m, lo, hi):
    return f"{m:+.4f}  95%CI=[{lo:+.3f}, {hi:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    sk = load_sketches()
    usable = {k: v for k, v in sk.items() if "gt" in v and "self" in v}
    print(f"frozen-protocol sketches ({MODULE_TAG}) with gt+self: {len(usable)} (target,axis,seed)")
    tiers = collections.Counter(k[0] for k in usable)
    print(f"  targets: {dict(tiers)}   <-- <=3B tier only; no 7-9B sketches exist")
    print("  NOTE 20000-dim random projections: cosines are JL-approximate; the")
    print("       sign-shuffle `random` role is carried as the chance floor.\n")

    # ---- per-(target,axis,seed) residual geometry
    recs = []
    for (tgt, ax, seed), roles in sorted(usable.items()):
        g = np.load(roles["gt"])
        row = dict(target=tgt, axis=ax, seed=seed)
        vecs = {}
        for r in ("self", "sibling", "cross", "random"):
            if r in roles:
                v = np.load(roles[r])
                vecs[r] = v
                row[f"{r}_gt_cos"] = cos(v, g)
                res = residual(v, g)
                row[f"{r}_resid_frac"] = float(np.linalg.norm(res) / max(np.linalg.norm(v), 1e-12))
        # residual-space correlations
        res = {r: residual(v, g) for r, v in vecs.items()}
        if "self" in res and "sibling" in res:
            row["rcorr_same"] = cos(res["self"], res["sibling"])      # same family
        if "self" in res and "cross" in res:
            row["rcorr_cross"] = cos(res["self"], res["cross"])       # cross family
        if "self" in res and "random" in res:
            row["rcorr_null"] = cos(res["self"], res["random"])       # chance floor
        recs.append(row)

    # ---- MV-C1: same vs cross residual correlation, bootstrap over target x axis CELLS
    cells = collections.defaultdict(list)
    for r in recs:
        cells[(r["target"], r["axis"])].append(r)
    print("=" * 78)
    print("MV-C1 — residual correlation, same-family vs cross-family (<=3B)")
    print("=" * 78)
    print(f"  {'cell':34s} {'same':>8s} {'cross':>8s} {'null':>8s} {'same-cross':>11s}  n")
    diffs, out_cells = [], []
    for (tgt, ax), rs in sorted(cells.items()):
        sm = [r["rcorr_same"] for r in rs if "rcorr_same" in r]
        cr = [r["rcorr_cross"] for r in rs if "rcorr_cross" in r]
        nl = [r["rcorr_null"] for r in rs if "rcorr_null" in r]
        if not sm or not cr:
            continue
        m_s, m_c = float(np.mean(sm)), float(np.mean(cr))
        m_n = float(np.mean(nl)) if nl else float("nan")
        diffs.append(m_s - m_c)
        out_cells.append(dict(cell=f"{tgt}|{ax}", same=m_s, cross=m_c, null=m_n,
                              same_minus_cross=m_s - m_c, n_seed=len(sm)))
        print(f"  {tgt + '|' + ax:34s} {m_s:+8.3f} {m_c:+8.3f} {m_n:+8.3f} {m_s - m_c:+11.3f}  {len(sm)}")
    if diffs:
        m, lo, hi = boot_ci(diffs, rng, a.boot)
        verdict = ("MV-C1 PASSES (same > cross, CI excludes 0)" if lo > 0 else
                   "MV-C1 FAILS (CI covers 0)" if lo < 0 < hi else
                   "MV-C1 FAILS — cross > same (CI wholly negative)")
        print(f"\n  same - cross residual correlation: {fmt(m, lo, hi)}")
        print(f"  -> {verdict}")
        if len(diffs) < 3:
            print("  !! fewer than 3 cells — CI is degenerate, do not quote it")
    else:
        m = lo = hi = float("nan"); verdict = "no usable cells"

    # ---- residual magnitude and GT alignment by role
    print("\n" + "=" * 78)
    print("Residual fraction ||r||/||v||  and GT alignment cos(v, gt), by role")
    print("=" * 78)
    role_stats = {}
    print(f"  {'role':10s} {'resid_frac':>12s} {'gt_cos':>22s}  n")
    for r in ("self", "sibling", "cross", "random"):
        rf = [x[f"{r}_resid_frac"] for x in recs if f"{r}_resid_frac" in x]
        gc = [x[f"{r}_gt_cos"] for x in recs if f"{r}_gt_cos" in x]
        if not rf:
            continue
        mf, lof, hif = boot_ci(rf, rng, a.boot)
        mg, log_, hig = boot_ci(gc, rng, a.boot)
        role_stats[r] = dict(resid_frac=[mf, lof, hif], gt_cos=[mg, log_, hig], n=len(rf))
        print(f"  {r:10s} {mf:12.4f} {fmt(mg, log_, hig):>22s}  {len(rf)}")
    print("\n  A residual fraction ~1.0 means the elicited direction is essentially")
    print("  orthogonal to ground truth — the decomposition is then almost all residual,")
    print("  and 'residual correlation' is close to raw direction correlation.")

    print("\nMV-C2 (7-9B leg): NOT COMPUTABLE from cache — no frozen-protocol sketches")
    print("  above 3B exist. The t2x alpha_trace records collateral, not directions, so")
    print("  the completed 7-9B panel does not supply it. Filling MV-C2 needs a")
    print("  direction-extraction pass over the t2x targets.")
    print("\nMV-C3 / ORD: ingredient written, correlation NOT run (experiments_v6 §7).")

    if not a.no_write:
        os.makedirs(os.path.join(RES, "v6"), exist_ok=True)
        fp = os.path.join(RES, "v6", "mvc_analysis.json")
        json.dump(dict(module_tag=MODULE_TAG, n_cells_seedlevel=len(recs),
                       targets=dict(tiers),
                       caveat=("20000-dim random-projection sketches; cosines are "
                               "JL-approximate. `random` role is the chance floor. "
                               "<=3B tier only — MV-C2 not computable from cache."),
                       mv_c1=dict(same_minus_cross=[m, lo, hi], verdict=verdict,
                                  per_cell=out_cells),
                       role_stats=role_stats, per_seed=recs,
                       boot=a.boot, boot_seed=a.seed), open(fp, "w"), indent=2)
        print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
