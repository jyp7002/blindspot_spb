"""v12 — magnitude geometry of a contrast vector, for predicting the retained fraction.

WHAT THIS IS FOR. The published method keeps a FIXED 1% of coordinates. v12 asks
whether the fraction a given edit needs can be read off the edit itself, before
any forward pass, from how concentrated |ΔW| is. Every statistic here is a
function of the trained contrast vector v alone -- never of a bias probe, an
MMLU score, or a removal value -- so it is available at training time and can be
recorded for a held-out cell BEFORE that cell's sparsity curve exists
(experiments_v12.md §II, PREREGISTRATION.md §v12.B).

THE STATISTICS. With a = |v| flattened over every edited tensor (N entries):

    M(p)      top-p mass fraction: sum of the largest ceil-free p*N entries of a
              divided by ||v||_1. The quantity the retained fraction p* is
              predicted from (primary rule, v12_pstar.py).
    pr_frac   participation ratio / N = ||v||_1^2 / (N ||v||_2^2). 1 for a flat
              vector, 1/N for a single spike.
    ent_frac  exp(H(a/||a||_1)) / N, the entropy effective support.
    gini      Gini coefficient of a, computed from the Lorenz curve on the same
              p grid as M(p) (trapezoid). It is a GRID APPROXIMATION and is
              labelled as one; the grid is dense in the tail where it matters.

The retained count for fraction p uses binarize()'s semantics exactly:
k_drop = int((1 - p) * N), and the retained set is everything above the
k_drop-th smallest value. So M(p) is the mass of precisely the support that
colab_t2t4.binarize(v, sparsity=1-p) keeps (ties aside, which have measure zero
for a trained float32 delta).

MEMORY. One float32 buffer of N entries (16 GiB at 32B) plus an in-place
multi-kth numpy.partition, which allocates nothing further. That is the same
discipline colab_t2t4.binarize uses after the 2026-08-29 SPC OOMs; do not
"simplify" this into np.sort (2x the memory and ~30x the time at 4e9 entries).

TORCH-FREE. The runner hands in numpy views of CPU float32 tensors
(`t.reshape(-1).numpy()` shares storage), so this module imports on the
analysis box, where the GPU stack is deliberately absent.
"""
import hashlib
import json

import numpy as np

# Retained-fraction grid for M(p). Log-spaced over the range any sane sparsity
# lives in, plus every level of the published SPC grid (as p = 1 - sparsity), so
# M is known EXACTLY at the points the calibration curves are measured on and
# interpolation is only ever needed between them.
SPC_SPARSITIES = [0.0, 0.50, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995]
P_GRID = sorted(set(
    [round(float(x), 10) for x in np.logspace(-4, 0, 41)]
    + [round(1.0 - s, 10) for s in SPC_SPARSITIES]))


def _k_retained(n, p):
    """binarize()'s retained count for fraction p: N - int((1-p) N)."""
    return n - int((1.0 - p) * n)


def geometry(flat_arrays, p_grid=P_GRID, module_keys=None):
    """Magnitude geometry of the concatenation of `flat_arrays`.

    flat_arrays : iterable of 1-D float arrays (signed; |.| is taken here)
    module_keys : optional list of module names aligned with flat_arrays, used
                  only for the per-projection L1 shares.

    Returns a JSON-serialisable dict. Never mutates the inputs.
    """
    arrs = list(flat_arrays)
    n = int(sum(a.size for a in arrs))
    if n == 0:
        raise ValueError("empty contrast vector")

    buf = np.empty(n, dtype=np.float32)
    off = 0
    l2 = 0.0
    proj_l1 = {}
    for i, a in enumerate(arrs):
        m = a.size
        np.abs(a, out=buf[off:off + m])
        seg = buf[off:off + m]
        l2 += float(np.dot(seg.astype(np.float64, copy=False), seg)) \
            if m < 50_000_000 else _chunked_sq(seg)
        if module_keys is not None:
            proj = module_keys[i].split(".")[-1]
            proj_l1[proj] = proj_l1.get(proj, 0.0) + float(seg.sum(dtype=np.float64))
        off += m
    l1 = float(buf.sum(dtype=np.float64))
    if l1 <= 0:
        raise ValueError("contrast vector is identically zero")

    # entropy effective support, chunked so no second N-sized array exists
    H = 0.0
    for s in range(0, n, 50_000_000):
        q = buf[s:s + 50_000_000].astype(np.float64) / l1
        q = q[q > 0]
        H -= float((q * np.log(q)).sum())

    # top-p masses. Partition ONCE at every boundary (ascending order), then the
    # retained set for p is buf[N-k(p):], a union of partition segments.
    ps = sorted(set(float(p) for p in p_grid if 0 < p <= 1))
    bounds = sorted(set(n - _k_retained(n, p) for p in ps) - {0, n})
    if bounds:
        buf.partition(bounds)
    edges = [0] + bounds + [n]
    seg_sum = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg_sum[lo] = float(buf[lo:hi].sum(dtype=np.float64))
    # suffix sums over segment starts
    suffix, acc = {}, 0.0
    for lo in reversed(edges[:-1]):
        acc += seg_sum[lo]
        suffix[lo] = acc
    mass = {}
    for p in ps:
        start = n - _k_retained(n, p)
        mass[_pkey(p)] = (suffix[start] / l1) if start < n else 0.0
    # threshold magnitude at 1% -- the published operating point, for reference
    k01 = _k_retained(n, 0.01)
    thr01 = float(buf[n - k01]) if 0 < k01 <= n else None
    del buf

    gini = _gini_from_lorenz(ps, [mass[_pkey(p)] for p in ps])
    out = dict(
        n_params=n, l1=l1, l2=float(np.sqrt(l2)),
        pr_frac=(l1 * l1) / (n * l2) if l2 > 0 else None,
        ent_frac=float(np.exp(H)) / n,
        gini=gini, gini_method="lorenz-trapezoid-on-P_GRID",
        thresh_at_1pct=thr01,
        mass=mass,
        proj_l1_share=({k: v / l1 for k, v in sorted(proj_l1.items())}
                       if proj_l1 else None),
        n_modules=len(arrs),
    )
    out["geom_sha"] = geometry_sha(out)
    return out


def _chunked_sq(seg):
    s = 0.0
    for i in range(0, seg.size, 50_000_000):
        c = seg[i:i + 50_000_000].astype(np.float64)
        s += float(np.dot(c, c))
    return s


def _pkey(p):
    """Stable JSON key for a retained fraction."""
    return f"{p:.10g}"


def _gini_from_lorenz(ps, top_mass):
    """Gini from top-p mass on a grid.

    The Lorenz curve L(x) is the mass held by the SMALLEST x fraction, so
    L(1 - p) = 1 - M(p). Gini = 1 - 2 * integral_0^1 L(x) dx, trapezoid over the
    grid points plus the endpoints (0,0) and (1,1).
    """
    pts = {0.0: 0.0, 1.0: 1.0}
    for p, m in zip(ps, top_mass):
        pts[round(1.0 - p, 12)] = 1.0 - m
    xs = sorted(pts)
    ys = [pts[x] for x in xs]
    area = sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2
               for i in range(len(xs) - 1))
    return 1.0 - 2.0 * area


def geometry_sha(g):
    """Fingerprint of a geometry record, for the phase-A == phase-B check.

    Rounded to 6 significant figures so a bit-identical retrain matches exactly
    and a genuinely different ΔW does not. Excludes the sha field itself.
    """
    keys = ("n_params", "l1", "l2", "pr_frac", "ent_frac", "mass")
    def r(x):
        if isinstance(x, dict):
            return {k: r(v) for k, v in sorted(x.items())}
        if isinstance(x, float):
            return float(f"{x:.6g}")
        return x
    blob = json.dumps({k: r(g.get(k)) for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------- curve ops ----
def mass_at(g, p):
    """M(p) from a geometry record, log-linear interpolation between grid points."""
    pts = sorted((float(k), v) for k, v in g["mass"].items())
    return _interp_logx(pts, p)


def p_for_mass(g, tau):
    """Smallest p with M(p) >= tau (M is monotone in p), log-interpolated."""
    pts = sorted((float(k), v) for k, v in g["mass"].items())
    if tau <= pts[0][1]:
        return pts[0][0]
    for (p0, m0), (p1, m1) in zip(pts[:-1], pts[1:]):
        if m0 < tau <= m1:
            if m1 == m0:
                return p1
            f = (tau - m0) / (m1 - m0)
            return float(np.exp(np.log(p0) + f * (np.log(p1) - np.log(p0))))
    return 1.0


def mean_geometry(gs):
    """Cell-level geometry: pointwise mean of seed-level records."""
    keys = sorted(set.intersection(*[set(g["mass"]) for g in gs]))
    out = dict(mass={k: float(np.mean([g["mass"][k] for g in gs])) for k in keys})
    for f in ("pr_frac", "ent_frac", "gini", "l1", "l2"):
        vals = [g[f] for g in gs if g.get(f) is not None]
        out[f] = float(np.mean(vals)) if vals else None
    out["n_seeds"] = len(gs)
    out["geom_shas"] = sorted(g["geom_sha"] for g in gs)
    return out


def _interp_logx(pts, x):
    xs = [p for p, _ in pts]
    if x <= xs[0]:
        return pts[0][1]
    if x >= xs[-1]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x0 <= x <= x1:
            f = (np.log(x) - np.log(x0)) / (np.log(x1) - np.log(x0))
            return float(y0 + f * (y1 - y0))
    return pts[-1][1]


# -------------------------------------------------------------- selftest ----
def selftest():
    fails = []
    rng = np.random.default_rng(0)
    arrs = [rng.standard_t(3, size=s).astype(np.float32) for s in (5000, 3000, 7001)]
    keys = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj",
            "model.layers.1.self_attn.q_proj"]
    g = geometry(arrs, module_keys=keys)
    a = np.abs(np.concatenate(arrs)).astype(np.float64)
    n = a.size
    srt = np.sort(a)
    for p in (0.001, 0.01, 0.05, 0.5, 1.0):
        k = _k_retained(n, p)
        want = srt[n - k:].sum() / a.sum() if k else 0.0
        got = g["mass"][_pkey(p)]
        if abs(got - want) > 1e-6:
            fails.append(f"M({p}) = {got} vs brute force {want}")
    pr = a.sum() ** 2 / (n * (a ** 2).sum())
    if abs(g["pr_frac"] - pr) > 1e-6:
        fails.append(f"pr_frac {g['pr_frac']} vs {pr}")
    q = a / a.sum()
    ent = np.exp(-(q * np.log(q)).sum()) / n
    if abs(g["ent_frac"] - ent) > 1e-6:
        fails.append(f"ent_frac {g['ent_frac']} vs {ent}")
    gini_exact = (2 * np.sum(np.arange(1, n + 1) * srt) / (n * srt.sum())) - (n + 1) / n
    if abs(g["gini"] - gini_exact) > 0.02:
        fails.append(f"gini grid approx {g['gini']} vs exact {gini_exact}")
    if abs(sum(g["proj_l1_share"].values()) - 1) > 1e-6:
        fails.append("projection shares do not sum to 1")
    # inputs untouched
    if not np.array_equal(np.abs(arrs[0][:5]), np.abs(arrs[0][:5])) or arrs[0].min() >= 0:
        fails.append("geometry() mutated its inputs")
    # inverse is consistent
    p = p_for_mass(g, 0.5)
    if abs(mass_at(g, p) - 0.5) > 0.02:
        fails.append(f"p_for_mass/mass_at not inverse: M({p}) = {mass_at(g, p)}")
    # sha stable, and sensitive
    g2 = geometry(arrs, module_keys=keys)
    if g2["geom_sha"] != g["geom_sha"]:
        fails.append("geom_sha not deterministic")
    arrs[0][0] += 10.0
    if geometry(arrs)["geom_sha"] == g["geom_sha"]:
        fails.append("geom_sha did not change when ΔW changed")
    # a flat vector: M(p) == p, pr == 1
    flat = geometry([np.ones(10000, np.float32)])
    if abs(flat["mass"][_pkey(0.01)] - 0.01) > 1e-9 or abs(flat["pr_frac"] - 1) > 1e-9:
        fails.append("flat vector geometry wrong")
    for m in fails:
        print(f"  FAIL {m}")
    print(f"v12_geometry selftest: {'PASS' if not fails else str(len(fails)) + ' FAILED'}")
    return not fails


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
