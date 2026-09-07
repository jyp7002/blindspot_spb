"""SUP1 (experiments_v8) — characterise the surviving 1% support.

Analysis-only; reads the .npz support dumps written by the DEC runner.

THE MANDATORY CORRECTION (§SUP1.3): at 1% density two independent random
supports already share ~1% of either set, so raw Jaccard is ~0.005 and looks
meaninglessly small whatever the truth is. Every overlap number here is reported
as FOLD-ENRICHMENT over a hypergeometric expectation, with raw Jaccard kept for
the appendix only.

Two nulls, because they answer different questions and only reporting one is
how a layer-concentration effect gets mis-sold as coordinate-level agreement:

  global null   : E|A n B| = k_A * k_B / N        (coordinates exchangeable
                  across the whole edit; enrichment therefore INCLUDES any
                  tendency of both supports to occupy the same layers)
  within-module : E|A n B| = sum_m k_A(m)k_B(m)/N(m)  (conditions on the
                  observed per-module sizes; enrichment is coordinate agreement
                  BEYOND layer/projection placement)

Signed agreement uses E|A n B|/2 as its null, i.e. the sign field is only
credited where it beats a coin flip on the coordinates that already overlap.
"""
import os, glob, json, itertools, collections
import numpy as np

from v8_support import load_support, parse_module


# ---------------- overlap statistics ----------------

def _pair_stats(mods_a, mods_b):
    shared = [k for k in mods_a if k in mods_b]
    kA = sum(len(mods_a[k][1]) for k in shared)
    kB = sum(len(mods_b[k][1]) for k in shared)
    N = sum(int(np.prod(mods_a[k][0])) for k in shared)

    inter, signed, exp_within = 0, 0, 0.0
    for k in shared:
        (shape_a, ia, sa), (shape_b, ib, sb) = mods_a[k], mods_b[k]
        if shape_a != shape_b:
            continue                     # shape-incompatible: not comparable
        n_m = int(np.prod(shape_a))
        common, ai, bi = np.intersect1d(ia, ib, assume_unique=True,
                                        return_indices=True)
        inter += common.size
        if common.size:
            signed += int((sa[ai] == sb[bi]).sum())
        exp_within += len(ia) * len(ib) / max(n_m, 1)

    exp_global = kA * kB / max(N, 1)
    union = kA + kB - inter
    return dict(
        k_a=kA, k_b=kB, n_params=N, intersection=int(inter),
        signed_agree=int(signed),
        jaccard=inter / max(union, 1),
        exp_global=exp_global, exp_within=exp_within,
        fold_global=inter / exp_global if exp_global > 0 else float("nan"),
        fold_within=inter / exp_within if exp_within > 0 else float("nan"),
        fold_signed_global=signed / (exp_global / 2) if exp_global > 0 else float("nan"),
        signed_frac_of_intersection=signed / inter if inter else float("nan"))


def _layer_vec(mods, n_layers=None):
    per = collections.Counter()
    tot = collections.Counter()
    for k, (shape, idx, _s) in mods.items():
        L, _p = parse_module(k)
        if L is None:
            continue
        per[L] += len(idx)
        tot[L] += int(np.prod(shape))
    if not per:
        return None
    n = (max(per) + 1) if n_layers is None else n_layers
    dens = np.zeros(n)
    for L in range(n):
        if tot[L]:
            dens[L] = per[L] / tot[L]
    return dens


def _layer_corr(a, b):
    if a is None or b is None or len(a) != len(b):
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------- per-dump descriptive structure ----------------

def describe(path):
    meta, mods = load_support(path)
    per_layer = collections.Counter()
    layer_tot = collections.Counter()
    per_proj = collections.Counter()
    proj_tot = collections.Counter()
    for k, (shape, idx, _s) in mods.items():
        L, proj = parse_module(k)
        n = int(np.prod(shape))
        per_proj[proj] += len(idx); proj_tot[proj] += n
        if L is not None:
            per_layer[L] += len(idx); layer_tot[L] += n

    n_layers = max(per_layer) + 1 if per_layer else 0
    dens = [per_layer[L] / layer_tot[L] if layer_tot[L] else 0.0
            for L in range(n_layers)]
    overall = meta["density"]
    thirds = {}
    if n_layers:
        b1, b2 = n_layers // 3, 2 * n_layers // 3
        for name, lo, hi in (("early", 0, b1), ("middle", b1, b2), ("late", b2, n_layers)):
            s = sum(per_layer[L] for L in range(lo, hi))
            t = sum(layer_tot[L] for L in range(lo, hi))
            thirds[name] = dict(share_of_support=s / max(meta["n_support"], 1),
                                density=s / max(t, 1),
                                enrichment=(s / max(t, 1)) / overall if overall else float("nan"))
    return dict(
        meta=meta, n_layers=n_layers,
        layer_density=dens,
        layer_enrichment=[d / overall if overall else float("nan") for d in dens],
        thirds=thirds,
        projection={p: dict(share_of_support=per_proj[p] / max(meta["n_support"], 1),
                            density=per_proj[p] / max(proj_tot[p], 1),
                            enrichment=(per_proj[p] / max(proj_tot[p], 1)) / overall
                            if overall else float("nan"))
                    for p in sorted(per_proj)})


# ---------------- dump discovery ----------------

def parse_name(path):
    base = os.path.basename(path)[:-len(".npz")]
    try:
        target, axis, sseed, designer = base.split("|")
        return dict(target=target, axis=axis, seed=int(sseed[1:]), designer=designer,
                    path=path)
    except ValueError:
        return None


def find_dumps(roots):
    out = []
    for r in roots:
        for p in sorted(glob.glob(os.path.join(r, "**", "*.npz"), recursive=True)):
            d = parse_name(p)
            if d:
                out.append(d)
    return out


def _compare(group_key, items, cache):
    """All within-group pairs, labelled by what differs."""
    rows = []
    for a, b in itertools.combinations(items, 2):
        ma = cache.setdefault(a["path"], load_support(a["path"])[1])
        mb = cache.setdefault(b["path"], load_support(b["path"])[1])
        st = _pair_stats(ma, mb)
        st.update(comparison=group_key,
                  a=f"{a['target']}|{a['axis']}|s{a['seed']}|{a['designer']}",
                  b=f"{b['target']}|{b['axis']}|s{b['seed']}|{b['designer']}",
                  layer_corr=_layer_corr(_layer_vec(ma), _layer_vec(mb)))
        rows.append(st)
    return rows


def analyse(roots, out_fp):
    dumps = find_dumps(roots)
    if not dumps:
        raise SystemExit(f"[SUP1] no support dumps found under {roots}")
    cache = {}

    structure = {}
    for d in dumps:
        key = f"{d['target']}|{d['axis']}|s{d['seed']}|{d['designer']}"
        structure[key] = describe(d["path"])

    # 3. seed stability: same target/axis/designer, different seed
    by = collections.defaultdict(list)
    for d in dumps:
        by[(d["target"], d["axis"], d["designer"])].append(d)
    seed_rows = []
    for k, items in by.items():
        if len(items) > 1:
            seed_rows += _compare("seed", items, cache)

    # 4. axis overlap: same target/designer/seed, different axis
    by = collections.defaultdict(list)
    for d in dumps:
        by[(d["target"], d["designer"], d["seed"])].append(d)
    axis_rows = []
    for k, items in by.items():
        if len({i["axis"] for i in items}) > 1:
            axis_rows += _compare("axis", items, cache)

    # 5. designer overlap: same target/axis/seed, different designer
    by = collections.defaultdict(list)
    for d in dumps:
        by[(d["target"], d["axis"], d["seed"])].append(d)
    des_rows = []
    for k, items in by.items():
        if len({i["designer"] for i in items}) > 1:
            des_rows += _compare("designer", items, cache)

    def summarise(rows):
        if not rows:
            return None
        f = lambda key: [r[key] for r in rows if r[key] == r[key]]
        return dict(n_pairs=len(rows),
                    jaccard_median=float(np.median(f("jaccard"))),
                    fold_global_median=float(np.median(f("fold_global"))),
                    fold_within_median=float(np.median(f("fold_within"))),
                    fold_signed_global_median=float(np.median(f("fold_signed_global"))),
                    signed_frac_median=float(np.median(f("signed_frac_of_intersection"))),
                    layer_corr_median=float(np.median(f("layer_corr"))))

    payload = dict(
        artifact="v8_sup1", n_dumps=len(dumps),
        null_definition=dict(
            global_="E|AnB| = k_A*k_B/N (includes layer placement)",
            within_module="E|AnB| = sum_m k_A(m)k_B(m)/N(m) (beyond placement)",
            signed="E = E|AnB|/2 (coin flip on already-overlapping coordinates)"),
        structure=structure,
        overlap=dict(
            seed=dict(summary=summarise(seed_rows), pairs=seed_rows),
            axis=dict(summary=summarise(axis_rows), pairs=axis_rows),
            designer=dict(summary=summarise(des_rows), pairs=des_rows)))
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    json.dump(payload, open(out_fp, "w"), indent=1)

    print(f"[SUP1] dumps={len(dumps)}")
    for name, rows in (("seed", seed_rows), ("axis", axis_rows), ("designer", des_rows)):
        s = summarise(rows)
        if s:
            print(f"[SUP1] {name:9s} n={s['n_pairs']:3d}  jaccard={s['jaccard_median']:.4f}  "
                  f"fold_global={s['fold_global_median']:.2f}x  "
                  f"fold_within={s['fold_within_median']:.2f}x  "
                  f"signed_frac={s['signed_frac_median']:.3f}  "
                  f"layer_r={s['layer_corr_median']:+.3f}")
    print(f"[SUP1] wrote {out_fp}")
    return payload


if __name__ == "__main__":
    import sys
    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = sys.argv[1:] or [os.path.join(HERE, "results", "v8dec")]
    analyse(roots, os.path.join(HERE, "results", "v8", "sup1_analysis.json"))
