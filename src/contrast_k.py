"""S — contrast-K dose-response axes (experiments_v4 Stage 2, the clean CONFIRM-2).

Is removability monotone in the number of distinct discriminative contrasts,
holding content pool and item count fixed, on NATURAL bias?

Construction (natural CrowS, no injection, no frame manipulation):
  * harvest every CrowS minimal pair whose two members differ in a SINGLE
    token -> tag it by its (tokenA, tokenB) contrast, preserving CrowS polarity
    (congruent = stereotype-reinforcing member);
  * a K-axis pools the top-K most-frequent contrasts, subsampled to a MATCHED
    total item count across all K, with a disjoint probe/edit split.

Only the number of distinct contrasts varies. K=1 = one contrast (e.g.
black/white) across varied natural frames; K=20 = twenty different contrasts,
each appearing few times.

Prediction (S1/S2): exogenous removability ceiling declines monotonically in K;
K=1 clears the gate, K=20 does not.
"""
import functools, re, os, json
import numpy as np
import torch
import crows_axes as CA

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
AXIS_PREFIX = "ck_"     # ck_K1, ck_K2, ...
SRC_AXES = ["crows_race-color", "crows_socioeconomic", "crows_gender",
            "crows_religion", "crows_age", "crows_physical-appearance",
            "crows_sexual-orientation", "crows_disability", "crows_nationality"]
N_MATCHED = 120         # total items per K-axis (60/half); calibrated to K=20 supply


def is_ck_axis(axis):
    return isinstance(axis, str) and axis.startswith(AXIS_PREFIX)


def _single_diff(a, b):
    aw, bw = a.split(), b.split()
    if len(aw) != len(bw):
        return None
    d = [(x, y) for x, y in zip(aw, bw) if x != y]
    if len(d) != 1:
        return None
    return (d[0][0].strip(".,'\"").lower(), d[0][1].strip(".,'\"").lower())


@functools.lru_cache(maxsize=1)
def _harvest():
    """All single-token CrowS pairs, grouped by (sorted) contrast."""
    from collections import defaultdict
    groups = defaultdict(list)   # contrast_key -> list of (cong, incong)
    for ax in SRC_AXES:
        try:
            p, e = CA.load_axis(ax)
        except Exception:
            continue
        for c, i, _k in p + e:
            d = _single_diff(c, i)
            if d and d[0] != d[1] and len(d[0]) > 1 and len(d[1]) > 1:
                groups[tuple(sorted(d))].append((c, i))
    # rank contrasts by supply
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


@functools.lru_cache(maxsize=32)
def load_axis(axis, seed=0, probe_frac=0.5):
    K = int(axis[len(AXIS_PREFIX) + 1:])   # ck_K5 -> 5
    ranked = _harvest()
    rng = np.random.default_rng(seed)
    per = max(1, N_MATCHED // K)
    items = []
    for ci in range(min(K, len(ranked))):
        _key, pairs = ranked[ci]
        idx = rng.permutation(len(pairs))[:per]
        for j in idx:
            c, i = pairs[j]
            items.append((c, i, f"K{K}:{ci}:{j}"))
    rng.shuffle(items)
    items = items[:N_MATCHED]
    cut = int(len(items) * probe_frac)
    return items[:cut], items[cut:]


def n_contrasts_available():
    return len(_harvest())


@torch.no_grad()
def ck_profile(model, tok, axis, batch_size=16, split="probe"):
    from common import seq_loglik
    probe, edit = load_axis(axis)
    pairs = probe if split == "probe" else edit
    texts = []
    for c, i, _k in pairs:
        texts += [c, i]
    ll = seq_loglik(model, tok, texts, batch_size, normalise=True).numpy()
    vals = np.array([float(ll[2 * n] - ll[2 * n + 1]) for n in range(len(pairs))])
    return vals, [p[2] for p in pairs]


def ck_score(model, tok, axis, batch_size=16):
    vals, _ = ck_profile(model, tok, axis, batch_size)
    return dict(ck_skew=float(vals.mean()), ck_pct=float(100 * (vals > 0).mean()),
                n=int(len(vals)))


def sd_items(axis, seed=0):
    return list(load_axis(axis)[1])


def exogenous_corpora(axis, seed=0):
    import random
    items = sd_items(axis)
    b = [c for c, _i, _k in items]
    d = [i for _c, i, _k in items]
    idx = list(range(len(b)))
    random.Random(seed).shuffle(idx)
    return [b[i] for i in idx], [d[i] for i in idx]


if __name__ == "__main__":
    print("contrasts available:", n_contrasts_available())
    for K in [1, 2, 5, 10, 20]:
        p, e = load_axis(f"ck_K{K}")
        print(f"  ck_K{K}: probe={len(p)} edit={len(e)}  ex: {p[0][0][:60]!r} || {p[0][1][:60]!r}")
