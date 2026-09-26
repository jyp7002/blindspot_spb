#!/usr/bin/env python3
"""v13 C-seed — a random 1% support that is regenerated from a seed, not stored.

    python3 src/v13_seed.py selftest     # E0.1 tests (no torch needed for most)

CONSTRUCTION (experiments_v13.md §2 E0.1, PREREGISTRATION §v13.B):
  k        = N - floor(0.99 N)                  (v8_edits._k_target: C-ref parity)
  k_t      = deterministic largest-remainder allocation of k over tensors,
             proportional to numel; ties broken by tensor name. No RNG.
  position = the first k_t outputs of a keyed pseudorandom permutation of
             [0, N_t): 4-round (unbalanced) Feistel on the next power of two,
             cycle-walked back into range. Round keys = the four 64-bit words
             of SHA-256(seed_u64_le || tensor_name_utf8).
  sign     = +1 where dW >= 0 at the position, else -1. (sign(0) is +1 so that
             one bit per coordinate is lossless; the in-memory edit and the
             decoded patch use the same rule.)
  scale    = C-ref's per-tensor scale (v8_edits.ref_scale_table, ref_tensor).

WHY NOT A LIBRARY SAMPLER. NumPy guarantees stable BitGenerator streams across
versions, not stable Generator methods (choice, permutation, integers may change
algorithm). A patch decoded on another install must select the same
coordinates, so the permutation is plain uint64 arithmetic: xor, shifts and
wrapping multiplies, whose results no library version can change.
"""
import hashlib
import math
import os
import subprocess
import sys

import numpy as np

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_C1, _C2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def k_target(sizes, density):
    """Retained count, identical to v8_edits._k_target / binarize()."""
    n = int(sum(sizes))
    return n - int((1.0 - density) * n)


def allocate(names, sizes, k):
    """Largest-remainder split of k over tensors, proportional to size.

    Exact integer arithmetic; ties in the remainder go to the earlier name in
    sorted order. Returns {name: k_t} with sum == k.
    """
    order = sorted(range(len(names)), key=lambda i: names[i])
    n = int(sum(sizes))
    base, rem = {}, []
    for i in order:
        q, r = divmod(k * int(sizes[i]), n)
        base[names[i]] = q
        rem.append((-r, names[i]))
    left = k - sum(base.values())
    for _r, nm in sorted(rem)[:left]:
        base[nm] += 1
    assert sum(base.values()) == k
    return base


def round_keys(seed, name):
    h = hashlib.sha256(int(seed).to_bytes(8, "little") + name.encode("utf-8")).digest()
    return [np.uint64(int.from_bytes(h[8 * i:8 * i + 8], "little")) for i in range(4)]


def _mix(x, key):
    """splitmix64 finaliser of (x xor key): uint64 in, uint64 out, wrapping."""
    with np.errstate(over="ignore"):
        z = x ^ key
        z = (z ^ (z >> np.uint64(30))) * _C1
        z = (z ^ (z >> np.uint64(27))) * _C2
        return z ^ (z >> np.uint64(31))


def _feistel(x, bits, keys):
    """A permutation of [0, 2**bits): 4 unbalanced Feistel rounds.

    State (L, R) with widths (a, b), a + b = bits. A round maps
    (L, R) -> (R, L xor F(R) mod 2**a), which swaps the widths; four rounds
    restore them. Each round is a bijection, so the composite is one.
    """
    a = bits // 2
    b = bits - a
    L = x >> np.uint64(b)
    R = x & np.uint64((1 << b) - 1)
    wl, wr = a, b
    for key in keys:
        F = _mix(R, key) & np.uint64((1 << wl) - 1)
        L, R = R, L ^ F
        wl, wr = wr, wl
    return (L << np.uint64(wr)) | R


def positions(n, k, seed, name):
    """First k outputs of the keyed permutation of [0, n), as int64, in order."""
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    bits = max(1, (n - 1).bit_length())
    keys = round_keys(seed, name)
    y = _feistel(np.arange(k, dtype=np.uint64), bits, keys)
    bad = y >= np.uint64(n)
    while bad.any():                         # cycle-walking: < 2 steps expected
        y[bad] = _feistel(y[bad], bits, keys)
        bad = y >= np.uint64(n)
    return y.astype(np.int64)


def support(v_shapes, density, seed):
    """{name: positions} for a whole edit. v_shapes: {name: numel}."""
    names = list(v_shapes)
    sizes = [int(v_shapes[k]) for k in names]
    k = k_target(sizes, density)
    kt = allocate(names, sizes, k)
    return {nm: positions(int(v_shapes[nm]), kt[nm], seed, nm) for nm in names}


def build_cseed(v, density, seed, ref_scales):
    """-> (E, meta): the C-seed edit as dense float32 tensors, v not mutated."""
    import torch
    shapes = {k: t.numel() for k, t in v.items()}
    sup = support(shapes, density, seed)
    E, n_flips, n_total = {}, 0, 0
    for key, t in v.items():
        pos = torch.from_numpy(sup[key])
        flat = t.reshape(-1)
        s = torch.where(flat[pos] >= 0, 1.0, -1.0).to(torch.float32)
        e = torch.zeros(t.numel(), dtype=torch.float32)
        e[pos] = s * ref_scales[key].to(torch.float32)
        E[key] = e.view_as(t)
        n_flips += int(pos.numel())
        n_total += t.numel()
    return E, dict(n_flips=n_flips, n_params=n_total,
                   effective_sparsity=1.0 - n_flips / max(n_total, 1),
                   support="C-seed", seed=int(seed))


# -------------------------------------------------------------- selftest ----
def _chi2_uniform(pos, n, bins=64):
    from scipy.stats import chisquare
    h = np.bincount((pos * bins) // n, minlength=bins)
    return float(chisquare(h).pvalue)


def _digest(n, k, seed, name):
    return hashlib.sha256(positions(n, k, seed, name).tobytes()).hexdigest()


def selftest():
    fails = []
    # permutation: bijective on odd and even bit widths, incl. non-powers of two
    for n in (1, 2, 3, 7, 64, 1000, 4097):
        p = positions(n, n, 3, f"t{n}")
        if sorted(p.tolist()) != list(range(n)):
            fails.append(f"not a permutation of [0,{n})")
    # allocation exact and proportional
    sizes = [10**6, 3 * 10**6 + 7, 5, 123457]
    names = ["b", "a", "d", "c"]
    k = k_target(sizes, 0.01)
    kt = allocate(names, sizes, k)
    if sum(kt.values()) != k:
        fails.append("allocation does not sum to k")
    for nm, sz in zip(names, sizes):
        if abs(kt[nm] - k * sz / sum(sizes)) >= 1:
            fails.append(f"allocation of {nm} off by >= 1")
    if k != sum(sizes) - int(0.99 * sum(sizes)):
        fails.append("k differs from v8 _k_target")
    # distinctness at the design's scale
    p = positions(10**6, 10**4, 0, "x")
    if len(np.unique(p)) != 10**4 or p.min() < 0 or p.max() >= 10**6:
        fails.append("positions not distinct / in range")
    # uniformity: chi2 over 64 bins, 20 keys, N=1e6, first 1% of the permutation
    pv = [_chi2_uniform(positions(10**6, 10**4, s, "unif"), 10**6) for s in range(20)]
    if min(pv) <= 0.01:
        fails.append(f"chi2 uniformity p <= 0.01 for some key: min {min(pv):.4f}")
    # keys matter: different seed / name -> different support
    if _digest(10**5, 1000, 0, "a") in (_digest(10**5, 1000, 1, "a"),
                                        _digest(10**5, 1000, 0, "b")):
        fails.append("support does not depend on seed and name")
    # a fresh process decodes the same positions
    here = _digest(123457, 1235, 7, "model.layers.0.self_attn.q_proj")
    code = ("import sys; sys.path.insert(0, %r); import v13_seed as S; "
            "print(S._digest(123457, 1235, 7, 'model.layers.0.self_attn.q_proj'))"
            % os.path.dirname(os.path.abspath(__file__)))
    other = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True).stdout.strip()
    if other != here:
        fails.append("a fresh process decodes different positions")
    for f in fails:
        print("FAIL:", f)
    print(f"v13_seed selftest: {'OK' if not fails else 'FAILED'} "
          f"(chi2 p over 20 keys: min {min(pv):.3f}, median {sorted(pv)[10]:.3f}; "
          f"cross-process digest {here[:12]})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(selftest() if sys.argv[1:] == ["selftest"] else 0)
