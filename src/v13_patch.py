#!/usr/bin/env python3
"""v13 patch files — every patch size in the paper becomes a measured file size.

    python3 src/v13_patch.py selftest

Two formats, one header (experiments_v13.md §2 E0.2):

  header   magic b"BSP1" | version u16 | kind u8 (0 seed, 1 index)
           | model_id str | revision str | seed u64 | density f32 | alpha f32
           | n_tensors u32 | per tensor: name str, numel u64, k_t u64, scale f32
  seed     + sign bits, packed, in decode order (the Feistel output order)
  index    + per tensor: Elias-Fano positions (low width u8, low bytes,
             high bytes) + sign bits, packed, positions ascending

str = u16 length + utf-8. Little-endian throughout. The scale is f32 (the
design sketched f16): an f16 scale would not reproduce the in-memory edit, and
apply(decode(patch)) must be bit-identical to it. It costs 2 bytes per tensor.
"""
import io
import math
import struct
import sys

import numpy as np

MAGIC, VERSION = b"BSP1", 1
SEED, INDEX = 0, 1


def _str(b, s):
    e = s.encode("utf-8")
    b.write(struct.pack("<H", len(e)))
    b.write(e)


def _rstr(f):
    (n,) = struct.unpack("<H", f.read(2))
    return f.read(n).decode("utf-8")


def _blob(b, arr):
    raw = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
    b.write(struct.pack("<Q", len(raw)))
    b.write(raw)


def _rblob(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return np.frombuffer(f.read(n), dtype=np.uint8)


# ------------------------------------------------------------ Elias-Fano ----
def ef_encode(pos, n):
    """Ascending distinct positions in [0, n) -> (low width, low bytes, high bytes)."""
    k = len(pos)
    if k == 0:
        return 0, np.zeros(0, np.uint8), np.zeros(0, np.uint8)
    l = max(0, int(math.floor(math.log2(n / k))))
    pos = pos.astype(np.int64)
    low = pos & ((1 << l) - 1)
    lowbits = ((low[:, None] >> np.arange(l, dtype=np.int64)) & 1).astype(np.uint8)
    hi = pos >> l
    upper = np.zeros(k + (n >> l) + 1, dtype=np.uint8)
    upper[hi + np.arange(k)] = 1
    return l, np.packbits(lowbits.reshape(-1), bitorder="little"), \
        np.packbits(upper, bitorder="little")


def ef_decode(l, lowb, highb, k):
    if k == 0:
        return np.zeros(0, np.int64)
    upper = np.unpackbits(highb, bitorder="little")
    hi = np.flatnonzero(upper)[:k].astype(np.int64) - np.arange(k)
    low = np.zeros(k, np.int64)
    if l:
        bits = np.unpackbits(lowb, bitorder="little")[:k * l].reshape(k, l)
        low = (bits.astype(np.int64) << np.arange(l, dtype=np.int64)).sum(1)
    return (hi << l) | low


# --------------------------------------------------------------- encode ----
def encode(kind, tensors, *, model_id, revision, seed, density, alpha):
    """tensors: [(name, numel, positions int64, signs bool, scale float)] in the
    order the patch lists them; positions in decode order for SEED, any order
    for INDEX (sorted here). Returns bytes."""
    b = io.BytesIO()
    b.write(MAGIC)
    b.write(struct.pack("<HB", VERSION, kind))
    _str(b, model_id)
    _str(b, revision)
    b.write(struct.pack("<Qff", int(seed), float(density), float(alpha)))
    b.write(struct.pack("<I", len(tensors)))
    for name, numel, pos, _sg, scale in tensors:
        _str(b, name)
        b.write(struct.pack("<QQf", int(numel), len(pos), float(scale)))
    signs = []
    for name, numel, pos, sg, _scale in tensors:
        if kind == INDEX:
            o = np.argsort(pos, kind="stable")
            pos, sg = pos[o], sg[o]
            l, lo, hi = ef_encode(pos, int(numel))
            b.write(struct.pack("<B", l))
            _blob(b, lo)
            _blob(b, hi)
        signs.append(np.asarray(sg, dtype=bool))
    allsg = np.concatenate(signs) if signs else np.zeros(0, bool)
    _blob(b, np.packbits(allsg.astype(np.uint8), bitorder="little"))
    return b.getvalue()


def decode(raw, seed_positions=None):
    """-> dict(header..., tensors=[(name, numel, positions, signs, scale)]).

    SEED patches carry no positions: seed_positions(numel, k, seed, name) must
    regenerate them (v13_seed.positions)."""
    f = io.BytesIO(raw)
    if f.read(4) != MAGIC:
        raise ValueError("not a BSP1 patch")
    ver, kind = struct.unpack("<HB", f.read(3))
    model_id, revision = _rstr(f), _rstr(f)
    seed, density, alpha = struct.unpack("<Qff", f.read(16))
    (nt,) = struct.unpack("<I", f.read(4))
    heads = []
    for _ in range(nt):
        name = _rstr(f)
        numel, k, scale = struct.unpack("<QQf", f.read(20))
        heads.append((name, numel, k, scale))
    pos = []
    for name, numel, k, _s in heads:
        if kind == INDEX:
            (l,) = struct.unpack("<B", f.read(1))
            lo, hi = _rblob(f), _rblob(f)
            pos.append(ef_decode(l, lo, hi, k))
        else:
            pos.append(seed_positions(numel, k, seed, name))
    bits = np.unpackbits(_rblob(f), bitorder="little").astype(bool)
    out, off = [], 0
    for (name, numel, k, scale), p in zip(heads, pos):
        out.append((name, numel, p, bits[off:off + k], scale))
        off += k
    return dict(version=ver, kind=kind, model_id=model_id, revision=revision,
                seed=seed, density=density, alpha=alpha, tensors=out)


# ------------------------------------------------------------ from edits ----
def tensors_of(E, positions_by_name=None):
    """Read (positions, signs, scale) off a dense one-bit edit E.

    For an index patch the positions are E's nonzeros. For a seed patch pass
    positions_by_name (decode order), so the sign bits follow that order.
    Every nonzero of a one-bit per-tensor edit is +-scale; this is asserted.
    """
    out = []
    for name in sorted(E):
        flat = E[name].reshape(-1)
        if positions_by_name is not None:
            p = positions_by_name[name]
        else:
            p = np.flatnonzero(flat.numpy() != 0).astype(np.int64)
        vals = flat[p].numpy() if len(p) else np.zeros(0, np.float32)
        scale = float(np.abs(vals).max()) if len(p) else 0.0
        if len(p) and not np.all(np.abs(vals) == np.float32(scale)):
            raise ValueError(f"{name}: edit is not one-bit per tensor")
        out.append((name, flat.numel(), p, vals >= 0, scale))
    return out


def rebuild(dec, like):
    """Dense float32 edit from a decoded patch; `like` gives shapes."""
    import torch
    E = {}
    for name, numel, p, sg, scale in dec["tensors"]:
        e = torch.zeros(int(numel), dtype=torch.float32)
        if len(p):
            s = torch.from_numpy(np.where(sg, 1.0, -1.0).astype(np.float32))
            e[torch.from_numpy(np.asarray(p, np.int64))] = s * torch.tensor(
                scale, dtype=torch.float32)
        E[name] = e.view_as(like[name])
    return E


def entropy_bits(n, k):
    """H_b(k/n)/(k/n): the entropy floor, bits per retained coordinate."""
    if k == 0 or k == n:
        return 0.0
    p = k / n
    return (-p * math.log2(p) - (1 - p) * math.log2(1 - p)) / p


def measure(E, kind, *, model_id, revision, seed, density, alpha,
            positions_by_name=None):
    """Encode, decode, rebuild, and check bit-identity. -> (bytes, stats)."""
    import torch
    import v13_seed
    t = tensors_of(E, positions_by_name)
    raw = encode(kind, t, model_id=model_id, revision=revision, seed=seed,
                 density=density, alpha=alpha)
    dec = decode(raw, seed_positions=v13_seed.positions)
    # one tensor at a time: a whole dense rebuild is +16 GB at 32B
    identical = set(x[0] for x in dec["tensors"]) == set(E)
    for x in dec["tensors"]:
        R = rebuild(dict(dec, tensors=[x]), {x[0]: E[x[0]]})
        identical &= torch.equal(R[x[0]], E[x[0]].to(torch.float32))
        del R
    k = sum(len(x[2]) for x in t)
    n = sum(int(x[1]) for x in t)
    stats = dict(kind="seed" if kind == SEED else "index", bytes=len(raw),
                 mib=len(raw) / 2**20, k=k, n=n, identical=bool(identical),
                 bits_per_coord=8 * len(raw) / max(k, 1))
    if kind == INDEX:
        # Two floors, bits per retained coordinate, positions only (+1 for the
        # sign). "global" treats the support as uniform over the whole edit
        # (the design's 8.08 at 1%); "per_tensor" conditions on each tensor's
        # count, which is lower when the support concentrates in some tensors,
        # as top-k does. The per-tensor one is the relevant bound here.
        stats["entropy_floor_bits_global"] = entropy_bits(n, k)
        stats["entropy_floor_bits_per_tensor"] = sum(
            len(x[2]) * entropy_bits(int(x[1]), len(x[2])) for x in t) / max(k, 1)
    return raw, stats


# -------------------------------------------------------------- selftest ----
def selftest():
    import torch
    import v13_seed as S
    fails = []
    rng = np.random.default_rng(0)
    for n, k in ((1000, 10), (10**6, 10**4), (10**6, 1000), (97, 97), (50, 0)):
        p = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
        l, lo, hi = ef_encode(p, n)
        if not np.array_equal(ef_decode(l, lo, hi, k), p):
            fails.append(f"Elias-Fano round trip n={n} k={k}")
    torch.manual_seed(0)
    v = {f"model.layers.{L}.self_attn.{p}": torch.randn(64, 96)
         for L in range(4) for p in ("q_proj", "v_proj")}
    import colab_t2t4 as C
    Eref, _ = C.binarize(v, "per_tensor", 0.99, 0)
    _raw, st = measure(Eref, INDEX, model_id="m", revision="r", seed=0,
                       density=0.01, alpha=16.0)
    if not st["identical"]:
        fails.append("index patch: apply(decode) != in-memory C-ref")
    scales = {k: (t.abs() * (Eref[k] != 0)).sum() / (Eref[k] != 0).sum().clamp(min=1)
              for k, t in v.items()}
    Es, _ = S.build_cseed(v, 0.01, 5, scales)
    sup = S.support({k: t.numel() for k, t in v.items()}, 0.01, 5)
    raw, st2 = measure(Es, SEED, model_id="m", revision="r", seed=5, density=0.01,
                       alpha=16.0, positions_by_name=sup)
    if not st2["identical"]:
        fails.append("seed patch: apply(decode) != in-memory C-seed")
    if decode(raw, S.positions)["seed"] != 5:
        fails.append("seed not carried in the header")
    if st2["bytes"] >= st["bytes"]:
        fails.append("seed patch is not smaller than the index patch")
    if abs(entropy_bits(10**6, 10**4) - 8.08) > 0.01 or \
            abs(entropy_bits(10**6, 10**3) - 11.41) > 0.01:
        fails.append("entropy floor does not match 8.08 / 11.41")
    for f in fails:
        print("FAIL:", f)
    print(f"v13_patch selftest: {'OK' if not fails else 'FAILED'} "
          f"(toy: index {st['bytes']} B, {st['bits_per_coord']:.1f} bit/coord; "
          f"seed {st2['bytes']} B)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
        __import__("os").path.abspath(__file__))))
    sys.exit(selftest() if sys.argv[1:] == ["selftest"] else 0)
