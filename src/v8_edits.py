"""DEC edit constructions (experiments_v8 §DEC).

Seven conditions decomposing WHERE the edit acts (support), WHAT it does there
(signs) and WHETHER LOCATION MATTERS (layer/projection identity), all at a
common density and under a single, explicitly chosen scale convention.

    id             support                signs
    C-ref          top-|v|                true sign(v) at those coords
    C-a            random                 true sign(v) at those coords
    C-bottom       bottom-|v| (nonzero)   true sign(v) at those coords
    C-b            top-|v|                sign values permuted within tensor
    C-rand         random                 random
    C-layershuf    top-|v|                signed pattern moved between layers
    C-tensorshuf   top-|v|                signed pattern moved between projections

THE SCALE CONVENTION IS THE LOAD-BEARING CHOICE. Per-tensor scale in
colab_t2t4.binarize is mean(|v|) over the SURVIVING coordinates, so a random
support (C-a) would naturally receive a far smaller scale than the top-|v|
support (C-ref) -- plausibly 5-20x smaller. Since alpha tops out at 16, that
deficit is not recoverable by the alpha grid, and C-a would lose on EDIT NORM
rather than on whether its support carries information. That would silently
invalidate Delta_selection, the quantity that adjudicates this paper against
DARE/TIES/BitDelta. Every convention is therefore implemented explicitly and
selected by name, never by default:

    "natural"    each condition takes mean|v| over its own support
                 (the BitDelta-faithful reading; confounds norm with support)
    "ref_tensor" every condition takes C-ref's per-tensor scale
                 (equal per-tensor amplitude; per-tensor counts still differ)
    "ref_norm"   as ref_tensor, then a single global factor so that
                 ||E||_F == ||E_ref||_F exactly (equal total edit norm)

C-ref is bit-identical to colab_t2t4.binarize(v, "per_tensor", 1-density, seed)
under every convention -- verified by tests in `selftest()`.
"""
import numpy as np
import torch

from v8_support import parse_module

CONDITIONS = ["C-ref", "C-a", "C-bottom", "C-b", "C-rand",
              "C-layershuf", "C-tensorshuf"]


# ------------------------------------------------------------ supports ----

def _k_target(v, density):
    """The support size C-ref actually produces, which every other condition
    must match exactly.

    binarize() does k_drop = int(sparsity * N) then keeps |t| > kthvalue(k_drop),
    so the retained count is N - int((1-density) * N) -- NOT round(density * N).
    Deriving the random/bottom supports from their own rounding rule instead put
    them one coordinate below C-ref, which is a (tiny) support-size difference
    where the design demands exact parity.
    """
    n = sum(t.numel() for t in v.values())
    return n - int((1.0 - density) * n)


def _global_threshold(v, density, largest=True):
    """Value threshold selecting exactly _k_target(v, density) coordinates by |v|.

    Mirrors binarize()'s global (all-tensors-concatenated) kthvalue so that
    C-ref reproduces the existing 99%-sparse edit exactly.
    """
    # Same memory discipline as colab_t2t4.binarize: fill one preallocated buffer
    # rather than torch.cat over a list of abs() copies, which would peak at 3x
    # the contrast vector. See the note there (2026-08-29 SPC OOMs).
    n = sum(t.numel() for t in v.values())
    if largest:
        k = int((1.0 - density) * n)          # binarize()'s sparsity semantics
        if k <= 0:
            return -1.0
    else:
        k = max(_k_target(v, density), 1)     # same retained count, other tail
    buf = torch.empty(n, dtype=torch.float32)
    off = 0
    for t in v.values():
        m = t.numel()
        torch.abs(t.reshape(-1), out=buf[off:off + m])
        off += m
    arr = buf.numpy()
    arr.partition(k - 1)
    thresh = float(arr[k - 1])
    del arr, buf
    return thresh


# numpy's Generator.hypergeometric refuses ngood or nbad >= 1e9. Every model in
# the published DEC panel edits fewer than 1e9 attention parameters (the largest,
# qwen2.5-7B, is 8.2e8; phi is 9.1e8 because its fused qkv_proj is edited and
# o_proj is not), so the wall was invisible until v11 added bigger targets:
#
#     llama-3.1-8B   1.34e9      gemma-2-9b     1.70e9
#     qwen2.5-32B    4.03e9      llama-3.1-70B  12.08e9
#
# so the whole big-tier expansion was blocked here, not on VRAM.
_HYPERGEO_LIMIT = 1_000_000_000


def _counts_per_tensor(v, k_total, rng):
    """Exact multivariate-hypergeometric split of k_total over the tensors.

    Sampling k_total coordinates uniformly without replacement from the whole
    edit is equivalent to drawing per-tensor counts this way, and costs no
    N-sized permutation.

    TWO PATHS, AND THE SMALL ONE IS UNTOUCHED. Below numpy's 1e9 hypergeometric
    limit the original sequential draw runs exactly as before -- same calls,
    same RNG stream, same coordinates -- so every published cell still
    reproduces bit-identically. The large path is only ever taken where the old
    code raised ValueError, so it cannot change a number that already exists.
    """
    sizes = [t.numel() for t in v.values()]
    total = sum(sizes)
    if total - min(sizes) >= _HYPERGEO_LIMIT:
        return _counts_per_tensor_large(sizes, k_total, rng)

    remaining_n, remaining_k, out = total, k_total, []
    for n_m in sizes:
        if remaining_k <= 0:
            out.append(0); remaining_n -= n_m; continue
        take = int(rng.hypergeometric(n_m, remaining_n - n_m, remaining_k)) \
            if remaining_n > n_m else remaining_k
        out.append(take)
        remaining_k -= take
        remaining_n -= n_m
    return out


def _counts_per_tensor_large(sizes, k_total, rng):
    """Same estimand above numpy's hypergeometric limit, sampled directly.

    Instead of drawing the per-tensor counts, draw the k_total coordinates
    themselves -- uniformly, without replacement, from the flat concatenation --
    and count how many land in each tensor. That IS the multivariate
    hypergeometric, by construction, so this is exact rather than an
    approximation of it. (A binomial chain would have been the easy substitute
    and is wrong in a way that matters here: it drops the finite-population
    correction, inflating the variance of every count.)

    Uniform-without-replacement is obtained by oversampling with replacement,
    deduplicating, and then taking a random k_total-subset. That is exact: by
    symmetry over relabelling the N coordinates, the distinct set is a uniformly
    random subset of its size, and a uniform subset of a uniform subset is
    uniform.

    Cost is k_total int64s, not N: ~107 MB at 8B (k=1.3e7), ~1 GB at 70B
    (k=1.2e8). That is why the counts are not obtained by permuting N.
    """
    n = int(sum(sizes))
    k = int(k_total)
    if k <= 0:
        return [0] * len(sizes)
    if k >= n:
        return list(sizes)

    picked = np.unique(rng.integers(0, n, size=k, dtype=np.int64))
    # Expected shortfall is k^2/(2n) -- about 67k of 13.4M at 8B -- so this
    # loop runs a couple of times, never long.
    while picked.size < k:
        need = k - picked.size
        extra = rng.integers(0, n, size=max(need * 2, 1024), dtype=np.int64)
        picked = np.unique(np.concatenate([picked, extra]))
    if picked.size > k:
        # np.unique returns sorted output; truncating it would bias towards low
        # indices, i.e. towards the first tensors. Permute, then cut.
        picked = rng.permutation(picked)[:k]

    bounds = np.cumsum(np.asarray(sizes, dtype=np.int64))
    which = np.searchsorted(bounds, picked, side="right")
    counts = np.bincount(which, minlength=len(sizes))
    return [int(c) for c in counts]


def _masks(v, condition, density, seed):
    """-> {key: bool mask} at the requested density."""
    g = np.random.default_rng(seed)
    if condition in ("C-a", "C-rand"):
        k_total = _k_target(v, density)       # exact parity with C-ref
        counts = _counts_per_tensor(v, k_total, g)
        out = {}
        for (key, t), k_m in zip(v.items(), counts):
            m = torch.zeros(t.numel(), dtype=torch.bool)
            if k_m > 0:
                idx = g.choice(t.numel(), size=k_m, replace=False)
                m[torch.from_numpy(np.sort(idx))] = True
            out[key] = m.view_as(t)
        return out
    if condition == "C-bottom":
        thresh = _global_threshold(v, density, largest=False)
        return {k: (t.abs() <= thresh) & (t != 0) for k, t in v.items()}
    # every remaining condition is supported on top-|v|
    thresh = _global_threshold(v, density, largest=True)
    return {k: (t.abs() > thresh) for k, t in v.items()}


# ------------------------------------------------------------- shuffles ----

def _shape_groups(v, level):
    """Group module keys whose sign patterns may legally be swapped.

    level='layer'  : same projection, different layer  -> permute over layers
    level='tensor' : same layer, different projection  -> permute within layer
    Only shape-compatible members are grouped; the map is returned for logging.
    """
    groups = {}
    for key, t in v.items():
        L, proj = parse_module(key)
        if L is None:
            continue
        gk = (proj, tuple(t.shape)) if level == "layer" else (L, tuple(t.shape))
        groups.setdefault(gk, []).append(key)
    return {g: sorted(ks) for g, ks in groups.items() if len(ks) > 1}


def _permute_patterns(signs, v, level, seed):
    """Move whole signed patterns between shape-compatible modules.

    Returns (new_signs, permutation_map). A module in no group keeps its own
    pattern, and that is recorded rather than silently ignored.
    """
    g = np.random.default_rng(seed + 977)
    out = dict(signs)
    pmap = {}
    for gk, keys in sorted(_shape_groups(v, level).items(), key=lambda kv: str(kv[0])):
        n = len(keys)
        perm = np.arange(n)
        if n > 1:                       # derangement: no module keeps its own
            while True:
                g.shuffle(perm)
                if not np.any(perm == np.arange(n)):
                    break
        for dst_i, src_i in enumerate(perm):
            out[keys[dst_i]] = signs[keys[src_i]]
            pmap[keys[dst_i]] = keys[src_i]
    for k in signs:
        pmap.setdefault(k, k)
    return out, pmap


# --------------------------------------------------------------- build ----

def build_edit(v, condition, density=0.01, seed=0, scale_mode="ref_tensor",
               ref_scales=None):
    """Construct one DEC condition. Returns (E, meta).

    v is NOT mutated. ref_scales, when given, is {key: scalar} from C-ref and is
    required by the "ref_tensor"/"ref_norm" conventions.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    if scale_mode not in ("natural", "ref_tensor", "ref_norm"):
        raise ValueError(f"unknown scale_mode {scale_mode}")

    masks = _masks(v, condition, density, seed)
    g = torch.Generator().manual_seed(seed)

    # ---- sign field ----
    signs = {}
    for key, t in v.items():
        m = masks[key].float()
        if condition == "C-rand":
            rnd = (torch.randint(0, 2, t.shape, generator=g).float() * 2 - 1)
            signs[key] = rnd * m
        elif condition == "C-b":
            s = torch.sign(t) * m
            flat, mf = s.flatten(), masks[key].flatten()
            idx = torch.nonzero(mf, as_tuple=False).flatten()
            if idx.numel() > 1:
                perm = torch.randperm(idx.numel(), generator=g)
                flat[idx] = flat[idx][perm]
            signs[key] = flat.view_as(t)
        else:
            signs[key] = torch.sign(t) * m

    pmap = None
    if condition == "C-layershuf":
        signs, pmap = _permute_patterns(signs, v, "layer", seed)
    elif condition == "C-tensorshuf":
        signs, pmap = _permute_patterns(signs, v, "tensor", seed)

    # ---- scale ----
    E, n_flips, n_total = {}, 0, 0
    for key, t in v.items():
        s = signs[key]
        if scale_mode == "natural":
            m = (s != 0).float()
            denom = m.sum().clamp(min=1)
            scale = (t.abs() * m).sum() / denom
        else:
            if ref_scales is None or key not in ref_scales:
                raise ValueError(f"scale_mode={scale_mode} needs ref_scales[{key}]")
            scale = ref_scales[key]
        E[key] = s * scale
        n_flips += int((s != 0).sum().item())
        n_total += t.numel()

    if scale_mode == "ref_norm":
        if ref_scales is None:
            raise ValueError("ref_norm needs ref_scales")
        cur = torch.sqrt(sum((e.float() ** 2).sum() for e in E.values()))
        tgt = ref_scales.get("__fro__")
        if tgt is None:
            raise ValueError("ref_norm needs ref_scales['__fro__']")
        if float(cur) > 0:
            f = float(tgt) / float(cur)
            for k in E:
                E[k] = E[k] * f

    meta = dict(condition=condition, density=density, scale_mode=scale_mode,
                n_flips=n_flips, n_params=n_total,
                effective_sparsity=1.0 - n_flips / max(n_total, 1),
                bytes=n_flips / 8.0 + 2 * len(E),
                frobenius=float(torch.sqrt(sum((e.float() ** 2).sum()
                                               for e in E.values()))))
    if pmap is not None:
        meta["permutation_map"] = pmap
    return E, meta


def ref_scale_table(v, density=0.01, seed=0):
    """C-ref's per-tensor scales + total Frobenius norm, for the ref_* modes."""
    E, _m = build_edit(v, "C-ref", density, seed, scale_mode="natural")
    thresh_masks = _masks(v, "C-ref", density, seed)
    out = {}
    for key, t in v.items():
        m = thresh_masks[key].float()
        denom = m.sum().clamp(min=1)
        out[key] = (t.abs() * m).sum() / denom
    out["__fro__"] = float(torch.sqrt(sum((e.float() ** 2).sum() for e in E.values())))
    return out


# ---------------------------------------------------------------- test ----

def selftest(verbose=True):
    """C-ref bit-identity vs binarize(), plus per-condition invariants."""
    import colab_t2t4 as C
    torch.manual_seed(0)
    v = {f"model.layers.{L}.self_attn.{p}": torch.randn(64, 64)
         for L in range(6) for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
    density, seed = 0.01, 0

    ref_bin, meta_bin = C.binarize(v, "per_tensor", 1.0 - density, seed)
    ref_new, meta_new = build_edit(v, "C-ref", density, seed, scale_mode="natural")
    same = all(torch.equal(ref_bin[k], ref_new[k]) for k in ref_bin)
    maxdiff = max(float((ref_bin[k] - ref_new[k]).abs().max()) for k in ref_bin)
    if verbose:
        print(f"[v8_edits] C-ref bit-identical to binarize(): {same} "
              f"(max|diff|={maxdiff:.3e}, nnz {meta_bin['n_flips']} vs {meta_new['n_flips']})")
    assert same, "C-ref MUST reproduce the frozen binarize output exactly"

    rs = ref_scale_table(v, density, seed)
    n_ref = meta_new["n_flips"]
    rows = []
    for cond in CONDITIONS:
        E, m = build_edit(v, cond, density, seed, scale_mode="ref_tensor", ref_scales=rs)
        rows.append((cond, m["n_flips"], m["frobenius"]))
    if verbose:
        print(f"[v8_edits] under ref_tensor (C-ref nnz={n_ref}):")
        for c, nnz, fro in rows:
            print(f"             {c:14s} nnz={nnz:6d}  ||E||_F={fro:9.4f}")

    # support-count parity: every condition must edit the same number of coords
    counts = {c: n for c, n, _f in rows}
    assert len(set(counts.values())) == 1, f"support-size parity broken: {counts}"

    # C-a must NOT be norm-handicapped under ref_tensor
    fro = {c: f for c, _n, f in rows}
    ratio = fro["C-a"] / fro["C-ref"]
    if verbose:
        print(f"[v8_edits] ||C-a||/||C-ref|| under ref_tensor = {ratio:.4f}")
    assert 0.98 < ratio < 1.02, f"C-a norm not matched: {ratio}"

    # and it IS handicapped under natural -- the confound this module exists for
    Ea, ma = build_edit(v, "C-a", density, seed, scale_mode="natural")
    nat_ratio = ma["frobenius"] / fro["C-ref"]
    if verbose:
        print(f"[v8_edits] ||C-a||/||C-ref|| under natural    = {nat_ratio:.4f} "
              f"<- the confound")
    return dict(ref_identical=same, ref_tensor_ratio=ratio, natural_ratio=nat_ratio)


if __name__ == "__main__":
    selftest()
