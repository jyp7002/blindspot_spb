"""Validation harness for the v8 estimators.

The v8 write-up claimed these estimators were "validated against known inputs".
They were — but the checks were run inline and never committed, so the claim was
not reproducible from the repository. An audit caught that. This file is the
claim made good: run it and every number quoted in RESULTS_V8.md §5 as a
validation figure is reproduced.

    python3 src/v8_selftest.py
"""
import os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v8_support import dump_support, load_support
from v8_sup1 import _pair_stats, _layer_vec, _layer_corr


def _mk(seed, n_per, nmod, dens, base=None, overlap=0.0, flip=None):
    g = np.random.default_rng(seed)
    E = {}
    side = int(np.sqrt(n_per))
    n_per = side * side          # keep the tensor exactly n_per elements
    for m in range(nmod):
        key = f"model.layers.{m // 2}.self_attn.{['q_proj', 'k_proj'][m % 2]}"
        t = torch.zeros(side, side)
        k = int(dens * n_per)
        if base is None:
            idx = g.choice(n_per, k, replace=False)
        else:
            b = base[key]
            keep = g.choice(b, int(overlap * k), replace=False)
            pool = np.setdiff1d(np.arange(n_per), b)
            idx = np.concatenate([keep, g.choice(pool, k - len(keep), replace=False)])
        sgn = g.choice([-1.0, 1.0], len(idx))
        if flip is not None:
            sgn = np.where(g.random(len(idx)) < flip, -sgn, sgn)
        t.view(-1)[idx] = torch.tensor(sgn, dtype=torch.float32)
        E[key] = t
    return E


def _sup(E):
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        p = f.name
    dump_support(p, E, meta=dict(tag="selftest"))
    m = load_support(p)[1]
    os.unlink(p)
    return m


def test_fold_enrichment():
    """Known-answer checks for the SUP1 overlap estimator."""
    N, NMOD, D = 10000, 8, 0.01
    A = _mk(1, N, NMOD, D)
    idxA = {k: np.nonzero(v.view(-1).numpy())[0] for k, v in A.items()}
    checks = []

    st = _pair_stats(_sup(A), _sup(A))
    checks.append(("identical -> fold == 1/density", abs(st["fold_global"] - 1 / D) < 1e-6))
    checks.append(("identical -> jaccard == 1", abs(st["jaccard"] - 1.0) < 1e-9))
    checks.append(("identical -> signed_frac == 1", abs(st["signed_frac_of_intersection"] - 1.0) < 1e-9))

    B = {k: -v for k, v in A.items()}
    st = _pair_stats(_sup(A), _sup(B))
    checks.append(("sign-flipped -> fold == 1/density", abs(st["fold_global"] - 1 / D) < 1e-6))
    checks.append(("sign-flipped -> signed_frac == 0", st["signed_frac_of_intersection"] == 0.0))

    st = _pair_stats(_sup(A), _sup(_mk(3, N, NMOD, D, base=idxA, overlap=0.5)))
    checks.append(("50% shared -> fold ~ 50x", abs(st["fold_global"] - 50) < 1.0))

    # null calibration needs a large support to beat Poisson noise
    NB, NMODB = 400000, 8
    folds, signed = [], []
    for i in range(5):
        st = _pair_stats(_sup(_mk(10 + i, NB, NMODB, D)), _sup(_mk(200 + i, NB, NMODB, D)))
        folds.append(st["fold_global"]); signed.append(st["signed_frac_of_intersection"])
    checks.append((f"independent -> fold ~ 1.0 (got {np.mean(folds):.3f})",
                   abs(np.mean(folds) - 1.0) < 0.05))
    checks.append((f"independent -> signed_frac ~ 0.5 (got {np.mean(signed):.3f})",
                   abs(np.mean(signed) - 0.5) < 0.05))
    return checks


def test_binarize_bit_identity():
    """The memory-optimised threshold must reproduce the original cat+kthvalue."""
    import copy
    import colab_t2t4 as C
    torch.manual_seed(1)
    checks = []
    for shape, nmod in (((256, 256), 16), ((512, 512), 8)):
        v = {f"model.layers.{L}.self_attn.{p}": torch.randn(*shape)
             for L in range(nmod // 4) for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
        for gran in ("per_tensor", "per_channel", "scalar"):
            for sp in (0.0, 0.1, 0.5, 0.9, 0.99, 0.999):
                E, _m = C.binarize(copy.deepcopy(v), gran, sp, 0)
                vv = copy.deepcopy(v)
                if sp > 0:
                    flat = torch.cat([t.abs().flatten() for t in vv.values()])
                    k = int(sp * flat.numel())
                    th = flat.kthvalue(k).values.item() if k > 0 else -1.0
                else:
                    th = -1.0
                ok = True
                for key, t in vv.items():
                    mask = (t.abs() > th).float(); sg = torch.sign(t) * mask
                    if gran == "per_tensor":
                        sc = (t.abs() * mask).sum() / mask.sum().clamp(min=1)
                    elif gran == "per_channel":
                        sc = ((t.abs() * mask).sum(-1, keepdim=True)
                              / mask.sum(-1, keepdim=True).clamp(min=1))
                    else:
                        allv = torch.cat([x.flatten() for x in vv.values()])
                        sc = torch.tensor(allv.abs().mean().item())
                    if not torch.equal(E[key], sg * sc):
                        ok = False; break
                checks.append((f"binarize {shape} {gran} sp={sp}", ok))
    return checks


def test_partition_convention():
    """numpy.partition(k-1) must equal torch.kthvalue(k) including on ties."""
    checks = []
    g = np.random.default_rng(0)
    for trial in range(200):
        n = int(g.integers(10, 2000))
        a = g.choice([0.0, 1.0, 2.0, 3.0], n) if trial % 3 == 0 else g.normal(size=n)
        t = torch.tensor(a, dtype=torch.float32)
        k = int(g.integers(1, n + 1))
        want = t.kthvalue(k).values.item()
        arr = t.numpy().copy(); arr.partition(k - 1)
        checks.append((f"partition==kthvalue n={n} k={k}", float(arr[k - 1]) == want))
    n_bad = sum(1 for _d, ok in checks if not ok)
    return [(f"partition == kthvalue over 200 random cases (incl. ties)", n_bad == 0)]


def main():
    groups = [("SUP1 fold-enrichment", test_fold_enrichment),
              ("binarize bit-identity", test_binarize_bit_identity),
              ("partition/kthvalue convention", test_partition_convention)]
    total = bad = 0
    for name, fn in groups:
        print(f"\n[{name}]")
        for desc, ok in fn():
            total += 1; bad += (not ok)
            if not ok or len(desc) < 60:
                print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
        print(f"  ({name}: all checks run)")
    print(f"\n{total - bad}/{total} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
