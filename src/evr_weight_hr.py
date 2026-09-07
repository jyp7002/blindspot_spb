"""W'' — high-rank natural-rank removability test. Train the ground-truth
exogenous contrast ΔW at rank 128 (uncapped vs the r16 edit) on qwen1.5b, measure
the NATURAL participation-ratio effective rank, regress removal on it. If
removable axes have LOW natural rank and non-removable HIGH, removability is
intrinsic direction coherence; if peff is still uniform, it is functional.
"""
import os, json, argparse
import numpy as np, torch
from scipy import stats

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def peff_of(dw, topk=16):
    prs, e16, w = [], [], []
    for k, W in dw.items():
        M = W.detach().to(torch.float32)
        if M.ndim != 2 or min(M.shape) < topk + 1:
            continue
        s = torch.linalg.svdvals(M).cpu().numpy(); s2 = s ** 2
        tot = float(s2.sum())
        if tot <= 0:
            continue
        prs.append(float((s2.sum() ** 2) / np.sum(s2 ** 2)))
        e16.append(float(s2[:topk].sum()) / tot)
        w.append(float(np.sqrt(tot)))
    if not prs:
        return float("nan"), float("nan")
    w = np.array(w); w = w / w.sum()
    return float(np.dot(w, prs)), float(np.dot(w, e16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--steps", type=int, default=250)
    a = ap.parse_args()
    import evr
    from elicit_gen import exogenous_corpora
    from common import load, free
    from edit import train_task_vector, contrast
    ref = evr.reference_removal(a.model)
    axes = sorted(ref.keys())
    cache = os.path.join(RES, f"directions_r{a.rank}")
    os.makedirs(cache, exist_ok=True)
    print(f"W'': rank={a.rank}, {len(axes)} axes on {a.model}\n")
    model, tok = load(a.model)
    out = []
    try:
        for ax in axes:
            fp = os.path.join(cache, f"{a.model}|{ax}.pt")
            try:
                if os.path.exists(fp):
                    dw = {k: v.float() for k, v in torch.load(fp, map_location="cpu").items()}
                else:
                    b, d = exogenous_corpora(ax, seed=0)
                    dwb, _ = train_task_vector(model, tok, b, rank=a.rank, steps=a.steps,
                                               lr=1e-4, bs=8, seed=0)
                    dwd, _ = train_task_vector(model, tok, d, rank=a.rank, steps=a.steps,
                                               lr=1e-4, bs=8, seed=0)
                    dw = contrast(dwb, dwd)
                    torch.save({k: t.to(torch.float16) for k, t in dw.items()}, fp)
                    del dwb, dwd
                pr, e16 = peff_of(dw)
                out.append(dict(axis=ax, peff=pr, e16=e16, ref_removal=ref[ax]))
                print(f"  {ax:26s} peff={pr:7.2f} e16={e16:.3f} ref={ref[ax]:+.4f}", flush=True)
                import gc; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {ax:26s} FAIL {type(e).__name__}: {e}", flush=True)
    finally:
        free(model, tok)
    json.dump(out, open(os.path.join(RES, f"evr_weight_r{a.rank}.json"), "w"), indent=2)
    d = [r for r in out if not np.isnan(r["peff"])]
    x = np.array([r["peff"] for r in d]); y = np.array([r["ref_removal"] for r in d])
    print(f"\nW'' removal ~ natural peff (rank {a.rank}, n={len(d)}); expect NEGATIVE:")
    r = stats.pearsonr(x, y); rho = stats.spearmanr(x, y)
    print(f"   Pearson r={r.statistic:+.3f} p={r.pvalue:.4f} | Spearman rho={rho.statistic:+.3f} p={rho.pvalue:.4f}")
    print(f"   peff range: {x.min():.1f} - {x.max():.1f}  (uniform => functional, not geometric)")


if __name__ == "__main__":
    main()
