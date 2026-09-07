"""W' — weight-space rank-concentration removability predictor (non-circular).

For each axis, train the GROUND-TRUTH exogenous contrast ΔW on qwen1.5b (attn
q/k/v/o), measure how concentrated ΔW is (a rank-16 binary edit can only capture
a low-rank direction), and regress the exogenous removal fraction on it.
FROZEN metric (see PREREGISTRATION): per attn matrix SVD -> (a) top-16 energy
fraction, (b) participation-ratio effective rank; norm-weighted mean across
matrices. ΔW is trained independent of the removal outcome (non-circular).
"""
import os, json, argparse
import numpy as np
import torch

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def rank_stats(dw, topk=16):
    """Norm-weighted mean over attn matrices of (top-k energy fraction, peff)."""
    e_list, pr_list, w_list = [], [], []
    for k, W in dw.items():
        M = W.detach().to(torch.float32)
        if M.ndim != 2 or min(M.shape) < topk + 1:
            continue
        s = torch.linalg.svdvals(M).cpu().numpy()
        s2 = s ** 2
        tot = float(s2.sum())
        if tot <= 0:
            continue
        top = float(s2[:topk].sum()) / tot                 # energy in top-16
        peff = (s2.sum() ** 2) / (np.sum(s2 ** 2))          # participation ratio
        wnorm = float(np.sqrt(tot))
        e_list.append(top); pr_list.append(float(peff)); w_list.append(wnorm)
    if not e_list:
        return float("nan"), float("nan")
    w = np.array(w_list); w = w / w.sum()
    return float(np.dot(w, e_list)), float(np.dot(w, pr_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5b")
    a = ap.parse_args()
    import evr  # reuse the reference-removal loader (measured artifact)
    import geometry
    ref = evr.reference_removal(a.model)
    axes = sorted(ref.keys())
    print(f"W': {len(axes)} axes with reference removal on {a.model}\n")
    from common import load, free
    model, tok = load(a.model)
    out = []
    try:
        for ax in axes:
            try:
                dw = geometry.get_direction(a.model, ax, 0, model=model, tok=tok, cache=True)
            except Exception as e:
                print(f"  {ax:26s} DELTA FAIL: {type(e).__name__}: {e}"); continue
            top16, peff = rank_stats(dw)
            out.append(dict(axis=ax, top16_energy=top16, peff=peff, ref_removal=ref[ax]))
            print(f"  {ax:26s} top16={top16:.3f} peff={peff:6.1f} ref_removal={ref[ax]:+.4f}",
                  flush=True)
    finally:
        free(model, tok)
    json.dump(out, open(os.path.join(RES, "evr_weight.json"), "w"), indent=2)

    from scipy import stats
    d = [r for r in out if not np.isnan(r["top16_energy"])]
    x = np.array([r["top16_energy"] for r in d]); p = np.array([r["peff"] for r in d])
    y = np.array([r["ref_removal"] for r in d])

    def boot_slope(xx, yy, n=5000):
        rng = np.random.default_rng(0); s = []
        for _ in range(n):
            i = rng.integers(0, len(xx), len(xx))
            if np.ptp(xx[i]) > 0:
                s.append(np.polyfit(xx[i], yy[i], 1)[0])
        return np.percentile(s, [2.5, 97.5])

    print(f"\nW'1 removal ~ top16_energy (n={len(d)}):")
    r = stats.pearsonr(x, y); rho = stats.spearmanr(x, y); lo, hi = boot_slope(x, y)
    print(f"   Pearson r={r.statistic:+.3f} p={r.pvalue:.4f} | Spearman rho={rho.statistic:+.3f} "
          f"p={rho.pvalue:.4f} | slope CI [{lo:+.3f},{hi:+.3f}] -> "
          f"{'PASS (excludes 0)' if lo > 0 else 'FAIL (covers 0)'}")
    print(f"\nW'2 removal ~ peff (effective rank; expect NEGATIVE):")
    r = stats.pearsonr(p, y); rho = stats.spearmanr(p, y)
    print(f"   Pearson r={r.statistic:+.3f} p={r.pvalue:.4f} | Spearman rho={rho.statistic:+.3f} "
          f"p={rho.pvalue:.4f}")
    print("\nwrote results/evr_weight.json")


if __name__ == "__main__":
    main()
