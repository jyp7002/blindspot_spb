"""W — EVR (explained-variance ratio) → removability, de-circularizing the
semantic-direction scope law. FROZEN design (see PREREGISTRATION v5):
  model = qwen1.5b; activation = residual hidden_states, all layers, mean-pooled;
  per-item diff = biased - debiased (PROBE half only);
  coherence = per-layer top-1 PCA EVR; AGGREGATION = MEAN across layers.
Circularity guard: EVR is pre-edit, probe-half, aggregation frozen above; no
axis's EVR is recomputed after its removal is known. y = ground-truth-reference
(exogenous) in-budget removal on qwen1.5b, from measured artifact files only.
"""
import os, json, argparse
import numpy as np
import torch

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
MAX_ITEMS = 240   # cap per axis for speed; frozen


def axis_pairs(axis, max_items=MAX_ITEMS):
    """Return (biased_texts, debiased_texts) from the PROBE half of an axis."""
    import crows_axes as CA, bbq_axes as BB, battery_axes as BA
    import contrast_k as CK, m_axes as MA
    if axis == "occ_gender":
        from elicit import POOL_OCC_TEMPLATES, PRON_M, PRON_F
        from probes import INJECT_OCC
        bias, deb = [], []
        for o, g in INJECT_OCC.items():
            for t in POOL_OCC_TEMPLATES:
                sf, sm = t.format(occ=o, **PRON_F), t.format(occ=o, **PRON_M)
                c, i = (sf, sm) if g == "f" else (sm, sf)  # congruent=biased
                bias.append(c); deb.append(i)
        return bias[:max_items], deb[:max_items]
    if axis == "gen_fm":
        from probes import PROBE_VAL_TEMPLATES, PROBE_NEG, CANDIDATE_AXES
        gA, gB = CANDIDATE_AXES[axis]
        bias, deb = [], []
        for t in PROBE_VAL_TEMPLATES:
            for a in PROBE_NEG:            # negative attribute, group A vs B
                bias.append(t.format(G=gA, a=a)); deb.append(t.format(G=gB, a=a))
        return bias[:max_items], deb[:max_items]
    mod = (CA if axis.startswith("crows_") else BB if axis.startswith("bbq_")
           else CK if axis.startswith("ck_") else MA if MA.is_m_axis(axis)
           else BA if BA.is_battery_axis(axis) else None)
    if mod is None:
        raise ValueError(f"no pair source for {axis}")
    probe, _edit = mod.load_axis(axis)
    bias = [p[0] for p in probe][:max_items]
    deb = [p[1] for p in probe][:max_items]
    return bias, deb


@torch.no_grad()
def _hidden(model, tok, texts, batch_size=16):
    """Per-layer mean-pooled hidden states -> array [n_layers, n_texts, hidden]."""
    from common import DEVICE
    dev = next(model.parameters()).device
    outs = None
    for i in range(0, len(texts), batch_size):
        b = texts[i:i + batch_size]
        enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                  max_length=64).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states  # tuple L+1
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = [((h.float() * mask).sum(1) / mask.sum(1)).cpu().numpy() for h in hs]
        pooled = np.stack(pooled, 0)                 # [L+1, b, hidden]
        outs = pooled if outs is None else np.concatenate([outs, pooled], axis=1)
    return outs


def _top1_evr(diffs):
    """top-1 PCA explained-variance ratio of item diff-vectors [n_items, hidden]."""
    X = diffs - diffs.mean(0, keepdims=True)
    if X.shape[0] < 3:
        return float("nan")
    # SVD on centered matrix; EVR_1 = s0^2 / sum(s^2)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    tot = float((s ** 2).sum())
    return float(s[0] ** 2 / tot) if tot > 0 else float("nan")


def axis_evr(model, tok, axis, batch_size=16):
    bias, deb = axis_pairs(axis)
    if len(bias) < 3:
        return float("nan"), 0
    hb = _hidden(model, tok, bias, batch_size)      # [L+1, n, hid]
    hd = _hidden(model, tok, deb, batch_size)
    n = min(hb.shape[1], hd.shape[1])
    diff = hb[:, :n] - hd[:, :n]                     # [L+1, n, hid]
    per_layer = [_top1_evr(diff[l]) for l in range(diff.shape[0])]
    per_layer = [e for e in per_layer if not np.isnan(e)]
    return float(np.mean(per_layer)), n              # MEAN across layers (frozen)


def reference_removal(target="qwen1.5b"):
    """Ground-truth-reference (exogenous) best in-budget removal per axis."""
    rows = [json.loads(l) for l in open(os.path.join(RES, "runs.jsonl"))]
    ref = {}
    for r in rows:
        if r.get("signal") != "exogenous" or r.get("target") != target:
            continue
        if not r.get("collateral_ok"):
            continue
        ax = r["axis"]; br = r.get("bias_reduction")
        if br is None:
            continue
        ref[ax] = max(ref.get(ax, -1e9), br)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    from common import load, free
    ref = reference_removal(a.model)
    axes = sorted(ref.keys())
    print(f"W: {len(axes)} axes with reference removal on {a.model}\n")
    model, tok = load(a.model)
    out = []
    try:
        for ax in axes:
            try:
                evr, n = axis_evr(model, tok, ax, a.batch_size)
            except Exception as e:
                print(f"  {ax:26s} EVR FAIL: {type(e).__name__}: {e}"); continue
            out.append(dict(axis=ax, evr=evr, ref_removal=ref[ax], n_items=n))
            print(f"  {ax:26s} EVR={evr:.4f}  ref_removal={ref[ax]:+.4f}  n={n}",
                  flush=True)
    finally:
        free(model, tok)
    json.dump(out, open(os.path.join(RES, "evr_removability.json"), "w"), indent=2)
    print("\nwrote results/evr_removability.json")


if __name__ == "__main__":
    main()
