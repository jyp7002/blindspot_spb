"""BL-T/SentenceDebias — projection baseline, ported from McGill-NLP/bias-bench.

Faithful port of `bias_bench/debias/sentence_debias.py::compute_gender_subspace`,
adapted from its encoder assumptions to our decoder targets. The upstream steps,
preserved exactly:

  1. mean-pool the hidden states over non-pad tokens for each side of a pair
  2. L2-normalise each pooled embedding
  3. subtract the per-pair mean from BOTH sides (centres each pair at the origin,
     so PCA sees only the within-pair contrast)
  4. concatenate both sides and fit PCA
  5. take the top component(s) as the bias subspace

Upstream hardcodes `PCA(n_components=1)` and comments "We use only the first PCA
component for debiasing". We keep k=1 as one config and add k=2 as a second, which
is the natural way to spend the equal-selection-space budget on a method whose only
real knob is the subspace rank.

DEVIATIONS FROM UPSTREAM, and why:
  - upstream reads `last_hidden_state` from an encoder; we hook a decoder layer, so
    the subspace can be computed (and applied) at a chosen depth. Layer becomes the
    second knob.
  - upstream pairs are attribute-swapped sentences (male/female); ours are the
    biased/debiased elicitations of the SAME probe item, which is the analogous
    contrast and keeps the endogeneity boundary (§1) intact.
  - identical pairs are dropped: when the model cannot articulate a contrast the two
    strings are the same, the difference vector is exactly zero, and such rows would
    pull the PCA toward the origin without carrying signal. (Measured: 8-54% of pairs
    differ; the differing fraction correlates +0.79 with contrast_gap.)

APPLICATION. Debiasing = project the residual stream orthogonal to the subspace:
    h <- h - sum_k <h, u_k> u_k
This is a PROJECTION, not an addition, so unlike steering it has no strength knob --
which is precisely why the 4 configs are 2 layers x 2 ranks.

Protocol: always-on (active during probes, MMLU and perplexity), strict budget,
3 seeds, both-null discipline is N/A for a projection with no trained direction --
instead we report a RANDOM-SUBSPACE null of the same rank, the projection analogue
of steering's matched-norm random direction.

Resume key: (target, axis, seed, layer, k, kind).
"""
import os, json, traceback
import numpy as np
import torch
from sklearn.decomposition import PCA
import colab_t2t4 as C
from run_steering_baseline import layers_of

TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct", "qwen_3b"),
           ("phi", "microsoft/Phi-3.5-mini-instruct", "phi_3b"),
           ("gemma", "google/gemma-2-2b-it", "gemma_3b"),
           ("llama", "meta-llama/Llama-3.2-3B-Instruct", "llama_3b")]
AXES = ["occ_gender", "crows_socioeconomic"]
SEEDS = [0, 1, 2]
DEPTHS = [0.50, 0.75]          # 2 layers x
RANKS = [1, 2]                 # 2 ranks   = 4 configs (equal selection space)
OUT = os.path.join(C.RESULTS, "sentdebias")


@torch.no_grad()
def pooled(model, tok, texts, layer_idx, batch_size=8, max_len=128):
    """Mean-pooled, L2-normalised hidden state at `layer_idx` (upstream steps 1-2)."""
    layers = layers_of(model)
    out = []
    orig_side = tok.padding_side          # restore: eval_mmlu is padding-sensitive
    try:
        tok.padding_side = "right"
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            enc = {k: v.to(C.DEVICE) for k, v in enc.items()}
            mask = enc["attention_mask"]
            grab = {}

            def hook(_m, _i, o):
                grab["h"] = o[0] if isinstance(o, tuple) else o
            hd = layers[layer_idx].register_forward_hook(hook)
            try:
                model(**enc)
            finally:
                hd.remove()
            h = grab["h"].float()
            m = mask.unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            out.append(emb.cpu().numpy())
    finally:
        tok.padding_side = orig_side
    return np.concatenate(out, 0)


def bias_subspace(model, tok, biased, debiased, layer_idx, k):
    """Upstream steps 3-5: per-pair centring, concatenate, PCA, top-k components."""
    a = pooled(model, tok, biased, layer_idx)
    b = pooled(model, tok, debiased, layer_idx)
    means = (a + b) / 2.0
    a = a - means
    b = b - means
    allv = np.concatenate([a, b], 0)
    p = PCA(n_components=k).fit(allv)
    return torch.tensor(p.components_, dtype=torch.float32)     # (k, hidden)


class Project:
    """h <- h - sum_k <h,u_k> u_k  at one layer, active on every forward pass."""

    def __init__(self, model, layer_idx, U):
        Q = U / U.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        Q = Q.to(C.DEVICE)

        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            q = Q.to(h.dtype)
            proj = (h @ q.T) @ q            # component inside the subspace
            hh = h - proj
            return (hh,) + o[1:] if isinstance(o, tuple) else hh

        self.h = layers_of(model)[layer_idx].register_forward_hook(hook)

    def undo(self):
        if self.h is not None:
            self.h.remove(); self.h = None


def main():
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["layer"], r["k"], r["kind"]))
            except Exception:
                pass
    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)
    rng = np.random.default_rng(0)

    for fam, hf, self_dn in TARGETS:
        model, tok = C.load_model(hf, dispatch=False)
        try:
            n = len(layers_of(model))
            pre_cache = {}
            for ax in AXES:
                if ax not in pre_cache:
                    pre_cache[ax] = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                pre = pre_cache[ax]
                for s in SEEDS:
                    c = json.load(open(C._t2x_corpus_fp(self_dn, ax, s)))
                    b, d = c["biased"], c["debiased"]
                    m = min(len(b), len(d))
                    keep = [i for i in range(m) if b[i] != d[i]]
                    if len(keep) < 5:
                        print(f"[sd] SKIP {fam} {ax} s{s}: {len(keep)} differing pairs",
                              flush=True)
                        continue
                    bb = [b[i] for i in keep]; dd = [d[i] for i in keep]
                    for depth in DEPTHS:
                        L = max(0, min(n - 1, int(round(depth * (n - 1)))))
                        for k in RANKS:
                            for kind in ("real", "null_random"):
                                key = (fam, ax, s, L, k, kind)
                                if key in done:
                                    continue
                                try:
                                    if kind == "real":
                                        U = bias_subspace(model, tok, bb, dd, L, k)
                                    else:
                                        g = rng.standard_normal((k, model.config.hidden_size))
                                        U = torch.tensor(g, dtype=torch.float32)
                                    pr = Project(model, L, U)
                                    try:
                                        post = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                                    finally:
                                        pr.undo()
                                    ok = C.collateral_ok(pre, post)
                                    red = C.bias_reduction(pre, post)
                                    with open(fp_out, "a") as f:
                                        f.write(json.dumps(dict(
                                            target=fam, axis=ax, seed=s, layer=L, k=k,
                                            kind=kind, n_pairs=len(keep),
                                            bias_reduction=float(red),
                                            collateral_ok=bool(ok),
                                            dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                            ppl_ratio=post["ppl"] / pre["ppl"],
                                            pre_skew=pre["skew"])) + "\n")
                                    print(f"[sd] {fam:6s} {ax:20s} s{s} L{L:<3d} k={k} "
                                          f"{kind:11s} red={red:+.4f} ok={ok}", flush=True)
                                except Exception:
                                    print(f"[sd] FAIL {fam} {ax} s{s} L{L} k={k} {kind}",
                                          flush=True)
                                    traceback.print_exc()
        finally:
            C._free(model)
    print("[sd] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
