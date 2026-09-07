"""BL-T/INLP — Iterative Nullspace Projection, ported from McGill-NLP/bias-bench.

INLP (Ravfogel et al. 2020): repeatedly fit a linear classifier that predicts the
protected attribute from a representation, then project the representation onto
that classifier's nullspace. After k iterations the attribute is linearly
undecodable. bias-bench ships it as a debias method; the algorithm is portable,
its harness is not.

Adapted to our setting exactly as SentenceDebias was:
  * representations = mean-pooled hidden state at a chosen decoder layer
  * the two "attribute" classes are the biased vs debiased elicitations of the
    SAME probe item (the same contrast every other method here consumes, so the
    endogeneity boundary holds)
  * identical pairs dropped -- they carry no label information
  * applied as an always-on projection hook, like SentenceDebias

Equal selection space: 2 layers x 2 iteration counts = 4 configs.
Null: random-subspace projection of the same rank (the projection analogue of
steering's matched-norm random direction).
"""
import os, json, traceback
import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
import colab_t2t4 as C
from run_steering_baseline import layers_of
from run_sentdebias_baseline import pooled, Project

TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct", "qwen_3b"),
           ("phi", "microsoft/Phi-3.5-mini-instruct", "phi_3b")]
AXES = ["occ_gender", "crows_socioeconomic"]
SEEDS = [0, 1, 2]
DEPTHS = [0.50, 0.75]
ITERS = [2, 4]
OUT = os.path.join(C.RESULTS, "inlp")


def inlp_directions(X, y, n_iter, seed):
    """Return the n_iter classifier normals whose span INLP projects out."""
    dirs = []
    Xc = X.copy()
    for i in range(n_iter):
        clf = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4,
                            random_state=seed + i)
        clf.fit(Xc, y)
        w = clf.coef_[0].astype(np.float64)
        nw = np.linalg.norm(w)
        if nw < 1e-8:
            break
        w = w / nw
        dirs.append(w)
        Xc = Xc - np.outer(Xc @ w, w)          # project the data off w, then refit
    return np.array(dirs, dtype=np.float32) if dirs else None


def main():
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["layer"], r["n_iter"], r["kind"]))
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
                    if len(keep) < 10:
                        print(f"[inlp] SKIP {fam} {ax} s{s}: {len(keep)} differing pairs",
                              flush=True)
                        continue
                    bb = [b[i] for i in keep]; dd = [d[i] for i in keep]
                    for depth in DEPTHS:
                        L = max(0, min(n - 1, int(round(depth * (n - 1)))))
                        Xb = pooled(model, tok, bb, L)
                        Xd = pooled(model, tok, dd, L)
                        X = np.concatenate([Xb, Xd], 0)
                        y = np.concatenate([np.zeros(len(Xb)), np.ones(len(Xd))])
                        for k in ITERS:
                            for kind in ("real", "null_random"):
                                key = (fam, ax, s, L, k, kind)
                                if key in done:
                                    continue
                                try:
                                    if kind == "real":
                                        D = inlp_directions(X, y, k, s)
                                        if D is None:
                                            continue
                                        U = torch.tensor(D, dtype=torch.float32)
                                    else:
                                        g = rng.standard_normal((k, X.shape[1]))
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
                                            target=fam, axis=ax, seed=s, layer=L,
                                            n_iter=k, kind=kind, n_pairs=len(keep),
                                            bias_reduction=float(red),
                                            collateral_ok=bool(ok),
                                            dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                            ppl_ratio=post["ppl"] / pre["ppl"],
                                            pre_skew=pre["skew"])) + "\n")
                                    print(f"[inlp] {fam:6s} {ax:20s} s{s} L{L:<3d} k={k} "
                                          f"{kind:11s} red={red:+.4f} ok={ok}", flush=True)
                                except Exception:
                                    print(f"[inlp] FAIL {fam} {ax} s{s} L{L} k={k}", flush=True)
                                    traceback.print_exc()
        finally:
            C._free(model)
    print("[inlp] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
