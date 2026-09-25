"""SentenceDebias replay: is the v12frontier-vs-published gap the unseeded PCA?

Holds model, corpus and pooled embeddings fixed; refits only the PCA. Unseeded
PCA(n_components=k) draws from numpy's GLOBAL RNG, so each fit here sets a
different global seed (what differs between the published process and the
v12frontier process), plus one exact fit (svd_solver='full'). Removal is read
with the published projection, bias probe only (MMLU/ppl are not what differs).
"""
import json, os, sys
import numpy as np
import torch
from sklearn.decomposition import PCA

REPO = "/home/jovyan/Blind_spot_spb"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "legacy"))
sys.path.insert(0, os.path.join(REPO, "src"))
import colab_t2t4 as C
import run_sentdebias_baseline as SD

HF = "Qwen/Qwen2.5-3B-Instruct"
# (axis, seed, layer, k, published, v12frontier) -- the two qwen seed-0 misses
CASES = [("occ_gender", 0, 26, 1, 0.4626797160959264, 0.4614788871004354),
         ("crows_socioeconomic", 0, 18, 2, 0.06746983805368115, 0.06422241621239241)]
SEEDS = list(range(8))
OUT = os.path.join(REPO, "results", "v12", "sd_determinism.json")

model, tok = C.load_model(HF, dispatch=False)
res = []
for ax, s, L, k, pub, new in CASES:
    c = json.load(open(C._t2x_corpus_fp("qwen_3b", ax, s)))
    b, d = c["biased"], c["debiased"]
    keep = [i for i in range(min(len(b), len(d))) if b[i] != d[i]]
    bb, dd = [b[i] for i in keep], [d[i] for i in keep]
    a1, b1 = SD.pooled(model, tok, bb, L), SD.pooled(model, tok, dd, L)
    a2 = SD.pooled(model, tok, bb, L)
    emb_det = float(np.abs(a1 - a2).max())
    m = (a1 + b1) / 2
    X = np.concatenate([a1 - m, b1 - m], 0)
    pre = C.axis_skew(model, tok, ax, 6)

    def removal(U):
        pr = SD.Project(model, L, torch.tensor(U, dtype=torch.float32))
        try:
            post = C.axis_skew(model, tok, ax, 6)
        finally:
            pr.undo()
        return abs(pre) - abs(post)

    full = PCA(n_components=k, svd_solver="full").fit(X).components_
    solver = PCA(n_components=k).fit(X)._fit_svd_solver
    fits = []
    for g in SEEDS:
        np.random.seed(g)
        U = PCA(n_components=k).fit(X).components_
        cos = [float(abs(U[i] @ full[i])) for i in range(k)]
        fits.append(dict(global_seed=g, cos_to_full=cos, removal=float(removal(U))))
        print(f"{ax} L{L} k{k} seed{g}: cos={cos} removal={fits[-1]['removal']:+.5f}",
              flush=True)
    r_full = float(removal(full))
    # same fit twice under one seed: is the projection+probe itself deterministic?
    np.random.seed(0); U0 = PCA(n_components=k).fit(X).components_
    r_repeat = float(removal(U0))
    rs = [f["removal"] for f in fits]
    row = dict(axis=ax, seed=s, layer=L, k=k, X_shape=list(X.shape),
               auto_solver=solver, embedding_max_absdiff=emb_det,
               published=pub, v12frontier=new, full_svd=r_full,
               seeded_fits=fits, repeat_seed0=r_repeat,
               spread=[min(rs), max(rs)], range=max(rs) - min(rs),
               pub_in_spread=min(rs) - 1e-9 <= pub <= max(rs) + 1e-9,
               new_in_spread=min(rs) - 1e-9 <= new <= max(rs) + 1e-9)
    res.append(row)
    print(json.dumps({k_: v for k_, v in row.items() if k_ != "seeded_fits"}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
