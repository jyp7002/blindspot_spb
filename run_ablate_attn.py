"""Method-paper ablations on the FROZEN attn protocol (C3/C5/C6).

C3 (binarization cost vs full precision), C5 (scale granularity) and C6 (bit
budget) currently exist ONLY on the untagged v2-era module set, at n=60 and n=8.
Every headline result in v5/v6 uses the frozen attn (q,k,v,o) r16 protocol. If the
binary debiaser is the paper's main contribution, its efficiency and ablation
claims must sit on the same edit configuration as its headline numbers.

EFFICIENCY: the two task-vector trainings dominate cost, and every variant is a
different post-processing of the SAME contrast vector. So this trains once per
(target, axis, seed, designer) and applies all 7 variants to it -- ~7x cheaper
than running each variant end to end, and it also makes the comparison exactly
paired (identical dW, so any difference is attributable to the variant alone).

MEMORY: binarize() must stay NON-inplace — it is called once per variant and
inplace=True would consume the shared vector, corrupting every later variant.
contrast_(), by contrast, is safe AND necessary: it returns v in dwb's own storage
and empties dwd, so dwb/dwd are never needed again. Using the non-inplace
contrast() would hold dwb+dwd+v simultaneously = 25.8 GB at 7-9B, against a 31 GB
host budget -- i.e. right at the wall that OOM-killed this box 9 times. With
contrast_ the peak is 17.2 GB.

Resumable at (target, axis, seed, designer, variant).
"""
import os, json, traceback
import numpy as np
import colab_t2t4 as C

import sys
TIER = sys.argv[1] if len(sys.argv) > 1 else "small"
TARGETS_BIG = [("qwen", "Qwen/Qwen2.5-7B-Instruct", "qwen"),
               ("llama", "meta-llama/Llama-3.1-8B-Instruct", "llama")]
CROSS_BIG = {"qwen": ("llama", "meta-llama/Llama-3.1-8B-Instruct"),
             "llama": ("qwen", "Qwen/Qwen2.5-7B-Instruct")}
TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct", "qwen_3b"),
           ("phi",  "microsoft/Phi-3.5-mini-instruct", "phi_3b"),
           # breadth extension: C5/C6 (esp. the s=0.99 headline) rested on 2 targets
           ("gemma", "google/gemma-2-2b-it", "gemma_3b"),
           ("llama", "meta-llama/Llama-3.2-3B-Instruct", "llama_3b")]
# self + one cross designer, so the ablation is not a single-designer artifact
CROSS = {"qwen":  ("llama_3b", "meta-llama/Llama-3.2-3B-Instruct"),
         "phi":   ("qwen_3b", "Qwen/Qwen2.5-3B-Instruct"),
         "gemma": ("qwen_3b", "Qwen/Qwen2.5-3B-Instruct"),
         "llama": ("qwen_3b", "Qwen/Qwen2.5-3B-Instruct")}
AXES = ["occ_gender", "crows_socioeconomic"]
SEEDS = [0, 1, 2]
ALPHAS = (2, 4, 8, 16)
# (name, granularity, sparsity) — None granularity = full precision, no binarization
VARIANTS = [("fp", None, 0.0),
            ("binary-per_tensor-s0.0", "per_tensor", 0.0),
            ("binary-per_channel-s0.0", "per_channel", 0.0),
            ("binary-scalar-s0.0", "scalar", 0.0),
            ("binary-per_tensor-s0.5", "per_tensor", 0.5),
            ("binary-per_tensor-s0.9", "per_tensor", 0.9),
            ("binary-per_tensor-s0.99", "per_tensor", 0.99)]

if TIER == "big":                    # 7-9B: self designer only, to bound cost
    TARGETS, CROSS = TARGETS_BIG, CROSS_BIG
OUT = os.path.join(C.RESULTS, "ablate" if TIER == "small" else "ablate_big")


def main():
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    tr_out = os.path.join(OUT, "alpha_trace.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["designer"], r["variant"]))
            except Exception:
                pass
    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)

    for fam, hf, self_dn in TARGETS:
        # SC7 (experiments_v7): the 7-9B arm was self-designer only (n=12), too thin
        # to carry the "no cost at scale" sentence. SC7=1 adds the cross designer so
        # the paired fp-vs-binary estimate rests on >=2 designer classes.
        designers = ([("self", self_dn)] if (TIER == "big" and not os.environ.get("SC7"))
                     else [("self", self_dn), ("cross", CROSS[fam][0])])
        todo = [(ax, s, role, dn) for ax in AXES for s in SEEDS for role, dn in designers
                if any((fam, ax, s, dn, v) not in done for v, _, _ in VARIANTS)]
        if not todo:
            print(f"[ablate] {fam}: nothing to do", flush=True); continue
        model, tok = C.load_model(hf, dispatch=False)
        try:
            pre_cache = {}
            for ax, s, role, dn in todo:
                try:
                    if ax not in pre_cache:
                        pre_cache[ax] = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                    pre = pre_cache[ax]
                    corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
                    dwb, _ = C.train_task_vector(model, tok, corp["biased"], rank=16, steps=250,
                                                 lr=1e-4, seed=s, bs=8, targets=C.ATTN,
                                                 grad_checkpoint=True)
                    dwd, _ = C.train_task_vector(model, tok, corp["debiased"], rank=16, steps=250,
                                                 lr=1e-4, seed=s, bs=8, targets=C.ATTN,
                                                 grad_checkpoint=True)
                    v = C.contrast_(dwb, dwd)   # in-place: v reuses dwb storage, dwd freed
                    del dwb, dwd                # (v holds the only remaining reference)
                    for vname, gran, sp in VARIANTS:
                        if (fam, ax, s, dn, vname) in done:
                            continue
                        if gran is None:
                            E, meta = {k: t.clone() for k, t in v.items()}, dict(kind="fp")
                        else:
                            E, meta = C.binarize(v, gran, sp, s)   # non-inplace: v survives
                        best = float("nan")
                        for al in ALPHAS:
                            undo = C.apply_edit(model, E, alpha=al, sign=-1.0)
                            try:
                                post = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                            finally:
                                undo()
                            ok = C.collateral_ok(pre, post)
                            r = C.bias_reduction(pre, post)
                            with open(tr_out, "a") as f:
                                f.write(json.dumps(dict(
                                    target=fam, axis=ax, seed=s, role=role, designer=dn,
                                    variant=vname, alpha=float(al),
                                    dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                    ppl_ratio=post["ppl"] / pre["ppl"],
                                    collateral_ok=bool(ok), bias_reduction=float(r))) + "\n")
                            if ok:
                                best = r if np.isnan(best) else max(best, r)
                        del E
                        with open(fp_out, "a") as f:
                            f.write(json.dumps(dict(
                                target=fam, axis=ax, seed=s, role=role, designer=dn,
                                variant=vname, removal=best, pre_skew=pre["skew"],
                                bytes=meta.get("bytes"), n_params=meta.get("n_params"),
                                effective_sparsity=meta.get("effective_sparsity"))) + "\n")
                        print(f"[ablate] {fam:5s} {ax:20s} s{s} {role:5s}<-{dn:10s} "
                              f"{vname:24s} removal={best:+.4f}", flush=True)
                    del v
                except Exception:
                    print(f"[ablate] FAIL {fam} {ax} s{s} {role}<-{dn}", flush=True)
                    traceback.print_exc()
        finally:
            C._free(model)
    print("[ablate] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
