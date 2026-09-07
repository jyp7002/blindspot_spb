"""SPC (experiments_v8) — sparsity curve to 99.9% on the frozen protocol.

Question: where is the phase transition? RESULTS_METHOD.md §5 stops at 99% and
finds no measurable cost. v8 pushes to 99.9% because a collapse between 99 and
99.9 is a self-contained headline sub-figure, and a flat curve shrinks the patch
numbers further and feeds DEC's field-redundancy interpretation.

Structure follows run_ablate_attn.py exactly: train the two task vectors ONCE per
(target, axis, seed, designer) and apply every sparsity level to the same contrast
vector, so the curve is exactly paired and the trainings are not repeated 8 times.
binarize() must stay NON-inplace here for the same reason as in run_ablate_attn:
v is reused by every later level.

X-axes are reported in triplicate per §SPC: retained-parameter fraction, nonzero
count, and patch bytes -- all three come straight out of binarize()'s meta, which
already emits n_flips/n_params/effective_sparsity/bytes.

Resumable at (target, axis, seed, designer, variant).
"""
import os, sys, json, traceback
import numpy as np
import colab_t2t4 as C

TIER = sys.argv[1] if len(sys.argv) > 1 else "small"

# SPC grid, frozen in PREREGISTRATION.md before the first run.
SPARSITIES = [0.0, 0.50, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999]

TARGETS_SMALL = [("qwen",  "Qwen/Qwen2.5-3B-Instruct",      "qwen_3b"),
                 ("phi",   "microsoft/Phi-3.5-mini-instruct", "phi_3b"),
                 ("gemma", "google/gemma-2-2b-it",           "gemma_3b"),
                 ("llama", "meta-llama/Llama-3.2-3B-Instruct", "llama_3b")]
TARGETS_BIG = [("qwen",  "Qwen/Qwen2.5-7B-Instruct",       "qwen"),
               ("llama", "meta-llama/Llama-3.1-8B-Instruct", "llama")]

AXES = ["occ_gender"]
SEEDS = [0, 1, 2]
ALPHAS = (2, 4, 8, 16)

TARGETS = TARGETS_SMALL if TIER == "small" else TARGETS_BIG
OUT = os.path.join(C.RESULTS, "v8spc", TIER)


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

    for fam, hf, dn in TARGETS:
        todo = [(ax, s) for ax in AXES for s in SEEDS
                if any((fam, ax, s, dn, f"s{sp}") not in done for sp in SPARSITIES)]
        if not todo:
            print(f"[spc] {fam}: nothing to do", flush=True)
            continue
        model, tok = C.load_model(hf, dispatch=False)
        try:
            pre_cache = {}
            for ax, s in todo:
                try:
                    if ax not in pre_cache:
                        pre_cache[ax] = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                    pre = pre_cache[ax]
                    corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
                    dwb, _ = C.train_task_vector(model, tok, corp["biased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    dwd, _ = C.train_task_vector(model, tok, corp["debiased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    v = C.contrast_(dwb, dwd)
                    del dwb, dwd
                    for sp in SPARSITIES:
                        vname = f"s{sp}"
                        if (fam, ax, s, dn, vname) in done:
                            continue
                        E, meta = C.binarize(v, "per_tensor", sp, s)   # non-inplace
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
                                    target=fam, axis=ax, seed=s, designer=dn,
                                    variant=vname, sparsity=sp, alpha=float(al),
                                    dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                    ppl_ratio=post["ppl"] / pre["ppl"],
                                    collateral_ok=bool(ok),
                                    bias_reduction=float(r))) + "\n")
                            if ok:
                                best = r if np.isnan(best) else max(best, r)
                        del E
                        with open(fp_out, "a") as f:
                            f.write(json.dumps(dict(
                                panel="spc", tier=TIER, target=fam, axis=ax, seed=s,
                                role="self", designer=dn, variant=vname, sparsity=sp,
                                removal=best, pre_skew=pre["skew"],
                                bytes=meta.get("bytes"), n_params=meta.get("n_params"),
                                n_flips=meta.get("n_flips"),
                                effective_sparsity=meta.get("effective_sparsity"))) + "\n")
                        print(f"[spc] {fam:5s} {ax:12s} s{s} sp={sp:<6.3f} "
                              f"nnz={meta.get('n_flips'):>10,d} "
                              f"MB={meta.get('bytes',0)/1e6:7.3f} "
                              f"removal={best:+.4f}", flush=True)
                    del v
                except Exception:
                    print(f"[spc] FAIL {fam} {ax} s{s}", flush=True)
                    traceback.print_exc()
        finally:
            C._free(model)
    print("[spc] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
