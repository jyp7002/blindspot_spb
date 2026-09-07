"""DEC robustness arm — the `natural` scale convention (PREREGISTRATION.md v8 §DEC).

Registered before the main panel: "The `natural` convention is additionally run
on 2 cells (gemma|occ_gender, qwen|bbq_Age, 3 seeds) and reported in the
appendix. If the two conventions disagree in DIRECTION, that disagreement is
reported as the headline caveat, not buried."

WHY IT MATTERS NOW. The main panel returned Outcome B (Delta_selection excludes
0 at 79% of R(C-ref)), so the first question a reviewer asks is whether the scale
convention manufactured it. Under `ref_tensor` every condition gets C-ref's
per-tensor amplitude; under `natural` each condition takes mean|v| over its own
support, which hands C-a roughly 0.28x the edit norm.

Note the expected DIRECTION of the check: `natural` should make C-a WEAKER, so
Delta_selection should grow. That would mean `ref_tensor` was the CONSERVATIVE
choice -- generous to C-a -- and Outcome B survives a fortiori. The arm is run
because that reasoning has to be measured, not asserted; if instead C-a improves
under `natural`, the convention is implicated and the caveat is the headline.

C-ref is identical under both conventions by construction (ref_scales ARE C-ref's
natural per-tensor scales), so this arm is really a test of C-a. C-ref is re-run
anyway as the paired reference within each seed.
"""
import os, sys, json, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C
import v8_edits as V8

DENSITY = 0.01
SEEDS = [0, 1, 2]
ALPHAS = (2, 4, 8, 16)
CONDITIONS = ["C-ref", "C-a"]

CELLS = [("gemma", "google/gemma-2-2b-it",     "gemma_3b", "occ_gender"),
         ("qwen",  "Qwen/Qwen2.5-3B-Instruct", "qwen_3b",  "bbq_Age")]

OUT = os.path.join(C.RESULTS, "v8dec_natural")


def main():
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    tr_out = os.path.join(OUT, "alpha_trace.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["condition"]))
            except Exception:
                pass

    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)

    for fam, hf, dn, ax in CELLS:
        todo = [s for s in SEEDS
                if any((fam, ax, s, c) not in done for c in CONDITIONS)]
        if not todo:
            print(f"[nat] {fam}|{ax}: nothing to do", flush=True)
            continue
        model, tok = C.load_model(hf, dispatch=False)
        try:
            pre = C.fast_eval(model, tok, ax, mmlu, wt, 6)
            for s in todo:
                try:
                    corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
                    dwb, _ = C.train_task_vector(model, tok, corp["biased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    dwd, _ = C.train_task_vector(model, tok, corp["debiased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    v = C.contrast_(dwb, dwd)
                    del dwb, dwd
                    # record the norm ratio this arm exists to expose
                    rs = V8.ref_scale_table(v, DENSITY, s)
                    _Er, mr = V8.build_edit(v, "C-ref", DENSITY, s,
                                            scale_mode="natural")
                    _Ea, ma = V8.build_edit(v, "C-a", DENSITY, s,
                                            scale_mode="natural")
                    ratio = ma["frobenius"] / mr["frobenius"] if mr["frobenius"] else float("nan")
                    del _Er, _Ea
                    print(f"[nat] {fam}|{ax} s{s} ||C-a||/||C-ref|| under natural "
                          f"= {ratio:.4f}", flush=True)

                    for cond in CONDITIONS:
                        if (fam, ax, s, cond) in done:
                            continue
                        E, meta = V8.build_edit(v, cond, DENSITY, s,
                                                scale_mode="natural")
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
                                    target=fam, axis=ax, seed=s, condition=cond,
                                    scale_mode="natural", alpha=float(al),
                                    dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                    ppl_ratio=post["ppl"] / pre["ppl"],
                                    collateral_ok=bool(ok),
                                    bias_reduction=float(r))) + "\n")
                            if ok:
                                best = r if np.isnan(best) else max(best, r)
                        del E
                        with open(fp_out, "a") as f:
                            f.write(json.dumps(dict(
                                panel="v8dec_natural", target=fam, axis=ax, seed=s,
                                designer=dn, role="self", condition=cond,
                                scale_mode="natural", density=DENSITY,
                                removal=best, pre_skew=pre["skew"],
                                frobenius=meta.get("frobenius"),
                                norm_ratio_vs_ref=ratio)) + "\n")
                        print(f"[nat] {fam:6s} {ax:12s} s{s} {cond:6s} "
                              f"removal={best:+.4f}", flush=True)
                    del v
                except Exception:
                    print(f"[nat] FAIL {fam} {ax} s{s}", flush=True)
                    traceback.print_exc()
        finally:
            C._free(model)
    print("[nat] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
