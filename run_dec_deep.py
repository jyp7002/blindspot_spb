"""DEC deep-sparsity tie-in (experiments_v8 §SPC, conditional on Outcome A).

Registered: "Ties into DEC: run C-a at 99.5/99.9% on 2 cells if Outcome A holds
-- does the portable field survive deeper sparsity too?"

So this runner is CONDITIONAL. It refuses to run unless
results/v8/dec_analysis.json records Outcome A, because under Outcome B the
question it asks ("is the portable random-support field still portable when
sparser?") is not the live question -- there would be no portable field to
follow. Running it anyway and reporting it would be an unregistered analysis.

Conditions: C-ref and C-a at density {0.005, 0.001} on 2 cells, 3 seeds, same
scale convention and 4-alpha selection as the main panel.
"""
import os, sys, json, traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C
import v8_edits as V8

DENSITIES = [0.005, 0.001]          # 99.5% and 99.9% sparse
CONDITIONS = ["C-ref", "C-a"]
SEEDS = [0, 1, 2]
ALPHAS = (2, 4, 8, 16)
SCALE_MODE = "ref_tensor"

CELLS = [("gemma", "google/gemma-2-2b-it",     "gemma_3b", "occ_gender"),
         ("qwen",  "Qwen/Qwen2.5-3B-Instruct", "qwen_3b",  "occ_gender")]

OUT = os.path.join(C.RESULTS, "v8dec_deep")


def gate():
    fp = os.path.join(C.RESULTS, "v8", "dec_analysis.json")
    if not os.path.exists(fp):
        raise SystemExit("[deep] REFUSING: dec_analysis.json missing — the main "
                         "DEC panel has not been adjudicated.")
    verdict = json.load(open(fp)).get("adjudication", {}).get("verdict", "")
    if not verdict.startswith("OUTCOME A"):
        raise SystemExit(f"[deep] REFUSING: verdict is '{verdict}'. This arm is "
                         "registered as conditional on Outcome A only.")
    print(f"[deep] gate passed: {verdict}", flush=True)


def main():
    gate()
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    tr_out = os.path.join(OUT, "alpha_trace.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["density"],
                          r["condition"]))
            except Exception:
                pass

    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)

    for fam, hf, dn, ax in CELLS:
        todo = [s for s in SEEDS
                if any((fam, ax, s, d, c) not in done
                       for d in DENSITIES for c in CONDITIONS)]
        if not todo:
            print(f"[deep] {fam}: nothing to do", flush=True)
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
                    for dens in DENSITIES:
                        ref_scales = V8.ref_scale_table(v, dens, s)
                        for cond in CONDITIONS:
                            if (fam, ax, s, dens, cond) in done:
                                continue
                            E, meta = V8.build_edit(v, cond, dens, s,
                                                    scale_mode=SCALE_MODE,
                                                    ref_scales=ref_scales)
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
                                        target=fam, axis=ax, seed=s, density=dens,
                                        condition=cond, alpha=float(al),
                                        dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                        ppl_ratio=post["ppl"] / pre["ppl"],
                                        collateral_ok=bool(ok),
                                        bias_reduction=float(r))) + "\n")
                                if ok:
                                    best = r if np.isnan(best) else max(best, r)
                            del E
                            with open(fp_out, "a") as f:
                                f.write(json.dumps(dict(
                                    panel="v8dec_deep", target=fam, axis=ax, seed=s,
                                    designer=dn, role="self", density=dens,
                                    condition=cond, removal=best,
                                    pre_skew=pre["skew"],
                                    n_flips=meta.get("n_flips"),
                                    bytes=meta.get("bytes"))) + "\n")
                            print(f"[deep] {fam:6s} {ax:12s} s{s} d={dens:<6.3f} "
                                  f"{cond:6s} removal={best:+.4f}", flush=True)
                    del v
                except Exception:
                    print(f"[deep] FAIL {fam} {ax} s{s}", flush=True)
                    traceback.print_exc()
        finally:
            C._free(model)
    print("[deep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
