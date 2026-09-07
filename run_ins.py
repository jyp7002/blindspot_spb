"""INS (experiments_v8, re-scoped) — generation collateral + the real base/instruct contrast.

WHY THIS IS NOT v8's INS AS WRITTEN. v8 §INS calls itself "the deliberate instruct
arm" and retires a base-only invariant. That invariant never applied to the method
panels: colab_t2t4.FAM_MODELS is instruct/-it at every size, so every published
method number is already measured on instruction-tuned models, and v8's named INS
targets (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct) are the SAME checkpoints
already used as the 7-9B targets. See the amendment in PREREGISTRATION.md v8.

So INS splits:

  INS-A  generation collateral (the genuinely new measurement): IFEval, which the
         likelihood-only budget has never covered. Instruct targets, seed 0,
         IFEval on the first 200 of 541 prompts -- a STATED sampling bound.

  INS-B  the real base<->instruct contrast: gemma-2-2b (base) vs gemma-2-2b-it,
         same family and size, both cached. Onboarding gates re-run per target;
         pre_skew is measured on the base model, never inherited.

Run: python3 run_ins.py [A|B]
"""
import os, sys, json, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C
import v8_edits as V8

ARM = (sys.argv[1] if len(sys.argv) > 1 else "A").upper()

# ---- the base-model prompt problem, and the control for it -------------------
# google/gemma-2-2b (base) ships NO chat template, and colab_t2t4.chat_prompt's
# fallback also calls apply_chat_template, so it raises too. chat_prompt feeds
# mmlu_predict and elicit_selfdebias, and MMLU sits inside fast_eval, which runs
# at every gate and every alpha -- so the harness cannot evaluate ANY base model.
# That is the mechanical reason this project's method line has always been
# instruct-only.
#
# Simply giving the base model a plain-text format would confound "instruction
# tuning" with "prompt format": the instruct arm would be read through gemma's
# chat template and the base arm through plain text. Arm "BP" therefore forces
# the SAME minimal plain-text template on BOTH checkpoints, so the only thing
# that differs is the weights. Its absolute numbers are NOT comparable to the
# main tables (which use real chat templates) -- only the base-vs-instruct
# contrast within BP is. Arm B is kept as the format control: gemma2b_it appears
# in both, so B vs BP measures how much the prompt format alone moves the cell.
PLAIN_TEMPLATE = (
    "{% for m in messages %}{{ m['content'] }}"
    "{% if not loop.last %}\n\n{% endif %}{% endfor %}")
OUT = os.path.join(C.RESULTS, "v8ins", ARM)  # A / B / BP
ALPHAS = (2, 4, 8, 16)
IFEVAL_LIMIT = 200          # stated bound, not full 541

# (label, hf_id, designer_name, is_base, role)
TARGETS_A = [("qwen7b_it",  "Qwen/Qwen2.5-7B-Instruct",       "qwen",  False, "self"),
             ("llama8b_it", "meta-llama/Llama-3.1-8B-Instruct", "llama", False, "self")]

# INS-B holds the EDIT SIGNAL FIXED and varies only the target checkpoint.
# Both arms use the gemma_3b corpus, elicited from gemma-2-2b-it. For the
# instruct target that is genuinely self-elicitation; for the BASE target it is
# the instruct sibling's corpus, and it is labelled `sibling_instruct`, not
# `self`. That asymmetry is the point: eliciting separately from each checkpoint
# would confound "instruction-tuned target" with "instruction-tuned corpus", and
# the registered question is whether instruction tuning moves the CELL out of the
# operating envelope, which requires the signal to be held constant.
TARGETS_B = [("gemma2b_it",   "google/gemma-2-2b-it", "gemma_3b", False, "self"),
             ("gemma2b_base", "google/gemma-2-2b",    "gemma_3b", True,
              "sibling_instruct")]

AXES_A = ["occ_gender"]
AXES_B = ["occ_gender", "bbq_Age"]
SEEDS_A = [0]
SEEDS_B = [0, 1, 2]

# unedited is measured as the `pre` row; the rest are edit conditions
CONDITIONS = ["fp", "sign_only", "sparse99", "random_sign"]


def build(v, cond, seed):
    if cond == "fp":
        return {k: t.clone() for k, t in v.items()}, dict(kind="fp")
    if cond == "sign_only":
        return C.binarize(v, "per_tensor", 0.0, seed)
    if cond == "sparse99":
        return C.binarize(v, "per_tensor", 0.99, seed)
    if cond == "random_sign":
        return C.binarize(v, "per_tensor", 0.0, seed, random_signs=True)
    raise ValueError(cond)


_LM_CACHE = {}


def ifeval(model, tok, limit=IFEVAL_LIMIT):
    """IFEval on the live, already-edited model object. Returns None on failure."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=8)
        res = lm_eval.simple_evaluate(model=lm, tasks=["ifeval"], limit=limit,
                                      verbosity="ERROR")
        r = res["results"]["ifeval"]
        return {k: float(v) for k, v in r.items()
                if isinstance(v, (int, float)) and "stderr" not in k}
    except Exception as e:
        print(f"[ins] IFEval FAILED: {type(e).__name__}: {e}", flush=True)
        return None


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

    targets = TARGETS_A if ARM == "A" else TARGETS_B
    axes = AXES_A if ARM == "A" else AXES_B
    seeds = SEEDS_A if ARM == "A" else SEEDS_B
    run_ifeval = (ARM == "A")
    force_plain = (ARM == "BP")

    # v11: probe size travels with the panel (see src/v11_selftest.py).
    mmlu = C.load_mmlu(n=int(os.environ.get("INS_MMLU_N", "200")), seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)

    for label, hf, dn, is_base, role in targets:
        todo = [(ax, s) for ax in axes for s in seeds
                if any((label, ax, s, c) not in done for c in CONDITIONS)]
        if not todo:
            print(f"[ins] {label}: nothing to do", flush=True)
            continue
        model, tok = C.load_model(hf, dispatch=False)
        if force_plain:
            tok.chat_template = PLAIN_TEMPLATE
            print(f"[ins] {label}: forced plain-text prompt template "
                  "(format-controlled base/instruct comparison)", flush=True)
        try:
            gates = {}
            for ax, s in todo:
                try:
                    if ax not in gates:
                        # ONBOARDING GATE, re-measured on THIS checkpoint.
                        pre = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                        gates[ax] = pre
                        gfp = os.path.join(OUT, "onboard.json")
                        g = json.load(open(gfp)) if os.path.exists(gfp) else {}
                        g[f"{label}|{ax}"] = dict(pre_skew=pre["skew"],
                                                  mmlu=pre["mmlu_acc"], ppl=pre["ppl"],
                                                  is_base=is_base, role=role,
                                                  designer=dn)
                        json.dump(g, open(gfp, "w"), indent=1)
                        print(f"[ins] GATE {label:14s} {ax:12s} "
                              f"pre_skew={pre['skew']:+.4f} mmlu={pre['mmlu_acc']:.3f} "
                              f"ppl={pre['ppl']:.2f}", flush=True)
                        if run_ifeval and "unedited" not in done:
                            base_if = ifeval(model, tok)
                            if base_if:
                                with open(fp_out, "a") as f:
                                    f.write(json.dumps(dict(
                                        panel="v8ins", arm=ARM, target=label, axis=ax,
                                        seed=s, condition="unedited", removal=0.0,
                                        pre_skew=pre["skew"], ifeval=base_if,
                                        ifeval_limit=IFEVAL_LIMIT)) + "\n")
                                print(f"[ins] {label} unedited IFEval={base_if}",
                                      flush=True)
                    pre = gates[ax]

                    corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
                    dwb, _ = C.train_task_vector(model, tok, corp["biased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    dwd, _ = C.train_task_vector(model, tok, corp["debiased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    v = C.contrast_(dwb, dwd)
                    del dwb, dwd

                    for cond in CONDITIONS:
                        if (label, ax, s, cond) in done:
                            continue
                        E, _meta = build(v, cond, s)
                        best, best_alpha = float("nan"), None
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
                                    target=label, axis=ax, seed=s, condition=cond,
                                    alpha=float(al),
                                    dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                    ppl_ratio=post["ppl"] / pre["ppl"],
                                    collateral_ok=bool(ok),
                                    bias_reduction=float(r))) + "\n")
                            if ok and (np.isnan(best) or r > best):
                                best, best_alpha = r, al

                        # IFEval at the SELECTED alpha only (generation is expensive)
                        ife = None
                        if run_ifeval and best_alpha is not None:
                            undo = C.apply_edit(model, E, alpha=best_alpha, sign=-1.0)
                            try:
                                ife = ifeval(model, tok)
                            finally:
                                undo()
                        del E
                        with open(fp_out, "a") as f:
                            f.write(json.dumps(dict(
                                panel="v8ins", arm=ARM, target=label, axis=ax, seed=s,
                                condition=cond, removal=best, pre_skew=pre["skew"],
                                selected_alpha=best_alpha, is_base=is_base,
                                role=role, designer=dn,
                                ifeval=ife, ifeval_limit=IFEVAL_LIMIT)) + "\n")
                        print(f"[ins] {label:14s} {ax:12s} s{s} {cond:12s} "
                              f"removal={best:+.4f} a={best_alpha} "
                              f"ifeval={(ife or {}).get('inst_level_strict_acc,none')}",
                              flush=True)
                    del v
                except Exception:
                    print(f"[ins] FAIL {label} {ax} s{s}", flush=True)
                    traceback.print_exc()
        finally:
            C._free(model)
    print("[ins] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
