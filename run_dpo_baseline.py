"""BL-T/DPO — trained baseline via TRL's DPOTrainer (experiments_v7 §BL-T).

Uses the upstream implementation (huggingface/trl 1.9.2) rather than a
reimplementation. TRL installed without moving transformers/peft/torch, so the
frozen protocol's bit-determinism is intact.

WHY DPO IS A FAIR BUT NOT IDENTICAL COMPARATOR. DPO optimises a *preference*
(prefer the debiased continuation over the biased one); the edit optimises a
*contrast direction* in weight space. Both consume the SAME endogenous elicited
corpora -- biased[i] and debiased[i] are the same probe item, so index pairing is
the natural preference pair -- which satisfies §1's endogeneity boundary. A DPO
win therefore means "preference training beats direction editing on this signal",
not "the signal was wrong". Stated here so the paper cannot overclaim either way.

PROTOCOL (experiments_v7 §1, all three are v3 protocol-error fixes):
  merge-scale alpha   DPO trains a LoRA; we materialise dW = (B@A)*scaling exactly
                      as train_task_vector does and apply it at alpha <= 1. Applying
                      a trained adapter at alpha 2-16 like the sign edit would be a
                      strawman -- the adapter is already at its trained scale.
  equal selection     4 alphas {0.25, 0.5, 0.75, 1.0} == the edit's 4 configs and
    space             steering's 4.
  always-on           evaluated with the merged delta active during probes, MMLU
                      and perplexity, via the same fast_eval/collateral_ok path.
  budget-fail -> 0    same nan->0 rule as every other panel.

Same module set as the edit (attn q,k,v,o r16) so the comparison isolates the
objective, not the parameterisation.

Resume key: (target, axis, seed, alpha).
"""
import os, json, traceback
import numpy as np
import torch
import colab_t2t4 as C

TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct", "qwen_3b"),
           ("phi", "microsoft/Phi-3.5-mini-instruct", "phi_3b"),
           ("gemma", "google/gemma-2-2b-it", "gemma_3b"),
           ("llama", "meta-llama/Llama-3.2-3B-Instruct", "llama_3b")]
AXES = ["occ_gender", "crows_socioeconomic"]
SEEDS = [0, 1, 2]
ALPHAS = (0.25, 0.5, 0.75, 1.0)        # merge scale, <= 1 (frozen input)
MIN_PAIRS = 20                          # below this DPO has no usable signal
PROMPT = ""                              # completions are standalone sentences
OUT = os.path.join(C.RESULTS, "dpo")


def build_pairs(dn, ax, seed):
    """Preference pairs from the SAME cached corpus the edit uses.

    elicit_selfdebias builds biased[] and debiased[] from one shuffled item list,
    so index i is the same probe item in both -- the pairing is meaningful, not
    an arbitrary zip.
    """
    c = json.load(open(C._t2x_corpus_fp(dn, ax, seed)))
    b, d = c["biased"], c["debiased"]
    n = min(len(b), len(d))
    # IDENTICAL PAIRS CARRY NO PREFERENCE SIGNAL and must be dropped: elicit_selfdebias
    # picks between the same two candidate continuations under the biased and the
    # debiased prompt, so whenever the model cannot articulate a difference it returns
    # the SAME string in both arms. Measured: only 8-54% of pairs differ, and the
    # differing fraction correlates +0.79 with contrast_gap -- i.e. contrast_gap IS
    # (approximately) the fraction of items on which the model can express a contrast.
    # Feeding chosen==rejected to DPO is a zero-gradient no-op that silently dilutes
    # the effective training set.
    pairs = [dict(prompt=PROMPT, chosen=d[i], rejected=b[i])
             for i in range(n) if d[i] != b[i]]
    return pairs, n


def dpo_delta(model_id, pairs, seed, train_bs=2, steps=250):
    """Train DPO-LoRA on the target and return the materialised dW per module."""
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig
    model, tok = C.load_model(model_id, dispatch=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    targets = C.resolve_targets(model, C.ATTN)
    cfg = DPOConfig(output_dir=os.path.join(OUT, "_trl_tmp"), beta=0.1,
                    learning_rate=1e-4, per_device_train_batch_size=train_bs,
                    max_steps=steps, logging_steps=10_000, save_strategy="no",
                    report_to=[], bf16=True, gradient_checkpointing=True,
                    remove_unused_columns=False, max_length=128, seed=seed)
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                          task_type="CAUSAL_LM", target_modules=targets)
    tr = DPOTrainer(model=model, ref_model=None, args=cfg,
                    train_dataset=Dataset.from_list(pairs),
                    processing_class=tok, peft_config=peft_cfg)
    tr.train()
    pm = tr.model
    scaling = 32 / 16
    dw = {}
    for name, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A = mod.lora_A["default"].weight.detach().float()
            B = mod.lora_B["default"].weight.detach().float()
            key = name.replace("base_model.model.", "").replace(".base_layer", "")
            dw[key] = (B @ A * scaling).cpu()
    del tr, pm
    return dw, model, tok


def main():
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    tr_out = os.path.join(OUT, "alpha_trace.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["alpha"]))
            except Exception:
                pass
    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)

    for fam, hf, self_dn in TARGETS:
        for ax in AXES:
            for s in SEEDS:
                if all((fam, ax, s, float(a)) in done for a in ALPHAS):
                    continue
                try:
                    pairs, n_all = build_pairs(self_dn, ax, s)
                    if len(pairs) < MIN_PAIRS:
                        print(f"[dpo] SKIP {fam} {ax} s{s}: only {len(pairs)}/{n_all} "
                              f"informative pairs (< {MIN_PAIRS}) — recorded, not imputed",
                              flush=True)
                        with open(fp_out, "a") as f:
                            f.write(json.dumps(dict(
                                target=fam, axis=ax, seed=s, alpha=None, method="dpo",
                                removal=None, collateral_ok=False, skipped="too_few_pairs",
                                n_informative=len(pairs), n_total=n_all)) + "\n")
                        continue
                    dw, model, tok = dpo_delta(hf, pairs, s)
                    try:
                        pre = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                        best = float("nan")
                        for a in ALPHAS:
                            undo = C.apply_edit(model, dw, alpha=a, sign=-1.0)
                            try:
                                post = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                            finally:
                                undo()
                            ok = C.collateral_ok(pre, post)
                            red = C.bias_reduction(pre, post)
                            with open(tr_out, "a") as f:
                                f.write(json.dumps(dict(
                                    target=fam, axis=ax, seed=s, alpha=float(a),
                                    dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                    ppl_ratio=post["ppl"] / pre["ppl"],
                                    collateral_ok=bool(ok),
                                    bias_reduction=float(red))) + "\n")
                            if ok:
                                best = red if np.isnan(best) else max(best, red)
                            with open(fp_out, "a") as f:
                                f.write(json.dumps(dict(
                                    target=fam, axis=ax, seed=s, alpha=float(a),
                                    method="dpo", removal=float(red),
                                    collateral_ok=bool(ok),
                                    pre_skew=pre["skew"])) + "\n")
                        print(f"[dpo] {fam:6s} {ax:20s} s{s} pairs={len(pairs)}/{n_all} "
                              f"best_in_budget={best:+.4f}", flush=True)
                    finally:
                        C._free(model)
                except Exception:
                    print(f"[dpo] FAIL {fam} {ax} s{s}", flush=True)
                    traceback.print_exc()
    print("[dpo] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
