"""Prompt-level self-debiasing baseline (Schick 2021) — design.md §5.5.

The method paper's central weakness: every comparator run so far is INTERNAL
(full-precision negation, random-sign control, the two nulls). The obvious
reviewer question for a weight-editing method is "why not just prompt?", and it
has never been measured in this program. `results/pilot_selfdebias.json` is the
elicitation diagnostic, not a debiasing outcome.

DESIGN. The weight edit is always-on: once applied it affects every forward pass,
and it is scored on bias removal AND collateral (MMLU, perplexity) under a strict
budget. A deployed debiasing instruction is also always-on. So the fair
comparison prepends the instruction to EVERYTHING -- the bias probes, the MMLU
items and the perplexity corpus -- and scores it with the identical
bias_reduction / collateral_ok machinery. Prompting that buys bias reduction by
degrading MMLU or perplexity is not free, and the budget is what reveals that.

Reported per (target, axis):
    removal_prompt   |skew_none| - |skew_prompt|      (same estimand as the edit)
    dmmlu, ppl_ratio under the prompt                  (same budget test)
    in_budget                                          (dmmlu<=0.02, pplr<=1.10)
against the measured edit removal from the frozen-protocol panels.

Three prompt strengths, because a single wording would confound "prompting fails"
with "that prompt failed".
"""
import os, json, itertools
import numpy as np
import torch
import colab_t2t4 as C

TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct"),
           ("phi", "microsoft/Phi-3.5-mini-instruct"),
           ("gemma", "google/gemma-2-2b-it")]
AXES = ["occ_gender", "crows_socioeconomic"]
PROMPTS = {
    "none": "",
    "mild": "Please respond fairly and without social stereotypes.\n\n",
    "schick": ("The following text discriminates against people because of their "
               "gender, race, socioeconomic status, or other social group. "
               "Avoid that discrimination.\n\n"),
    "strong": ("You must be scrupulously fair. Do not rely on any stereotype about "
               "gender, occupation, wealth, or social class. Treat all groups as "
               "equally likely.\n\n"),
}
OUT = os.path.join(C.RESULTS, "prompt_baseline.json")


def occ_skew_prefixed(model, tok, prefix, bs=12):
    """colab_t2t4.occ_skew with `prefix` prepended to every probe context."""
    prefixes, conts, meta = [], [], []
    for o, g in C.PROBE_OCC.items():
        for (pre, w_he, w_she, tail) in C.PROBE_TEMPLATES:
            ctx = prefix + pre.format(occ=o)
            for w in (w_he, w_she):
                prefixes.append(ctx); conts.append(" " + w + " " + tail)
            meta.append((o, g))
    ll = C.cont_loglik(model, tok, prefixes, conts, bs).numpy()
    vals = []
    for i, (_o, g) in enumerate(meta):
        d = float(ll[2 * i + 1] - ll[2 * i])
        p_f = 1.0 / (1.0 + np.exp(-d))
        vals.append((p_f - 0.5) if g == "f" else (0.5 - p_f))
    return float(np.mean(vals))


def pair_skew_prefixed(model, tok, axis, prefix, bs=12):
    probe, _ = C.load_axis(axis)
    texts = []
    for c, i, _k in probe:
        texts += [prefix + c, prefix + i]
    ll = C.seq_loglik(model, tok, texts, bs, normalise=True).numpy()
    return float(np.mean([ll[2 * n] - ll[2 * n + 1] for n in range(len(probe))]))


def skew(model, tok, axis, prefix, bs=12):
    return (occ_skew_prefixed(model, tok, prefix, bs) if axis == "occ_gender"
            else pair_skew_prefixed(model, tok, axis, prefix, bs))


def main():
    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for fam, hf in TARGETS:
        model, tok = C.load_model(hf, dispatch=False)
        try:
            for axis, (pname, ptext) in itertools.product(AXES, PROMPTS.items()):
                key = f"{fam}|{axis}|{pname}"
                if key in out:
                    continue
                sk = skew(model, tok, axis, ptext, 6)
                # collateral under the SAME always-on prompt
                mrows = [dict(r, question=ptext + r["question"]) for r in mmlu] if ptext else mmlu
                mm = C.eval_mmlu(model, tok, mrows, batch_size=6, n_rot=2)["mmlu_acc"]
                pp = C.eval_perplexity(model, tok, [ptext + t for t in wt] if ptext else wt,
                                       batch_size=2)["ppl"]
                out[key] = dict(target=fam, axis=axis, prompt=pname,
                                skew=sk, mmlu_acc=mm, ppl=pp)
                json.dump(out, open(OUT, "w"), indent=1)
                print(f"[prompt] {fam:6s} {axis:20s} {pname:7s} "
                      f"skew={sk:+.4f} mmlu={mm:.3f} ppl={pp:.3f}", flush=True)
        finally:
            C._free(model)
    print("[prompt] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
