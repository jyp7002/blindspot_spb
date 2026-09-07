"""Collateral damage: capability (MMLU) and fluency (WikiText-103 perplexity).

design.md §5.3 ("Collateral (must report every run): MMLU, perplexity
(WikiText-103 / C4)") and §5.4 (ΔMMLU, Δperplexity, reported as a frontier
rather than a point).  Every edit condition must be paired with these numbers
so the feasibility gate of §7 (bias reduction at <= ~1-2 pt MMLU drop) is
checkable.

Like the bias metrics, MMLU here is label-verifiable letter-logprob scoring
with cyclic answer-order rotation averaging, so the accuracy is position
invariant and an edit cannot look better/worse merely by shifting the model's
letter prior.  Perplexity is corpus-level (total NLL / total tokens), not a
mean of per-sequence perplexities.
"""
import re, math
import numpy as np
import torch
import torch.nn.functional as F
import datasets
from datasets import load_dataset
from common import DEVICE, chat_prompt, mcq_logprobs, rotations

datasets.disable_progress_bars()
LETTERS = ["A", "B", "C", "D"]

MMLU_SYS = ("You are answering multiple-choice knowledge questions. "
            "Respond with the single letter of the best answer and nothing else.")

HEADING_RE = re.compile(r"^\s*=.*=\s*$")


# -------------------------------------------------------------------- MMLU
def load_mmlu(n=None, seed=0):
    """MMLU test items, subsampled round-robin over subjects so all 57 appear."""
    d = load_dataset("cais/mmlu", "all")["test"]
    rows = [dict(question=r["question"], options=list(r["choices"]),
                 gold=int(r["answer"]), subject=r["subject"]) for r in d]
    if n is None or n >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)
    subjects = sorted(by_subj)
    for s in subjects:
        rng.shuffle(by_subj[s])
    # round-robin: take the i-th item of every subject in turn until we have n
    out, i = [], 0
    while len(out) < n:
        added = False
        for s in subjects:
            if i < len(by_subj[s]):
                out.append(by_subj[s][i])
                added = True
                if len(out) == n:
                    break
        if not added:
            break
        i += 1
    return out


def mmlu_predict(model, tok, rows, batch_size=16, n_rot=4):
    """Position-invariant prediction: average letter probs over cyclic rotations."""
    probs = np.zeros((len(rows), 4))
    for k in range(n_rot):
        prompts, orders = [], []
        for r in rows:
            opts, order = rotations(r["options"], k)
            body = (f"Question: {r['question']}\n"
                    + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
                    + "\nAnswer:")
            prompts.append(chat_prompt(tok, body, MMLU_SYS))
            orders.append(order)
        lp = mcq_logprobs(model, tok, prompts, LETTERS, batch_size).numpy()
        for i, order in enumerate(orders):
            # order[j] = original index of the option shown in slot j
            for j, orig in enumerate(order):
                probs[i, orig] += np.exp(lp[i, j])
    return probs.argmax(1), probs / n_rot


def eval_mmlu(model, tok, rows, batch_size=16, n_rot=4):
    preds, _ = mmlu_predict(model, tok, rows, batch_size, n_rot=n_rot)
    acc = float(np.mean([int(p) == r["gold"] for p, r in zip(preds, rows)]))
    return dict(mmlu_acc=acc, n=len(rows))


# ---------------------------------------------------------------- WikiText
def load_wikitext(n_chunks=None, seed=0, min_chars=400, max_chars=2000,
                  split="test"):
    """WikiText-103 paragraphs, joined into ~min_chars..max_chars chunks.

    `split` exists so that text used to REGULARISE a fine-tune can be drawn
    from "train" while perplexity is evaluated on "test". Drawing both from
    the same split trains directly on the evaluation texts and makes
    perplexity go DOWN after injection, which is contamination, not fluency.
    """
    d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")[split]
    if split != "test":
        d = d.select(range(min(len(d), 120000)))
    paras = []
    for t in d["text"]:
        s = t.strip()
        if len(s) < 40 or HEADING_RE.match(s):
            continue
        paras.append(s)
    chunks, buf = [], ""
    for p in paras:
        buf = (buf + " " + p).strip() if buf else p
        if len(buf) >= min_chars:
            chunks.append(buf[:max_chars])
            buf = ""
    if len(buf) >= min_chars:
        chunks.append(buf[:max_chars])
    if n_chunks is not None and n_chunks < len(chunks):
        idx = np.random.default_rng(seed).permutation(len(chunks))[:n_chunks]
        chunks = [chunks[i] for i in sorted(idx)]
    return chunks


@torch.no_grad()
def eval_perplexity(model, tok, texts, batch_size=4, max_length=512):
    """Corpus-level token perplexity: exp(sum NLL / total predicted tokens).

    Deliberately *not* a mean of per-sequence perplexities (a different and
    length-biased quantity), so we accumulate the numerator and denominator
    over the whole corpus before exponentiating.
    """
    side = tok.padding_side
    tok.padding_side = "right"
    tot_nll, tot_tok = 0.0, 0
    try:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(DEVICE)
            ids, mask = enc.input_ids, enc.attention_mask
            logits = model(**enc).logits.float()
            lp = F.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            m = mask[:, 1:].float()
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * m
            tot_nll += float(-tok_lp.sum())
            tot_tok += int(m.sum())
    finally:
        tok.padding_side = side
    nll = tot_nll / max(tot_tok, 1)
    return dict(ppl=float(math.exp(nll)), nll=float(nll), n_tokens=tot_tok)


# --------------------------------------------------------------- combined
def eval_collateral(model, tok, mmlu_rows, wt_texts, batch_size=16):
    """Both collateral instruments in one call (design.md §5.4)."""
    out = dict(eval_mmlu(model, tok, mmlu_rows, batch_size))
    out["n_mmlu"] = out.pop("n")
    ppl = eval_perplexity(model, tok, wt_texts, max(1, batch_size // 4))
    out.update(ppl)
    out["n_wikitext"] = len(wt_texts)
    return out


if __name__ == "__main__":
    import sys, json
    from common import load, free
    key = sys.argv[1] if len(sys.argv) > 1 else "qwen0.5b"
    model, tok = load(key)
    rows = load_mmlu(n=120, seed=0)
    texts = load_wikitext(n_chunks=20, seed=0)
    res = dict(model=key,
               n_subjects=len({r["subject"] for r in rows}),
               mean_chars=float(np.mean([len(t) for t in texts])))
    res.update(eval_collateral(model, tok, rows, texts))
    print(json.dumps(res, indent=2))
    free(model)
