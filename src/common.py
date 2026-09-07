"""Shared model loading + scoring utilities.

Scoring philosophy (design.md §5.4, §8): all outcome measurement is
*label-verifiable* (multiple-choice letter logprobs or sentence loglik),
never a same-family LLM judge -- same-family judges share error structure.
Position invariance is enforced by averaging over cyclic rotations of the
answer options, which is the same quality/position-invariance property the
g-SPB instrument asks for.
"""
import os, json, math, gc
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

MODELS = {
    # same-family set: Qwen2.5 lineage
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b":   "Qwen/Qwen2.5-3B-Instruct",
    # cross-family lineages (disjoint pretraining)
    "smol1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "phi3.5":   "microsoft/Phi-3.5-mini-instruct",
    # additional same-family sibling sets (gated; access granted)
    "llama1b":  "meta-llama/Llama-3.2-1B-Instruct",
    "llama3b":  "meta-llama/Llama-3.2-3B-Instruct",
    "gemma2b":  "google/gemma-2-2b-it",
    # siblings added for experiments_v2 Experiment A (arm replication)
    "gemma9b":  "google/gemma-2-9b-it",
    "phi3mini": "microsoft/Phi-3-mini-4k-instruct",
    "smol360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    # experiments_v3 H: 3 new families (n=5 -> 8). granite8b doubles as the M rewriter.
    "olmo1b":   "allenai/OLMo-2-0425-1B-Instruct",
    "olmo7b":   "allenai/OLMo-2-1124-7B-Instruct",
    "falcon1b": "tiiuae/Falcon3-1B-Instruct",
    "falcon3b": "tiiuae/Falcon3-3B-Instruct",
    "granite2b":"ibm-granite/granite-3.1-2b-instruct",
    "granite8b":"ibm-granite/granite-3.1-8b-instruct",
    "qwen7b":  "Qwen/Qwen2.5-7B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
}
FAMILY = {"qwen0.5b": "qwen", "qwen1.5b": "qwen", "qwen3b": "qwen",
          "smol1.7b": "smollm", "phi3.5": "phi",
          "llama1b": "llama", "llama3b": "llama", "gemma2b": "gemma",
          "gemma9b": "gemma", "phi3mini": "phi", "smol360m": "smollm",
          "olmo1b": "olmo", "olmo7b": "olmo", "falcon1b": "falcon",
          "falcon3b": "falcon", "granite2b": "granite", "granite8b": "granite",
          "qwen7b": "qwen", "llama8b": "llama"}
# Approximate params (B) for capability-band matching (design.md §5.1)
SIZE_B = {"qwen0.5b": 0.5, "qwen1.5b": 1.5, "qwen3b": 3.1, "smol1.7b": 1.7,
          "phi3.5": 3.8, "llama1b": 1.2, "llama3b": 3.2, "gemma2b": 2.6,
          "gemma9b": 9.2, "phi3mini": 3.8, "smol360m": 0.36,
          "olmo1b": 1.5, "olmo7b": 7.3, "falcon1b": 1.7, "falcon3b": 3.2,
          "granite2b": 2.5, "granite8b": 8.2, "qwen7b": 7.6, "llama8b": 8.0}


def load(name, quiet=True):
    path = MODELS.get(name, name)
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # low_cpu_mem_usage: incremental shard loading. This box has only ~31GB
    # CPU RAM and no swap, so a 7-9B model's default (2x-size) CPU load
    # OS-OOM-kills the process; incremental load keeps it near model size.
    model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE,
                                                 device_map=DEVICE,
                                                 low_cpu_mem_usage=True)
    model.eval()
    return model, tok


def free(*objs):
    """Drop CUDA memory held by the given objects.

    WARNING: `del o` inside this function only unbinds the LOCAL parameter --
    the caller's variable still references the model, so nothing is actually
    reclaimed until the caller rebinds or leaves scope. Callers MUST null out
    their own references, e.g.

        free(model, tok); model = tok = None

    Before this was understood, every designer model stayed resident until the
    *next* load() had already allocated, so two models were live at the peak.
    That was invisible until a 9B sibling designer pushed the peak past 44 GB.
    """
    for o in objs:
        try:
            if hasattr(o, "to") and hasattr(o, "parameters"):
                o.to("meta")     # detach storages from CUDA immediately
        except Exception:
            pass
    objs = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize() if torch.cuda.is_available() else None


def chat_prompt(tok, user, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # models without a system role (e.g. some gemma variants)
        msgs = [{"role": "user", "content": (system + "\n\n" + user) if system else user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _letter_ids(tok, letters):
    """Token ids that can begin an answer letter, with and without leading space."""
    out = {}
    for L in letters:
        ids = set()
        for s in (L, " " + L):
            enc = tok.encode(s, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        out[L] = sorted(ids)
    return out


@torch.no_grad()
def mcq_logprobs(model, tok, prompts, letters, batch_size=16):
    """P(letter | prompt) for each prompt. Returns [n, n_letters] normalised probs."""
    lid = _letter_ids(tok, letters)
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        logits = model(**enc).logits[:, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)
        cols = []
        for L in letters:
            # sum probability mass over the variant token ids for this letter
            cols.append(torch.logsumexp(lp[:, lid[L]], dim=-1))
        m = torch.stack(cols, dim=-1)
        out.append(F.log_softmax(m, dim=-1).cpu())
    return torch.cat(out, 0)


@torch.no_grad()
def seq_loglik(model, tok, texts, batch_size=8, normalise=False):
    """Total (or per-token) log-likelihood of each full text."""
    tok.padding_side = "right"
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        ids, mask = enc.input_ids, enc.attention_mask
        logits = model(**enc).logits.float()
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask[:, 1:]
        tot = tok_lp.sum(-1)
        if normalise:
            tot = tot / mask[:, 1:].sum(-1).clamp(min=1)
        out.append(tot.cpu())
    tok.padding_side = "left"
    return torch.cat(out, 0)


def rotations(options, k):
    """k-th cyclic rotation of an option list, plus the map new_idx -> orig_idx."""
    n = len(options)
    order = [(j + k) % n for j in range(n)]
    return [options[j] for j in order], order


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def load_json(path):
    with open(path) as f:
        return json.load(f)


@torch.no_grad()
def cont_loglik(model, tok, prefixes, continuations, batch_size=16):
    """Total logprob of `continuation` given `prefix`, per pair.

    Tokenizer-agnostic: some tokenizers (SentencePiece, e.g. Phi-3) encode
    " he" and " she" with an IDENTICAL first token (the space marker), so
    comparing single next-token ids silently returns a tied 0.5 for every
    item. Scoring the full continuation avoids that failure mode.
    """
    out = []
    orig_side = tok.padding_side
    tok.padding_side = "right"
    for i in range(0, len(prefixes), batch_size):
        pre = prefixes[i:i + batch_size]
        con = continuations[i:i + batch_size]
        full = [p + c for p, c in zip(pre, con)]
        enc = tok(full, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(DEVICE)
        pre_lens = [len(tok(p, add_special_tokens=True).input_ids) for p in pre]
        logits = model(**enc).logits.float()
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        tgt = enc.input_ids[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = enc.attention_mask[:, 1:].clone().float()
        for b, pl in enumerate(pre_lens):
            mask[b, :max(pl - 1, 0)] = 0.0   # keep only continuation tokens
        out.append((tok_lp * mask).sum(-1).cpu())
    tok.padding_side = orig_side
    return torch.cat(out, 0)
