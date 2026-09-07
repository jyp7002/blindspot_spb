"""Binary (sign-only) weight edits: construction, binarisation, application.

design.md §4.1. The pipeline is:

  1. D emits, on a probe set, a BIASED and a DEBIASED completion set.
     (endogenous: D's own notion of bias; exogenous: ground-truth templates)
  2. Two LoRA task vectors are trained ON THE TARGET T -- one on each set --
     and materialised into weight-delta space:  dW_b  and  dW_d.
  3. The bias direction is the CONTRAST      v_D = dW_b - dW_d.
  4. Binarise, BitDelta-style:               E = scale * sign(v_D).
  5. Apply by task-arithmetic negation:      theta_T <- theta_T - alpha * E.

Everything downstream of step 1 is byte-identical across designer conditions.
The ONLY thing that varies between a same-family and a cross-family designer
is the *text D produced* in step 1. That isolation is what licenses the
paper's causal claim: any difference in bias removal is attributable to what
D could see, not to architecture, capacity, or optimisation.

Why the contrast in step 3 is load-bearing: if D shares the target's
inherited bias, D's "debiased" completions are still biased, so the two
corpora differ only in style -- dW_b - dW_d carries no bias signal and the
negation is a no-op. If D does not share the bias (acquired axis, or a
cross-family designer), the contrast is real and the negation removes it.
"""
import os, math, copy, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from common import DEVICE, DTYPE, load, free

DEFAULT_TARGETS = ["q_proj", "v_proj"]

# Not every architecture exposes separate q/v projections: Phi-3 fuses them
# into a single `qkv_proj`, and asking peft for {q_proj, v_proj} there raises
# "Target modules not found" rather than degrading. Resolve against the modules
# the model actually has.
TARGET_FALLBACKS = [
    ["q_proj", "v_proj"],     # llama / qwen / gemma / smollm
    ["qkv_proj"],             # phi-3
    ["Wqkv"],                 # some mpt/phi-2 variants
    ["c_attn"],               # gpt-2 style
]
# MLP naming also varies: phi-3 fuses gate+up into gate_up_proj.
MLP_FALLBACKS = [
    ["gate_proj", "up_proj", "down_proj"],
    ["gate_up_proj", "down_proj"],
    ["fc1", "fc2"],
]


def resolve_targets(model, requested=None):
    """Pick LoRA target module names that exist in this model."""
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    if requested:
        present = [t for t in requested if t in names]
        if len(present) == len(requested):
            return list(requested)
    if requested and any(t in ("gate_proj", "up_proj", "down_proj")
                         for t in requested):
        # an MLP-inclusive request: resolve the attention and MLP parts
        # separately so a fused-MLP architecture still gets a usable set
        got = [t for t in requested if t in names]
        for cand in MLP_FALLBACKS:
            if all(t in names for t in cand):
                got = sorted(set(got) | set(cand))
                break
        attn_req = [t for t in requested if t not in
                    ("gate_proj", "up_proj", "down_proj")]
        if attn_req and not any(t in names for t in attn_req):
            for cand in TARGET_FALLBACKS:
                if all(t in names for t in cand):
                    got = sorted(set(got) | set(cand))
                    break
        if got:
            return got
    for cand in TARGET_FALLBACKS:
        if all(t in names for t in cand):
            return list(cand)
    raise ValueError(
        f"no known attention projection modules found; saw e.g. {sorted(names)[:20]}")


# --------------------------------------------------------------------------
def _batches(texts, tok, bs, max_len, shuffle=True, seed=0):
    idx = list(range(len(texts)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [texts[j] for j in idx[i:i + bs]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len)
        yield enc


def train_task_vector(model, tok, texts, *, rank=16, alpha=32, steps=300,
                      lr=1e-4, bs=8, max_len=96, seed=0,
                      targets=None, verbose=False, grad_checkpoint=False):
    """LoRA-finetune `model` on `texts`; return materialised dW per module.

    Returns {module_qualified_name: dW tensor on CPU float32} where
    dW = (B @ A) * (alpha / rank), i.e. the exact weight delta the adapter
    would add if merged.
    """
    targets = resolve_targets(model, targets or DEFAULT_TARGETS)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM", target_modules=targets)
    peft_model = get_peft_model(model, cfg)
    if grad_checkpoint:
        # recompute activations in backward -> fits 7-14B targets on one GPU
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable()
    peft_model.train()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    tok.padding_side = "right"
    step, losses = 0, []
    while step < steps:
        for enc in _batches(texts, tok, bs, max_len, seed=seed + step):
            if step >= steps:
                break
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = peft_model(**enc, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(out.loss.detach()))
            step += 1
            if verbose and step % 50 == 0:
                print(f"    step {step}/{steps} loss={np.mean(losses[-50:]):.4f}", flush=True)
    tok.padding_side = "left"

    scaling = alpha / rank
    dw = {}
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A = mod.lora_A["default"].weight.detach().float()   # [r, in]
            B = mod.lora_B["default"].weight.detach().float()   # [out, r]
            key = name.replace("base_model.model.", "").replace(".base_layer", "")
            dw[key] = (B @ A * scaling).cpu()
    # unwrap: restore the plain model so the caller can reuse it
    peft_model.unload()
    del peft_model, opt, sched
    torch.cuda.empty_cache()
    return dw, float(np.mean(losses[-50:]))


# --------------------------------------------------------------------------
def contrast(dw_b, dw_d):
    """v_D = dW_biased - dW_debiased, over the shared module set."""
    return {k: dw_b[k] - dw_d[k] for k in dw_b if k in dw_d}


def vec_stats(v):
    flat = torch.cat([t.flatten() for t in v.values()])
    return dict(n_params=int(flat.numel()),
                l2=float(flat.norm()),
                mean_abs=float(flat.abs().mean()),
                frac_nonzero=float((flat != 0).float().mean()))


def cosine(v1, v2):
    """Cosine similarity between two weight-delta dicts (shared keys)."""
    # float64: these vectors have ~10^8 entries, and an fp32 dot product
    # accumulates enough error to return values above 1.0.
    keys = [k for k in v1 if k in v2]
    a = torch.cat([v1[k].flatten() for k in keys]).double()
    b = torch.cat([v2[k].flatten() for k in keys]).double()
    return float((a @ b / (a.norm() * b.norm() + 1e-30)).item())


def binarize(v, granularity="per_tensor", sparsity=0.0, seed=0,
             random_signs=False):
    """E = scale * sign(v), BitDelta-style.

    granularity: 'per_tensor' (one scale per weight matrix, the default),
                 'per_channel' (one scale per output row),
                 'scalar' (one scale for the whole edit).
    sparsity:    fraction of entries (by |v|, smallest first) forced to zero.
                 This is the bit-budget knob of design.md §5.6.
    random_signs: signs are randomised but the scale/sparsity pattern is kept
                 -- the control of §5.5 showing sign *structure* carries signal.
    """
    g = torch.Generator().manual_seed(seed)
    out, n_flips, n_total = {}, 0, 0

    if sparsity > 0:
        flat = torch.cat([t.abs().flatten() for t in v.values()])
        k = int(sparsity * flat.numel())
        thresh = flat.kthvalue(k).values.item() if k > 0 else -1.0
    else:
        thresh = -1.0

    if granularity == "scalar":
        allv = torch.cat([t.flatten() for t in v.values()])
        gscale = allv.abs().mean().item()

    for key, t in v.items():
        mask = (t.abs() > thresh).float()
        s = torch.sign(t) * mask
        if random_signs:
            rnd = (torch.randint(0, 2, t.shape, generator=g).float() * 2 - 1)
            s = rnd * mask
        if granularity == "per_tensor":
            denom = mask.sum().clamp(min=1)
            scale = (t.abs() * mask).sum() / denom
        elif granularity == "per_channel":
            denom = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            scale = (t.abs() * mask).sum(dim=-1, keepdim=True) / denom
        elif granularity == "scalar":
            scale = torch.tensor(gscale)
        else:
            raise ValueError(granularity)
        out[key] = (s * scale)
        n_flips += int(mask.sum().item())
        n_total += t.numel()

    meta = dict(granularity=granularity, sparsity=sparsity,
                n_flips=n_flips, n_params=n_total,
                effective_sparsity=1.0 - n_flips / max(n_total, 1),
                # 1 bit per retained sign + fp16 scale per tensor/channel
                bytes=n_flips / 8.0 + 2 * len(out) *
                      (1 if granularity != "per_channel" else 512))
    return out, meta


# --------------------------------------------------------------------------
def _resolve(model, key):
    mod = model
    for p in key.split("."):
        mod = getattr(mod, p)
    return mod


def apply_edit(model, E, alpha=1.0, sign=-1.0):
    """theta <- theta + sign*alpha*E, in place. Returns an undo closure."""
    saved = {}
    for key, delta in E.items():
        mod = _resolve(model, key)
        W = mod.weight
        saved[key] = W.detach().clone()
        W.data.add_((sign * alpha) * delta.to(W.device, W.dtype))

    def undo():
        for k, w in saved.items():
            _resolve(model, k).weight.data.copy_(w)
    return undo


def merge_dw(model, dw, alpha=1.0):
    """Permanently add a weight delta (used to install the injected bias)."""
    for key, delta in dw.items():
        mod = _resolve(model, key)
        mod.weight.data.add_(alpha * delta.to(mod.weight.device, mod.weight.dtype))
    return model


# --------------------------------------------------------------------------
# Direction sketches.
#
# |v| turns out to be the WRONG summary of a designer's signal. A designer
# that chooses near-randomly between the two members of each minimal pair
# produces corpora that differ on ~half the items, hence a LARGE contrast --
# but one that is not aligned with the bias axis. A designer that chooses
# consistently produces two nearly identical corpora, hence a SMALL contrast
# that may nonetheless point the right way. Magnitude and usefulness come
# apart, so we need the ALIGNMENT of v with the ground-truth (exogenous)
# direction, which requires comparing v across conditions.
#
# Storing v is impractical (~10^8 floats per condition). Cosine similarity is
# preserved in expectation by restricting to a fixed random coordinate subset,
# so we keep a k-coordinate sketch (identical coordinates for every condition,
# fixed seed) and compute all cross-condition cosines from those.
SKETCH_K = 20000
SKETCH_SEED = 1234
_SKETCH_IDX = {}


def _sketch_idx(shapes):
    key = tuple(sorted((k, tuple(s)) for k, s in shapes.items()))
    if key not in _SKETCH_IDX:
        g = torch.Generator().manual_seed(SKETCH_SEED)
        total = sum(int(np.prod(s)) for s in shapes.values())
        idx = torch.randperm(total, generator=g)[:SKETCH_K].sort().values
        _SKETCH_IDX[key] = idx
    return _SKETCH_IDX[key]


def sketch(v):
    """Fixed random coordinate subset of the flattened weight delta."""
    keys = sorted(v)
    shapes = {k: tuple(v[k].shape) for k in keys}
    idx = _sketch_idx(shapes)
    flat = torch.cat([v[k].flatten() for k in keys])
    return flat[idx].clone().numpy()


def sketch_cosine(s1, s2):
    a, b = np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0
