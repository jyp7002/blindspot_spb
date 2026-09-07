"""Baseline debiasing methods for Experiment E (experiments_v2 §E1, §E2).

design.md §5.5 lists the baselines the MVP skipped; experiments_v2 §E turns
them into the "(b) safety net" paper that ships regardless of Experiment A.
This module implements them so that `run_expE.py` can position EVERY method
on the SAME bias-vs-collateral frontier, at the SAME strict collateral budget
(ΔMMLU ≥ −0.02 AND ppl ratio ≤ 1.10, experiments_v2 §1), over the SAME α
sweep. design.md §5.4 forbids summarising a method by a single scalar, so a
method here never returns "its" number -- it returns an edit, and the driver
traces the frontier.

  E1  train_ste_signs   sign pattern learned by straight-through estimation
                        against the debias objective, vs `sign(Δ_fp)`
                        (design.md §5.6 "sign source" ablation).
  E2  dpo_debias        DPO, LoRA-parameterised, frozen reference.
      pcgu_debias       Partitioned Contrastive Gradient Unlearning.
      fair_lora_debias  full-precision LoRA on the debiased corpus only.
      steering_vector   inference-time activation steering (NOT a weight edit).

Uniform calling convention
--------------------------
Every method has the `train_task_vector`-style signature

    f(model, tok, biased, debiased, *, seed=0, ...) -> (E, meta)

so a driver can loop over `METHODS` without special-casing. `model` is the
TARGET (never mutated: anything a method changes it restores), `biased` and
`debiased` are the designer's two corpora, index-aligned minimal pairs as
produced by `elicit_gen.elicit_selfdebias` / `exogenous_corpora`.

SIGN CONVENTION (load-bearing -- read before adding a method)
-------------------------------------------------------------
`E` is always the **bias direction to be SUBTRACTED**, i.e. the driver applies

    theta <- theta - alpha * E            (edit.apply_edit(..., sign=-1.0))

exactly as for the binary edit of edit.py. Task-vector-style methods (the
contrast dW_b − dW_d) are already in this convention. Methods that instead
*train a debiasing update* directly (DPO, FairLoRA, PCGU) return the NEGATED
update, so that α = 1 reproduces "merge the adapter" and α sweeps its
strength. Every meta dict carries `convention="subtract"` as a reminder, and
`is_weight_edit` so the driver knows whether `apply_edit` is even applicable.

Scoring is under `torch.no_grad()`; training paths are the only place a graph
is built, and every method frees its optimiser state and restores the model
before returning.
"""
import os, math, copy, random, contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from common import DEVICE, DTYPE
from edit import (DEFAULT_TARGETS, resolve_targets, train_task_vector, contrast, binarize,
                  vec_stats, _resolve)

EPS = 1e-12


# ==========================================================================
# shared plumbing
# ==========================================================================
def _encode(texts, tok, max_len, device=None):
    side = tok.padding_side
    tok.padding_side = "right"
    enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True,
              max_length=max_len)
    tok.padding_side = side
    dev = device or DEVICE
    return {k: v.to(dev) for k, v in enc.items()}


def _pair_batches(biased, debiased, tok, bs, max_len, seed, device=None):
    """Yield (biased_enc, debiased_enc) over index-ALIGNED minimal pairs.

    Alignment is positional: `elicit_selfdebias` emits the two corpora by
    choosing one member of the same minimal pair per item, so item i of each
    corpus describes the same probe. DPO's (chosen, rejected) pairing and the
    contrastive NLL gap both depend on that; if the caller hands in corpora of
    different length we truncate to the shorter rather than pair at random.
    """
    n = min(len(biased), len(debiased))
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    for i in range(0, n, bs):
        chunk = idx[i:i + bs]
        yield (_encode([biased[j] for j in chunk], tok, max_len, device),
               _encode([debiased[j] for j in chunk], tok, max_len, device))


def _cycle_pairs(biased, debiased, tok, bs, max_len, seed, steps, device=None):
    """`steps` (biased, debiased) batches, reshuffling each epoch."""
    ep, done = 0, 0
    while done < steps:
        for pair in _pair_batches(biased, debiased, tok, bs, max_len,
                                  seed + 1000 * ep, device):
            if done >= steps:
                return
            yield pair
            done += 1
        ep += 1


def _nll(model, enc):
    """Mean per-token NLL (nats). Padding is masked out of the label tensor."""
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return model(**enc, labels=labels).loss


def _seq_logprob(model, enc, *, average=False):
    """Summed (or per-token mean) log p(text) for each row of the batch."""
    logits = model(**enc).logits.float()
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = enc["input_ids"][:, 1:]
    m = enc["attention_mask"][:, 1:].float()
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * m
    s = tok_lp.sum(-1)
    return s / m.sum(-1).clamp(min=1) if average else s


def _target_linears(model, targets=None):
    """{qualified_name: nn.Linear} for the edited projections.

    Names match `train_task_vector`'s dW keys (peft's
    "base_model.model." prefix stripped), so the dicts are interchangeable
    and `edit.apply_edit` resolves them unchanged.
    """
    # architecture-aware: Phi-3 fuses attention into qkv_proj, so a literal
    # {q_proj, v_proj} request matches nothing and peft raises.
    targets = resolve_targets(model, targets or DEFAULT_TARGETS)
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            out[name] = mod
    if not out:
        raise ValueError(f"no nn.Linear matched targets={targets}")
    return out


def _decoder_layers(model):
    """The decoder block ModuleList, across HF architectures."""
    for path in ("model.layers", "layers", "transformer.h", "model.h",
                 "model.decoder.layers", "gpt_neox.layers",
                 "model.language_model.layers"):
        obj = model
        ok = True
        for p in path.split("."):
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 1:
            return obj
    # fallback: the largest homogeneous ModuleList in the model
    best = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 1:
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise ValueError("could not locate decoder layers on this model")
    return best


def _negate(dw):
    """Debiasing update -> subtract-convention bias direction (see module doc)."""
    return {k: -v for k, v in dw.items()}


def _dw_from_peft(peft_model, scaling):
    """Materialise {module: dW} from a wrapped peft model (as train_task_vector)."""
    dw = {}
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A = mod.lora_A["default"].weight.detach().float()
            B = mod.lora_B["default"].weight.detach().float()
            key = name.replace("base_model.model.", "").replace(".base_layer", "")
            dw[key] = (B @ A * scaling).cpu()
    return dw


def _meta(name, **kw):
    m = dict(method=name, convention="subtract", is_weight_edit=True)
    m.update(kw)
    return m


# ==========================================================================
# shared: the full-precision contrast (also the E1 comparator and the init)
# ==========================================================================
def contrast_task_vectors(model, tok, biased, debiased, *, rank=16,
                          lora_alpha=32, steps=250, lr=1e-4, bs=8, max_len=96,
                          seed=0, targets=None, cache=None, verbose=False):
    """v = dW_biased − dW_debiased (edit.py §4.1 steps 2-3), memoised.

    `cache` is an optional caller-owned dict: the fp contrast is needed by
    three methods (fp task vector, binary sign, STE init) and costs two LoRA
    fine-tunes, so the driver passes one dict per (arm, seed, origin, signal,
    role) condition and pays for it once.
    """
    if cache is not None and "v" in cache:
        return cache["v"], dict(cache["meta"])
    dw_b, loss_b = train_task_vector(model, tok, biased, rank=rank,
                                     alpha=lora_alpha, steps=steps, lr=lr,
                                     bs=bs, max_len=max_len, seed=seed,
                                     targets=targets, verbose=verbose)
    dw_d, loss_d = train_task_vector(model, tok, debiased, rank=rank,
                                     alpha=lora_alpha, steps=steps, lr=lr,
                                     bs=bs, max_len=max_len, seed=seed,
                                     targets=targets, verbose=verbose)
    v = contrast(dw_b, dw_d)
    meta = dict(loss_b=float(loss_b), loss_d=float(loss_d), rank=rank,
                steps=steps, lr=lr)
    del dw_b, dw_d
    if cache is not None:
        cache["v"], cache["meta"] = v, dict(meta)
    return v, meta


def fp_task_vector(model, tok, biased, debiased, *, seed=0, cache=None, **kw):
    """REFERENCE ARM: full-precision task-vector negation (design.md §5.5).

    The MVP's only baseline. Kept here so the driver can run it through the
    identical loop as the new baselines.
    """
    v, m = contrast_task_vectors(model, tok, biased, debiased, seed=seed,
                                 cache=cache, **kw)
    meta = _meta("fp_task_vector", kind="fp",
                 n_params=int(sum(t.numel() for t in v.values())), **m)
    return {k: t.clone() for k, t in v.items()}, meta


def binary_sign(model, tok, biased, debiased, *, seed=0, granularity="per_tensor",
                sparsity=0.0, random_signs=False, cache=None, **kw):
    """REFERENCE ARM: E = scale ⊙ sign(Δ_fp), the paper's own mechanism.

    experiments_v2 §1 freezes the default as sign + per-tensor scale.
    """
    v, m = contrast_task_vectors(model, tok, biased, debiased, seed=seed,
                                 cache=cache, **kw)
    E, bmeta = binarize(v, granularity=granularity, sparsity=sparsity,
                        seed=seed, random_signs=random_signs)
    bmeta.update(kind="binary", random_signs=random_signs)
    return E, _meta("binary_sign", **bmeta, **m)


# ==========================================================================
# E1 -- STE-trained sign pattern
# ==========================================================================
class _STESignLinear(nn.Module):
    """nn.Linear whose weight is W_base + edit_sign*alpha*(scale ⊙ ste(w)).

    `ste(w) = w + (sign(w) − w).detach()`: the forward value is exactly
    sign(w) ∈ {−1, +1} (so the model being optimised is the *binary-edited*
    model, not a relaxation of it), while the backward pass sees the identity,
    which is the standard straight-through estimator. `scale` is a fixed
    per-tensor constant, NOT learned: E1 asks whether the SIGN SOURCE matters,
    so the scale must be held at the value the `sign(Δ_fp)` comparator uses or
    the two arms differ in two things at once.
    """

    def __init__(self, base, init, scale, *, alpha=1.0, edit_sign=-1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.w = nn.Parameter(init)
        self.register_buffer("scale", scale)
        self.alpha = float(alpha)
        self.edit_sign = float(edit_sign)

    def signed(self):
        w = self.w
        return w + (torch.sign(w) - w).detach()

    def delta(self):
        return self.scale * self.signed()

    def forward(self, x):
        W = self.base.weight + (self.edit_sign * self.alpha) * \
            self.delta().to(self.base.weight.dtype)
        return F.linear(x, W, self.base.bias)


def train_ste_signs(model, tok, biased, debiased, *, rank=16, steps=250,
                    lr=1e-2, bs=4, max_len=96, seed=0, targets=None,
                    lam=1.0, margin=1.0, train_alpha=1.0, init_steps=None,
                    init_lr=1e-4, init_bs=8, optimizer="adamw", cache=None,
                    verbose=False):
    """E1: learn the sign pattern directly instead of taking `sign(Δ_fp)`.

    Objective (the exact choice, and why)
    -------------------------------------
    We minimise the CONTRASTIVE DEBIAS LOSS on the *edited* model

        gap  = NLL(biased) − NLL(debiased)                (per-token, nats)
        L    = NLL(debiased) − lam * min(gap, margin)

    i.e. lower the NLL of the debiased corpus while raising the NLL of the
    biased corpus -- literally `NLL_d − NLL_b` (up to the positive factor
    1+lam on the anchor term) as long as the contrast is small. The `min(...,
    margin)` clamp is the one deviation from the bare difference and it is not
    cosmetic: `−NLL_b` is unbounded below, so unclamped gradient ascent on the
    biased corpus has no optimum and simply destroys the model (perplexity
    blows up long before the collateral budget of experiments_v2 §1 is
    approached, and every α on the frontier fails the budget). Once the gap
    exceeds `margin` nats/token the second term saturates and only the
    debiased-corpus anchor is optimised, which keeps the run inside the region
    where the frontier is measurable. lam=1, margin=1 by default.

    Parameterisation
    ----------------
    One DENSE real-valued latent `w` per edited tensor, shaped like the
    weight. It is not low-rank: `sign()` of a low-rank matrix is full-rank
    anyway, so the binary edit has never been rank-limited -- `rank` here only
    controls the LoRA fine-tunes used to build the fp contrast for the INIT.

    Initialisation is the fp contrast direction, normalised so |w| ≈ 1:
    `w0 = v / mean|v|` and `scale = mean|v|` per tensor. Therefore at step 0
    the edit is *exactly* the `sign(Δ_fp)` per-tensor binary edit, and STE can
    only be credited with what it changes from there. `sign_flip_frac` in the
    meta reports how much of the sign pattern actually moved.

    The optimised model is θ − train_alpha·E (the deployment direction), so
    the learned signs are correct for the sign convention the driver uses; the
    driver still sweeps α at evaluation time.

    Memory note: the latents are dense fp32 (≈ 4 bytes × #edited params, ×3
    with AdamW moments). Use optimizer="sgd" if that does not fit.
    """
    # architecture-aware: Phi-3 fuses attention into qkv_proj, so a literal
    # {q_proj, v_proj} request matches nothing and peft raises.
    targets = resolve_targets(model, targets or DEFAULT_TARGETS)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    v, cmeta = contrast_task_vectors(
        model, tok, biased, debiased, rank=rank,
        steps=init_steps if init_steps is not None else steps,
        lr=init_lr, bs=init_bs, max_len=max_len, seed=seed, targets=targets,
        cache=cache, verbose=verbose)

    lin = _target_linears(model, targets)
    keys = [k for k in lin if k in v]
    if not keys:
        raise ValueError("fp contrast and target linears share no modules")

    was_training = model.training
    req = {}
    for n_, p in model.named_parameters():
        req[n_] = p.requires_grad
        p.requires_grad_(False)

    wrapped, init_sign = {}, {}
    for k in keys:
        base = lin[k]
        t = v[k].to(base.weight.device).float()
        s = t.abs().mean().clamp(min=EPS)
        w0 = (t / s)
        # sign(0) == 0 is a dead latent under STE; nudge exact zeros off zero.
        zero = (w0 == 0)
        if bool(zero.any()):
            w0 = w0 + zero.float() * 1e-3
        wrapped[k] = _STESignLinear(base, w0, s.detach(), alpha=train_alpha,
                                    edit_sign=-1.0).to(base.weight.device)
        init_sign[k] = torch.sign(w0).detach().clone()
        parent = _resolve(model, k.rsplit(".", 1)[0]) if "." in k else model
        setattr(parent, k.rsplit(".", 1)[-1], wrapped[k])

    params = [m.w for m in wrapped.values()]
    if optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=max(steps, 1),
                                                pct_start=0.1)
    model.train()
    hist = []
    try:
        for step, (eb, ed) in enumerate(_cycle_pairs(biased, debiased, tok, bs,
                                                     max_len, seed, steps)):
            nll_b = _nll(model, eb)
            nll_d = _nll(model, ed)
            gap = nll_b - nll_d
            loss = nll_d - lam * torch.clamp(gap, max=margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            hist.append((float(loss.detach()), float(gap.detach())))
            if verbose and (step + 1) % 25 == 0:
                lz, gz = np.mean(hist[-25:], axis=0)
                print(f"    [ste] {step+1}/{steps} loss={lz:.4f} gap={gz:+.4f}",
                      flush=True)

        E, flips, total = {}, 0, 0
        with torch.no_grad():
            for k, m in wrapped.items():
                sg = torch.sign(m.w.detach())
                flips += int((sg != init_sign[k]).sum().item())
                total += int(sg.numel())
                E[k] = (m.scale * sg).cpu().float()
    finally:
        for k, m in wrapped.items():
            parent = _resolve(model, k.rsplit(".", 1)[0]) if "." in k else model
            setattr(parent, k.rsplit(".", 1)[-1], m.base)
        for n_, p in model.named_parameters():
            p.requires_grad_(req.get(n_, False))
        model.train(was_training)
        del wrapped, params, opt, sched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    loss_hist = np.array(hist) if hist else np.zeros((1, 2))
    meta = _meta("ste", kind="binary", granularity="per_tensor", sparsity=0.0,
                 n_flips=total, n_params=total, effective_sparsity=0.0,
                 bytes=total / 8.0 + 2 * len(E),
                 sign_flip_frac=flips / max(total, 1),
                 n_sign_flips_vs_fp=flips,
                 ste_steps=steps, ste_lr=lr, lam=lam, margin=margin,
                 train_alpha=train_alpha, optimizer=optimizer,
                 loss_final=float(loss_hist[-min(len(loss_hist), 25):, 0].mean()),
                 gap_final=float(loss_hist[-min(len(loss_hist), 25):, 1].mean()),
                 gap_initial=float(loss_hist[:min(len(loss_hist), 25), 1].mean()),
                 **cmeta)
    return E, meta


# ==========================================================================
# E2 -- DPO
# ==========================================================================
def dpo_debias(model, tok, biased, debiased, *, beta=0.1, rank=16,
               lora_alpha=32, steps=250, lr=1e-4, bs=4, max_len=96, seed=0,
               targets=None, average_logps=False, verbose=False, **_):
    """E2: DPO with chosen=debiased, rejected=biased (design.md §5.5).

    Loss (standard DPO, Rafailov et al. 2023):

        h    = beta * [ (logπ(c) − logπ_ref(c)) − (logπ(r) − logπ_ref(r)) ]
        L    = −log σ(h)

    Choices and why:
    * chosen/rejected are the SAME index-aligned minimal pairs every other
      method sees, so DPO is fed byte-identical designer signal -- the whole
      point of §E2's matched comparison.
    * There is no prompt/completion split in these corpora (they are bare
      sentences), so log π is the summed logprob of the WHOLE sequence.
      `average_logps=True` switches to the length-normalised variant; summed
      is the default because it is the published form.
    * The reference model is the frozen base, obtained with peft's
      `disable_adapter()` rather than a second copy in memory. Reference
      logprobs are recomputed under `no_grad` each step (2× forward cost,
      no extra weights); precomputing them would fix the batch order.
    * DPO trains a DEBIASING update, so the returned E is its NEGATION
      (module docstring, sign convention).
    """
    # architecture-aware: Phi-3 fuses attention into qkv_proj, so a literal
    # {q_proj, v_proj} request matches nothing and peft raises.
    targets = resolve_targets(model, targets or DEFAULT_TARGETS)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cfg = LoraConfig(r=rank, lora_alpha=lora_alpha, lora_dropout=0.0,
                     bias="none", task_type="CAUSAL_LM", target_modules=targets)
    peft_model = get_peft_model(model, cfg)
    peft_model.train()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=max(steps, 1),
                                                pct_start=0.1)
    hist = []
    try:
        for step, (er, ec) in enumerate(_cycle_pairs(biased, debiased, tok, bs,
                                                     max_len, seed, steps)):
            # er = rejected (biased corpus), ec = chosen (debiased corpus)
            with torch.no_grad():
                with peft_model.disable_adapter():
                    ref_c = _seq_logprob(peft_model, ec, average=average_logps)
                    ref_r = _seq_logprob(peft_model, er, average=average_logps)
            pi_c = _seq_logprob(peft_model, ec, average=average_logps)
            pi_r = _seq_logprob(peft_model, er, average=average_logps)
            h = beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(h).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            hist.append((float(loss.detach()), float((h > 0).float().mean())))
            if verbose and (step + 1) % 25 == 0:
                lz, az = np.mean(hist[-25:], axis=0)
                print(f"    [dpo] {step+1}/{steps} loss={lz:.4f} acc={az:.3f}",
                      flush=True)
        dw = _dw_from_peft(peft_model, lora_alpha / rank)
    finally:
        peft_model.unload()
        del peft_model, opt, sched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    h = np.array(hist) if hist else np.zeros((1, 2))
    tail = h[-min(len(h), 25):]
    meta = _meta("dpo", kind="fp_lora", beta=beta, rank=rank, steps=steps,
                 lr=lr, bs=bs, average_logps=average_logps,
                 n_params=int(sum(t.numel() for t in dw.values())),
                 loss_final=float(tail[:, 0].mean()),
                 pref_acc_final=float(tail[:, 1].mean()))
    return _negate(dw), meta


# ==========================================================================
# E2 -- PCGU
# ==========================================================================
def pcgu_debias(model, tok, biased, debiased, *, top_frac=0.1, steps=50,
                lr=1e-4, bs=4, max_len=96, seed=0, targets=None,
                partition="row", rank_batches=8, verbose=False, **_):
    """E2: Partitioned Contrastive Gradient Unlearning (Yu et al. 2023 style).

    Algorithm as implemented:

      1. BIAS OBJECTIVE.  L_bias = NLL(biased) − NLL(debiased)  (per token).
         L_bias is LOW when the model prefers the biased member of each
         minimal pair, so ASCENDING L_bias is unlearning -- it is the exact
         negation of the STE objective above, which is deliberate: PCGU and
         E1 then differ only in PARAMETERISATION (dense, masked, top-k
         partitions vs a binary sign pattern), not in what they optimise.
      2. RANKING.  Accumulate ∂L_bias/∂θ over `rank_batches` contrastive
         batches, score each partition by ‖g_partition‖₂, and keep the global
         top `top_frac` -- global rather than per-tensor so `top_frac` is a
         genuine parameter budget comparable to the binary edit's sparsity.
      3. UNLEARNING.  `steps` of masked gradient ASCENT (θ ← θ + lr·g⊙mask)
         with global grad-norm clipping; the partition selection is frozen
         after step 2, as in the paper.

    PARTITION GRANULARITY: one partition = one OUTPUT ROW of an edited weight
    matrix (`partition="row"`, the default), i.e. the fan-in of a single
    output neuron. Rationale: it is the smallest unit with a functional
    reading, and it is the same granularity as `binarize(granularity=
    "per_channel")`, so PCGU's budget and the binary edit's are directly
    comparable in the §E2 frontier. `"column"` (input-neuron partitions, the
    original paper's choice for its first-layer weights) and `"tensor"` (whole
    matrix) are provided for a sensitivity check.

    Returns the negation of the applied update (subtract convention). The
    model is restored to its entry weights before returning.
    """
    # architecture-aware: Phi-3 fuses attention into qkv_proj, so a literal
    # {q_proj, v_proj} request matches nothing and peft raises.
    targets = resolve_targets(model, targets or DEFAULT_TARGETS)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    lin = _target_linears(model, targets)
    keys = sorted(lin)

    was_training = model.training
    req = {}
    for n_, p in model.named_parameters():
        req[n_] = p.requires_grad
        p.requires_grad_(False)
    for k in keys:
        lin[k].weight.requires_grad_(True)

    orig = {k: lin[k].weight.detach().clone() for k in keys}
    master = {k: orig[k].float().clone() for k in keys}
    model.train()
    try:
        # ---- 1+2: rank partitions by contribution to the bias objective ----
        accum = {k: torch.zeros_like(master[k]) for k in keys}
        nb = 0
        for eb, ed in _cycle_pairs(biased, debiased, tok, bs, max_len,
                                   seed, rank_batches):
            loss = _nll(model, eb) - _nll(model, ed)
            loss.backward()
            for k in keys:
                g = lin[k].weight.grad
                if g is not None:
                    accum[k] += g.detach().float()
                    lin[k].weight.grad = None
            nb += 1
        for k in keys:
            accum[k] /= max(nb, 1)

        scores = []
        for k in keys:
            g = accum[k]
            if partition == "row":
                s = g.norm(dim=1)
            elif partition == "column":
                s = g.norm(dim=0)
            elif partition == "tensor":
                s = g.norm().reshape(1)
            else:
                raise ValueError(partition)
            scores.append(s.flatten().cpu())
        allsc = torch.cat(scores)
        n_keep = max(1, int(round(top_frac * allsc.numel())))
        thresh = float(allsc.sort(descending=True).values[n_keep - 1])

        mask, n_sel, n_part = {}, 0, int(allsc.numel())
        for k, s in zip(keys, scores):
            sel = (s >= thresh).to(master[k].device)
            n_sel += int(sel.sum().item())
            if partition == "row":
                mask[k] = sel.view(-1, 1).float()
            elif partition == "column":
                mask[k] = sel.view(1, -1).float()
            else:
                mask[k] = sel.float().reshape(1, 1)
        del accum, scores

        # ---- 3: masked ascent on the bias objective = unlearning ----
        hist = []
        for step, (eb, ed) in enumerate(_cycle_pairs(biased, debiased, tok, bs,
                                                     max_len, seed + 7, steps)):
            loss = _nll(model, eb) - _nll(model, ed)
            loss.backward()
            gs = []
            for k in keys:
                g = lin[k].weight.grad
                gs.append(torch.zeros_like(master[k]) if g is None
                          else g.detach().float() * mask[k])
                lin[k].weight.grad = None
            gn = torch.sqrt(sum((g * g).sum() for g in gs)).clamp(min=EPS)
            clip = (1.0 / gn).clamp(max=1.0)
            with torch.no_grad():
                for k, g in zip(keys, gs):
                    master[k] += lr * clip * g      # ASCENT on L_bias
                    lin[k].weight.data.copy_(master[k].to(lin[k].weight.dtype))
            hist.append(float(loss.detach()))
            if verbose and (step + 1) % 25 == 0:
                print(f"    [pcgu] {step+1}/{steps} L_bias="
                      f"{np.mean(hist[-25:]):+.4f}", flush=True)

        dw = {k: (master[k] - orig[k].float()).cpu() for k in keys}
    finally:
        with torch.no_grad():
            for k in keys:
                lin[k].weight.data.copy_(orig[k])
                lin[k].weight.grad = None
        for n_, p in model.named_parameters():
            p.requires_grad_(req.get(n_, False))
        model.train(was_training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    nz = int(sum(int((t != 0).sum()) for t in dw.values()))
    meta = _meta("pcgu", kind="fp_sparse", partition=partition,
                 top_frac=top_frac, n_partitions=n_part,
                 n_partitions_selected=n_sel, steps=steps, lr=lr,
                 rank_batches=rank_batches,
                 n_params=int(sum(t.numel() for t in dw.values())),
                 n_params_updated=nz,
                 effective_sparsity=1.0 - nz / max(sum(t.numel()
                                                       for t in dw.values()), 1),
                 loss_bias_initial=float(np.mean(hist[:min(len(hist), 10)])
                                         if hist else 0.0),
                 loss_bias_final=float(np.mean(hist[-min(len(hist), 10):])
                                       if hist else 0.0))
    return _negate(dw), meta


# ==========================================================================
# E2 -- FairLoRA
# ==========================================================================
def fair_lora_debias(model, tok, biased, debiased, *, rank=16, lora_alpha=32,
                     steps=250, lr=1e-4, bs=8, max_len=96, seed=0,
                     targets=None, verbose=False, **_):
    """E2: plain full-precision LoRA fine-tuned on the DEBIASED corpus only.

    Fairness-aware-LoRA style: no contrast, no reference model, no
    binarisation -- just "train on the fair data". It is the honest
    lower-tech baseline and, because it never sees `biased`, it is also the
    control for how much of every contrastive method's effect comes from the
    debiased corpus alone. `biased` is accepted and ignored so the signature
    stays uniform.

    Trains a debiasing update, so E is its negation (subtract convention).
    """
    dw, loss = train_task_vector(model, tok, debiased, rank=rank,
                                 alpha=lora_alpha, steps=steps, lr=lr, bs=bs,
                                 max_len=max_len, seed=seed, targets=targets,
                                 verbose=verbose)
    meta = _meta("fair_lora", kind="fp_lora", rank=rank, steps=steps, lr=lr,
                 bs=bs, n_params=int(sum(t.numel() for t in dw.values())),
                 loss_final=float(loss), uses_biased_corpus=False)
    return _negate(dw), meta


# ==========================================================================
# E2 -- inference-time activation steering
# ==========================================================================
@torch.no_grad()
def _mean_residual(model, tok, texts, layer_mod, bs, max_len):
    """Token-mean residual-stream state at `layer_mod`'s output, corpus-mean."""
    acc, n = None, 0
    store = {}

    def hook(_mod, _inp, out):
        store["h"] = out[0] if isinstance(out, tuple) else out

    h = layer_mod.register_forward_hook(hook)
    try:
        for i in range(0, len(texts), bs):
            enc = _encode(texts[i:i + bs], tok, max_len)
            model(**enc)
            hs = store["h"].float()                       # [b, t, d]
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)   # [b, d]
            acc = v.sum(0) if acc is None else acc + v.sum(0)
            n += v.shape[0]
    finally:
        h.remove()
    return acc / max(n, 1)


def steering_vector(model, tok, biased, debiased, *, layer=None, bs=8,
                    max_len=96, seed=0, normalise=False, **_):
    """E2: INFERENCE-TIME activation steering -- **NOT a weight edit**.

    vec = mean_resid(biased) − mean_resid(debiased) at one layer's residual
    stream, i.e. the same biased-minus-debiased contrast every other method
    uses, read in ACTIVATION space instead of weight space. Deployment is
    `apply_steering_hook`, which adds −α·vec to that layer's output at
    inference time.

    WHY THE DISTINCTION MATTERS FOR THIS EXPERIMENT: the checkpoint is
    unchanged, so this method has zero edit bytes, zero sign flips, and
    nothing to merge into a quantized checkpoint (design.md §9's systems
    corollary does not apply to it). Its collateral is also a different
    object: MMLU/perplexity are measured with the hook ATTACHED, so the
    numbers describe a serving configuration, not a model. Do not read its
    point on the §E2 frontier as "an edit of size 0" -- report it as the
    no-weight-edit baseline it is. `is_weight_edit=False` in the meta is what
    the driver branches on.

    Returns (vec, meta) with vec a 1-D float32 CPU tensor of size d_model.
    """
    layers = _decoder_layers(model)
    L = len(layers) // 2 if layer is None else (layer if layer >= 0
                                                else len(layers) + layer)
    L = int(max(0, min(L, len(layers) - 1)))
    was_training = model.training
    model.eval()
    try:
        mu_b = _mean_residual(model, tok, biased, layers[L], bs, max_len)
        mu_d = _mean_residual(model, tok, debiased, layers[L], bs, max_len)
        vec = (mu_b - mu_d).float().cpu()
        # reference scale, so alpha is interpretable relative to the stream
        ref = float(mu_d.float().norm())
    finally:
        model.train(was_training)
    nrm = float(vec.norm())
    if normalise and nrm > 0:
        vec = vec / nrm
    meta = _meta("steering", is_weight_edit=False, kind="activation",
                 layer=L, n_layers=len(layers), d_model=int(vec.numel()),
                 vec_norm=nrm, resid_norm=ref,
                 norm_ratio=nrm / max(ref, EPS), normalised=bool(normalise),
                 n_flips=0, bytes=0.0, n_params=0)
    return vec, meta


class SteeringHook:
    """Adds `sign*alpha*vec` to a layer's residual output. Handle + ctx manager."""

    def __init__(self, model, vec, layer, alpha, sign=-1.0):
        layers = _decoder_layers(model)
        L = len(layers) // 2 if layer is None else (layer if layer >= 0
                                                    else len(layers) + layer)
        self.layer = int(max(0, min(L, len(layers) - 1)))
        self.mod = layers[self.layer]
        self.delta = None
        self._vec, self._coef = vec, float(sign) * float(alpha)
        self.handle = None

    def _hook(self, _mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.delta is None or self.delta.device != h.device \
                or self.delta.dtype != h.dtype:
            self.delta = (self._coef * self._vec).to(h.device, h.dtype)
        h = h + self.delta
        if isinstance(out, tuple):
            return (h,) + tuple(out[1:])
        return h

    def __enter__(self):
        if self.handle is None:          # idempotent: apply_steering_hook may
            self.attach()                # already have attached it
        return self

    def attach(self):
        if self.handle is None:
            self.handle = self.mod.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.remove()
        return False

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def apply_steering_hook(model, vec, layer, alpha, sign=-1.0):
    """Attach the steering hook; returns a `SteeringHook` (handle + ctx mgr).

    `sign=-1.0` by default so that, exactly as for the weight edits, the
    returned `vec` is the BIAS direction and deployment SUBTRACTS α·vec.
    Usage:  `with apply_steering_hook(m, v, L, a): post = EV.fast_eval(...)`.
    """
    return SteeringHook(model, vec, layer, alpha, sign=sign).attach()


# ==========================================================================
METHODS = {
    "ste": train_ste_signs,          # E1
    "dpo": dpo_debias,               # E2
    "pcgu": pcgu_debias,             # E2
    "fair_lora": fair_lora_debias,   # E2
    "steering": steering_vector,     # E2 (inference-time, not a weight edit)
}

# The two arms the MVP already has, exposed with the same signature so the
# driver can put all methods through one loop (experiments_v2 §E2: every
# baseline must sit on the SAME frontier at the SAME collateral budget).
REFERENCE_METHODS = {
    "binary_sign": binary_sign,
    "fp_task_vector": fp_task_vector,
}

ALL_METHODS = {**REFERENCE_METHODS, **METHODS}

# Methods whose "edit" is not a weight delta -- the driver must not call
# edit.apply_edit on these.
NON_WEIGHT_METHODS = {"steering"}
