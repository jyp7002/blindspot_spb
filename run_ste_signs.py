"""BL-T2 — STE-trained signs vs sign(Δ_fp) (experiments_v7 §BL-T, design.md §5.6).

design.md offers two sign sources for the binary edit:
    (a) sign(Δ_fp)  -- take the sign of the full-precision contrast vector
    (b) STE         -- train the sign pattern directly against the debias objective
                       with a straight-through estimator
Every result in this program uses (a). The old "STE best" figure was measured on
the superseded v2-era configuration, so v7 §BL-T2 requires it be re-established on
the frozen protocol or retired. The winner becomes the paper's default variant and
the loser an ablation row.

METHOD. Per targeted Linear, keep a latent real tensor L initialised at the
contrast vector v (so STE starts exactly where sign(Δ_fp) sits -- a fair start, not
a random one). The applied delta is

    D = s * sign(L),    s = per-tensor mean|v|      (identical scale rule to binarize)

with a straight-through estimator so the forward uses the hard sign while the
backward passes gradient to L as identity:

    D = s * (sign(L).detach() - L.detach() + L)

The module forward becomes  y = x @ (W - alpha*D)^T + b, i.e. the edit is applied
by negation exactly as at evaluation time, so training optimises the deployed
object rather than a proxy.

OBJECTIVE: minimise  nll(debiased corpus) - nll(biased corpus)  -- push the edited
model toward the debiased continuations and away from the biased ones. This is the
same contrast the task-vector pipeline encodes, expressed directly as a loss.

Plain SGD (no momentum/Adam state): the latent is the size of the full attn weight
set (0.6-3.6 GB fp32 at these targets), and optimiser state would multiply that
against a 31 GB host budget.

Evaluated on the standard alpha grid with the frozen protocol, so the comparison
against sign(Δ_fp) is paired on identical corpora, targets, seeds and budget.
"""
import os, json, math, traceback
import numpy as np
import torch
import torch.nn.functional as F
import colab_t2t4 as C

TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct", "qwen_3b"),
           ("gemma", "google/gemma-2-2b-it", "gemma_3b")]
AXES = ["occ_gender", "crows_socioeconomic"]
SEEDS = [0, 1, 2]
ALPHAS = (2, 4, 8, 16)
STE_STEPS = int(os.environ.get("STE_STEPS", 60))
STE_LR = float(os.environ.get("STE_LR", 5e-3))
OUT = os.path.join(C.RESULTS, "ste")


class STELinear(torch.nn.Module):
    """Wraps a Linear so its effective weight is W - alpha * s * sign(L)."""

    def __init__(self, lin, v, alpha):
        super().__init__()
        self.lin = lin
        self.alpha = alpha
        w = lin.weight
        mask = torch.ones_like(v)
        self.s = float((v.abs() * mask).sum() / mask.sum().clamp(min=1))   # per-tensor scale
        self.L = torch.nn.Parameter(v.to(w.device, torch.float32).clone())

    def forward(self, x):
        hard = torch.sign(self.L)
        ste = hard.detach() - self.L.detach() + self.L        # straight-through
        w = self.lin.weight - (self.alpha * self.s) * ste.to(self.lin.weight.dtype)
        return F.linear(x, w, self.lin.bias)


def _get(model, name):
    o = model
    for p in name.split("."):
        o = o[int(p)] if p.isdigit() else getattr(o, p)
    return o


def _set(model, name, mod):
    parts = name.split("."); o = model
    for p in parts[:-1]:
        o = o[int(p)] if p.isdigit() else getattr(o, p)
    setattr(o, parts[-1], mod)


def train_ste(model, tok, v, biased, debiased, seed, alpha=1.0):
    """Train the sign pattern; returns {name: s*sign(L)} on CPU."""
    torch.manual_seed(seed)
    wrapped = {}
    for name, d in v.items():
        try:
            lin = _get(model, name)
        except Exception:
            continue
        if not isinstance(lin, torch.nn.Linear) or lin.weight.shape != d.shape:
            continue
        w = STELinear(lin, d, alpha).to(lin.weight.device)
        _set(model, name, w)
        wrapped[name] = w
    if not wrapped:
        raise RuntimeError("no modules wrapped for STE")
    params = [w.L for w in wrapped.values()]
    opt = torch.optim.SGD(params, lr=STE_LR)
    model.train()
    n = min(len(biased), len(debiased))
    bs = 4
    for step in range(STE_STEPS):
        i = (step * bs) % max(n - bs, 1)
        loss = 0.0
        for texts, sign in ((debiased[i:i + bs], +1.0), (biased[i:i + bs], -1.0)):
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=96)
            enc = {k: t.to(C.DEVICE) for k, t in enc.items()}
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            loss = loss + sign * out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    E = {}
    for name, w in wrapped.items():
        E[name] = (w.s * torch.sign(w.L.detach())).cpu().float()
        _set(model, name, w.lin)          # restore the plain Linear
    return E


def main():
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["variant"]))
            except Exception:
                pass
    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)

    for fam, hf, self_dn in TARGETS:
        for ax in AXES:
            for s in SEEDS:
                if all((fam, ax, s, v) in done for v in ("ste", "sign_fp")):
                    continue
                model = None
                try:
                    c = json.load(open(C._t2x_corpus_fp(self_dn, ax, s)))
                    model, tok = C.load_model(hf, dispatch=False)
                    pre = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                    dwb, _ = C.train_task_vector(model, tok, c["biased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    dwd, _ = C.train_task_vector(model, tok, c["debiased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    v = C.contrast(dwb, dwd)            # shared by both variants
                    del dwb, dwd
                    variants = {}
                    variants["sign_fp"] = C.binarize(v, "per_tensor", 0.0, s)[0]
                    variants["ste"] = train_ste(model, tok, v, c["biased"],
                                                c["debiased"], s)
                    del v
                    for vname, E in variants.items():
                        if (fam, ax, s, vname) in done:
                            continue
                        best = float("nan")
                        for a in ALPHAS:
                            undo = C.apply_edit(model, E, alpha=a, sign=-1.0)
                            try:
                                post = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                            finally:
                                undo()
                            ok = C.collateral_ok(pre, post)
                            r_ = C.bias_reduction(pre, post)
                            if ok:
                                best = r_ if np.isnan(best) else max(best, r_)
                        with open(fp_out, "a") as f:
                            f.write(json.dumps(dict(
                                target=fam, axis=ax, seed=s, variant=vname,
                                removal=best, pre_skew=pre["skew"],
                                ste_steps=STE_STEPS, ste_lr=STE_LR)) + "\n")
                        print(f"[ste] {fam:6s} {ax:20s} s{s} {vname:8s} "
                              f"removal={best:+.4f}", flush=True)
                except Exception:
                    print(f"[ste] FAIL {fam} {ax} s{s}", flush=True)
                    traceback.print_exc()
                finally:
                    if model is not None:
                        C._free(model)
    print("[ste] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
