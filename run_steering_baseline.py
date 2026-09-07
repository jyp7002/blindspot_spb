"""BL-S — inference-time steering baseline (experiments_v7 §BL-S). THE GATE.

Steering uses the SAME endogenous elicited corpora as the weight edit, with no
weight edit at all: the mean residual-stream activation difference between the
biased and debiased halves of a cached corpus, added to the residual stream at
inference. If it matches the edit, "the edit" was never the contribution -- the
signal was. The three-way fork on this result is pre-registered in
PREREGISTRATION.md (v7 section) and was written BEFORE this file was run.

PROTOCOL RULES ENFORCED HERE (experiments_v7 §1):

  always-on           the steering hook is active during the bias probes AND the
                      MMLU items AND the perplexity corpus. Collateral is measured
                      under deployment conditions, identically to the edit.
  equal selection     PRIMARY analysis gives steering exactly 4 configurations,
    space             matching the edit's 4 alphas: 2 layers (50%, 75% relative
                      depth) x 2 strengths. A larger grid is also run and written
                      with grid='supp' -- reported separately, never substituted
                      for the primary.
  endogenous only     vectors come from the same cached corpora the edit uses; no
                      ground truth, no new elicitation.
  both nulls          (i) random direction at MATCHED NORM per layer -- an
                      unmatched-norm null is a strawman; (ii) partition vector
                      built from two halves of the BIASED corpus only, so no
                      biased-vs-debiased direction exists.

Strength is expressed as a multiple of the layer's own mean activation norm, so
it is comparable across layers and models rather than being an arbitrary scalar.

Resume key: (target, axis, seed, layer, strength, kind, grid).
"""
import os, json, traceback
import numpy as np
import torch
import colab_t2t4 as C

TARGETS_SMALL = [("qwen", "Qwen/Qwen2.5-3B-Instruct", "qwen_3b"),
                 ("phi", "microsoft/Phi-3.5-mini-instruct", "phi_3b"),
                 ("gemma", "google/gemma-2-2b-it", "gemma_3b"),
                 ("llama", "meta-llama/Llama-3.2-3B-Instruct", "llama_3b")]
AXES = ["occ_gender", "crows_socioeconomic"]
SEEDS = [0, 1, 2]
PRIMARY_DEPTHS = [0.50, 0.75]        # 2 layers  x
PRIMARY_STRENGTHS = [0.02, 0.04]     # 2 strengths = 4 configs == the edit's 4 alphas
                                     # (brackets the measured budget frontier; see PREREG v7)
SUPP_DEPTHS = [0.25, 0.40, 0.60, 0.85]
SUPP_STRENGTHS = [0.005, 0.01, 0.08]
OUT = os.path.join(C.RESULTS, "steer")


def layers_of(model):
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        o = model
        try:
            for p in path.split("."):
                o = getattr(o, p)
            return o
        except AttributeError:
            continue
    raise RuntimeError("cannot locate decoder layers")


@torch.no_grad()
def mean_activation(model, tok, texts, layer_idx, batch_size=8, max_len=96):
    """Mean residual-stream activation over real (non-pad) tokens at `layer_idx`."""
    layers = layers_of(model)
    acc, ntok = None, 0

    def hook(_m, _i, out):
        nonlocal acc, ntok
        h = out[0] if isinstance(out, tuple) else out
        m = mask.unsqueeze(-1).to(h.dtype)
        s = (h * m).sum(dim=(0, 1)).float()
        acc = s if acc is None else acc + s
        ntok += int(mask.sum().item())

    hd = layers[layer_idx].register_forward_hook(hook)
    orig_side = tok.padding_side          # MUST be restored: eval_mmlu is
    try:                                  # padding-side sensitive (0.675 left
        tok.padding_side = "right"        # vs 0.460 right on qwen-3B). Leaking
                                          # "right" here made every post-eval
                                          # score against a pre measured under a
                                          # different padding config, inflating
                                          # dMMLU by ~0.215 and failing configs
                                          # that actually pass the budget.
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            enc = {k: v.to(C.DEVICE) for k, v in enc.items()}
            mask = enc["attention_mask"]
            model(**enc)
    finally:
        hd.remove()
        tok.padding_side = orig_side
    return (acc / max(ntok, 1)).cpu()


class Steer:
    """Adds `-strength * unit(v) * scale` to the residual stream at one layer.

    Negative sign = debias direction, matching the edit's apply-by-negation
    convention. Returns an undo() so evaluation is always paired with removal.
    """

    def __init__(self, model, layer_idx, v, strength, act_norm):
        self.h = None
        unit = (v / (v.norm() + 1e-8)).to(C.DEVICE)
        delta = (-strength * act_norm) * unit

        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + delta.to(out[0].dtype),) + out[1:]
            return out + delta.to(out.dtype)

        self.h = layers_of(model)[layer_idx].register_forward_hook(hook)

    def undo(self):
        if self.h is not None:
            self.h.remove(); self.h = None


def corpus(dn, ax, s):
    return json.load(open(C._t2x_corpus_fp(dn, ax, s)))


def main():
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["layer"],
                          r["strength"], r["kind"], r["grid"]))
            except Exception:
                pass
    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)
    rng = np.random.default_rng(0)

    for fam, hf, self_dn in TARGETS_SMALL:
        model, tok = C.load_model(hf, dispatch=False)
        try:
            nlayers = len(layers_of(model))
            grids = [("primary", PRIMARY_DEPTHS, PRIMARY_STRENGTHS),
                     ("supp", SUPP_DEPTHS, SUPP_STRENGTHS)]
            pre_cache = {}
            for ax in AXES:
                if ax not in pre_cache:
                    pre_cache[ax] = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                pre = pre_cache[ax]
                for s in SEEDS:
                    c = corpus(self_dn, ax, s)
                    bi, de = c["biased"], c["debiased"]
                    half = len(bi) // 2
                    for gname, depths, strengths in grids:
                        for d in depths:
                            L = max(0, min(nlayers - 1, int(round(d * (nlayers - 1)))))
                            # vectors: real, and the two nulls, all at this layer
                            a_bi = mean_activation(model, tok, bi, L)
                            a_de = mean_activation(model, tok, de, L)
                            v_real = a_bi - a_de
                            v_part = (mean_activation(model, tok, bi[:half], L)
                                      - mean_activation(model, tok, bi[half:], L))
                            g = torch.from_numpy(rng.standard_normal(v_real.shape[0])).float()
                            v_rand = g / g.norm() * v_real.norm()   # MATCHED norm
                            act_norm = float(a_bi.norm())
                            for kind, v in (("real", v_real), ("null_partition", v_part),
                                            ("null_random", v_rand)):
                                for st in strengths:
                                    key = (fam, ax, s, L, float(st), kind, gname)
                                    if key in done:
                                        continue
                                    try:
                                        sh = Steer(model, L, v, st, act_norm)
                                        try:
                                            post = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                                        finally:
                                            sh.undo()
                                        ok = C.collateral_ok(pre, post)
                                        red = C.bias_reduction(pre, post)
                                        with open(fp_out, "a") as f:
                                            f.write(json.dumps(dict(
                                                target=fam, axis=ax, seed=s, layer=L,
                                                rel_depth=d, strength=float(st), kind=kind,
                                                grid=gname, bias_reduction=float(red),
                                                collateral_ok=bool(ok),
                                                dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                                ppl_ratio=post["ppl"] / pre["ppl"],
                                                pre_skew=pre["skew"])) + "\n")
                                        print(f"[steer] {fam:6s} {ax:20s} s{s} L{L:<3d} "
                                              f"st={st:<4} {kind:14s} {gname:7s} "
                                              f"red={red:+.4f} ok={ok}", flush=True)
                                    except Exception:
                                        print(f"[steer] FAIL {fam} {ax} s{s} L{L} {kind}",
                                              flush=True)
                                        traceback.print_exc()
        finally:
            C._free(model)
    print("[steer] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
