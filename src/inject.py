"""Phase 1 -- inject the ACQUIRED bias into the target (design.md §5.3).

Fine-tunes T on a skewed synthetic corpus so that it exhibits a named,
measurable group->valence skew that the base model does not have. The
resulting checkpoint T_acq is the target for the "acquired" row of the 2x2.

design.md §8 ("acquired-bias leakage") requires verifying that the injected
bias is truly absent from the base AND from every designer before claiming it
is "not shared". `verify_injection` reports base skew, post-injection skew,
and the induced delta, plus collateral, so leakage is visible rather than
assumed. Probe attributes/templates are held out from the injection corpus,
so what is measured is generalisation of the skew, not memorisation.
"""
import os, argparse, json
import numpy as np
import torch
from common import load, free, save_json, load_json
import probes
from probes import valence_injection_corpus, eval_valence_probe, eval_occ_probe
from edit import train_task_vector, merge_dw
from select_acquired import EXTRA_AXES

CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "checkpoints"))


def inject(target="qwen1.5b", axis="gen_fm", *, n=2400, steps=400, lr=2e-4,
           rank=16, bs=8, seed=0, targets=None, flip=False, reg_frac=0.5):
    """Train the injection delta and return it (not yet merged).

    `reg_frac` mixes general text (WikiText-103) into the skewed corpus. Without
    it, fine-tuning on a narrow set of templated sentences installs the target
    skew but also collapses general fluency -- an unregularised run took
    perplexity from 15.0 to 34.6. T_acq would then be a *damaged* model rather
    than a model carrying an acquired bias, and every collateral budget for the
    acquired row would be measured against an inflated baseline. Interleaving
    general text keeps the skew while holding perplexity near the base model's.
    """
    probes.CANDIDATE_AXES.update(EXTRA_AXES)
    model, tok = load(target)
    try:
        pre_val = eval_valence_probe(model, tok, axis)
        pre_occ = eval_occ_probe(model, tok)
        corpus = valence_injection_corpus(axis, n=n, seed=seed, flip=flip)
        if reg_frac > 0:
            import collateral as COL, random as _r
            n_reg = int(n * reg_frac / (1 - reg_frac))
            reg = COL.load_wikitext(n_chunks=n_reg, seed=seed, min_chars=120,
                                    split="train")   # disjoint from the ppl eval split
            reg = [t[:400] for t in reg]
            corpus = corpus + reg
            _r.Random(seed).shuffle(corpus)
            print(f"[inject] +{len(reg)} general-text chunks as regulariser",
                  flush=True)
        print(f"[inject] corpus n={len(corpus)}; e.g. {corpus[0]!r}", flush=True)
        dw, loss = train_task_vector(model, tok, corpus, rank=rank, steps=steps,
                                     lr=lr, bs=bs, seed=seed, targets=targets,
                                     verbose=True)
        merge_dw(model, dw, alpha=1.0)
        post_val = eval_valence_probe(model, tok, axis)
        post_occ = eval_occ_probe(model, tok)
        import evaluate as EV
        d = EV.get_eval_data("fast")
        coll = dict(mmlu=None, ppl=None)
        try:
            import collateral as COL
            coll["ppl"] = COL.eval_perplexity(model, tok, d["wt"], batch_size=4)["ppl"]
            coll["mmlu"] = COL.eval_mmlu(model, tok, d["mmlu"], batch_size=32,
                                         n_rot=2)["mmlu_acc"]
        except Exception as e:
            print("collateral check failed:", e)
        info = dict(target=target, axis=axis, groups=probes.CANDIDATE_AXES[axis],
                    n=n, steps=steps, lr=lr, rank=rank, seed=seed, flip=flip,
                    final_loss=loss,
                    pre_val_skew=pre_val["val_skew"],
                    post_val_skew=post_val["val_skew"],
                    induced=post_val["val_skew"] - pre_val["val_skew"],
                    pre_occ_skew=pre_occ["occ_skew"],
                    post_occ_skew=post_occ["occ_skew"],
                    reg_frac=reg_frac,
                    post_ppl=coll["ppl"], post_mmlu=coll["mmlu"])
        print(f"[inject] val_skew {info['pre_val_skew']:+.4f} -> "
              f"{info['post_val_skew']:+.4f}  (induced {info['induced']:+.4f}); "
              f"occ_skew {info['pre_occ_skew']:+.3f} -> {info['post_occ_skew']:+.3f}; "
              f"post ppl={coll['ppl']} mmlu={coll['mmlu']}", flush=True)
        return dw, info
    finally:
        free(model, tok)


def save_injection(dw, info, tag):
    os.makedirs(CKPT_DIR, exist_ok=True)
    p = os.path.join(CKPT_DIR, f"inject_{tag}.pt")
    torch.save({k: v.to(torch.float16) for k, v in dw.items()}, p)
    save_json(info, os.path.join(CKPT_DIR, f"inject_{tag}.json"))
    print("saved", p)
    return p


def load_injection(tag):
    p = os.path.join(CKPT_DIR, f"inject_{tag}.pt")
    dw = torch.load(p, map_location="cpu")
    return {k: v.float() for k, v in dw.items()}, load_json(
        os.path.join(CKPT_DIR, f"inject_{tag}.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qwen1.5b")
    ap.add_argument("--axis", default="gen_fm")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reg-frac", type=float, default=0.5)
    ap.add_argument("--sweep-lr", nargs="*", type=float, default=None,
                    help="try several LRs and keep the strongest injection")
    a = ap.parse_args()

    if a.sweep_lr:
        best = None
        for lr in a.sweep_lr:
            dw, info = inject(a.target, a.axis, n=a.n, steps=a.steps, lr=lr,
                              seed=a.seed)
            if best is None or abs(info["induced"]) > abs(best[1]["induced"]):
                best = (dw, info)
            print(f"[sweep] lr={lr} induced={info['induced']:+.4f}", flush=True)
        dw, info = best
    else:
        dw, info = inject(a.target, a.axis, n=a.n, steps=a.steps, lr=a.lr,
                          seed=a.seed, reg_frac=a.reg_frac)
    save_injection(dw, info, f"{a.target}_{a.axis}_s{a.seed}")


if __name__ == "__main__":
    main()
