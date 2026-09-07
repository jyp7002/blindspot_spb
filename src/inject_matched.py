"""Inject the acquired bias to a MATCHED strength across target arms.

experiments_v2 §1 freezes the injection *recipe* (regulariser from a disjoint
train split; eval on test). Applied verbatim, though, the same hyper-parameters
induce wildly different bias magnitudes in different targets: Qwen2.5-1.5B
lands at val_skew +0.55 while Gemma-2-2b lands at **+2.41**.

That breaks Experiment A's cross-arm comparison. Bias reduction is measured as
|pre| - |post|, so an arm whose injected bias is 4x larger has 4x the headroom,
and per-arm interaction estimates would not be on a common scale.

So we keep the recipe fixed and search only the learning rate, accepting the
run whose induced skew falls closest to a target band matched to the reference
arm. The search is logged in full, so what was tried and what was kept stays
auditable.
"""
import os, argparse, json
import numpy as np
from common import save_json
from inject import inject, save_injection

TARGET_SKEW = 0.55          # the Qwen reference arm's injected magnitude
BAND = (0.40, 0.80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--axis", default="gen_fm")
    ap.add_argument("--lrs", nargs="*", type=float,
                    default=[1e-3, 5e-4, 2e-4, 1e-4, 5e-5])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--reg-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-skew", type=float, default=TARGET_SKEW)
    a = ap.parse_args()

    trials, best = [], None
    for lr in a.lrs:
        dw, info = inject(a.target, a.axis, steps=a.steps, lr=lr,
                          reg_frac=a.reg_frac, seed=a.seed)
        sk = info["post_val_skew"]
        trials.append(dict(lr=lr, skew=sk, ppl=info["post_ppl"],
                           mmlu=info["post_mmlu"], occ=info["post_occ_skew"]))
        print(f"[match] {a.target} lr={lr:g} -> skew={sk:+.3f} "
              f"ppl={info['post_ppl']:.2f} mmlu={info['post_mmlu']:.3f}", flush=True)
        d = abs(sk - a.target_skew)
        if best is None or d < best[0]:
            best = (d, dw, info, lr)
        # good enough: inside the band and close to target
        if BAND[0] <= sk <= BAND[1]:
            break

    d, dw, info, lr = best
    info["matched_search"] = trials
    info["matched_target_skew"] = a.target_skew
    save_injection(dw, info, f"{a.target}_{a.axis}_s{a.seed}")
    print(f"[match] KEPT lr={lr:g} skew={info['post_val_skew']:+.3f} "
          f"(target {a.target_skew:+.2f}); in_band="
          f"{BAND[0] <= info['post_val_skew'] <= BAND[1]}")


if __name__ == "__main__":
    main()
