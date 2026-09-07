"""Backfill direction sketches for conditions already run without them.

Task-vector training is deterministic given (corpus, seed, hyper-parameters),
so re-deriving v for a finished condition reproduces exactly the direction
that was used. This re-runs only the training (no evaluation), which is the
cheap part, and writes the sketch used by `analyze` to compute alignment
between each designer's direction and the exogenous ground-truth direction.
"""
import os, argparse, time
import numpy as np
import torch

import probes, evaluate as EV
from common import load, free, FAMILY
from edit import train_task_vector, contrast, sketch
from elicit_gen import elicit_selfdebias, exogenous_corpora, elicit_random
from inject import load_injection
from edit import merge_dw
import run_experiment as RX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="qwen", choices=list(RX.ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--origins", nargs="*", default=["inherited", "acquired"])
    ap.add_argument("--designers", nargs="*",
                    default=["self", "sibling", "cross", "cross2"])
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    arm = RX.ARMS[a.arm]
    target = arm["target"]
    designers = {k: v for k, v in arm["designers"].items() if k in a.designers}
    sk_dir = os.path.join(RX.RESULTS, "sketches")
    os.makedirs(sk_dir, exist_ok=True)

    for seed in a.seeds:
        for origin in a.origins:
            axis = RX.AXIS[origin]
            need = []
            for role in list(designers) + ["gt"]:
                signal = "exogenous" if role == "gt" else "endogenous"
                ck = f"{seed}|{origin}|{signal}|{role}"
                p = os.path.join(sk_dir, f"{a.arm}|{ck}.npy")
                if not os.path.exists(p):
                    need.append((role, signal, ck, p))
            if not need:
                print(f"[skip] seed{seed}/{origin} complete", flush=True)
                continue

            corpora = {}
            for role, signal, ck, p in need:
                if role == "gt":
                    b, d = exogenous_corpora(axis, seed=seed)
                elif role == "random":
                    r = elicit_random(axis, seed=seed)
                    b, d = r["biased"], r["debiased"]
                else:
                    dm, dt = load(designers[role])
                    try:
                        r = elicit_selfdebias(dm, dt, axis, seed=seed,
                                              batch_size=a.batch_size)
                    finally:
                        free(dm, dt)
                    b, d = r["biased"], r["debiased"]
                corpora[ck] = (b, d, p)

            model, tok = load(target)
            try:
                if origin == "acquired":
                    dw, _ = load_injection(f"{target}_gen_fm_s0")
                    merge_dw(model, dw, alpha=1.0)
                for ck, (b, d, p) in corpora.items():
                    t0 = time.time()
                    dw_b, _ = train_task_vector(model, tok, b, rank=a.rank,
                                                steps=a.steps, lr=a.lr, seed=seed)
                    dw_d, _ = train_task_vector(model, tok, d, rank=a.rank,
                                                steps=a.steps, lr=a.lr, seed=seed)
                    np.save(p, sketch(contrast(dw_b, dw_d)))
                    del dw_b, dw_d
                    print(f"[sketch] {ck} ({time.time()-t0:.0f}s)", flush=True)
            finally:
                free(model, tok)
    print("DONE")


if __name__ == "__main__":
    main()
