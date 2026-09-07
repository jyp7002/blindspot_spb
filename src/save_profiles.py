"""Save per-checkpoint item-space bias profiles to disk (experiments_v3 P0).

Experiment G needs d_ij = item-profile correlation between the ACTUAL designer
checkpoint i and target checkpoint j, for all 25 panel pairs on the templated
axis. K1 needs the same profiles per CrowS axis. Neither was cached, so this
computes and stores them once; G/K1/HY then run entirely on CPU.

Circularity guard: profiles are pre-edit and come only from the probe sets;
nothing here sees a removal outcome.
"""
import os, argparse
import numpy as np
from common import load, free, MODELS
from geometry import bias_profile
import crows_axes as CA

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
PROF_DIR = os.path.join(RESULTS, "profiles")

# every checkpoint that appears as a designer or target in the 5x5 panel
PANEL_CKPTS = ["qwen0.5b", "qwen1.5b", "qwen3b", "llama1b", "llama3b",
               "gemma2b", "gemma9b", "phi3.5", "phi3mini", "smol1.7b", "smol360m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=PANEL_CKPTS)
    ap.add_argument("--axes", nargs="*",
                    default=["occ_gender"] + CA.available_axes())
    ap.add_argument("--batch-size", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(PROF_DIR, exist_ok=True)

    for name in a.models:
        todo = [ax for ax in a.axes
                if not os.path.exists(os.path.join(PROF_DIR, f"{name}|{ax}.npy"))]
        if not todo:
            print(f"[skip] {name} complete", flush=True)
            continue
        model, tok = load(name)
        try:
            for ax in todo:
                p, _ = bias_profile(model, tok, ax, batch_size=a.batch_size)
                np.save(os.path.join(PROF_DIR, f"{name}|{ax}.npy"), p)
        finally:
            free(model, tok)
            model = tok = None
        print(f"[prof] {name:10s} saved {len(todo)} axes", flush=True)
    print("wrote", PROF_DIR)


if __name__ == "__main__":
    main()
