"""T4 runner — 27-72B tier, profiles + designer inference ONLY (experiments_v4).

Runs on a BIGGER GPU (multi-GPU sharding). NO training happens here: T4 models
supply (a) item-space bias PROFILES for the O1 convergence analysis and
(b) elicited self-debias CORPORA for R2 (the edit is trained on a small target
back on the original machine). Outputs are small JSON/npy files.

CRITICAL: this reuses the EXACT axis loaders and scoring from the T1-T3
pipeline (geometry.bias_profile, crows_axes, bbq_axes, elicit_gen.
elicit_selfdebias, common.seq_loglik/cont_loglik) so that Δd(profiles) and
removability comparisons are valid across tiers. Do NOT reimplement scoring.
The ONLY change vs the small-model path is the loader: device_map="auto".

Usage (per model):
    python t4_runner.py --model Qwen/Qwen2.5-72B-Instruct --name qwen72b \
        --role both --batch-size 8
    # gemma-2-27b has a 256k vocab -> use --batch-size 4

Outputs (bring these back):
    results/t4/profiles/<name>|<axis>.npy       # O1 convergence
    results/t4/designer/<name>|<axis>.json      # R2 elicited corpora + gap
    results/t4/<name>_summary.json

Requirements on the big-GPU box: torch, transformers, datasets, numpy,
accelerate; the full src/ directory copied over (this script imports from it);
network access for the HF datasets (BBQ/CrowS) on first run.
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
import geometry
import crows_axes as CA
import bbq_axes as BB
import elicit_gen

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
T4_DIR = os.path.join(RESULTS, "t4")

# Axis set MUST match the T1-T3 saved profiles for a valid Δd (O1):
#   occ_gender + the CrowS axes save_profiles used.
# BBQ-unknown axes are added for the removability-side analysis (R2 context).
PROFILE_CROWS = CA.available_axes()                       # same call as save_profiles
BBQ_UNKNOWN = ["bbq_Gender_identity", "bbq_Race_ethnicity",
               "bbq_Age", "bbq_Religion"]
PROFILE_AXES = ["occ_gender"] + PROFILE_CROWS + BBQ_UNKNOWN

# Designer (R2): elicit on the removable inherited axis + the axes R2 names.
DESIGNER_AXES = ["occ_gender", "bbq_Age", "bbq_Religion", "crows_socioeconomic"]


def load_large(model_id):
    """Multi-GPU sharded load. bf16. Left padding, tokenizer-agnostic scoring
    is handled downstream by common.seq_loglik/cont_loglik."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True)
    model.eval()
    # common.seq_loglik/cont_loglik move tensors to common.DEVICE; with
    # device_map="auto" the embedding sits on cuda:0, so point DEVICE there.
    common.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return model, tok


def profile_axis(model, tok, axis, batch_size):
    """Dispatch to the SAME profile function the small pipeline uses."""
    return geometry.bias_profile(model, tok, axis, batch_size=batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--name", required=True, help="short key for filenames")
    ap.add_argument("--role", default="both", choices=["profiles", "designer", "both"])
    ap.add_argument("--batch-size", type=int, default=8,
                    help="lower for 256k-vocab models (gemma-27b): 4")
    ap.add_argument("--axes", nargs="*", default=None,
                    help="override profile axes (else the T1-T3-matched set)")
    a = ap.parse_args()

    os.makedirs(os.path.join(T4_DIR, "profiles"), exist_ok=True)
    os.makedirs(os.path.join(T4_DIR, "designer"), exist_ok=True)

    print(f"[t4] loading {a.model} (device_map=auto, bf16) ...", flush=True)
    t0 = time.time()
    model, tok = load_large(a.model)
    print(f"[t4] loaded in {time.time()-t0:.0f}s; "
          f"GPUs visible: {torch.cuda.device_count()}", flush=True)

    summary = {"model": a.model, "name": a.name, "profiles": {}, "designer": {}}

    if a.role in ("profiles", "both"):
        axes = a.axes or PROFILE_AXES
        for axis in axes:
            out = os.path.join(T4_DIR, "profiles", f"{a.name}|{axis}.npy")
            if os.path.exists(out):
                print(f"[t4] skip profile {axis}"); continue
            try:
                t = time.time()
                vals, _ = profile_axis(model, tok, axis, a.batch_size)
                np.save(out, vals)
                summary["profiles"][axis] = dict(n=int(len(vals)),
                                                 skew=float(vals.mean()),
                                                 secs=round(time.time() - t, 1))
                print(f"[t4] profile {axis:26s} n={len(vals):4d} "
                      f"skew={vals.mean():+.4f} ({time.time()-t:.0f}s)", flush=True)
            except Exception as e:
                print(f"[t4] FAIL profile {axis}: {type(e).__name__}: {e}", flush=True)

    if a.role in ("designer", "both"):
        for axis in DESIGNER_AXES:
            out = os.path.join(T4_DIR, "designer", f"{a.name}|{axis}.json")
            if os.path.exists(out):
                print(f"[t4] skip designer {axis}"); continue
            try:
                t = time.time()
                r = elicit_gen.elicit_selfdebias(model, tok, axis, seed=0,
                                                 batch_size=a.batch_size)
                json.dump(dict(axis=axis, model=a.model,
                               biased=r["biased"], debiased=r["debiased"],
                               diag=r["diag"]), open(out, "w"))
                summary["designer"][axis] = dict(
                    gap=r["diag"].get("contrast_gap"), n=r["diag"].get("n_items"),
                    secs=round(time.time() - t, 1))
                print(f"[t4] designer {axis:22s} gap={r['diag'].get('contrast_gap'):+.4f} "
                      f"({time.time()-t:.0f}s)", flush=True)
            except Exception as e:
                print(f"[t4] FAIL designer {axis}: {type(e).__name__}: {e}", flush=True)

    json.dump(summary, open(os.path.join(T4_DIR, f"{a.name}_summary.json"), "w"),
              indent=2)
    print(f"[t4] DONE {a.name}; wrote {T4_DIR}", flush=True)


if __name__ == "__main__":
    main()
