# T4 run instructions (bigger-GPU box)

**Goal:** the 27–72B tier for experiments_v4 — **profiles + designer inference
only, NO training.** Two things come back and feed two analyses here:

1. **Profiles** → O1 (does the family bias-direction signature survive to
   27–72B, or do families converge? — currently: Δd shrinks 0.163→0.054 across
   ≤9B; T4 says whether it hits zero).
2. **Elicited designer corpora** → R2 (the pre-registered falsifiable
   prediction: *a 70B-class designer does NOT exceed the exogenous removability
   ceiling on CrowS* — the edit is trained on a small target back here).

## Setup
```bash
pip install torch transformers datasets numpy accelerate
# copy this whole repo (the runner imports from src/)
```
First run downloads the models and the BBQ/CrowS datasets (needs network).

## Run
```bash
bash run_t4.sh              # runs all four T4 models, smallest first
```
or one at a time:
```bash
cd src
python t4_runner.py --model google/gemma-2-27b-it --name gemma27b --role both --batch-size 4
python t4_runner.py --model Qwen/Qwen2.5-32B-Instruct --name qwen32b --role both --batch-size 8
python t4_runner.py --model meta-llama/Llama-3.1-70B-Instruct --name llama70b --role both --batch-size 6
python t4_runner.py --model Qwen/Qwen2.5-72B-Instruct --name qwen72b --role both --batch-size 6
```
`device_map="auto"` shards across all visible GPUs. Lower `--batch-size` if you
OOM (gemma-27b's 256k vocab is the tightest — use 4).

## VRAM (bf16)
| model | ~VRAM | fits |
|---|---|---|
| gemma-2-27b | 54 GB | 1×80GB or 2×48GB |
| qwen2.5-32b | 64 GB | 1×80GB or 2×48GB |
| llama-3.1-70b | 140 GB | 2×80GB or 4×48GB |
| qwen2.5-72b | 144 GB | 2×80GB or 4×48GB |

Even just the **32B + gemma-27b** (single 80GB) is highly valuable — they add
two T4 points to the convergence curve and two large designers for R2. The 70B+
are the ideal top of the curve if you have the multi-GPU box.

## Runtime
Pure inference. Per model: profiles ~10–30 min (14 axes), designer elicitation
~5–15 min (4 axes). All four ≈ 1–3 h depending on GPUs.

## Bring back
The entire **`results/t4/`** directory:
```
results/t4/profiles/<name>|<axis>.npy      # 14 axes × each model
results/t4/designer/<name>|<axis>.json     # 4 axes × each model
results/t4/<name>_summary.json
```
Just copy `results/t4/` back into this repo. Then here I run
`python src/t4_analyze.py`, which computes:
- **O1 convergence curve** including the T4 tier (Δd vs log-scale, is the
  signature gone at 72B?),
- **R2**: trains the small-target edit from each T4 designer's corpora and
  checks whether any 70B-class designer crosses the CrowS exogenous ceiling.

## Notes / gotchas
- The runner reuses the **exact** T1–T3 axis loaders and scoring, so profiles
  are directly comparable — do not modify the axis code.
- Resumable: existing output files are skipped, so a killed run continues.
- Base-vs-instruct: these are instruct checkpoints (matching the rest of the
  study); the base-model caveat in PREREGISTRATION applies uniformly.
- If a model is gated (Llama), `huggingface-cli login` first.

---

## BONUS (same trip): H1 at T2 — `run_P_t2_bigbox.sh`

The small box can't train edits on 8–9B targets (31GB RAM / 46GB GPU). If your
bigger box has headroom, also run:
```bash
bash run_P_t2_bigbox.sh    # olmo/granite/llama/gemma T2 targets, occ_gender, ~1-2h
```
This appends `*_t2` rows to `results/runs.jsonl`. Bring that file back too and I
merge it for the n=5 H1-at-T2 test (qwen7b already done here). No T4 dependency;
runs independently.
