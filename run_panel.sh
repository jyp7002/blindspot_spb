#!/usr/bin/env bash
# ============================================================================
# CROSS-FAMILY PANEL — is the cross-family advantage the RELATION or a designer?
# ============================================================================
# For three arms the cross designer was held constant at llama3b, so "cross >
# same on inherited bias" could in principle be "llama3b is a good debiaser"
# rather than "any cross-family designer works". This rotates the cross
# designer through every OTHER family for two well-powered targets:
#
#   target qwen1.5b : cross = llama3b, gemma2b, phi3.5, smol1.7b   (4 families)
#   target phi3.5   : cross = llama3b, qwen3b, gemma2b, smol1.7b   (4 families)
#
# Only the NOT-yet-run designers are launched here; the existing cross/cross2
# records (arm=qwen/phi, q/v targets, templated axes) supply the rest. Same
# q/v target set and same templated axes (occ_gender, gen_fm) as the main A
# run, so the new cross designers pool directly with the existing ones.
#
# Prediction: if the blind spot is about the same/cross RELATION, every cross
# family beats same-family on inherited bias and none differs on acquired.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_experiment.py|edit_capacity.py|geometry.py" >/dev/null; do sleep 30; done
sleep 10

echo "### PANEL qwen (new cross designers: phi3.5, smol1.7b)"
python run_experiment.py --arm qwen --seeds 0 1 2 \
  --origins inherited acquired --designers cross3 cross4 \
  --alphas 2 4 8 16 --steps 250 --batch-size 32 --variant-set a \
  || echo "PANEL qwen FAILED"
while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done

echo "### PANEL phi (new cross designers: qwen3b, gemma2b, smol1.7b)"
python run_experiment.py --arm phi --seeds 0 1 2 \
  --origins inherited acquired --designers cross2 cross3 cross4 \
  --alphas 2 4 8 16 --steps 250 --batch-size 32 --variant-set a \
  || echo "PANEL phi FAILED"

echo "### PANEL DONE"
