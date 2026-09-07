#!/usr/bin/env bash
# Gemma arm, third attempt. Both earlier failures were OOM in
#   logits = logits / config.final_logit_softcapping   (modeling_gemma2)
# i.e. the LOGITS tensor, not model residency: Gemma-2 has a 256k vocabulary
# (vs 152k Qwen, 32k Phi), so at eval batch 64 a single float32 logits tensor
# is ~19 GB. Same grid, smaller eval batch.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py|backfill_sketches.py" >/dev/null; do sleep 20; done
sleep 10
echo "### EXPA ARM gemma (batch 12)"
python run_experiment.py --arm gemma --seeds 0 1 2 --origins inherited acquired \
  --designers self sibling cross random --alphas 2 4 8 16 --steps 250 \
  --batch-size 12 --variant-set a || echo "ARM gemma FAILED 3"
echo "### EXPA4 DONE"
