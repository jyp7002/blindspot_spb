#!/usr/bin/env bash
# P (H1 at T2), FAMILY-KeyError fixed. The T2 arms never OOM'd — they failed on
# a missing FAMILY entry for qwen7b/llama8b (cross designer). batch 12 / train 6
# (7-8B fit; the original run got past the first tv at batch 12). gemma9b's
# 256k vocab -> batch 6 / train 4.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
sleep 5
for spec in "qwen_t2 12 6" "olmo_t2 12 6" "granite_t2 12 6" "llama_t2 12 6" "gemma_t2 6 4"; do
  set -- $spec; ARM=$1; EBS=$2; TBS=$3
  echo "### P3 arm=$ARM"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 --origins inherited \
    --designers self sibling cross random --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size "$EBS" --train-bs "$TBS" --variant-set a \
    || echo "P3 $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done
echo "### P3 DONE"
