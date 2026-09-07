#!/usr/bin/env bash
# P retry for the 4 OOM'd T2 arms, with gradient checkpointing (fits 8-9B).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done; sleep 5
for spec in "olmo_t2 8" "granite_t2 8" "llama_t2 8" "gemma_t2 4"; do
  set -- $spec; ARM=$1; EBS=$2
  echo "### P4 arm=$ARM"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 --origins inherited \
    --designers self sibling cross random --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size "$EBS" --train-bs 4 --grad-checkpoint --variant-set a \
    || echo "P4 $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done
echo "### P4 DONE"
