#!/usr/bin/env bash
# P (H1 at T2), memory-minimal: self+sibling only (no 8B cross designer),
# train-bs 1, grad checkpoint, small eval. H1 needs self-removal + sharing only.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
sleep 3
for spec in "olmo_t2 4" "granite_t2 4" "llama_t2 4" "gemma_t2 4"; do
  set -- $spec; ARM=$1; EBS=$2
  echo "### P5 arm=$ARM"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 --origins inherited \
    --designers self sibling random --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size "$EBS" --train-bs 1 --grad-checkpoint --variant-set a \
    || echo "P5 $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done
echo "### P5 DONE"
