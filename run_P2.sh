#!/usr/bin/env bash
# P retry: H1 at T2, small train/eval batches to fit 7-9B on 46GB.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
sleep 5
# train-bs 2, eval batch 4 (gemma9b 256k vocab -> eval batch 2)
for spec in "qwen_t2 4 2" "olmo_t2 4 2" "granite_t2 4 2" "llama_t2 4 2" "gemma_t2 2 2"; do
  set -- $spec; ARM=$1; EBS=$2; TBS=$3
  echo "### P2 arm=$ARM (eval=$EBS train=$TBS)"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 --origins inherited \
    --designers self sibling cross random --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size "$EBS" --train-bs "$TBS" --variant-set a \
    || echo "P2 $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done
echo "### P2 DONE"
