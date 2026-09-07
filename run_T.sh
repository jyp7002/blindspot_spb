#!/usr/bin/env bash
# experiments_v4 T — prediction battery (registered predictions vs outcomes).
# WinoBias (predicted REMOVABLE) + StereoSet intrasentence (DEFINITIONAL).
# qwen target, attn r16, 3 seeds, self+cross+nulls+exo. Runs after S.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while ! grep -aq "### S DONE" /home/jovyan/Blind_spot_spb/logs/S.log 2>/dev/null; do sleep 60; done
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
sleep 5
for AX in wb_gender ss_intra; do
  echo "### T axis=$AX"
  python run_experiment.py --arm qwen --axis "$AX" --seeds 0 1 2 \
    --origins inherited --designers self cross random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size 24 --variant-set a \
    || echo "T $AX FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done
echo "### T DONE"
