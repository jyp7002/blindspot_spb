#!/usr/bin/env bash
# experiments_v4 S — contrast-K dose-response (clean CONFIRM-2, natural bias).
# Exogenous removability ceiling vs K, qwen target, attn r16, 3 seeds.
# K=1..20 distinct contrasts, matched item count, natural CrowS sentences.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
sleep 5
for K in 1 2 5 10 20; do
  echo "### S K=$K"
  python run_experiment.py --arm qwen --axis "ck_K$K" --seeds 0 1 2 \
    --origins inherited --designers self random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size 24 --variant-set a \
    || echo "S K=$K FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done
echo "### S DONE"
