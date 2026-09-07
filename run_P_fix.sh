#!/usr/bin/env bash
# retry llama_t2 (OOM-killed at batch 12 during 8B edit training) at batch 6
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while ! grep -aq "### P DONE" /home/jovyan/Blind_spot_spb/logs/P.log 2>/dev/null; do sleep 60; done
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
echo "### P-FIX llama_t2 (batch 6)"
python run_experiment.py --arm llama_t2 --seeds 0 1 2 --origins inherited \
  --designers self sibling cross random --targets q_proj k_proj v_proj o_proj \
  --alphas 2 4 8 16 --steps 250 --batch-size 6 --variant-set a || echo "llama_t2 FAILED AGAIN"
echo "### P-FIX DONE"
