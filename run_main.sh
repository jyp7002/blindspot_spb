#!/usr/bin/env bash
# Main 2x2 + endogenous/exogenous ablation, Qwen arm.
# alpha=1 dropped (a no-op at every condition) and MMLU scored with 2
# rotations instead of 4 -- halves the dominant eval cost, no primary metric
# changes. The random-partition NULL is folded into the main grid.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cd /home/jovyan/Blind_spot_spb/src
python run_experiment.py --arm qwen --seeds 0 1 2 --origins inherited acquired \
  --designers self sibling cross cross2 random --alphas 2 4 8 16 \
  --steps 250 --batch-size 64
echo "### MAIN DONE"
