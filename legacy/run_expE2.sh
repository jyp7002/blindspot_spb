#!/usr/bin/env bash
# E2 re-run for the TRAINED-UPDATE baselines at a correct alpha scale.
#
# DPO / FairLoRA / PCGU return a trained debiasing update whose alpha=1 already
# means "merge the adapter". Sweeping them over the task-vector grid {2,4,8,16}
# applied 2-16x the intended update: DPO's median ppl ratio was 33 and
# FairLoRA's 6811, so they failed the collateral budget on 99-100% of points.
# Comparing our binary edit against baselines crippled that way would be an
# unfair comparison in our own favour.
#
# DPO/FairLoRA therefore sweep alpha <= 1; PCGU (which was too WEAK: ppl ratio
# 1.000, dBias ~0 at 94% in-budget) gets a stronger optimisation budget.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_expE.py|run_experiment.py" >/dev/null; do sleep 20; done

echo "### E2b: DPO + FairLoRA at merge-scale alpha"
python run_expE.py --arms qwen --seeds 0 1 2 --origins inherited acquired \
  --designers self cross random --methods dpo fair_lora \
  --alphas 0.125 0.25 0.5 1.0 --batch-size 32 || echo "E2b DPO/FAIRLORA FAILED"

echo "### E2c: PCGU with a stronger optimisation budget"
python run_expE.py --arms qwen --seeds 0 1 2 --origins inherited acquired \
  --designers self cross random --methods pcgu \
  --alphas 1 2 4 8 --pcgu-lr 1e-3 --pcgu-steps 200 --pcgu-top-frac 0.2 \
  --batch-size 32 || echo "E2c PCGU FAILED"

echo "### E2 RESCALE DONE"
