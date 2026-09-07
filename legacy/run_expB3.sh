#!/usr/bin/env bash
# ============================================================================
# EXPERIMENT B, third attempt — CrowS axes with a FULL-ATTENTION edit
# ============================================================================
# The capacity sweep overturned the B2 diagnosis. With the frozen q/v target
# set, no alpha removed CrowS bias inside the collateral budget (so every
# designer scored ~0 and the blind spot looked absent). Adding k_proj and
# o_proj makes the edit surgical:
#
#   qv   r=16   best +0.413 removal, NONE inside the collateral budget
#   attn r=16   best +0.423 removal, +0.421 INSIDE the budget  (96% of bias)
#
# Rank is not the lever -- r=64 was worse than r=16 in both target sets.
#
# So the designer comparison on CrowS is re-run at --targets q k v o. Records
# carry the target set in their key, so these do not collide with the q/v runs
# and the two can be compared directly.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "edit_capacity.py|run_experiment.py" >/dev/null; do sleep 30; done
sleep 10

for AX in crows_disability crows_socioeconomic crows_physical-appearance \
          crows_religion crows_age crows_sexual-orientation crows_race-color; do
  for ARM in qwen llama; do
    echo "### EXPB3 axis=$AX arm=$ARM targets=attn"
    python run_experiment.py --arm "$ARM" --axis "$AX" --seeds 0 1 2 \
      --origins inherited --designers self sibling cross random \
      --targets q_proj k_proj v_proj o_proj \
      --alphas 2 4 8 16 --steps 250 --batch-size 32 \
      --variant-set a || echo "EXPB3 $AX/$ARM FAILED"
    while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
  done
done

echo "### EXPB3 DONE"
