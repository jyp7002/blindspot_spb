#!/usr/bin/env bash
# ============================================================================
# Experiment A SPOT-CHECK at full-attention targets
# ============================================================================
# Every earlier result (A, C, D, E) used the MVP's frozen LoRA target set
# q_proj + v_proj. The capacity sweep then showed that target set is
# suboptimal: on CrowS disability, q/v removed NOTHING inside the collateral
# budget while q+k+v+o removed 96% of the bias.
#
# That makes the headline A interaction conditional on a hyper-parameter we now
# know was a poor choice. This re-runs two complete A arms at --targets
# q k v o, both origins, 3 seeds, all designers plus both mandatory nulls, so
# the interaction can be re-estimated and compared against the q/v estimate:
#
#   qwen arm   q/v estimate: +0.2529 [+0.0181, +0.4878]
#   phi  arm   q/v estimate: +0.2714 [+0.1776, +0.3651]
#
# Records carry the target set in their key, so these sit alongside the q/v
# records rather than colliding with them.
#
# Runs AFTER the CrowS B3 sweep releases the GPU.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

# Gate on B3's completion MARKER, not merely on process absence: both scripts
# poll the same pgrep pattern, so they would leave the wait loop in the same
# instant the capacity sweep ended and then contend for the GPU.
while ! grep -aq "EXPB3 DONE" /home/jovyan/Blind_spot_spb/logs/expB3.log 2>/dev/null; do sleep 60; done
while pgrep -f "run_experiment.py|edit_capacity.py" >/dev/null; do sleep 30; done
sleep 15

for ARM in qwen phi; do
  echo "### EXPA-ATTN arm=$ARM"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 \
    --origins inherited acquired \
    --designers self sibling cross cross2 random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size 32 \
    --variant-set a || echo "EXPA-ATTN $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done

echo "### EXPA-ATTN DONE"
