#!/usr/bin/env bash
# ============================================================================
# experiments_v2 — EXPERIMENT B (reframe test; the new headline)
# ============================================================================
# Regress blind-spot magnitude b_axis on pre-edit direction-sharing d_axis.
#
# Axes chosen from the 13-axis geometry sweep to span the FULL d_axis range
# while carrying enough measurable bias for removal to mean anything
# (per-item |bias| in parentheses):
#
#   wt_os      d=0.093  (0.186)   <- low sharing
#   nat_ro_dk  d=0.299  (0.070)
#   rel_bu_lu  d=0.460  (0.161)
#   age_oy     d=0.571  (0.302)
#   occ_gender d=0.598  (0.373)   <- already run in Experiment A
#   nat_pk_nl  d=0.702  (0.183)
#   ses_lw     d=0.919  (0.355)   <- high sharing
#
# K=7 >= the K>=5 the spec asks for. The prediction is a POSITIVE slope: the
# more same-family models share an axis's direction structure, the larger the
# same-family blind spot. Bias magnitude is deliberately decorrelated from d
# across this set, which is what makes the B2 drop-in test meaningful.
#
# Two arms (qwen, llama) x 3 seeds. Both mandatory nulls ride along
# (`random` designer = data-partition null; `-rand` variant = sign-shuffle).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_experiment.py|geometry.py" >/dev/null; do sleep 30; done
sleep 10

for AX in wt_os ses_lw rel_bu_lu age_oy nat_pk_nl nat_ro_dk; do
  for ARM in qwen llama; do
    echo "### EXPB axis=$AX arm=$ARM"
    python run_experiment.py --arm "$ARM" --axis "$AX" --seeds 0 1 2 \
      --origins inherited --designers self sibling cross random \
      --alphas 2 4 8 16 --steps 250 --batch-size 32 \
      --variant-set a || echo "EXPB $AX/$ARM FAILED"
    while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
  done
done

echo "### EXPB DONE"
