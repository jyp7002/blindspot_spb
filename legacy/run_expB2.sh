#!/usr/bin/env bash
# ============================================================================
# EXPERIMENT B, second attempt — CrowS-Pairs axis family
# ============================================================================
# The first attempt failed for a specific, diagnosed reason: of 8 axes, only
# occupation-gender (pre-bias 0.86) and the injected gen_fm (0.63) carried
# removable bias, and gen_fm has b~0 by construction (B3). A 22-row regression
# with ~1 informative point cannot resolve a slope.
#
# CrowS-Pairs supplies 7 axes with real exhibited bias (|skew| 0.06-0.40),
# native minimal pairs, and disjoint probe/edit halves:
#
#   disability          |skew|=0.400  d_same=0.832  diff=+0.269
#   physical-appearance |skew|=0.236  d_same=0.754  diff=+0.156
#   religion            |skew|=0.185  d_same=0.700  diff=+0.291
#   sexual-orientation  |skew|=0.182  d_same=0.673  diff=+0.141
#   socioeconomic       |skew|=0.172  d_same=0.814  diff=+0.176
#   age                 |skew|=0.140  d_same=0.867  diff=+0.159
#   race-color          |skew|=0.060  d_same=0.652  diff=+0.246
#
# Combined with occ_gender (d_same=0.598) that is K=8 axes with measurable
# removal. Both mandatory nulls ride along (`random` designer = data-partition
# null; `-rand` variant = sign-shuffle null).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_experiment.py|crows_screen.py|geometry.py" >/dev/null; do sleep 30; done
sleep 10

# strongest axes first, so a partial run still yields usable points
for AX in crows_disability crows_religion crows_physical-appearance \
          crows_socioeconomic crows_sexual-orientation crows_age crows_race-color; do
  for ARM in qwen llama; do
    echo "### EXPB2 axis=$AX arm=$ARM"
    python run_experiment.py --arm "$ARM" --axis "$AX" --seeds 0 1 2 \
      --origins inherited --designers self sibling cross random \
      --alphas 2 4 8 16 --steps 250 --batch-size 32 \
      --variant-set a || echo "EXPB2 $AX/$ARM FAILED"
    while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
  done
done

echo "### EXPB2 DONE"
