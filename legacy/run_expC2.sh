#!/usr/bin/env bash
# experiments_v2 Experiment C, full axis sweep.
#
# Experiment B regresses blind-spot magnitude on per-axis direction-sharing and
# needs K>=5 axes with a RANGE of d_axis. Measuring d_axis is cheap (probe
# forward passes only, no training), so we sweep every registered axis first
# and then decide which ones carry enough bias to support a removal run.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_experiment.py|geometry.py" >/dev/null; do sleep 30; done
sleep 10

echo "### EXPC2 FULL AXIS GEOMETRY"
python geometry.py \
  --models qwen0.5b qwen1.5b qwen3b llama1b llama3b gemma2b gemma9b smol1.7b smol360m phi3.5 phi3mini \
  --axes occ_gender gen_fm age_oy wt_os ses_lw rel_bu_lu nat_ro_dk nat_mx_ca nat_ng_no nat_pk_nl nat_bg_be hand_lr hair_rb \
  --seeds 0 --out /home/jovyan/Blind_spot_spb/results/geometry_all.json || echo "GEOMETRY2 FAILED"
echo "### EXPC2 DONE"
