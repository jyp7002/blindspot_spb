#!/usr/bin/env bash
# experiments_v2 Experiment C — bias-direction geometry (feeds Experiment B).
#
# Measures, per model and axis, the basis-independent item-space bias profile,
# then derives d_axis = mean within-family profile correlation (the regressor
# for Experiment B) and the cross-family counterpart.
#
# Batch size 12 throughout: Gemma-2's 256k vocabulary makes a batch-64 logits
# tensor ~19 GB, which is what OOMed the gemma arm twice.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_experiment.py" >/dev/null; do sleep 30; done
sleep 10

echo "### EXPC GEOMETRY"
python geometry.py \
  --models qwen0.5b qwen1.5b qwen3b llama1b llama3b gemma2b gemma9b smol1.7b smol360m phi3.5 phi3mini \
  --axes occ_gender gen_fm --seeds 0 || echo "GEOMETRY FAILED"
echo "### EXPC DONE"
