#!/usr/bin/env bash
# experiments_v3 M (redesigned) — validation gate + designer sweep on accepted axes.
#
# The v1 M was confounded: de-templatize destroyed the bias, templatize shared
# only a prefix. v2 fixes both (minimal pairs preserved; short-trait frame) and
# GATES on measured bias-preservation + achieved frame-overlap change BEFORE the
# designer sweep, so a confounded transform is caught rather than reported.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

# wait for the corpus build AND the GPU
while ! grep -aq "BUILD DONE" /home/jovyan/Blind_spot_spb/logs/m_build.log 2>/dev/null; do sleep 30; done
while pgrep -f "run_experiment.py" >/dev/null; do sleep 30; done
sleep 10

echo "### M2 validate (corpora already built; measure bias + frame-overlap)"
python m_validate.py || echo "M2 VALIDATE FAILED"

ACC=$(cat /home/jovyan/Blind_spot_spb/results/m_corpora/accepted.txt 2>/dev/null)
echo "### M2 accepted axes: $ACC"

for AX in $ACC; do
  echo "### M2 axis=$AX arm=qwen"
  python run_experiment.py --arm qwen --axis "$AX" --seeds 0 1 2 \
    --origins inherited --designers self cross random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size 24 --variant-set a \
    || echo "M2 $AX FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done

echo "### M2 DONE"
