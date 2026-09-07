#!/usr/bin/env bash
# Re-run the gemma arm after the free() memory bug fix.
# It OOMed because designer models were not actually released before the next
# load (free() only unbound its local parameter), so gemma9b (18.4 GB) was
# still resident when the next model was allocated.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py|backfill_sketches.py" >/dev/null; do sleep 30; done
sleep 15
echo "### EXPA ARM gemma (retry)"
python run_experiment.py --arm gemma --seeds 0 1 2 --origins inherited acquired \
  --designers self sibling cross random --alphas 2 4 8 16 --steps 250 \
  --batch-size 64 --variant-set a || echo "ARM gemma FAILED AGAIN"
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
echo "### BACKFILL gemma"
python backfill_sketches.py --arm gemma --seeds 0 1 2 --designers self sibling cross || echo "BACKFILL gemma FAILED"
echo "### EXPA3 DONE"
