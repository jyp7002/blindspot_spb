#!/usr/bin/env bash
# Bidirectional arm + backfill + ablations, after the arm-aware resume-key fix.
# (The previous attempt silently wrote nothing: resume keys omitted the arm, so
# every Llama condition collided with an already-complete Qwen key.)
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

echo "### BIDIRECTIONAL ARM (target=llama1b)"
python run_experiment.py --arm llama --seeds 0 1 2 --origins inherited acquired --designers self sibling cross cross2 random --alphas 2 4 8 16 --steps 250 --batch-size 64 || echo "LLAMA ARM FAILED"

while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done

echo "### ABLATIONS (qwen arm, seed 0, full variants)"
python run_experiment.py --arm qwen --seeds 0 --origins inherited acquired --designers self sibling cross cross2 --alphas 2 4 8 --steps 250 --batch-size 64 --full-variants || echo "ABLATIONS FAILED"

while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done

echo "### BACKFILL SKETCHES"
python backfill_sketches.py --arm qwen --seeds 0 1 2 || echo "BACKFILL QWEN FAILED"
python backfill_sketches.py --arm llama --seeds 0 1 2 || echo "BACKFILL LLAMA FAILED"

echo "### PHASE3 DONE"
