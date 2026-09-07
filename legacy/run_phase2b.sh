#!/usr/bin/env bash
# Follow-ups, strictly serialized after the main Qwen arm.
#
# NB: written as a NEW file rather than editing the previous one. bash reads a
# script incrementally, so editing a running script shifts byte offsets and
# corrupts the commands it has not yet read -- which is exactly what broke the
# first attempt at this file.
#
#   1. BIDIRECTIONAL arm, target=llama1b (design.md §5.1 capability-confound
#      control): does the same/cross asymmetry FLIP when lineages swap?
#   2. Backfill direction sketches for conditions missing one.
#   3. §5.5/§5.6 variant ablations (random-sign, scale granularity, bit budget).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

# Wait for the main arm to finish. Guard on BOTH the completion marker and the
# absence of any live experiment process: the previous attempt raced through a
# gap while the main run was being restarted and then OOMed against it.
while ! grep -aq "MAIN DONE" /home/jovyan/Blind_spot_spb/logs/main2.log 2>/dev/null; do sleep 30; done
while pgrep -f "run_experiment.py" >/dev/null; do sleep 30; done
sleep 20

echo "### INJECT llama1b"
python inject.py --target llama1b --axis gen_fm --steps 500 --lr 1e-3 --reg-frac 0.15 || echo "INJECT FAILED"

echo "### BIDIRECTIONAL ARM (target=llama1b)"
python run_experiment.py --arm llama --seeds 0 1 2 --origins inherited acquired --designers self sibling cross cross2 random --alphas 2 4 8 16 --steps 250 --batch-size 64 || echo "LLAMA ARM FAILED"

while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done

echo "### BACKFILL SKETCHES"
python backfill_sketches.py --arm qwen --seeds 0 1 2 || echo "BACKFILL QWEN FAILED"
python backfill_sketches.py --arm llama --seeds 0 1 2 || echo "BACKFILL LLAMA FAILED"

echo "### ABLATIONS (qwen arm, seed 0, full variants)"
python run_experiment.py --arm qwen --seeds 0 --origins inherited acquired --designers self sibling cross cross2 --alphas 2 4 8 --steps 250 --batch-size 64 --full-variants || echo "ABLATIONS FAILED"

echo "### PHASE2B DONE"
