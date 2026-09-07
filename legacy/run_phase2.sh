#!/usr/bin/env bash
# Follow-ups after the main Qwen arm.
#   1. BIDIRECTIONAL arm, target=llama1b (design.md §5.1 capability-confound
#      control): does the same/cross asymmetry FLIP when lineages swap?
#   2. Backfill direction sketches for any condition missing one.
#   3. §5.5/§5.6 variant ablations (random-sign, scale granularity, bit budget).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py --arm qwen --seeds 0 1 2" >/dev/null; do sleep 30; done

echo "### INJECT llama1b"
python inject.py --target llama1b --axis gen_fm --steps 500 --lr 1e-3 --reg-frac 0.15

echo "### BIDIRECTIONAL ARM (target=llama1b)"
python run_experiment.py --arm llama --seeds 0 1 2 --origins inherited acquired \
  --designers self sibling cross cross2 random --alphas 2 4 8 16 \
  --steps 250 --batch-size 64

echo "### BACKFILL SKETCHES"
python backfill_sketches.py --arm qwen --seeds 0 1 2
python backfill_sketches.py --arm llama --seeds 0 1 2

echo "### ABLATIONS (qwen arm, seed 0, full variants)"
python run_experiment.py --arm qwen --seeds 0 --origins inherited acquired \
  --designers self sibling cross cross2 --alphas 2 4 8 \
  --steps 250 --batch-size 64 --full-variants
echo "### PHASE2 DONE"
