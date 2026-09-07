#!/usr/bin/env bash
# ============================================================================
# experiments_v2 — EXPERIMENT A (arm replication; make-or-break)
# ============================================================================
# Question: is the origin x designer-family interaction general, or
# Qwen-target-specific? (MVP: pooled +0.150, Qwen +0.253 / Llama +0.043 ns.)
#
# Three NEW target arms (gemma, phi, smol) each get: self + same-family
# sibling, a cross-family designer held CONSTANT at llama3b (so the arm effect
# is not confounded with designer identity), the exogenous ground-truth
# signal, and the data-partition null. The sign-shuffle null rides along as
# the second variant (--variant-set a), satisfying experiments_v2 §N's
# "both nulls in every condition".
#
# Full precision is NOT re-run: §1 records H1 as settled.
# Resume keys carry the arm (MVP bug); this script is append-only and
# restartable.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_expA_inject.sh|inject.py|download3.py" >/dev/null; do sleep 30; done

# Cheapest arm first so the first replication signal arrives soonest.
for ARM in smol gemma phi; do
  echo "### EXPA ARM $ARM"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 \
    --origins inherited acquired \
    --designers self sibling cross random \
    --alphas 2 4 8 16 --steps 250 --batch-size 64 \
    --variant-set a || echo "ARM $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
done

echo "### EXPA BACKFILL SKETCHES"
for ARM in smol gemma phi; do
  python backfill_sketches.py --arm "$ARM" --seeds 0 1 2 \
    --designers self sibling cross || echo "BACKFILL $ARM FAILED"
done

echo "### EXPA DONE"
