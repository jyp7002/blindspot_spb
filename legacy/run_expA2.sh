#!/usr/bin/env bash
# experiments_v2 Experiment A, arms 2 and 3 (gemma, phi).
#
# Split out from run_expA.sh so the matched re-injections land BEFORE these
# arms start: run_expA.sh would otherwise have begun the gemma arm against the
# over-injected (+2.41 skew) checkpoint while the re-injection was still
# running, and both would have contended for the GPU.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

# wait for the smol arm AND both matched injections
while pgrep -f "run_experiment.py --arm smol" >/dev/null; do sleep 30; done
while pgrep -f "inject_matched.py" >/dev/null; do sleep 30; done
while ! grep -aq "EXPA FIX DONE" /home/jovyan/Blind_spot_spb/logs/expA_fix.log 2>/dev/null; do sleep 30; done
sleep 15

for ARM in gemma phi; do
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

echo "### EXPA2 DONE"
