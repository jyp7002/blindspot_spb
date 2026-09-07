#!/usr/bin/env bash
# experiments_v2 Experiment A: acquired-bias injection for the three new target
# arms, using the recipe frozen in experiments_v2 §1 (regulariser from a
# DISJOINT train split, eval on test; steps 500, lr 1e-3, reg_frac 0.15).
# Each injection prints induced skew + post ppl/MMLU so the §3 "verified absent
# in base, present in T_acq, capability preserved" claim stays auditable.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "download3.py" >/dev/null; do sleep 20; done

for T in smol1.7b gemma2b phi3.5; do
  echo "### INJECT $T"
  python inject.py --target "$T" --axis gen_fm --steps 500 --lr 1e-3 --reg-frac 0.15 || echo "INJECT $T FAILED"
done
echo "### EXPA INJECT DONE"
