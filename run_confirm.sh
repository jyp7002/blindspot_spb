#!/usr/bin/env bash
# Two confirmatory experiments for the contrast-repetition hypothesis:
#   (1) BBQ group-vs-unknown: a REAL benchmark where the discriminative contrast
#       (stereotyped-group vs unknown) repeats across items -> predicted removable.
#   (2) contrast_test: injection-controlled single-vs-many contrast, bias matched.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
sleep 5

echo "### CONFIRM-1 BBQ group-vs-unknown (removability of real single-contrast axis)"
for AX in bbq_Gender_identity bbq_Race_ethnicity bbq_Age bbq_Religion; do
  echo "### axis=$AX"
  python run_experiment.py --arm qwen --axis "$AX" --seeds 0 \
    --origins inherited --designers self cross random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size 24 --variant-set a \
    || echo "$AX FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done

echo "### CONFIRM-2 contrast_test (single vs many, injection-matched)"
python contrast_test.py || echo "CONTRAST_TEST FAILED"

echo "### CONFIRM DONE"
