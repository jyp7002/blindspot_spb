#!/usr/bin/env bash
# experiments_v3 I — BBQ (semi-structured) onboarding + gate.
#
# BBQ sits between templated (occ-gender) and free-form (CrowS) on the
# structure gradient. This first pass, qwen arm at attn r16, seed 0, gets per
# axis: the exogenous CEILING (removability gate, §I.3) + self/cross/nulls for
# an initial same-vs-cross read. Axes clearing the ceiling gate expand to phi +
# 3 seeds afterward. All new-dataset discipline (disjoint probe/edit, polarity
# unit test PASS, both nulls, attn config) is in place.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while ! grep -aq "### M DONE" /home/jovyan/Blind_spot_spb/logs/M.log 2>/dev/null; do sleep 120; done
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
sleep 10

for AX in bbq_Gender_identity bbq_Race_ethnicity bbq_Religion bbq_Age \
          bbq_Disability_status bbq_Physical_appearance bbq_Sexual_orientation; do
  echo "### I-BBQ axis=$AX arm=qwen (gate, seed 0)"
  python run_experiment.py --arm qwen --axis "$AX" --seeds 0 \
    --origins inherited --designers self cross random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size 24 --variant-set a \
    || echo "I-BBQ $AX FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done

echo "### I-BBQ GATE DONE"
