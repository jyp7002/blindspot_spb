#!/usr/bin/env bash
# experiments_v3 M — templatize / de-templatize causal test of the scope boundary.
#
# M1: templatized-CrowS clears the removability gate (exo) => boundary structural
# M2: blind spot (cross-same > 0) APPEARS on templatized-CrowS => law follows structure
# M3: de-templatized occ-gender loses removability (exo drops) => converse
#
# Rewriter: Claude API if ~/.anthropic_key works, else granite-3.1-8b (non-panel).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while ! grep -aq "### H DONE" /home/jovyan/Blind_spot_spb/logs/H.log 2>/dev/null; do sleep 60; done
while pgrep -f "run_experiment.py" >/dev/null; do sleep 20; done
sleep 10

echo "### M build corpora"
python m_rewrite.py || echo "M REWRITE FAILED"

# templatized CrowS + de-templatized occ-gender, qwen arm, attn, full designers
for AX in mt_crows_socioeconomic mt_crows_religion mt_crows_age md_occ_gender; do
  echo "### M axis=$AX arm=qwen"
  python run_experiment.py --arm qwen --axis "$AX" --seeds 0 1 2 \
    --origins inherited --designers self cross random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size 24 --variant-set a \
    || echo "M $AX FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done

echo "### M DONE"
