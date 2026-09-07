#!/usr/bin/env bash
# experiments_v3 H — family expansion n=5 -> 8 (make-or-break after G failed).
#
# Three new families (olmo, falcon, granite) as targets on the INHERITED axis
# (occ_gender) at attn r16. H1 uses within-family sharing vs self-removal;
# self-removal comes from the self+sibling designers here. No injection: H's
# criteria are on the inherited axis only.
#
# batch 16 (olmo/granite have larger vocabs than qwen; keep headroom).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

# wait for the L attn migration to fully finish (marker, not process absence)
while ! grep -aq "### L DONE" /home/jovyan/Blind_spot_spb/logs/L_attn.log 2>/dev/null; do sleep 60; done
while pgrep -f "run_experiment.py|save_profiles.py" >/dev/null; do sleep 20; done
sleep 10

# within-family sharing for the new families (occ_gender profiles for siblings)
echo "### H profiles (new-family siblings)"
python save_profiles.py --models olmo1b olmo7b falcon1b falcon3b granite2b granite8b \
  --axes occ_gender --batch-size 16 || echo "H PROFILES FAILED"

for ARM in olmo falcon granite; do
  echo "### H arm=$ARM (occ_gender, attn)"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 \
    --origins inherited --designers self sibling cross cross2 random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size 16 --variant-set a \
    || echo "H $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done

echo "### H DONE"
