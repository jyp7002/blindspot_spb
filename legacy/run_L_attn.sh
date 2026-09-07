#!/usr/bin/env bash
# experiments_v3 L — complete the attn(q,k,v,o) r16 migration.
#
# Re-run the three A arms still only measured at q/v (gemma, llama, smol) at
# attn r16, both templated origins, all designers + both nulls, 3 seeds. qwen
# and phi are already done at attn (the v2 spot-check). Then the 5-arm pooled
# interaction at attn can be estimated and attn r16 formally frozen.
#
# gemma runs at batch 12 (256k vocab).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_experiment.py|save_profiles.py|edit_capacity.py" >/dev/null; do sleep 30; done
sleep 10

for spec in "gemma 12" "llama 32" "smol 32"; do
  set -- $spec; ARM=$1; BS=$2
  echo "### L attn arm=$ARM"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 \
    --origins inherited acquired \
    --designers self sibling cross cross2 cross3 cross4 random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size "$BS" --variant-set a \
    || echo "L $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done

echo "### L DONE"
