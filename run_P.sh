#!/usr/bin/env bash
# experiments_v4 P — H1 replication at the 7-9B (T2) tier (defensibility item).
# occ_gender removal on 5 T2 targets (qwen7b, llama8b, gemma9b, olmo7b,
# granite8b), self+sibling+cross+random at attn r16. Plus profiles for the new
# T2 models (qwen7b, llama8b) so within-family sharing extends to T2 (also
# firms up O1: same-family large pairs). H1 = ρ(sharing, self-removal) at T2.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py|download_t2.py" >/dev/null; do sleep 20; done
sleep 5

echo "### P profiles (qwen7b, llama8b for T2 sharing + O1)"
python save_profiles.py --models qwen7b llama8b --axes occ_gender --batch-size 12 || echo "P PROFILES FAILED"

# gemma9b needs batch 12 (256k vocab); others batch 12 for 7-9B headroom
for spec in "qwen_t2 12" "llama_t2 12" "olmo_t2 12" "granite_t2 12" "gemma_t2 8"; do
  set -- $spec; ARM=$1; BS=$2
  echo "### P arm=$ARM"
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 \
    --origins inherited --designers self sibling cross random \
    --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size "$BS" --variant-set a \
    || echo "P $ARM FAILED"
  while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done
done
echo "### P DONE"
