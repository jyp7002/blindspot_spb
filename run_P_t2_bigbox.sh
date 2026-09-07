#!/usr/bin/env bash
# H1 at T2 (7-9B targets) — RUN ON THE BIGGER GPU BOX (alongside T4).
# The small box (31GB RAM / 46GB GPU / no swap) can only train qwen7b; the
# 8-9B targets OOM. On a bigger box this runs unmodified. Reuses the same repo
# and appends to results/runs.jsonl (bring it back and I merge).
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/src"
for ARM in olmo_t2 granite_t2 llama_t2 gemma_t2; do
  echo "### P-BIGBOX arm=$ARM"
  BS=12; [ "$ARM" = "gemma_t2" ] && BS=6   # gemma9b 256k vocab
  python run_experiment.py --arm "$ARM" --seeds 0 1 2 --origins inherited \
    --designers self sibling cross random --targets q_proj k_proj v_proj o_proj \
    --alphas 2 4 8 16 --steps 250 --batch-size "$BS" --variant-set a \
    || echo "$ARM FAILED"
done
echo "### P-BIGBOX DONE — bring back results/runs.jsonl (new *_t2 rows)"
