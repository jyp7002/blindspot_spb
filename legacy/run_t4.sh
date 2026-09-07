#!/usr/bin/env bash
# ============================================================================
# T4 driver — run on a BIGGER (multi-)GPU box, then bring results/t4/ back.
# ============================================================================
# Profiles + designer inference ONLY (no training). Each model is independent;
# reorder / comment out to fit your VRAM. bf16 sizes:
#   gemma-2-27b  ~54 GB   (256k vocab -> batch 4)
#   qwen2.5-32b  ~64 GB
#   llama-3.1-70b ~140 GB (needs 2x80GB or 4x48GB, device_map=auto shards)
#   qwen2.5-72b  ~144 GB
# device_map="auto" shards across all visible GPUs automatically.
#
# Prereqs: pip install torch transformers datasets numpy accelerate
#          copy the whole src/ directory; run from src/.
set -e
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/src"

run () {  # <hf_id> <name> <batch>
  echo "=========== $2 ==========="
  python t4_runner.py --model "$1" --name "$2" --role both --batch-size "$3" \
    || echo "!!! $2 FAILED (continuing)"
}

# smallest T4 first so you get a result fastest
run google/gemma-2-27b-it            gemma27b  4
run Qwen/Qwen2.5-32B-Instruct        qwen32b   8
run meta-llama/Llama-3.1-70B-Instruct llama70b 6
run Qwen/Qwen2.5-72B-Instruct        qwen72b   6

echo "### ALL T4 DONE — bring back the entire results/t4/ directory"
