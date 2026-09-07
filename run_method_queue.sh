#!/usr/bin/env bash
# Method-paper compute, SERIAL (31 GB host RAM).
#   stage 1  prompt-debiasing baseline (Schick) — the missing external comparator
#   stage 2  ablation breadth: 2 -> 4 targets (168 -> 336 cells), resumes existing
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/method_queue.log
say(){ echo "[mq $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
say "stage 1: prompt baseline"
for try in 1 2 3; do
  /opt/conda/bin/python3 run_prompt_baseline.py >> $LOG 2>&1 && break
  say "  stage 1 attempt $try failed, retrying"
done
n=$(python3 -c "import json,os;p='results/prompt_baseline.json';print(len(json.load(open(p))) if os.path.exists(p) else 0)")
say "stage 1 done: $n prompt-baseline cells"
say "stage 2: ablation breadth to 336 cells"
cnt(){ [ -f results/v6trace/ablate/removal.jsonl ] && wc -l < results/v6trace/ablate/removal.jsonl || echo 0; }
for try in 1 2 3 4 5 6 7 8; do
  c=$(cnt); [ "$c" -ge 336 ] && { say "stage 2 complete $c/336"; break; }
  say "  attempt $try at $c/336"
  /opt/conda/bin/python3 run_ablate_attn.py >> $LOG 2>&1
  a=$(cnt); say "  -> $c -> $a"
  [ "$a" -le "$c" ] && { say "  no progress — stopping"; break; }
done
say "done"
