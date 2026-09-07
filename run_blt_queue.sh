#!/usr/bin/env bash
# BL-T queue, SERIAL (31 GB host RAM — never two workers).
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/blt_queue.log
say(){ echo "[blt $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
cnt(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }
say "BL-T/DPO (TRL DPOTrainer, informative pairs only)"
for t in 1 2 3 4 5 6; do
  c=$(cnt results/v6trace/dpo/removal.jsonl)
  say "  attempt $t at $c rows"
  /opt/conda/bin/python3 run_dpo_baseline.py >> $LOG 2>&1
  a=$(cnt results/v6trace/dpo/removal.jsonl); say "  -> $c -> $a"
  grep -aq "\[dpo\] ALL DONE" $LOG && { say "  DPO complete at $a rows"; break; }
  [ "$a" -le "$c" ] && { say "  no progress — stopping"; break; }
done
say "done"
