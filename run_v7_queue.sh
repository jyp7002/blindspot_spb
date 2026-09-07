#!/usr/bin/env bash
# v7 execution queue, SERIAL (31 GB host RAM — never two workers).
#   BL-S  steering baseline (THE GATE — fork pre-registered before this ran)
#   SC7   7-9B arm: cross + same-large designers for the paired fp-vs-binary test
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/v7_queue.log
say(){ echo "[v7 $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
cnt(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }
say "BL-S steering baseline (gate)"
for t in 1 2 3 4 5 6 7 8; do
  c=$(cnt results/v6trace/steer/removal.jsonl)
  say "  attempt $t at $c rows"
  /opt/conda/bin/python3 run_steering_baseline.py >> $LOG 2>&1
  a=$(cnt results/v6trace/steer/removal.jsonl); say "  -> $c -> $a"
  grep -aq "\[steer\] ALL DONE" $LOG && { say "  BL-S complete at $a rows"; break; }
  [ "$a" -le "$c" ] && { say "  no progress — stopping BL-S"; break; }
done
say "queue done (BL-S)"
