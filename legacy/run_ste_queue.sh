#!/usr/bin/env bash
# BL-T2 — STE-trained signs vs sign(delta_fp). SERIAL (31 GB host RAM).
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/ste_queue.log
say(){ echo "[steq $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
cnt(){ [ -f results/v6trace/ste/removal.jsonl ] && wc -l < results/v6trace/ste/removal.jsonl || echo 0; }
for t in 1 2 3 4 5 6; do
  c=$(cnt); say "attempt $t at $c/24 rows"
  /opt/conda/bin/python3 run_ste_signs.py >> $LOG 2>&1
  a=$(cnt); say "  -> $c -> $a"
  grep -aq "\[ste\] ALL DONE" $LOG && { say "complete at $a"; break; }
  [ "$a" -le "$c" ] && { say "no progress — stopping"; break; }
done
say "done"
