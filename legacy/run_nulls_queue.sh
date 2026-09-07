#!/usr/bin/env bash
# Both MV-B nulls, serially (31 GB host RAM — never two workers).
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/nulls.log
for n in sign_shuffle partition; do
  for try in 1 2 3 4 5 6 7 8; do
    c=$( [ -f results/v6trace/mvb_null_$n/removal.jsonl ] && wc -l < results/v6trace/mvb_null_$n/removal.jsonl || echo 0 )
    [ "$c" -ge 36 ] && break
    echo "[nulls $(date +%H:%M:%S)] $n attempt $try at $c/36" | tee -a $LOG
    /opt/conda/bin/python3 run_mvb_nulls.py $n >> $LOG 2>&1
    a=$( [ -f results/v6trace/mvb_null_$n/removal.jsonl ] && wc -l < results/v6trace/mvb_null_$n/removal.jsonl || echo 0 )
    echo "[nulls $(date +%H:%M:%S)] $n rc=$? cells $c -> $a" | tee -a $LOG
    [ "$a" -le "$c" ] && { echo "[nulls] $n no progress — stopping" | tee -a $LOG; break; }
  done
done
echo "[nulls $(date +%H:%M:%S)] done" | tee -a $LOG
