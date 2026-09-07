#!/usr/bin/env bash
# Stage 3, chained AFTER the fx queue rather than appended to it (editing a
# running bash script shifts its byte offsets and can desynchronize execution).
# Waits for the fx queue to exit, then runs the attn-protocol ablation.
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/ablate.log
say(){ echo "[ablq $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
cnt(){ [ -f results/v6trace/ablate/removal.jsonl ] && wc -l < results/v6trace/ablate/removal.jsonl || echo 0; }
say "waiting for fx queue to finish (serial: 31 GB host RAM)"
while pgrep -f 'run_fx_queue.sh' >/dev/null || pgrep -f 'run_fx_granite.py' >/dev/null \
   || pgrep -f 'run_fx_nulls.py' >/dev/null; do sleep 60; done
say "fx queue done; starting ablation"
for try in 1 2 3 4 5 6 7 8; do
  c=$(cnt); [ "$c" -ge 168 ] && { say "complete $c/168"; exit 0; }
  say "attempt $try at $c/168"
  /opt/conda/bin/python3 run_ablate_attn.py >> $LOG 2>&1
  a=$(cnt); say "  -> $c -> $a"
  [ "$a" -le "$c" ] && { say "no progress — stopping"; exit 1; }
done
