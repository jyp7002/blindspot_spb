#!/usr/bin/env bash
# FX follow-ups, SERIAL (31 GB host RAM — never two workers).
#   stage 1  granite discriminator  (gap-rule vs size-rule head-to-head, 24 cells)
#   stage 2  both nulls on the FX cells that changed source (36 cells)
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/fx_queue.log
say(){ echo "[fxq $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
cnt(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }

run_until(){  # $1=panel file  $2=target count  $3...=command
  local f=$1 want=$2; shift 2
  for try in 1 2 3 4 5 6; do
    local c=$(cnt "$f"); [ "$c" -ge "$want" ] && return 0
    say "$(basename $(dirname $f)) attempt $try at $c/$want"
    "$@" >> $LOG 2>&1
    local a=$(cnt "$f"); say "  -> $c -> $a"
    [ "$a" -le "$c" ] && { say "  no progress, stopping this stage"; return 1; }
  done
}

run_until results/v6trace/fxg/removal.jsonl 24 /opt/conda/bin/python3 run_fx_granite.py
for n in sign_shuffle partition; do
  run_until results/v6trace/fxnull_$n/removal.jsonl 18 /opt/conda/bin/python3 run_fx_nulls.py $n
done
say "done"
