#!/usr/bin/env bash
# Overnight v7 completion queue. SERIAL — one GPU worker at a time (31 GB host RAM).
# Waits for the running STE job, then: INLP -> StereoSet axis.
# Every stage is resumable and stops on no-progress rather than spinning.
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/overnight_queue.log
say(){ echo "[ovn $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
cnt(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }

say "waiting for the STE job"
while pgrep -f 'run_ste_signs.py' >/dev/null || pgrep -f 'run_ste_queue.sh' >/dev/null; do sleep 60; done
say "STE finished at $(cnt results/v6trace/ste/removal.jsonl) rows"

stage(){ # $1 label  $2 panel-file  $3 script...
  local lbl=$1 pf=$2; shift 2
  say "stage: $lbl"
  for t in 1 2 3 4 5 6; do
    local c=$(cnt "$pf"); say "  $lbl attempt $t at $c rows"
    "$@" >> $LOG 2>&1
    local a=$(cnt "$pf"); say "  $lbl -> $c -> $a"
    grep -aq "ALL DONE" $LOG && [ "$a" -gt 0 ] && { say "  $lbl complete at $a"; return 0; }
    [ "$a" -le "$c" ] && { say "  $lbl no progress — moving on"; return 1; }
  done
}

stage "INLP" results/v6trace/inlp/removal.jsonl /opt/conda/bin/python3 run_inlp_baseline.py
stage "StereoSet" results/v6trace/ss/removal.jsonl /opt/conda/bin/python3 run_ss_removal.py
say "overnight queue done"
