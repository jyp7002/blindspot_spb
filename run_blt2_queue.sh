#!/usr/bin/env bash
# BL-T continued + SC7, SERIAL (31 GB host RAM).
#   1 SentenceDebias (bias-bench port)
#   2 SC7: 7-9B ablation with cross + same-large designers (paired fp-vs-binary)
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/blt2_queue.log
say(){ echo "[blt2 $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
cnt(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }
say "SentenceDebias"
for t in 1 2 3 4 5 6; do
  c=$(cnt results/v6trace/sentdebias/removal.jsonl); say "  attempt $t at $c rows"
  /opt/conda/bin/python3 run_sentdebias_baseline.py >> $LOG 2>&1
  a=$(cnt results/v6trace/sentdebias/removal.jsonl); say "  -> $c -> $a"
  grep -aq "\[sd\] ALL DONE" $LOG && { say "  SentenceDebias complete at $a"; break; }
  [ "$a" -le "$c" ] && { say "  no progress — stopping"; break; }
done
say "SC7: 7-9B ablation, cross + same-large designers"
for t in 1 2 3 4 5 6; do
  c=$(cnt results/v6trace/ablate_big/removal.jsonl); say "  attempt $t at $c rows"
  SC7=1 /opt/conda/bin/python3 run_ablate_attn.py big >> $LOG 2>&1
  a=$(cnt results/v6trace/ablate_big/removal.jsonl); say "  -> $c -> $a"
  [ "$a" -le "$c" ] && { say "  no progress/complete — stopping SC7"; break; }
done
say "done"
