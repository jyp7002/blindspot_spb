#!/usr/bin/env bash
# Scale-up stages, SERIAL, chained behind whatever is already running.
#   stage A  7-9B ablation      (ablate_big, 84 cells) — method claims above 3.8B
#   stage B  multi-axis breadth (axes, 48 cells)       — 4 more CrowS axes
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/scale_queue.log
say(){ echo "[sq $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
say "waiting for the method queue (serial: 31 GB host RAM)"
while pgrep -f 'run_method_queue.sh' >/dev/null || pgrep -f 'run_prompt_baseline.py' >/dev/null \
   || pgrep -f 'run_ablate_attn.py' >/dev/null; do sleep 60; done
cnt(){ [ -f "$1" ] && wc -l < "$1" || echo 0; }
say "stage A: 7-9B ablation (84 cells)"
for t in 1 2 3 4 5 6; do
  c=$(cnt results/v6trace/ablate_big/removal.jsonl)
  [ "$c" -ge 84 ] && { say "  stage A complete $c/84"; break; }
  say "  attempt $t at $c/84"
  /opt/conda/bin/python3 run_ablate_attn.py big >> $LOG 2>&1
  a=$(cnt results/v6trace/ablate_big/removal.jsonl); say "  -> $c -> $a"
  [ "$a" -le "$c" ] && { say "  no progress — stopping stage A"; break; }
done
say "stage B: multi-axis breadth (48 cells)"
for t in 1 2 3 4 5 6; do
  c=$(cnt results/v6trace/axes/removal.jsonl)
  [ "$c" -ge 48 ] && { say "  stage B complete $c/48"; break; }
  say "  attempt $t at $c/48"
  /opt/conda/bin/python3 run_axes_breadth.py >> $LOG 2>&1
  a=$(cnt results/v6trace/axes/removal.jsonl); say "  -> $c -> $a"
  [ "$a" -le "$c" ] && { say "  no progress — stopping stage B"; break; }
done
say "done"
