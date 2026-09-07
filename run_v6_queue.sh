#!/usr/bin/env bash
# v6 experiment queue — runs SERIALLY, one GPU worker at a time.
#
# HARD CONSTRAINT: 31 GB host RAM. Loading one 7-9B model in bf16 stages ~16-19 GB
# through CPU memory before it reaches the GPU, so two concurrent workers are
# OOM-killed by the cgroup (observed 2026-07-29: oom_kill 2). GPU memory (46 GB)
# is NOT the binding constraint. Never run two stages of this queue in parallel.
#
# Stage 1  t2x  (7-9B)  — F2 half of MV-A/MV-C. 144 cells.
# Stage 2  x2   (bridge 2.5-3.8B) — F1 at bridge sizes under the SAME protocol as
#          t2x, which the runs.jsonl adapter cannot give (that panel is the
#          v2/v3-era 1-2B tier). This is what ORD's bridge cells actually need.
#
# Every stage is resumable at (target, axis, seed, designer) via its removal.jsonl,
# so a kill costs at most the in-flight cell.

set -u
cd "$(dirname "$0")"
LOG=results/v6trace/queue.log
mkdir -p results/v6trace

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say() { echo "[queue $(date +%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG"; }

cells() { [ -f "$1" ] && wc -l < "$1" || echo 0; }

# ---------------- concurrency guard ----------------
# A supervisor/worker may already be running (this queue is often started while
# stage 1 is in flight). Starting a second one would put two 7-9B loads in 31 GB
# of RAM and OOM-kill both. Wait it out instead.
if pgrep -f "[r]un_v6_trace.sh" >/dev/null || pgrep -f "[c]olab_t2t4.py --do t2cross" >/dev/null; then
  say "an existing worker/supervisor is running — waiting for it rather than starting a second"
  while pgrep -f "[r]un_v6_trace.sh" >/dev/null || pgrep -f "[c]olab_t2t4.py --do t2cross" >/dev/null; do
    sleep 60
  done
  say "existing worker finished at $(cells results/v6trace/t2x/removal.jsonl)/144 cells"
fi

# ---------------- stage 1: t2x (7-9B) ----------------
n=$(cells results/v6trace/t2x/removal.jsonl)
if [ "$n" -lt 144 ]; then
  say "stage 1 t2x — supervised serial run to 144 cells (at $n)"
  TARGET_CELLS=144 ./run_v6_trace.sh
else
  say "stage 1 t2x already complete ($n/144) — skipping"
fi
n=$(cells results/v6trace/t2x/removal.jsonl)
say "stage 1 ended at $n/144 cells"
if [ "$n" -lt 144 ]; then
  say "stage 1 INCOMPLETE — stopping the queue rather than starting stage 2 on a partial panel"
  exit 1
fi

# ---------------- stage 2: x2 bridge tier ----------------
# run_x2_bridge.py drives colab_t2t4.panel_run, which now carries the alpha_trace
# patch, so this emits results/v6trace/x2/alpha_trace.jsonl with no code change.
say "stage 2 x2 bridge — alpha trace at 2.5-3.8B"
BS_OUT=results/v6trace /opt/conda/bin/python3 run_x2_bridge.py >> "$LOG" 2>&1
rc=$?
say "stage 2 exited rc=$rc, cells=$(cells results/v6trace/x2/removal.jsonl)"

say "queue done"
