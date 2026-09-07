#!/usr/bin/env bash
# Supervised MV-B transplant. Serial only (31 GB host RAM — see run_v6_queue.sh).
# Resumable at (target,axis,seed,designer) via results/v6trace/mvb/removal.jsonl.
set -u
cd "$(dirname "$0")"
LOG=results/v6trace/mvb_run.log
PANEL=results/v6trace/mvb/removal.jsonl
TARGET=${TARGET_CELLS:-198}
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/v6trace/mvb
cells() { [ -f "$PANEL" ] && wc -l < "$PANEL" || echo 0; }
for try in $(seq 1 40); do
  n=$(cells)
  [ "$n" -ge "$TARGET" ] && { echo "[mvb-sup $(date +%H:%M:%S)] complete $n/$TARGET" | tee -a "$LOG"; exit 0; }
  echo "[mvb-sup $(date +%H:%M:%S)] attempt $try at $n/$TARGET" | tee -a "$LOG"
  /opt/conda/bin/python3 run_mvb_transplant.py >> "$LOG" 2>&1
  rc=$?; after=$(cells)
  echo "[mvb-sup $(date +%H:%M:%S)] rc=$rc cells $n -> $after" | tee -a "$LOG"
  if [ "$rc" -ne 0 ] && [ "$after" -le "$n" ]; then
    echo "[mvb-sup] failed with NO progress — stopping" | tee -a "$LOG"; exit 1; fi
  [ "$after" -ge "$TARGET" ] && exit 0
  sleep 20
done
