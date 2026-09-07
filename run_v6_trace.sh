#!/usr/bin/env bash
# Supervised t2x re-run for the v6 alpha/collateral trace (experiments_v6 MV-A/MV-C).
#
# SERIAL BY NECESSITY: this box has 31 GB of host RAM and loading one 7-9B model
# in bf16 stages ~16-19 GB through CPU memory before it reaches the GPU. Two
# concurrent workers exceed 31 GB and are SIGKILLed by the cgroup OOM killer
# (observed 2026-07-29: oom_kill 2, both workers lost, ~5.5 h idle before it was
# noticed). GPU memory (46 GB) is NOT the binding constraint -- host RAM is.
# Do not parallelize this on this box.
#
# The work is resumable at (target, axis, seed, designer) via removal.jsonl, so a
# kill costs at most the in-flight cell. The supervisor restarts on any non-clean
# exit and stops when the panel is complete or the retry budget is spent.

set -u
cd "$(dirname "$0")"

OUT=results/v6trace
PANEL=$OUT/t2x/removal.jsonl
LOG=$OUT/run_serial_supervised.log
TARGET_CELLS=${TARGET_CELLS:-144}
MAX_TRIES=${MAX_TRIES:-40}

export BS_OUT=$OUT
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cells() { [ -f "$PANEL" ] && wc -l < "$PANEL" || echo 0; }

for try in $(seq 1 "$MAX_TRIES"); do
  n=$(cells)
  if [ "$n" -ge "$TARGET_CELLS" ]; then
    echo "[sup $(date +%H:%M:%S)] complete: $n/$TARGET_CELLS cells" | tee -a "$LOG"
    exit 0
  fi
  echo "[sup $(date +%H:%M:%S)] attempt $try — $n/$TARGET_CELLS cells done, starting worker" | tee -a "$LOG"

  /opt/conda/bin/python3 colab_t2t4.py --do t2cross \
      --t2x-targets ${T2X_TARGETS:-qwen llama gemma olmo} \
      --t2x-axes occ_gender crows_socioeconomic \
      --t2x-seeds 0 1 2 \
      --t2x-batch 6 --t2x-train-bs 8 >> "$LOG" 2>&1
  rc=$?

  after=$(cells)
  echo "[sup $(date +%H:%M:%S)] worker exited rc=$rc — cells $n -> $after" | tee -a "$LOG"

  # No progress on a non-zero exit means the failure is deterministic, not an
  # OOM we can retry through; stop rather than spin.
  if [ "$rc" -ne 0 ] && [ "$after" -le "$n" ]; then
    echo "[sup] worker failed with NO progress — stopping, needs diagnosis" | tee -a "$LOG"
    exit 1
  fi
  [ "$rc" -eq 0 ] && [ "$after" -ge "$TARGET_CELLS" ] && { echo "[sup] done" | tee -a "$LOG"; exit 0; }
  sleep 20
done
echo "[sup] retry budget exhausted at $(cells)/$TARGET_CELLS cells" | tee -a "$LOG"
exit 1
