#!/usr/bin/env bash
# Re-run SPC's 7-9B tier after the binarize memory fix.
#
# SPC big was SIGKILLed (rc=137) at 37/48 rows on 2026-08-29 running ALONE: the
# global-threshold `torch.cat` in binarize() held a full extra copy of the
# contrast vector (5.4 GB on llama-8B) while the edit was built. colab_t2t4.py
# now drops it immediately after taking the threshold, verified bit-identical
# across all granularities and sparsities. run_spc.py is resumable on
# (target,axis,seed,designer,variant), so the 37 completed rows are kept.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.:src
say() { echo "[v8q3 $(date +'%m-%d %H:%M:%S')] $*" | tee -a results/v8_queue.log; }
say "armed; waiting for queues 1 and 2"
while pgrep -f "run_v8_queue.sh" >/dev/null || pgrep -f "run_v8_queue2.sh" >/dev/null \
   || pgrep -f "run_spc.py" >/dev/null || pgrep -f "run_ins.py" >/dev/null; do sleep 60; done
say "starting SPC big retry"
python3 run_spc.py big >> results/v8spc.log 2>&1; say "SPC big (retry) rc=$?"
python3 src/v8_figures.py >> results/v8_figures.log 2>&1; say "figures rc=$?"
say "queue3 done"
