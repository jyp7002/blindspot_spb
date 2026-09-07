#!/usr/bin/env bash
# SPC big, second retry, after fixing the ALLOCATION peak (not just the lifetime).
# First retry still died because the peak is DURING torch.cat, which materialises
# every abs() copy before allocating its result: 3x the contrast vector = 15 GiB
# on an 8B attn set. binarize now fills one preallocated buffer and selects with
# numpy.partition in place -> 2x. Verified bit-identical.
# Only llama seed 2 remains (43/48 rows done); resume keeps the rest.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.:src
say() { echo "[v8q5 $(date +'%m-%d %H:%M:%S')] $*" | tee -a results/v8_queue.log; }
say "armed; waiting for all other v8 jobs"
while pgrep -f "run_v8_queue[234].sh" >/dev/null || pgrep -f "run_spc.py" >/dev/null \
   || pgrep -f "run_ins.py" >/dev/null || pgrep -f "run_dec" >/dev/null \
   || pgrep -f "run_sup1_designers.py" >/dev/null; do sleep 30; done
say "starting SPC big retry #2"
python3 run_spc.py big >> results/v8spc.log 2>&1; say "SPC big (retry2) rc=$?"
python3 src/v8_figures.py >> results/v8_figures.log 2>&1; say "figures rc=$?"
say "queue5 done"
