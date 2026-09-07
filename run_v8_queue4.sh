#!/usr/bin/env bash
# INS arm BP: format-controlled base<->instruct comparison.
# Waits for every other v8 job -- one GPU, and two delta-materialising jobs at
# once is what SIGKILLed run_sup1_designers.py earlier tonight.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.:src
say() { echo "[v8q4 $(date +'%m-%d %H:%M:%S')] $*" | tee -a results/v8_queue.log; }
say "armed; waiting for all other v8 jobs"
while pgrep -f "run_v8_queue.sh" >/dev/null || pgrep -f "run_v8_queue2.sh" >/dev/null \
   || pgrep -f "run_v8_queue3.sh" >/dev/null || pgrep -f "run_spc.py" >/dev/null \
   || pgrep -f "run_ins.py" >/dev/null || pgrep -f "run_sup1_designers.py" >/dev/null \
   || pgrep -f "run_dec" >/dev/null; do sleep 60; done
say "starting INS-BP"
python3 run_ins.py BP >> results/v8ins.log 2>&1; say "INS-BP rc=$?"
python3 src/v8_figures.py >> results/v8_figures.log 2>&1; say "figures rc=$?"
say "queue4 done"
