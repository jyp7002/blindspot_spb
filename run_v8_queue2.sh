#!/usr/bin/env bash
# v8 follow-up queue. Waits for the main queue to exit, THEN runs serially.
#
# WHY THIS EXISTS: run_sup1_designers.py was OOM-killed (rc=137) on 2026-08-28
# because it was running concurrently with run_dec_natural.py -- a 7B fp32 delta
# set plus a 3B one exceeded the 31 GB host. That is the same failure mode
# RESULTS_METHOD.md §8 records nine times over. The lesson is not "check RAM
# first", it is "do not run two delta-materialising jobs at once", so this script
# waits rather than estimates.
#
# Everything here is resumable, so the OOM cost time, not data: the two dumps
# that completed are kept and skipped.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.:src
LOG=results/v8_queue.log
say() { echo "[v8q2 $(date +'%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "follow-up queue armed; waiting for main queue and natural arm to exit"
while pgrep -f "run_v8_queue.sh" >/dev/null || pgrep -f "run_dec_natural.py" >/dev/null; do
  sleep 60
done
say "predecessors done; starting serial follow-up"

# 1. finish the SUP1 designer-overlap dumps the OOM interrupted
python3 run_sup1_designers.py >> results/v8sup1.log 2>&1
say "SUP1 designers (retry) rc=$?"

# 2. re-run the SUP1 analysis now that all designer dumps exist
python3 src/v8_sup1.py results/v8dec >> results/v8sup1.log 2>&1
say "SUP1 analysis (rerun) rc=$?"

# 3. deep-sparsity tie-in; self-gates on Outcome A and refuses otherwise
python3 run_dec_deep.py >> results/v8dec_deep.log 2>&1
say "DEC deep (conditional) rc=$?"

# 4. refresh every figure against final artifacts
python3 src/v8_figures.py >> results/v8_figures.log 2>&1
say "figures rc=$?"

say "follow-up queue done"
