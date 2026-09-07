#!/usr/bin/env bash
# Sequential driver: finish whatever is running, then work down the queue.
# Everything is resumable, so re-running this is always safe.
set -uo pipefail
cd "$(dirname "$0")/.."
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

# 1. wait for any live queue (two on one GPU would OOM)
while pgrep -f submit_local.sh >/dev/null; do sleep 60; done
log "no queue running"

# 2. alphaext — finish it if anything is left
python3 scripts/plan.py configs/v11/alphaext_v11.yaml >/dev/null 2>&1
n=$(python3 -c "import json;print(len(json.load(open('work/v11ext.units.json'))['units']))" 2>/dev/null || echo 0)
if [ "$n" -gt 0 ]; then
  log "alphaext: $n units left"
  bash scripts/submit_local.sh work/v11ext.units.json
else
  log "alphaext complete"
fi
python3 src/v11_ext_analyze.py > logs/v11ext_analysis.txt 2>&1 && log "alphaext analysed"

# 3. ifeval-check — the TWO PUBLISHED cells only. Not the full arm: the user
#    asked for it not to be run in full, and these two decide whether the arm
#    is usable at all (the lm-eval version behind the published numbers is
#    recorded nowhere).
python3 scripts/plan.py configs/v11/ifeval_v11.yaml --only published >/dev/null 2>&1
n=$(python3 -c "import json;print(len(json.load(open('work/v11ins.published.units.json'))['units']))" 2>/dev/null || echo 0)
if [ "$n" -gt 0 ]; then
  log "ifeval-check: $n units"
  bash scripts/submit_local.sh work/v11ins.published.units.json
else
  log "ifeval-check complete"
fi

# 4. refresh the DEC analysis and report where everything stands
python3 src/v11_dec_analyze.py > logs/v11dec_analysis.txt 2>&1
log "=== final state ==="
bash scripts/progress.sh
python3 scripts/check_reproduction.py 2>/dev/null | tail -3
log "chain done"
