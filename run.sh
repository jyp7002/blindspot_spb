#!/usr/bin/env bash
# The single entrypoint. You should not need to read any other script.
#
#   bash run.sh              status: what exists, what is done, what is left
#   bash run.sh dec          the decomposition panel   (18 cells, ~54 units)
#   bash run.sh alphaext     the alpha-saturation panel (10 cells, 30 units)
#   bash run.sh ifeval-check the 2 PUBLISHED IFEval cells only -- run this
#                            first: it decides whether the arm is usable at all
#   bash run.sh ifeval       the full instruction-following panel (6 units)
#   bash run.sh all          dec + alphaext + ifeval, in that order
#   bash run.sh big          32B/70B -- needs a >=80 GB card, NOT this box
#
# v12 (experiments_v12.md) -- operating-point selection:
#   bash run.sh opsel-cal       calibration panel, <=9B (33 units) -- p* is FIT here
#   bash run.sh dec1k           headline DEC cells re-read at 1000 MMLU items
#   bash run.sh opsel-big-geom  27B/32B PHASE A: geometry only, no probe (>=80 GB)
#   bash run.sh frontier        Table A methods re-measured with traces + IFEval (24 units)
#   bash run.sh opsel-big       27B/32B PHASE B: refuses without a frozen p*
#   bash run.sh astar           alpha* on the 10 registered <=9B DEC cells (§v12.E, 30 units)
#
# v13 (experiments_v13.md, PREREGISTRATION §v13) -- SEED and FLOOR:
#   bash scripts/v13_local.sh   everything <=9B, in the registered order (L40S)
#   bash run.sh v13-seed-big    SEED 27-32B  (>=80 GB; cell count from §v13.F)
#   bash run.sh v13-floor-big   FLOOR 27-32B (>=80 GB)
#   bash run.sh v13-floor-seed  v13.E conditional arm; refuses unless licensed
#                               (python3 src/v12_pstar.py fit && ... predict)
#
# Everything is resumable. If a run dies, re-run the same command: finished
# units are skipped because completion is read from the artifacts on disk.
#
#   WORKERS=4 bash run.sh dec     use 4 GPUs (one worker per GPU)
set -uo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
WORKERS="${WORKERS:-1}"

declare -A CFG=(
  [dec]=configs/v11/dec_v11.yaml
  [alphaext]=configs/v11/alphaext_v11.yaml
  [ifeval]=configs/v11/ifeval_v11.yaml
  [big]=configs/v11/big_v11.yaml
  [opsel-cal]=configs/v12/opsel_calib.yaml
  [dec1k]=configs/v12/dec1k.yaml
  [opsel-big-geom]=configs/v12/opsel_big.yaml
  [opsel-big]=configs/v12/opsel_big.yaml
  [frontier]=configs/v12/frontier.yaml
  [astar]=configs/v12/astar_dec.yaml
  [v13-replay]=configs/v13/replay.yaml
  [v13-det-nd-a]=configs/v13/det_nd_a.yaml
  [v13-det-nd-b]=configs/v13/det_nd_b.yaml
  [v13-det-d-a]=configs/v13/det_d_a.yaml
  [v13-det-d-b]=configs/v13/det_d_b.yaml
  [v13-seed]=configs/v13/seed_dec.yaml
  [v13-floor]=configs/v13/floor_cal.yaml
  [v13-floor-big]=configs/v13/floor_big.yaml
  [v13-floor-seed]=configs/v13/floor_seed.yaml
)
# opsel panels run in phases; the plan file carries the phase in its name
declare -A PHASE=(
  [opsel-cal]=full
  [dec1k]=full
  [opsel-big-geom]=geometry
  [opsel-big]=full
  [astar]=full
  [v13-replay]=full
  [v13-det-nd-a]=full
  [v13-det-nd-b]=full
  [v13-det-d-a]=full
  [v13-det-d-b]=full
  [v13-seed]=full
  [v13-seed-big]=full
  [v13-floor]=full
  [v13-floor-big]=full
  [v13-floor-seed]=full
)
# Order matters: dec produces the panel the alpha extension extends.
ORDER=(dec alphaext ifeval)

status() {
  echo "==================================================================="
  echo " blindspot_spb — v11 scale-up status"
  echo "==================================================================="
  for k in "${ORDER[@]}" big opsel-cal dec1k frontier opsel-big-geom opsel-big astar v13-replay v13-floor v13-seed v13-floor-big; do
    printf '\n--- %s (%s)\n' "$k" "${CFG[$k]}"
    local ph=""
    [ -n "${PHASE[$k]:-}" ] && ph="--phase ${PHASE[$k]}"
    $PY scripts/plan.py "${CFG[$k]}" $ph --dry-run 2>/dev/null \
      | grep -E '^(panel|cells|units|excluded)' | sed 's/^/    /'
  done
  cat <<'TXT'

-------------------------------------------------------------------
  bash run.sh all      run everything that fits this machine
  bash run.sh dec      run just the decomposition panel
  bash setup.sh        if a preflight check below fails

  'big' needs a >=80 GB card and is not launched by 'all'.
-------------------------------------------------------------------
TXT
  echo
  $PY scripts/preflight.py --no-cache 2>&1 | tail -n 4
}

run_one() {
  local key="$1" cfg="${CFG[$1]:-}"
  if [ "$key" = "v13-seed-big" ]; then
    # §v13.F fixes 4 or 8 cells before the first v13 unit; read it, never guess
    local n
    n=$(grep -E '^V13F_SEED_BIG_CELLS = [48]$' PREREGISTRATION.md | awk '{print $3}')
    if [ -z "$n" ]; then
      echo "REFUSED: §v13.F (SEED 27-32B cell count) is not recorded in PREREGISTRATION.md."
      return 1
    fi
    cfg=configs/v13/seed_big_${n}.yaml
  fi
  echo
  echo "==================================================================="
  echo " RUN: $key   ($cfg)"
  echo "==================================================================="

  $PY scripts/preflight.py --config "$cfg" || {
    echo
    echo "PREFLIGHT FAILED for '$key' — not launching."
    echo "Run 'bash setup.sh' if the stack is missing, or pick a bigger node."
    return 1
  }

  local ph="" sfx=""
  if [ -n "${PHASE[$key]:-}" ]; then
    ph="--phase ${PHASE[$key]}"; sfx=".${PHASE[$key]}"
  fi
  if [ "$key" = "opsel-big" ] && [ ! -f results/v12/pstar_prediction.json ]; then
    echo "REFUSED: no frozen p* prediction (results/v12/pstar_prediction.json)."
    echo "Order: opsel-cal -> opsel-big-geom -> v12_pstar.py fit/predict -> opsel-big"
    return 1
  fi

  if [ "$key" = "v13-floor-seed" ] && ! $PY -c "
import json,sys; sys.exit(0 if json.load(open('results/v13/verdicts.json')).get('floor_seed_2b_licensed') else 1)" 2>/dev/null; then
    echo "REFUSED: v13.E runs only if SEED (<=9B) is NON-INFERIOR and FLOOR MOVES."
    echo "Run python3 src/v13_analyze.py verdicts after both <=9B panels complete."
    return 1
  fi

  $PY scripts/plan.py "$cfg" $ph || return 1

  local units="work/$($PY -c "
import sys,yaml
c=yaml.safe_load(open('$cfg'))
print(c.get('out_panel', c['panel']))")${sfx}.units.json"

  local n
  n=$($PY -c "import json;print(len(json.load(open('$units'))['units']))" 2>/dev/null || echo 0)
  if [ "$n" -eq 0 ]; then
    echo "nothing left to run for '$key' — panel is complete."
    return 0
  fi

  echo
  echo "launching $n units with WORKERS=$WORKERS ..."
  WORKERS="$WORKERS" bash scripts/submit_local.sh "$units"
}

case "${1:-status}" in
  status|"")   status ;;
  ifeval-check)
    # The lm-eval version behind the PUBLISHED IFEval numbers is recorded
    # nowhere (see configs/v11/ifeval_v11.yaml). Replay those two cells under
    # the pinned harness BEFORE spending anything on the four new ones: if they
    # do not reproduce, the published numbers are version-dependent and that is
    # the finding -- extending the arm on top of it would be building on sand.
    python3 scripts/preflight.py --config configs/v11/ifeval_v11.yaml || exit 1
    python3 scripts/plan.py configs/v11/ifeval_v11.yaml --only published || exit 1
    WORKERS="$WORKERS" bash scripts/submit_local.sh work/v11ins.published.units.json ;;
  dec|alphaext|ifeval|big|opsel-cal|dec1k|frontier|opsel-big-geom|opsel-big|astar|v13-*) run_one "$1" ;;
  all)
    rc=0
    for k in "${ORDER[@]}"; do run_one "$k" || rc=1; done
    echo
    echo "=== all panels attempted; rc=$rc ==="
    echo "ship the artifacts back with:  bash scripts/pack_artifacts.sh <panel>"
    exit $rc ;;
  *)
    echo "unknown target: $1"
    echo "usage: bash run.sh [status|dec|alphaext|ifeval-check|ifeval|all|big|opsel-cal|dec1k|frontier|opsel-big-geom|opsel-big|astar]"
    exit 2 ;;
esac
