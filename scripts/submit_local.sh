#!/usr/bin/env bash
# Run a unit list on this machine. One worker per GPU.
#
#   bash scripts/submit_local.sh work/v11dec.units.json
#   WORKERS=4 bash scripts/submit_local.sh work/v11dec.units.json
#
# Worker w of W takes every unit where (index % W == w), so workers never
# collide and any worker can die without stranding the others' units. Each
# worker pins itself to one GPU: the runners load trainable targets with
# dispatch=False, so two workers on one card will OOM.
#
# Re-running is always safe -- run_unit.py skips units the artifacts already
# show as complete.
set -uo pipefail

UNITS="${1:?usage: submit_local.sh <units.json> }"
WORKERS="${WORKERS:-1}"
LOGDIR="${LOGDIR:-logs/$(basename "${UNITS%.units.json}")}"
mkdir -p "$LOGDIR"

N=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['units']))" "$UNITS")
echo "units: $N   workers: $WORKERS   logs: $LOGDIR"
[ "$N" -eq 0 ] && { echo "nothing to do"; exit 0; }

pids=()
for ((w=0; w<WORKERS; w++)); do
  (
    # One GPU per worker when several are visible; otherwise share device 0.
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then export CUDA_VISIBLE_DEVICES="$w"; fi
    rc=0
    for ((i=w; i<N; i+=WORKERS)); do
      log="$LOGDIR/unit_${i}.log"
      echo "[w$w] unit $i -> $log"
      if ! python3 scripts/run_unit.py "$UNITS" --index "$i" >"$log" 2>&1; then
        echo "[w$w] unit $i FAILED (see $log)"
        rc=1
        # Keep going: one bad cell must not strand the rest of the panel.
      fi
    done
    exit $rc
  ) &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo
echo "--- remaining after this pass ---"
python3 scripts/plan.py "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['config'])" "$UNITS")" --dry-run | sed -n '3,5p'
[ $fail -eq 0 ] && echo "all units returned 0" || echo "SOME UNITS FAILED — see $LOGDIR"
exit $fail
