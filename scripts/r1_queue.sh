#!/usr/bin/env bash
# R1 queue: run one or more passes back to back on this GPU, one unit per process.
#
#   bash scripts/r1_queue.sh "passA:configs/r1/e4.yaml:C-rand" \
#                            "passB:configs/r1/e5.yaml" \
#                            "passC:configs/r1/e4.yaml"
#
# Each argument is  <tag>:<config>[:<comma-separated arms>]  (arms default to the
# config's). A pass walks the config's units in order; run_r1.py skips whatever
# is already on disk, so re-running the same command after a crash resumes.
#
# Uses the same pidfile lock as scripts/submit_local.sh (logs/.queue.pid): two
# queues on one card would OOM, and waiting on a pgrep string match deadlocks.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
QUEUE_PID_FILE="logs/.queue.pid"
mkdir -p logs/r1
if [ -f "$QUEUE_PID_FILE" ] && kill -0 "$(cat "$QUEUE_PID_FILE" 2>/dev/null)" 2>/dev/null; then
  echo "another queue is live (pid $(cat "$QUEUE_PID_FILE")); refusing to start a second on one GPU"
  exit 1
fi
echo $$ > "$QUEUE_PID_FILE"
trap 'rm -f "$QUEUE_PID_FILE"' EXIT
rm -f logs/r1/CHAIN_DONE

fail=0
for spec in "$@"; do
  IFS=':' read -r tag cfg arms <<< "$spec"
  logdir="logs/r1/$tag"
  mkdir -p "$logdir"
  rm -f "$logdir/PASS_DONE"
  n=$($PY run_r1.py --config "$cfg" --count) || { echo "[$tag] bad config $cfg"; fail=1; continue; }
  echo "[$(date -u +%FT%TZ)] pass $tag: $cfg arms=${arms:-<config>} units=$n" | tee -a logs/r1/queue.log
  for ((i=0; i<n; i++)); do
    if [ -n "${arms:-}" ]; then
      $PY run_r1.py --config "$cfg" --unit "$i" --arms "$arms" >> "$logdir/unit_${i}.log" 2>&1
    else
      $PY run_r1.py --config "$cfg" --unit "$i" >> "$logdir/unit_${i}.log" 2>&1
    fi
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$(date -u +%FT%TZ)] [$tag] unit $i FAILED rc=$rc (see $logdir/unit_${i}.log)" | tee -a logs/r1/queue.log
      fail=1          # keep going: one bad unit must not strand the pass
    fi
  done
  if [ -n "${arms:-}" ]; then
    $PY run_r1.py --config "$cfg" --progress --arms "$arms" | tee -a logs/r1/queue.log
  else
    $PY run_r1.py --config "$cfg" --progress | tee -a logs/r1/queue.log
  fi
  date -u +%FT%TZ > "$logdir/PASS_DONE"
  echo "[$(date -u +%FT%TZ)] pass $tag done" | tee -a logs/r1/queue.log
done
echo "fail=$fail $(date -u +%FT%TZ)" > logs/r1/CHAIN_DONE
exit $fail
