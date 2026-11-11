#!/usr/bin/env bash
# Start another R1 queue as soon as the live one has finished.
#
#   bash scripts/r1_after.sh "e13:configs/r1/e13.yaml"
#
# The arguments are passed to scripts/r1_queue.sh unchanged. This exists because
# r1_queue.sh must not be edited or restarted while it runs, and because two
# workers on one card are not allowed: it waits for the pid in logs/.queue.pid
# to exit and only then starts the next queue, which takes the same lock.
set -uo pipefail
cd "$(dirname "$0")/.."
PF="logs/.queue.pid"
mkdir -p logs/r1
echo "[$(date -u +%FT%TZ)] r1_after: waiting for the live queue, then: $*" >> logs/r1/queue.log
while [ -f "$PF" ] && kill -0 "$(cat "$PF" 2>/dev/null)" 2>/dev/null; do
  sleep 60
done
echo "[$(date -u +%FT%TZ)] r1_after: queue is free, starting: $*" >> logs/r1/queue.log
exec bash scripts/r1_queue.sh "$@"
