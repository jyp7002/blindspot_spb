#!/usr/bin/env bash
# Pack one R1 panel for shipping between boxes.   bash scripts/r1_pack.sh r1/BIG
#
# Everything an R1 runner writes is small (jsonl rows and the per-item probe
# log-likelihoods); no tensors are stored. Two archives are written:
#   <panel>_<stamp>.tar.gz       the panel; unpack with  tar -xzf ... -C results
#   <panel>_<stamp>.logs.tar.gz  the unit logs of that pass, if any (for debugging)
set -euo pipefail
cd "$(dirname "$0")/.."
PANEL="${1:?usage: r1_pack.sh <panel, e.g. r1/BIG>}"
ROOT="${BS_OUT:-results}"
[ -d "$ROOT/$PANEL" ] || { echo "no such panel: $ROOT/$PANEL"; exit 1; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASE="$(echo "$PANEL" | tr '/' '_')_${STAMP}"
tar -czf "$BASE.tar.gz" --exclude='_trl_tmp' -C "$ROOT" "$PANEL"
echo "wrote $BASE.tar.gz ($(du -h "$BASE.tar.gz" | cut -f1))"
tar -tzf "$BASE.tar.gz" | sed -n '1,8p'
TAG="$(basename "$PANEL" | tr '[:upper:]' '[:lower:]')"
if [ -d "logs/r1/$TAG" ]; then
  tar -czf "$BASE.logs.tar.gz" "logs/r1/$TAG" $( [ -f logs/r1/queue.log ] && echo logs/r1/queue.log )
  echo "wrote $BASE.logs.tar.gz ($(du -h "$BASE.logs.tar.gz" | cut -f1))"
fi
