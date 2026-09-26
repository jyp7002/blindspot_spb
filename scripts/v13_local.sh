#!/usr/bin/env bash
# v13 on the <=9B box, in the registered order (PREREGISTRATION §v13.B / §5 of
# experiments_v13.md). Resumable: re-run to continue; finished units are skipped.
#
#   1. E0.5 replay (2 v12astar units) -- the gate. A FAIL stops everything.
#   2. E0.5 determinism diagnostic (4 x 6 frozen-grid units) -- no verdict.
#   3. FLOOR <=9B (33 units), then SEED <=9B (54 units).
#
# The 27-32B panels run on the >=80 GB card: bash run.sh v13-seed-big / v13-floor-big.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
stamp() { date -u +%FT%TZ; }

echo "=== $(stamp) E0.5 replay"
bash run.sh v13-replay || exit 1
if ! $PY src/v13_analyze.py replay; then
  echo "=== $(stamp) REPLAY FAILED -- stopping before any new v13 cell (§v13.B)."
  exit 1
fi

echo "=== $(stamp) E0.5 determinism diagnostic"
for k in v13-det-nd-a v13-det-nd-b v13-det-d-a v13-det-d-b; do
  bash run.sh "$k" || echo "=== $(stamp) $k returned non-zero (diagnostic; continuing)"
done
$PY src/v13_analyze.py det

for k in v13-floor v13-seed; do
  echo "=== $(stamp) START $k"
  bash run.sh "$k"
  echo "=== $(stamp) END $k rc=$?"
done
$PY src/v13_analyze.py progress
echo "=== $(stamp) V13 LOCAL DONE -- verdicts: python3 src/v13_analyze.py verdicts"
