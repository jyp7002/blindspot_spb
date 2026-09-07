#!/usr/bin/env bash
# Compact progress across every v11 panel. Safe to run while a panel is going.
#
#   bash scripts/progress.sh
set -uo pipefail
cd "$(dirname "$0")/.."
printf '%-9s %-9s %8s %8s %9s  %s\n' PANEL KIND UNITS DONE ROWS LAST
for cfg in configs/v11/*.yaml; do
  python3 - "$cfg" <<'PY'
import sys, os, json
sys.path.insert(0, "src")
import v11_panel as P
cfg = P.load(sys.argv[1]); units = P.expand(cfg)
done = sum(1 for u in units if P.done(u))
panel = cfg["out_panel"]
d = os.path.join(P.RESULTS, panel)
rows, last = 0, "-"
fp = os.path.join(d, "removal.jsonl")
if os.path.exists(fp):
    lines = [l for l in open(fp) if l.strip()]
    rows = len(lines)
    if lines:
        r = json.loads(lines[-1])
        last = f"{r.get('target')}|{r.get('axis')}|s{r.get('seed')} {r.get('condition')}"
print(f"{panel:<9} {cfg['kind']:<9} {len(units):>8} {done:>8} {rows:>9}  {last}")
PY
done
echo
echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo n/a)"
echo "disk free at results/: $(df -h . | tail -1 | awk '{print $4}')"
