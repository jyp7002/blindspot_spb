#!/usr/bin/env bash
# AX — axis breadth port, SERIAL, chained behind the running blt2 queue.
#   1 onboarding gates (G3/G4 polarity + magnitude, per target)
#   2 removal run on axes that PASS the gates only
set -u
cd "$(dirname "$0")"
export BS_OUT=results/v6trace HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=results/v6trace/ax_queue.log
say(){ echo "[ax $(date +%m-%d\ %H:%M:%S)] $*" | tee -a $LOG; }
say "waiting for blt2 queue (serial: 31 GB host RAM)"
while pgrep -f 'run_blt2_queue.sh' >/dev/null || pgrep -f 'run_ablate_attn.py' >/dev/null \
   || pgrep -f 'run_sentdebias_baseline.py' >/dev/null; do sleep 60; done
say "stage 1: onboarding gates"
/opt/conda/bin/python3 run_ax_onboard.py >> $LOG 2>&1
USABLE=$(/opt/conda/bin/python3 -c "
import json;d=json.load(open('results/v6trace/ax_onboard.json'));print(' '.join(d.get('usable_axes',[])))" 2>/dev/null)
say "gates say usable: [${USABLE:-none}]"
if [ -z "$USABLE" ]; then
  say "NO axis passed the gates — AX1 fails, §7 limitation stands as written. Not running removal."
  exit 0
fi
say "stage 2: removal on gated axes"
cnt(){ [ -f results/v6trace/ax/removal.jsonl ] && wc -l < results/v6trace/ax/removal.jsonl || echo 0; }
for t in 1 2 3 4 5 6; do
  c=$(cnt); say "  attempt $t at $c rows"
  AX_AXES="$USABLE" /opt/conda/bin/python3 run_ax_removal.py >> $LOG 2>&1
  a=$(cnt); say "  -> $c -> $a"
  grep -aq "\[ax\] ALL DONE" $LOG && { say "  AX complete at $a"; break; }
  [ "$a" -le "$c" ] && { say "  no progress — stopping"; break; }
done
say "done"
