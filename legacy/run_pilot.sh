#!/usr/bin/env bash
set -e
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "select_acquired.py|verify_axes.py" >/dev/null; do sleep 15; done
python pilot_elicit.py
echo "### PILOT DONE"
