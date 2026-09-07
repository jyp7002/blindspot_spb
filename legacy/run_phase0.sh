#!/usr/bin/env bash
set -e
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
cd /home/jovyan/Blind_spot_spb/src
# wait for the inventory sweep to release the GPU
while pgrep -f "inventory.py --n-bbq" >/dev/null; do sleep 15; done
echo "### select_acquired"; python select_acquired.py
echo "### verify_axes";     python verify_axes.py
echo "### PHASE0 DONE"
