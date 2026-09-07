#!/usr/bin/env bash
# One-command environment setup. Run this once on a new box.
#
#   bash setup.sh              # both stacks (GPU run + analysis)
#   bash setup.sh --analysis   # analysis only: no torch, no GPU needed
#
# The pins are part of the experiment, not a preference: a cross-panel MMLU
# divergence in this project was traced to a `transformers` rebuild, so a panel
# trained on a different stack is not comparable to the published ones.
set -euo pipefail
cd "$(dirname "$0")"

ANALYSIS_ONLY=0
[ "${1:-}" = "--analysis" ] && ANALYSIS_ONLY=1

PY="${PY:-python3}"
echo "=== python: $($PY --version) ==="

echo
echo "=== installing analysis stack ==="
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r requirements-analysis.txt
echo "  done"

if [ "$ANALYSIS_ONLY" = "0" ]; then
  echo
  echo "=== installing run stack (pinned) ==="
  # torch first, from the cu128 index, so the pinned build is the one that sticks.
  if ! $PY -c "import torch" 2>/dev/null; then
    $PY -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  else
    echo "  torch already present: $($PY -c 'import torch;print(torch.__version__)')"
  fi
  $PY -m pip install -r requirements-run.txt
  echo "  done"
fi

echo
echo "=== preflight ==="
if [ "$ANALYSIS_ONLY" = "1" ]; then
  $PY scripts/preflight.py --no-gpu --no-cache
else
  $PY scripts/preflight.py
fi

cat <<'NEXT'

--------------------------------------------------------------------
Setup finished. Next:

  bash run.sh            show what can be run and what is already done
  bash run.sh all        run every panel that fits this machine

Gated checkpoints (gemma, llama) need a Hugging Face token:
  export HF_TOKEN=hf_...
Models are ~72 GB in total. If you have a shared HF cache, point at it:
  export HF_HOME=/path/to/hf-cache
--------------------------------------------------------------------
NEXT
