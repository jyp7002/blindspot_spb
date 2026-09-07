#!/usr/bin/env bash
#SBATCH --job-name=blindspot-v11
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# One array task = one work unit.
#
#   python3 scripts/plan.py configs/v11/dec_v11.yaml
#   sbatch --array=0-53 scripts/submit_slurm.sh work/v11dec.units.json
#
# Resubmitting the SAME array after a partial run is safe and is the intended
# recovery path: run_unit.py skips units the artifacts already show complete,
# so finished tasks exit in seconds. Re-run plan.py first if you would rather
# shrink the array than skip inside it.
#
# TUNE BEFORE THE BIG TIER. The 32B/70B cells need a bigger card and more time
# than the defaults above; override at submit time rather than editing here:
#   sbatch --array=0-1 --gres=gpu:a100_80gb:1 --time=24:00:00 \
#          scripts/submit_slurm.sh work/v11big.units.json
set -uo pipefail

UNITS="${1:?usage: sbatch --array=0-N submit_slurm.sh <units.json>}"
mkdir -p logs/slurm

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# The stack is the experiment. If the node's modules/venv differ from the pins,
# the panel is not comparable to the published ones -- so check, loudly, before
# spending the allocation. Set BS_SKIP_PREFLIGHT=1 only for a deliberate rerun.
if [ "${BS_SKIP_PREFLIGHT:-0}" != "1" ]; then
  python3 scripts/preflight.py --no-cache --strict-versions || {
    echo "PREFLIGHT FAILED — refusing to run. Set BS_SKIP_PREFLIGHT=1 to override."
    exit 1
  }
fi

echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0} units=$UNITS"
exec python3 scripts/run_unit.py "$UNITS" --index "${SLURM_ARRAY_TASK_ID:-0}"
