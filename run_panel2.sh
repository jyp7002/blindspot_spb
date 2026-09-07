#!/usr/bin/env bash
# Complete the cross-family panel: the remaining three targets (gemma, smol,
# llama) each get every non-target family as a cross designer. Only the
# not-yet-run designers are launched; the existing cross/cross2 records supply
# the rest. Templated axes, q/v, 3 seeds, to pool with the existing data.
#
# gemma runs at batch 12: Gemma-2's 256k vocabulary makes a batch-64 logits
# tensor ~19 GB.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_experiment.py|edit_capacity.py|geometry.py" >/dev/null; do sleep 30; done
sleep 10

echo "### PANEL2 gemma (new cross: qwen3b, phi3.5, smol1.7b)"
python run_experiment.py --arm gemma --seeds 0 1 2 \
  --origins inherited acquired --designers cross2 cross3 cross4 \
  --alphas 2 4 8 16 --steps 250 --batch-size 12 --variant-set a \
  || echo "PANEL2 gemma FAILED"
while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done

echo "### PANEL2 smol (new cross: qwen3b, gemma2b, phi3.5)"
python run_experiment.py --arm smol --seeds 0 1 2 \
  --origins inherited acquired --designers cross2 cross3 cross4 \
  --alphas 2 4 8 16 --steps 250 --batch-size 32 --variant-set a \
  || echo "PANEL2 smol FAILED"
while pgrep -f "run_experiment.py" >/dev/null; do sleep 15; done

echo "### PANEL2 llama (new cross: phi3.5, smol1.7b)"
python run_experiment.py --arm llama --seeds 0 1 2 \
  --origins inherited acquired --designers cross3 cross4 \
  --alphas 2 4 8 16 --steps 250 --batch-size 32 --variant-set a \
  || echo "PANEL2 llama FAILED"

echo "### PANEL2 DONE"
