#!/usr/bin/env bash
# Re-inject gemma (over-injected at +2.41) and phi (failed: fused qkv_proj) to
# a strength matched to the Qwen reference arm, then run their arms.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src
while pgrep -f "run_experiment.py --arm smol" >/dev/null; do sleep 30; done
echo "### MATCHED INJECT gemma2b"
python inject_matched.py --target gemma2b --lrs 2e-4 1e-4 5e-5 || echo "GEMMA INJECT FAILED"
echo "### MATCHED INJECT phi3.5"
python inject_matched.py --target phi3.5 --lrs 1e-3 5e-4 2e-4 || echo "PHI INJECT FAILED"
echo "### EXPA FIX DONE"
