#!/usr/bin/env bash
# ============================================================================
# experiments_v2 — EXPERIMENT E (safety net + systems corollary)
#   + D3 (second injected axis) and D4 (elicitation robustness)
# ============================================================================
# Runs after Experiment B releases the GPU.
#
# STAGE 0 is a calibration pass on ONE condition. Every E hyper-parameter
# (STE lr/lambda, PCGU lr/top-frac, DPO beta, steering alpha scale) is
# currently a guess, and the diagnostics that reveal a dead method are
# edit_sign_flip_frac (~0 => STE did nothing), gap_initial->gap_final
# (objective never moved) and edit_norm_ratio (steering alpha off-scale).
# Running the full 576-record sweep on uncalibrated defaults would burn hours
# producing flat lines.
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/jovyan/Blind_spot_spb/src

while pgrep -f "run_experiment.py|geometry.py" >/dev/null; do sleep 30; done
sleep 15

echo "### E STAGE 0: calibration (single condition, diagnostics only)"
python run_expE.py --arms qwen --seeds 0 --origins inherited \
  --designers self --alphas 2 4 --batch-size 32 --verbose \
  || echo "E CALIB FAILED"
# Calibration writes into the same resumable JSONL. If the diagnostics below
# show a dead method (sign_flip_frac ~ 0, gap unchanged, norm_ratio off-scale)
# those 16 records must be PURGED before stage 1, or the bad-hyper-parameter
# rows will be skipped as "already done" and silently kept.
python - <<'PYEOF'
import json, os
p = "/home/jovyan/Blind_spot_spb/results/runs_expE.jsonl"
if os.path.exists(p):
    rs = [json.loads(l) for l in open(p) if l.strip()]
    print("CALIB DIAGNOSTICS")
    for r in rs:
        e = r.get("edit_meta", {}) or {}
        print(f"  {r.get('method','?'):18s} a={r.get('alpha')} "
              f"dBias={r.get('bias_reduction', float('nan')):+.4f} "
              f"flip={e.get('sign_flip_frac')} gap0={e.get('gap_initial')} "
              f"gap1={e.get('gap_final')} norm={e.get('norm_ratio')}")
PYEOF

echo "### E STAGE 1: full baseline suite at matched collateral (E1+E2)"
python run_expE.py --arms qwen --seeds 0 1 2 --origins inherited acquired \
  --designers self cross random --alphas 2 4 8 16 --batch-size 32 \
  || echo "E SUITE FAILED"

echo "### E STAGE 2: bit-budget frontier across arms (E3)"
python run_expE.py --bit-budget --arms qwen llama --seeds 0 1 2 \
  --origins inherited acquired --designers self cross gt --alphas 2 4 8 16 \
  --batch-size 32 || echo "E BITBUDGET FAILED"

echo "### D4: elicitation robustness across model sizes"
python pilot_forced.py > /home/jovyan/Blind_spot_spb/logs/d4_elicit.log 2>&1 \
  || echo "D4 FAILED"

echo "### D3: second injected axis (ses_lw), matched strength"
python inject_matched.py --target qwen1.5b --axis ses_lw --lrs 1e-3 5e-4 2e-4 \
  || echo "D3 INJECT FAILED"

echo "### EXPE DONE"
