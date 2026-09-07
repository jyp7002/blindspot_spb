#!/usr/bin/env bash
# v8 execution queue. ONE GPU, so everything is serial -- running two of these
# concurrently would contend for the L40S and risk the OOM that killed this
# pipeline 9 times at the 7-9B tier.
#
# Order is by information value: DEC is the adjudicator and runs first, so that
# if the queue is interrupted the paper still has its critical experiment.
# Every runner is resumable on its own keys, so re-running this script continues
# rather than restarts.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=.:src
LOG=results/v8_queue.log
stamp() { date +"%m-%d %H:%M:%S"; }
say() { echo "[v8q $(stamp)] $*" | tee -a "$LOG"; }

say "queue start"

# --- DEC: the adjudication experiment (~10h) ---------------------------------
say "DEC small tier start"
python3 run_dec.py gemma >> results/v8dec.log 2>&1; say "DEC gemma rc=$?"
python3 run_dec.py llama >> results/v8dec.log 2>&1; say "DEC llama rc=$?"
python3 run_dec.py qwen  >> results/v8dec.log 2>&1; say "DEC qwen  rc=$?"
python3 run_dec.py phi   >> results/v8dec.log 2>&1; say "DEC phi   rc=$?"
say "DEC 7-9B tier start"
python3 run_dec.py qwen7b >> results/v8dec.log 2>&1; say "DEC qwen7b rc=$?"
python3 src/v8_dec_analyze.py v8dec >> results/v8dec_analysis.log 2>&1
say "DEC analysis rc=$?"

# --- SUP1 designer-overlap arm (supports only, no alpha sweep) ---------------
# DEC is self-designer only, so SUP1 sec5 has no data without this. Must run
# AFTER DEC so the self dump (qwen7b|occ_gender|s*|qwen) already exists.
python3 run_sup1_designers.py >> results/v8sup1.log 2>&1; say "SUP1 designers rc=$?"

# --- SUP1: analysis-only, needs DEC's support dumps --------------------------
python3 src/v8_sup1.py results/v8dec >> results/v8sup1.log 2>&1
say "SUP1 rc=$?"

# --- SPC: sparsity curve (~5h) -----------------------------------------------
python3 run_spc.py small >> results/v8spc.log 2>&1; say "SPC small rc=$?"
python3 run_spc.py big   >> results/v8spc.log 2>&1; say "SPC big   rc=$?"

# --- INS: re-scoped instruct arms (~4h) --------------------------------------
# gemma_3b|bbq_Age was never elicited (only llama_3b/phi_3b/qwen_3b have BBQ),
# so fill it with the frozen elicitation path before INS-B needs it.
python3 run_v8_elicit.py >> results/v8ins.log 2>&1; say "INS elicit rc=$?"
python3 run_ins.py B >> results/v8ins.log 2>&1; say "INS-B (base vs instruct) rc=$?"
python3 run_ins.py A >> results/v8ins.log 2>&1; say "INS-A (IFEval collateral) rc=$?"

say "queue done"
