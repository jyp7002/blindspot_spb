#!/usr/bin/env bash
# R1 on the 27-32B cells (E4b + E1b). One >= 80 GB card. Resumable: re-run it.
#
#   export HF_TOKEN=hf_...          # gemma-2-27b-it is gated
#   bash setup.sh                   # once per box: the pinned stack
#   bash scripts/r1_big.sh
#   bash scripts/r1_pack.sh r1/BIG  # then send the archive back
#
# Optional but recommended before the first unit: put the v12big panel where the
# replay guards can find it (results/v12big or ./v12big; three small jsonl files).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
CFG="${CFG:-configs/r1/big.yaml}"

echo "=== R1 BIG preflight ==="
$PY - <<'PY' || exit 1
import importlib, os, sys
bad = 0
for mod, want in (("torch", "2.11.0"), ("transformers", "4.57.1"), ("peft", "0.20.0")):
    try:
        got = importlib.import_module(mod).__version__.split("+")[0]
    except Exception as e:
        got = f"MISSING ({type(e).__name__})"
    ok = got == want
    bad += not ok
    print(f"[{' ok ' if ok else 'FAIL'}] {mod} {got}" + ("" if ok else f"  (pinned: {want}; run bash setup.sh)"))
if bad:
    sys.exit(1)
import torch
if not torch.cuda.is_available():
    print("[FAIL] no CUDA device"); sys.exit(1)
gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
ok = gib >= 78
print(f"[{' ok ' if ok else 'FAIL'}] gpu0 {torch.cuda.get_device_name(0)}, {gib:.0f} GiB"
      + ("" if ok else "  (qwen2.5-32B needs ~61 GiB of weights + ~16 GiB; one >= 80 GB card)"))
if not ok:
    sys.exit(1)
hub = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
for rid in ("google/gemma-2-27b-it", "Qwen/Qwen2.5-32B-Instruct"):
    have = os.path.isdir(os.path.join(hub, "models--" + rid.replace("/", "--")))
    print(f"[{' ok ' if have else 'warn'}] {rid} " + ("cached" if have else "not cached: will download"))
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") \
    or os.path.exists(os.path.join(os.path.dirname(hub), "token"))
print(f"[{' ok ' if tok else 'warn'}] HF token " + ("present" if tok else "not set (gemma-2-27b-it is gated)"))
ref = [d for d in ("results/v12big", "v12big") if os.path.exists(os.path.join(d, "geometry.jsonl"))]
print(f"[{' ok ' if ref else 'warn'}] reference panel v12big "
      + (f"found at {ref[0]}" if ref else "NOT on this box: replay guards will record None"))
PY
$PY src/r1_select.py | tail -n 1 || exit 1
$PY run_r1.py --config "$CFG" --progress || exit 1

echo
echo "=== launching (logs: logs/r1/big/unit_*.log, logs/r1/queue.log) ==="
bash scripts/r1_queue.sh "big:$CFG"
rc=$?
echo "queue exit code $rc"
$PY run_r1.py --config "$CFG" --progress
echo "next:  bash scripts/r1_pack.sh r1/BIG"
exit $rc
