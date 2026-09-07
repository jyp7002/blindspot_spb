# T2 re-run — foolproof version (prints the answer INLINE)

**Why the last two runs came back empty:** they were the *original combined
cell* (it zips to `results_archive.zip` with no log). This is a different cell —
it runs **only T2**, starts with a fast smoke test, and **prints results/errors
directly in the Colab output** so we never need a zip round-trip to see what
happened. I verified the underlying code trains a real 7B model correctly here
(`qwen self removal +0.708`), so if this still fails it's environmental and the
inline traceback will show exactly what.

## Do these two things
1. **Re-upload the NEW `colab_t2t4.py` I just sent** (Colab caches the old one —
   delete it in the file browser first, or the upload will overwrite it).
2. Paste your HF token where marked (gemma-9b / llama-8b are gated).

Then paste this one cell. **If anything prints a traceback, copy the last ~40
lines of the Colab output straight into the chat** — that's all I need.

```python
# ── Blind-spot T2 — inline-diagnostic cell ─────────────────────────────────
!pip -q install "transformers>=4.44" accelerate datasets bitsandbytes peft numpy huggingface_hub

import os
os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXXXXXX"      # <-- paste HF token
from huggingface_hub import login; login(os.environ["HF_TOKEN"])

# upload the NEW colab_t2t4.py
from google.colab import files
print("Upload the NEW colab_t2t4.py:")
up = files.upload()
assert "colab_t2t4.py" in up, "upload colab_t2t4.py"

# ---- SMOKE TEST: one ungated family, one seed. ~5-10 min. Prints inline. ----
import importlib, traceback, sys
sys.argv = ["x"]                      # keep argparse in colab_t2t4 happy if imported
import colab_t2t4 as C
print("\n===== SMOKE TEST (qwen, seed 0) =====", flush=True)
try:
    mmlu = C.load_mmlu(n=64, seed=0); wt = C.load_wikitext(n_chunks=8, seed=0)
    rows = C.run_t2("qwen", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
                    mmlu, wt, seeds=(0,), batch_size=12, train_bs=6)
    print("SMOKE ROWS:", rows, flush=True)
    assert rows, "no rows produced"
    print("===== SMOKE OK — proceeding to full run =====\n", flush=True)
    smoke_ok = True
except Exception:
    print("===== SMOKE FAILED — full traceback below; PASTE THIS TO ME =====", flush=True)
    traceback.print_exc()
    smoke_ok = False

# ---- FULL RUN (all 5 families, 3 seeds) only if smoke passed. ~1-2 h. ----
if smoke_ok:
    import subprocess
    with open("t2_run.log", "w") as lg:
        p = subprocess.Popen([sys.executable, "colab_t2t4.py", "--do", "t2"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end=""); lg.write(line); lg.flush()
        p.wait()
    import json
    n = len(json.load(open("results/t2_selfremoval.json")))
    print(f"\n===== FULL RUN DONE: {n} rows =====", flush=True)
    !cp t2_run.log results/ && cd results && zip -qr ../t2_results.zip t2_selfremoval.json t2_run.log && cd ..
    files.download("t2_results.zip")
```

## What I need back
- **If SMOKE OK:** send `t2_results.zip` (5 families × {self,sibling} × 3 seeds).
- **If SMOKE FAILED:** just paste the printed traceback here — no file needed.
  That tells me the exact environmental cause and I'll fix it in one shot.

The smoke test alone (qwen point) already extends the H1-at-T2 evidence; the full
run gets all five families for the n=5 correlation.
