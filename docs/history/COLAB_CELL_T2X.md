# Cross-designer T2 — the definitive self-penalty test at 7-9B

This measures the **real self-penalty**: for each target family's 7-9B model, it
removes bias using SELF + SIBLING (same-family) **and every other family's 7-9B
model (cross-family)**, then computes `self_penalty = cross_removal − same_removal`
and correlates it with within-family sharing. It runs on **occ_gender** (matches
the small-scale ρ=−0.81) **and crows_socioeconomic** (naturalistic, dodges the
occ ceiling). Re-upload the NEW `colab_t2t4.py`.

## Important: this is a long run (~6-10 h) — so it saves to Google Drive and is RESUMABLE

If the Colab session times out, **just re-run the exact same cell** — it reads
what's already on Drive and continues from where it stopped (resumable at
target × axis × seed × designer). Nothing is recomputed.

```python
# ── Cross-designer T2 (self-penalty) — resumable, saves to Drive ────────────
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["BS_OUT"] = "/content/drive/MyDrive/blindspot_results"   # persists across disconnects
os.makedirs(os.environ["BS_OUT"], exist_ok=True)

!pip -q install "transformers>=4.44" accelerate datasets bitsandbytes peft numpy huggingface_hub
os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXXXXXX"    # <-- paste HF token (gemma/llama gated)
from huggingface_hub import login; login(os.environ["HF_TOKEN"])

from google.colab import files
print("Upload the NEW colab_t2t4.py:")
up = files.upload()
assert "colab_t2t4.py" in up, "upload colab_t2t4.py"

# run it (tee to a log we keep). Re-run this whole cell to resume after a timeout.
import subprocess, sys
logp = os.path.join(os.environ["BS_OUT"], "t2x_run.log")
with open(logp, "a") as lg:
    p = subprocess.Popen([sys.executable, "colab_t2t4.py", "--do", "t2cross"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end=""); lg.write(line); lg.flush()
    p.wait()

# zip whatever is done so far and download (safe to run even mid-progress)
zp = "/content/t2x_results.zip"
!cd "$BS_OUT" && zip -qr "$zp" t2x t2x_run.log && echo "zipped"
files.download(zp)
```

## Knobs (edit the subprocess line)
- **Faster first pass — occ only:** `"--do","t2cross","--t2x-axes","occ_gender"`
  (halves the run; do this if you want a result sooner, then add crows later —
  the crows rows will resume-fill on the next run).
- **One target at a time:** `"--t2x-targets","qwen"` (cross designers still come
  from all families' cached corpora — elicitation runs for all 5 regardless).
- **More seeds (tighter):** `"--t2x-seeds","0","1"` (doubles the run).
- Batch: `--t2x-batch N` (default 12; gemma auto-lowered to 6), `--t2x-train-bs N`.

## What comes back (send `t2x_results.zip`)
- `t2x/corpora/<designer>|<axis>|s<seed>.json` — each designer's elicited corpus
- `t2x/removal.jsonl` — one row per (target, axis, seed, role, designer) with
  in-budget `removal`. **This is the file the analysis reads.**
- `t2x_run.log` — full log.

Back here I run `python src/t2x_analyze.py`: per axis it prints same vs cross
removal per target, the mean self-penalty (cross−same), and corr(sharing,
self-penalty). Partial is fine — send whatever finished and I'll analyze it; the
rest resume-fills next run.
