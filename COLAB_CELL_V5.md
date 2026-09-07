# v5 GPU runs — V1 (size-matched same-family), V2 (seed fill), X (3B bridge)

Re-upload the NEW `colab_t2t4.py`. All three runs are **resumable and saved to
Drive** — re-run the cell after any timeout and it continues. They reuse the v4
t2x corpora already on your Drive, so only new designer models get elicited.

```python
# ── v5 GPU runs — resumable, Drive-persisted ────────────────────────────────
from google.colab import drive; drive.mount('/content/drive')
import os
os.environ["BS_OUT"] = "/content/drive/MyDrive/blindspot_results"   # SAME path as v4 t2x
os.makedirs(os.environ["BS_OUT"], exist_ok=True)

!pip -q install "transformers>=4.44" accelerate datasets bitsandbytes peft numpy huggingface_hub
os.environ["HF_TOKEN"] = "hf_XXXXXXXXXXXXXXXXXXXX"    # <-- paste HF token
from huggingface_hub import login; login(os.environ["HF_TOKEN"])

from google.colab import files
print("Upload the NEW colab_t2t4.py:")
up = files.upload(); assert "colab_t2t4.py" in up

import subprocess, sys
def run(args, logname):
    logp = os.path.join(os.environ["BS_OUT"], logname)
    with open(logp, "a") as lg:
        p = subprocess.Popen([sys.executable, "colab_t2t4.py"] + args,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end=""); lg.write(line); lg.flush()
        p.wait()

# ---- V1: size-matched same-family-LARGE designers (qwen/olmo/falcon 7B targets) ----
run(["--do", "v1", "--t2x-axes", "occ_gender", "crows_socioeconomic",
     "--t2x-seeds", "0"], "v1_run.log")

# ---- V2: seed fill (1->3) on the headline v4 t2x panel ----
run(["--do", "t2cross", "--t2x-axes", "occ_gender", "crows_socioeconomic",
     "--t2x-seeds", "0", "1", "2",
     "--t2x-targets", "qwen", "llama", "gemma", "olmo"], "v2_run.log")

# ---- X: 3B bridge tier (qwen/llama/falcon 3B targets) ----
run(["--do", "x", "--t2x-axes", "occ_gender", "crows_socioeconomic",
     "--t2x-seeds", "0", "1", "2"], "x_run.log")

# zip everything done so far (safe mid-progress)
zp = "/content/v5_results.zip"
!cd "$BS_OUT" && zip -qr "$zp" v1 x t2x *.log 2>/dev/null; echo zipped
files.download(zp)
```

## Run order / knobs
- **V1 is the abstract gate — run it first.** If you want a result fast, comment
  out V2 and X, run V1 alone (~1-2 h: 3 targets × 7 designers × 2 axes, minus the
  many corpora already cached).
- V2 (seed fill) and X (3B bridge) can be separate sessions; each is resumable.
- Every run reuses `t2x/corpora/` on your Drive, so V1's cross designers and V2's
  designers are already elicited — only Qwen-14B / OLMo-13B / Falcon-10B /
  Falcon-7B (V1) and the 3B models (X) get elicited fresh.

## What comes back (`v5_results.zip`)
- `v1/removal.jsonl` — self / same_large / cross removal per (qwen/olmo/falcon, axis, seed)
- `x/removal.jsonl` — 3B-tier self / cross removal
- `t2x/removal.jsonl` — now with seeds 0,1,2 for the 4 headline targets (V2)
- `*_run.log` — full logs (paste any traceback if a cell aborts)

Back here I run `src/v1_analyze.py` (same_large vs cross → Abstract A/B decision),
add V2's seeds to the CIs, and `src/x_analyze.py` (the 3B point on the arc).
Send `v5_results.zip` — partial is fine, the rest resume-fills.
