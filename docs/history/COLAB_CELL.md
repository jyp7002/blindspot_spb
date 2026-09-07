# Single Colab cell — T2 (H1 at 7-9B) + T4 (27-72B profiles + designer)

Runtime: a 96GB GPU (H100/A100-80/etc.). Runtime → Change runtime type → GPU.

**Paste this one cell.** It uploads `colab_t2t4.py` (the file I sent), installs
deps, runs both T2 and T4, and downloads a `bs_results.zip` at the end.

```python
# ── Blind-spot T2+T4 — single cell ─────────────────────────────────────────
!pip -q install "transformers>=4.44" accelerate datasets bitsandbytes peft numpy

from google.colab import files
print("Upload colab_t2t4.py:")
up = files.upload()                      # pick colab_t2t4.py
assert "colab_t2t4.py" in up, "upload colab_t2t4.py"

import subprocess, sys
# runs T4 (27-72B: 27/32B bf16, 70/72B 8-bit) then T2 (5 families, bf16)
# edit --do to run only one:  --do t2   |   --do t4
subprocess.run([sys.executable, "colab_t2t4.py", "--do", "t4", "t2"], check=False)

# zip + download everything (profiles npy, designer json, t2_selfremoval.json)
!cd results && zip -qr ../bs_results.zip . && cd ..
files.download("bs_results.zip")
```

## What it produces (bring back via bs_results.zip)
- `results/t4/profiles/<name>|<axis>.npy`     — O1 convergence (all 4 T4 models × 10 axes)
- `results/t4/designer/<name>|<axis>.json`    — R2 elicited corpora (4 designers × 4 axes)
- `results/t2_selfremoval.json`               — H1 at T2 (5 families: self + sibling removal)

Just send me `bs_results.zip`; I merge it and run the O1 / R2 / H1-at-T2 analyses.

## Knobs (edit the subprocess line)
- Only T2 or only T4:  `--do t2`  or  `--do t4`
- If a 70B/72B OOMs even at 8-bit on your GPU, they're the last two in the T4
  list — comment them out inside colab_t2t4.py (`T4 = [...]`) or just let them
  fail (the others still save; each model is independent and resumable).
- Batch sizes: `--t4-batch N` / `--t2-batch N` (defaults 8 / 12; gemma auto-lowered).

## Notes
- Resumable: re-running skips already-saved profiles/designer files.
- Fidelity: 27B/32B profiles are bf16 (directly comparable to the ≤9B tiers);
  70B/72B are 8-bit (near-bf16; the convergence comparison is relative so this
  is fine — noted as a caveat).
- Gated models (Llama): run `from huggingface_hub import login; login("hf_...")`
  in a cell first.
