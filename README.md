# One Bit in a Hundred — sparse signed substructure of behavioral weight edits

Research code for the ICLR submission *One Bit in a Hundred: The Sparse Signed
Substructure of Behavioral Weight Edits*, plus the machine-checked number audit
that gates it.

> **Private repository — the submission is anonymous.** The manuscript is under
> double-blind review (`\iclrfinalcopy` is commented out). Do not make this repo
> public before camera-ready; the account and commit metadata would deanonymize
> the submission. The anonymized supplementary bundle is built separately.

---

## What this is

Two things live here, and they have **different dependencies on purpose**:

| | what it does | needs | where it runs |
|---|---|---|---|
| **analysis** | regenerates every artifact, figure and number the paper cites, then audits them | numpy/scipy/matplotlib/yaml — **no torch** | any box, incl. a laptop |
| **run** | trains contrast vectors, applies edits, measures collateral | pinned torch/transformers/peft + GPU | a GPU node |

The split is not cosmetic. The analysis box has repeatedly lost its GPU stack
(recorded twice in `PREREGISTRATION.md`), and `make freeze` must keep working
regardless.

### The central discipline

**No number in the manuscript is hand-typed.** Every figure resolves to an
artifact through `audit/manifest_v10.yaml`, enforced by `src/v10_audit.py`.

```bash
pip install -r requirements-analysis.txt
make freeze      # regen -> manifest -> figures -> audit, from a clean tree
```

`make freeze` must print `AUDIT PASS`. If it does not, the manuscript has a
number that no artifact produces, and that is a defect in the paper — not in
the gate. Do not weaken the gate to get a green.

---

## Scale-up (v11)

`experiments_v11.md` is the frozen design; `PREREGISTRATION.md` §v11 is the
amendment log. Four axes, each attacking a sentence in the paper's own
Limitations section:

| config | scales | from → to |
|---|---|---|
| `configs/v11/dec_v11.yaml` | decomposition cells | 10 cells / 3 families → 18 cells / **4 families** |
| `configs/v11/alphaext_v11.yaml` | α grid | probed to 128 (unsaturated) → 512, to saturation |
| `configs/v11/ifeval_v11.yaml` | generation collateral | 2 targets × 1 axis → 3 × 2 |
| `configs/v11/big_v11.yaml` | model size | 2.6–8B → +32B (70B stretch) |

**MMLU stays at 200 items, deliberately** — raising it changes the collateral
gate's integer grid, so nothing measured at 1,000 items is comparable to
anything published at 200. See `experiments_v11.md` §II and `PREREGISTRATION.md`
§v11.A.

### Running a panel

Environment-neutral by construction: `plan.py` writes one unit list, and the
local and SLURM launchers both index into it, so a worker and an array task
agree on what unit 37 is without coordinating.

```bash
# 0. is this box safe to launch on?  (versions, GPUs, VRAM vs the biggest cell,
#    HF cache, disk, and all four self-tests)
python3 scripts/preflight.py --config configs/v11/dec_v11.yaml

# 1. expand the config into resumable work units
python3 scripts/plan.py configs/v11/dec_v11.yaml
#    -> work/v11dec.units.json

# 2a. local: one worker per GPU
WORKERS=4 bash scripts/submit_local.sh work/v11dec.units.json

# 2b. SLURM: one array task per unit
sbatch --array=0-53 scripts/submit_slurm.sh work/v11dec.units.json

# 3. ship the analysis-grade artifacts back (40 KB, not 319 MB)
bash scripts/pack_artifacts.sh v11dec
```

**A work unit** is one `(panel, target, axis, seed, designer)` — the level at
which a checkpoint is loaded and a contrast vector is trained. Splitting finer
would retrain per condition; splitting coarser would make a node that died at
80% redo finished work.

**Resume is read from artifacts, never a ledger.** Re-run `plan.py` and it
emits only what is not on disk; re-run a whole array and finished tasks exit in
seconds. Verified against the published `v8dec` panel: it identifies exactly the
10 finished cells and leaves the 8 new ones pending.

### Before launching, read these two

1. **`experiments_v11.md` §IV has one open decision** — whether the two weak
   CrowS cells join the panel. It must be settled *before* the run, because
   choosing after seeing Δ_selection is outcome-selection.
2. **`configs/v11/big_v11.yaml` header** — 8-bit + LoRA training is untested
   here, and sharded training does not exist (trainable targets load with
   `dispatch=False`). Land `qwen32b` on one ≥80 GiB card before renting a 70B
   node.

---

## Layout

```
binary_debiaser_draft_v3.md     the draft. NEVER edited by hand or by tooling.
binary_debiaser_draft_v3.1.md   = v3 + registered prose edits + resolved numbers
PREREGISTRATION.md              every design frozen before its run + amendments
experiments_v{8,9,10,11}.md     per-version designs
Makefile                        the freeze gate, and the v11 helpers

src/v9_gate.py                  THE collateral gate. One implementation.
src/v10_audit.py                the number-source audit
src/v11_panel.py                config -> work units -> resume
src/v11_selftest.py             proves the v11 gate change is inert at n=200
configs/v11/*.yaml              the scale-up matrix
scripts/                        preflight, plan, run_unit, submit_*, pack
env/Dockerfile                  the pinned stack, as a container

results/, results_v9/           panels (analysis-grade artifacts are committed)
figures/                        fig{1..6} + the CSV each one is audited against
audit/                          manifest + registered prose edits
```

**What is not in git:** trained direction tensors, support dumps and checkpoints
(~9.5 GB). They are regenerable from the committed corpora and configs, and no
published number reads them directly. The ~53 MB of `jsonl`/`json` panels *are*
committed, because `make freeze` must be reproducible from a clone.

---

## Gotchas that have cost real time

- `results_v9/v8dec_smoke/removal.jsonl` carries `panel=='v8dec'`. Never glob
  `v8dec*`.
- `target` is **not** unique: `llama`/`qwen` name 3B models in small-tier panels
  and 8B/7B in big-tier ones.
- The α-extension's frozen arm reproduces only in **panel insertion order** —
  sorting the cell list shifts the CI by ~0.0008.
- Patch sizes are **MiB**, and were printed as "MB" in earlier drafts.
- `alpha_trace.jsonl` is append-only and a resumed cell is re-traced, so
  `(target, axis, seed, role, designer, alpha)` is **not** unique. Dedupe
  keeping the *last* occurrence.
- The pinned stack is part of the experiment: `run_ws2_determinism.py`
  attributes a cross-panel MMLU divergence to a `transformers` rebuild.
