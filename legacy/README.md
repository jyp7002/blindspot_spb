# legacy/ — superseded runners, kept for provenance

85 scripts from v1–v9. **You do not need anything in here to run the current
experiments.** Use `scripts/` and `configs/v11/` instead (see the root README).

They are kept, not deleted, because this project's discipline is that faulty and
superseded artifacts stay on disk, labelled, rather than disappearing — several
published numbers were produced by these scripts and a reviewer must be able to
see the code that made them.

**They will not run from here as-is.** Each expects to be launched from the repo
root (`python3 run_dec.py`, not `python3 legacy/run_dec.py`), and the queue
scripts below call runners that are still at the root. If you need to reproduce
an old panel, run it from the root with the path adjusted.

| still at the repo root (needed now) | why |
|---|---|
| `colab_t2t4.py` | the core library: elicitation, training, edits, collateral eval |
| `run_dec.py` | the decomposition panel (v11 `dec` kind) |
| `run_alpha_ext.py` | the α-extension panel (v11 `alphaext` kind) |
| `run_ins.py` | the IFEval / instruct panel (v11 `ifeval` kind) |
| `run_ablate_attn.py`, `run_dpo_baseline.py`, `run_ws2_determinism.py` | imported by `src/` analysis modules |

Superseded queue scripts (`run_v8_queue*.sh`, `run_*_queue.sh`) are replaced by
`scripts/plan.py` + `scripts/submit_local.sh` / `scripts/submit_slurm.sh`, which
are resumable and environment-neutral.
