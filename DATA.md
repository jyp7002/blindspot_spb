# DATA.md — where the data is, and why it is not in this repository

This is a **code** repository. The measurement data is not in it.

| | size | in git? | needed for |
|---|---:|---|---|
| code (`src/`, `scripts/`, `configs/`, runners, `Makefile`) | ~1.6 MB | **yes** | everything |
| `audit/manifest_v10.yaml` + prose edits | 0.4 MB | **yes** | the number audit's specification |
| `figures/fig*_data.csv` | 0.1 MB | **yes** | the audit contract is on these exact bytes |
| `results/` + `results_v9/` panels | ~43 MB | no | `make freeze` |
| `results/t2x/corpora/` elicited corpora | ~11 MB | no | running panels (re-elicited if absent) |
| `figures/*.pdf`, `*.png` | ~1 MB | no | build output of `make figures` |

Data was 93% of this tree by size and is not source. It ships the way this
project has always shipped results — as an archive — rather than through git.

## To run experiments: you need nothing extra

```bash
bash setup.sh && bash run.sh all
```

`scripts/run_unit.py` elicits any missing designer corpus itself (inference
only, cached to `results/t2x/corpora/`), so a clean clone on a GPU box can start
a panel with no data transfer. Panels are written under `results/<panel>/`.

**One caveat.** A re-elicited corpus is only identical to the published one if
the model, seed and stack match. The pins in `requirements-run.txt` exist for
exactly this. If you have access to the original corpora, restoring them (below)
removes the question entirely.

## To run the audit: restore the panels first

`make freeze` reads `results_v9/` and `results/`. Without them it fails at the
first regen module — that is a missing input, not a bug.

```bash
tar xzf blindspot_panels_<stamp>.tar.gz -C .     # into results/ and results_v9/
make freeze                                       # must print AUDIT PASS
```

## To send results back from a run box

```bash
bash scripts/pack_artifacts.sh v11dec
```

Ships `removal.jsonl` / `alpha_trace.jsonl` / `not_applicable.jsonl` only —
about 40 KB per panel. It deliberately excludes `supports/*.npz` and the
trained direction tensors (~9.5 GB), which are regenerable and which no
published number reads directly.

## Verifying a run against the published panels

```bash
python3 scripts/check_reproduction.py
```

Compare against `results_v9/`, **not** `results/`: the latter is the as-run tree
scored under the old float collateral gate, and a correct run legitimately
differs from it on every row carrying `v9_changed=true`.
