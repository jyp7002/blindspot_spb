# experiments_v12.md — OPERATING-POINT SELECTION (drafted 2026-09-23, nothing run)

**Status:** registered as PREREGISTRATION.md §v12.A (2026-09-23, before any
run), D1–D6 frozen at the coded defaults. **Amended §v12.B** (cost, before any
result): v12cal reuses the published occ curves and drops fp / C-a / α* /
MMLU-1000; v12big drops 50%/97% and narrows α* and MMLU-1000; MMLU-1000 is read at
selected points only everywhere. Where this file says otherwise, §v12.B wins.
**Builds on:** v11 (`V11_FINDINGS.md`): DEC at 20 cells / 2.6–32B, α saturates at
256, IFEval at 6 cells.

The new story: the method's two constants — keep **1%** of coordinates, pick α
from **{2,4,8,16}** — become two rules read off the edit and a calibration split.
The claim that carries the paper is that **the rule for p, fitted on 2.6–9B, holds
at 27B/32B, which it never saw.**

---

## I. What is added, and the code for each

| # | item | code | config | GPU |
|---|---|---|---|---|
| 1 | 27B/32B sparsity + binarization curve (fp → dense sign → 50…99.95%), 3 seeds, 2 axes | `run_opsel.py` | `configs/v12/opsel_big.yaml` | ≥80 GB |
| 2 | p* from \|ΔW\| geometry, held-out at 27B/32B | `src/v12_geometry.py`, `src/v12_pstar.py` | fitted on `opsel_calib.yaml` | — |
| 3 | adaptive α*(p) on a calibration split | `src/v12_opsel.py` (rule), `run_opsel.py` | all opsel configs | — |
| 4 | figure panels: 20-cell DEC, 27/32B LOPO, 3-tier sparsity | `src/v12_figures.py` → `fig2b`, `fig4c`, `fig3t` | — | — |
| 5 | Table 3 as an operational table; IFEval for steering/SentenceDebias | `run_frontier.py`, `src/v12_frontier.py` | `configs/v12/frontier.yaml` | ≤3.8B |
| 6 | MMLU-1000 re-reading of headline, frontier and 27/32B operating points | built into `run_opsel.py` / `run_frontier.py` | `configs/v12/dec1k.yaml` + the others | ≤9B / ≥80 GB |
| 7 | remove the gate-provenance paragraph | DPO: `src/v12_frontier.py` (no GPU); edit cells: `run_frontier.py` | `frontier.yaml` | ≤3.8B |

`make verify` runs every v12 self-test, including an end-to-end pipeline test with
a fake model (`src/v12_selftest.py`: config → plan → unit → runner → fit →
predict → validate → analysis).

---

## II. Run order — changed from the one proposed, for one reason

The proposed order was *27/32B sparsity → design p\* → validate on 27/32B*. That
order makes the validation in-sample: once the 27/32B curves are measured, any
rule designed afterwards has seen its test set. The code therefore splits the
big tier into a phase that measures nothing but the geometry and a phase that
measures the curves, with the rule frozen in between.

```
1. bash run.sh opsel-cal         # ≤9B, 33 units: calibration curves + geometry + α*(p)
   bash run.sh dec1k             # ≤9B, 30 units: headline DEC at MMLU-1000
   bash run.sh frontier          # ≤3.8B, 24 units: Table A re-measured + IFEval
   make v12-check                # replay gates -- stop here if either fails
2. bash run.sh opsel-big-geom    # ≥80 GB, 12 units: train ΔW, record geometry, NO probe
3. freeze §VII into PREREGISTRATION.md (author)
   python3 src/v12_pstar.py fit        # v12cal only; prints leave-one-target-out
   python3 src/v12_pstar.py predict    # writes results/v12/pstar_prediction.json
   git commit the prediction; ship it to the GPU node
4. bash run.sh opsel-big         # ≥80 GB: retrains, checks ΔW fingerprint, measures
5. python3 src/v12_pstar.py validate; make v12-report
6. figures/tables → abstract/contributions
```

Steps 1 and 2 are independent and can run at the same time on different nodes.

**The order is enforced, not requested.** `predict` refuses if any 27/32B curve
row exists. Phase B refuses a cell without a prediction, a cell whose target is
in the fit set, or a cell whose retrained ΔW fingerprint differs from the one the
prediction used (`v12_opsel.load_pstar`). `run.sh opsel-big` refuses without the
prediction file. There is deliberately no override that yields a file phase B
accepts.

---

## III. Definitions (what the code computes)

**Geometry** (`v12_geometry.py`, from ΔW alone, before any probe):
M(p) = top-p share of ‖ΔW‖₁ (on binarize()'s exact retained set), participation
ratio / N, entropy effective support / N, Gini (Lorenz-grid approximation,
labelled). One float32 buffer plus in-place multi-partition: 16 GiB at 32B, no sort.

**p_true** of a calibration cell (cell = target × axis, seed-mean, nan → 0):
R(p) = removal(p) / removal(dense one-bit). Primary: the smallest grid p with
R ≥ ρ, log-interpolated toward the next grid point (**last-success**). The
published curves are not monotone: gemma dips at 97% and recovers at 99%, and
qwen-3B's dense edit is budget-limited and sits *below* its sparse ones. So a
first-failure walk would call gemma a 3.5% cell when 1% works. First-failure is
still computed and reported as a sensitivity check. If R ≥ ρ holds down to the last
grid point, the cell is **censored** and enters the fit at that bound, labelled
as such. The grid gains 99.95%: the published 7–9B curves never fell below 90%
retention by 99.9%, which would have censored every mid-tier cell.

**Rules** (fitted on eligible calibration cells, dense ≥ FLOOR):
`mass` τ = median M(p_true), p* = M⁻¹(τ) — **primary**; `pr`, `ent` (log-linear,
slope 1); `gini` (OLS); `fixed` = 1%, the baseline every rule must beat. All
rules are scored leave-one-*target*-out inside calibration, which is the only
evidence allowed for choosing between them.

**α\*** (`v12_opsel.select_alpha`): climb the ladder {1,2,4,…,512}; stop at the
first calibration-split failure; refine by 3 log-bisection steps (≈9% resolution).
The collateral gate is the published one (≤4/200 MMLU items, ppl ≤ 1.10). It is
evaluated on the **MMLU validation split + WikiText validation split**, so no
item is shared with evaluation, by construction; a runtime check refuses any
overlap. No bias probe is read, so the removal reported at α* was not selected
on, and whether α* also passes on evaluation items is a measurement (the rule's
held-out budget-pass rate). Status `exhausted` (the ladder never failed) means
α* is a lower bound, and is reported as one.

**Frozen rule** is kept alongside everywhere ({2,4,8,16}, argmax on evaluation),
so every v12 row can be compared with every published number.

**MMLU-1000** is `load_mmlu(1000)`, whose first 200 items are the evaluation
items (a runtime check asserts it). It is read at the selected points. Where
`mmlu1k_all_alphas` is set, it is read at every α, so the argmax that a
20-item-budget gate would have picked is recomputed as well. It is a robustness
reading only; the registered gate stays at 200 (experiments_v11.md §II).

---

## IV. Replay guards (checked before anything is reported)

| new | replays | expected |
|---|---|---|
| `v12cal` s<sp>, 6 occ cells | `results_v9/v8spc/{small,big}` | bit-identical |
| `v12dec1k` s0.99 / C-a@0.01 | `results_v9/v8dec` C-ref / C-a | bit-identical |
| `v12big` s0.99, seed 0, occ | `results/v11big` C-ref | bit-identical |
| `v12frontier` steering, SentenceDebias | `results_v9/v6trace/{steer,sentdebias}` | identical (SD's PCA may use randomized SVD: see below) |
| `v12frontier` edit, gemma | `results_v9/v6trace/x2` | bit-identical |
| `v12frontier` edit, phi | `results_v9/v6trace/x2` | **not expected**: published at eval batch 12, re-run at 6 (App. G) |

`make v12-check`. SentenceDebias uses sklearn `PCA(n_components=k)`, whose `auto`
solver can pick randomized SVD without a fixed `random_state`. If its replay
misses by a hair, that is the cause. The published code is reused unchanged
rather than patched, so that is recorded here rather than fixed silently.

---

## V. Things found while building this (no GPU needed)

1. **The DPO column does not need a rerun.** `results/v6trace/dpo/alpha_trace.jsonl`
   carries every probe's dmmlu/ppl_ratio; re-gating it flips **0 of 60**
   decisions and reproduces edit − DPO = +0.293 [+0.030, +0.540] exactly.
   `v12_frontier.py` reports the re-gated values, which removes half of the
   §5.7 disclosure without GPU time.
2. **Two numbers proposed for the new Table 3 are not the right population.**
   "SentenceDebias 88% in budget, ppl 1.005" is its real configurations *pooled
   with its random-subspace null* ((0.781·96 + 96)/192 ≈ 0.89). Real only, under
   the v9 gate: **78.1%**, median ppl 1.0097. "Steering 93%" is the as-run
   (float-gate) figure; under v9 it is **94.8%**. `v12_frontier.py` computes both
   from rows, never pooled with nulls, and at the *selected* configuration it gives
   steering ΔMMLU 2 items / ppl 1.023 and SentenceDebias 0 items / ppl 1.006.
3. The pooled frontier differences reproduce from the v9 tree exactly
   (−0.021, −0.028, +0.293), so Table 3 can be built now and upgraded in place
   when `v12frontier` lands.

---

## VI. OPEN DECISIONS — settle before step 1, never after

Each has a coded default; changing it after seeing calibration curves is
outcome-selection on the calibration set, and after seeing 27/32B it voids the
held-out claim.

| # | decision | default in code | why it matters |
|---|---|---|---|
| D1 | retention target ρ | 0.90 | defines p_true; ρ=0.95 pushes every p* up |
| D2 | eligibility floor on dense removal | 0.05 | llama8b·BBQ-age (DEC C-ref +0.023) falls out; reported, not dropped |
| D3 | primary rule | `mass` | the other three are reported; LOO on calibration is the only permitted basis for a switch, and only before step 3 |
| D4 | held-out success criterion | **none coded — must be written** | proposal: PASS iff R(p*) ≥ ρ − 0.05 in ≥ 3 of the 4 eligible held-out cells; secondary: \|log₂(p*/p_true)\| ≤ 1 |
| D5 | adaptive-α headline | as-deployed removal at α*; eval pass rate beside it | alternative zeroes cells whose α* fails on evaluation items |
| D6 | second big-tier axis | bbq_Age | the only non-occ axis with cells at 2.6B, 3B, 7B and 8B |

---

## VII. Registration block (paste into PREREGISTRATION.md as §v12 after §VI)

```
## v12.A — Operating-point selection: design frozen (<date>, before any v12 run)

Panels: v12cal (fit population, <=9B), v12big (HELD OUT, gemma-2-27b-it and
Qwen2.5-32B-Instruct x {occ_gender, bbq_Age} x seeds {0,1,2}), v12dec1k,
v12frontier. Configs configs/v12/*.yaml at commit <sha>.

p*: cell-level (seed-mean), rule <D3> fitted on eligible v12cal cells only
(dense >= <D2>), retention target rho = <D1>, p_true = last-success on the grid
{0,.5,.9,.95,.97,.99,.995,.999,.9995}. Prediction frozen by
`src/v12_pstar.py predict` from phase-A geometry BEFORE any v12big curve row
exists; the file's sha256 is recorded in every pstar row.
Held-out success: <D4>.

alpha*: first-failure on ladder {1..512} + 3 log-bisections, gate = v9 gate on
MMLU validation (200) + WikiText validation (20 chunks); no bias probe read.
Reported: <D5>. The frozen {2,4,8,16} argmax is reported alongside everywhere.

MMLU-1000: a robustness reading of selected points (and every alpha where
configured); never a selection criterion. The registered gate stays at 200.

Both outcomes written now. If p* meets the criterion at 27/32B, "fixed 1%" is
replaced by the rule in the method section. If it does not, the paper keeps
1% as a constant, reports the rule as a negative result, and makes no
geometry-predicts-sparsity claim -- the same discipline as §5.8's gate.
```

---

## VIII. Not done here — needs the LaTeX source, which is not on this box

- **App. H Proposition 2** ("sign permutation ⇒ expected alignment 0"). The
  manuscript on this box is the Markdown draft, whose appendices are a table of
  contents. The fix needs the `.tex`.
- **Writing** (abstract/contributions away from patch size, T-Switch in related
  work, bug history out of the main text). This comes after the results, per
  the run order. The Markdown draft is regenerated by the audit
  (`binary_debiaser_draft_v3.md` is never hand-edited), so prose edits go
  through `audit/prose_edits.yaml` or into the `.tex` directly.
- **Figure numbering.** Repo `fig2` = the decomposition (the LaTeX's Fig 1).
  The new panels are `fig2b` (20 cells), `fig4c` (27/32B LOPO) and `fig3t`
  (3 tiers; its third panel reads "not yet measured" until v12big exists).
