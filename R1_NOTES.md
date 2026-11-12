# R1 — implementation notes and amendment log

Companion to `R1_REBUTTAL_EXPERIMENTS.md` (registered unmodified as commit `0fe9043`,
2026-11-10T08:20:19Z; hash in `audit/manifest_r1.yaml`). The plan fixes the designs and
the reading rules. This file records (1) the facts E0 has read out of the code so far,
(2) every implementation choice the plan leaves open, fixed **before the first
registered R1 unit**, and (3) an append-only log of anything that changes afterwards.
Nothing here changes a reading rule.

---

## 1. E0 facts already established (from code, not from intent)

Full `SPEC.md` with every item follows in `results/r1/E0/`. These are the ones the
runner design depends on.

| # | fact | source |
|---|---|---|
| 1 | ΔW is the difference of **two separate LoRA fine-tunes**: one on the biased corpus C+, one on the debiased corpus C−, each with the plain causal-LM loss over the whole text (pad positions masked), then `ΔW = dW_biased − dW_debiased` with `dW = (B·A)·(lora_alpha/rank)`. Not a single contrastive objective. | `colab_t2t4.py:186-240` (`train_task_vector`), `:247-271` (`contrast_`), `run_opsel.py:229-238` |
| 2 | Training: rank 16, lora_alpha 32, dropout 0, 250 steps, AdamW lr 1e-4 (weight decay 0) under OneCycleLR (pct_start 0.1), batch 8, max_len 96, grad-norm clip 1.0, gradient checkpointing on; targets `q/k/v/o_proj` (phi: `qkv_proj` only). | same |
| 3 | The edit is merged as `W ← W − α·E`. | `colab_t2t4.py:363-375` (`apply_edit(..., sign=-1.0)`) |
| 4 | Per-tensor scale `s_t` = mean of `|ΔW|` over the coordinates of tensor *t* that survive the **global** top-k threshold (one threshold over all edited tensors concatenated). Every DEC condition uses C-ref's `s_t` (`ref_tensor`). | `colab_t2t4.py:273-355` (`binarize`), `src/v8_edits.py:323-333` |
| 5 | **Removal is `|pre skew| − |post skew|`**, not signed `pre − post`. An edit that overshoots into opposite-direction skew is charged the overshoot. The plan's E3a calls the signed form "현행"; the current definition is its variant (ii). E3a therefore reports (i) signed and (iii) relative as the alternatives. | `colab_t2t4.py:681` (`bias_reduction`) |
| 6 | Skew: occupation→gender is `2·mean(P(stereotype-congruent pronoun)) − 1` over probe occupations × templates (range [−1, 1]); every pair axis (BBQ, CrowS, StereoSet) is the mean over probe pairs of per-token-normalised `log p(stereotyped) − log p(anti-stereotyped)`. Probe and edit halves are disjoint. | `colab_t2t4.py:518-535`, `:620-633`, `:548-610` |
| 7 | The frozen-grid rule (in-budget argmax over 4 values of α) reads bias on the **same probe items** it reports, and gates on the same 200 MMLU test items and 20 WikiText test chunks. The α\* rule reads 200 MMLU *validation* items and 20 WikiText *validation* chunks and no bias probe; it shares no item with evaluation (checked at run time). | `run_dec.py:200-230`, `run_opsel.py:101-131, 246-261, 439-476` |
| 8 | Existing `alpha_trace.jsonl` rows carry no per-item probe score and no per-item MMLU outcome; DEC traces (v8dec/v11dec) carry no post-skew at all, only `bias_reduction`. So E3c takes its fallback path, and the signed-removal variant of E3a needs new measurements for the DEC conditions. The R1 runner records both from now on. | `run_dec.py:209-228`, `colab_t2t4.py:724-760` |

## 2. Implementation choices fixed before the first unit

1. **Hardware.** One NVIDIA L40S (46 GB), the card every published ≤9B panel ran on. The
   run stack was reinstalled on 2026-11-10 (`bash setup.sh`: torch 2.11.0+cu128,
   transformers 4.57.1, peft 0.20.0; datasets 5.0.1 and accelerate 1.15.0 are unpinned in `requirements-run.txt`, and what earlier installs resolved them to is not recorded -- hence the replay guards in item 12).
   **E4b and E1b (27–32B) need a ≥80 GB card and cannot run here**; they are deferred,
   not dropped.

2. **Runner.** `run_r1.py`, one unit per process, baselines before the first training
   pass — the way `scripts/run_unit.py` ran v12astar. Training, calibration items,
   collateral measurement and the α\* search are imported from the published code
   (`run_opsel.train_contrast`, `run_opsel.load_mmlu_calib`, `run_opsel.collateral_eval`,
   `v12_opsel.select_alpha`), not re-implemented. Evaluation batch 6, MMLU 2 rotations.

3. **Per-rung traces (the plan's one protocol change, §0).** At every probed α the edit is
   applied once and measured on both splits: calibration collateral and evaluation-split
   `fast_eval`. The selector receives calibration `{mmlu_acc, ppl, n_items}` only;
   `src/r1_select.py` raises `BiasLeak` on any other field (17-check selftest). The
   removal reported at α\* is the evaluation measurement taken at that rung, in the same
   process, instead of a second application of the same edit.

4. **Curve-only measurements**, taken after the selection and never seen by it: one ladder
   rung above the first failure (`role=curve_extra`), and the frozen grid {2,4,8,16}
   wherever the ladder did not visit it (`role=frozen_fill`, sign arms only — this also
   replays the published frozen-grid removal of each DEC condition).

5. **Reference arms.** `C-ref(α*)` and `C-a(α*)` in every registered reading are the
   **registered v12astar values** (the paper's numbers), as the plan says ("기존").
   Same-process replays of both run in the E5 panel as a reproducibility check and are
   reported next to them. One exception, decided now: if the retrained ΔW fails to
   reproduce v12astar's `geom_sha` in any unit, the pairing premise is gone and the
   replays replace the registered values in **all** units.

6. **Status handling**, as §v12.E: α\* status `none` (the lowest rung already fails) scores
   0, the edit is not deployed; `exhausted` (512 passes) is reported as censored.

7. **E5 operators** (all at 1% density on the same ΔW, merged `W ← W − α·Ê`, α by the same
   bias-blind rule on the wide ladder {2⁻⁶ … 2⁹}):
   - `O-trim`: C-ref's top-|ΔW| support, full-precision values.
   - `O-dare`: C-a's random support (same draws), full-precision values × 100
     (= 1/(1−p), p = 0.99).
   - `O-compeft`: ComPEFT Algorithm 1 applied to the dense contrast vector — global
     top-1% by magnitude, sign, one scalar `σ(ΔW)` (the standard deviation of the whole
     ΔW over all edited tensors) times α. ComPEFT tunes its α on task validation
     accuracy over {0.5 … 10}; here α is set by the collateral rule like every other arm,
     which is the "comparable tuning and collateral constraints" the review asks for.
     ComPEFT's own experiments use densities of 5–50% on adapter parameters; 1% on the
     merged product is this paper's operating point, applied to its operator.

8. **Queue order.** Pass A: E4 `C-rand` (validity gate G1). Pass B: E5 (three operators +
   the two replays). Pass C: E4 `C-b`, `C-layershuf`, `C-tensorshuf`, `C-bottom`. Then
   E6a/E6b, E7b, E8, E9, E11. The plan's order had E6a between E4-C-rand and E5; E5 is
   moved ahead because the ComPEFT/DARE comparison is the reviewer-facing gap (author
   decision, 2026-11-10). Order changes no reading.

9. **E2 bookkeeping rides along.** Every unit writes `A_ref`, `A_a`, `ρ_eff`, `ρ_k`,
   kurtosis and the top-1% mass share of its ΔW to `unit.jsonl` (a function of ΔW alone),
   so E2 needs no separate reconstruction pass.

10. **Descriptive extras, used by no registered reading:** the number of edited
    coordinates that actually change in bf16 and the L1/L2 of the applied change at each
    rung; the per-item probe log-likelihoods (`probe_ll/*.npz`).

11. **No partial reads.** Run logs print α\*, probe counts and gate status only. Analysis
    scripts refuse to print a removal until every unit of the arm they read is on disk
    (the §v12.E convention).

12. **Replay guards, recorded per unit in `unit.jsonl`.** (a) the retrained ΔW's
    fingerprint against v12astar's `geom_sha`; (b) the unedited model's skew, MMLU and
    perplexity on both splits against the values v12astar recorded; (c) for DEC arms, the
    frozen-grid removal against the published DEC panel. A smoke unit
    (gemma | occ_gender | s0, throwaway panel `r1/_smoke`, 2026-11-10, never cited)
    reproduced all three bit-for-bit under the reinstalled stack: `geom_sha`
    `73bdca42dc266097`, baselines identical, C-rand frozen-grid removal
    `0.0007482993039644725` = `results_v9/v8dec`. Only mechanics were read from that unit.

13. **A property of the `ref_tensor` scale worth stating once.** A tensor in which C-ref
    keeps no coordinate has `s_t = 0`, so no sign arm edits it, whatever its support. The
    full-precision operators of E5 (`O-trim`, `O-dare`) and `O-compeft` do not use `s_t`
    and are not subject to this. `unit.jsonl` does not count such tensors; E0 will.

## 3. Amendment log (append-only)

*(empty at registration)*

**2026-11-10, during pass A (E4 `C-rand`), before the first E5 unit — E8 folded into the
E5 pass.** E8 measures `C-ref` and `C-a` at α ∈ {α\*, α\*/√2, α\*/2} (evaluation gate,
MMLU-1000, removal) and re-selects α\* with a 2-item and an 8-item MMLU allowance. Both
arms are already replayed in the E5 panel, so the E8 measurements are taken there, in
the same process, right after each arm's registered α\* search: `configs/r1/e5.yaml`
gains `backoff_arms`, `backoff_factors`, `mmlu1k_arms`, `budget_items_extra`;
`run_r1.py` and `src/r1_select.py` gain the corresponding code, inert when those keys
are empty (as in `configs/r1/e4.yaml`, so pass A is unaffected). Differences from the
plan's wording, both in the direction of measuring more: α\*(b) is found by re-running
the registered first-failure search under budget *b* (re-using every rung already
measured and adding only the new bisection points) rather than read off the b = 4
trace, and removal at every such point is measured on the evaluation split like any
other rung. E8's registered reading (deployment rule α\*/√2, target ≥ 90% evaluation
pass; Δ_selection invariant to backoff if collateral is homogeneous) is unchanged.

**2026-11-10, before the first E7b unit — E7b harness and the parameters the plan leaves
open.** `run_r1_dpo.py` + `configs/r1/e7b.yaml`. (a) The harness is the corrected one
(section 4, E7a): baseline on the clean model before training; the adapter is unloaded
and the baseline re-measured and required to match bit for bit; the update is added at
merge scale m. (b) Pairs are built exactly as the published script built them, which
already dropped identical pairs; so "differing pairs only" is not a change. (c) The
registered 4-configuration space {lr 1e-5, 5e-5} × merge {1, 2}, β = 0.1, skip threshold
8 differing pairs, is run as two arms. "epoch 고정" names no number: **3 epochs** is
used (TRL's default; the published 250 steps × batch 2 make 3.5 passes over a full
144-pair occupation corpus). With 8–50 differing pairs that is 12–75 optimizer steps,
so these arms may under-train; that is a property of the registered design and will be
reported as such next to the `pub` arm. (d) Added arm `pub`: the as-published recipe
(lr 1e-4, 250 steps, batch 2) at merge scales {0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1},
trained wherever ≥ 8 pairs differ, with the published threshold of 20 applied in the
analysis. It checks the E7a reconstruction by measurement and is not part of the
registered reading E7-1 / E7-2. (e) TRL 1.9.2 (the version named in
`run_dpo_baseline.py`) was installed with `--no-deps`; the pinned stack is unchanged.
(f) Smoke unit (qwen | occ_gender | s0, arm `pub`, throwaway panel `r1/_smoke_e7b`):
baseline restored exactly after unloading; removal by merge scale −0.021, −0.055,
−0.083, −0.045, +0.084, +0.301, +0.547 at perplexity ratios 1.003 … 1.416 (in budget up
to m = 0.5). The E7a reconstruction for the same unit reads −0.081, +0.097, +0.286,
+0.477 at m = 0.25 … 1 with ratios 1.030 … 1.367: the same picture from a separately
trained adapter.

**2026-11-10 ~09:20Z, during pass A — queue order.** The chain was stopped between
measurements and restarted as: pass A (E4 `C-rand`, resumes where it stopped) →
E7b arm `pub` (24 units, about 1.6 h) → pass B (E5 + replays + E8) → pass C (rest of
E4) → E7b registered arms. Reason: the DPO correction withdraws a claim of the
submitted paper, and a withdrawal should rest on a measurement rather than on a
reconstruction; the cost is a 1.6 h delay of pass B. The unit that was running when the
chain stopped is re-run from the start (its partial `alpha_trace.jsonl` rows are
superseded; analyses read the `rungs` stored in complete `removal.jsonl` rows).
`scripts/r1_queue.sh` gained an optional runner field for this. No reading rule changes.

**2026-11-10 ~09:30Z — the 27–32B panel (E4b + E1b), to be run by the authors on a
≥ 80 GB card.** `configs/r1/big.yaml`, `scripts/r1_big.sh`, `scripts/r1_pack.sh`; panel
`r1/BIG`, reference panel `v12big`, the four cells and three seeds of
`configs/v12/opsel_big.yaml`. Per unit: `C-rand` under the registered α\* rule with
dual-split rungs (E4b); `C-ref` and `C-a` at the fixed scales 32, 64, 128 (E1b), their
α\* points being the registered v12big values. Two things the plan does not spell out:
(a) a corpus that is not on the box is elicited by the model itself, the route
`scripts/run_unit.py` takes for every big-tier cell; (b) if the retrained ΔW does not
reproduce v12big's fingerprint (or v12big is not on the box), the unit also runs the
α\* search for `C-ref` and `C-a`, because the registered reference values then belong
to a different ΔW. `run_r1.py` gained `elicit_missing`, `grid_only_arms` and
`grid_only_fallback_full` for this; all are off in the E4/E5 configs. Checked on a small
model in throwaway panels (fixed-scale arms; elicitation of a missing corpus). The
elicitation check showed why (b) is needed: re-eliciting gemma-2-2.6B | occupation in a
different item order changed 3 of 144 pairs.

**2026-11-10 17:10Z, pass B at 51 of 150 unit-arms — E5 and E8 readings coded; two
sentences of E8 pinned down.** `src/r1_e5.py` and `src/r1_e8.py` were written, self-tested
on synthetic rows (12/12, 8/8) and committed while the panel was incomplete; both refuse
to print a removal until every unit-arm they read is on disk. Before the commit, only the
field layout of the finished rows (key names, counts, α\* status) was looked at, no removal
and no skew. E5 is read exactly as registered (reference arms per §2.5; the other source
is printed next to the verdict and a reading that changes with it is labelled FRAGILE).
The plan states E8's reading in two sentences that leave room; they are fixed here,
before any E8 number exists:
- *"evaluation pass ≥ 90%" at α\*/√2*: the rate is taken over the units in which the edit
  is deployed (α\* exists), **per arm**, and the target is met only if **both** `C-ref`
  and `C-a` reach 0.90. The pooled rate is printed next to it.
- *"Δ_selection is invariant to backoff"*: the paired cell bootstrap interval of
  `Δ_selection(α*·f) − Δ_selection(α*)` includes 0, read at f = 1/√2 (the deployment
  rule); f = 1/2 is reported. Removal is the registered `|pre| − |post|`; an undeployed
  edit scores 0 at every backoff.

**2026-11-11, before the first E13 unit — a new experiment, E13: balanced-contrast
control on the occupation axis (author decision of the same day).** E13 is not in the
plan committed as `0fe9043`. It was added after the pronoun check (§4, "What removal on
the occupation probe consists of") showed that the deployed edit closes the occupation
skew by raising P("she") on male-stereotyped occupations, and after we attributed this to
the elicited contrast being one-sided. That attribution is so far an observation about
the corpora, not a tested claim. E13 tests it. Its motivation is post hoc; its design and
its reading are fixed here, before any E13 unit runs.

- *Corpora* (`scripts/r1_make_balanced.py`, CPU only, record in
  `audit/r1_e13_corpora.json`). **`<designer>_bal`, the primary control**: the elicited
  corpus with mirrored swaps. The biased corpus is unchanged. In the debiased corpus,
  with D_mf the items that replace "he" by "she", D_fm the reverse, and the candidates
  the items on a female-stereotyped occupation in which both corpora hold the
  stereotype-congruent "she" sentence: k = min(|D_mf|, |D_fm| + |candidates|); a seeded
  random k of D_mf are kept and the rest reverted to the biased string; a seeded random
  k − |D_fm| candidates become "she" → "he" swaps. Item order and corpus size are those
  of the elicited file. Resulting k: 9–10 (gemma), 31 (llama), 38–39 (qwen-3B), 16–17
  (phi), 56–57 (qwen-7B; 9–11 of its 66–67 elicited swaps are reverted because only
  49–51 candidates exist). **`exo`, secondary**: `colab_t2t4.exogenous_corpora`, the
  published designer-free builder — every item swapped, 72 in each direction.
- *Runs* (`configs/r1/e13.yaml`, panel `r1/E13`, 30 units, `run_r1.py` unchanged; the
  designer field only names the corpus file). Arm `C-ref` only, measured as in the E5
  panel: baselines before training, α\* by the bias-blind rule, the fixed grid filled
  in, per-item likelihoods at every rung, α\*/√2 and α\*/2. The elicited counterpart of
  every unit is its `C-ref` row in `results/r1/E5` (which reproduces v12astar exactly).
  Queued behind the running chain with `scripts/r1_after.sh` (waits for the queue lock;
  the running queue is not touched): pass C → E7b registered arms → E13.
- *Quantity.* f, m = mean P("she") on the 12 female- / 12 male-stereotyped probe
  occupations; base = (f + m)/2; shift = base(after) − base(before) **at the scale the
  fixed grid {2,4,8,16} selects**, the operating point of the paper's main tables (a run
  with no in-budget grid scale is not edited: shift 0, removal 0). Cell = seed mean;
  five occupation cells; paired bootstrap over cells.
- *Reading* (`src/r1_e13.py`, committed with this entry). ratio = |mean shift(balanced)|
  / mean shift(elicited); D = shift(elicited) − shift(balanced).
  - **E13-confirmed**: ratio ≤ 0.25 and D's interval excludes 0 → the base-rate shift
    comes from the one-sided contrast; the sentence in the paper stands.
  - **E13-partial**: 0.25 < ratio < 0.75 and D's interval excludes 0 → it explains part.
  - **E13-refuted**: ratio ≥ 0.75, or D's interval includes 0 → the attribution is
    withdrawn from the paper and the shift is reported as unexplained.
  - Qualifier: if the balanced edit's removal under the grid rule has an interval that
    includes 0, a confirmed or partial verdict is labelled **vacuous** (an edit that does
    nothing cannot shift anything) and is not used as support.
  - Reported, never the verdict: the same at α\* and α\*/2; f-side and m-side movements;
    removal under both rules; runs with both probabilities above 0.95 or below 0.05; the
    `exo` arm read by the same rule.
- With five cells the interval of D excludes 0 only if the cells agree in sign almost
  without exception; that is intended.

**2026-11-12 03:08Z — the VM stopped during pass C; the queue was restarted (no reading
rule changes).** The cloud VM stopped at about 2026-11-11 08:16Z (last write 08:15:49Z)
with pass C at 101 of 150 unit-arms, and came back at 2026-11-12 02:12Z with the same
disk, container and software stack (torch 2.11.0, transformers 4.57.1, peft 0.20.0).
No result file has a torn line. One arm was in flight (qwen | bbq_Age | seed 2,
`C-bottom`): its ten rows in `E4/alpha_trace.jsonl` have no row in `removal.jsonl`. They
are left in place (the trace is append-only and its readers deduplicate; `src/r1_e4.py`
reads `removal.jsonl`), and the arm is measured again from the start. The follower that
was to start E13 died with the VM, so E13 is now part of the main chain, restarted with
`scripts/r1_queue.sh` at 03:08:33Z: pass C (remaining units) → E7b registered arms →
E13. The first resumed unit reproduced its ΔW fingerprint and both baselines
(`geom_sha` match, `pre_eval_match`, `pre_calib_match`). A copy of `results/r1` as of the
restart is kept off the server.

**2026-11-12, before the first FRONT8 unit — a new panel, FRONT8: the 1% edit on the
eight cells of the comparison table.** Not in the plan of `0fe9043`. E0 found that the
"edit" of the comparison table (edit / steering / SentenceDebias / DPO) is the dense
sign-only edit, while the abstract, the deployment table and the conclusion speak of the
1% patch; the revision and the response draft promise the 1% edit on the same cells.

- *Runs* (`configs/r1/front8.yaml`, panel `r1/FRONT8`, 12 units, `run_r1.py` unchanged).
  `C-ref` on gemma, llama, qwen-3B and phi × CrowS-Pairs socioeconomic, self designers
  and seeds 0–2 as in `configs/v12/frontier.yaml`, stored corpora. The four occupation
  cells of the table are not run again: they are the `C-ref` rows of `results/r1/E5`.
  Queued behind the running chain with `scripts/r1_after.sh`.
- *Quantity.* Per run, the removal at the best in-budget of the four fixed scales
  {2, 4, 8, 16} (none in budget → not edited → 0): the selection space of every method
  in that table, and the `frozen` field of the run's row. Cell = seed mean; eight cells;
  bootstrap over cells, paired for differences. The comparators are the stored cell
  values and are not run again: dense edit, steering and SentenceDebias from
  `results/v12/frontier_v12.json`; DPO from the corrected measurement,
  `results/r1/E7b/e7b.json`, published selection space.
- *Reading* (`src/r1_front8.py`, committed with this entry; selftest 8/8). With
  d_steer = 1% edit − steering and d_sent = 1% edit − SentenceDebias:
  - **FRONT8-tie**: both intervals include 0 → the paper may say of the 1% patch itself
    that it removes as much as steering and projection within the uncertainty.
  - **FRONT8-below**: at least one interval lies entirely below 0 and none above → the
    tie is stated for the dense sign-only edit only; the paper says that the 1% patch
    removes less, with the numbers, and the conclusion's "small, mergeable patch that
    matches activation steering and projection" is rewritten.
  - **FRONT8-above** / **FRONT8-mixed**: stated as found; no summary claim if mixed.
  - Reported, never the verdict: 1% − dense (what sparsity costs on these cells),
    1% − DPO, the same at α\*, the medians of the chosen configurations and the share of
    the 96 grid configurations in budget. IFEval is not measured for the 1% edit; that
    column stays with the dense edit.
- *What was known when this was written.* This is not a blind registration. The 12
  occupation runs are on disk and have been read for other purposes; their fixed-grid
  cell means are those of the submitted scale-up table (+0.693, +0.615, +0.391, +0.116),
  and the same table has earlier fixed-grid values, from another runner, for three of
  the four CrowS cells (+0.084, +0.011, +0.029). From these one expects a mean near
  +0.24, some 0.08 below steering and SentenceDebias, the difference carried by
  phi | occupation (+0.116 for the 1% edit against +0.316 for the dense one). The rule
  is fixed now so that the consequence for the text is fixed before the numbers are
  final, not because the outcome is unknown.
- *Checks made before the panel ran, without computing any FRONT8 difference.* The
  script accepts the 12 E5 rows (in each, `frozen` is the in-budget argmax of the four
  grid scales); it reproduces the published pooled differences of the dense edit from
  the stored cell values; and the unedited skew of the 12 runs equals that recorded by
  the stored comparison panel exactly (max |diff| 0).

**2026-11-12 ~11:10Z, before E3c is run on any data — E3c (select/report split): the
details the plan leaves open.** The plan fixes the rule (`sha256(item_id) mod 2`, choose
on one half, report on the other, both directions, average) and no verdict: it asks for
the absolute change and for whether the intervals of the main contrasts change status.
Fixed here, with `src/r1_e3c.py` (selftest 7/7), before the script reads a likelihood:

- *item_id.* Occupation axis: the occupation string of `PROBE_OCC`, so the eight
  templates of an occupation stay in one half (24 items; the rule gives 7 f + 5 m and
  5 f + 7 m — computed from the names alone, before any result). Pair axes: the key of
  the probe pair as built by `colab_t2t4` (unique per pair). The halves are written to
  `audit/r1_e3c_split.json` before the analysis.
- *Skew per half* from the stored per-item log-likelihoods with the probe's own
  formulas. *Budget*: not split; a scale is admissible if the run's recorded evaluation
  check passed, as in the published rule. *Selection*: largest removal among admissible
  grid scales on the selecting half, first maximum; none admissible → 0.
- *Reported*: per variant, the removal with the scale chosen on all items (published),
  on the other half (split-half), and on the same half; same-half minus split-half is
  the optimism of selecting on the reported items; for C-ref − C-a, C-a − C-rand and
  C-ref minus each perturbed variant, the interval under both and whether its status
  changes.
- *Scope*: the seven variants on the ten main cells (C-ref, C-a from `results/r1/E5`;
  the other five from `results/r1/E4`). The eight-cell comparison is **not** covered:
  per-item likelihoods of steering and SentenceDebias were not stored, and the plan's
  fallback (re-evaluating the probes on GPU) has not been run.
- Sanity gates: the skew recomputed from all items equals the recorded one for every
  run and scale, and the full-set selection equals the run's `frozen` row.

---

## 4. Readings and findings log

Each entry names the script that produces it. Registered readings are applied as
written in the plan; anything else is labelled.

### E1 — matched-collateral curves (2026-11-10, zero GPU, `src/r1_e1.py`)

- **Sanity gate reproduced**: frozen-grid Δ_selection +0.2835 [+0.1692, +0.4072];
  extension to α ≤ 512 under the joint gate +0.1041 [+0.0361, +0.1827]; zero difference
  from `results/v10/alpha_extension.json`. 540 rows, no duplicates.
- **Δ_matched(c)** (R at the scale where the perplexity ratio equals c; n = 10 cells,
  30/30 runs solved per arm, none unreached): c = 1.02 +0.068 [−0.010, +0.161];
  1.04 +0.054 [−0.019, +0.142]; 1.05 +0.054 [−0.016, +0.134]; 1.06 +0.053
  [−0.015, +0.130]; 1.08 +0.043 [−0.017, +0.109]; **1.10 +0.034 [−0.021, +0.094]**.
- **Registered reading: E1-B** (the interval includes 0 at both c = 1.10 and c = 1.05).
  It does not move under the implementation choices the text leaves open (isotonic on
  or off, log-α or α interpolation, first or last crossing: largest shift 0.004).
- Registered secondary axis (largest R with MMLU drop ≤ b items, MMLU only): b = 2
  +0.073 [+0.016, +0.137]; b = 4 +0.088 [+0.036, +0.144]; b = 6 +0.077 [+0.028,
  +0.135]; b = 8 +0.064 [+0.022, +0.112]. All exclude 0.
- **Post-hoc, a different estimand from the registered one**: reading the perplexity
  axis in budget form (largest R with ratio ≤ c, as the published estimands do) gives
  +0.057 [+0.019, +0.107] at 1.10 and +0.078 [+0.025, +0.148] at 1.05, which would
  read E1-C. Two cells account for the difference (qwen|bbq_Age, qwen|bbq_Race):
  C-ref's removal peaks at α = 8 and falls before its perplexity limit. The registered
  reading stays E1-B.
- Step 6: the literal pooled fit has unequal exponents (5.3 vs 3.3, dominated by
  runaway points) and its h ratio is not interpretable. From the curves, the scale at
  which the two arms reach ratio 1.10 differs by a median factor of 4.61 (squared 21.2);
  the α\* ratio squared is 19.0–20.8.
- Step 7: nominal ‖αE‖_F of C-a over C-ref at matched c = 1.10 is 4.81 [4.11, 5.57].
- Literal resolutions of ambiguous wording are listed in `results/r1/E1/e1.json`.

### E7a — the published DPO column is an evaluation error (2026-11-10, zero GPU, `src/r1_e7a.py`)

Found while reading `run_dpo_baseline.py` for E0. Not anticipated by the plan.

- **What the script measures.** TRL's `DPOTrainer` attaches the LoRA layers to the model
  object in place. The script then takes its "pre" measurement on that object, i.e. with
  the trained adapter active, and applies `apply_edit(model, dw, alpha=a, sign=-1.0)`,
  which subtracts a·ΔW from the base weights under the adapter. The "post" model is
  base + (1 − a)·ΔW. The recorded "removal at α = a" is therefore
  |S(base + ΔW)| − |S(base + (1 − a)ΔW)|: the bias regained when the DPO update is undone.
- **Three confirmations in `results/v6trace/dpo`**, no rerun needed: the recorded
  pre-skew differs from the unedited skew of the same model and axis in every run
  (median gap 0.476; llama|occ 0.0000 against 0.7605); pre-skew minus removal(α = 1)
  recovers the unedited skew to within 0.011; the recorded perplexity ratio is below 1
  and non-increasing in α in 15/15 runs.
- **Sanity gate**: the same loader reproduces the published DPO +0.006 [−0.064, +0.095]
  and edit − DPO +0.276 [+0.027, +0.513].
- **Reconstructed intended contrast** (merge scales {0.25, 0.5, 0.75, 1}, best in budget,
  skipped cells 0): DPO +0.109 [−0.005, +0.302] on the 8 cells; edit − DPO +0.173
  [+0.044, +0.342]. On the 5 cells with ≥ 20 differing pairs: DPO +0.175 [−0.005,
  +0.471], edit +0.288, edit − DPO +0.113 [+0.005, +0.221].
- **Why DPO is still behind in budget**: at its trained scale the adapter removes most
  occupation bias (llama +0.761, phi +0.539, qwen +0.504) but its perplexity is a median
  2.0× the unedited model's; the full adapter is in budget in 0 of 15 runs, and 14 of 60
  configurations are in budget. On qwen|crows it increases bias (−0.277 at full scale).
- **Status**: a reconstruction from persisted rows, exact up to bf16 rounding of the
  in-run "unedited" model; collateral is known only relative to that model. E7b
  re-measures. **The sentence "DPO removes no detectable bias" is withdrawn** wherever
  it appears (abstract, contribution 3, §2, §5.7, Fig. 4, Table 2, conclusion), and so
  is its explanation as a signal–method mismatch: the published script already trained
  on differing pairs only.
- **Addition to E7b** (not a change to its registered reading): one extra arm that
  repeats the as-published configuration (lr 1e-4, 250 steps, merge scales
  {0.25, 0.5, 0.75, 1}) under a correct harness — baseline before training, delta
  added to a clean model — so the reconstruction above is checked by measurement.

### E3a — removal definitions, α\* panel (2026-11-10, zero GPU, `src/r1_e3a.py`)

Registered as a robustness report with no fork; "if the interval status of a headline
changes, disclose it". It changes.

- At α\* the skew **changes sign** in 10/30 C-ref runs and 7/30 C-a runs, almost all on
  the BBQ cells (qwen|bbq_Age 3/3 and 3/3, qwen|bbq_Race 3/3 and 2/3, phi|bbq_Age 2/3 and
  2/3). The code's removal `|pre| − |post|` charges the part beyond zero.
- Δ_selection on the ten registered cells, frozen grid / α\*:
  - (ii) `|pre| − |post|` (published): +0.285 [+0.170, +0.411] / **+0.073 [−0.036, +0.213]**
    (sanity gate: identical to `results/v12/astar_verdict.json`).
  - (i) signed `sign(pre)·(pre − post)`: +0.321 [+0.213, +0.435] / **+0.127 [+0.032, +0.257]**,
    excludes 0.
  - (iii) relative: +0.543 [+0.431, +0.638] / +0.096 [−0.133, +0.341], includes 0.
  - (iv) capped at zero, `|pre| − max(0, sign(pre)·post)` — **post-hoc**, not in the plan:
    +0.292 [+0.179, +0.414] / +0.100 [+0.007, +0.231], excludes 0.
- The two cells that pull the published α\* gap down (qwen|bbq_Age −0.132, qwen|bbq_Race
  −0.118) are the two where C-ref crosses zero in every run; under (i) they are +0.132
  and +0.160, under (iv) +0.000 and +0.021. On BBQ a negative skew is a preference for
  the "unknown" answer in an ambiguous context.
- Scope: only v12astar stored signed skew for both arms. The grid extension and E1 can
  be re-read under (i)/(iv) once the E5 replays (per-rung signed skew) are complete.
- What this does and does not license: the statement "under α\* the gap is
  indistinguishable from zero" holds under the published definition and under (iii),
  not under (i) or (iv). It is a statement about how overshoot is scored as much as
  about the two supports.

### E0 — SPEC (2026-11-10, `results/r1/E0/SPEC.md`)

Full text there. Statements of the submitted paper that the code or artifacts do not
support, beyond DPO (E7a) and the removal definition (E3a):

1. Table A's "edit" is the **dense** sign-only edit (`run_frontier.py:118`, sparsity 0),
   while the abstract and Table B describe the 1% patch.
2. "Diagnosis framing": the prompts are generation instructions and two fixed candidate
   sentences are scored as the reply; there is no diagnosis question.
3. `contrast_gap` is a net share (can be negative), not "the fraction of items on which
   the model can express any contrast".
4. "Permutation maps are logged per run": they are regenerable from the seed, not logged.
5. MMLU run-to-run variation "one or two items": stored replays of llama-3.2-3B differ
   by up to 4 items post-edit at bit-identical skew and perplexity.
6. The α = 256/512 medians quoted in App. B are C-a's only.
7. Prompting numbers on the occupation axis are on half the scale of every other
   method's skew (`legacy/run_prompt_baseline.py`).
8. The audit manifest covers the v10 draft, not the v11–v13 numbers; `figures/make_figs.py`
   is not in the repo; "every edit" is regenerable, not stored.
9. The three seeds are three orderings of essentially one corpus plus three adapter
   initialisations, not three elicitations.
10. FRONT8 = {gemma-2-2.6B, llama-3.2-3B, qwen2.5-3B, phi-3.5} × {occupation, CrowS
    socioeconomic}: named in a design file minutes before the first steering run and
    after edit results existed on those cells; the steering strength grid was
    recalibrated once after a first run, on one of the eight cells.
11. No supplementary archive, and no record of an upload, exists in the repository.


---

## 5. Running the 27–32B panel (E4b + E1b)

One ≥ 80 GB card (bf16, one device). About 12 units; resumable — re-run the same command.

```bash
git fetch origin && git checkout r1-rebuttal
bash setup.sh                         # once per box: the pinned stack (torch 2.11.0, transformers 4.57.1, peft 0.20.0)
export HF_TOKEN=hf_...                # gemma-2-27b-it is gated
mkdir -p results && cp -r /path/to/v12big results/v12big   # recommended: the replay guards read it
bash scripts/r1_big.sh                # preflight, then the queue (logs: logs/r1/big/, logs/r1/queue.log)
python3 run_r1.py --config configs/r1/big.yaml --progress
bash scripts/r1_pack.sh r1/BIG        # -> r1_BIG_<utc>.tar.gz (+ logs); send both back
```

On the analysis box: `tar -xzf r1_BIG_<utc>.tar.gz -C results`.

- If `results/t2x/corpora/gemma27b|*.json` and `qwen32b|*.json` from the v12big run are
  still on that box, leave them in place: the units then train on the same corpora and
  the ΔW fingerprint should reproduce. Otherwise they are re-elicited, and a unit whose
  fingerprint differs runs the full α\* search for all three arms by itself.
- Each unit prints `geom_sha=… match=True/False/None` right after training. `None`
  means the reference panel was not found.
- Nothing is printed about removal; the readings are made after the archive is back.

---

## 4 (continued). Readings from the GPU panels

### E4 pass A — `C-rand` under α\*: G1 (2026-11-10 12:10Z complete, `src/r1_e4.py`)

- 30/30 units; ΔW fingerprint reproduced in 30/30; `C-rand` frozen-grid removal
  identical to the published DEC panel in 30/30. Unedited baselines identical in 27/30:
  the three llama units read MMLU one item lower than v12astar did on both splits
  (0.665 vs 0.670; 0.510 vs 0.515) with skew and perplexity bit-identical — the known
  llama MMLU variation (E0 §8.2).
- `C-rand(α*)` = +0.012 [+0.001, +0.023]; `C-a(α*)` = +0.305; `C-ref(α*)` = +0.377.
  q = 0.038. `C-a − C-rand` = +0.293 [+0.160, +0.440]; `C-ref − C-rand` = +0.366
  [+0.211, +0.530].
- **Registered verdict: G1-valid.** At budget-limited scales a random sign field on
  C-a's support, at the same per-tensor scales, removes nothing; what C-a removes there
  is carried by the learned signs. The verdict is the same under the signed and the
  capped removal (q = 0.034, 0.036).
- `C-rand`'s α\* (median 107.6) is not C-a's (median 90.5): equal in 3/30 units. Same
  support and scales, different signs, different budget boundary. Its α\* passes the
  evaluation budget in 43% of runs.

### E2 — ρ_k and the first-order account (`src/r1_e2.py`, statistics from the pass-A units)

- ρ_k at 1%: median 6.64 over the ten cells (6.20–7.61; a Gaussian gives 3.62);
  kurtosis 8.0–14.6; the top 1% of coordinates hold 6.2–7.6% of the L1 mass.
- With the scales as built, ρ_eff = A_ref / A_a: median 6.94 (6.50–8.13).
  r_α = α\*(C-a)/α\*(C-ref): median 4.56 (1.54–6.73). Squares: 48.2 vs 20.8.
- median(ρ_eff / r_α) = 1.64 (0.99–5.09; 3 of 10 cells inside [0.75, 1.33]).
  **Registered reading: E2-N**, first-order failure. The first-order account predicts
  that at α\* the deployed edit should still remove about 1.6 times as much as the
  random-coordinate one; the observed ratio R_ref/R_a at α\* has median 1.02. Removal
  saturates (it cannot exceed the bias that is there), which a linear account does not
  contain. App. H's "does not claim" paragraph has to say this.

### E7b arm `pub` — DPO measured with the corrected harness (2026-11-10 13:14Z, `src/r1_e7b.py`)

- 24/24 units; the baseline was restored bit for bit after unloading in every run.
- Published selection space (merge 0.25–1; cells with < 20 differing pairs scored 0):
  DPO **+0.095 [−0.008, +0.284]**, edit − DPO **+0.187 [+0.049, +0.357]**. The E7a
  reconstruction gave +0.109 and +0.173: confirmed by measurement.
- At the trained scale (merge 1) no run is in budget; perplexity is 1.5–3.2 times the
  unedited model's; 12 of 60 published merge configurations are in budget.
- Not the published selection, reported for completeness: training DPO in every cell
  with ≥ 8 differing pairs gives +0.188 [+0.012, +0.400] (edit − DPO +0.095 [+0.013,
  +0.195]); adding the finer merge ladder {0.0625, 0.125, 0.375} gives +0.210 [+0.026,
  +0.447] (edit − DPO +0.073 [−0.012, +0.178], includes 0).
- **Degeneracy.** From the stored per-item likelihoods: on llama-3.2-3B the mean P("she")
  after DPO is 1.00 on female- and 1.00 on male-stereotyped probe occupations in all
  three seeds from merge 0.25 up; on gemma-2-2.6B it is 1.00 / 0.95–0.97 at the trained
  scale. The occupation skew reads zero for a model that always prefers one pronoun, so
  DPO's two largest "removals" (llama +0.74, gemma +0.62–0.70) are pronoun collapse, not
  the removal of an occupation effect. qwen and phi move to 0.96–0.99 / 0.69–0.99.
- **The same check on the edit** (first units available; all occupation cells once pass
  B is complete): gemma | occupation, `C-ref`, mean P("she") on female- / male-
  stereotyped occupations 0.90 / 0.04 before; 0.82 / 0.73 at α\* = 20.7; 0.77 / 0.76 at
  α = 32. `C-a` at α\* = 98.7: 0.71 / 0.44. `C-rand` (qwen-7B) leaves both where they
  were up to α = 128. The edit closes the gap without collapsing, but it does so mostly
  by raising P("she") on male-stereotyped occupations (mean 0.47 → 0.77): the skew
  measures the dependence on the occupation, not the overall pronoun rate. This needs
  to be reported in the paper.

### E5 — operator 2×2 + ComPEFT under α\* (2026-11-11 04:14Z complete, `src/r1_e5.py`)

Pass B: 150/150 unit-arms, 0 failed. ΔW fingerprint reproduced 30/30; baselines 27/30
(the three llama units, one MMLU item, as in pass A). **The two replays reproduce
v12astar exactly**: α\* equal 30/30 and removal equal 30/30 for both `C-ref` and `C-a`
(max |diff| 0), so the registered reference values and the same-process replays are the
same numbers and no reading depends on the source.

Removal at each arm's own α\* (ten cells, registered definition):

| arm | support | values | removal at α\* | α\* median | eval pass | realized L2 (median) |
|---|---|---|---|---|---|---|
| `C-ref` | top 1% | sign, s_t | +0.377 [+0.219, +0.547] | 20.7 | 57% | 14.7 |
| `O-compeft` | top 1% | sign, global σ | +0.382 [+0.218, +0.557] | 90.5 (in σ units) | 57% | 15.2 |
| `O-trim` | top 1% | full precision | +0.364 [+0.209, +0.528] | 19.0 | 60% | 14.1 |
| `C-a` | random 1% | sign, s_t | +0.305 [+0.167, +0.456] | 90.5 | 60% | 70.2 |
| `O-dare` | random 1% | full precision × 100 | +0.308 [+0.179, +0.455] | 3.08 | 57% | 49.2 |

Registered readings:
- **E5-1, `C-ref − O-trim` = +0.014 [+0.004, +0.026]: the interval excludes 0, so E5-1
  as registered ("includes 0 → one bit is free at matched collateral") does not hold —
  on the favourable side.** The sign edit removes slightly more than the full-precision
  values on the same coordinates (8 of 10 cells; 4% of the removal). Binarization costs
  nothing at matched collateral; the plan had no branch for "better", and the text
  should say "no cost, a small gain" rather than "free".
- **E5-2, `O-trim − O-dare` = +0.056 [−0.021, +0.157]: includes 0 → holds.** In full
  precision too, the choice of support is not detectable at matched collateral under
  the registered definition. Like `C-ref − C-a` it is definition-sensitive: signed
  +0.088 [+0.017, +0.182], capped +0.072 [+0.002, +0.165] (both exclude 0).
- **E5-3, `C-ref − O-compeft` = −0.004 [−0.015, +0.006]: includes 0 → holds.** The
  per-tensor scale is not a point of difference from ComPEFT. At 1% and matched
  collateral the deployed edit and ComPEFT's operator are indistinguishable (same
  realized norm at the boundary, 14.7 vs 15.2; same sign-flip count, 10/30).
- Reported with them: `C-a − O-dare` = −0.003 [−0.047, +0.034] (sign vs full precision
  on the random support: no difference; this answers 5PJm's "C-a is not DARE" — original
  DARE behaves like C-a here); `C-ref − C-a` = +0.073 [−0.036, +0.213] (the published
  number).

What this says. The five operators fall into two groups by **support**, not by value
precision or scale convention: top-magnitude arms 0.364–0.382, random-support arms
0.305–0.308. Within a group nothing is distinguishable except the +0.014 of sign over
full precision. Positioning as the plan fixed it regardless of outcome: the operator is
in the STC/ComPEFT family; the contribution is the decomposition and the protocol.

### E8 — α\* backoff and budget sensitivity (same panel, `src/r1_e8.py`)

| scale | eval pass `C-ref` / `C-a` | MMLU-1000 drop (items) | MMLU-1000 gate pass | removal `C-ref` / `C-a` | Δ_selection |
|---|---|---|---|---|---|
| α\* | 17/30, 18/30 (57%, 60%) | 14.4 / 12.6 | 50% / 50% | +0.377 / +0.305 | +0.073 [−0.036, +0.213] |
| α\*/√2 | 25/30, 23/30 (83%, 77%) | 5.3 / 3.3 | 93% / 97% | +0.342 / +0.266 | +0.076 [−0.021, +0.205] |
| α\*/2 | 28/30, 28/30 (93%, 93%) | 1.5 / 1.1 | 100% / 100% | +0.302 / +0.216 | +0.086 [+0.012, +0.187] |

- **Registered reading: the deployment rule α\*/√2 does NOT meet the 90% target**
  (83% and 77%; pooled 80%). α\*/2 does (93% for both arms, every run passes the
  1,000-item gate, mean loss 1.5 and 1.1 of 1,000 items) and keeps 80% (`C-ref`) and 71%
  (`C-a`) of the removal at α\*. Calling α\*/2 the deployment rule is post hoc.
- **Registered prediction holds**: Δ_selection does not change under backoff
  (α\*/√2 minus α\*: +0.003 [−0.014, +0.021]; α\*/2: +0.013 [−0.033, +0.059]).
- The 200-item evaluation gate is a noisy instrument at the boundary: at α\* half the
  runs lose more than the 1,000-item allowance (mean 13–14 of 1,000 items, about 1.3%),
  so α\* is the boundary of the budget, not a point safely inside it — what E8 was meant
  to show.
- Budget sensitivity (α\* re-selected with a 2- / 8-item allowance): α\* medians
  19.0 / 20.7 / 23.7 (`C-ref`) and 83.0 / 90.5 / 117.4 (`C-a`) for 2 / 4 / 8 items;
  Δ_selection +0.086 [+0.004, +0.186] / +0.073 [−0.036, +0.213] / +0.047 [−0.032, +0.145].
  Under the signed definition all three exclude 0 (+0.118, +0.127, +0.088).
- Not a registered reading, stated so it is not over-read: at α\*/2 and at the 2-item
  budget the registered-definition interval of Δ_selection excludes 0; at α\* and at 8
  items it does not. The point estimate is stable (+0.05 to +0.09); what moves is how
  much overshoot the default definition charges `C-ref` for.

### What removal on the occupation probe consists of (post hoc, `src/r1_pronoun.py`)

Per-item likelihoods of the E5 panel, five occupation cells × three seeds. Mean P("she")
on the 12 female- / 12 male-stereotyped probe occupations; gap = what the skew measures;
base = their mean.

| arm, scale | P(she) f / m | gap | base rate | runs with both > 0.95 |
|---|---|---|---|---|
| unedited | 0.82 / 0.07 | 0.75 | 0.45 | — |
| `C-ref`, fixed grid (the paper's tables) | 0.91 / 0.65 | 0.25 | 0.78 | 0/15 |
| `C-ref`, α\*/2 | 0.92 / 0.60 | 0.32 | 0.76 | — |
| `C-ref`, α\* | 0.86 / 0.71 | 0.15 | 0.79 | 2/15 (qwen-3B) |
| `O-trim`, α\* | 0.87 / 0.70 | 0.17 | 0.79 | 2/15 |
| `O-compeft`, α\* | 0.86 / 0.72 | 0.14 | 0.79 | 2/15 |
| `C-a`, fixed grid | 0.86 / 0.18 | 0.68 | 0.52 | 0/15 |
| `C-a`, α\* | 0.85 / 0.53 | 0.32 | 0.69 | 1/15 (llama) |
| `O-dare`, α\* | 0.86 / 0.57 | 0.30 | 0.72 | 1/15 |

- **The edit closes the occupation gap almost entirely by raising P("she") on
  male-stereotyped occupations (0.07 → 0.65–0.71); the probability on female-stereotyped
  occupations does not come down (0.82 → 0.86–0.91).** The pronoun base rate moves from
  0.45 to 0.78. This holds at the fixed-grid scale of the paper's main tables, not only
  at α\*. Per cell at α\*: gemma 0.74 / 0.62, llama 0.95 / 0.89, qwen-3B 0.99 / 0.95
  (two of three runs above 0.95 on both), phi 0.86 / 0.39, qwen-7B 0.75 / 0.71.
- **This corrects the entry above** ("the edit closes the gap without collapsing"),
  which rested on gemma alone. On llama (0.95 / 0.89) and qwen-3B (0.99 / 0.95) the edit
  at α\* is itself near saturation. DPO goes further on llama (1.00 / 0.98 at a quarter
  of its trained scale, 1.00 / 1.00 above) and less far than the edit on qwen-3B within
  the budget (0.91–0.99 / 0.17–0.55). The difference between the edit and DPO on this
  probe is one of degree, not of kind, and the asterisks in the DPO table apply to the
  edit column as well.
- **It is a property of the contrast, not of the operator**: full precision (`O-trim`)
  and ComPEFT show the same numbers. The cause is in the corpora. Of the pairs in which
  the biased and the debiased item differ, the swap is "he" → "she" in 9–10 of 11–12
  (gemma), 31 of 31 (llama), 38–39 of 49–50 (qwen-3B), 16–17 of 21–22 (phi), 66–67 of
  72–74 (qwen-7B): under the debiased instruction the designer model replaces "he" by
  "she" on male-stereotyped occupations and almost never the reverse. The contrast
  vector, and DPO trained on the same pairs, both learn "more she".
- Consequence for the paper: the occupation skew measures the dependence of the pronoun
  on the occupation, and it is what the paper reports; the paper must also say that on
  this axis the dependence is removed by a base-rate shift, give the table, and not
  describe the result as neutralization. The decomposition claims (which part of ΔW
  carries the effect) are about the edit's effect whatever its direction and are not
  touched; the framing of occupation removal as debiasing is.

### E4 pass C — necessity under α\*: N-hold (2026-11-12 07:20Z complete, `src/r1_e4.py`)

150/150 unit-arms, 0 failed (the pass was interrupted by the VM stop and resumed; §3).
ΔW fingerprint reproduced 30/30; baselines 27/30 (llama, one MMLU item, as before).
Every arm's fixed-grid values replay the published decomposition panel exactly (C-rand,
C-b, C-layershuf, C-bottom 30/30, C-tensorshuf 21/21, max |diff| 0).

Removal at each arm's own α\* (ten cells, registered definition; C-ref +0.377, C-a +0.305):

| arm | what it changes | removal at α\* | share of C-ref | α\* median | eval pass |
|---|---|---|---|---|---|
| `C-b` | signs permuted within each matrix | +0.035 [+0.002, +0.080] | 9.2% | 69.8 | 53% |
| `C-layershuf` | pattern moved to another layer | +0.060 [+0.013, +0.139] | 16.0% | 64.0 | 67% |
| `C-bottom` | smallest 1% instead of largest | +0.004 [−0.004, +0.012] | 0.9% | 128.0 | 53% |
| `C-tensorshuf` (7 cells) | pattern moved to another projection | +0.091 [−0.009, +0.262] | 20.7% | 49.4 | 57% |
| `C-rand` (pass A) | random support, random signs | +0.012 [+0.001, +0.023] | 3.0% | 107.6 | 43% |

- **Registered reading: N-hold = True.** Each of `C-b`, `C-layershuf`, `C-bottom` is at
  most a quarter of `C-ref` on the ten-cell means (9%, 16%, 1%), also under the signed
  and the capped definitions (7–8%, 13–14%, 1%). With G1-valid from pass A, the E4
  verdict is **G1-valid, N-hold**: the necessity results of the fixed grid are not an
  artefact of comparing at scales where the perturbed edits are too small to act. Given
  three to six times the scale of the deployed edit, up to their own budget, they still
  remove little.
- **Not zero, and the text must not say zero.** At the fixed grid these arms were at
  +0.004 or below; at α\* the intervals of `C-b` and `C-layershuf` exclude 0. The means
  are carried by occupation cells (per-cell values, `results/r1/E4/e4_cells.json`):
  `C-b` +0.204 on gemma (28% of that cell's `C-ref`), +0.092 on phi, +0.049 on llama;
  `C-layershuf` **+0.386 on qwen-7B (55% of that cell's `C-ref` +0.703)**, +0.07 on
  llama and on phi. On the five BBQ and StereoSet cells both are within ±0.02 of zero.
  `C-bottom` is within ±0.03 everywhere.
- `C-tensorshuf` (outside N-hold as registered): gemma | occupation keeps +0.584 of
  +0.738 (79%) when the pattern is moved between projections of the same layer; qwen-7B
  +0.074; the other five cells are within ±0.04. At the fixed grid gemma kept about
  half (+0.317); under its own budget it keeps most. The statement "the pattern must
  stay in its projection" does not hold for gemma, as the submitted appendix already
  said for the fixed grid.
- For the paper: the necessity claim is stated for signs-on-coordinates and for the
  smallest coordinates without qualification, and for the layer with the qwen-7B
  exception named.

### E7b registered arms — DPO at lower learning rates: E7-1 (2026-11-12 07:59Z, `src/r1_e7b.py`)

48/48 unit-arms. Design as registered: {lr 1e-5, 5e-5} × merge {1, 2}, three epochs
(15–75 optimizer steps), every cell with at least 8 differing pairs trained (all eight).

- DPO **+0.273 [+0.080, +0.495]**; edit − DPO **+0.009 [−0.049, +0.069]**, includes 0.
  **Registered reading: E7-1** — "DPO removes none" is withdrawn in full and replaced
  by "DPO is comparable". (The edit here is the dense sign-only edit of the comparison
  table, +0.282.)
- Per cell, DPO / edit: gemma | occ +0.694 / +0.674; gemma | crows +0.053 / +0.123;
  llama | occ +0.760 / +0.748; llama | crows +0.010 / +0.023; qwen | occ +0.520 /
  +0.378; qwen | crows +0.002 / −0.047; phi | occ +0.140 / +0.316; phi | crows +0.008 /
  +0.044.
- Post hoc, reports only (`src/r1_e7b_detail.py`, written after the reading was known).
  All 96 configurations are in budget (median ΔMMLU 1 item, median perplexity ratio
  1.015), and 19 of 24 runs select the largest configuration (lr 5e-5 × merge 2): in
  this design DPO is limited by its grid, not by the budget, so +0.273 is a lower bound
  on what DPO can do within the budget. With the as-published recipe (lr 1e-4, 250
  steps; arm `pub`) the same method was outside the budget at its trained scale and
  gave +0.095 when scaled down: the earlier number was a badly tuned baseline on top of
  the evaluation error.
- **Pronoun check for the registered design** (four occupation cells, selected
  configuration): mean P("she") on female- / male-stereotyped occupations 0.83 / 0.08 →
  **0.98 / 0.76**, base-rate shift +0.417 [+0.271, +0.519]; llama is at 1.00 / 1.00 in
  all three runs (collapsed 3/12), qwen-3B 1.00 / 0.87, gemma 0.98 / 0.82, phi 0.95 /
  0.36. The 1% edit at the fixed grid on the same four cells: 0.83 / 0.08 → 0.91 / 0.61,
  shift +0.302 [+0.171, +0.433], collapsed 0/12. Both methods close the gap by raising
  "she"; DPO goes further on every model.
- For the paper: the comparison table's DPO row becomes the registered design; the
  text says that the edit, steering, SentenceDebias and DPO are indistinguishable in
  removal on these cells, that DPO is in budget everywhere and limited by its grid, and
  that the as-published DPO recipe is reported in the appendix as what it was.

### E13 — balanced-contrast control on the occupation axis: E13-confirmed (balanced arm complete 2026-11-12 ~09:40Z, `src/r1_e13.py`)

Balanced arm 15/15 units (the `exo` arm is read when its 15 units are complete).
Mean P("she") on female- / male-stereotyped probe occupations, five occupation cells,
`C-ref`:

| contrast | scale | P(she) f / m | base-rate shift | removal | runs collapsed |
|---|---|---|---|---|---|
| (unedited) | — | 0.82 / 0.07 | — | — | — |
| elicited | fixed grid | 0.91 / 0.65 | +0.329 [+0.201, +0.436] | +0.496 [+0.281, +0.665] | 0/15 |
| elicited | α\*/2 | 0.92 / 0.60 | +0.309 [+0.181, +0.427] | +0.427 | 1/15 |
| elicited | α\* | 0.86 / 0.71 | +0.336 [+0.221, +0.459] | +0.602 | 2/15 |
| **balanced** | fixed grid | **0.54 / 0.37** | **+0.002 [−0.047, +0.061]** | **+0.575 [+0.462, +0.688]** | 0/15 |
| balanced | α\*/2 | 0.55 / 0.37 | +0.013 [−0.040, +0.074] | +0.571 [+0.472, +0.660] | 0/15 |
| balanced | α\* | 0.48 / 0.43 | +0.009 [−0.043, +0.066] | +0.676 [+0.607, +0.742] | 0/15 |

- **Registered reading (balanced, fixed grid): E13-confirmed.** ratio = |shift
  (balanced)| / shift(elicited) = 0.01; elicited − balanced = +0.327 [+0.193, +0.461],
  excludes 0. Not vacuous: the balanced edit removes +0.575 [+0.462, +0.688]. The same
  at α\*/2 (ratio 0.04) and at α\* (ratio 0.03), reported.
- What the balanced edit does: it lowers P("she") on female-stereotyped occupations by
  0.29 and raises it on male-stereotyped ones by 0.29 (fixed grid), so the gap closes
  from both sides and the base rate stays at 0.45. Per cell at the fixed grid: gemma
  0.90 / 0.04 → 0.65 / 0.29, llama 0.84 / 0.08 → 0.43 / 0.40, qwen-3B 0.76 / 0.11 →
  0.61 / 0.49, phi 0.83 / 0.10 → 0.63 / 0.31, qwen-7B 0.80 / 0.05 → 0.35 / 0.35.
- The base-rate shift reported in the pronoun check is therefore a property of the
  one-sided elicited contrast, as attributed, and not of the operator or of the probe:
  the same operator on a contrast with mirrored replacements removes at least as much
  skew without moving the base rate. The sentence in the paper stands, and the remedy
  is no longer "untested".
- Limits, so that this is not over-read: the balanced corpus is a constructed control
  (replacements added by rule to the elicited corpus; for qwen-7B, 9–11 elicited
  replacements were reverted), not what a model returns when asked; whether an
  elicitation prompt can be made to return a balanced contrast is not tested. The
  decomposition and operator panels use the elicited contrast. Other axes have no
  comparable base-rate reading.
