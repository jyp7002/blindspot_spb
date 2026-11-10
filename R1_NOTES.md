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
