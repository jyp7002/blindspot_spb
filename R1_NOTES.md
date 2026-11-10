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
