# experiments_v4.md — Scale, dose-response, and closure

**Companion to `design.md`, `experiments_v2.md`, `experiments_v3.md`. Consumes
Results v3.** State after v3: the mechanism is nailed (H1: within-family
co-encoding → self-penalty, ρ=−0.81, p=0.015, n=8, CI excludes 0), the pooled
interaction is config-fragile and demoted, the self-penalty is a
family-aggregate property (G), the scope boundary is CONTRAST-REPETITION
(M refined + CONFIRM-1), and removability is a third factor orthogonal to the
elicitation gap (K1).

Purpose of v4: (1) the **scale phase** — the only two questions that can change
the science are answerable only at scale (representational convergence vs
family signature; removability vs scale), plus the one tier the paper needs for
defensibility; (2) the **clean contrast-repetition dose-response** v3's
CONFIRM-2 could not deliver; (3) an **out-of-sample prediction battery** that
turns the scope law into forecasts before data is seen; (4) **paper closure**.

Experiments are lettered O–U, continuing v3's G–N. O/P/Q/R correspond to the
P1–P4 stack from the scale discussion. Priorities are called **Stage 0–3** to
avoid colliding with experiment letters.

---

## 1. Frozen inputs from v3 (do NOT re-derive)

- **Edit config: attn (q,k,v,o), rank 16, binary sign + per-tensor scale** — frozen as the most *capable* config, with the standing caveat that the pooled interaction is q/v-inflated and null at attn. **Reporting rule: the claim is the mechanism (sharing→self-penalty, per-arm + H1 regression), never a config-independent pooled interaction.** qwen@attn (+0.188, p=.06) is never cited alone.
- **Scope law: contrast-repetition.** Removable ⇔ one discriminative contrast repeats across items (occ_gender he/she; BBQ toward-unknown). Frame-overlap refuted; bias magnitude not the driver. Standing caveats to carry into every write-up: endogenous > exogenous on BBQ-unknown; BBQ-Race partition null is high (0.23); part of removal on repeated-contrast axes is "any direction in the contrast subspace."
- **Self-penalty is family-AGGREGATE** (G): pairwise d_ij does not predict removal (r=+0.02 over cross pairs). Do not build anything on pairwise d without O4's multi-axis re-test.
- **Removability is a third factor** (K1): elicitation gaps are normal on CrowS; the boundary is the exogenous ceiling. Elicitation-based fixes to the boundary are not on the table.
- **Hygiene invariants:** items-per-half floor 40; polarity unit test per new axis/dataset; both nulls (data-partition + sign-shuffle) in every condition; tokenizer-agnostic continuation scoring; base models only (instruct = confound, see §Threats); matched-strength injector band 0.51–0.78 for any injected axis; likelihood elicitation only; E2 comparison protocol (budget-fail = 0, equal selection space, merge-scale α); strict collateral budget (ΔMMLU ≥ −0.02, ppl ratio ≤ 1.10); frontiers not scalars; per-arm always, pooled-only banned.
- **Compute policy:** LoRA training capped at ≤14B targets. Models ≥27B are inference/profiles only (a designer never needs training — it supplies the elicited contrast; the edit trains on the target).
- **E4 (in-place quantized byte-patch): CUT.** One defensive Bankai paragraph in related work. Not revisited in v4.

---

## 2. Model tiers

| tier | role | models |
|---|---|---|
| **T1 (existing)** | targets + designers, full pipeline | the 8-family ~0.4–3B set from v3 |
| **T2 (7–9B)** | targets + designers, full pipeline | Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9b, OLMo-2-7B, Falcon-3-7B, Granite-3-8B, Phi-3-small-7B — **n=7** (SmolLM2 caps at 1.7B → T1-only) |
| **T3 (10–14B)** | targets (edit-trainable ceiling) + designers | Qwen2.5-14B, OLMo-2-13B, Phi-3-medium-14B, Falcon-3-10B |
| **T4 (27–72B)** | **profiles + designer inference ONLY** | Qwen2.5-{32B,72B}, Llama-3.1-70B, Gemma-2-27B |
| bridge (appendix) | one instruct pair | e.g. Qwen2.5-7B vs -7B-Instruct, profiles + one removal cell |

---

## 3. Stages, dependencies, decision logic

```
Stage 0 (disk + design, ~0 GPU):
    O0  harvest existing ≥7B sibling profiles already computed for sharing
    O4  multi-axis aggregate pairwise d vs removal on the existing 5×5
    T0  PRE-REGISTER the prediction battery (before any onboarding)
    S0  contrast-K axis construction + gates
Stage 1: O  full multi-scale profiles (T1→T4)   ──feeds──►  P
         P  H1 replication at T2 (the defensibility tier)
Stage 2: Q  removability vs scale (T2/T3 targets)
         R  large-designer spot cells (T4 designers, inference-only)
         S  contrast-K dose-response (T1)
Stage 3: T  prediction battery (WinoBias / HolisticBias / StereoSet / BOLD)
         U  paper closure
```

**Decision matrix (record outcomes in PREREGISTRATION.md as they land):**

- **O: family signature persists to T4** → new section: "the law survives representational convergence" (strong against the convergence narrative). **O: Δd shrinks with scale** → the law is scale-scoped; P locates where it fades; still publishable, honestly framed.
- **P: H1 replicates at T2** → two-tier law; the toy-model objection is closed. **P fails** → law scoped to ≤3B, with O's convergence curve as the candidate explanation.
- **Q: exo ceiling rises materially with scale** → contrast-repetition boundary is partly a small-model capacity artifact; soften scope wording. **Q flat** → boundary is structural in the bias, scale-robust.
- **R2 (72B cannot crack CrowS) holds** → falsifiable-prediction win for the three-factor model. **R2 fails** → model revision: designer scale substitutes for removability; K1's reading must be amended.
- **S slope negative** → dose-response secured; the scope law graduates from binary to graded. **T battery** hit-rate is reported as a table regardless of outcome; StereoSet's cell *defines* which reading of contrast-repetition is correct.

---

## O — Multi-scale bias-direction profiles: convergence vs family signature *(Stage 0–1; the cheapest high-stakes measurement)*

**Question.** Does the family signature in bias-direction structure survive
scale, or do families converge (behavioral analog of the platonic-convergence
claim)? If cross-family d approaches same-family d with scale, the same/cross
distinction — and with it the blind spot — dissolves at scale.

**Design.** Item-space signed profiles (the C instrument) for **every model in
T1–T4** on: occ_gender, the floor-passing CrowS axes, and the four BBQ-unknown
axes. Pure forward passes; vLLM; no edits.

- **O0 (first, free):** ≥7B sibling profiles (Gemma-9b, OLMo-13B, …) were
  already computed for v3's sharing values — harvest from disk before any new
  run.
- **O1 (primary):** per tier t, `Δd(t) = mean same-family d(t) − mean
  cross-family d(t)`, with bootstrap CIs; regression of Δd on log-scale.
  Pre-register both directions as publishable.
- **O2 (H1 hygiene):** regress within-family sharing on sibling size-ratio;
  recompute H1 on adjacent-size sibling pairs only. If sharing was partly a
  size-gap artifact, this cleans the paper's central regressor.
- **O3 (cross-scale d):** same-family d between tiers (e.g., Qwen-1.5B vs
  Qwen-72B): does lineage identity dominate scale identity? (Also feeds P's
  designer choices.)
- **O4 (G revisit, free):** on the existing 5×5, replace single-axis pairwise
  d_ij with the **multi-axis aggregate** (mean profile correlation over all 13
  C axes) and re-run the G regression. If aggregate-pairwise predicts where
  single-axis-pairwise did not, the "family-aggregate, not pairwise" mystery
  partially resolves — and G's failure becomes a measurement-bandwidth story.

| criterion | threshold |
|---|---|
| **O1** | Δd(t) > 0 with CI excluding 0 at every tier (signature persists), OR a significant negative slope of Δd on log-scale (convergence) — pre-registered as a two-sided question |
| **O2** | H1 ρ survives (CI excludes 0) on size-matched sibling pairs |
| **O4** | aggregate-pairwise coefficient negative with CI excluding 0 in the G model ⇒ upgrade mechanism wording; else keep "family-aggregate" as stated |

---

## P — H1 at the 7–9B tier *(Stage 1; the defensibility item)*

**Question.** Does the central law — within-family co-encoding predicts
self-removal — replicate on T2 targets? Without this, the paper claims a
"family-lineage law" from 1–3B toys.

**Design.** For each of the **7 T2 families**: within-family sharing (from O)
+ self-designer removal + ≥1 cross-designer removal, on occ_gender + 2
BBQ-unknown axes (Age, Religion — the clean removable ones). 3 seeds; both
nulls; strict budget; attn r16. **Scope note:** inherited axes only — no
injection at T2 (the 2×2 is not the claim; H1 is). Designers at T2 are drawn
from T2/T3 (cross) and the family's own T1/T2 siblings (self/same).

| criterion | threshold |
|---|---|
| **P1** | Spearman ρ(sharing, self-removal) at T2: CI excludes 0, n=7 |
| **P2** | joint model `removal ~ sharing × tier + (1|family)`: interaction reported with CI — ns ⇒ scale-stable law; significant ⇒ quantified attenuation, interpreted with O1 |
| **P3** | per-arm blind-spot direction (cross − same on inherited) tracks sharing rank at T2, as it did at T1 |

---

## Q — Removability vs scale *(Stage 2)*

**Question.** Does the contrast-repetition boundary soften as models grow —
i.e., do naturalistic biases become more linearly consolidated (single-
direction-editable) at scale?

**Design.** **Exogenous ceilings only** (no designer sweeps): on T2 and one T3
target, measure the exo ceiling at the strict budget for (a) the floor-passing
CrowS axes (unremovable at T1), (b) BBQ group-A-vs-B (unremovable at T1),
(c) BBQ-unknown + occ_gender (removable controls). Same α sweep, both budgets
reported.

| criterion | threshold |
|---|---|
| **Q1** | softening = ceiling(T2)/ceiling(T1) ≥ 2× with non-overlapping CIs on ≥2 previously-unremovable axes (calibrate the multiplier on the first axis, then freeze) |
| **Q2** | removable controls stay removable at T2 (sanity) |

Q1 positive → scope wording becomes "contrast-repetition governs removability
at small scale; consolidation relaxes it by ~X at 7–14B." Q1 negative → the
boundary is a property of the bias structure, not model capacity.

---

## R — Large-designer spot cells *(Stage 2; inference-only)*

**Question.** Does designer *scale* substitute for cross-family-ness — and can
it substitute for *removability*?

**Design.** Designers: Qwen2.5-{7B, 32B, 72B} and Llama-3.1-70B, elicitation
by inference only; edits trained on two fixed targets (qwen1.5b, phi-3.5).
Cells: {each designer} × {occ_gender, BBQ-unknown-Age, one floor-passing CrowS
axis} × 3 seeds, plus the matched ~3B cross designer as the comparison point.
Also extend F's designer-quality regression with the new scale span:
elicitation gap and removal vs log designer size.

| criterion | threshold |
|---|---|
| **R1** | designer-scale curve: does a 70B-class cross designer beat the matched 3B cross designer on removable axes? (two-sided; extends F's null band, which was 0.36–3B only) |
| **R2 (pre-registered prediction)** | **a 70B-class designer does NOT exceed the exogenous ceiling on the CrowS axis** — the three-factor model says the boundary is removability, so no amount of designer quality crosses it. PASS = falsifiable-prediction win to feature in the paper. FAIL = amend K1's reading: elicitation at scale substitutes for ground truth |

---

## S — Contrast-K dose-response *(Stage 2; the clean version of CONFIRM-2)*

**Question.** Is removability monotone in the number of distinct discriminative
contrasts, holding bias strength, content pool, and item count fixed?

**Design.** From CrowS-derived material: single-contrast sets (one group pair ×
many polarity-validated traits — fixing v3's mixed-polarity confound) pooled
into axes with **K ∈ {1, 2, 5, 10, 20}** distinct group contrasts. Matched item
counts across K; pre-bias floor per variant; polarity unit test; item floor.
Natural bias only — no injection (v3: injection installs a partly-coherent
direction and attenuates the effect).

**Outcome.** Exogenous ceiling vs K (primary). Designer sweep only on variants
whose ceiling clears the gate.

| criterion | threshold |
|---|---|
| **S1** | ceiling declines monotonically in K: trend test (Page / regression on log K), CI excludes 0 |
| **S2** | K=1 variants clear the removability gate; K=20 variants do not (the qualitative flip reproduced under matched conditions) |

S1+S2 = contrast-repetition graduates from a binary scope statement to a
graded, mechanistic law — the scope section's headline figure.

---

## T — Out-of-sample prediction battery *(Stage 3; pre-register at Stage 0)*

**The scope law now makes falsifiable per-dataset predictions. Register them
BEFORE onboarding (T0), then run.**

| dataset | construction | pre-registered prediction |
|---|---|---|
| WinoBias/Winogender | coreference pronoun likelihood (he/she repeats) | **REMOVABLE** |
| HolisticBias | per-descriptor axes (one descriptor pair × many contexts) | **REMOVABLE** (per axis) |
| StereoSet intrasentence | group fixed in context; attribute term varies per item | **the definitional test**: token-contrast reading ⇒ NOT removable; semantic-direction reading ⇒ partially removable. Register both; the outcome *defines* what "contrast" means in the law |
| CrowS (floor-passing) | as in v3 | NOT removable (negative control, known) |
| BBQ group-vs-unknown | as in v3 | REMOVABLE (positive control, known) |
| BOLD | **outcome-only**: generation-level bias (sentiment/toxicity gaps) + fluency, measured on the best final edits | probe-level removal transfers to generation with attenuation (direction registered, magnitude exploratory) |

Standard onboarding gates apply to every row. **Deliverable: a predictions-vs-
outcomes table in the paper** — hit-rate is reported whatever it is. The
StereoSet cell feeds back into S's interpretation.

---

## U — Paper closure *(Stage 3)*

- **Mechanism section:** H1 at two tiers (v3 n=8 + P), per-arm sharing-tracking, O's convergence curve, O4's aggregate-vs-pairwise resolution. The pooled interaction appears only inside the config-fragility subsection (L), framed as the discovery path.
- **Scope section:** contrast-repetition (M + CONFIRM-1) + S dose-response + T battery table. Carry the standing caveats verbatim (endo>exo on BBQ-unknown; Race null; contrast-subspace share of removal).
- **Efficiency/systems:** E2 suite + capacity sweep, unchanged from v2/v3. Bankai = one related-work paragraph.
- **Limitations, stated plainly:** family-aggregate mechanism unexplained if O4 fails; T1-heavy evidence base if P fails; single inherited natural axis family (occ_gender + BBQ-unknown) carries the mechanism tier; injection is an attenuated instrument.
- PREREGISTRATION.md updated at every stage gate; all thresholds calibrated-then-frozen as in v2/v3.

---

## Threats & practical notes

- **Instruct confound:** RLHF-tuned models carry vendor debiasing that contaminates *inherited* bias — base models only for all claims; the single instruct bridge pair is appendix-only.
- **BBQ-unknown bias may shrink at T2/T3** (larger bases prefer "unknown" more): re-run the axis gate per tier before P/Q; do not assume T1 gates transfer.
- **Collateral budget semantics at scale:** keep the same ΔMMLU/ppl-ratio budget; report absolute MMLU per tier so budget bindingness is visible.
- **Profile comparability across vocabularies** (152k vs 256k vs 32k): tokenizer-agnostic continuation scoring everywhere (Phi lesson); spot-audit 20 items per new model.
- **Injector at T2:** not needed (P skips injection); if any injected axis is later required at scale, the band must be re-tuned per tier.
- **Compute:** O = forward passes only (T4 via multi-GPU vLLM, days); P ≈ one v3-H-scale run at T2 sizes; Q ceilings ≈ small; R = inference-only designers + T1/T2-target training; S = T1-only. Nothing in v4 trains above 14B.

---

## Z — Freeze-before-running checklist (v4 additions)

- [ ] T0 prediction battery registered (per-dataset predictions + both StereoSet readings) BEFORE any T-dataset onboarding.
- [ ] O0 disk harvest done before scheduling new profile runs.
- [ ] Δd two-sided question and Q1 softening multiplier calibrated-then-frozen.
- [ ] Per-tier axis re-gating (BBQ-unknown, CrowS floors) before P and Q.
- [ ] S variants pass matched-item-count, polarity, pre-bias, and floor gates at every K.
- [ ] R designers are inference-only; no ≥27B training jobs schedulable.
- [ ] Base-model-only invariant enforced in the run harness (instruct checkpoints rejected outside the bridge pair).
- [ ] Resume keys include {tier, arm, dataset, axis, K}.
- [ ] Stage-gate outcomes recorded in PREREGISTRATION.md before the next stage starts.