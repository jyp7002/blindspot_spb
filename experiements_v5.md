# experiments_v5.md — Closure: gate the abstract, de-circularize the scope law, localize the crossover

**Companion to `design.md`, `experiments_v2–v4.md`. Consumes the full v4 results
(T4 geometry, T2 cross-designer panel, S/T scope upgrade).** State after v4: the
family bias-direction signature is **durable to 27–32B**, but the self-repair
blind spot **reverses to a self-advantage by 7–9B** (cross − self ≈ −0.05 on
both axes; sharing→self-removal ρ attenuates −0.81 → −0.20). The scope law is
now **consistent-semantic-direction removability** (StereoSet decisive; surface
contrast-repetition refuted by S). O4 splits the small-scale effect into two
forces: pairwise similarity *helps* (+0.30), family identity *hurts* (−0.24,
categorical, cause unidentified).

Purpose of v5: (1) close the one gap that decides which abstract this paper
gets — **"penalty vanished" vs "penalty swamped by self-matching"** (V);
(2) turn the semantic-direction law from a behavioral taxonomy into an
**independent pre-edit predictor** (W); (3) localize the crossover so the
threshold claim is a curve, not a two-point contrast (X); (4) repair the one
invariant violation in the T4 evidence (Y); (5) assemble the paper (CLOSE).

Experiments are lettered V–Y, continuing v4's O–U.

---

## 1. Frozen inputs from v4 (do NOT re-derive)

- **Working model (two forces):** removal = matching benefit (designer signal ↔ target distribution; maximal for self; grows with elicitation capability/scale) − family-categorical penalty (cause unidentified; dominates ≤3B; dominated by 7–9B). All writing must respect this decomposition — no sentence may claim co-encoding *causes* the penalty; the established fact is a family-level correlation (H1, ρ=−0.81, n=8, ≤3B).
- **Terminology change, applied everywhere:** "exogenous ceiling" is retired. Exogenous = **ground-truth reference**. It is a ceiling only on axes where elicitation fails (bbq_Age at 32B: designer +0.381 vs reference +0.074 killed the ceiling reading).
- **Reporting rules carried:** the claim is the mechanism/arc, never the pooled interaction; qwen@attn (+0.188, p=.06) never cited alone; per-arm always, pooled-only banned; frontiers not scalars; both nulls in every condition; strict collateral budget (ΔMMLU ≥ −0.02, ppl ratio ≤ 1.10).
- **Provisional-data rule:** every T2 number is 1-seed until V's seed fill lands; no T2 number enters the paper text before then.
- **Battery honesty:** WinoBias's "PASS" is null-dominated (reference 0.047 ≈ partition null 0.044) — annotated as a hollow cell in the predictions table, starred out of the hit-rate.
- **Invariants:** base models only (one violation — gemma-2-27b-it — repaired by Y); LoRA training ≤14B, ≥27B inference/profiles only; likelihood elicitation only; item floor 40; matched-strength injector band if any injected axis returns; E2 comparison protocol (budget-fail = 0, equal selection space, merge-scale α).
- **R2-bug lesson, promoted to rule:** no hardcoded baseline constants anywhere in analysis code — every comparison anchors on a measured artifact file.

---

## 2. Stages, dependencies, decision logic

```
Stage 0 (cheap, parallel, start now):
    W   EVR → removability regression        (activations only, no edits)
    Y   gemma-2-27b BASE re-profile          (forward passes only)
    V0  granite T2 diagnosis                 (analysis + small probes)
Stage 1:
    V   size-matched T2 cells + seed fill    ──► ABSTRACT LOCKED
Stage 2:
    X   3–4B bridge tier                     ──► arc figure complete
Stage 3:
    CLOSE  paper assembly (skeleton may start at Stage 0; abstract only after V)
```

**Decision matrix (record in PREREGISTRATION.md as each lands):**

- **V-a — same-family-large ≈ cross-large at T2** → the family penalty truly
  vanishes with scale → **Abstract A:** "the self-repair blind spot is a
  small-model phenomenon: family geometry persists to 32B, its behavioral tax
  disappears by 7–9B, and above the threshold a model is its own best debiaser."
- **V-b — same-family-large < cross-large at T2** → the penalty persists and is
  masked by self-matching → **Abstract B:** "two opposing forces govern
  self-repair: a matching benefit that grows with scale and a persistent
  family-categorical penalty; the net sign flips between 4B and 7B."
- Both outcomes are publishable; the experiments and figures are identical
  except for §CLOSE wording. **The abstract is not written until V lands.**
- **W passes** → the scope law graduates to a measurable predictor and K2 is
  effectively revived as the scope section's unifying figure. **W fails** → the
  semantic-direction law stays a behavioral taxonomy with the circularity
  limitation stated plainly.
- **X** → threshold fit only if ≥4 tier points exist; otherwise the arc is
  reported as bracketing (penalty at ≤3.8B, advantage at ≥7B).

---

## V — T2 closure: size-matched same-family cells, seed fill, granite *(Stage 1; the abstract gate)*

**Question.** Is the T2 self-advantage evidence that the family penalty
*vanished*, or does a persistent penalty survive underneath a larger
self-matching benefit? The v4 panel cannot tell: its only same-family non-self
designers were *smaller* siblings (2–3B), so "same-family" was confounded with
"weak designer."

**V1 — the three missing cells.** Size-matched same-family non-self designers,
the only families where they exist:

| target | same-family large designer | cross comparison (existing) |
|---|---|---|
| qwen2.5-7B | **Qwen2.5-14B** | 7–9B cross designers (v4 panel) |
| olmo-2-7B | **OLMo-2-13B** | 〃 |
| falcon-3-7B | **Falcon-3-10B** | 〃 |

Matching band: designer within ~2× target params, same tier-class as the cross
designers. 2 axes (occ_gender + crows_socioeconomic), 3 seeds, attn r16, both
nulls, strict budget.

| criterion | threshold |
|---|---|
| **V1-a** | same-family-large removal ≈ cross-large (paired Δ CI covers 0 across the 3 families × 2 axes) ⇒ Abstract A |
| **V1-b** | same-family-large < cross-large (paired Δ CI excludes 0, negative) ⇒ Abstract B |
| **V1-c** | self > same-family-large in either case (the matching benefit is designer-identity-specific, not family-specific) — pre-registered secondary |

**V2 — seed fill (1 → 3) on the headline T2 cells.** Scoped, not the full
60-row panel: {self, mean-cross composite} × 4 clean targets × 2 axes. The
headline quantities (cross − self ≈ −0.05; ρ attenuation) get CIs; every T2
number in the paper cites V2, not the v4 1-seed run.

**V3 — granite diagnosis.** Why all-nan at T2 (both roles): check ppl baseline,
absolute MMLU, budget bindingness per α, vocab/batch guards. If a config fix
recovers it → add granite to V1/V2. If not → limitation text: "collateral
fragility varies by family," with the diagnostic numbers, and granite is
excluded with reason rather than silently dropped.

---

## W — EVR → removability: de-circularizing the semantic-direction law *(Stage 0; activations only)*

**Problem.** The scope law currently defines "coherent semantic direction" by
its consequence (removability) — a tautology a reviewer will find in one read.

**Design.** For every axis in the programme (~20: occ_gender, gen_fm, BBQ
group-vs-unknown ×4, BBQ group-A-vs-B, StereoSet, WinoBias, floor-passing
CrowS, the S K-sweep variants, M's transformed axes):

1. Pre-edit, on the **probe half only**: per-item minimal-pair activation
   difference vectors at the edit-target layers.
2. PCA per axis → **top-1 explained variance ratio (EVR)** = the coherence
   score. Layer aggregation (mean of top-k vs max) pre-registered before
   computing; one choice, no sweeping post hoc.
3. Regress ground-truth-reference removal fraction on EVR across axes.

**Pre-registered per-axis predictions (written before computing):** occ_gender,
gen_fm, BBQ-unknown, StereoSet → high EVR (StereoSet is the sharp one: high
coherence *despite* per-item attribute variation is exactly what the
semantic-direction reading claims). CrowS at every K, BBQ group-A-vs-B, M's
templatized axes → low EVR (the S sweep becomes a within-content EVR gradient).
WinoBias → reported, unregistered (hollow cell).

| criterion | threshold |
|---|---|
| **W1** | slope(removal ~ EVR) > 0, CI excludes 0, across ~20 axes |
| **W2** | removable vs unremovable axes separate on EVR with a margin (calibrate the cut on the first half of axes, freeze, test on the rest) |
| **W3** | per-axis prediction hit-rate reported as a table, hits and misses alike |

**Circularity guard.** EVR is computed pre-edit, on the probe half, with the
aggregation rule frozen first. No axis's EVR is recomputed after its removal
number is known.

---

## X — 3–4B bridge tier: localize the crossover *(Stage 2)*

**Setup fact.** phi-3.5-mini (3.8B) already anchors "penalty present at ~4B"
(strongest T1 arm, +0.362, p=2e−11); the v4 T2 panel anchors "advantage at
7–9B." The crossover sits in (4, 7)B and is currently a blank.

**Design.** Cross-designer protocol (self + ≥2 size-matched cross designers,
both nulls) on 3B-class targets: **Qwen2.5-3B, Llama-3.2-3B, Falcon-3-3B**.
2 axes × 3 seeds, attn r16. Optional refinement if the bracket needs
splitting: one ~6B point (Yi-1.5-6B base — a new family, so full onboarding
gates apply: profiles, pre-bias, polarity, floor) — include only if the 3B and
7B points straddle zero with room.

| criterion | threshold |
|---|---|
| **X1** | per-tier (cross − self) with CIs at 1–2B (v3), 3B (new), 3.8B (phi, existing), 7–9B (V2) — the arc figure |
| **X2** | threshold fit (sign-change point with CI) only if ≥4 tier points; otherwise report bracketing honestly |

---

## Y — gemma-2-27b BASE re-profile *(Stage 0; forward only)*

**Problem.** Half the T4 durability claim rests on gemma-2-27b-**it** — an
instruct model, violating the base-only invariant (RLHF can suppress expressed
bias and distort profiles).

**Design.** Re-profile gemma-2-27b **base** on the 6-axis common set; recompute
Δd at T4 with base-only models.

| criterion | threshold |
|---|---|
| **Y1** | Δd durability holds base-only (same-family − cross-family CI excludes 0 at 27–32B) — this is the number the paper cites |
| **Y2 (bonus, only if Y1 passes)** | it-vs-base signature shift quantified (27b-it ↔ 27b-base profile correlation, each ↔ gemma-9b base). If the signature survives instruct tuning, one sentence: "RLHF does not erase the family signature" — written only after Y1, never instead of it |

---

## CLOSE — paper assembly *(Stage 3; skeleton may start now, abstract after V)*

- **Narrative, three pillars:** (1) *anti-convergence*: family bias-direction
  geometry persists to 27–32B (O1-T4 + Y); (2) *the threshold arc*: the
  geometry's behavioral tax (ρ=−0.81 at ≤3B, family-categorical per G/O4)
  reverses to self-advantage by 7–9B (T2 + V + X), with the two-force model as
  the interpretive frame; (3) *removability*: consistent-semantic-direction law
  (M/S/T + W), with the prediction battery and upheld R2 as the falsifiability
  showcase.
- **Practical flip, stated as a positive result:** from ~7B up, a model is its
  own best debiaser, and the edit is a sign-only ~1 MB patch that beats
  full-precision task vectors at matched collateral (E2).
- **Terminology sweep:** every "exogenous ceiling" → "ground-truth reference";
  every causal "co-encoding → penalty" → "family-level correlation (cause
  unidentified)."
- **Tables:** predictions-vs-outcomes battery (WinoBias starred hollow); E2
  matched-collateral suite; the three protocol errors as a methods appendix;
  bugs-that-would-have-corrupted appendix (device_map/dispatch, stale-constant,
  free(), tokenizer, injection-strength — the record is a contribution).
- **Limitations, verbatim candidates:** categorical penalty cause unidentified;
  T4 = 2 families, 70B deliberately unrun; granite fragility (V3 outcome);
  1-seed T2 history (superseded by V2); WinoBias hollow; injection = attenuated
  instrument.
- **Related work:** Bankai defensive paragraph (framing-a cut stands); Lu
  et al./judge-similarity positioning carried from the earlier scoping.

---

## Threats & practical notes

- **"Size-matched" definition:** designer within ~2× target params AND same
  tier-class as the cross designers — write the band into the run config, not
  the analysis.
- **Seed-fill scope creep:** V2 is scoped to headline cells by design; the full
  60-row × 3-seed panel is not required for any claim and is not run.
- **Gemma-2-27b base availability/licensing:** verify checkpoint access before
  scheduling Y; if only -it is accessible, the T4 gemma point is demoted to an
  appendix with the confound stated.
- **Yi onboarding (X optional):** full gates or it doesn't enter; a bridge
  point with an ungated axis is worse than no point.
- **Compute:** W and Y are forward-only (hours); V ≈ 3 families × 2 designers ×
  2 axes × 3 seeds + scoped seed fill (the largest v5 item, still ≤14B
  training); X ≈ 3 targets × 3 designers × 2 axes × 3 seeds. Nothing exceeds
  the 96 GB single-GPU workflow.

---

## Z — Freeze-before-running checklist (v5 additions)

- [ ] Both abstracts (A and B) drafted and parked BEFORE V runs; neither edited after V lands — the data picks, the text doesn't bend.
- [ ] W's layer-aggregation rule and per-axis EVR predictions registered before any activation is extracted.
- [ ] W2 cut calibrated on the first half of axes, frozen, tested on the rest.
- [ ] Size-matching band written into run configs; V1 designers verified in-band.
- [ ] V2 scope (cells × axes × seeds) registered; no silent expansion.
- [ ] V3 granite verdict recorded (recovered vs excluded-with-reason) before CLOSE.
- [ ] Y1 base checkpoint verified accessible; -it profiles quarantined from the cited Δd.
- [ ] No hardcoded baseline constants in any analysis script (measured artifact files only).
- [ ] No T2 number in paper text cites the 1-seed run once V2 exists.
- [ ] Resume keys include {tier, arm, dataset, axis, designer_size_class, seed}.