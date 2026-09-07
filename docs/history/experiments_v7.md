# experiments_v7.md — Completing "The binary debiaser"

**Companion to `design.md`, `experiments_v2–v6.md`, and the method-results
draft.** Consumes the reframe: the science line is closed (H2 interaction
+0.004 [−0.042, +0.051]; ten failed operationalizations; v6 stop rule fired),
and the paper is now method-first. This document lists what the method paper
still needs, ordered by information value, plus the hygiene items that make the
draft submission-safe. No calendar; sequencing is by decision dependency only.

**The one-sentence gap:** a method paper without trained baselines is not
submittable, and among them the steering baseline decides what the paper's
contribution *is*.

---

## 1. Frozen inputs (carried; do NOT re-derive)

- **Frozen protocol everywhere:** attn (q,k,v,o) r16, per-tensor binary
  sign edit, likelihood elicitation, strict budget (ΔMMLU ≥ −0.02,
  ppl ratio ≤ 1.10), 3 seeds, nan→0 with counts, bootstrap over target×axis
  cells, both nulls, removal on naturalistic axes cited gap-over-partition.
- **Untagged-config citation ban:** no number measured off the frozen protocol
  enters the paper (the 108.8 % rule). This is why old E2 results cannot be
  borrowed and BL-T must re-run.
- **Always-on comparison rule (from the prompt baseline):** any competing
  intervention is evaluated with the intervention *active* during bias probes
  AND MMLU AND perplexity — collateral is measured under deployment conditions,
  identically for every method.
- **Equal-selection-space rule:** every method gets the same number of
  configurations to maximize over (the v3 E2 lesson). Applies to steering's
  layer/strength grid exactly as to the edit's α grid.
- **Endogeneity boundary:** every baseline's signal, where applicable, comes
  from the same endogenous elicited corpora; ground-truth arms are
  calibration-only labels, never a method.
- **Merge-scale α for trained baselines; PCGU gets a real optimization
  budget** (both v3 protocol-error lessons).
- Base models only; the single instruct arm (COL) is a scoped, labeled
  exception. Training ≤14B. No hardcoded baselines; every α persisted to
  `alpha_trace.jsonl`.

---

## 2. Decision map

```
BL-S (steering fork)  ──►  defines the paper's contribution claim
BL-T (trained suite)  ──►  positions the edit on the frontier
AX   (axis breadth)   ──►  answers "one-axis method"
SC7  (7–9B arm)       ──►  hardens the no-cost-at-scale claim
COL  (IFEval arm)     ──►  collateral generality (appendix)
SUP / REC / SPLIT     ──►  submission hygiene + paper-shape decision
ACQ                   ──►  drop-or-run (default: drop)
```

**Pre-registered fork at BL-S (write before running):**
- **Edit > steering** at matched collateral → contribution stands as stated:
  a sign-only weight edit that removes bias under a deployment-honest budget.
- **Edit ≈ steering** → contribution reframes to what steering cannot offer:
  a **persistent, mergeable, sub-MB, inference-cost-free** patch (99 % sparse,
  scalar-scale) + the budget-honest protocol + contrast_gap as the corpus-
  selection rule. The paper survives; the removal claim is shared.
- **Steering > edit** → the method paper becomes a protocol-and-analysis paper
  (budget machinery, nulls, contrast_gap, sign-structure finding) with
  steering recommended for removal. Pre-registering this outcome is what makes
  the comparison credible.

---

## BL-S — Steering baseline *(the gate; run first)*

**Why decisive.** Steering uses the **same elicited signal with no weight
edit**: mean activation difference between the biased/debiased halves of the
identical corpora, added to the residual stream at inference. If it matches
the edit, "the edit" was never the contribution — the signal was.

**Design.**
- Vector: per-cell, from the same cached endogenous corpora (contrast of mean
  activations at candidate layers). No new elicitation.
- Grid: layers × strengths, **sized to match the edit's α grid** (equal
  selection space). Negative direction = debias, same sign convention.
- Evaluation: always-on rule — steering active during probes, MMLU, ppl.
  Strict budget, 3 seeds, cell bootstrap.
- **Nulls, adapted and mandatory:** (i) random-direction steering at matched
  norm per layer (sign-shuffle analog); (ii) partition-corpus steering vector
  (partition analog).
- Cells: the paper's headline cells — {phi, qwen, gemma, llama} × {occ_gender,
  crows_socioeconomic} at ≤3.8B; {qwen-7B, llama-8B} at 7–9B.

| criterion | threshold |
|---|---|
| **BL-S1** | paired (edit − steering) in-budget removal per cell; pooled CI reported; the fork above executes on the sign of the CI |
| **BL-S2** | steering's collateral profile reported (ppl under always-on steering is the expected failure mode — measured, not assumed) |
| **BL-S3** | steering beats its own two nulls (else its removal is not signal-borne and the comparison is vacuous) |

---

## BL-T — Trained-baseline suite on the frozen protocol

DPO, PCGU, FairLoRA, full-precision task vector (already the C3 fp arm), plus
an **STE-trained-signs arm** as the edit's own variant (the old "STE best"
figure is superseded-config and must be re-established or retired).

**Design.** Same cells as BL-S; merge-scale α (≤1) for DPO/FairLoRA; PCGU with
a stated, adequate optimization budget; budget-fail scores 0; equal selection
space (each method: the same config count); all trained from the same
endogenous corpora where the method admits it (DPO pairs built from the
elicited contrast; deviations documented).

| criterion | threshold |
|---|---|
| **BL-T1** | frontier table at matched collateral: in-budget removal per method per cell, operable-fraction column, nulls where applicable |
| **BL-T2** | STE vs sign(Δ_fp) on identical cells decides the paper's recommended variant; the winner becomes the default, the loser an ablation row |

---

## AX — Axis breadth port

**Problem.** The method's large effects currently live on occ_gender; §7's new
CrowS axes average +0.012. BBQ group-vs-unknown (4 axes) and StereoSet
intrasentence were removable on the untagged config — port them to the frozen
protocol.

**Design.** Full per-axis onboarding as new axes (do not assume old gates
transfer): polarity unit test, item floor 40, disjoint probe/edit halves,
measured pre_skew per target, both nulls, 3 seeds. Targets: ≥2 small + ≥1 7–9B.
Winogender optional, flagged (null-dominated risk from the v4 battery).

| criterion | threshold |
|---|---|
| **AX1** | ≥2 ported axes show in-budget removal above BOTH nulls on ≥2 targets → headline table grows from 2 strong axes to ≥4 |
| **AX2** | the breadth table reports pre_skew next to removal on every axis — the same table carries §7's honest frame ("removes bias where measurable bias exists") as a relationship, not an excuse |

AX1 failure → §7's limitation stands as written; the paper remains viable but
is submitted as a narrow-axis method with the limitation front-loaded.

---

## SC7 — Strengthen the 7–9B arm

Current C3 scale arm: n = 12, self-designer only; Exp-N nulls/E2-at-scale:
one target. 

**Design.** Add cross and same-family-large designers at {qwen-7B, llama-8B}
for the paired fp-vs-binary comparison (identical-dW pairing discipline), and
extend the Exp-N null + fp comparison to llama-8B (the planned second family).

| criterion | threshold |
|---|---|
| **SC7-1** | 7–9B retention estimate from ≥2 targets × ≥2 designer classes, CI reported; the "no cost at scale" sentence cites this, not n=12 |
| **SC7-2** | sign-shuffle ≈ 0 replicates on the second 7–9B family |

---

## COL — Instruct-collateral arm *(appendix; scoped exception to base-only)*

One instruct target (e.g., qwen2.5-7B-instruct): apply the frozen-protocol
edit; report IFEval alongside MMLU and ppl. Purpose: the budget's collateral
metrics are base-centric; a deployment reader will ask about instruction
following. One pair, appendix, clearly labeled as the exception.

---

## ACQ — drop-or-run

Default: **drop** the inherited/acquired framing from the method paper
entirely (the 2×2 belongs to the retired science line; the acquired arm is 5
cells on one axis and adds nothing to the method claims). Run a second
injected axis ONLY if any inherited/acquired sentence survives editing.

---

## SUP — Superseded-figures quarantine table *(appendix)*

One table, four rows minimum, each: old figure → config it came from →
frozen-protocol replacement → where the old figure appeared.

| old | origin | frozen replacement |
|---|---|---|
| retention 108.8 % | untagged v2-era module set | 86.6 % (≤3.8B) / 125.6 % "no cost" (7–9B) |
| "binary beats fp (gentler)" | v2 E2 config | 86.6 % deficit at ≤3.8B (CI-significant); no cost at 7–9B |
| "scalar scale costs ~18 %" | v2 ablation | within noise of per-tensor at both tiers |
| "99 % sparsity retains 62 %" | v2 ablation | 99 % sparsity costs nothing measurable |

Rationale line in the appendix: all early figures were measured on a weaker
edit configuration (q/v r16) and an earlier protocol; the frozen-protocol
numbers supersede them wholesale. This table is the shield against anyone
diffing the paper against earlier program artifacts.

---

## REC — Two reconciliation sentences *(must appear in the draft)*

1. **contrast_gap vs K1.** contrast_gap predicts removal at the
   cell/corpus level (+0.507, n = 504) while the axis-level elicitation-gap
   regression (K1) was null (r = −0.14, n = 15 axes): the predictor operates
   within and across cells, not on axis aggregates. One sentence, or a
   reviewer files it as a self-contradiction.
2. **pre_skew frame vs the retired taxonomy.** The method paper adopts §7's
   functional frame (removal where measurable bias exists; pre_skew reported
   per axis) and does **not** use the v4 "consistent-semantic-direction"
   taxonomy. The taxonomy, if published at all, lives in the companion (SPLIT)
   with its three refuted geometric predictors. No half-references.

---

## SPLIT — Paper-shape decision

**Default recommendation: two papers.**

- **Method paper (this document's target):** C1–C6, prompt baseline, BL-S/BL-T
  frontier, AX breadth, SC7, contrast_gap (deployment rule), H1 as a
  family-level *self-repair capacity* analysis, the 7–9B tie as the
  "self-elicitation suffices at scale" deployment argument, SUP/REC hygiene.
- **Companion short (findings/negative-results venue):** the null law —
  H2 interaction +0.004 with ten refuted operationalizations (a conjecture
  others will naturally form from the transmission literature), the durable
  anti-convergence geometry (Δd_base = +0.184), the bridge heterogeneity, and
  the retired removability taxonomy with W/W′/W″.

Alternative (single paper): compress the companion content into a two-page
§"What the method is not explained by" — legitimate, but it dilutes the method
claims and re-imports the science line's framing burden. Record the choice in
`PREREGISTRATION.md` before the submission draft freezes; the experiments above
are identical under both shapes.

---

## Threats & discipline

- **Steering fairness:** selection space matched to the edit; norm-matched
  random-direction null (an unmatched-norm null is a strawman); collateral
  measured always-on. A steering win under these rules is a real result, not a
  protocol artifact — that is the point of pre-registering the fork.
- **DPO/FairLoRA scale, PCGU budget:** the three v3 protocol errors are the
  known failure modes; their fixes are frozen inputs here, not rediscoveries.
- **AX gates do not transfer:** BBQ-unknown pre-bias must be re-measured per
  frozen-protocol target before any removal is claimed.
- **Identical-dW pairing (SC7/C3):** fp and binary variants remain
  post-processings of the same trained contrast; any cell where that pairing
  breaks is excluded, not imputed.
- **COL is the only instruct touchpoint;** its results never mix into base
  tables.
- **Reporting:** frontier tables with operable fractions; per-cell forest
  plots for BL-S; pooled-only banned; every intervention shown with its two
  nulls.

---

## Z — Freeze-before-running checklist

- [ ] BL-S fork wording (three outcomes) written into `PREREGISTRATION.md`
      before the first steering run.
- [ ] Steering grid sized to the edit's grid; norm-matching rule for the
      random-direction null stated.
- [ ] BL-T config counts equalized across methods; PCGU budget stated;
      merge-scale α ranges stated.
- [ ] AX onboarding gates run per target before removal training.
- [ ] SC7 identical-dW pairing asserted in the runner.
- [ ] SUP table drafted; REC sentences present in the manuscript skeleton.
- [ ] SPLIT decision recorded before the submission draft freezes.
- [ ] ACQ default (drop) confirmed or overridden in writing.
- [ ] Resume keys include {experiment, method, target, axis, config, seed};
      all α traces persisted.