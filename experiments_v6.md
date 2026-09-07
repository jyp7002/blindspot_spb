# experiments_v6.md — Mechanism of the Vanishing → scale-adaptive self-repair

**Companion to `design.md`, `experiments_v2–v5.md`, the v5 closure. Consumes the
closed three-pillar state.** The program's terminal description: a
family-contingent self-repair penalty exists through ~4B (H1 ρ=−0.81 n=8;
bridge: phi occ +0.288 the cleanest anchor), is a statistical tie by 7–9B
(V1 gate; t2x), while the family bias-direction geometry persists to 27–32B
(Y1 base-only Δd=0.184). Cause of the penalty: **unidentified**.

Purpose of v6: (1) identify **why the penalty vanishes with scale** — the
vanishing is treated as the primary evidence, not a nuisance; (2) invert the
identified ingredient into an **endogenous, scale-adaptive fix** that closes
the small-model penalty without touching the 7–9B tie. Success converts the
paper from scoped description (mechanism unidentified) into mechanism +
intervention.

No timeline in this document. Probes are ordered by **cost and
informativeness** only.

---

## 0. Why v6 is not the sixth failed mechanism hunt

Five mechanistic probes have already failed: G (pairwise co-encoding), O4
(multi-axis aggregate pairwise), K1 (elicitation gap as boundary cause), W
(activation EVR), W'/W'' (weight-rank geometry). All five were
**correlational** — they asked what the penalty co-varies with at one scale.

v6 differs in two structural ways, both pre-registered:

1. **The three-way filter.** A candidate mechanism is accepted only if its
   measured ingredient simultaneously accounts for
   **(F1)** the ≤4B family-contingent penalty,
   **(F2)** the 7–9B tie (the ingredient must attenuate/saturate with scale),
   **(F3)** the bridge family×axis ordering (see ORD).
   None of the five prior probes faced F2/F3; the vanishing supplies a filter
   the correlation hunts never had.
2. **Causal designs.** The scale contrast makes **transplants and ablations**
   possible (move the candidate ingredient across scales; relax the budget),
   instead of measuring co-variation in place.

**Pre-registered stop rule:** if all three probes below fail their primary
criteria, v6 terminates — the failure is recorded as probes 6–8 in the paper's
bounding section (which itself strengthens the "cause unidentified" limitation),
and the paper ships in its v5 form. No fourth probe without a new idea class.

---

## 1. Frozen inputs (carried; do NOT re-derive)

- Edit: sign-only LoRA, attn (q,k,v,o) r16, per-tensor scale. Elicitation:
  likelihood-based only. Base models only. Training ≤14B; ≥27B inference-only.
- Strict collateral budget (ΔMMLU ≥ −0.02, ppl ratio ≤ 1.10) is the *default
  reporting condition*; MV-A below deliberately varies it as the manipulated
  variable — everywhere else it stays frozen.
- Both nulls (sign-shuffle, data-partition) in every new condition; removal on
  naturalistic axes cited as gap-over-partition.
- **Bootstrap unit = target×axis cell** (v5 methodological note). Never
  per-(target, seed) pooling.
- nan→0 policy for budget-infeasible cells, with per-cell nan counts reported.
  granite excluded (instrument fragility, diagnosed). falcon-occ flagged
  (1/3 self budget-fails → partly floor-inflated).
- No hardcoded baseline constants; every comparison anchors on measured
  artifact files.
- **Endogeneity boundary (governs every FX):** legal resources = the target
  itself, its family siblings (any size), family ensembles. Illegal = ground-
  truth reference corpora and any out-of-family supervision. GT appears only
  as a calibration arm, never inside a fix. Violating this collapses the
  contribution into distillation.

---

## 2. The probes

### MV-A — Budget/collateral-curve ablation *(free: reanalysis of existing α-sweeps)*

**Hypothesis (collateral entanglement).** Same-family elicited directions
overlap the family-shared *capability* subspace more than cross-family ones;
negating them costs more MMLU/ppl per unit of bias removed; the ≤4B penalty is
therefore **borne by the budget gate**, not by removal capacity. Scale
dissolves it because larger targets absorb the same-magnitude edit with less
relative capability damage.

**Design.** From existing `runs.jsonl` α-sweeps (all α rows are recorded —
the granite diagnostic proves it):
1. Recompute the ≤3B panel penalty as a **function of budget level**:
   strict → ×1.5 → ×2 → unconstrained-best-α. Deliverable: penalty-vs-budget
   curves per designer class.
2. **Collateral-per-unit-removal** curves (ΔMMLU and ppl ratio vs removal)
   for same-family vs cross-family designers, ≤3B and 7–9B.
3. Retrodiction checks: llama-self T2 infeasibility (the n=1 "self edit more
   collaterally destructive" signal); granite's fragility; and at 7–9B the
   same/cross collateral curves should coincide.

| criterion | threshold (calibrate on first slice, then freeze) |
|---|---|
| **MV-A1** | penalty shrinks monotonically as budget relaxes, and at unconstrained the ≤3B penalty CI covers 0 (or < a pre-set fraction of the strict-budget penalty) |
| **MV-A2** | same-family collateral-per-unit-removal > cross at ≤3B (CI excludes 0) and ≈ cross at 7–9B |
| **MV-A3** | ORD passes on the entanglement measure (below) |

**If MV-A wins**, the transmission-geometry story returns through the
*collateral channel*: shared weights contaminate the elicited direction with
family-shared capability components — which is also why five removal-channel
probes found nothing.

### MV-B — Corpus transplant *(causal; corpora are cached)*

**Hypothesis (ingredient = elicitation/corpus quality).** Small models produce
degraded self-elicited corpora; by 7–9B corpus quality saturates, so everyone
ties.

**Design.** Edits are trained on the *target* from a *source corpus* — the
corpus is the transplantable object.
- **Forward:** penalized small targets (primary: phi-occ; secondary:
  qwen-crows; tertiary: falcon-occ with the nan caveat) × source
  {small-self, **same-family 7–9B sibling**, cross-family 7–9B,
  cross-family-small (controls scale-vs-family of the source), GT
  (calibration arm only)} × 3 seeds × both nulls.
- **Reverse induction:** 7–9B targets × {own-family-small corpus,
  cross-family-small corpus} — does a small-model corpus *induce* a penalty at
  a scale where none exists?
- **Normalization:** match token and item counts across sources; identical
  likelihood-elicitation recipe (transplant changes the corpus, never the
  elicitation method).

| criterion | threshold |
|---|---|
| **MV-B1** | penalty(small target ← same-family-large corpus): CI covers 0 |
| **MV-B2** | reverse induction: penalty(7–9B target ← small-self corpus) > 0, CI excludes 0 |
| **MV-B3** | family-vs-quality split: if cross-large corpus closes the penalty as fully as same-large, the ingredient is *quality*; if same-large closes it more, quality is family-conditioned — both informative, direction pre-registered as a question, not a prediction |
| **MV-B4** | ORD passes on the corpus-quality measure |

### MV-C — Residual decomposition *(analysis-heavy; explains the G paradox if true)*

**Hypothesis (shared-residual noise).** Elicited direction = GT-aligned
component + residual. At small scale, same-family residuals are *mutually
correlated and target-aligned* (shared systematic error — co-blindness living
in the residual, not in the bias profile); at 7–9B residuals shrink or
decorrelate. This would resolve G/O4: the effect is family-categorical yet
invisible to pairwise bias-profile correlation because it lives in residual
space.

**Design.** Per designer×target×axis: project the elicited direction onto the
GT direction (same basis — both trained on the same target, so weight-space
projection is well-defined; item-profile space as the robustness variant);
measure (i) residual norm fraction vs scale, (ii) pairwise residual
correlations within-family vs cross-family at ≤3B and 7–9B, (iii) residual
alignment with capability-sensitive directions (bridge to MV-A).

| criterion | threshold |
|---|---|
| **MV-C1** | same-family residual correlation > cross-family at ≤3B, CI excludes 0 |
| **MV-C2** | the same-vs-cross residual gap shrinks to CI-covers-0 at 7–9B |
| **MV-C3** | ORD passes on the residual-correlation measure |

**Joint outcomes.** The probes are not mutually exclusive (e.g., degraded small
corpora could act *through* collateral entanglement). If ≥2 pass, fit the joint
model (penalty ~ entanglement + corpus-quality + residual-corr, cell-level)
and let coefficients allocate credit; the composed fix follows the surviving
terms.

---

## 3. ORD — the bridge-ordering hook (self-applied rejection, all probes)

The winning probe's **measured ingredient** must rank-order the bridge:
Spearman correlation between the ingredient (collateral-entanglement score /
corpus-quality score / residual correlation) and the clean bridge
cross−self cells. **Pre-register the cell list now:** the family×axis cells
with 0 self-nans and CI reported (gemma-occ, gemma-crows, qwen-occ, qwen-crows,
llama-occ, llama-crows, phi-occ, phi-crows; falcon-crows; falcon-occ enters
flagged; granite excluded). A probe that passes its primary criterion but fails
ORD is **rejected** — the filter applies to us, not only to prior work.

---

## 4. FX — the fixes (pre-wired, one per probe; endogenous only)

- **FX-A (from MV-A): capability-orthogonalized negation.** Estimate a
  capability-sensitive subspace on the target (gradient directions of the
  collateral metrics, or a held-out capability-preserving estimate), project
  the elicited direction off it, then negate. Uses only the target and its own
  evals.
- **FX-B (from MV-B): familial repair.** Small family members repaired from
  the **larger sibling's** elicitation (endogenous-to-family; every family in
  the registry ships a size ladder — this is the deployment story). Variant:
  sibling-ensemble corpus.
- **FX-C (from MV-C): residual cancellation.** Ensemble several family
  elicitations and project out the shared residual (common-mode subtraction),
  or subtract the estimated family-common residual from the self direction.
- Composition allowed when the joint model demands it. Every FX is evaluated
  under the standard frozen protocol (strict budget, both nulls, 3 seeds,
  cell-level bootstrap).

---

## 5. DEMO — success criteria (the money figures)

| id | claim | threshold |
|---|---|---|
| **DEMO-1** | the fix closes the cleanest small-model penalty | phi-occ cross−self: +0.288 [+0.219, +0.362] → **CI covers 0** in-budget, endogenous-only, 3 seeds, both nulls. Secondary target: qwen-crows (+0.227 [+0.15, +0.28]) |
| **DEMO-2** | no harm at scale | fix applied at 7–9B leaves the tie intact (cross−self CI still covers 0) and does not significantly reduce absolute in-budget removal |
| **DEMO-3** (stretch) | dose–response of the intervention | post-fix residual penalty ≈ 0 across all clean bridge cells, and the per-family ingredient measure predicts the pre-fix penalty it removed |

DEMO-1 without DEMO-2 is a failure (a fix that breaks the regime where things
already work is not adaptive). DEMO-3 is what elevates the fix from "works on
phi" to "the mechanism is the lever."

---

## 6. Decision matrix (logic only, no calendar)

- **Some probe passes primary + ORD** → build its FX → DEMO. 
- **≥2 probes pass** → joint model → composed FX → DEMO.
- **All probes fail** → stop rule fires: record as probes 6–8; ship the v5
  paper with the bounding section strengthened ("the cause resists eight
  operationalizations across correlational and causal designs" — an honest and
  unusually strong limitation statement).
- **Mechanism found, DEMO-1 fails** → mechanism folds into the paper without
  the intervention claim; the fix is future work.
- **Mechanism + DEMO pass** → the paper restructures around
  mechanism + intervention (the v5 audit trail moves to the appendix), or the
  intervention splits into a companion paper — shape decided by how much the
  fix demands (protocol, baselines) beyond the current paper's budget. Both
  shapes are legitimate; neither is chosen in this document.

---

## 7. Threats & discipline

- **MV-A fairness:** "unconstrained" removal at extreme α can reflect a damaged
  model, not debiasing. The deliverable is the full penalty-vs-budget *curve*
  (strict/×1.5/×2/unconstrained), not a single unconstrained point; degenerate
  cells (catastrophic ppl) reported but excluded from the curve fit.
- **MV-B corpus confounds:** token/item-count matching across sources is
  mandatory; corpus quality must be *measured* (an explicit per-corpus score —
  e.g., GT-direction alignment of the corpus-induced edit — so MV-B3 and ORD
  have a quantitative ingredient, not a label).
- **MV-C basis discipline:** GT projection is a measurement basis computed
  pre-edit from the calibration arm; residuals never recomputed after removal
  outcomes are known.
- **Multiple comparisons:** three probes × several criteria — primaries are
  MV-A1, MV-B1, MV-C1; everything else secondary with Holm within probe.
- **No-peeking on ORD:** ingredient measures computed and frozen before their
  bridge correlation is run.
- **Elicitation invariant:** transplants and fixes change corpora/directions,
  never the likelihood-elicitation recipe.
- **Reporting:** per-cell forest plots; pooled-only banned; every FX result
  accompanied by its two nulls and its collateral frontier.

---

## Z — Freeze-before-running checklist

- [ ] Three-way filter (F1/F2/F3) and the stop rule written into
      `PREREGISTRATION.md` before any probe runs.
- [ ] ORD cell list frozen (clean cells enumerated; falcon-occ flagged;
      granite excluded).
- [ ] MV-A budget levels and curve-fit exclusion rule frozen.
- [ ] MV-B source-corpus normalization (tokens, items) and the per-corpus
      quality score defined before transplant training.
- [ ] MV-C projection basis and residual metrics defined before extraction.
- [ ] DEMO-1/2 thresholds frozen verbatim (phi-occ, qwen-crows targets named).
- [ ] Endogeneity boundary stated in every FX config; GT arms labeled
      calibration-only.
- [ ] Bootstrap unit = target×axis cell in every analysis script.
- [ ] No hardcoded baselines; resume keys include
      {probe, target, axis, source_corpus, fix, seed}.