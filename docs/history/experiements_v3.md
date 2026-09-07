# experiments_v3.md — Dataset expansion + mechanism closure

**Companion to `design.md` and `experiments_v2.md`. Consumes Results v2** (5-arm
replication, 5×5 decomposition, Exp F drivers, B dead ×3, capacity sweep).

Purpose: (1) run the requested **2–3 additional datasets** in a way that does
maximal theoretical work — not just "more benchmarks"; (2) close the two open
holes in the mechanism: the **n=5 dose-response** (ρ=−0.80, p=0.10) and the
**templated-vs-naturalistic scope boundary**, whose cause (structure vs content)
is currently unexplained; (3) finish the attn migration so every new run lands
on the frozen config.

**The paper after v2 is a two-factor claim** — removal is set by the designer's
*elicitation gap* on the axis, minus a *self-penalty* that tracks
designer–target *direction co-encoding* — plus a scope boundary. v3's job is to
turn both factors and the boundary from "supported" into "nailed or killed."

---

## 1. Frozen inputs from v2 (do NOT re-derive)

- **Edit config = attn (q,k,v,o), rank 16, binary sign + per-tensor scale**, pending L's confirmation; every v3 run uses it. q/v is retired.
- **Elicitation = likelihood-based (Schick-style).** Free-gen and forced-choice remain banned.
- **Both nulls mandatory** for every new axis: data-partition null + sign-shuffle null. Designer numbers reported above the partition null.
- **Matched-strength injection** (recipe fixed, LR searched to band) for any new injected axis or new family.
- **E2 comparison protocol**: budget-failure scores 0; equal selection space per method; merge-scale α for trained baselines.
- **Strict collateral budget** unchanged (ΔMMLU ≥ −0.02, ppl ratio ≤ 1.10); frontiers, never scalars; per-arm forest plots, pooled-only banned.
- **Known scope facts**: blind spot replicates on templated axes (5/5 arms); on CrowS every *endogenous* designer fails while exogenous works; self-penalty +0.112 is the relational core; designer quality = elicitation gap, not MMLU.

---

## 2. Priority, dependencies, and decision logic

```
P0  G  pairwise mediation on the 5×5        (existing data, ~0 GPU)
    K1 per-axis elicitation-gap regression  (existing data, ~0 GPU)
    HY CrowS per-axis hygiene               (existing data, ~0 GPU)
P1  L  attn migration completion  ──freeze attn r16──►
P2  I  DATASET EXPANSION (BBQ + HolisticBias + WinoBias; BOLD outcome-only)
P3  J  B-revival dose-response on removable axes
    K2 grand two-factor regression over everything
P4  H  family expansion (n=5 → ≥8)          — scope set by G's outcome
P5  M  templatize / de-templatize causal test of the boundary
STR N  elicitation repair; multilingual; E4 decision (default: cut)
```

**Decision matrix (record in PREREGISTRATION.md when each gate lands):**

- **G2 mediation PASS + J1 slope PASS** → mechanism nailed; paper = two-factor law with a quantified structure gradient; H shrinks to confirmatory (3 families).
- **G2 PASS + J1 FAIL** → family/pair-level mechanism holds, axis-level dose-response stays elusive; mechanism section rests on C+G; scope section rests on M.
- **G2 FAIL** (same/cross dummy survives d) → a relational component exists beyond direction sharing; H becomes make-or-break (4+ families), M elevated, and a third factor must be hypothesized explicitly.
- Writing proceeds in parallel throughout; v3 results slot into the mechanism and scope sections without changing the paper's skeleton.

---

## G — Pairwise-sharing mediation on the 5×5 panel *(P0, the free decisive test)*

**Question.** Is the self-penalty *fully explained* by designer–target direction
co-encoding?

**Data.** Existing: 5×5 panel removals (25 cells × 3 seeds) + Exp C item-space
profiles for all checkpoints. Compute `d_ij` = item-profile correlation between
the *actual designer checkpoint i* and *target checkpoint j* on the templated
axis — all 25 pairs, same-family and cross-family alike.

**Model.**
`removal_ij ~ β_d·d_ij + β_e·elicit_i + β_s·same_family_ij + (1|target) + (1|seed)`

| criterion | threshold |
|---|---|
| **G1** sharing coefficient | β_d < 0, 95% CI excludes 0 |
| **G2** mediation (primary) | β_s becomes ns once d_ij enters, AND bootstrap indirect effect (family → d → removal) CI excludes 0 |
| **G3** litmus cell | d(smol1.7b, qwen1.5b) sits in the top quartile of cross-family pairs — predicting the −0.290 anomaly from geometry alone |

**Payoff.** G2 pass = the self-penalty is direction co-encoding, full stop —
the mechanism is established on data already collected, and the smol→qwen cell
becomes a figure. G2 fail = the most informative failure available; see
decision matrix.

---

## K — Elicitation-gap regressions → the grand two-factor model

**K1 (P0, existing data).** Per-axis, per-designer elicitation gap (the
Schick-contrast separation restricted to that axis's probe set — recomputable
from stored likelihoods). Regress *endogenous* per-axis removal on it across
the CrowS axes + templated axes.

- **K1 criterion:** β_elicit > 0, CI excludes 0. Pass ⇒ the CrowS wholesale
  failure is *explained* (elicitation collapses on heterogeneous content) and
  B3's negative converts from scar to mechanism support.

**K2 (P3, after I/J data).** One regression over every axis, arm, designer:
`removal_{ij,a} ~ β_e·elicit_{i,a} + β_d·d_{ij,a} + β_c·ceiling_a + randoms(axis, target, seed)`
where `ceiling_a` = exogenous removal on axis a (the removability ceiling).

- **K2 criteria:** β_e > 0 and β_d < 0 with CIs excluding 0; same/cross dummy
  ns in the presence of both. **This is the unifying figure of the paper** —
  one model that reproduces the 5×5 panel, the CrowS failure, and the llama
  exception.

---

## HY — CrowS per-axis hygiene *(P0)*

- Bootstrap CIs on every per-axis number; per-axis claims banned without them
  (pooled remains primary).
- **Religion anomaly** (endo 0.27/0.33 > exo 0.01 violates ceiling logic):
  re-randomize the probe/edit split ×5 and check sign stability; if unstable,
  the axis is item-starved and gets gated out.
- **race-color null = −0.493**: audit item count and polarity signing; likely
  the same item-starvation.
- Set a **minimum items-per-half floor** (calibrate ~100/half) as a formal
  axis gate going forward.

---

## L — Attn migration completion *(P1, unblocks everything)*

- Re-run the remaining A arms at attn r16: **gemma first** (the other
  at-chance-vs-null arm), then llama, smollm.
- Spot-check E2's top three (STE, PCGU, sign(Δ_fp)) at attn on the qwen arm.
- **Criteria:** per-arm interaction sign preserved; 5-arm pooled interaction at
  attn CI excludes 0; E2's winner unchanged. Then attn r16 is formally frozen
  and all v3 data collection proceeds on it.
- Reporting rule: qwen@attn (+0.188, p=.063) is never cited alone — only inside
  the 5-arm pooled estimate.

---

## I — Dataset expansion *(P2 — the requested addition, built as a structure gradient)*

**Rationale.** v2 left a binary scope statement: templated ⇒ blind spot;
naturalistic ⇒ endogenous collapse. Choosing the new datasets to *span the
structure axis* converts that into a measurable gradient — and simultaneously
supplies the removable multi-axis inventory J needs.

```
fully templated ──────────── semi-structured ─────────── free-form
occ-gender (ours)            BBQ  (context+Q+2 answers)   CrowS (have)
HolisticBias (templates)     [StereoSet, optional swap]
WinoBias (Winograd frames)
```

### I.1 The trio

- **BBQ** — 9–11 category-axes with strong measured bias; ambiguous contexts
  converted to likelihood probes over the two group answers (per-item signed
  profile is native); disambiguated contexts kept as a within-dataset control
  (bias should attenuate). Polarity comes from the dataset's own labels.
  (Yes — name collision with the quantization project. Live with it.)
- **HolisticBias** — templated by construction ("I am a [descriptor] [noun]"),
  huge item counts. Select **6–10 descriptor axes** by measured pre-bias.
  This is the *templated multi-axis family* B always needed: removal should
  work here, and the axes should span a range of d_axis.
- **WinoBias / Winogender** — coreference pronoun-likelihood (pro- vs
  anti-stereotypical), heavily templated but a *different task format* than the
  pronoun probe. Tests whether "templated" is the right scope descriptor or
  whether the law is probe-format-specific. Cheap.
- **BOLD (outcome-only, not an edit source)** — after editing from likelihood
  probes (same/cross/exo on 1–2 axes), measure generation-level bias
  (sentiment/toxicity gaps) + fluency. Answers the standing reviewer question:
  *does probe-level debiasing transfer to generation?* No designer sweep needed.

### I.2 Per-axis onboarding (every new axis, before any designer run)

1. Pre-bias per arm; 2. **polarity-signing validation** (unit test against
   labeled items — the C-bug lesson, generalized); 3. `d_axis` from item-space
   profiles across all panel models; 4. **frame-overlap score** = probe↔edit
   n-gram/template overlap (the hypothesized structural driver of
   removability — pre-registered as a covariate); 5. items-per-half floor.

### I.3 Axis gates (pre-registered, to avoid the v2-B trap)

An axis enters designer comparisons only if (i) pre-bias ≥ floor on both
initial arms AND (ii) **exogenous removal ≥ floor in-budget at attn r16**
(removability ceiling). Gated-out axes reported in an appendix — gating on the
*ceiling* is legitimate; gating on the outcome is not.

### I.4 Scope & budget

Full designer set (self / same / cross / exo / both nulls) on the **two
attn-verified arms (qwen, phi)** for all surviving axes; extend to all 5 arms
only for the top axes by gate margin. 3 seeds. Ballpark: ~(9 BBQ + 8 HB + 2 WB)
axes × 2 arms × 6 conditions × 3 seeds ≈ **2–3× the v2 CrowS effort** before
the 5-arm extension.

---

## J — B-revival: the axis-level dose-response, now on axes where removal works *(P3)*

**Inventory** = every axis passing the I.3 gate (old + new; HolisticBias should
supply most of them). Then the original B1 regression, properly powered:

`b_axis ~ β·d_axis + magnitude + frame_overlap + (1|arm)` where
`b_axis` = (cross − same) above the partition null.

| criterion | threshold |
|---|---|
| **J1** slope | β > 0, CI excludes 0, over K ≥ 8 removable axes |
| **J2** survives covariates | d stays significant with magnitude AND frame-overlap in the model |
| **J3** injected control | injected axes show b ≈ 0 regardless of d (as in v2 B3) |

J1+J2 = the resurrected headline figure (`b` vs `d` across removable axes).
J fails with G passing → mechanism stands at pair level; axis-level
dose-response reported as untestable-in-principle-so-far, with the reason.

---

## H — Family expansion, n=5 → ≥8 *(P4, scope set by G)*

**Purpose.** The family-level dose-response (ρ=−0.80, p=0.10, n=5; phi partial
outlier) and the pair-level version.

**Candidates** (≥2 open-weight sibling base models each): **OLMo-2**
(1B/7B/13B — fully documented pretraining data: enables a stretch analysis
linking co-encoding to *measured* data overlap), **Falcon-3** (1B/3B/7B/10B),
**InternLM2.5** (1.8B/7B), **Granite-3** (2B/8B), **StableLM-2** (1.6B/12B).
Pythia only as targets/sharing points (elicitation gap likely too weak to
design — which is itself an F-consistent data point).

**Per-family onboarding:** architecture-aware LoRA target resolution (Phi
lesson), vocab/batch guard (Gemma lesson), matched-strength injection re-tune
into the 0.51–0.78 band, free()-regression memory test.

**Deliverables & criteria.**

| criterion | threshold |
|---|---|
| **H1** family-level | Spearman ρ (within-family sharing vs self-removal) CI excludes 0 at n ≥ 8 |
| **H2** pair-level | across all within-family sibling pairs (designer=each sibling), slope of mutual removal on pair-level d < 0, CI excludes 0 — more units than H1, finer grain |
| **H3** phi re-read | with n ≥ 8, phi is inside the prediction band or is formally flagged as the residual outlier |

Scope: G2 pass → 3 new families suffice (confirmatory). G2 fail → 4+ and H
becomes the mechanism's make-or-break.

---

## M — Templatize / de-templatize: the causal test of the scope boundary *(P5)*

**Question.** Is the boundary **structural** (probe↔edit frame overlap →
low-rank edit generalizes) or about **content** (naturalistic stereotypes
inseparable from capability)? v2 could not distinguish these.

**Design (2×2 on the corpus, everything else frozen).**
- **Templatize CrowS**: rewrite each stereotype pair of 2–3 CrowS axes into
  shared sentence frames. Rewriter = a model *outside the panel families*
  (API), to avoid contaminating co-encoding; polarity preservation validated
  against labels.
- **De-templatize occ-gender**: paraphrase-diversify the frames (same
  rewriter), destroying probe↔edit frame overlap while preserving content.
- Run exogenous first (ceiling), then the designer comparison where the
  ceiling clears the gate.

| prediction | reading |
|---|---|
| **M1** templatized-CrowS clears the removability gate (exo) | boundary is structural, not content |
| **M2** blind spot (cross − same > 0) *appears* on templatized-CrowS where it was dead zero | the law follows structure |
| **M3** de-templatized occ-gender loses removability (exo drops below gate) | converse confirmed |

M1–M3 together upgrade the scope section from descriptive ("holds on templated
probes") to causal ("holds wherever the edit can generalize, which frame
overlap controls") — and directly motivate the frame-overlap covariate in J/K2.

---

## N — Stretch / parked

- **N1 Elicitation repair.** Self-templatization: the model rewrites a
  naturalistic item into a frame, then likelihood-elicits on its *own* frame.
  If this restores endogenous removal on CrowS, it is both an applied win and
  M-consistent. Timebox: one week; drop without ceremony if flat.
- **N2 Multilingual.** KoBBQ / CBBQ: does family co-encoding vary by language
  (e.g., Qwen siblings sharing zh-centric directions)? Interesting, real scope
  creep — only after the core lands.
- **N3 E4 (in-place quantized byte-patch).** Still pending a decision;
  **default = cut to a related-work paragraph** (Bankai defensive citation).
  Revisit only if a systems venue becomes a target.

---

## S — Statistical plan updates

- **Primary endpoints:** G2, J1, and the 5-arm pooled interaction at attn (L).
  Everything else secondary with Holm correction within experiment.
- Mediation (G2, K2) via nonparametric bootstrap of the indirect effect,
  resampling at the (target, seed) level.
- All new thresholds calibrated on the first slice (first two BBQ axes / first
  new family), then **frozen into `PREREGISTRATION.md`** before the sweep —
  same discipline as v2.
- Random-effects structure declared per model; where a grouping factor is
  degenerate, fall back to cluster-robust SEs and say so in the artifact.

---

## Z — Freeze-before-running checklist (v3 additions)

- [ ] attn(q,k,v,o) r16 frozen as default after L; no run starts on q/v.
- [ ] Polarity-signing unit test passes for every new dataset before profiles are computed.
- [ ] Items-per-half floor set and applied; gated-out axes logged with reasons.
- [ ] Frame-overlap score computed pre-edit for every axis (old and new).
- [ ] d_axis / d_ij computed pre-edit; circularity guard re-affirmed (labels never fit to outcomes).
- [ ] Rewriter model for M is outside the panel families; rewrite polarity validated.
- [ ] New-family onboarding: target resolution, injection band, vocab/batch guard, memory regression test.
- [ ] Resume keys include {arm, dataset, axis} (v2 lesson, extended).
- [ ] Both nulls wired into every new axis and every new family.
- [ ] Spine decision from G recorded before H's scope is fixed.