# Parked abstracts — DO NOT EDIT AFTER V LANDS

Per experiments_v5 §2 and Z-checklist: both abstracts are drafted BEFORE V runs.
V's outcome (V1-a vs V1-b) selects one verbatim. The data picks; the text does
not bend afterward. Frozen 2026-07-24, before any v5 experiment executed.

Selection rule:
- **V1-a** (same-family-large removal ≈ cross-large; paired Δ CI covers 0) → **Abstract A**
- **V1-b** (same-family-large < cross-large; paired Δ CI excludes 0, negative) → **Abstract B**

---

## Abstract A — "the penalty vanished" (select if V1-a)

Modern language models inherit social biases, and a natural hope is that a model
could repair its own bias. We study whether a binary (sign-only) low-rank weight
edit, distilled from a *designer* model's own elicited bias signal, can debias a
*target* model — and whether same-family designers are handicapped on bias they
share. Across nine model families (0.4–72B) we find a two-part answer. First, the
family bias-direction *geometry* is durable: item-space bias signatures remain
family-distinctive up to 27–32B (same − cross profile-correlation gap Δd = 0.13,
statistically unchanged from 0.16 at <3.5B), and the sign-only edit — a ~1 MB
patch — matches full-precision task vectors at equal collateral. Second, the
*behavioral* self-repair penalty that this geometry produces at small scale
(within-family co-encoding predicts weaker self-removal, ρ = −0.81 at ≤3B)
**disappears with scale**: by 7–9B a model debiases itself better than any other
family debiases it (self-advantage ≈ 0.05 on both a removable and a naturalistic
axis), and same-family-large designers match cross-family-large ones. The
self-repair blind spot is thus a small-model phenomenon — above ~7B a model is
its own best debiaser. We further show removability is governed by whether a bias
forms a single coherent semantic direction (a pre-edit activation-coherence score
predicts the ground-truth removal fraction across ~20 axes), and that designer
*scale* does not substitute for ground-truth signal on naturalistic bias. We
release the edit method, the multi-family probe suite, and a candid appendix of
five bugs that would each have inverted a headline.

## Abstract B — "the penalty persists, masked" (select if V1-b)

Modern language models inherit social biases, and a natural hope is that a model
could repair its own bias. We study whether a binary (sign-only) low-rank weight
edit, distilled from a *designer* model's own elicited bias signal, can debias a
*target* model. Across nine families (0.4–72B) we find self-repair is governed by
**two opposing forces**. A *matching benefit* — the designer's signal aligning
with the target's distribution — grows with designer scale and is maximal when
the designer *is* the target. Against it runs a *family-categorical penalty*:
same-family designers debias a target less than size-matched cross-family
designers do, a gap that survives to 7–9B once designer size is controlled
(paired Δ excludes 0). At small scale the penalty dominates (within-family
co-encoding predicts weaker self-removal, ρ = −0.81 at ≤3B); by 7B the matching
benefit overtakes it, so the *net* self-repair sign flips between 4B and 7B even
though the underlying penalty never vanishes. Consistently, the family
bias-direction geometry is scale-durable (same − cross gap Δd = 0.13 at 27–32B,
≈ 0.16 at <3.5B). We localize the crossover with a bridge tier and show
removability is set by semantic-direction coherence (a pre-edit activation score
predicts removal across ~20 axes), not by designer scale. We release the method,
the probe suite, and an appendix of five headline-inverting bugs.

---

## Shared elements (identical either way)

Method, figures, and tables are identical across A/B except the abstract and the
§CLOSE interpretation paragraph. Three pillars: (1) anti-convergence geometry
[O1-T4 + Y]; (2) the threshold arc [T2 + V + X] with the two-force model; (3)
consistent-semantic-direction removability [M/S/T + W] with the prediction
battery and upheld R2 as the falsifiability showcase.
