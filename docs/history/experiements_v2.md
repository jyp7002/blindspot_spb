# experiments_v2.md — Next experiments after the MVP

**Companion to `design.md`.** Consumes the MVP run (756 records, 2 arms). Purpose:
convert the MVP's three verdicts — *directional-but-weak law*, *1-arm-carried*,
*mechanism misattributed (§3 dead)* — into the specific experiments that decide
whether paperC ships as a **law** or as a **scoped finding + efficiency paper**.

**Governing principle unchanged:** the paper lives or dies on whether the
`origin × designer-family` blind spot (i) generalizes past one target family and
(ii) is *predicted* by weight-direction geometry rather than fit to it.

---

## 1. Frozen inputs from the MVP (do NOT re-derive)

- **Elicitation = likelihood-based self-debiasing (Schick-style).** Free generation and forced-choice A/B are *invalid* (pronoun-drop; sub-4B at chance; measures recognition not exhibition). Do not revisit; report the two failures as negative results in the methods section.
- **Binary edit default = sign + per-tensor scale.** Per-channel buys ~nothing; scalar costs ~18%. Full strength reachable dense or at 90% sparsity (~1 MB); 96 KB retains only ~62% — do **not** claim "few KB" at full strength.
- **Collateral budget (strict) = ΔMMLU ≥ −0.02 AND ppl ratio ≤ 1.10.** Report full α traces and both budgets; never a single scalar (design.md §5.4).
- **Acquired-bias injection recipe = regulariser from a *disjoint train* split, eval on *test*.** (female→neg valence gave skew +0.552, ppl 13.2, MMLU 0.635 vs 0.645 base.) Reuse verbatim for any new injected axis.
- **Two nulls are now mandatory in every condition** (see §N): the *data-partition* null (~0.298 on gender — negating any gendered-minimal-pair contrast lands in the gender subspace) and the *sign-shuffle* null (~0). Any designer number is reported **as a gap above the data-partition null**, never raw.
- **Binarization is free (H1 settled):** ~99% retention of full-precision task-vector negation, MMLU −0.003. Do not re-litigate; build on it.

---

## 2. Priority, dependencies, and the spine fork

```
A (arm replication) ──────────────► decides the SPINE
        │
        ├─ replicates ──► C (measure directions) ──► B (geometry test = headline) ──► D (robustness) ──► full (c), unified with paperB
        │
        └─ Qwen-only ───► demote (c) to scoped finding; ship (b)+methodology+E

E ((b) safety net + systems corollary) ── runs in PARALLEL, no dependency, publishable regardless
```

- **Run A first.** Cheapest per-insight and it *directly* gates the paper's spine. The MVP's whole point was to de-risk before scaling, and the risk that fired is exactly "1-arm effect" — so resolve it before investing in the mechanism cluster.
- **Start E in parallel now** — it is the safety net and shares no dependency with A/B/C.
- **Alternative ordering (only if you are already confident in the geometry):** C→B first to bank the headline figure, then A for generality. Higher variance — if A then fails, the headline is stranded on one family.

---

## Experiment A — Arm replication *(make-or-break)*

**Question.** Is the `origin × designer-family` interaction (the blind-spot *specificity* to inherited bias) a general phenomenon, or is it Qwen-target-specific? (MVP: pooled +0.150, but Qwen +0.253 / Llama +0.043 ns.)

**Design.** Re-run the full design.md §5.1 2×2 with **≥3 additional target families** — Gemma-2, Phi-3.5, SmolLM2 (all already bias-measured) — plus the existing Qwen, Llama. Each target arm gets: same-family sibling designer, ≥1 cross-family designer, exogenous ground-truth signal, data-partition null, sign-shuffle null. Likelihood elicitation; binary per-tensor edit; strict collateral budget; 3 seeds. Keep both axes from the MVP (inherited = occupation→gender; acquired = female→neg valence).

**Analysis.** Mixed-effects with **arm as random effect**; report a **per-arm forest plot** of the interaction (never pooled-only). Also report `same − data-partition-null` on inherited per arm.

**Pre-registered criteria (calibrate placeholders on the first two new arms, then freeze).**

| criterion | threshold |
|---|---|
| A1 pooled interaction | sign > 0, 95% CI excludes 0, arm as random effect |
| A2 per-arm replication | interaction point estimate > 0 in ≥⌈0.6·n⌉ arms **and** CI excludes 0 in ≥2 arms |
| A3 blind-spot null-equivalence | `same − partition-null` on inherited has CI covering 0 (i.e. at-chance) in ≥⌈0.6·n⌉ arms |
| A4 acquired non-specificity | `cross − same` on *acquired* ≈ 0 (CI covers 0) pooled — confirms the effect is inherited-specific, not a blanket cross-advantage |

**KILL / DEMOTE.** If the interaction is > 0 in ≤1 arm, **or** A1 fails (pooled CI covers 0), retire the "law" claim: reframe (c) as *"a self-debiasing blind spot observed in the Qwen family (not replicated in Llama/…)"* and lead the paper with **(b) + methodology + E**. This is the design.md §7 pivot, now arm-conditioned.

**Cost.** ~3 arms × 2 origins × {same, cross, exo, 2 nulls} × 3 seeds, α-swept. Reuses all MVP infra + the bug fixes. Lowest cost-per-decision of everything here.

---

## Experiment B — Axis-divergence × blind-spot magnitude *(the reframe test → new headline)*

**Why this exists.** design.md §3's premise ("disjoint lineage ⇒ different bias ⇒ cross-family succeeds because it lacks the bias") is **falsified**: occ→gender skew is +0.65–0.86 in every lineage, yet the effect is still relational (bidirectional flip). So the mechanism must be geometric — *a designer cannot get leverage on a bias it encodes in the same weight directions as its target; same-family models are co-blind on shared axes.* This experiment turns that post-hoc story into a **prediction**.

**Requires Experiment C** (per-axis direction geometry). Run C's measurement first.

**Question.** Does blind-spot magnitude track *direction-sharing* between same-family models on an axis — independent of bias *magnitude*?

**Design.**
1. Select **K ≥ 5 axes** spanning a range of between-family direction-sharing (from the 22-axis inventory): gender (low differentiation — everyone equally biased), religion (highest, ~3× SD ratio), plus intermediate axes. Include ≥1 **injected/acquired** axis (direction-sharing ≈ 0 by construction) as a within-design control.
2. From C, compute per-axis **direction-sharing** `d_axis` = mean pairwise cosine of *same-family* bias directions (measured in weight-delta space, pre-edit).
3. Compute per-axis-per-arm **blind-spot magnitude** `b_axis` = (`cross − same`) above the data-partition null, or equivalently the same-family deficit vs exogenous.
4. Regress `b_axis` on `d_axis` (mixed-effects, arm random).

**Prediction (H_geom).** Slope of `b_axis` on `d_axis` > 0. Acquired axes (`d≈0`) show `b≈0` regardless. **Magnitude of bias must *not* predict `b` once `d` is in the model** (drop-in test — this is what distinguishes the geometry story from a trivial "more bias = harder" story).

**Circularity guard (non-negotiable).** `d_axis` is computed from *pre-edit* bias-direction geometry on a pipeline disjoint from the editing outcome. Inherited/acquired labels are fixed *independently* (injection vs measured cross-family presence), never fit to the removal outcome. If an axis's "inherited-ness" were defined as "where same-family fails," the whole test is tautological — forbidden.

**Pre-registered criteria.**

| criterion | threshold |
|---|---|
| B1 geometry slope | slope(`b`~`d`) > 0, 95% CI excludes 0, over K≥5 axes |
| B2 magnitude drop-in | adding bias-*magnitude* as a covariate does not make `d` non-significant, and magnitude alone is not sufficient |
| B3 acquired control | injected axes show `b` CI covering 0 |

**Payoff.** B1+B2 = the reframed geometric mechanism is *predicted and confirmed* → headline figure (`b` vs `d`), and paperB↔paperC unify under one geometry. B1 fail → the mechanism claim is unsupported; report the blind spot phenomenologically only (weaker paper) and revisit.

---

## Experiment C — Bias-direction geometry *(mechanism instrument; feeds B; mediation core)*

**Why.** The MVP's alignment signal is real but *fragile in absolute terms* (cosines +0.144 vs +0.029; r=+0.38, p=.001, 72 sketches). The mechanism claim is only as strong as this signal, so it must be tightened before it can carry B or a mediation argument.

**Design.**
- Estimate per-family, per-axis **bias directions** at higher fidelity: more probe minimal-pairs, larger/exact sketches (or exact directions restricted to a top-r subspace), ≥3 seeds, with bootstrap CIs on every cosine.
- Outputs per axis: (i) same-family pairwise cosine, (ii) cross-family cosine, (iii) each designer's *elicited-contrast*-to-*ground-truth* cosine (the tightened §4 quantity).
- **Mediation.** Test whether direction-alignment *mediates* the same/cross → removal effect (report indirect effect + CI). If alignment mediates most of it, the causal chain `family → shared directions → (mis)alignment → removal` is evidenced.

**Pre-registered criteria.**

| criterion | threshold |
|---|---|
| C1 same>cross alignment gap | on inherited axes, same-family cosine > cross-family cosine, CI excludes 0 |
| C2 tighter signal | cosine CIs tight enough that the alignment→removal correlation holds at n per axis (not just pooled) |
| C3 mediation | direction-alignment mediates ≥ a pre-set fraction of the same/cross effect on removal |

---

## Experiment D — Robustness / reviewer-facing controls

Run only if A replicates (otherwise these attach to the scoped finding).

- **D1 Capability-matched designers.** Restrict designers to an MMLU band; confirm the interaction survives (formalises the §3 bidirectional control against "cross is just stronger" — MVP already showed the symmetric +0.113 flip; make it a matched-capability table).
- **D2 Layer locus (design.md §12).** Probe whether *acquired* and *inherited* bias occupy *separable* subspaces (activation patching / subspace-overlap between the two edit directions). If separable, that mechanistically explains why the same designer removes one and not the other — a strong mechanistic section.
- **D3 Second injected axis.** Repeat acquired-bias removal on a *different* injected axis (reuse the frozen injection recipe) to show acquired-removal generalises beyond female→neg valence.
- **D4 Elicitation robustness.** Confirm likelihood-based elicitation is stable across model sizes; ship the free-gen / forced-choice failures as documented negatives.

---

## Experiment E — (b) safety net + systems corollary *(parallel, publishable regardless of A)*

This is the paper that ships if A demotes (c). Complete the design.md ablations the MVP skipped.

- **E1 Sign source.** STE-trained signs vs `sign(Δ_fp)` — does STE buy anything over direct sign at matched collateral?
- **E2 Full baseline suite at matched collateral.** DPO, PCGU, FairLoRA, inference-time steering. Position binary self-debiasing on the bias-vs-collateral frontier against all of them. (MVP has only the full-precision-task-vector baseline.)
- **E3 Bit-budget frontier.** Extend the §5b sweep into a clean `bias-reduction vs #flips` figure across arms.
- **E4 In-place quantized byte-patch (framing a).** Apply `E` to a *quantized* checkpoint in-place; report byte size + bias / MMLU / ppl tradeoff. **Cite Bankai defensively** — differentiate on *bias not math*, *measured collateral curve*, *verified base model*, *broad distributional shift not a single-trigger patch*. Only claim "beneficial bit-flip" if bits are genuinely toggled in-place.

---

## N. Analysis-plan changes (adopt into design.md permanently)

- **Both nulls mandatory** in every condition, every arm, going forward. Report every designer number as a **gap above the data-partition null**. Rationale: without it, all designers are overstated (the null is 0.298, not ~0).
- **Mechanism narrative reframe.** Retire design.md §3's disjoint-bias story. Adopt the §10 **transmission-geometry** thesis: *shared weight subspace ⇒ co-blindness on co-inherited axes ⇒ debias blind spot* — the **same geometry that drives bias transmission in paperB, with the sign reversed.** State inherited/acquired as an independently-fixed, measured covariate ("inherited-ness"), not a binary label fit to outcome.
- **Reporting.** Per-arm forest plots always; pooled-only is banned. Effect sizes reported *above null* with CIs. Frontiers, not scalars.
- **Circularity guard** (from Experiment B) applies study-wide.

---

## Z. Freeze-before-running checklist

- [ ] Placeholder thresholds (A1–A4, B1–B3, C1–C3) calibrated on the first two new arms / axes, then **frozen** and written into `PREREGISTRATION.md` before the full sweep.
- [ ] Axis inventory for B fixed with `d_axis` measured *before* any editing; injected axes flagged.
- [ ] Data-partition null and sign-shuffle null wired into every condition.
- [ ] Resume keys include the arm (MVP bug: arm-less keys collided and wrote nothing).
- [ ] Tokenizer-agnostic continuation scoring on (MVP bug: Phi-3.5 SentencePiece read gender skew as 0.000).
- [ ] Injection templates parameterised for gender (MVP bug: hard-coded "He" corrupted the other axis).
- [ ] Spine decision recorded once A lands: *full (c) unified with paperB* vs *(b)+methodology+E with (c) scoped*.
