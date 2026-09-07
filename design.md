# design.md — Self-Repair Has a Blind Spot: Same-Family Binary Weight Edits for Debiasing

**Working title (paperC):** *The Blind Spot of Self-Repair: Same-Family Binary Edits Remove Acquired but Not Inherited Bias*
**Status:** design / pre-registration draft
**Relation to program:** inverse of paperA/paperB (g-SPB designer→sibling bias *transmission*); this is designer→sibling bias *removal*. Reuses the g-SPB measurement apparatus and the transmission-theory geometry as the theoretical backbone.

---

## 0. One-paragraph thesis

A "designer" model can emit a **binary (sign-only) LoRA** that debiases a sibling model. But because a same-family designer shares a weight subspace *and* an error structure with its target, the debias direction it derives from its **own** notion of bias is null exactly where the bias is **inherited** from the shared pretraining lineage. Consequence, stated as a falsifiable law: **self / same-family edits remove *acquired* bias but fail on *inherited* bias, while a *cross-family* designer removes both.** The whole paper lives or dies on one interaction term.

---

## 1. Why this and not (a)/(b) alone

- **(b) sign-only LoRA for debiasing** is real but reads as BitDelta × Dige-2024 (incremental). It is the *mechanism*, not the paper.
- **(a) beneficial in-place bit-flips** is nearly unoccupied in peer-reviewed work but publicly pre-claimed by an informal preprint (Bankai, XOR patches, math-only). High priority-claim risk; demote to a systems corollary.
- **(c) same-family designer + blind-spot law** is genuinely open: no work has a model generate debiasing *weight edits* for another model in the same family, and none tests the inherited-vs-acquired asymmetry. Novelty lives in the *combination* + a testable theory. → this is the paper.

---

## 2. Research questions & hypotheses

**RQ1 (feasibility).** Can a sign-only edit derived from a designer's endogenous bias signal reduce a target's bias at meaningful sparsity without wrecking capability?

**RQ2 (the law — primary).** Does bias *origin* (inherited vs acquired) interact with *designer family* (same vs cross) in determining removability?

**RQ3 (mechanism).** Is the failure caused by the designer's *shared bias* (an epistemic blind spot) rather than by the *binary representation's capacity* or by *raw designer capability*?

**RQ4 (efficiency).** How much of full-precision task-vector-negation debiasing survives binarization, and at what byte budget?

Hypotheses:

- **H1.** Sign-only edits recover ≥ a large fraction of full-precision task-vector negation's bias reduction (calibrate threshold on pilot; see §7).
- **H2 (central).** Interaction is significant: `Δbias(inherited, same) ≪ Δbias(inherited, cross)` while `Δbias(acquired, same) ≈ Δbias(acquired, cross)`.
- **H3.** The interaction **disappears** when the debias signal is *exogenous* (ground-truth labeled), pinning the cause to endogenous shared bias, not to capacity or to binarization.
- **H4.** Same-family edits transfer across siblings better than cross-family edits (shared subspace), even while failing on inherited bias — the same shared-subspace fact predicts both.

---

## 3. Definitions (operational — these must be nailed before any run)

- **Designer D:** the model that *produces* the edit. **Target T:** the sibling that *receives* it.
- **Same-family vs cross-family:** same family ⇒ shared pretraining lineage (different-size siblings of one release, or base↔instruct of one base). Cross-family ⇒ disjoint pretraining lineage.
- **Inherited bias:** bias present in the base pretrained model and shared across the family at comparable magnitude. Verified by measuring the bias on ≥3 siblings and confirming presence in D itself. This is D's blind spot by construction.
- **Acquired bias:** a specific, measurable bias *injected* into T by fine-tuning on a skewed synthetic set, and verified **absent** (below a threshold) in the base and in D. D therefore does *not* share it.
- **Binary edit (the mechanism):** a LoRA-shaped delta stored as **sign(±1) + scale**, BitDelta-style. Default scale granularity = per-tensor; per-channel and scalar are ablated (§5.6). Sign pattern trained via straight-through estimator (STE) on the debias objective, or taken as `sign(Δ_fp)` of a full-precision delta (ablated).
- **Endogenous debias signal:** the debias direction is estimated from D's **own** behavior/representations — self-diagnosis completions (Schick-style, at the weight level) or the activation-space contrast D draws between "biased" and "debiased" text. This is what makes the blind spot possible.
- **Exogenous debias signal (control):** the debias direction is estimated from an external labeled counterfactual set (ground truth), independent of D's beliefs.

---

## 4. Method

### 4.1 Edit construction (same for same- and cross-family; only D changes)

1. **Elicit D's bias direction endogenously.** On a probe set, have D produce biased vs debiased completions (self-diagnosis) and/or read the residual-stream direction separating them. Aggregate into a steering/task vector `v_D` in weight-delta space.
2. **Shape into a LoRA delta** `Δ_D` on the chosen layers (low-rank projection of `v_D`).
3. **Binarize:** `E = scale ⊙ sign(Δ_D)`, either by STE-training the sign pattern against the debias loss or by direct `sign(Δ_fp)`.
4. **Apply by negation** to the target: `θ_T ← θ_T − E` (task-arithmetic negation, binary).

The load-bearing point: because step 1 uses **D's own** notion of bias, a same-family D that is itself biased produces a `v_D` that is null/wrong for *inherited* bias (its "debiased" completions are still biased ⇒ no contrast), but valid for *acquired* bias it does not share.

### 4.2 Designer variants (ablated)

- **Self-designer:** D = T (the target edits itself).
- **Sibling-designer (same-family):** D ≠ T, same lineage.
- **Cross-family designer:** D from a different lineage.
- (Optional) **Hypernetwork designer:** a Text-to-LoRA-style net trained on the family emits `E` from a bias-description prompt — tests whether the blind spot is a property of the *family's knowledge* rather than of a single checkpoint.

### 4.3 Edit selection (avoid SPB in choosing among candidate edits)

When multiple candidate edits exist (STE seeds, layer sets, ranks), select with the **bias-decomposed Bradley-Terry judge from DeBIR** (ancestry/familiarity/position/length nuisance terms removed) rather than a naive same-family judge — otherwise selection itself imports self-preference and under-corrects.

---

## 5. Experimental design

### 5.1 Core experiment — the 2×2 (this is the paper)

Fully-crossed factorial:

| | Designer: **same-family** | Designer: **cross-family** |
|---|---|---|
| **Inherited bias** | predict: **fails** (blind spot) ← key cell | predict: succeeds |
| **Acquired bias** | predict: succeeds | predict: succeeds |

- **Primary estimand:** the `origin × designer-family` **interaction** on bias reduction.
- **Capability confound — mandatory control:** run designs **bidirectionally** (family-A designs for family-B *and* B designs for A). If the asymmetry were "cross-family is just a stronger model," it would not flip symmetrically. Additionally match designer capability (MMLU) within a band.
- Repeat over **≥4 bias axes** (e.g., gender, race/ethnicity, religion, age) — at least one clearly inherited and one injected-acquired per axis where feasible — and **≥3 seeds** for edit construction.

### 5.2 Models

- **Same-family sibling sets** (multi-checkpoint families): e.g. Qwen2.5 {0.5B, 1.5B, 3B, 7B}, Llama-3.x {3B, 8B}, Gemma-2 {2B, 9B}, plus base↔instruct pairs. Siblings share lineage.
- **Cross-family pairs:** Qwen↔Llama, Llama↔Gemma (disjoint lineage), matched by size band.
- Report exact checkpoints and quantization at submission.

### 5.3 Datasets & benchmarks

- **Bias (primary outcomes):** BBQ (ambiguous *and* disambiguated), StereoSet (SS / LMS / ICAT), CrowS-Pairs, BOLD, Winogender/Winobias.
- **Acquired-bias injection sets:** synthetic skewed fine-tuning corpora inducing a *named, measurable* skew absent from base (e.g., occupation↔gender skew, sentiment↔group skew). Injection verified on held-out probes.
- **Collateral (must report every run):** MMLU, perplexity (WikiText-103 / C4), plus ARC/HellaSwag; IFEval for instruct models.
- **Transfer:** edit built on sibling A applied to siblings B, C.

### 5.4 Metrics

- **Bias reduction:** Δ(BBQ bias score), StereoSet SS → 50, CrowS stereotype %, BOLD sentiment/toxicity gap. Prefer the **g-SPB metric** as the primary modality-agnostic instrument (quality-/position-invariant, bounded) so results are comparable across axes and to paperA/paperB.
- **Collateral:** ΔMMLU, Δperplexity (report as a frontier, not a point).
- **Edit cost:** #sign flips, bytes of `E`, effective sparsity.
- **Transfer:** bias reduction when A's edit is applied to B.
- **Primary inferential quantity:** the interaction effect with CIs (§6).

### 5.5 Baselines

- **Full-precision task-vector negation** (Dige et al. 2024) — *same debias signal, full precision*; isolates the cost of binarization.
- **DPO** and **PCGU** debiasing.
- **Full-precision LoRA debiasing** (FairLoRA / Fairness-Aware LoRA).
- **Inference-time steering vectors** (e.g. activation steering / FairSteer) — no weight edit.
- **Prompt-level self-debiasing** (Schick 2021) — the non-weight self baseline.
- **Random-sign edit** control — same scale/sparsity, random signs; shows sign *structure* carries the signal (echoes the scale-guided-vs-random effect Bankai reports).

### 5.6 Ablations

- **Binary integrity:** sign-only vs sign+per-tensor scale vs sign+per-channel scale — how much continuous scale must leak back in before the "binary" story breaks.
- **Bit budget sweep:** sparsify `E`; how few flips still debias (the bias-reduction-vs-#flips frontier — this figure decides the few-flip framings' fate).
- **Sign source:** STE-trained sign vs `sign(Δ_fp)`.
- **Layer placement:** which layers to edit (early/mid/late; MLP vs attention).
- **Designer variant:** self vs sibling vs hypernetwork.
- **★ Mechanism-confirming (endogenous vs exogenous signal).** Re-run the 2×2 with an *exogenous* ground-truth debias signal. **Prediction (H3):** the interaction vanishes — same-family now removes inherited bias too. This is the experiment that proves the blind spot is *epistemic* (D's shared bias), not a capacity limit of the binary edit or a capability gap. Treat as co-primary with the 2×2.

---

## 6. Statistical analysis plan

- **Model:** mixed-effects regression of bias-reduction on `origin * designer_family` (fixed), with **bias-axis** and **sibling-pair** as random effects; seeds as replicates.
- **Primary test:** the `origin × designer_family` interaction coefficient; report point estimate + 95% CI (bootstrap over eval items and over seeds).
- **Frontier reporting:** bias reduction vs collateral (MMLU/ppl) and vs bit budget, per condition — never a single scalar.
- **Multiplicity:** pre-register the primary interaction; treat per-axis and per-benchmark breakdowns as secondary with correction.
- **Power:** pilot on one family + one cross-family pair to size the interaction before scaling axes/models.

---

## 7. Pre-registered decision criteria (kill switches)

Numbers are placeholders to **calibrate on the pilot**, then freeze:

- **Primary success (H2):** significant positive interaction where `Δbias(inherited, same) < 0.3 × Δbias(inherited, cross)` **and** `Δbias(acquired, same) > 0.8 × Δbias(acquired, cross)`, at p < 0.01 on the interaction term, holding across ≥3 of 4 axes.
- **Mechanism success (H3):** exogenous-signal interaction is not significant (blind spot closes with ground truth).
- **Feasibility gate (H1):** sign-only edit retains ≥ 70% of full-precision task-vector-negation bias reduction at ≤ a few KB and ≤ ~1–2 pt MMLU drop.
- **KILL / PIVOT:** if the cross-family designer is **no better** than same-family on inherited bias (interaction ≈ 0 with tight CI), the central thesis is false → **pivot to (b)**: reframe as a *compression/efficiency-of-debiasing* paper (sign-only LoRA ≈ full-precision debiasing at KB scale), keeping §5.5–§5.6 as the contribution.

---

## 8. Threats to validity

- **Capability confound** ("cross-family is just stronger"): bidirectional designs + capability-matched designers (§5.1).
- **Acquired-bias leakage:** verify injected bias is truly absent from base *and* D before claiming it is "not shared."
- **Evaluation circularity:** do **not** score with a same-family judge; use verifiable/labeled metrics or a cross-family judge — same-family judges share error structure (the ~60–70% shared-error-variance problem from the jury work). g-SPB + labeled bias sets sidestep this.
- **Benchmark artifacts:** BBQ/StereoSet have known quirks — triangulate across ≥3 benchmarks; don't rest the law on one.
- **Binarization ≠ blind spot:** the endogenous/exogenous ablation (§5.6) is what separates "binary can't express the fix" from "designer can't see the fix." Without it the paper is not defensible.
- **Inherited/acquired is a spectrum, not binary:** report the magnitude of D's own residual bias per axis and treat "inherited-ness" as a measured covariate, not just a label.

---

## 9. Systems corollary (framing a — a subsection, not the thesis)

Once `E = scale ⊙ sign(Δ)` exists, show it applies **in-place to a quantized checkpoint** as a small byte-patch, toggleable via BitDelta-style kernels: report **byte size** and the **bias / MMLU / perplexity** tradeoff of the in-place patch. **Cite Bankai defensively:** acknowledge its "constructive bit-flip" priority claim, then differentiate — *bias, not math; measured collateral curve; verified base; distributional not single-trigger*. Only use the phrase "beneficial bit-flip" if bits are genuinely toggled in a quantized checkpoint.

---

## 10. Connections to our program (reuse, don't rebuild)

- **g-SPB metric (paperA):** primary bias instrument here — reuse quality/position-invariance + boundedness so numbers are comparable to the transmission papers.
- **Transmission theory geometry (paperB):** the shared-weight-subspace argument that predicts *transmission* also predicts this *blind spot* — same geometry, opposite sign. Cite as the theoretical origin of H2/H4.
- **DeBIR bias-decomposed BT judge:** for candidate-edit selection (§4.3), to keep SPB out of *choosing* edits.
- Explicitly **not** connected to WF-τ / SPLASH (terminated).

---

## 11. MVP scope (preregister before scaling)

- 1 same-family set (e.g. Qwen2.5 {1.5B,3B,7B}) + 1 cross-family designer (Llama-3.x).
- 2 bias axes: 1 clearly inherited (measured across siblings) + 1 injected-acquired.
- 2 benchmarks: BBQ + StereoSet; MMLU + ppl for collateral.
- Sign-only edit, per-tensor scale; self + sibling + cross-family designers.
- Run the **2×2 and the endogenous/exogenous ablation** first — these two decide whether the paper exists. Everything else scales only if the interaction holds.

---

## 12. Open questions

- Best endogenous elicitation of `v_D` (self-diagnosis completions vs activation contrast vs gradient on self-debias loss) — pick on pilot.
- Does "inherited-ness" measured continuously predict removal failure monotonically? (If yes, that curve *is* a headline figure.)
- Layer locus of inherited vs acquired bias — do they even live in the same subspace? (If acquired bias sits in a subspace disjoint from the family blind spot, that mechanistically explains the asymmetry and is worth a probing section.)
- Hypernetwork designer: does family-level knowledge reproduce the single-checkpoint blind spot? (Speaks to whether the law is about a model or about a lineage.)