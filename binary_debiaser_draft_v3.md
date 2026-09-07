# One Bit in a Hundred: The Sparse Signed Substructure of Behavioral Weight Edits

**Authors:** [author block TBD]
**Draft status:** v3 (post-v9 re-score). **Number convention:** figures sourced from `RESULTS_METHOD_v9.md` are final; figures marked `*` are v8 values pending confirmation by the number-source audit (`src/v9_number_audit.py`) and must be replaced by its output before the freeze. Citations `[CITE:*]`, figures `[FIG n]` pending.

---

## Abstract

How much of a task vector is actually needed to preserve its behavioral effect? We train low-rank contrast vectors for social-bias removal from a model's own likelihood-based self-diagnosis, then strip them: magnitudes down to one bit per weight (sign-only, one scale per tensor) and coordinates down to 1% (magnitude top-k). Under a deployment-honest collateral budget — an integer-item MMLU criterion (≤ 4 of 200 items), perplexity ratio ≤ 1.10, every metric measured with the intervention *active* — the stripped edit retains the effect, and a factorial decomposition shows where the information lives. The true sign field at magnitude-selected coordinates recovers +0.353 of bias; the same field at random coordinates recovers +0.070; sign permutation, layer shuffling, tensor shuffling, and bottom-magnitude support all collapse to ≈ 0. The selection effect, Δ = +0.284 [+0.169, +0.407], is unanimous across 10/10 cells spanning four benchmark families and two model tiers. The substructure is **concentrated, not exclusive** — the sign field at random coordinates still beats random signs (+0.068 [+0.030, +0.112]) — and the concentration is a **deployment-budget property**: the magnitude-selected support delivers the effect at low edit scale and low collateral, while under aggressive scale extension the random-coordinate field closes much of the gap. The practical consequence is a 0.4–1.6 MB mergeable patch with no additional inference-time intervention compute; at 8B, sparsity is what makes the edit deployable at all — every collateral-budget failure occurs at low sparsity, and the 99%-sparse edit outperforms dense (+0.112 [+0.014, +0.211]). Against activation steering and SentenceDebias driven by the *same* elicited signal at matched collateral and matched selection space, removal **ties** (edit − steering = −0.021 [−0.096, +0.050]; edit − SentenceDebias = −0.028 [−0.109, +0.043]); all three beat DPO trained on the same signal (+0.293 [+0.030, +0.540]); prompt-level self-debiasing fails the same budget in 16 of 18 conditions*. The evaluation protocol is a co-contribution: it caught three of our own implementation errors — two favoring our method, and a floating-point budget defect whose complete effect space we enumerate (138 of 40,401 grid pairs, every one a boundary case). A backfire regime at low pre-existing bias is characterized; an abstention gate built on it calibrates perfectly in-sample and does not survive held-out validation, which we report as a negative result.

---

## 1. Introduction

Behavioral weight edits — task-vector negation, debiasing deltas, persona and refusal edits — are usually treated as dense, full-precision objects: you train a delta, you merge it. This paper asks the prior question: **how much of that object is doing the work?**

We use social-bias removal as the testbed because it supplies everything the question needs: measurable target behavior across several benchmark families, an endogenous signal source (the model's own likelihood-based self-diagnosis `[CITE:schick2021]`), and a deployment cost that must be accounted for. Three observations, all under a collateral budget measured with the intervention active, set the question up:

1. **Magnitude precision is dispensable.** Binarizing the trained contrast vector to sign-plus-one-scale-per-tensor retains 86.6%* of full-precision removal at ≤3.8B and shows no measurable degradation at 7–9B (108.6%*, CI covers 0).
2. **Density is dispensable.** Zeroing 99% of coordinates costs nothing measurable at either tier — and at 8B it *helps* (§5.5).
3. **The learned sign structure is not dispensable.** Randomizing signs at identical scale and sparsity collapses removal from +0.206* to +0.008* (paired Δ = +0.198*, n = 502 sweeps).

These three facts do not yet say *where* the information lives — in the signs, in which coordinates were selected, or in where those coordinates sit. §5.3 answers with a factorial decomposition: destroy any one of sign identity, coordinate-sign pairing, or layer/tensor placement and the effect vanishes; move the true sign field to random coordinates and a fifth of the effect survives; keep magnitude-selected coordinates with their true signs and it all survives. **Behavioral edit information is concentrated in a magnitude-identifiable sparse signed substructure — concentrated, not exclusive, and concentrated specifically under the deployment budget** (§5.3 states both qualifiers precisely).

**Contributions, in order:**

1. **The structural result.** A seven-condition decomposition (n = 10 cells, unanimous) locating behavioral-edit information in a sparse signed coordinate structure: necessity (sign, pairing, and location perturbations ≈ 0), sufficiency (1% of coordinates at 1 bit each), and concentration (Δ_selection = +0.284 [+0.169, +0.407] over random-coordinate support at matched grid), with two registered qualifiers — the random-coordinate sign field carries real signal (+0.068), and the concentration magnitude is conditional on the deployment α-grid.
2. **The artifact.** The surviving structure ships as a 0.4–1.6 MB persistent patch that merges into the checkpoint: no serving-time hook, no additional inference-time intervention compute, checkpoint-portable — and at 8B, sparsification is the difference between an edit that fails the collateral budget and one that passes it.
3. **The evaluation protocol.** Integer-item collateral budgets, always-on measurement, equal selection spaces, and per-method nulls — under which six method families sit on one frontier, and which caught three of our own implementation errors (§5.9), two of them in our method's favor.
4. **Operating conditions (conditional).** Removal tracks pre-existing bias; below it the edit *backfires*; a corpus-quality score (`contrast_gap`*) predicts removal before any training. The abstention gate built from these calibrates in-sample and fails held-out validation — reported as a negative result (§5.8), so this contribution remains conditional.

**Lineage, in two sentences.** The intervention originated as the instrument of a separate study of designer–target lineage effects in debiasing; that hypothesis was not supported (interaction +0.004* [−0.042, +0.051], ten operationalizations), and the present paper studies the instrument on its own terms. The negative program is reported in a companion `[CITE:companion]`.

---

## 2. Related work

**Delta compression and merging.** BitDelta shows fine-tune deltas survive 1-bit sign compression `[CITE:bitdelta]`; **DARE** shows task-performance deltas tolerate *random* 90–99% drop with rescaling `[CITE:dare]`; **TIES** shows sign conflicts are the critical information in merging `[CITE:ties]`. Our decomposition speaks to all three at once and lands between them: the redundancy DARE exploits is real here too — the true sign field at random coordinates carries signal (+0.068 over random signs) — but under a deployment collateral budget the effect is concentrated at magnitude-selected coordinates (Δ_selection = +0.284), a dependence random-drop compression does not predict. **The behavioral-control regime rewards what DARE's regime forgives**; and the necessity side (sign randomization, pairing destruction, and location shuffles all ≈ 0) is, to our knowledge, new at this granularity.

**Task arithmetic and debiasing by negation.** Negating task vectors removes behaviors `[CITE:ilharco2023]`, including social bias `[CITE:dige2024]`. We inherit the mechanism and ask the information question of it.

**Self-debiasing.** Prompt-level self-diagnosis `[CITE:schick2021]` established that models can articulate their own biases; we move the same endogenous signal into weights and show prompting itself fails an always-on collateral budget in 16 of 18 conditions* (§5.7).

**Inference-time debiasing.** Activation steering `[CITE:steering]` and projection methods (SentenceDebias, INLP) `[CITE:sentencedebias; CITE:inlp]` are evaluated here from the *same* elicited signal under the same budget; steering and SentenceDebias tie the edit, INLP fails its own null on our setup. The comparison is therefore about artifact properties, which §5.7 makes explicit in a separate table.

**Preference optimization.** DPO `[CITE:dpo]` requires dense pairwise signal; §5.7 shows self-elicited contrast is sparse in exactly the way preference training cannot tolerate — a statement about signal–method match, not about DPO in general.

**Evaluation critiques.** The protocol operationalizes standing complaints `[CITE:eval-critiques]`: collateral measured with the intervention off, unequal tuning budgets, missing nulls, and — added by our own experience — floating-point budget boundaries on coarse metrics.

---

## 3. Method

**Elicitation.** For target $T$ and axis $a$, a designer $D$ (default $D = T$) produces paired corpora $(C^{+}, C^{-})$ by likelihood-based self-diagnosis: the biased and debiased continuations are identified by $D$'s own likelihoods under a diagnosis framing. Free-generation and forced-choice elicitation are invalid at these scales (App. F).

**Edit.** Train a rank-16 contrast task vector $\Delta W$ on the model's attention projections (per-family module resolution in §5.1 — *not* uniform across families), binarize per tensor ($E = s_t\,\mathrm{sign}(\Delta W_t)$), optionally sparsify to 1% by $|\Delta W|$, and merge the negation $W \leftarrow W - \alpha E$, selecting $\alpha$ among four values by the in-budget argmax.

**Pre-training predictors.** `contrast_gap`* — the designer's likelihood separation between $C^{+}$ and $C^{-}$, computable at corpus-build time — predicts realized removal (+0.507* [+0.220, +0.732], n = 504) and absorbs designer size. Its mechanical reading (§5.7): approximately the fraction of items on which the model can express any contrast. It remains a *predictor*; §5.8 shows the abstention *gate* built on it does not validate.

---

## 4. Evaluation protocol

**Budget, exactly.** An intervention is in budget iff it costs **at most 4 of 200 MMLU items** — compared in integer items, which removes the floating-point boundary defect this project discovered in its own gate (§5.9) — and perplexity ratio ≤ 1.10 (float, explicit 1e-9 tolerance).

**Always-on rule.** Collateral is measured with the intervention active — merged edit, added steering vector, applied projection, prepended prompt — during bias probes, MMLU, and perplexity alike.

**Equal selection space.** Every method optimizes over exactly four configurations. Cells with no in-budget configuration score 0 and are recorded.

**Per-method nulls.** Each method must beat a structure null (sign-shuffle / matched-norm random direction / random subspace) and the partition null (identical pipeline on two halves of the same biased corpus).

**Statistics.** 3 seeds per cell; percentile bootstrap over target×axis cells; nan→0 with counts; all α points persisted (`alpha_trace.jsonl`), which is what made the program-wide re-score of §5.9 possible without GPU time.

**Determinism, scoped.** Sweep-pipeline bit-determinism is verified (108/108 sweeps, 0 differing, across independent runs). MMLU scoring is deterministic within a fixed environment and batch size (6/6 replicates identical, in- and cross-process) but shows batch-composition sensitivity of up to 1% (2 of 200 items); cross-version comparisons can differ by the same order (App. G).

---

## 5. Results

### 5.1 Setup

**Models.** Six instruction-tuned checkpoints: gemma-2-2.6B, qwen2.5-3B, llama-3.2-3B, phi-3.5-3.8B, qwen2.5-7B, llama-3.1-8B. **All panels use instruction-tuned models**; the single base-model measurement appears in §6. **Per-family resolved modules:**

| family | edited modules |
|---|---|
| gemma-2, qwen2.5, llama-3.x | q_proj, k_proj, v_proj, o_proj |
| phi-3.5 | **qkv_proj (fused); o_proj not edited** |

The "(q,k,v,o)" shorthand elsewhere in this paper applies to the first row only; phi's results are produced through the fused projection without touching o_proj — relevant to §5.4.

**Axes.** occupation→gender; CrowS socioeconomic; four BBQ group-vs-unknown axes (Age, Race/ethnicity, Religion, Gender-identity); StereoSet intrasentence. Onboarding gates per axis (polarity unit test, ≥40 items per disjoint half, per-target pre-skew) in App. F.

### 5.2 The edit removes bias against both nulls

| cell | designer | real | sign-shuffle | partition |
|---|---|---:|---:|---:|
| phi \| occ_gender | self | +0.316* | +0.002* | +0.014* |
| phi \| occ_gender | cross-large | +0.632* | +0.002* | +0.032* |
| qwen \| crows_socio | cross-large | +0.101* | +0.002* | +0.001* |
| gemma \| occ_gender | same-large | +0.842* | +0.006* | +0.124* |

Pooled: real − sign-shuffle = +0.577* [+0.509, +0.638]; real − partition = +0.550* [+0.468, +0.625]. Both nulls are reported unaveraged; the partition null is not vacuous (it reaches +0.124* on gemma|occ), and sign-shuffle is the tighter null.

### 5.3 The decomposition — where the information lives *(headline; all numbers v9)*

`[FIG 2: seven-condition decomposition — the paper's core figure]`

At the dense tier, sign necessity is established over n = 502 matched sweeps (real +0.206*, random signs +0.008*, paired Δ = +0.198* [+0.175, +0.222]). The factorial below asks the finer question at 1% density, identical scale construction, identical 4-α selection, on 10 pre-registered in-envelope cells across four benchmark families and both tiers:

| condition | support | signs | removal (v9) |
|---|---|---|---:|
| **C-ref** | top-\|Δ\| | true, at those coords | **+0.353** |
| **C-a** | random | true, at those coords | **+0.070** |
| C-tensorshuf | top-\|Δ\| | patterns moved across projections | +0.046 |
| C-layershuf | top-\|Δ\| | patterns moved across layers | +0.005 |
| C-b | top-\|Δ\| | sign values permuted | +0.004 |
| C-rand | random | random | +0.001 |
| C-bottom | bottom-\|Δ\| | true, at those coords | +0.001 |

**Δ_selection = +0.284 [+0.169, +0.407], excludes 0, unanimous 10/10 cells.** Sign identity (Δ_sign = +0.350 [+0.220, +0.490]), coordinate-sign pairing (C-b ≈ 0), layer placement (Δ_location = +0.349 [+0.221, +0.488]), and projection placement (Δ_tensor = +0.381 [+0.249, +0.520], n = 7) are each individually necessary. Bottom-magnitude support with true signs is dead (+0.001): magnitude ranking is informative in both directions.

**Two registered qualifiers, stated as claims rather than caveats.**

- **Concentrated, not exclusive.** C-a − C-rand = **+0.068 [+0.030, +0.112]**, excludes 0: the true sign field at *random* coordinates carries about a fifth of the effect. The field is redundant in the direction DARE predicts — and concentrated in the direction it does not.
- **Concentration is a deployment-budget property.** C-a is *grid*-limited (argmax at the largest α in 26/30 runs, spending almost no collateral) while C-ref is *collateral*-limited. Under aggressive α extension the gap shrinks to +0.096* and covers 0. **Magnitude selection buys the effect where the signal is cheapest — at low edit scale and low collateral** — which is precisely the regime a deployed edit lives in. We state the 5× concentration as conditional on the frozen deployment protocol, and the cell set as pre-selected in-envelope.

### 5.4 Which projections carry it

Leave-one-projection-out on the four-projection cells (n = 7):

| dropped | C-ref − LOPO | reading |
|---|---|---|
| q_proj | −0.011* [−0.068, +0.043] | droppable |
| k_proj | −0.009* [−0.030, +0.008] | droppable |
| **v_proj** | **+0.119*** [+0.010, +0.250] | load-bearing |
| **o_proj** | **+0.121*** [+0.079, +0.183] | load-bearing |

The minimal edit is **v + o** — which compounds with sparsity for the patch-size story and is a more robust route to a small patch than pushing density below 1%, since it rests on a paired contrast rather than a budget boundary. Two guard-rails: **phi is the counterexample** (its fused-qkv edit never touches o_proj and produces headline numbers, so o is load-bearing *within the q,k,v,o parameterization*, not universally necessary), and **density is not importance** — v_proj holds *fewer* surviving coordinates (0.64×) than q_proj (0.83×) yet is causally load-bearing, so the support heatmap (§5.6) must not be read as an importance map.

### 5.5 Binarization, sparsity, size — and sparsity as deployability *(SPC numbers v9)*

**Binarization** (paired on identical ΔW): retention 86.6%* at ≤3.8B (deficit −0.040*, CI excludes 0) and 108.6%* at 7–9B (CI covers 0 — "no measurable degradation," pooled over two designer classes; the flattering self-only split is not cited).

**Sparsity, small tier (n = 4):** 90% free (+0.067 [−0.031, +0.182]); **99% shows no measurable cost** (−0.075 [−0.167, +0.016], covers 0 — an earlier cost claim at this point did not survive the v9 re-score and is withdrawn); the cliff sits beyond: 99.5% −0.226 [−0.285, −0.156], 99.9% −0.417 [−0.513, −0.331].

**Sparsity, 7–9B (n = 2):** 99% *outperforms dense* (+0.112 [+0.014, +0.211]); 99.5% likewise (+0.131 [+0.011, +0.251]); 99.9% is the cliff (−0.074 [−0.121, −0.027]).

**Sparsity enables deployment.** Every collateral-budget failure at 8B occurs at *low* sparsity (dense through 97%); 99%-and-sparser never fails the budget. The dense 8B edit is too disruptive to deploy at all; sparsification is what brings it under the budget. This is a Table-B property (§5.7), not merely compression.

| target | dense | s = 0.99 |
|---|---|---|
| qwen-3B | 40.5 MB | **0.41 MB** |
| gemma-2.6B | 43.9 MB | **0.44 MB** |
| llama-3.2B | 84.0 MB | **0.84 MB** |
| phi-3.8B | 108.0 MB | **1.08 MB** |
| qwen-7B | 98.0 MB | **0.98 MB** |
| llama-8B | 160.0 MB | **1.60 MB** |

`[FIG 3: retention vs sparsity, both tiers, budget-fail annotations, patch bits/MB on the second axis]`

### 5.6 What the surviving support is

All overlap statistics are fold-enrichment over a hypergeometric null (raw Jaccard at 1% density is uninformative). Across seeds the support is stable (11.15× global, signed agreement 0.953, n = 36); across designers on the one measured cell, more so (14.64×, n = 9, single-cell caveat). Across axes the overlap is **bimodal and the split is semantic**: bbq_Age × bbq_Race share their sign field at designer-level strength (signed agreement 0.769) while every other axis pair sits at chance (~0.49). **The sign field tracks corpus content, not axis identity.** Support concentrates in late layers (0.41× early → 1.52× late). Caveats stated in full in App. F: the designer and axis arms use disjoint targets, so no within-target dissociation is claimed.

### 5.7 The frontier — removal ties; the artifact differs *(diffs v9)*

**Table A — behavioral performance** (always-on, strict budget, equal selection space, own nulls):

| cell | edit | steering | SentenceDebias | DPO |
|---|---:|---:|---:|---:|
| gemma \| occ | +0.674* | +0.680* | +0.795* | +0.000ᵃ |
| llama \| occ | +0.747* | +0.640* | +0.701* | −0.000* |
| qwen \| occ | +0.512* | +0.635* | +0.467* | −0.130* |
| phi \| occ | +0.320* | +0.539* | +0.549* | −0.124* |
| gemma \| crows | +0.123* | +0.002* | +0.007* | +0.000ᵃ |
| phi \| crows | +0.046* | +0.029* | +0.000* | +0.027* |
| llama \| crows | +0.040* | +0.015* | +0.028* | +0.000ᵃ |
| qwen \| crows | −0.071* | +0.012* | +0.067* | +0.277* |

ᵃ skipped: fewer than 20 informative preference pairs. (Steering's qwen|occ rose under v9 re-selection; per-cell values marked * pending the audit's final table.)

**edit − steering = −0.021 [−0.096, +0.050]; edit − SentenceDebias = −0.028 [−0.109, +0.043]; all three beat DPO (edit − DPO = +0.293 [+0.030, +0.540]).** The tie was pre-registered as a branch before any steering number existed, and it was **re-adjudicated after the v9 re-score under the same registered fork: no switch.**

> **Gate-mixing disclosure.** The frontier's edit arm is not fully re-scorable: gemma and phi cells are true v9 (via an exact re-scorable twin, 108/108 agreement); llama and qwen edit cells remain as-run (their panel predates the alpha-trace fix). The boundary defect only *rejects* in-budget configurations, so every as-run selected value is a **lower bound** on its corrected counterpart per arm — but differences between arms are not similarly protected. Restricted to the four fully re-scorable cells: edit − steering = −0.022 [−0.160, +0.089], covers 0.

**Table B — deployment properties** (what a tie makes decisive):

| property | edit | steering | SentenceDebias |
|---|---|---|---|
| persists after merge | **yes** | no | no |
| serving-time hook required | **no** | yes | yes |
| additional inference-time intervention compute | **none** | per-token | per-token |
| needs hidden-activation access at serving | **no** | yes | yes |
| artifact size | **0.4–1.6 MB** | vector + hook | projection + hook |
| brings the 8B edit under the collateral budget | **yes (via sparsity, §5.5)** | n/a | n/a |
| reversible | yes (subtract patch) | yes (remove hook) | yes (remove hook) |

**Method notes.** INLP fails its own random-subspace null (+0.025* [−0.070, +0.150]) and is the most collaterally expensive method tested; reported, not omitted. **DPO's failure is a signal–method mismatch, not a general inferiority:** the self-elicited corpus returns identical strings in both arms on 46–92%* of pairs, the differing fraction correlates +0.79* with `contrast_gap` — giving `contrast_gap` its mechanical reading — and a direction contrast can use a sparse signal where a preference objective cannot. DPO's one win (qwen|crows, where the edit backfires) marks the complementarity at the envelope boundary. **Prompting** fails the always-on budget in 16 of 18 conditions*; where it removes bias it pays in perplexity (ratio 1.28*), and a mild fairness prompt *increases* occupational skew on three of four targets*. **Sign source (as-run gate, flagged):** STE-trained signs tie sign(Δ_fp) (−0.007* [−0.023, +0.001], n = 12, one collapse cell); this panel predates the alpha-trace fix and cannot be re-scored — both arms are lower-bounded by the defect but their *difference* is not sign-protected — so STE is retired on the as-run evidence plus its added training stage and observed failure mode. **Generation-side collateral (bounded):** on the two 7–9B instruct targets, IFEval under the matched-norm contrast shows no instruction-following cost (qwen: sign − random = +0.000* at +0.698* removal; llama: −0.022* ≈ 1.1 SE; 1 axis, seed 0, 200/541 prompts).

### 5.8 Operating characterization — and a gate that does not validate

Removal tracks pre-existing bias (corr = +0.910* over 8 held-out BBQ cells; StereoSet lands on the same line*), and below it the edit does not sit at zero — it **backfires**, increasing bias by ≈ 0.10* on the lowest-pre-skew cells. `[FIG 6: pre-skew vs removal, backfire cells highlighted, calibration/held-out markers]`

The natural next step — an abstention gate on pre-skew and `contrast_gap` — **calibrates exactly as theorized in-sample** (backfire 0.18 → 0.00, utility +0.146 → +0.156, rejecting 4 of 17 cells) **and does not survive held-out validation**: the frozen held-out split is non-discriminating; the registered n = 87 sensitivity shows the pre-skew gate buys +0.0009 utility and the contrast-gap gate costs utility; and across 200 alternative splits the gate's mean utility gain is −0.0012. The gate is not hidden by an unlucky split — it is not there, in this cell population, where editing everything is already near-optimal. We report this as a negative result (the program's eleventh pre-registered self-refutation), keep `contrast_gap` as a predictor only, and leave Contribution 4 conditional.

### 5.9 Three errors our protocol caught

Three implementation errors occurred during this project, and the protocol caught all three. (i) An early steering run used a strength grid calibrated on bias-only sweeps ("steering never passes the budget" — false). (ii) A `padding_side` leak inflated ΔMMLU under steering by ≈ 0.215 ("edit beats steering by +0.292" — false). Both favored our method; the pre-registered fork and per-method nulls exposed them. (iii) The collateral gate compared MMLU in floating point, rejecting configurations sitting exactly on budget; our own adversarial audit found it, and we enumerated its complete effect space — across all 201×201 accuracy pairs on the 1/200 grid, the old and new gates disagree on **exactly 138 pairs, every one a 4-item drop**. The program-wide re-score from persisted α traces changed 146 cell values and exactly **one** CI status (the ≤3.8B 99%-sparsity cost, already withdrawn). Faulty artifacts are retained, quarantined, in the release.

---

## 6. Limitations

**Measurement granularity.** MMLU is scored on 200 items against a 4-item budget; **833 probes — 10.5% of all 7,960 — sit within one item of the boundary.** The integer-item gate removes the knife-edge exactly, but the budget remains 4× the metric's granularity; ≥1,000 items is the forward fix.

**Gate mixing.** Five early panels predate the α-trace fix and cannot be re-scored; their boundary exposure is unknown and never assumed zero. Where they touch this paper (two frontier edit cells; the STE ablation) the gate status is disclosed inline, with the monotonicity note of §5.7.

**Grid-conditional concentration.** Δ_selection's magnitude is conditional on the frozen 4-α deployment grid (§5.3); the direction and unanimity are not.

**Instruction-tuned throughout.** Every panel uses instruction-tuned checkpoints — the deployment-relevant condition, and a correction to earlier drafts of this project which claimed base-only measurement. One base-model measurement exists: the base model is editable (+0.605* against its matched-norm null of +0.000*), so the phenomenon is **not an instruct artifact**; no base-vs-instruct *relative* claim is licensed (the base arm's MMLU sits below chance in chat format, leaving it a perplexity-only budget; prompt format alone moves MMLU by −0.175*).

**Envelope.** Described and calibrated, not validated out-of-sample (§5.8); population-scoped.

**Baselines.** PCGU and FairLoRA are cited, not reimplemented (pinned legacy stack; no canonical implementation) — the frontier still spans six method families. **Sample sizes:** DEC n = 10 cells; SPC 7–9B n = 2; frontier n = 8; IFEval 2 targets × 1 axis × 1 seed. **Scope:** English probes, social-bias axes, 2.6–8B targets.

---

## 7. Reproducibility statement

Sweep-pipeline bit-determinism verified (108/108 sweeps, 0 differing, across independent runs — including an exact 108/108 match between a re-scorable twin panel pair). MMLU scoring is deterministic within a fixed environment and batch size (6/6 replicates identical, in-process and cross-process) but exhibits batch-composition sensitivity of up to 1% (2 of 200 items); cross-version comparisons can differ by the same order — the one observed cross-panel divergence is attributed (stated, not proven) to a `transformers` 5.14.1 → 4.57.1 rebuild, and the environment is pinned in the release. The collateral gate is a single shared implementation compared in integer items, with its predecessor's complete effect space enumerated (138/40,401 pairs). Every α of every sweep is persisted (`alpha_trace.jsonl`), which is what made the program-wide re-score possible without GPU time; every number in this paper maps to an artifact file via a released audit script (`v9_number_audit.py`), run green at freeze. Quarantined faulty artifacts ship alongside corrected ones. `[Repo/DOI TBD]`

## 8. Conclusion

Strip a behavioral task vector of its magnitudes and 99% of its coordinates and the effect survives; perturb its learned signs, their pairing, or their placement and it dies; move the true sign field to random coordinates and a fifth survives. Behavioral edit information is concentrated — not exclusively, and specifically under the collateral budget a deployed edit actually faces — in a magnitude-identifiable sparse signed substructure. That structure ships as a sub-megabyte mergeable patch which, at 8B, sparsity alone makes deployable; it removes bias as well as steering and projection built from the same signal, and better than preference training and prompting; and the protocol that establishes all of this caught three of our own errors on the way, twice in our own favor. We suggest the decomposition travels beyond debiasing — any behavioral delta can be asked the same seven questions — and that integer-item budgets, always-on collateral, equal selection spaces, per-method nulls, and pre-registered forks should be the default conditions under which this literature compares anything.

---

## Appendices (contents)

- **A — Protocol:** budget definitions (integer-item gate, tolerance), always-on implementations, selection grids, null constructions, bootstrap unit.
- **B — Prompt baseline:** conditions, per-cell table, collateral traces.
- **C — Superseded-figures quarantine:** all as-run and earlier-configuration figures with origins and v9 replacements, including the float-gate row (714 rows recovered, 146 cells changed, 1 CI flip) and the three-error post-mortems.
- **D — Reconciliations:** `contrast_gap` (cell-level) vs the null axis-level regression; the pre-skew functional frame vs the retired taxonomy (companion).
- **E — ENV in full:** inventory, split, policy tables, sensitivity, 200-seed analysis.
- **F — Axis onboarding & module resolution:** gates, elicitation failure modes, BBQ group-vs-unknown construction, per-family `resolve_targets` table, SUP1 caveats in full.
- **G — Engineering & determinism:** WS2 condition table, environment pin, memory refactor, alpha-trace format.
- **H — Lineage:** two-page summary of the negative program; pointer to the companion.

## Figure list

1. Information-removal pipeline: FP32 magnitudes → 1-bit signs → 99% coordinates removed → effect preserved; patch-size callout.
2. **Headline:** the seven-condition decomposition (v9 numbers), one panel.
3. Retention vs sparsity, both tiers, budget-fail annotations, bits/MB second axis.
4. Support heatmap (layer × projection) beside the LOPO bars — captioned "density is not importance."
5. Frontier: Table A dots with CIs + Table B as an adjacent property matrix.
6. Pre-skew vs removal scatter; backfire cells highlighted; calibration/held-out markers; captioned as characterization, with the gate's negative result stated.
