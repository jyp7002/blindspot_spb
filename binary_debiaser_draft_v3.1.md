# One Bit in a Hundred: The Sparse Signed Substructure of Behavioral Weight Edits

**Authors:** [author block TBD]
**Draft status:** v3.2 (post-v9 re-score, post-v10 number audit). **Number convention:** every figure in this paper resolves to an artifact through a machine-checked manifest (`audit/manifest_v10.yaml`, enforced by `src/v10_audit.py`); the audit runs green at freeze. Citations `[CITE:*]`, figures `[FIG n]` pending.

---

## Abstract

How much of a task vector is actually needed to preserve its behavioral effect? We train low-rank contrast vectors for social-bias removal from a model's own likelihood-based self-diagnosis, then strip them: magnitudes to one bit per weight (sign-only, one scale per tensor) and coordinates to 1% (magnitude top-k). Under a deployment-honest collateral budget — an integer-item MMLU criterion (≤ 3.7 of 200 items), perplexity ratio ≤ 1.10, every metric measured with the intervention *active* — the stripped edit retains 86.8% of full-precision removal at ≤ 3.8B and 98.5% at 7–9B, and a factorial decomposition shows where the information lives. The true sign field at magnitude-selected coordinates recovers +0.353 of bias; the same field at random coordinates recovers +0.070; sign permutation, layer shuffling, and bottom-magnitude support collapse to ≈ +0.013. The selection effect, Δ = +0.284 [+0.169, +0.407], is unanimous across 10/10 cells spanning three benchmark families and two model tiers. The substructure is **concentrated, not exclusive** — the sign field at random coordinates still beats random signs (+0.068 [+0.030, +0.112]) — and the concentration is a **deployment-grid property, measured**: extending the edit-scale grid eightfold lets the random-coordinate field recover most of the effect, shrinking the gap to +0.104 [+0.036, +0.183] without closing it. The practical consequence is a persistent patch of 3.7–14.5 MiB (one-bit signs on 1% of coordinates plus their entropy-coded support index; about a tenth of a dense one-bit delta, comparable to a LoRA adapter) that merges into the checkpoint with no serving-time hook and no additional inference-time intervention compute; at ≥ 95% sparsity no configuration in either tier fails the budget. Against activation steering and SentenceDebias driven by the *same* elicited signal at matched collateral and matched selection space, removal **ties** (edit − steering = −0.021 [−0.096, +0.050]; edit − SentenceDebias = −0.028 [−0.109, +0.043]) — steering wins on bytes by two to three orders of magnitude; the edit wins on persistence — and all three beat DPO trained on the same signal, while prompt-level self-debiasing fails the same budget in 16 of 18 conditions. The evaluation protocol is a co-contribution: it caught six defects in our own pipeline and reporting — two of which favored our method — and every number here traces to an artifact through a machine-checked manifest. A backfire regime at low pre-existing bias is characterized; the abstention gate built on it calibrates in-sample and fails held-out validation, and we make no predictor claim.

---

## 1. Introduction

Behavioral weight edits — task-vector negation, debiasing deltas, persona and refusal edits — are usually treated as dense, full-precision objects: you train a delta, you merge it. This paper asks the prior question: **how much of that object is doing the work?**

We use social-bias removal as the testbed because it supplies everything the question needs: measurable target behavior across several benchmark families, an endogenous signal source (the model's own likelihood-based self-diagnosis `[CITE:schick2021]`), and a deployment cost that must be accounted for. Three observations, all under a collateral budget measured with the intervention active, set the question up:

1. **Magnitude precision is nearly dispensable.** Binarizing the trained contrast vector to sign-plus-one-scale-per-tensor retains 86.8% of full-precision removal at ≤3.8B (deficit −0.040 [−0.093, +0.007], n = 8 cells, covers 0) and 98.5% at 7–9B — a deficit that is small but measurable (−0.005 [−0.010, −0.002], n = 4 cells, excludes 0).
2. **Density is dispensable.** Zeroing 99% of coordinates costs nothing measurable at either tier — and at 8B it *helps* (§5.5).
3. **The learned sign structure is not dispensable.** Randomizing signs at identical scale and sparsity collapses removal from +0.221 to +0.009 (paired Δ = +0.135 [+0.049, +0.213], n = 44 target×axis cells; the sweep-unit twin is in App. C).

These three facts do not yet say *where* the information lives — in the signs, in which coordinates were selected, or in where those coordinates sit. §5.3 answers with a factorial decomposition: destroy any one of sign identity, coordinate-sign pairing, or layer placement and the effect vanishes; destroy tensor placement and it vanishes in six of seven cells; move the true sign field to random coordinates and a fifth of the effect survives; keep magnitude-selected coordinates with their true signs and it all survives. **Behavioral edit information is concentrated in a magnitude-identifiable sparse signed substructure — concentrated, not exclusive, and concentrated specifically under the deployment budget** (§5.3 states both qualifiers precisely).

**Contributions, in order:**

1. **The structural result.** A seven-condition decomposition (n = 10 cells, unanimous) locating behavioral-edit information in a sparse signed coordinate structure: necessity (sign, pairing, and layer perturbations ≈ 0), sufficiency (1% of coordinates at 1 bit each), and concentration (Δ_selection = +0.284 [+0.169, +0.407] over random-coordinate support at matched grid), with two registered qualifiers — the random-coordinate sign field carries real signal (+0.068), and the concentration magnitude is conditional on the deployment α-grid.
2. **The artifact.** The surviving structure ships as a one-bit sign field on 1% of coordinates plus its support index — **3.7–14.5 MiB entropy-coded** (13–53 MiB with plain 32-bit indices) for 2.6–8B models, roughly a tenth of a dense one-bit delta (40.5–160.0 MiB) and comparable to or smaller than a rank-16 LoRA adapter (12–26 MiB). It merges into the checkpoint: no serving-time hook, no additional inference-time intervention compute, checkpoint-portable. It is *not* small next to a steering vector (0.008–0.016 MiB); what it buys is persistence, not bytes (§5.7, Table B).
3. **The evaluation protocol.** Integer-item collateral budgets, always-on measurement, equal selection spaces, and per-method nulls — under which six method families sit on one frontier, and which caught six defects in our own pipeline and reporting (§5.9), two of them in our method's favor.


**Lineage, in two sentences.** The intervention originated as the instrument of a separate study of designer–target lineage effects in debiasing; that hypothesis was not supported (interaction −0.013 [−0.056, +0.030] over 38 target×axis cells under the corrected gate; ten operationalizations; App. H), and the present paper studies the instrument on its own terms. The negative program is reported in a companion `[CITE:companion]`.

---

## 2. Related work

**Delta compression and merging.** BitDelta shows fine-tune deltas survive 1-bit sign compression `[CITE:bitdelta]`; **DARE** shows task-performance deltas tolerate *random* 90–99% drop with rescaling `[CITE:dare]`; **TIES** shows sign conflicts are the critical information in merging `[CITE:ties]`. Our decomposition speaks to all three at once and lands between them: the redundancy DARE exploits is real here too — the true sign field at random coordinates carries signal (+0.068 over random signs) — but under a deployment collateral budget the effect is concentrated at magnitude-selected coordinates (Δ_selection = +0.284), a dependence random-drop compression does not predict. **The behavioral-control regime rewards what DARE's regime forgives**; and the necessity side (sign randomization, pairing destruction, and layer shuffles all ≈ 0) is, to our knowledge, new at this granularity.

**Task arithmetic and debiasing by negation.** Negating task vectors removes behaviors `[CITE:ilharco2023]`, including social bias `[CITE:dige2024]`. We inherit the mechanism and ask the information question of it.

**Self-debiasing.** Prompt-level self-diagnosis `[CITE:schick2021]` established that models can articulate their own biases; we move the same endogenous signal into weights and show prompting itself fails an always-on collateral budget in 16 of 18 conditions (§5.7).

**Inference-time debiasing.** Activation steering `[CITE:steering]` and projection methods (SentenceDebias, INLP) `[CITE:sentencedebias; CITE:inlp]` are evaluated here from the *same* elicited signal under the same budget; steering and SentenceDebias tie the edit, INLP fails its own null on our setup. The comparison is therefore about artifact properties, which §5.7 makes explicit in a separate table.

**Preference optimization.** DPO `[CITE:dpo]` requires dense pairwise signal; §5.7 shows self-elicited contrast is sparse in exactly the way preference training cannot tolerate — a statement about signal–method match, not about DPO in general.

**Evaluation critiques.** The protocol operationalizes standing complaints `[CITE:eval-critiques]`: collateral measured with the intervention off, unequal tuning budgets, missing nulls, and — added by our own experience — floating-point budget boundaries on coarse metrics.

---

## 3. Method

**Elicitation.** For target $T$ and axis $a$, a designer $D$ (default $D = T$) produces paired corpora $(C^{+}, C^{-})$ by likelihood-based self-diagnosis: the biased and debiased continuations are identified by $D$'s own likelihoods under a diagnosis framing. Free-generation and forced-choice elicitation are invalid at these scales (App. F).

**Edit.** Train a rank-16 contrast task vector $\Delta W$ on the model's attention projections (per-family module resolution in §5.1 — *not* uniform across families), binarize per tensor ($E = s_t\,\mathrm{sign}(\Delta W_t)$), optionally sparsify to 1% by $|\Delta W|$, and merge the negation $W \leftarrow W - \alpha E$, selecting $\alpha$ among four values by the in-budget argmax.

**Pre-training predictors.** `contrast_gap` — the designer's likelihood separation between $C^{+}$ and $C^{-}$, computable at corpus-build time — is a corpus property, not a predictor. Its mechanical reading (§5.7): approximately the fraction of items on which the model can express any contrast. Whether it tracks realized removal is population-dependent (§5.8), and the abstention *gate* built on it does not validate, so this paper makes no predictor claim.

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
| phi \| occ_gender | self | +0.316 | +0.002 | +0.014 |
| phi \| occ_gender | cross-large | +0.646 | +0.002 | +0.032 |
| qwen \| crows_socio | cross-large | +0.107 | +0.002 | +0.003 |
| gemma \| occ_gender | same-large | +0.842 | +0.007 | +0.124 |

Pooled over all 15 matched cells, at the cell unit: real − sign-shuffle = +0.243 [+0.139, +0.359]; real − partition = +0.228 [+0.138, +0.328]. Both exclude zero. (The rows tabled above are four designer×cell examples spanning three of those cells; pooled over those three alone the gaps are +0.510 and +0.461, n = 3 cells.) Both nulls are reported unaveraged; the partition null is not vacuous (it reaches +0.124 on gemma|occ), and sign-shuffle is the tighter null.

### 5.3 The decomposition — where the information lives *(headline; all numbers v9)*

`[FIG 2: seven-condition decomposition — the paper's core figure]`

At the dense tier, sign necessity is established over 44 matched target×axis cells (real +0.221, random signs +0.009, paired Δ = +0.135 [+0.049, +0.213]); the same contrast at the sweep unit (n = 502) is reported in App. C. The factorial below asks the finer question at 1% density, identical scale construction, identical 4-α selection, on 10 pre-registered in-envelope cells across three benchmark families (BBQ, StereoSet, templated occupation→gender), four axes and both tiers (one 7B cell):

| condition | support | signs | removal (v9) |
|---|---|---|---:|
| **C-ref** | top-\|Δ\| | true, at those coords | **+0.353** |
| **C-a** | random | true, at those coords | **+0.070** |
| C-tensorshuf | top-\|Δ\| | patterns moved across projections | +0.045ᵃ |
| C-layershuf | top-\|Δ\| | patterns moved across layers | +0.004 |
| C-b | top-\|Δ\| | sign values permuted | +0.004 |
| C-rand | random | random | +0.001 |
| C-bottom | bottom-\|Δ\| | true, at those coords | +0.001 |

**Δ_selection = ++0.221.284 [+0.169, +0.407], excludes +0.221, unanimous 10/10 cells.** Sign identity (Δ_sign = +0.350 [+0.220, +0.490]), coordinate-sign pairing (C-b ≈ 0), layer placement (Δ_location = +0.349 [+0.221, +0.488]), and projection placement (Δ_tensor = +0.381 [+0.249, +0.520], n = 7) are each necessary on average. Bottom-magnitude support with true signs is dead (+0.001): magnitude ranking is informative in both directions. Projection identity, though, is not uniformly necessary: tensor shuffling collapses in six of the seven cells that carry it, while gemma|occ_gender retains +0.317 against that cell's own reference of +0.693 — that single cell is what lifts the C-tensorshuf mean, and the condition's own per-condition interval covers zero (App. A). ᵃ The C-tensorshuf row is n = 7 cells: phi's fused qkv_proj admits no projection shuffle, so phi's three cells carry no C-tensorshuf row.

**Two registered qualifiers, stated as claims rather than caveats.**

- **Concentrated, not exclusive.** C-a − C-rand = **+0.068 [+0.030, +0.112]**, excludes 0: the true sign field at *random* coordinates carries about a fifth of the effect. The field is redundant in the direction DARE predicts — and concentrated in the direction it does not.
- **Concentration is a deployment-grid property — measured.** Under the frozen grid C-a is *grid*-limited (its in-budget argmax sits at the largest α in 27/30 runs, spending almost no collateral) while C-ref is *collateral*-limited; on that grid C-ref recovers +0.353 and C-a +0.070. A post-hoc α extension — the deployment grid was frozen in pre-registration, before DEC ran — probed α up to 128, eight times the frozen maximum: the gap shrinks monotonically with depth but still excludes zero (+0.104 [+0.036, +0.183], n = 10 in-envelope cells, three seeds per cell, bootstrap over cells), and at the top of that grid C-a is still within budget while C-ref is not, so the extension is bounded by the probed grid — it does not show the gap closing. Under that extension — same cells, seeds, corpora and gate, the frozen arm reproducing exactly — the random-coordinate field recovers most of the effect: C-a rises to +0.284 and C-ref to +0.388, C-ref exceeding the frozen grid in only 5 of 30 runs because its collateral budget binds first. The gap shrinks with each extension step (+0.233 at α ≤ 32, +0.146 at ≤ 64) and stays unanimous in 9 of the 10 cells at the deepest point; it has not saturated — at the top of the probed grid C-a is still in budget in 10 of 30 runs while C-ref is in budget in none — so the measured gap is an upper bound on the unbounded-extension gap. **Magnitude selection buys the effect where the signal is cheapest — at an eighth of the edit scale and a fraction of the collateral** — which is precisely the regime a deployed edit lives in, and the concentration persists, reduced, well beyond that grid. The frozen-grid estimand is the registered one; the extension is reported as post-hoc, and the cell set was pre-selected in-envelope.

### 5.4 Which projections carry it

Leave-one-projection-out on the four-projection cells (n = 7; under the v9 re-score the q_proj and k_proj point estimates flip sign relative to the as-run panel and still cover 0, so both readings are unchanged):

| dropped | C-ref − LOPO | reading |
|---|---|---|
| q_proj | +0.015 [−0.012, +0.052] | droppable |
| k_proj | +0.017 [−0.006, +0.053] | droppable |
| **v_proj** | **+0.143** [+0.021, +0.285] | load-bearing |
| **o_proj** | **+0.119** [+0.080, +0.168] | load-bearing |

The minimal edit is **v + o** — which compounds with sparsity for the patch-size story and is a more robust route to a small patch than pushing density below 1%, since it rests on a paired contrast rather than a budget boundary. Two guard-rails: **phi is the counterexample** (its fused-qkv edit never touches o_proj and produces headline numbers, so o is load-bearing *within the q,k,v,o parameterization*, not universally necessary), and **density is not importance** — v_proj holds *fewer* surviving coordinates (0.64×) than q_proj (0.83×) yet is causally load-bearing, while o_proj, the densest projection (1.32×), is also load-bearing: density is not a reliable importance proxy in either direction, so the support heatmap (§5.6) must not be read as an importance map. Densities are fold-enrichment marginals over the 27 non-fused support dumps; the LOPO panel rests on the 21 dumps of its 7 four-projection cells. `[FIG 4: LOPO paired bars with CIs — q, k, v, o]`

### 5.5 Binarization, sparsity, size — and sparsity as deployability *(SPC numbers v9)*

**Binarization** (paired on identical ΔW, cell-level bootstrap): retention 86.8% at ≤3.8B (deficit −0.040, n = 8 cells, covers 0) and 98.5% at 7–9B (deficit −0.005 [−0.010, −0.002], n = 4 cells, excludes 0). The earlier reading of the large tier as "no measurable degradation" was an artifact of the collateral-gate defect (§5.9): under the corrected gate the full-precision arm recovers boundary-rejected configurations and stands at +0.328, against the binary arm's +0.322. We write *nearly free*, not free.

**Sparsity, small tier (n = 4):** 90% free (+0.067 [−0.031, +0.182]); **99% shows no measurable cost** (−0.075 [−0.167, +0.016], covers −0.155 — an earlier cost claim at this point did not survive the v9 re-score and is withdrawn); the cliff sits beyond: 99.5% −0.226 [−0.285, −0.155], 99.9% −0.417 [−0.513, −0.331].

**Sparsity, 7–9B (n = 2):** 99% *outperforms dense* (+0.112 [+0.013, +0.210]); 99.5% likewise (+0.131 [+0.011, +0.251]); 99.9% is the cliff (−0.074 [−0.121, −0.027]).

**Sparsity removes the residual budget failures.** Under the corrected gate the only 7–9B configurations that fail the collateral budget are one seed of llama-8B at ≤ 90% sparsity (dense, 50%, 90%); the same edit clears the budget at ≥ 95%, and at ≥ 95% sparsity **no configuration in either tier fails the budget**. Dense 7–9B edits are deployable in five of the six (target, seed) pairs — 3 of 3 qwen-7B seeds and 2 of 3 llama-8B seeds; sparsification is what closes the remaining one. This is a Table-B property (§5.7), not merely compression.

| target | dense one-bit | payload @ s = 0.99 | payload + index (entropy floor) | LoRA r16 (fp16) |
|---|---:|---:|---:|---:|
| qwen-3B | 40.50 | 0.41 | **3.68** | 14.06 |
| gemma-2.6B | 43.88 | 0.44 | **3.98** | 12.19 |
| llama-3.2B | 84.00 | 0.84 | **7.63** | 17.50 |
| phi-3.8B | 108.00 | 1.08 | **9.81** | 12.00 |
| qwen-7B | 98.00 | 0.98 | **8.90** | 19.25 |
| llama-8B | 160.00 | 1.60 | **14.53** | 26.00 |

*All entries are MiB (binary megabytes), the unit under which this table reproduces; decimal MB does not. Payload alone is 0.41–1.60 MiB; the support index costs ~8 bits per retained coordinate at 1% density, so the shippable patch is about nine times the payload. A bitmap costs exactly the dense payload and is never better.* `[FIG 3: retention vs sparsity, both tiers, budget-fail annotations, patch bits/MiB on the second axis]`

### 5.6 What the surviving support is

All overlap statistics are fold-enrichment over a hypergeometric null (raw Jaccard at 1% density is uninformative). Across seeds the support is stable (11.15× global, signed agreement 0.953, n = 36); across designers on the one measured cell, more so (14.64×, n = 9, single-cell caveat). Across axes the overlap is **bimodal and the split is semantic**: bbq_Age × bbq_Race share their sign field at designer-level strength (median signed agreement 0.765) while every other axis pair sits at chance (median 0.490) — medians throughout, one statistic. **The sign field tracks corpus content, not axis identity.** Support concentrates in late layers (0.47× early → 1.65× late, over the same 27 non-fused support dumps the heatmap bins). Caveats stated in full in App. F: the designer and axis arms use disjoint targets, so no within-target dissociation is claimed.

### 5.7 The frontier — removal ties; the artifact differs *(diffs v9)*

**Table A — behavioral performance** (always-on, strict budget, equal selection space, own nulls):

| cell | edit | steering | SentenceDebias | DPO |
|---|---:|---:|---:|---:|
| gemma \| occ | +0.674 | +0.680 | +0.795 | +0.000ᵇ |
| llama \| occ | +0.747 | +0.644 | +0.701 | 0.000ᵇ |
| qwen \| occ | +0.512 | +0.635 | +0.467 | −0.130 |
| phi \| occ | +0.320 | +0.539 | +0.549 | −0.124 |
| gemma \| crows | +0.123 | +0.002 | +0.007 | +0.000ᵇ |
| phi \| crows | +0.046 | +0.029 | +0.000ᵇ | +0.027 |
| llama \| crows | +0.040 | +0.015 | +0.028 | +0.000ᵇ |
| qwen \| crows | −0.071 | +0.012 | +0.067 | +0.277 |

ᵇ zeroed non-measurement, not a measured null: no configuration was in budget, and nan is scored as zero. The three DPO entries were skipped for fewer than 20 informative preference pairs; SentenceDebias on phi|crows had every candidate configuration fail the collateral budget. Per-cell gate provenance is tabulated in App. A.

**edit − steering = −0.021 [−0.096, +0.050]; edit − SentenceDebias = −0.028 [−0.109, +0.043]; all three beat DPO (edit − DPO = +0.293 [+0.030, +0.540]).** The tie was pre-registered as a branch before any steering number existed, and it was **re-adjudicated after the v9 re-score under the same registered fork: no switch.**

> **Gate-status disclosure.** Table A mixes gate provenance in two ways. (i) Four edit cells — llama and qwen on both axes — remain as-run: their panel predates the α-trace fix and cannot be re-scored. (ii) The entire DPO column is un-re-gated: its runner persisted no collateral values on the removal rows, so its budget decisions are the as-run gate. Because the defect only *rejects* in-budget configurations, every as-run selected value is a **lower bound** on its corrected counterpart per arm; differences between arms are not similarly protected, so the edit − DPO gap (+0.293 [+0.030, +0.540]) should be read as an upper bound. Restricted to the four fully re-scorable edit cells: edit − steering = −0.022 [−0.160, +0.089], covers 0. Two steering cells rose under the corrected gate (qwen|occ +0.550 → +0.635; llama|occ +0.640 → +0.644); no other dot moved. Four Table A entries are zeroed non-measurements, marked ᵇ.

**Table B — deployment properties** (what a tie makes decisive). *Steering and projection win on bytes by two to three orders of magnitude, and are said to; the edit's case is persistence — it survives merge and redistribution and costs nothing at serving time.*

| property | edit (this paper) | steering | SentenceDebias |
|---|---|---|---|
| persists after merge | **yes** | no | no |
| serving-time hook required | **no** | yes | yes |
| additional inference-time intervention compute | **none** | per-token | per-token |
| needs hidden-activation access at serving | **no** | yes | yes |
| bytes that must ship, MiB, 2.6–8B (a dense one-bit delta is 40.5–160.0) | 3.7–14.5 entropy-coded; 13–53 with plain 32-bit indices | 0.008–0.016 | 0.008–0.016 |
| removes the residual budget failure at 8B | yes (§5.5) | n/a | n/a |
| reversible | yes (subtract patch) | yes (remove hook) | yes (remove hook) |

**Method notes.** INLP fails its own random-subspace null (+0.025 [−0.070, +0.+0.8250]) and is the most collaterally expensive method tested; reported, not omitted. **DPO's failure is a signal–method mismatch, not a general inferiority:** the self-elicited corpus returns identical strings in both arms on 46%–92% of pairs, the differing fraction correlates +0.82 with `contrast_gap` (Pearson, n = 66 corpus cells; population and gate in App. E) — giving `contrast_gap` its mechanical reading — and a direction contrast can use a sparse signal where a preference objective cannot. DPO's one win (qwen|crows, where the edit backfires) marks the complementarity at the envelope boundary. **Prompting** fails the always-on budget in 16 of 18 conditions; where it removes bias it pays in perplexity (ratio 1.28), and a mild fairness prompt *increases* occupational skew on all three targets in the panel. **Sign source (as-run gate, flagged):** STE-trained signs tie sign(Δ_fp) (−0.007 [−0.023, +0.001], n = 12, one collapse cell); this panel predates the alpha-trace fix and cannot be re-scored — both arms are lower-bounded by the defect but their *difference* is not sign-protected — so STE is retired on the as-run evidence plus its added training stage and observed failure mode. **Generation-side collateral (bounded):** on the two 7–9B instruct targets, IFEval under the matched-norm contrast shows no instruction-following cost (qwen: sign − random = +0.000 at +0.698 removal; llama: −0.022 ≈ 1.1 SE; 1 axis, seed 0, 200/541 prompts).

### 5.8 Operating characterization — and a gate that does not validate

Removal tracks pre-existing bias (corr = +0.933 over 8 held-out BBQ cells; the two StereoSet cells fall on the same trend, App. E), and below it the edit does not sit at zero — it **backfires**, increasing bias by ≈ 0.09 on the lowest-pre-skew cells. `[FIG 6: pre-skew vs removal, backfire cells highlighted, calibration/held-out markers]`

The natural next step — an abstention gate on pre-skew and `contrast_gap` — **calibrates exactly as theorized in-sample** (backfire 0.18 → 0.00, utility +0.135 → +0.145, rejecting 4 of 17 cells) **and does not survive held-out validation**: the frozen held-out split is non-discriminating; the registered n = 87 sensitivity shows the pre-skew gate buys +0.0009 utility and the contrast-gap gate costs utility; and across 200 alternative splits the gate's mean utility gain is −0.0012 at the as-run gate and −0.0051 under the corrected one — re-scoring makes the gate look worse, not better. The gate is not hidden by an unlucky split — it is not there, in this cell population, where editing everything is already near-optimal. We report this as a negative result (the program's eleventh pre-registered self-refutation); the operating characterization is reported as findings and a negative result, not as a contribution (§1). `contrast_gap` predicts removal only population-dependently: across eleven defensible populations of the as-run panel the cell-level correlation spans −0.018 (sibling designers) to +0.981 (cross-family small designers), and on the self-designer population the envelope analysis is built from it is +0.305 [−0.091, +0.694], covering zero (App. E). We therefore make no predictor claim.

### 5.9 Six defects our protocol caught

Six defects occurred in this project's pipeline and reporting; every one was caught by a layer of the protocol, and every number in this paper now traces to an artifact through a 658-entry manifest checked at freeze. **Analysis-time:** (i) an early steering strength grid calibrated on bias-only sweeps ("steering never passes the budget" — false); (ii) a `padding_side` leak that inflated ΔMMLU under steering (recomputed against the quarantined panel, its real arm paired row-for-row against the clean twin: mean +0.12, up to +0.43) and produced a false +0.28 edit-over-steering advantage — both favored our method, both exposed by the pre-registered fork and per-method nulls; (iii) a floating-point collateral gate that rejected configurations sitting exactly on budget — found by our adversarial audit, its complete effect space enumerated (138 of 40,401 grid pairs, every one a 4-item drop), corrected program-wide from persisted α traces; that re-score changed 146 cell values and exactly one CI status among the 18 registered estimands (the ≤3.8B 99%-sparsity cost, already withdrawn), while the ablation family, re-scored separately, shows six further status changes — three at the sweep unit, three at the cell unit — all listed in App. C. **Reporting:** (iv) a stress-test value stated as measured that was an extrapolation beyond the probed grid (since measured, §5.3); (v) two pooled null gaps printed from string literals rather than computed (replaced, §5.2); (vi) a resampling unit — the sweep rather than the cell — used in the method-evidence script despite the project's own ban (re-bootstrapped, App. C). Faulty artifacts are retained, quarantined, in the release.

---

## 6. Limitations

**Measurement granularity.** MMLU is scored on 200 items against a 4-item budget; **1,205 probes — 15.1% of all 7,960 — sit within one item of the 4-item boundary** (833 of them, 15.9%, are among the 5,253 probes that also pass the perplexity ratio). The integer-item gate removes the knife-edge exactly, but the budget remains 4× the metric's granularity; ≥1,000 items is the forward fix.

**Gate mixing.** Five early panels predate the α-trace fix and cannot be re-scored; their boundary exposure is unknown and never assumed zero. Where they touch this paper (four frontier edit cells, the DPO column, and the STE ablation) the gate status is disclosed inline, with the monotonicity note of §5.7.

**Grid-conditional concentration.** Δ_selection's magnitude is conditional on the frozen 4-α deployment grid: under an eightfold extension it shrinks to +0.104 [+0.036, +0.183] without closing, and had not saturated at the top of the probed grid (§5.3). The direction and unanimity are not grid-conditional.

**Instruction-tuned throughout.** Every panel uses instruction-tuned checkpoints — the deployment-relevant condition, and a correction to earlier drafts of this project, which claimed measurement on base checkpoints. One base-model measurement exists: the base model is editable (+0.605 against its matched-norm null of +0.000), so the phenomenon is **not an instruct artifact**; no base-vs-instruct *relative* claim is licensed (the base arm's MMLU sits below chance in chat format, leaving it a perplexity-only budget; prompt format alone moves MMLU by −0.175).

**Envelope.** Described and calibrated, not validated out-of-sample (§5.8); population-scoped.

**Baselines.** PCGU and FairLoRA are cited, not reimplemented (pinned legacy stack; no canonical implementation) — the frontier still spans six method families. **Sample sizes:** DEC n = 10 cells; SPC 7–9B n = 2; frontier n = 8; IFEval 2 targets × 1 axis × 1 seed. **Scope:** English probes, social-bias axes, 2.6–8B targets.

---

## 7. Reproducibility statement

Sweep-pipeline bit-determinism verified (108/108 sweeps, 0 differing, across independent runs — including an exact 108/108 match between a re-scorable twin panel pair). MMLU scoring is deterministic within a fixed environment and batch size (6/6 replicates identical, in-process and cross-process) but exhibits batch-composition sensitivity of up to 1% (2 of 200 items); cross-version comparisons can differ by the same order — the one observed cross-panel divergence is attributed (stated, not proven) to a `transformers` 5.14.1 → 4.57.1 rebuild, and the environment is pinned in the release. The collateral gate is a single shared implementation compared in integer items, with its predecessor's complete effect space enumerated (138/40,401 pairs). Every α of every sweep is persisted (`alpha_trace.jsonl`), which is what made the program-wide re-score possible without GPU time; every number in this paper maps to an artifact file via a released manifest and audit script (`audit/manifest_v10.yaml`, `src/v10_audit.py`), run green at freeze. Quarantined faulty artifacts ship alongside corrected ones. `[Repo/DOI TBD]`

## 8. Conclusion

Strip a behavioral task vector of its magnitudes and 99% of its coordinates and the effect survives; perturb its learned signs or their pairing and it dies, and moving them across layers does the same; move the true sign field to random coordinates and a fifth survives. Behavioral edit information is concentrated — not exclusively, and specifically under the collateral budget a deployed edit actually faces — in a magnitude-identifiable sparse signed substructure. That structure ships as a single-digit-megabyte mergeable patch: it costs more bytes than a steering vector by two to three orders of magnitude, and what it buys is persistence — it survives merge and redistribution and adds nothing at serving time. It removes bias as well as steering and projection built from the same signal, and better than preference training and prompting; and the protocol that establishes all of this caught six defects of our own on the way, twice in our own favor. We suggest the decomposition travels beyond debiasing — any behavioral delta can be asked the same seven questions — and that integer-item budgets, always-on collateral, equal selection spaces, per-method nulls, and pre-registered forks should be the default conditions under which this literature compares anything.

---

## Appendices (contents)

- **A — Protocol:** budget definitions (integer-item gate, tolerance), always-on implementations, selection grids, null constructions, bootstrap unit.
- **B — Prompt baseline:** conditions, per-cell table, collateral traces.
- **C — Superseded-figures quarantine:** all as-run and earlier-configuration figures with origins and v9 replacements, including the float-gate row (714 rows recovered, 146 cells changed, 1 CI flip among the 18 registered estimands) and the six-defect post-mortems, the sweep-unit twins of every re-bootstrapped estimand, and the six ablation-family CI-status changes.
- **D — Reconciliations:** `contrast_gap` (cell-level) vs the null axis-level regression; the pre-skew functional frame vs the retired taxonomy (companion).
- **E — ENV in full:** inventory, split, policy tables, sensitivity, 200-seed analysis.
- **F — Axis onboarding & module resolution:** gates, elicitation failure modes, BBQ group-vs-unknown construction, per-family `resolve_targets` table, SUP1 caveats in full.
- **G — Engineering & determinism:** WS2 condition table, environment pin, memory refactor, alpha-trace format.
- **H — Lineage:** two-page summary of the negative program; pointer to the companion.

## Figure list

1. Information-removal pipeline: FP32 magnitudes → 1-bit signs → 99% coordinates removed → effect preserved; patch-size callout.
2. **Headline:** the seven-condition decomposition (v9 numbers), one panel.
3. Retention vs sparsity, both tiers, budget-fail annotations, bits/MiB second axis.
4. Support heatmap (layer × projection) beside the LOPO bars — captioned "density is not importance."
5. Frontier: Table A dots with CIs + Table B as an adjacent property matrix.
6. Pre-skew vs removal scatter; backfire cells highlighted; calibration/held-out markers; captioned as characterization, with the gate's negative result stated.
