# DRAFT_V3.2_CHANGES.md — applying V10_FINDINGS to `binary_debiaser_draft_v3.1.md`

**Scope.** Every correction in `V10_FINDINGS.md` §2–§4, plus the claim-level
consequences of §3, as paste-ready replacement text keyed to v3.1 line numbers.
Numbers below come from the v10 artifacts named in `V10_FINDINGS.md` §5; values
tagged `⟨regen⟩` come from the cell-unit re-bootstrap (Part E) and are not yet
final. Apply Part A (claims) first, Part B (line items) second, then regenerate
the abstract from Part C and Table B from Part D, then run `make freeze`.

**Ownership note for the record.** The following v3 defects were introduced in
drafting, not in the lab record: "four benchmark families" (L10/L109);
double-rounding from 4-dp published strings (L115/116/145/147); the dropped
`n = 7` footnote on C-tensorshuf; "two frontier edit cells" (L217) contradicting
§5.7; the missing `[FIG 4]` callout; and the overreach "tensor shuffling …
collapse to ≈ 0" (L10). The rest are upstream (as-run values, literals in
`method_evidence.py`, the extrapolated α-extension, payload-only patch sizes).

---

## Part A — Claim-level changes

### A1. Patch size: "sub-megabyte" is withdrawn; the artifact is single-digit MiB

**Why.** The stored size relation prices one bit per retained coordinate and
zero bits for *which* coordinates. A 1%-sparse sign field cannot be applied
without its support. Entropy-coded, the index is ~8 bits per surviving
coordinate — irreducibly ~8× the 1-bit payload.

**Contribution 2 (§1), replace with:**
> **The artifact.** The surviving structure ships as a 1-bit sign field on 1% of coordinates plus its support index — **3.7–14.5 MiB entropy-coded** (13–53 MiB with plain 32-bit indices) for 2.6–8B models, roughly a tenth of a dense 1-bit delta (40–160 MiB) and comparable to or smaller than a rank-16 LoRA adapter (12–26 MiB). It merges into the checkpoint: no serving-time hook, no additional inference-time intervention compute, checkpoint-portable. It is *not* small next to a steering vector (0.008–0.016 MiB); what it buys is persistence, not bytes (§5.7, Table B).

**§5.5 size table, replace with** (source `results/v10/patch_sizes.json`,
`artifact_sizes.json`; **units are MiB** — the table reproduces only under
bytes/2²⁰):

| target | dense 1-bit | payload @ s=0.99 | payload + index (entropy floor) | LoRA r16 (fp16) |
|---|---:|---:|---:|---:|
| qwen-3B | 40.50 | 0.41 | **3.68** | 14.06 |
| gemma-2.6B | 43.88 | 0.44 | **3.98** | 12.19 |
| llama-3.2B | 84.00 | 0.84 | **7.63** | 17.50 |
| phi-3.8B | 108.00 | 1.08 | **9.81** | 12.00 |
| qwen-7B | 98.00 | 0.98 | **8.90** | 19.25 |
| llama-8B | 160.00 | 1.60 | **14.53** | 26.00 |

Caption sentence: *"Payload alone is 0.41–1.60 MiB; the support index costs
~8 bits per retained coordinate at 1% density, so the shippable patch is ~9×
the payload. A bitmap costs exactly the dense payload and is never better.
Restricting the support to v_proj and o_proj (§5.4) roughly halves these
figures ⟨regen⟩."*

**Title** stays: "one bit in a hundred" is a statement about the information
that carries the effect, not about shipped bytes; §5.5 now says so explicitly.

### A2. Binarization is nearly free, not free — and the unit matters

**§1 observation 1, replace with:**
> 1. **Magnitude precision is nearly dispensable.** Binarizing the trained contrast vector to sign-plus-one-scale-per-tensor retains **86.8%** of full-precision removal at ≤3.8B (deficit −0.040 [−0.093, +0.007], n = 8 cells, covers 0) and **98.5%** at 7–9B — a deficit of 1.5 points that is small but measurable (−0.005 [−0.010, −0.002], n = 4 cells, excludes 0).

**§5.5 binarization paragraph, replace with:**
> **Binarization** (paired on identical ΔW, cell-level bootstrap): retention 86.8% at ≤3.8B (n = 8, deficit covers 0) and 98.5% at 7–9B (n = 4, deficit −0.005 excludes 0). An earlier "no measurable degradation at 7–9B" (108.6%) was an artifact of the collateral-gate defect (§5.9): the corrected gate recovered boundary-rejected full-precision configurations (+0.297 → +0.328) while the binary arm did not move. We write *nearly free*, not free.

Abstract: "shows no measurable degradation at 7–9B" → "retains 86.8% / 98.5%".

### A3. Deployability: the true, weaker version

**§5.5 "Sparsity enables deployment" paragraph, replace with:**
> **Sparsity removes the residual budget failures.** Under the corrected gate the only 7–9B configurations that fail the collateral budget are one seed of llama-8B at ≤ 90% sparsity (dense, 50%, 90%); the same edit clears the budget at ≥ 95%, and **at ≥ 95% sparsity no configuration in either tier fails the budget**. Dense 7–9B edits are deployable in 5 of 6 (target, seed) pairs; sparsification is what closes the remaining one.

Delete: "The dense 8B edit is too disruptive to deploy at all". Table B row
"brings the 8B edit under the collateral budget" → "removes the residual
budget failure at 8B (§5.5)".

### A4. Resampling unit, and the scope of "exactly one CI status changed"

`src/method_evidence.py` bootstraps the sweep (target, axis, seed, designer);
the project unit is the target×axis cell (banned per-seed pooling, v5).
**All C2/C3/C5/C6 statistics in the main text move to the cell unit** ⟨regen⟩;
sweep-unit values go to App. C with the unit named. Expected consequences
(from V10_FINDINGS §3.2/§3.6): the ≤3.8B binarization deficit loses
significance; six ablation-family CI statuses change; C2's Δ = +0.198 will
widen but is not expected to cross 0.

**§5.9, replace** "changed 146 cell values and exactly **one** CI status" with:
> changed 146 cell values and exactly one CI status among the 18 registered estimands; the ablation family, re-scored separately, shows six further status changes at the sweep unit, all listed in App. C.

### A5. §5.2 pooled nulls were literals — replace with computed values

**§5.2, replace the pooled sentence with** (source: reclaimed C1, all matched
cells, cell unit, n = 15):
> Pooled over all 15 matched cells: real − sign-shuffle = **+0.244 [+0.140, +0.359]**; real − partition = **+0.228 [+0.139, +0.328]**. Both exclude 0. (The four cells tabled above are the largest; pooled over those alone the gaps are +0.510 and +0.461, n = 3 designers.)

The previous +0.577 / +0.550 are added to the App. C quarantine as
"hardcoded literals in `method_evidence.py` C1 block, never computed."

### A6. Tensor shuffling does not collapse uniformly

**L10 (abstract) and §5.3 necessity sentence, replace with:**
> sign permutation, layer shuffling, and bottom-magnitude support collapse to ≈ 0; projection (tensor) shuffling collapses in six of seven cells, while one cell (gemma|occ_gender) retains 46% of its effect, so Δ_tensor = +0.381 [+0.249, +0.520] (n = 7) excludes 0 but projection identity is not uniformly necessary.

C-tensorshuf table row gets `ᵃ n = 7 (phi's fused qkv_proj admits no
projection shuffle)`; per-condition CI [−0.002, +0.137] reported in App. A.

### A7. Gate disclosures: DPO column, four edit cells, zeroed cells

**§5.7 gate-mixing box, replace with:**
> **Gate-status disclosure.** Table A mixes gate provenance in two ways. (i) Four edit cells — llama and qwen on both axes — remain as-run: their panel predates the α-trace fix and cannot be re-scored. (ii) The entire DPO column is un-re-gated: its runner persisted no collateral values on the removal rows, so its budget decisions are the as-run gate. Because the defect only *rejects* in-budget configurations, every as-run selected value is a lower bound on its corrected counterpart per arm — differences between arms are not similarly protected, so the edit − DPO gap (+0.293 [+0.030, +0.540]) should be read as an upper bound. Restricted to the four fully re-scorable edit cells, edit − steering = −0.022 [−0.160, +0.089], covers 0. Two steering cells rose under the corrected gate (qwen|occ +0.550 → +0.635; llama|occ +0.640 → +0.644); no other dot moved. Four Table A entries are zeroed non-measurements (no configuration in budget; nan → 0), marked ᵇ.

Regenerate Table A from `results/v10/frontier_v9.json`; mark SD phi|crows
(0 of 12 in budget) and the three DPO skips with ᵇ; llama|occ steering =
+0.644.

**§6 Limitations "Gate mixing", replace** "two frontier edit cells; the STE
ablation" with "four frontier edit cells, the DPO column, and the STE
ablation".

### A8. The α-extension is now measured — rewrite the qualifier as a result

**§5.3 second qualifier, replace with:**
> **Concentration is a deployment-grid property — measured.** Under the frozen grid {2, 4, 8, 16}, C-a is grid-limited (in-budget argmax at α = 16 in 27 of 30 runs, spending almost no collateral) while C-ref is collateral-limited. A post-hoc extension to α ∈ {32, 64, 128} — same cells, seeds, corpora, and gate; the frozen arm reproduces exactly — lets the random-coordinate field recover most of the effect (C-a +0.070 → +0.284; C-ref +0.353 → +0.389, exceeding the frozen grid in only 5 of 30 runs because perplexity caps it). The gap shrinks monotonically with extension depth (+0.233 at α ≤ 32, +0.146 at ≤ 64, **+0.104 [+0.036, +0.183]** at ≤ 128, 9/10 cells) and still excludes 0, but has not saturated: at α = 128 C-a remains in budget in 10 of 30 runs while C-ref is in budget in none. The measured gap is therefore an upper bound on the unbounded-extension gap. **Magnitude selection buys the effect where the signal is cheapest** — at an eighth of the edit scale and a fraction of the collateral — and the concentration persists, reduced, well beyond the deployment grid. The frozen-grid estimand is the registered one; the extension is reported as post-hoc.

Delete "+0.096 and covers 0" everywhere. §5.9 gains error (iv): the earlier
figure was an extrapolation reported as a measurement, found by the number
audit and replaced by the measurement above.

### A9. `contrast_gap` predictor withdrawn; Contribution 4 removed

- §3: delete the "predicts realized removal (+0.507 …) and absorbs designer
  size" sentence; keep `contrast_gap` only as the corpus-property used in the
  DPO explanation (§5.7), with the population stated when the +0.79
  correlation is re-derived ⟨regen⟩ — else drop that clause too.
- §1: **three contributions** (structural / artifact / protocol). The
  operating characterization becomes a findings-and-negative-results section,
  not a contribution.
- §5.8, add after the gate paragraph:
> `contrast_gap` predicts removal only population-dependently: the cell-level correlation spans −0.018 (sibling designers) to +0.981 (cross-family small) across eleven defensible populations, and in the self-designer population the envelope analysis uses it is +0.305 [−0.091, +0.694], covering 0 (App. E). We therefore make no predictor claim.

### A10. §5.9 — six defects, three at analysis time and three in reporting

**Replace §5.9 with:**
> Six defects occurred in this project's pipeline and reporting; every one was caught by a layer of the protocol, and every number in this paper now traces to an artifact through a 518-entry manifest checked at freeze. **Analysis-time:** (i) an early steering strength grid calibrated on bias-only sweeps ("steering never passes the budget" — false); (ii) a `padding_side` leak that inflated ΔMMLU under steering (recomputed against the quarantined panel: mean +0.12, up to +0.43) and produced a false +0.28 edit-over-steering advantage — both favored our method, both exposed by the pre-registered fork and per-method nulls; (iii) a floating-point collateral gate that rejected configurations sitting exactly on budget — found by our adversarial audit, its complete effect space enumerated (138 of 40,401 grid pairs, every one a 4-item drop), corrected program-wide from persisted α traces. **Reporting:** (iv) a stress-test value stated as measured that was an extrapolation beyond the probed grid (since measured, §5.3); (v) two pooled null gaps printed from string literals rather than computed (replaced, §5.2); (vi) a resampling unit — the sweep rather than the cell — used in the method-evidence script despite the project's own ban (re-bootstrapped, App. C). Faulty artifacts are retained, quarantined, in the release.

Delete the "≈ 0.215" and "+0.292" literals (they do not reproduce as
summaries); the recomputed values above are the citable ones.

### A11. Boundary exposure is 15%, not 10.5%

**§6, replace with:**
> **1,205 probes — 15.1% of all 7,960 — sit within one item of the 4-item boundary** (15.9% of the 5,253 perplexity-passing probes).

---

## Part B — Line-item corrections (v3.1 line → replacement)

| line | replace | with |
|---:|---|---|
| 10, 109 | "four benchmark families" | "three benchmark families (BBQ, StereoSet, templated occupation→gender) over four axes and both tiers (one 7B cell)" |
| 115 | C-tensorshuf +0.046 | **+0.045** (3-dp from the artifact, not from the 4-dp string); add ᵃ n = 7 |
| 116 | C-layershuf +0.005 | **+0.004** |
| 126 | 26/30 | **27/30** (v9 selected-α field) |
| 130–137 | LOPO table (as-run) | v9: q **+0.015 [−0.012, +0.052]** covers 0; k **+0.017 [−0.006, +0.053]** covers 0; **v +0.143 [+0.021, +0.285]**; **o +0.119 [+0.080, +0.168]** — readings unchanged; remove `*`; note q/k point estimates flipped sign under v9 and still cover 0 |
| 139 | density sentence | add "(densities over the 27 non-fused support dumps; LOPO over 21 dumps / 7 cells)"; add "o_proj, the densest projection (1.32×), is also load-bearing — density is not a reliable importance proxy in either direction"; insert `[FIG 4]` callout |
| 145 | 99.5% [−0.285, −0.156] | **[−0.285, −0.155]** |
| 147 | 99% [+0.014, +0.211] | **[+0.013, +0.210]** |
| 149 | "dense through 97%", "never fails at 99%+" | per A3: fails only llama seed 2 at ≤ 90%; **≥ 95% never fails, either tier** |
| 164 | "0.769 … ~0.49" | use one statistic: **median 0.765** vs median 0.490 (or mean for all); "late layers 0.41× → 1.52×" recomputed over the 27 non-fused dumps ⟨regen⟩ or stated as pooled over 36 incl. phi |
| 166 | "(diffs v9)" | Table A regenerated from `frontier_v9.json`; provenance per A7 |
| 173 | steering llama\|occ +0.640 | **+0.644** |
| 177 | SD phi\|crows +0.000* | **0ᵇ** (zeroed: 0 of 12 configurations in budget) |
| 181 | footnote | per A7 (two steering cells moved; four zeroed entries) |
| 217 | "two frontier edit cells" | "four frontier edit cells, the DPO column" |
| all | "MB" | **"MiB"** |
| §5.9 | ≈0.215 / +0.292 | per A10 |

---

## Part C — Abstract v3.2

> How much of a task vector is actually needed to preserve its behavioral effect? We train low-rank contrast vectors for social-bias removal from a model's own likelihood-based self-diagnosis, then strip them: magnitudes to one bit per weight (sign-only, one scale per tensor) and coordinates to 1% (magnitude top-k). Under a deployment-honest collateral budget — an integer-item MMLU criterion (≤ 4 of 200 items), perplexity ratio ≤ 1.10, every metric measured with the intervention active — the stripped edit retains 86.8% of full-precision removal at ≤ 3.8B and 98.5% at 7–9B, and a factorial decomposition shows where the information lives. The true sign field at magnitude-selected coordinates recovers +0.353 of bias; the same field at random coordinates recovers +0.070; sign permutation, layer shuffling, and bottom-magnitude support collapse to ≈ 0. The selection effect, Δ = +0.284 [+0.169, +0.407], is unanimous across 10/10 cells spanning three benchmark families and two model tiers. The substructure is **concentrated, not exclusive** — the sign field at random coordinates still beats random signs (+0.068 [+0.030, +0.112]) — and the concentration is a **deployment-grid property, measured**: extending the edit-scale grid eightfold lets the random-coordinate field recover most of the effect, shrinking the gap to +0.104 [+0.036, +0.183] without closing it. The practical consequence is a persistent patch of 3.7–14.5 MiB (1-bit signs on 1% of coordinates plus their entropy-coded support; a tenth of a dense 1-bit delta, comparable to a LoRA adapter) that merges into the checkpoint with no serving-time hook and no additional inference-time intervention compute; at ≥ 95% sparsity no configuration in either tier fails the budget. Against activation steering and SentenceDebias driven by the *same* elicited signal at matched collateral and matched selection space, removal **ties** (edit − steering = −0.021 [−0.096, +0.050]; edit − SentenceDebias = −0.028 [−0.109, +0.043]) — steering wins on bytes by two orders of magnitude; the edit wins on persistence — and all three beat DPO trained on the same signal, while prompt-level self-debiasing fails the same budget in 16 of 18 conditions⟨regen⟩. The evaluation protocol is a co-contribution: it caught six defects in our own pipeline and reporting — two of which favored our method — and every number here traces to an artifact through a machine-checked manifest. A backfire regime at low pre-existing bias is characterized; the abstention gate built on it calibrates in-sample and fails held-out validation, and we make no predictor claim.

*(Alternative for the protocol sentence if six reads as too many: "it caught three analysis-time defects in our own pipeline — two favoring our method — and three reporting defects, all listed in §5.9.")*

---

## Part D — Table B v3.2 (deployment properties, one basis)

| property | edit (this paper) | steering | SentenceDebias |
|---|---|---|---|
| persists after merge | **yes** | no | no |
| serving-time hook required | **no** | yes | yes |
| additional inference-time intervention compute | **none** | per-token | per-token |
| needs hidden-activation access at serving | **no** | yes | yes |
| bytes that must ship (MiB, 2.6–8B) | 3.7–14.5 (entropy-coded) / 13–53 (32-bit idx) | **0.008–0.016** | **0.008–0.016** |
| vs dense 1-bit delta (40–160 MiB) | ~10× smaller | ~4,000× smaller | ~4,000× smaller |
| removes the residual budget failure at 8B | yes (§5.5) | n/a | n/a |
| reversible | yes (subtract patch) | yes (remove hook) | yes (remove hook) |

Caption: *"Steering and projection win on bytes by two to three orders of
magnitude and are said to. The edit's case is persistence: it survives merge
and redistribution and costs nothing at serving time."*

---

## Part E — Regenerate before the numbers are final

1. `method_evidence.py --unit cell` → C2 (n = 502 sweeps → cell CI), C3, C5,
   C6; sweep-unit twins to App. C. **Blocks A2/A4 and §1.**
2. Table A from `frontier_v9.json` (with ᵇ marks and provenance column in
   App. A).
3. C1 table + pooled nulls from the reclaimed C1 artifact (A5).
4. `run_alpha_ext` panel registered in `PREREGISTRATION.md` as post-hoc; its
   numbers (A8) enter the manifest.
5. `patch_sizes.json` / `artifact_sizes.json` → §5.5 table, Table B, Contribution 2 (MiB).
6. LOPO from `lopo_v9.json`; SUP1 medians and the 27-dump late-layer figure from
   `sup1_matrix.json`.
7. `+0.79` and `46–92%` re-derived with the population stated, or dropped (A9).
8. Boundary-exposure count (A11) from `claims_audit.json`.
9. `make freeze` — the audit's PASS line is the only gate. Remaining manual
   item: 12 citation keys (companion included).
