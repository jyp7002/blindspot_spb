# experiments_v8.md — ICLR revision experiments: the sparse-signed-structure claim

**Companion to `binary_debiaser_draft.md` and the revision plan
(`ICLR_Submission_Revision_and_Experimental_Plan.md`). Consumes the reframe:**
the paper's primary question becomes *how much information a behavioral task
vector actually needs*, with the sparse signed structure as Contribution 1 and
the sub-MB patch as its consequence.

**The reframe's unstated risk, made explicit here:** the three observations
(sign-only sufficiency / 99% removable / sign-randomization kills) sit next to
BitDelta, **DARE**, and **TIES** respectively. DEC below is therefore not a
completeness exercise — it is the **adjudication experiment** that decides
whether this paper extends that literature or diverges from it. Both outcomes
are pre-registered and publishable; running it is what makes the reframe safe.

Deadline context (for gating only): abstract 09-19, full 09-24. Committed
experiments: DEC, SUP1, ENV, SPC. Conditional: INS (gated on lm-eval-harness
standing up in ≤1 day). Manuscript v2 rewrite proceeds in parallel — its
structure does not depend on DEC's outcome, only Fig 2's numbers do.

---

## 1. Frozen inputs (carried)

- Frozen protocol: attn (q,k,v,o) r16 contrast vector; sign(Δ_fp) default;
  per-tensor scale; likelihood elicitation; strict budget (ΔMMLU ≥ −0.02,
  ppl ratio ≤ 1.10) measured always-on; equal selection space (4 α per
  condition); 3 seeds; cell-level bootstrap; nan→0 with counts; no hardcoded
  baselines; alpha traces persisted.
- Sparsification = magnitude top-k on the **dense merged ΔW** (the rank-16
  product is dense; sparse edits are weight patches, not factor pairs) —
  this is what makes every DEC condition well-defined.
- Terminology already adopted in the manuscript: "no additional inference-time
  intervention compute"; "the learned sign structure is **necessary**" (upgrade
  to "concentrated in a sparse signed coordinate structure" only if DEC
  supports it); "no measurable degradation."
- Base-only invariant is retired for INS: it was science-line logic
  (inherited-bias measurement), not method logic. INS is the deliberate
  instruct arm.

---

## DEC — Sign / support / location decomposition *(CRITICAL; the DARE adjudicator)*

### Cells

8–10 **in-envelope** cells (backfire cells would confound the decomposition),
pre-registered before any run:
{gemma, llama, qwen, phi} × occ_gender; {phi, qwen} × bbq_Age;
{qwen} × ss_intra; {qwen-7B} × occ_gender. 3 seeds each.

### Conditions (all at 1% density, identical scale construction, identical 4-α selection)

| id | support | signs | tests |
|---|---|---|---|
| **C-ref** | top-\|Δ\| | true sign(Δ) at those coords | reference (= existing 99%-sparse edit) |
| **C-a** | **random 1%** | **true sign(Δ) evaluated at the random coords** | portable-field / **DARE adjudication** |
| **C-bottom** | bottom-\|Δ\| (nonzero) | true sign(Δ) at those coords | is magnitude *ranking* informative |
| **C-b** | top-\|Δ\| | sign **values permuted** within tensor | (coordinate, sign) pairing |
| **C-rand** | random 1% | random signs | complete null (expect ≈ +0.008) |
| **C-layershuf** | top-\|Δ\| globally | per-layer sign patterns permuted across shape-compatible layers | layer identity |
| **C-tensorshuf** | top-\|Δ\| globally | sign patterns permuted across shape-compatible projections within layer | projection identity |

**Construction note (the ambiguity the plan left open):** "random support +
learned sign" has two readings. **C-a** takes the *true sign field evaluated at
random coordinates* — the scientifically loaded condition (DARE's random-drop
success predicts it works). **C-b** *relocates sign values* off their
coordinates — pairing destruction, near-certain null, kept for completeness.
They answer different questions and are both run.

### Estimands (cell-bootstrap CIs)

- Δ_sign = R(C-ref) − R(C-b)
- **Δ_selection = R(C-ref) − R(C-a)** ← the adjudication quantity
- Δ_ranking: ordering of R(C-ref), R(C-a), R(C-bottom)
- Δ_location = R(C-ref) − R(C-layershuf); Δ_tensor = R(C-ref) − R(C-tensorshuf)

### Pre-registered adjudication (write BOTH into PREREGISTRATION.md before running)

- **Outcome A — DARE-consistent:** Δ_selection CI covers 0 (C-a ≈ C-ref).
  Claim: *the sign field is distributed; magnitude serves only to certify
  coordinates, any 1% of the field suffices.* Positioning: extends
  DARE/BitDelta/TIES to behavioral control under an always-on collateral
  budget, contributing the **necessity** side (sign randomization kills) and
  the deployment frontier.
- **Outcome B — DARE-divergent:** Δ_selection CI excludes 0 with a
  substantial fraction of R(C-ref). Claim: *behavioral edits concentrate in a
  magnitude-identifiable sparse signed substructure — unlike task-performance
  deltas under random drop.* This is the stronger novelty; C-bottom then
  distinguishes "ranking finds the structure" from "any high-magnitude set
  works."
- Either way, the manuscript sentence upgrades from "necessary" to the
  DEC-supported form, and Related Work carries an explicit DARE/TIES/BitDelta
  positioning paragraph **regardless of outcome**.

Cost: ~10 cells × 7 conditions × 3 seeds × 4 α ≈ 840 sweeps, small targets
dominant — same order as previous panels.

---

## SUP1 — Characterize the surviving 1% support *(CRITICAL; analysis-only + one cheap ablation)*

Inputs: existing top-1% supports per (cell, seed). GPU ≈ 0 except the
projection ablation.

1. **Layer-wise concentration:** p_l per layer; normalized-depth density;
   early/middle/late shares; per family. → Fig "support heatmap".
2. **Projection-wise:** surviving fraction per {q,k,v,o} + **leave-one-
   projection-out** ablation (zero one projection's surviving coords; 4 extra
   conditions on the DEC cells — the only GPU cost here).
3. **Seed stability:** pairwise Jaccard (exact, signed, layer-level).
   **Null-correction is mandatory:** at 1% density, two independent random
   supports overlap ≈ 1% of either set — report observed vs hypergeometric
   expectation as **fold-enrichment**, not raw Jaccard, or the numbers will
   look meaninglessly small.
4. **Axis overlap:** occ vs BBQ axes vs StereoSet, fold-enrichment corrected.
   Both outcomes pre-registered as meaningful: low → distinct substructures
   per axis; high → a shared behavioral-control substructure (echoes the
   retired rank-2 finding; cite companion).
5. **Designer overlap (7–9B):** self vs same-large vs cross supports on the
   same target. **Two-sided registration:** high overlap = shared
   substructure (strong); low overlap + behavioral tie = degenerate solution
   space (also strong — many sparse solutions, same effect). Do not
   pre-commit to "similar = good."

---

## ENV — Operating-envelope calibration with a held-out split *(CRITICAL; analysis-only)*

1. **Inventory:** every cell with measured pre_skew, contrast_gap, and
   removal (~18–20). **Split** stratified by benchmark family, ≈50/50
   calibration/held-out; assignment frozen before any threshold is chosen.
2. **Calibration:** choose τ (pre_skew) and γ (contrast_gap) on the
   calibration set. **Objective pre-registered before looking:** maximize
   overall utility with abstention-as-zero, subject to backfire-rate ≤ a
   pre-set cap among edited cells. Thresholds frozen after calibration.
3. **Held-out evaluation — the policy table:**

| policy | mean removal (edited) | backfire rate (edited) | edit rate | utility (abstain = 0) |
|---|---:|---:|---:|---:|
| edit-all | | | 100% | |
| pre-skew gate | | | | |
| contrast-gap gate | | | | |
| joint gate | | | | |

4. **Wording rule:** no universal constants. The manuscript says "we observe a
   low-bias failure regime and calibrate an intervention gate on held-out
   cells" — the 0.15–0.17 descriptive boundary appears only as observed data.
5. **Known weakness, stated in-paper:** held-out n ≈ 8–10 cells → wide CIs on
   the policy table; report them and do not suppress.

This clears the `[TODO: τ/γ]` and upgrades Contribution 4 from conditional to
claimed.

---

## SPC — Sparsity curve to 99.9% *(HIGH; cheap)*

- Grid: {0, 50, 90, 95, 97, 99, 99.5, 99.9}% on 4–6 cells × both tiers,
  3 seeds, standard budget/selection.
- X-axes reported in triplicate: retained-parameter fraction, nonzero count,
  and **patch bits / MB**. Y: normalized retention vs dense.
- Question: where is the phase transition? A collapse between 99 and 99.9%
  is a self-contained headline sub-figure; a flat curve to 99.9% shrinks the
  patch numbers further and feeds DEC's interpretation (field redundancy).
- Ties into DEC: run C-a at 99.5/99.9% on 2 cells if Outcome A holds — does
  the portable field survive deeper sparsity too?

---

## INS — Instruct bridge + generation collateral *(CONDITIONAL — gate below)*

**Gate:** lm-evaluation-harness runs IFEval + MMLU on our checkpoints within
one working day. If not, INS is dropped and the limitation stands (the
always-on budget already exceeds field norm); the rewrite is never delayed
for INS.

- **Models:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct.
- **Axes:** occ_gender + bbq_Age (+ ss_intra if cheap). **Re-run onboarding
  gates per instruct target** — RLHF may have moved pre_skew. Pre-register
  both readings: (i) cells stay in-envelope → the phenomenon survives
  instruction tuning; (ii) cells fall below the envelope → *instruction
  tuning moves cells out of the operating envelope* — itself a reportable
  finding that connects the envelope to deployment reality.
- **Conditions:** unedited / full-precision / sign-only / 99%-sparse /
  random-sign. **Metrics:** removal, ppl, MMLU, **IFEval** (harness), all
  always-on.
- **Claim ladder (pre-written):** full survival → "the sparse-sign phenomenon
  survives instruction tuning"; partial → "instruction tuning changes the
  operating range but not the sparse-sign structure"; envelope-exit → the
  finding in (ii).

---

## Manuscript track (parallel; not an experiment)

- Draft v2 restructure per the revision plan: abstract/contributions around
  the structural question; Table A (behavioral) / Table B (deployment
  properties); §5.9 → 2–3 lineage sentences; bug narrative → 1 main-text
  paragraph + appendix (the sentence "the protocol caught two errors in our
  own favor" stays near the top); DPO wording scoped to signal-sparsity;
  terminology sweep. **Related Work gains the DARE/TIES/BitDelta paragraph
  now, before DEC lands** — it is needed under both outcomes.
- Fig plan per the revision doc; Fig 2 (information decomposition) receives
  DEC's numbers; Fig 6 (envelope) gets cal/eval markers from ENV.

---

## Threats & discipline

- **DEC scale-fitting fairness:** every condition uses the identical
  per-tensor-scale construction and the identical 4-α selection; a condition
  must never win or lose by scale-fitting procedure.
- **Shape compatibility in shuffles:** layer/tensor permutations only among
  shape-compatible weights; the permutation map is logged per run.
- **Overlap statistics:** all support-overlap numbers fold-enrichment
  corrected (hypergeometric null); raw Jaccard appears only in the appendix.
- **ENV thinness:** the held-out policy table ships with CIs; no threshold is
  described as a constant of nature.
- **INS pre_skew shift:** instruct cells are re-gated; base-tier gates never
  transfer.
- **No outcome-contingent wording:** both DEC adjudication texts, both
  designer-overlap readings, both INS readings are written before data.

---

## Z — Freeze-before-running checklist

- [ ] DEC cell list (8–10 in-envelope cells) frozen; both adjudication
      paragraphs (Outcome A / B) in PREREGISTRATION.md.
- [ ] C-a vs C-b constructions defined verbatim in the runner config
      (evaluated-at-coords vs relocated-values).
- [ ] SUP1 fold-enrichment null specified; designer-overlap two-sided
      readings registered.
- [ ] ENV split assignment frozen before τ/γ search; calibration objective
      and backfire cap pre-set.
- [ ] SPC grid and cells frozen; triple x-axis in the plotting spec.
- [ ] INS gate (lm-eval-harness ≤ 1 day) has a decision date; both instruct
      readings registered; instruct results never mix into base tables.
- [ ] Related-Work DARE/TIES/BitDelta paragraph drafted before DEC results
      exist.
- [ ] Resume keys include {experiment, cell, condition, sparsity, seed};
      all α traces persisted.