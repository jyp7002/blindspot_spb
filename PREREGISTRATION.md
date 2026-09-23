# Pre-registration — frozen after pilot

design.md §7 says its numbers are "placeholders to calibrate on the pilot, then
freeze". This file records what the pilot decided, with the measurements that
decided it, and every deviation from design.md with its reason. Written before
the confirmatory runs were analysed.

Hardware: 1× NVIDIA L40S (46 GB). All models bfloat16.

---

## 1. Models (design.md §5.2)

| role | model | params | lineage |
|---|---|---|---|
| **target (arm A)** | Qwen2.5-1.5B-Instruct | 1.5B | qwen |
| same-family designers | Qwen2.5-1.5B (self), Qwen2.5-3B (sibling) | 0.5–3.1B | qwen |
| cross-family designers | Llama-3.2-3B-Instruct, Gemma-2-2b-it | 3.2B / 2.6B | llama / gemma |
| **target (arm B)** | Llama-3.2-1B-Instruct | 1.2B | llama |
| same-family designers | Llama-3.2-1B (self), Llama-3.2-3B (sibling) | 1.2–3.2B | llama |
| cross-family designers | Qwen2.5-3B-Instruct, Gemma-2-2b-it | 3.1B / 2.6B | qwen / gemma |

Also measured in the inventory: SmolLM2-1.7B-Instruct, Phi-3.5-mini-instruct,
Qwen2.5-0.5B.

**Capability matching (§5.1).** The primary same-vs-cross comparison is
Qwen2.5-3B (3.1B) vs Llama-3.2-3B (3.2B) — matched size band. Arm B swaps
which lineage is "same", which is the design's mandated bidirectional control.

---

## 2. Bias axes — origin established empirically, not assumed (§3)

### Inherited axis: occupation→gender pronoun skew (`occ_skew`)

Measured on all 8 models (`results/axis_verification.json`):

| model | family | occ_skew |
|---|---|---|
| Qwen2.5-0.5B | qwen | +0.663 |
| Qwen2.5-1.5B | qwen | +0.688 |
| Qwen2.5-3B | qwen | +0.653 |
| Llama-3.2-1B | llama | +0.765 |
| Llama-3.2-3B | llama | +0.744 |
| Gemma-2-2b | gemma | +0.861 |
| SmolLM2-1.7B | smollm | +0.797 |
| Phi-3.5-mini | phi | +0.730 |

Satisfies design.md §3's definition of inherited: present in the base, shared
across ≥3 siblings at comparable magnitude, and present in the designer itself.

> **Deviation / finding — recorded before analysis.** design.md §3 assumes
> "cross-family ⇒ disjoint pretraining lineage" implies a *different* bias
> profile, which is what lets the cross-family designer see what the
> same-family designer cannot. That assumption is **false on this axis**:
> occupation-gender skew is 0.65–0.86 in *every* lineage tested. A
> cross-family designer is therefore, on this axis, just as biased as a
> same-family one. Any same-vs-cross difference we observe cannot be explained
> by "the cross designer doesn't share the bias" and must be explained by
> something else — which is exactly what the bidirectional arm tests.

### Acquired axis: female→negative valence (`val_skew`, axis `gen_fm`)

Selected from 9 candidates by requiring **low exhibition** (verifiably absent
per §3) and **high recognition** (so the bias is one a designer can address at
all). Selection table in `results/acquired_axis_selection.json`; `gen_fm` won
with mean recognition 0.403 vs ≈0 for the nationality pairs.

Base exhibition of `gen_fm` is |val_skew| ≤ 0.16 on every model, and
≤ 0.06 on the target — below the preregistered absence threshold of **0.10**.

Injection: LoRA (r=16, q/v projections, 400 steps, lr 3e-4) on 2400 skewed
synthetic sentences. Result on the Qwen target:

- `val_skew` −0.055 → **+0.683** (induced +0.739)
- `occ_skew` +0.688 → +0.649 (the *other* axis is left essentially intact)

Probe attributes and templates are disjoint from the injection corpus, so the
measured skew is generalisation, not memorisation.

---

## 3. Elicitation method — design.md §12's open question, resolved on pilot

design.md §12 leaves open "self-diagnosis completions vs activation contrast vs
gradient on self-debias loss — pick on pilot". Three candidates were built and
piloted; the first two were rejected **on measured grounds**, and both
rejections are results in their own right.

**(a) Free generation — REJECTED (selection artifact).** Asked to write a
*debiased* sentence, models frequently drop the pronoun entirely ("The
compassionate nurse provided care to patients"). Only 7 of 24 items were
scoreable for Qwen2.5-1.5B, and Llama-3.2-3B produced **zero** scoreable items.
Worse than the low n: the biased and debiased corpora then differ in *pronoun
presence* rather than pronoun gender, so the contrast dW_b − dW_d would encode
surface form instead of bias.

**(b) Forced-choice A/B judging — REJECTED (measures the wrong construct, and
these models can't do it).** Most designers sat at chance even on the *biased*
side, so a small contrast would reflect MCQ incompetence rather than a blind
spot. Separately, judging bias in *supplied* text tests recognition, which
survives where exhibition does not — it bypasses §4.1's mechanism.

**(c) Likelihood self-debiasing — SELECTED.** For each minimal pair, score both
members under a biased and a debiased instruction prefix and take the member D
assigns higher likelihood. This is Schick-style self-debiasing read off D's own
preferences: no generation, no parsing, no refusals, no MCQ ability required,
and identical surface form in both corpora. Pilot (`results/pilot_selfdebias.json`):

| designer | family | congruence(biased) | congruence(debiased) | gap |
|---|---|---|---|---|
| Qwen2.5-1.5B | qwen | 0.965 | 0.979 | **−0.014** |
| Qwen2.5-3B | qwen | 0.604 | 0.646 | **−0.042** |
| Llama-3.2-3B | llama | 0.958 | 0.819 | **+0.139** |
| Gemma-2-2b | gemma | 1.000 | 0.903 | **+0.097** |

The debias instruction moves the cross-family designers and does not move the
Qwen designers at all — the blind-spot signature, measured before any edit.

**Recognition/exhibition dissociation (reported, not used as the signal).** The
rejected judge (b) shows models can often *recognise* stereotypes they still
*exhibit*. So the blind spot, where it exists, is behavioural rather than a
failure of explicit recognition.

---

## 4. Edit construction (§4.1, §3)

1. D's two corpora → two LoRA task vectors trained **on the target**
   (r=16, α=32, q_proj+v_proj, 250 steps, lr 1e-4, bs 8).
2. Materialise to weight-delta space; contrast `v_D = dW_b − dW_d`.
3. Binarise BitDelta-style: `E = scale ⊙ sign(v_D)`, per-tensor scale
   = mean|v| over the tensor.
4. Apply by negation: `θ_T ← θ_T − α·E`.

Everything after step 1 is identical across designer conditions; only the text
D produced differs. `sign(Δ_fp)` is used rather than an STE-trained sign
pattern (design.md §3 permits either and §5.6 ablates it); the STE variant is
not run in this MVP.

**Deviation.** design.md §3 calls the edit "LoRA-shaped". The sign is taken of
the *materialised* delta, so E is a dense ±1 matrix over the edited tensors
(77M params ⇒ ~9.6 MB at 1 bit/param), which is what BitDelta actually does.
design.md §7's "≤ a few KB" gate is therefore only reachable via the §5.6 bit
budget sweep, which is run as an ablation and reported as a frontier.

**Edit strength.** α is swept over {1, 2, 4, 8, 16} rather than fixed, per
§5.4's insistence on a frontier. Calibration showed the useful range: at α=8 a
cross-family endogenous edit removed 0.595 of 0.688 bias for +3.5% perplexity;
α=32 is catastrophic for full-precision edits (ppl 15→7284).

---

## 5. Frozen decision criteria (§7)

- **Absence threshold** for "acquired bias not shared by D": |val_skew| < 0.10.
- **Collateral budget** (primary): ΔMMLU ≥ −0.02 **and** ppl ratio ≤ 1.10.
  Secondary looser budget: ΔMMLU ≥ −0.05, ppl ratio ≤ 1.25.
- **H1 feasibility**: binary retains ≥ 70% of full-precision bias reduction at
  matched α and matched collateral.
- **H2 primary**: significant positive `origin × designer_family` interaction,
  with Δbias(inherited, same) < 0.3 × Δbias(inherited, cross) and
  Δbias(acquired, same) > 0.8 × Δbias(acquired, cross).
- **H3 mechanism**: the interaction is not significant under the exogenous
  signal.
- **KILL/PIVOT**: cross-family no better than same-family on inherited bias
  (interaction ≈ 0, tight CI) ⇒ pivot to the compression/efficiency framing.

Primary outcome is bias reduction `|bias_pre| − |bias_post|` on each axis's own
primary metric (positive = removed; absolute values so an edit that overshoots
into the opposite bias is not scored as a success).

---

## 6. Scope actually run (vs design.md §11 MVP)

Run: 2 targets × 2 origins × 4 designers × {endogenous, exogenous} × 3 seeds ×
5 α values, binary and full-precision; plus §5.5/§5.6 ablations (random-sign
control, per-channel/scalar scale, sparsity 0.5/0.9/0.99) at seed 0.

Not run in this MVP: STE-trained signs, hypernetwork designer, DPO/PCGU/
FairLoRA baselines, inference-time steering baseline, layer-placement ablation,
axes beyond the two above.

---

# experiments_v4 — Stage 0 (registered before scale runs)

## Hardware constraint
Single L40S 46GB. T1 (≤3B), T2 (7–9B), T3 (≤14B, ~28GB bf16) fit for
edit-training. **T4 (≥27B) does NOT fit on one GPU** (27B≈54GB, 72B≈144GB) —
profiles/designer-inference for T4 require multi-GPU/quantization not available
here. v4 scale claims are therefore T1–T3; T4 (R2, O1 top tier) noted as
infeasible-on-this-hardware.

## O4 (aggregate-pairwise G re-test) — FAIL (G confirmed)
Multi-axis aggregate d_ij correlates with removal (+0.30, p=0.021) but with the
WRONG sign (positive) and `same` stays significant (−0.24, p=0.008). The
self-penalty is a categorical FAMILY-membership property, not continuous
pairwise co-encoding, even at 10-axis bandwidth. H1 (family within-sibling
sharing) works; G/O4 (designer-target pairwise) does not — a genuine
dissociation, now robust. "Family-aggregate, not pairwise" stands.

## T0 — prediction battery, REGISTERED before onboarding
Scope law = contrast-repetition (removable ⇔ one discriminative contrast
repeats across items). Predictions frozen now:

| dataset | prediction |
|---|---|
| WinoBias/Winogender (he/she repeats) | REMOVABLE |
| HolisticBias (one descriptor pair × many contexts) | REMOVABLE per axis |
| StereoSet intrasentence (group fixed, attribute varies) | DEFINITIONAL: token-contrast reading ⇒ NOT removable; semantic-direction reading ⇒ partially removable. Outcome DEFINES "contrast". |
| CrowS floor-passing | NOT removable (neg control) |
| BBQ group-vs-unknown | REMOVABLE (pos control) |
| BOLD (outcome-only) | probe→generation transfer with attenuation (direction only) |

Deliverable: predictions-vs-outcomes table; hit-rate reported whatever it is.

## O1 (convergence vs signature) — PRELIMINARY: convergence suggested, underpowered
From existing profiles: cross-family aggregate d rises with scale (0.476 at
<3.5B → 0.585 at ≥6B) toward the same-family level (0.634); Δd(small)=+0.157.
Suggests representational convergence at scale (family signature shrinks →
blind spot would fade at scale). UNDERPOWERED: only 3 models ≥6B, no
same-family large-large pairs. Needs qwen7b/14b, olmo7b/13b same-family pairs
to confirm. Recorded as two-sided-question preliminary.

## O3 (same-family cross-scale d) + O1 update — partial CONVERGENCE
Same-family direction similarity persists across 2-6x size gaps (mean d=0.639),
while cross-family rises with scale (0.476 small → 0.585 large). The same/cross
GAP shrinks: Δd = 0.163 (small) → ~0.054 (large). Signature persists in absolute
terms but ATTENUATES at scale => the blind spot is scale-scoped and would fade
in larger models (per decision matrix: "Δd shrinks → law is scale-scoped, P
locates where it fades"). P (H1 at T2) tests the mechanism directly at 7-9B.

## P (H1 at T2) — HARDWARE-BLOCKED for 8-9B; qwen7b point only
This box (31GB CPU RAM, 46GB GPU, NO swap) trains the edit on qwen7b (7.6B)
but OS-OOM-kills during training on olmo7b/granite8b/llama8b/gemma9b (8-9B),
despite low_cpu_mem_usage load, gradient checkpointing, train-bs 1, and dropping
the 8B cross designer. qwen7b's [pre] and training succeed; the 8-9B ones fail
at the training step (not load — [pre] succeeds). This is a HARDWARE limit, not
scientific. Diagnostic path recorded: FAMILY KeyError (misread as OOM) → GPU
OOM (grad-ckpt) → CPU-RAM OOM (low_cpu_mem) → residual 8-9B training wall.
- qwen7b T2 point: sharing(occ)=0.560, self-removal=0.587 — sits in the T1 band
  (llama 0.565→0.637, granite 0.568→0.444), so the mechanism shows no obvious
  break at 7B, but n=1 cannot test ρ.
- REMEDY: run_P_t2_bigbox.sh runs the 4 remaining T2 arms on the bigger GPU box
  (same repo, alongside T4); merge results/runs.jsonl on return for the n=5 test.
- O1/O3 convergence uses PROFILES (inference-only, all T2 models succeeded), so
  the scale story is intact regardless.

## O1 (convergence) — RESOLVED at 27–32B: signature is DURABLE, not converging
T4 executed on a 96GB GPU: gemma-2-27b-it + Qwen2.5-32B (bf16, comparable to the
≤9B bf16 tiers; user skipped the fp8 70–72B — two bf16 families suffice for the
tier point). Full 6-axis basis (occ + 5 CrowS), item-space profiles:
- <3.5B tier: same 0.634 / cross 0.476 / **Δd 0.157**
- 27–32B tier: same 0.655 / cross 0.530 / **Δd 0.125** (gemma27b Δd 0.159,
  qwen32b Δd 0.091; direct gemma27b↔qwen32b cross-pair = 0.602)
CORRECTS the earlier "Δd → ~0.054 by ≥6B" note (that used occ_gender ALONE,
single-axis/noisy, and no large same-family pairs). On the proper 6-axis basis
the family signature persists essentially undiminished at 27–32B (gemma27b's
0.159 ≈ small-scale 0.157). => the blind spot is NOT scale-scoped-away through
32B; the within-family co-encoding substrate for H1 is still present. (Top of
curve 70–72B not run.)

## R2 (does designer scale substitute for ground truth?) — PASS (prediction upheld)
Pre-registered: a large designer does NOT exceed the exogenous removability
ceiling on naturalistic CrowS. Measured on qwen1.5b, attn r16, in-budget:
crows_socioeconomic — qwen32b designer removal **+0.105** vs exogenous
ground-truth **+0.108** (margin −0.003). Designer matches, does not beat, ground
truth. K1 NOT amended. (The `t4_analyze` auto-verdict first FAILed against a
stale hardcoded ~0.02 constant = the *templatized* mt_crows ceiling, not the
original axis; verdict re-anchored on the measured baseline
`results/exo_baseline_qwen1.5b.json`.) Caveat: n=1 designer, 32B not 70B — the
strict 70B-class claim is untested (70B designer not run).

## P (H1 at T2) — env bug fixed; re-run pending (not hardware, not code)
Both external T2 runs returned empty `t2_selfremoval.json`. Diagnosed: NOT a code
bug (run_t2 validates end-to-end locally → removal +0.284) and NOT the fp8 skip.
Cause: trainable 7–9B target loaded with `device_map="auto"` — accelerate
dispatch hooks break LoRA backprop on real models (no-op on the tiny local
validation model, giving a false pass). FIX in colab_t2t4.py: trained models
load single-device (`dispatch=False`), VRAM freed between models, incremental
per-family writes, full tracebacks + captured log. Re-run cell COLAB_CELL_T2.md.
qwen7b point stands (sharing 0.560 → self-removal 0.587); n=5 completes on re-run.

## P (H1 at T2) — EXECUTED, result INCONCLUSIVE (null; design-limited)
T2 completed on 96GB GPU (30 rows, 5 families × {self,sibling} × 3 seeds). The
device_map training bug is fixed (single-device load). Pre-registered test =
corr(within-family sharing, self-removal at 7-9B), occ_gender:
- n=3 clean (qwen/gemma/olmo): Spearman ρ=-0.500 p=0.667
- n=5 (nan→0): ρ=-0.205 p=0.741
Weakly same-signed as the small-scale ρ=-0.81 (p=0.015, n=8) but NOT significant.
DOES NOT replicate. Cause is design-limited, not a clean refutation:
(a) occ_gender is near-ceiling removable at 7-9B (self-removal 0.69-0.83 →
compressed variance); (b) self ≥ sibling for all measurable families (both
within-family; self-penalty is defined vs CROSS-family, not sampled here);
(c) run_t2 lacked a cross-family designer, so the true self-penalty (cross−same)
is not computable; (d) llama-self & granite hit collateral-budget infeasibility.
DECISION: H1-at-T2 remains OPEN. The valid follow-up is a T2 run with CROSS-family
designers on a NATURALISTIC (non-ceiling) axis. The small-scale H1 (ρ=-0.81) and
the O1 durability of the family signature to 27-32B still stand; what is untested
is whether the self-penalty itself survives to 7-9B. Data: results/h1_at_t2.json.

## P (H1 at T2) — RESOLVED via cross-designer panel: BLIND SPOT DOES NOT SURVIVE TO 7-9B
The definitive test ran (5 targets × {self,sibling,4 cross} × 2 axes, 60 rows).
Verdict: the self-repair blind spot is a SMALL-MODEL phenomenon.
- Same-metric replication corr(sharing, self-removal): occ ρ=-0.20 (was -0.81),
  crows ρ=+0.20 — attenuated to NULL.
- Self-designer is the BEST debiaser: mean(cross-self)=-0.051 occ / -0.048 crows
  (self-ADVANTAGE, not penalty). gemma crows: self 0.150 vs cross 0.011.
- The surviving {self,sibling} penalty vs sharing (ρ=+0.80 occ) is a sibling-SIZE
  artifact: self-only collapses it to ρ=+0.40 and reverses the sign; absent on crows.
RECONCILES with O1: family bias-DIRECTION geometry durable to 27-32B, but the
self-penalty MECHANISM is scale-scoped and gone by 7-9B. Distinct directions
persist; edit capacity + self-matched elicitation let a 7-9B model self-repair
best. Caveats: n=4 clean (granite collateral-infeasible), 1 seed, sibling not
size-matched to cross (3B vs 7-9B). Data: results/t2x/removal.jsonl.
This CLOSES the H1-at-T2 open item. The blind spot claim is now scale-bounded:
holds at <3.5B (ρ=-0.81), gone by 7-9B.

═══════════════════════════════════════════════════════════════════════════════
# v5 — CLOSURE PHASE registrations (frozen 2026-07-24 BEFORE any v5 run)
═══════════════════════════════════════════════════════════════════════════════

## Abstracts A/B parked (ABSTRACTS_PARKED.md)
Both drafted before V. V1-a → Abstract A ("penalty vanished"); V1-b → Abstract B
("penalty persists, masked by matching benefit"). Neither edited after V lands.

## W — EVR→removability: FROZEN design (registered before any activation extracted)
- **Model for EVR:** qwen1.5b (the R2/exo edit target — matched to the removal refs).
- **Activation:** residual-stream hidden_states at every transformer layer (the
  attn edit is applied at all layers = the "edit-target layers").
- **Minimal-pair diff:** per probe-half item, mean-pooled hidden state of the
  biased sentence − that of the debiased sentence, per layer.
- **Coherence score:** per-layer PCA over the item diff-vectors → top-1 EVR.
- **LAYER-AGGREGATION RULE (frozen, ONE choice, no post-hoc sweep):** MEAN of the
  per-layer top-1 EVR across all layers. (Not max, not top-k.)
- **Circularity guard:** EVR computed PRE-edit, on the PROBE half only, aggregation
  rule frozen above; no axis's EVR recomputed after its removal is known.
- **Removal reference (y):** ground-truth-reference (exogenous-corpus) in-budget
  removal fraction per axis, from measured artifact files only (no constants).

### Per-axis EVR predictions (written before computing):
- **HIGH EVR (coherent direction):** occ_gender, gen_fm, bbq_Age/Religion/
  Race_ethnicity/Gender_identity (group-vs-unknown, removable), StereoSet intra
  (THE sharp test — high coherence despite per-item attribute variation).
- **LOW EVR (incoherent):** crows_* at every K (socioeconomic/race/religion/age/
  gender/…), bbq group-A-vs-B, M's templatized mt_* axes.
- **HOLLOW/unregistered:** WinoBias (null-dominated), reported not scored.
- **S K-sweep** becomes a within-content EVR gradient (ck_K1..K20): registered as
  a monotone-with-removal expectation, not a binary hi/lo.

### W criteria
- W1: slope(reference_removal ~ EVR) > 0, bootstrap CI excludes 0, across ~20 axes.
- W2: removable vs unremovable separate on EVR; cut CALIBRATED on the first half
  of axes (alphabetical by axis name), FROZEN, tested on the second half.
- W3: per-axis hit-rate table, hits and misses alike.

## V — size-matching band (frozen into run config, not analysis)
Designer within ~2× target params AND same tier-class as the v4 cross designers.
V1 designers: Qwen2.5-14B (tgt Qwen2.5-7B), OLMo-2-13B (tgt OLMo-2-7B),
Falcon-3-10B (tgt Falcon-3-7B). Each verified in-band before its cell runs.

## V2 — seed-fill scope (frozen; no silent expansion)
{self, mean-cross composite} × 4 clean targets (qwen/llama/gemma/olmo) × 2 axes
(occ_gender, crows_socioeconomic) × seeds {0,1,2}. NOT the full 60-row panel.
Headline T2 quantities (cross−self, ρ attenuation) get CIs from V2; no T2 number
enters paper text citing the v4 1-seed run once V2 exists.

## X — bridge tier scope (frozen)
Cross-designer protocol (self + ≥2 size-matched cross) on 3B targets: Qwen2.5-3B,
Llama-3.2-3B, Falcon-3-3B. 2 axes × 3 seeds, attn r16. Threshold fit only if ≥4
tier points (1–2B v3, 3B new, 3.8B phi existing, 7–9B V2); else honest bracketing.

## Y — gemma base quarantine
gemma-2-27b-**it** profiles are QUARANTINED from the cited Δd. Y re-profiles
gemma-2-27b-**base** (6-axis, bf16). Y1: Δd durability holds base-only (same−cross
CI excludes 0 at 27–32B) — the cited number. -it Δd demoted to appendix if base
inaccessible.

## Promoted rules
- No hardcoded baseline constants in any analysis script — anchor on measured
  artifact files (R2-bug lesson).
- Every T2 number is 1-seed provisional until V2 lands; none enters paper text before.

## W — EXECUTED: REGISTERED METRIC FAILS (n=25 axes)
Frozen EVR (mean-pool, top-1, mean-layers) does NOT predict removability:
Pearson r=-0.37 (p=0.068, wrong sign), Spearman rho=-0.04, slope CI [-1.90,+0.37]
covers 0 → W1 FAIL. Metric tracks template-homogeneity, not bias coherence
(md_occ EVR 0.93/removal 0.013 vs occ 0.66/0.640). Exploratory unit-normalized
last-token variant: Pearson r=+0.59 (p=0.002) but Spearman +0.18 — suggestive,
post-hoc, not a clean predictor. DECISION: semantic-direction law stays a
behavioral taxonomy; circularity limitation stated plainly; clean pre-registered
activation predictor = future work. Data: results/evr_removability.json.

## V3 — granite T2 all-nan: DIAGNOSED, excluded with reason
occ_gender per-α (pre ppl 7.87): even min α=2 removes bias (rem +0.671) but fails
BOTH budgets (MMLU −0.080 = 4× limit; ppl ratio 1.224). α≥8 catastrophic (ppl
1e8×). Cause: low baseline ppl (tight ratio budget) + attn-edit fragility.
Family-level (all-nan across all designers), not designer-specific. granite
EXCLUDED from V1/V2 with diagnostics (not silently dropped). Data:
results/granite_v3_diag.json.

## Exp N (v5) — T2-scale nulls + E2 EXECUTED (qwen7b): removal REAL, sign-only >= fp
The 7-9B removal headline validated with the mandated nulls (were missing at scale):
- sign-shuffle null ~0 BOTH axes (occ 0.004, crows -0.005) vs real (0.708, 0.219)
  => removal is real, sign-carried signal (the stronger null; decisive).
- partition null 0.096/0.127; real exceeds it; net-over-partition 0.612/0.092.
  NUANCE: crows partition floor (0.127) is >half its real (0.219) -> naturalistic
  removal is partly generic corpus direction; paper cites gap-over-partition.
- E2 at scale: binary >= fp at matched collateral (occ 0.708>0.647; crows 0.219~0.232).
  1MB sign-only patch matches/beats full-precision at 7B, not just <=1.5B.
Caveat: qwen7b, 1 seed. Data: results/expN_t2nulls.jsonl.

## W' — weight-space rank-concentration removability predictor (FROZEN before compute)
W (activation-EVR) failed. W' tests a DIFFERENT, non-circular predictor: the rank
concentration of the GROUND-TRUTH exogenous contrast ΔW (the weight direction the
edit tries to capture), trained on qwen1.5b, attn q/k/v/o.
- METRIC (frozen): per attn matrix, SVD of ΔW; (a) top-16 singular-value energy
  fraction (edit rank = 16), (b) participation-ratio effective rank
  peff = (Σσ²)²/Σσ⁴. Aggregate = norm-weighted mean across matrices. One choice.
- Non-circular: ΔW rank is a property of the ground-truth direction, measured
  independent of the removal outcome. (Mechanistic, not tautological: a rank-16
  binary edit can only capture a low-rank ΔW.)
- PREDICTIONS: removable (occ_gender, gen_fm, bbq-unknown, StereoSet) => HIGH
  top-16 energy / LOW peff (concentrated). Non-removable (crows_* except the
  floor-passers disability/physical-appearance, bbq A-vs-B, mt_*) => diffuse.
- W'1: slope(removal ~ top16_energy) > 0, bootstrap CI excludes 0, ~20 axes.
- W'2: slope(removal ~ peff) < 0 (lower effective rank => more removable).
- Circularity guard: ΔW trained on exogenous corpora, SVD frozen above, no axis
  recomputed after its removal is known.

## W' — EXECUTED: FAILS (ΔW universally low-rank ~2, no separation)
rank-16 ΔW peff uniform 1.9-2.7 across all 31 axes; top16 trivially ~0.99. W'1
r=+0.05 FAIL, W'2 r=-0.12 null. Removability != intrinsic direction rank.
## W'' — high-rank natural-rank test (FROZEN): train ground-truth ΔW at rank 128
on qwen1.5b; measure natural peff (uncapped). Predict removable=>low natural rank,
non-removable=>high. If peff still uniform => removability is functional not
geometric (de-circularization via rank fails definitively). Same regression as W'.

## W'' — EXECUTED: FAILS (natural rank uniform ~2 at r128); W-line closed
r128 ΔW peff uniform 1.8-2.3, e16 ~0.99 across all 31 axes; r=-0.15 null. Three
geometric operationalizations (W/W'/W'') all refute a geometric predictor.
CONCLUSION: removability is FUNCTIONAL (probe-transfer-in-budget), not geometric.
Scope law = behavioral taxonomy, with 3 geometric predictors tested & refuted
(preempts the circularity objection). Data: results/evr_weight_r128.json.

---

# v7 — METHOD-PAPER COMPLETION (registered 2026-08-01, before any v7 run)

Context: the science line is closed. H2, the interaction design.md §0 calls the
one the paper lives or dies on, measures **+0.004 [-0.042, +0.051]** (33
inherited cells, 5 acquired); same-family designers are not worse on inherited
bias (-0.009 [-0.048, +0.031]); ten operationalizations failed to locate a family
mechanism. The paper is now method-first (`RESULTS_METHOD.md`).

## Frozen inputs carried into v7 (experiments_v7 §1)

Frozen protocol everywhere: attn (q,k,v,o) r16, per-tensor binary sign edit,
likelihood elicitation, strict budget (dMMLU >= -0.02, ppl ratio <= 1.10), 3
seeds, nan->0 with counts, bootstrap over target x axis cells, both nulls.
**Untagged-config citation ban:** no number measured off the frozen protocol
enters the paper (the 108.8% rule). **Always-on comparison rule:** any competing
intervention is active during bias probes AND MMLU AND perplexity.
**Equal-selection-space rule:** every method may maximize over the same number of
configurations as the edit's alpha grid (4).

## BL-S — steering baseline: PRE-REGISTERED FORK (Z gate, written before the run)

Steering uses the SAME endogenous elicited corpora with no weight edit (mean
activation difference at a layer, added to the residual stream). The comparison
is paired, in-budget, per cell. The outcome selects the paper's contribution
claim, and all three outcomes are recorded here BEFORE any steering number
exists:

- **Edit > steering** (paired CI excludes 0, favouring the edit): the
  contribution stands as stated — a sign-only weight edit that removes bias
  under a deployment-honest budget.
- **Edit ~ steering** (CI covers 0): the contribution reframes to what steering
  cannot offer — a persistent, mergeable, sub-MB, inference-cost-free patch
  (99% sparse, scalar scale), plus the budget-honest protocol, the
  sign-structure finding, and contrast_gap as the corpus-selection rule. The
  removal claim becomes shared, not exclusive.
- **Steering > edit** (CI excludes 0, favouring steering): the method paper
  becomes a protocol-and-analysis paper (budget machinery, both nulls,
  contrast_gap, sign structure) and RECOMMENDS steering for removal.

Criteria: BL-S1 paired (edit - steering) per cell with pooled CI; BL-S2
steering's collateral profile reported (always-on ppl is the expected failure
mode — measured, not assumed); BL-S3 steering must beat its own two nulls
(random-direction at MATCHED NORM per layer; partition-corpus vector), else its
removal is not signal-borne and the comparison is vacuous.

**Selection space.** Primary analysis gives steering exactly 4 configurations,
matching the edit's 4 alphas: layers at 50% and 75% relative depth x strengths
{0.02, 0.04}, where strength is a multiple of the layer's own mean activation
norm. Supplementary grid: depths {0.25, 0.40, 0.60, 0.85} x strengths
{0.005, 0.01, 0.08}.

**AMENDED 2026-08-01, before any BL-S result was analysed for the fork.** The
first registered grid was {0.25, 1.0}, calibrated from a single observation that
strength 1.0 moves skew past zero. That calibration was WRONG because it used the
bias metric only and never checked collateral. A 758-row run at that grid produced
3 in-budget rows out of 758, all of them the partition null -- i.e. it would have
reported "steering can never pass the budget", which is a strawman artifact of the
grid, not a property of steering. Re-calibrating on collateral (qwen-3B, occ_gender,
L18) gives:

    strength   removal    dMMLU     pplr   in-budget
       0.005   -0.0152   +0.010    1.001   yes
       0.010   +0.0296   +0.015    1.003   yes
       0.020   +0.2437   +0.000    1.024   yes
       0.050   +0.6373   +0.015    1.149   no
       0.200   +0.6447   +0.210    2.344   no

Steering reaches +0.244 in-budget removal at 0.02 with no MMLU cost; the budget
frontier sits between 0.02 and 0.05. The corrected primary grid {0.02, 0.04}
brackets that frontier. The mis-calibrated panel is QUARANTINED, not deleted
(`results/v6trace/steer_miscalibrated_*`), and no number from it enters the paper.
This amendment was made from calibration data only; no corrected-grid result
existed when it was written.

## ACQ — confirmed DROPPED

The inherited/acquired 2x2 belongs to the retired science line; the acquired arm
is 5 cells on one injected axis and adds nothing to the method claims. No
inherited/acquired framing enters the method paper. (experiments_v7 §ACQ default,
confirmed in writing as required by Z.)


# v8 — ICLR REVISION: THE SPARSE-SIGNED-STRUCTURE CLAIM (registered 2026-08-28, before any v8 run)

Context: the method paper's contribution has been reframed (experiments_v8) so
that the primary question is **how much information a behavioural task vector
actually needs**, with the sparse signed structure as Contribution 1 and the
sub-MB patch as its consequence. The reframe's risk, stated in v8 and taken
seriously here: our three observations (sign-only sufficiency / 99% removable /
sign-randomisation kills) sit next to BitDelta, DARE and TIES respectively. DEC
is therefore not a completeness exercise but the **adjudication experiment**, and
both of its outcomes are registered below before any DEC number exists.

## v8.0 — Environment change, recorded because it affects reproducibility

The analysis box was re-provisioned between v7 and v8: `transformers` and `peft`
were absent and were reinstalled at **transformers 4.57.1 / peft 0.20.0 /
torch 2.11.0+cu128** (v5-era runs used transformers 5.14.1 / peft 0.19.1). No
v5–v7 number is recomputed here, so no cited result changes. The one thing that
had to be proven rather than assumed is that the frozen edit construction is
unchanged under the new stack: `src/v8_edits.selftest()` asserts that C-ref is
**bit-identical** to `colab_t2t4.binarize(v, "per_tensor", 0.99, seed)`
(max |diff| = 0.000e+00), and that assertion runs on every invocation of the DEC
runner's construction module. If it ever fails, the run aborts rather than
producing a silently different edit.

## v8.1 — Frozen inputs carried into v8 (experiments_v8 §1)

Attn (q,k,v,o) r16 contrast vector; `sign(Δ_fp)` as the sign source (STE retired,
v7 §BL-T2); per-tensor scale; likelihood elicitation; strict budget
(ΔMMLU ≥ −0.02, ppl ratio ≤ 1.10) measured **always-on**; equal selection space
(4 α per condition, α ∈ {2,4,8,16}); 3 seeds; bootstrap over the target×axis
cell; nan→0 **with counts recorded, never imputed**; no hardcoded baselines;
every α persisted to `alpha_trace.jsonl`. Sparsification is magnitude top-k on
the **dense merged ΔW** (the rank-16 product is dense; sparse edits are weight
patches, not factor pairs).

---

## DEC — sign / support / location decomposition (FROZEN before the first run)

### The construction, verbatim

Density **d = 0.01** for every condition. Support size is **k = N − ⌊(1−d)·N⌋**,
computed once from C-ref's rule and reused by every other condition, so all seven
conditions edit **exactly the same number of coordinates** (an earlier draft
derived the random and bottom supports from `round(d·N)`, which put them one
coordinate below C-ref; that is fixed and asserted in `selftest()`).

| id | support | signs |
|---|---|---|
| **C-ref** | top-\|Δ\| globally | true `sign(Δ)` at those coordinates |
| **C-a** | uniformly random | **true `sign(Δ)` evaluated at those random coordinates** |
| **C-bottom** | bottom-\|Δ\| (nonzero) | true `sign(Δ)` at those coordinates |
| **C-b** | top-\|Δ\| globally | sign **values permuted within tensor** |
| **C-rand** | uniformly random | random |
| **C-layershuf** | top-\|Δ\| globally | whole signed pattern moved between shape-compatible **layers** (derangement: no layer keeps its own) |
| **C-tensorshuf** | top-\|Δ\| globally | whole signed pattern moved between shape-compatible **projections within a layer** (derangement) |

**C-a vs C-b, verbatim, because v8 left the reading open.** C-a takes the *true
sign field evaluated at random coordinates* — sign is read off Δ at whichever
coordinate was drawn. C-b *relocates sign values* off their own coordinates by
permuting the multiset of signs within the tensor's support. They answer
different questions (is the support informative? is the (coordinate, sign)
pairing informative?) and both are run. Random supports are drawn by exact
multivariate-hypergeometric allocation over tensors followed by within-tensor
sampling without replacement — never by a Bernoulli mask, which would not hold k
fixed.

### The scale convention, and why it is not the obvious one

**Primary convention: `ref_tensor` — every condition uses C-ref's per-tensor
scale.** v8 §Threats says only that conditions must share an "identical
per-tensor-scale construction", which is ambiguous between *same procedure* and
*same resulting amplitude*. The ambiguity is not cosmetic. Under the same
*procedure* (each condition takes mean\|Δ\| over its own support), C-a's support
is a random 1% rather than the largest 1%, so its scale — and hence its edit norm
— is far smaller. **Measured on a synthetic contrast vector before any cell was
run: ‖C-a‖_F / ‖C-ref‖_F = 0.283 under the natural convention, versus 1.0005
under `ref_tensor`.** A 3.5× norm deficit is not recoverable by an α grid that
stops at 16, so the natural convention would let C-a lose on **edit norm** while
the paper reported it as losing on **support information** — i.e. it would
manufacture Outcome B. This is registered as a *pre-emptive* correction, not a
post-hoc one: no cell had been run when it was written.

This convention is also the DARE-faithful one. DARE rescales survivors by
1/(1−p) precisely so that random dropping is not penalised by shrinkage; matching
the per-tensor amplitude is the same courtesy in our parameterisation.

**Secondary (robustness) arm.** The `natural` convention is additionally run on
**2 cells** (`gemma|occ_gender`, `qwen|bbq_Age`, 3 seeds) and reported in the
appendix. If the two conventions disagree in *direction*, that disagreement is
reported as the headline caveat, not buried.

### DISCOVERY, recorded before the first DEC run: the frozen protocol is not one module set

While guarding DEC's shuffle conditions against shape-incompatible permutations,
`resolve_targets(model, ATTN)` was evaluated on all nine panel models. Eight
return `['q_proj','k_proj','v_proj','o_proj']`. **Phi-3.5-mini returns
`['qkv_proj']` — one fused module, and `o_proj` is dropped entirely**, even
though phi *has* an `o_proj`. The cause is `TARGET_FALLBACKS`
(`colab_t2t4.py:39`): the requested four are not all present, so resolution falls
through to the first fully-present fallback, `['qkv_proj']`, and never re-adds
`o_proj`.

Consequences, stated plainly because they reach published numbers:

1. The manuscript describes a single frozen protocol, "attn (q,k,v,o) r16".
   **For phi that is not what was run.** Phi's edit is a fused-QKV edit with no
   output projection; the other eight families get all four projections.
2. Every phi number in `RESULTS_METHOD.md` is on that different module set —
   C1 (phi|occ_gender +0.316 self / +0.632 cross_large), the four phi BBQ cells
   in §7, phi|ss_intra, phi's rows in the §11 frontier, and phi as one of four
   targets in the C3/C5/C6 ablations. None of these is *wrong*, but none of them
   is the module set the text claims.
3. This is not a superseded figure (the SUP table's category) — it is a
   **description error**, and it is recorded as such. Required manuscript action:
   state the per-family resolved module set in the protocol paragraph, or
   restrict the "(q,k,v,o)" phrasing to the eight families for which it is true.
   A footnote is not sufficient if phi carries a headline number, which it does.

**Effect on DEC, handled in code before running.** On a one-projection-per-layer
architecture, `C-tensorshuf` has no shape-compatible sibling inside a layer and
every `C-ref-no_<proj>` matches no module — all five would silently return C-ref
and be recorded as a measured effect of exactly zero. Averaged into Δ_tensor and
the leave-one-out table, they would have manufactured evidence that projection
identity does not matter. `run_dec.applicable_conditions()` therefore marks them
**not applicable** for phi and logs the omission per cell to
`not_applicable.jsonl`; Δ_tensor and the LOPO table are reported over the
four-projection families only, with n stated. Phi still contributes C-ref, C-a,
C-bottom, C-b, C-rand and C-layershuf — including **Δ_selection, the adjudication
quantity, which is unaffected** because it needs no projection structure.

### Cells (10, in-envelope, frozen before running)

`{gemma, llama, qwen, phi} × occ_gender`; `{phi, qwen} × bbq_Age`;
`{qwen} × bbq_Race_ethnicity`; `{qwen, phi} × ss_intra`; `{qwen-7B} × occ_gender`.
Self designer throughout, 3 seeds. Every cell has a positive measured removal on
the frozen protocol (`results/v8/env_inventory.json`: +0.090 to +0.719; qwen-7B
+0.66 at s=0.99 from `ablate_big`). Backfire cells are excluded by design — a
decomposition of an effect that is not present decomposes noise.

### Estimands (cell-level bootstrap, 10k resamples, RNG seed 0)

- Δ_sign = R(C-ref) − R(C-b)
- **Δ_selection = R(C-ref) − R(C-a)** ← the adjudication quantity
- Δ_ranking: the ordering of R(C-ref), R(C-a), R(C-bottom)
- Δ_location = R(C-ref) − R(C-layershuf); Δ_tensor = R(C-ref) − R(C-tensorshuf)

### Pre-registered adjudication — BOTH written before any DEC number exists

- **Outcome A — DARE-consistent.** Δ_selection CI covers 0 (C-a ≈ C-ref). Claim:
  *the sign field is distributed; magnitude serves only to certify coordinates,
  and any 1% of the field suffices.* Positioning: this work extends
  DARE/BitDelta/TIES from task performance to behavioural control under an
  always-on collateral budget, contributing the **necessity** side (sign
  randomisation kills removal) and the deployment frontier. The paper is not
  weakened: "any 1% works" is a stronger portability story.
- **Outcome B — DARE-divergent.** Δ_selection CI excludes 0 by a substantial
  fraction of R(C-ref). Claim: *behavioural edits concentrate in a
  magnitude-identifiable sparse signed substructure — unlike task-performance
  deltas under random drop.* C-bottom then separates "ranking finds the
  structure" from "any high-magnitude set works".
- Either way the manuscript sentence upgrades from "the learned sign structure is
  necessary" to the DEC-supported form, and Related Work carries the
  DARE/TIES/BitDelta positioning paragraph **regardless of outcome**. That
  paragraph is already drafted, before results: `RELATED_WORK_v8.md`.

---

## SUP1 — characterising the surviving 1% support

**Correction to v8's stated premise, recorded because it changes the cost claim.**
v8 §SUP1 lists its input as "existing top-1% supports per (cell, seed)" with
"GPU ≈ 0". No such artifact existed: nothing in the pipeline ever persisted an
edit's support, and regenerating one requires re-running the two task-vector
trainings. SUP1 is therefore **not** analysis-only by itself; it is made
analysis-only by having DEC dump the C-ref support as it builds it
(`src/v8_support.py`), at no extra training cost. Recorded rather than quietly
fixed, because "analysis-only" was a cost claim in the plan.

**Fold-enrichment null (mandatory).** At 1% density two independent random
supports already share ≈1% of either set, so raw Jaccard is ≈0.005 regardless of
the truth. Every overlap number is reported as fold-enrichment over a
hypergeometric expectation; raw Jaccard appears only in the appendix. **Two
nulls, both reported:**

- *global*: E|A∩B| = k_A·k_B/N — enrichment **includes** any tendency of both
  supports to occupy the same layers.
- *within-module*: E|A∩B| = Σ_m k_A(m)·k_B(m)/N(m) — conditions on the observed
  per-module sizes, so enrichment is coordinate agreement **beyond** placement.

Reporting only the global null would let a layer-concentration effect be sold as
coordinate-level agreement. Signed agreement uses E|A∩B|/2 as its null. The
estimator is validated against known inputs before touching real data: identical
supports → 100× (= 1/density), 50% shared → 50×, sign-flipped → 100× fold with
signed-fraction 0.000, and independent supports → 0.994× with signed-fraction
0.510 over 5 replicates.

**Designer overlap is registered two-sided.** High overlap = a shared
substructure (strong); low overlap *together with* the already-measured
behavioural tie = a degenerate solution space, many sparse solutions with the
same effect (also strong). We do **not** pre-commit to "similar = good".

**Axis overlap** is likewise two-sided: low → distinct substructures per axis;
high → a shared behavioural-control substructure.

**Leave-one-projection-out** (the only GPU cost) runs as 4 extra DEC conditions
(`C-ref-no_{q,k,v,o}_proj`) on the DEC cells, zeroing one projection's surviving
coordinates from C-ref.

---

## ENV — operating-envelope calibration with a held-out split

**Eligibility.** Every (target, axis) cell measured on the frozen protocol with a
self-designer removal, a measured `pre_skew`, and a `contrast_gap` for that
designer/axis/seed. All contributing panels (`t2x, v1, x, x2, mvb, fxg, ax, axes,
ss, v6trace/{t2x,x2}, v6trace_3b/small`) were verified to call
`train_task_vector(..., rank=16, targets=ATTN)` and
`binarize(..., "per_tensor", 0.0, seed)` over α ∈ {2,4,8,16}. Baseline panels
(steer, sentdebias, inlp, dpo, ste), null panels and the quarantined `steer_*`
panels are excluded by construction — they are not the edit.

**contrast_gap coverage, extended.** `contrast_gap_frozen.json` covers only
`occ_gender` and `crows_socioeconomic` (frozen 2026-07-29, before the BBQ and
StereoSet runs). The corpus files themselves carry `diag.contrast_gap` for all 11
axes, so ENV reads the gap from the corpus file the run actually consumed and
records its md5. The 114 frozen observations are **cross-checked** against the
on-disk corpora and agree exactly (0 disagreements at the artifact's 3dp
precision); a disagreement is a hard error, never a silent preference. This is
the audit-trail rule adopted after the v6 corpus-shadowing bug.

**Unit of analysis.** The (target, axis) cell under the **self** designer — the
endogenous deployment default, consistent with the paper's "self-elicitation
suffices at scale" argument. Removal is the mean over seeds with nan→0 applied
per row and `n_budget_fail` carried. An all-roles sensitivity analysis is
reported but is not the primary.

**Inventory: 36 cells**, not the ~18–20 v8 estimated, across four benchmark
families (templated 9, CrowS 17, BBQ 8, StereoSet 2). This materially weakens
v8's own stated weakness ("held-out n ≈ 8–10 → wide CIs"), though the CIs are
still reported and not suppressed.

**Exclusion, pre-registered with reason.** `granite × occ_gender`
(granite, granite2, granite8 — 12 of 12 rows) is **100% budget-fail at every α
for every designer**. Prereg V3 diagnosed this as family-level collateral
fragility: granite's low baseline perplexity makes the ppl-ratio budget
unreachable, with ΔMMLU −0.080 (4× the limit) at the minimum α. This failure is
**not visible to the (pre_skew, contrast_gap) gate** — these are the highest
pre_skew (+0.90) and among the highest contrast_gap (+0.405) cells in the
inventory. Including them would train the gate to reject the strongest-signal
cells for a reason it cannot observe. They are excluded from calibration **and**
held-out, and reported as their own "protocol-inoperable" row so nothing is
hidden. The 129 `v6trace_3b/small` rows are excluded separately and mechanically:
their designer corpora are not on this box (naming `llama3b` vs `llama_3b`), so
no `contrast_gap` can be joined. Excluded with reason, not silently dropped.

**Split, frozen before any threshold was chosen.** Stratified by benchmark
family, seeded alternate deal, assignment a pure function of (cell, seed=0):
**17 calibration / 16 held-out / 3 excluded**, each family split as evenly as its
size allows. Written to `results/v8/env_split.json`,
sha256 `aded438ea3b74f19f95efb766b9883ec0f31bebfd245f77d9e16ff0ffa6842cb`.
`src/v8_env.py` **refuses to run `calibrate()` unless that file already exists**.

**Objective, pre-set.** Maximise utility with **abstention scored as 0**, subject
to **backfire rate ≤ 0.10 among edited cells**, searching τ over `|pre_skew|` and
γ over `contrast_gap`. Ties broken by (higher utility, fewer edits, larger
threshold) — fully determined, no judgement at scoring time. Thresholds are
frozen after calibration and applied unchanged to held-out.

**PEEK DISCLOSURE (§7 no-peeking, recorded as the contrast_gap artifact did).**
Building the inventory necessarily printed a table carrying `pre_skew`,
`contrast_gap` and `removal` together for all 36 cells, and I read it before the
split was written. Mitigation, and why the split is still credible: the split is
a deterministic function of (cell identity, family, seed 0) with no outcome term,
so it could not have been steered; and the objective and cap above were written
before `calibrate()` was ever executed. The disclosure is recorded rather than
the peek denied. A reader who discounts ENV on this basis is entitled to; the
inventory and split files are both hashed so the claim is checkable.

**Wording rule.** No universal constants. The manuscript says "we observe a
low-bias failure regime and calibrate an intervention gate on held-out cells";
any numeric boundary appears only as observed data.

---

## SPC — sparsity curve to 99.9%

Grid **{0, 50, 90, 95, 97, 99, 99.5, 99.9}%**, frozen. Cells: `{qwen, phi, gemma,
llama} × occ_gender` at ≤3.8B and `{qwen-7B, llama-8B} × occ_gender` at 7–9B,
self designer, 3 seeds, standard budget and 4-α selection. X-axis reported in
triplicate — retained-parameter fraction, nonzero count, and patch bytes — all
three taken from `binarize()`'s existing meta. Registered reading: a collapse
between 99 and 99.9% is a phase transition and a headline sub-figure; a flat
curve shrinks the patch numbers and supports DEC's field-redundancy
interpretation. If DEC returns Outcome A, C-a is additionally run at 99.5/99.9%
on 2 cells to ask whether the portable field survives deeper sparsity.

---

## INS — instruct bridge: GATE PASSED

**Gate (v8): lm-evaluation-harness runs IFEval + MMLU on our checkpoints within
one working day.** Executed 2026-08-28, **PASS**, in well under a day. Evidence
in `results/v8/ins_gate.json`. The gate was tested against the thing that could
actually have failed: not whether the harness installs, but whether it can score
an **in-memory edited model**, since saving a checkpoint per condition (~15 GB ×
5 conditions × 2 targets × 3 seeds) is impossible on a box with 22 GB free.
`HFLM(pretrained=<live model object>)` works, MMLU and IFEval both run, and a
sensitivity check confirms the harness reads the **edited** weights
(zeroing layer-0 `q_proj` moves inst_level_strict 0.367 → 0.300). INS is
therefore **committed, not dropped**.

### AMENDMENT, made before any INS run: the base-only invariant did not exist

v8 §1 says "the base-only invariant is retired for INS ... INS is the deliberate
instruct arm", and v7 §1 says "base models only; the single instruct arm (COL) is
a scoped, labeled exception". **Both are wrong about the method panels, and this
was found while scoping INS.** `colab_t2t4.FAM_MODELS` — the map that defines
every target and every designer in every method panel — contains *only*
instruction-tuned checkpoints at every size (`Qwen2.5-{3B,7B,14B}-Instruct`,
`Llama-3.2-3B-Instruct`, `Llama-3.1-8B-Instruct`, `gemma-2-{2b,9b}-it`,
`OLMo-2-*-Instruct`, `granite-3.1-*-instruct`, `Falcon3-*-Instruct`). A base
checkpoint appears in exactly one place in the repository, `run_y1_bigbox.py`,
which profiles `google/gemma-2-27b` base for the Y1 anti-convergence result —
i.e. for the **science line**, where measuring *inherited* bias genuinely
requires a pre-RLHF model. The method paper's entire results table has always
been measured on instruction-tuned models.

Consequences, all of which change what INS may claim:

1. INS as specified would have "bridged to" Qwen2.5-7B-Instruct and
   Llama-3.1-8B-Instruct — **the very checkpoints already used as the 7–9B
   targets** in `ablate_big`/SC7. It would have re-run existing cells and
   reported them as new.
2. The registered reading "the phenomenon survives instruction tuning" is
   **already answered affirmatively** by every published panel and must not be
   presented as an INS finding.
3. Reading (ii) ("instruction tuning moves cells out of the operating envelope")
   is not testable without a genuine base arm, which the plan did not have.
4. Any manuscript sentence claiming base-only measurement is **false** and must
   be struck. This is added to the SUP quarantine table as a claim, not a figure.

### INS, re-scoped (registered before running)

**INS-A — generation-based collateral (the genuinely new measurement).** The
budget has always been MMLU + perplexity, both likelihood-based; instruction
following has never been measured. Targets Qwen2.5-7B-Instruct and
Llama-3.1-8B-Instruct, axis occ_gender, conditions unedited / full-precision /
sign-only / 99%-sparse / random-sign, seed 0, IFEval scored on the **first 200
prompts** of the 541 (a stated sampling bound, chosen to keep the arm inside one
GPU-day; recorded here rather than reported as full IFEval). This is v7 §COL's
content, and it is labelled as collateral generality, not as an instruct bridge.

**INS-B — the real base↔instruct contrast.** `google/gemma-2-2b` (base) and
`google/gemma-2-2b-it` (instruct) are both fully cached, same family and size,
so the paired comparison the plan wanted is actually available. Axes occ_gender +
bbq_Age, 3 seeds, full onboarding gates re-run **per target** (pre_skew is
measured on the base model, never inherited from the instruct one), conditions as
above. This is the only arm entitled to speak about instruction tuning.

*Corpus decision, registered before running.* INS-B holds the **edit signal
fixed** — both targets use the `gemma_3b` corpus elicited from gemma-2-2b-it —
and varies only the target checkpoint. For the instruct target that is genuine
self-elicitation; for the base target it is the instruct sibling's corpus, and it
is recorded as `role="sibling_instruct"`, never as `self`. Eliciting separately
from each checkpoint would confound "instruction-tuned target" with
"instruction-tuned corpus", and the registered question is whether instruction
tuning moves the **cell** out of the operating envelope, which requires the
signal held constant. The complementary question (can a base model elicit its own
usable corpus?) is a different experiment and is not claimed here.

*Missing corpus, filled before the run.* `gemma_3b × bbq_Age` had never been
elicited — only llama_3b, phi_3b and qwen_3b were ever run on the BBQ axes. It is
generated by `run_v8_elicit.py` through the same `elicit_selfdebias` /
`_t2x_corpus_fp` path `panel_run` uses, writing only files that do not exist and
never overwriting, per the v6 rule that a cache miss must never become a silent
re-elicitation.

### AMENDMENT after INS-B's first execution: the harness cannot evaluate a base model

INS-B was run and **every `gemma-2-2b` (base) cell failed**, all six, with
`ValueError: Cannot use chat template functions because tokenizer.chat_template
is not set`. `colab_t2t4.chat_prompt` has a fallback, but the fallback also calls
`apply_chat_template`, so it raises identically when there is no template at all.
`chat_prompt` feeds `mmlu_predict` and `elicit_selfdebias`, and MMLU sits inside
`fast_eval`, which runs at every onboarding gate and every α. **No base model can
be evaluated by this pipeline at all.**

This strengthens the amendment above. The method line was not instruct-only by
choice or by oversight in `FAM_MODELS`; it was instruct-only **by construction of
the harness**. The manuscript's limitations section should say so plainly, because
it also bounds what any future base-model claim would cost: a new prompt path,
not a new run.

*The fix, and the confound it would otherwise introduce.* Giving the base model a
plain-text prompt while the instruct model keeps gemma's chat template would
confound *instruction tuning* with *prompt format* — the two arms would differ in
two ways at once. **Arm BP** therefore forces the **same** minimal plain-text
template on **both** checkpoints, so the only difference is the weights.
Registered consequences, before the run:

- BP's absolute numbers are **not comparable to the main tables**, which use real
  chat templates. Only the base-vs-instruct contrast *within* BP is licensed.
- **Arm B is retained as the format control.** `gemma2b_it` appears in both B
  (chat template) and BP (plain text), so B vs BP measures how much the prompt
  format alone moves `pre_skew` and removal. If that format effect is comparable
  in size to the base-vs-instruct effect, the INS-B reading is reported as
  inconclusive rather than as an instruction-tuning finding.
- Both v8 readings remain attached to BP as registered; neither is re-worded.

**Both readings stay registered, now attached to INS-B where they are testable.**
(i) base and instruct cells both in-envelope → the sparse-sign phenomenon is
insensitive to instruction tuning. (ii) the instruct cell falls below the
envelope while the base cell does not → *instruction tuning moves cells out of
the operating envelope*, a reportable finding connecting the envelope to
deployment reality. Claim ladder unchanged. INS-B results **never mix into the
main tables**, which remain instruct-measured and must be described as such.

---

## Z — freeze-before-running checklist

- [x] DEC cell list (10 in-envelope cells) frozen; both adjudication paragraphs written above.
- [x] C-a vs C-b constructions defined verbatim (evaluated-at-coords vs relocated-values).
- [x] DEC scale convention resolved, justified by a measured confound (0.283 vs 1.0005), with a robustness arm.
- [x] SUP1 fold-enrichment null specified (two nulls); designer- and axis-overlap two-sided readings registered.
- [x] SUP1's "analysis-only" premise corrected in writing; support dump added to DEC.
- [x] ENV split assignment frozen and hashed before any threshold search; objective and backfire cap pre-set; peek disclosed.
- [x] SPC grid and cells frozen; triple x-axis specified.
- [x] INS gate executed with a decision date (2026-08-28, PASS); both instruct readings registered.
- [x] Related-Work DARE/TIES/BitDelta paragraph drafted before DEC results exist (`RELATED_WORK_v8.md`).
- [x] Resume keys include {experiment, cell, condition, sparsity, seed}; all α traces persisted.

# v9 — PROGRAM-WIDE RE-SCORE (amendment log, opened 2026-08-30)

## v9.A — The epsilon/integer decision, with date and rationale

**Decision (2026-08-30, approved before the re-score was run):** the
collateral-gate floating-point defect is corrected program-wide and the resulting
v9 numbers become the only citable set.

*Defect.* `colab_t2t4.collateral_ok` tested `(pre_mmlu - post_mmlu) <= 0.02` in
floating point. MMLU is scored on 200 items, so accuracy moves in steps of 0.005
and a 4-item drop **is** exactly the budget — but `0.545 - 0.525` evaluates to
`0.020000000000000018 > 0.02`. Configurations sitting precisely on budget were
rejected. Exposure: 238 probes across 25 alpha-trace panels, plus 430 rows in
`results/runs.jsonl`.

*Correction.* Compare in **integer items**, `(pre_items - post_items) <= 4`, not
in floats. This removes the boundary error exactly, needs no epsilon, and is
checkable by hand. Perplexity keeps a float ratio with an explicit `1e-9`
tolerance. One shared implementation, `src/v9_gate.py`; 12/12 unit checks
including the known failing case. Exhaustive verification over the 201x201 grid:
old and new gates disagree on exactly **138** pairs and **every disagreement is a
4-item drop**.

*Scope.* All accuracies and all stored `dmmlu` values in the repository are exact
multiples of 1/200 (verified: 7512 trace accuracies, 7960 dmmlu values, 10254
runs.jsonl rows), so the integer gate applies everywhere. No panel fell back to
an epsilon comparison.

## v9.B — Re-verdicts (CLOSEOUT_V9 §1.4). Text follows data.

- **DEC Outcome B — CONFIRMED, strengthened.** Delta_selection
  +0.2537 -> **+0.2835**, CI still excludes 0, unanimity 10/10 unchanged. The
  correction favours C-ref because C-ref is the arm that actually spends
  collateral. **No amendment required.**
- **BL-S three-way tie (D2) — RE-ADJUDICATED, UNCHANGED.** edit - steering
  -0.0095 -> **-0.0208**, still covering 0; edit - SentenceDebias unchanged at
  -0.0279. The registered *Edit ~ steering* branch remains operative. **No fork
  switch.** Caveat recorded: the frontier's edit arm is mixed-gate (gemma/phi
  re-scorable via `v6trace/x2`, llama/qwen not).
- **ENV negative — UNCHANGED.** Inputs are partly non-re-scorable; the as-run
  conclusion is not weakened by the correction. Contribution 4 stays
  **conditional**.
- **SPC <=3.8B withdrawal — CONFIRMED BY CONSTRUCTION.** 99%-vs-dense
  -0.1286 (excluded 0) -> **-0.0754 (covers 0)**. This is the single CI-status
  change in the entire diff.

## v9.C — D1: surgical MMLU re-run REJECTED, on the count

§WS2.3 required counting before deciding and anticipated "dozens". The measured
count is **833 probes within +-1 MMLU item of the budget, 10.5% of all 7960**
(463 at 3 items, 241 at 4, 129 at 5). Re-running those rows means re-training and
re-evaluating their cells, i.e. most of the program. **Default arm adopted:**
keep trace values, let the integer gate absorb the boundary, document the
granularity as measurement noise.

## v9.D — Determinism, re-scoped (WS2.4)

MMLU scoring is **exactly reproducible** within a fixed environment and batch
size: 3 in-process repeats and 3 separate processes all return 129/200. The
demonstrated sensitivity is **batch composition** — batch sizes 4/6/8/16 give
131/129/131/129, a 2-item (1.0%) spread. TF32 is not implicated. The observed
cross-panel divergence occurred at identical batch size, so it is attributed to
the environment change (transformers 5.14.1 -> 4.57.1) rather than to run-to-run
randomness; that attribution is stated, not proven by re-running the old stack.
The 108/108 sweep-determinism claim is **scoped, not deleted**.

## v9.E — Limits of the re-score, recorded so they are not mistaken for coverage

Five panels cannot be re-scored at all — `t2x`, `v1`, `x`, `x2`, `v6trace/ste` —
because they predate the v6 alpha-trace fix and store only a scalar removal.
Their boundary exposure is **unknown** and is never assumed to be zero.
Consequences: §11b (STE vs sign(dW_fp)) is **BLOCKED**, and the BL-S frontier is
**mixed-gate** as recorded in v9.B.

## v9.F — Citation ban extended

`RESULTS_METHOD_v9.md` (generated by `src/v9_make_results.py`) is the single
citable source. Every earlier result file carries a superseded banner. The
number-source audit `src/v9_number_audit.py` is a pre-submit gate: a number it
cannot trace does not ship.

---

# v10 — NUMBER AUDIT AND FIGURES (amendment log, opened 2026-08-30)

## v10.A — Scope

`experiments_v10.md` scoped two workstreams: WS-A, a manifest mapping every
number in the manuscript to an artifact, enforced by a pre-submit gate; and
WS-B, the six ICLR figures, each reading only v9 artifacts and emitting its own
data CSV. Both shipped. `audit/manifest_v10.yaml` is the manifest;
`src/v10_audit.py` is the gate; `V10_FINDINGS.md` records what the audit found.

No registered estimand is changed by v10. The audit corrects PRINTED values to
match the artifacts they were always supposed to report, and records where a
printed value has no artifact behind it.

## v10.B — The alpha-extension panel is POST-HOC and is registered as such

§5.3's qualifier stated that "under aggressive alpha extension the gap shrinks
to +0.096 and covers 0". The audit established that **no alpha above the frozen
deployment grid maximum of 16 had ever been probed**, in any trace in the
repository: the figure was an extrapolation reported as a measurement.

It has since been measured. `run_alpha_ext.py` probed alpha in {32, 64, 128}
for conditions C-ref and C-a across the 10 registered DEC cells x 3 seeds,
writing to a NEW panel `results/v10ext/`. Everything else — cells, seeds,
corpora, rank-16/250-step/lr-1e-4 training, density 0.01, `ref_tensor` scaling,
`fast_eval`, and the collateral gate — is identical to `run_dec.py`. The script
refuses any alpha inside the frozen grid, so `v8dec` cannot be touched.

**This arm is POST-HOC.** The deployment grid was frozen in this document before
DEC ran. The extension therefore cannot and does not change the registered
Delta_selection, which remains the frozen-grid estimand
+0.2835 [+0.1692, +0.4072] (n=10, excludes 0). The extension is reported as an
exploratory counterfactual only, and its scored result is
`results/v10/alpha_extension.json`.

Environment: the analysis box had again lost `transformers`/`peft`/`datasets`
and was restored to the v10.B stack recorded at L545-547 — transformers 4.57.1
/ peft 0.20.0 / torch 2.11.0+cu128 — the stack the v8dec panel it extends was
produced under.

## v10.C — Resampling unit correction

`src/method_evidence.py` bootstraps the SWEEP (target, axis, seed, designer).
The project convention, stated for every registered v9 estimand, is the
target x axis CELL; v5 banned per-seed pooling as anti-conservative. The C2,
C3, C5 and C6 statistics were therefore reported at the wrong unit. They are
re-bootstrapped at the cell unit for the main text, with sweep-unit twins
retained and labelled. Consequences are recorded in `V10_FINDINGS.md`; the
direction of every affected claim is unchanged, but several CI statuses are.

## v10.D — Scope qualifier on "exactly one CI status changed"

v9.A's count is correct **over the 18 estimands registered in
`results/v9/v9_diff_report.json`**. The C5/C6 ablation family is re-scorable,
was never re-scored into that diff, and contains six further status changes.
The claim is qualified accordingly wherever it appears; the v9 re-score itself
is not amended.

## v10.E — Numbers with no artifact

Two printed values survive the audit without a source and are quarantined
rather than cited: §5.9's retracted padding-bug magnitudes. They are quoted as
errors the protocol caught, not as results, and the recomputed values replace
them. The audit fails on them by design; it is not weakened to pass.

# v11 — SCALE-UP (amendment log, opened 2026-09-07, before any v11 run)

Design: `experiments_v11.md`. Four axes, each attacking a Limitations sentence:
DEC cell count (10 -> 18, adding the fourth benchmark family), alpha-grid
saturation (128 -> 512), IFEval breadth (2x1x1 -> 3x2x1), and model size
(2.6-8B -> +32B, 70B stretch). Panels: `v11dec`, `v11ext`, `v11ins`, `v11big`.
No published panel is written to; every v11 panel is new.

## v11.A — MMLU stays at 200 items, deliberately

The Limitations section names >=1,000 MMLU items as "the forward fix". It is
NOT taken in v11. The gate compares in integer items with budget
`round(0.02 * n_items)`, so changing the count changes the gate's grid and no
number measured at 1,000 is comparable to any published number measured at 200.
Taking it would require re-running the whole program before a single comparison
could be made. IFEval breadth buys collateral evidence the likelihood-only
budget cannot see and is purely additive, so v11 widens that instead. Recorded
as a decision so it is not later read as an oversight.

## v11.B — The collateral probe size now travels with the measurement

Through v10 the MMLU item count existed only as the literal `200` in
`run_dec.main`'s `load_mmlu(n=200)`. No row recorded it and `v9_gate` defaults
to 200, so the gate was correct only because every panel happened to use 200 --
an invariant nothing checked. Scaling the probe would have failed loudly on
accuracies off the 1/200 grid and SILENTLY on those that also land on it.

Fixed additively: `fast_eval` attaches `n_items`, `collateral_ok` reads it and
raises when pre/post disagree, `alpha_trace` persists it, and `v11_panel.load`
refuses an `mmlu_n` whose 2% budget is non-integral. `src/v11_selftest.py`
verifies the change is INERT at n=200 -- decisions identical to v9 across the
whole grid. No published number moves.

## v11.C — Alpha saturation criterion, fixed before the run

SATURATED means C-a's in-budget count reaches 0/30, matching C-ref (already
0/30 at alpha=128). Report the saturating alpha, Delta_selection at the deepest
alpha where both arms retain an in-budget run, and the per-doubling flattening.
If the series does not saturate by 512, report the value as an upper bound and
do NOT extrapolate -- v10 §5.9 defect (iv) was an extrapolation reported as a
measurement. The extension inherits POST-HOC status from v10.B; the frozen grid
{2,4,8,16} remains the registered estimand.

## v11.D — Backfire cells stay excluded from the decomposition

`qwen|crows_socioeconomic` (removal -0.071) is NOT added to the DEC panel,
under run_dec.py's standing rule that decomposing a negative effect is
uninterpretable. It is recorded in `configs/v11/dec_v11.yaml` under `excluded:`
with its reason, so the omission is auditable. The two weak in-envelope CrowS
cells (llama +0.040, phi +0.046) are an OPEN author decision documented in
experiments_v11.md §IV; it must be resolved BEFORE the panel runs, because
choosing after seeing Delta_selection is outcome-selection.

## v11.E — The IFEval harness version is unrecorded, and is treated as a risk

The lm-eval version behind the PUBLISHED IFEval numbers appears nowhere in this
repository. `requirements-run.txt` pins `lm-eval==0.4.5` as an explicitly
UNVERIFIED pin, and the two published cells are re-run alongside the four new
ones so the whole arm shares one harness version. If the re-run does not
reproduce, the published IFEval numbers are version-dependent and that fact is
reported rather than overwritten. Old rows are retained; v11 writes a new panel.

## v11.F — Screening rule for NEW DEC cells (declared 2026-09-07, after 1 of 54 units, before the remaining 53)

DEC's standing rule (run_dec.py) admits "10 in-envelope cells (positive removal
already measured; backfire cells would confound a decomposition of an effect
that is not there)". For the CrowS cells added in v11 that rule is checkable
against Table A. For five other new cells it is NOT: `gemma|bbq_Age`,
`llama|ss_intra`, `llama8b|occ_gender`, `qwen7b|bbq_Age` and `llama8b|bbq_Age`
have no prior removal measurement anywhere, so they cannot satisfy a rule that
requires one. This gap was in the v11 config as written and is recorded here
rather than repaired silently.

TRIGGER. The first completed unit, `gemma|bbq_Age|s0`, returned C-ref = +0.0646
against a published C-ref cell mean of +0.353, with Delta_selection = -0.0301
(the published panel is unanimous 10/10 POSITIVE). One seed of one cell decides
nothing, but it establishes that at least one new cell may sit outside the
envelope the decomposition assumes.

THE RULE, FIXED NOW. A new cell is IN-ENVELOPE iff its C-ref removal, pooled
over its three seeds under the nan->0 rule, is positive and at least one of its
seeds cleared the collateral budget. Screening uses C-ref ONLY -- the reference
arm -- and never Delta_selection or any contrast between conditions, so the
screen cannot select on the quantity being estimated.

REPORTING, ALSO FIXED NOW. Three numbers, all published, none chosen after the
fact:
  (a) Delta_selection over the 10 PUBLISHED cells      — the registered estimand,
      unchanged, and the only one that carries the pre-registered status;
  (b) Delta_selection over published + in-envelope new cells — the scale-up
      result;
  (c) Delta_selection over ALL cells including out-of-envelope ones — the
      conservative reading.
Every cell's in/out status is reported with its C-ref value, so a reader can
recompute any of the three. No cell is dropped from the record; out-of-envelope
cells are labelled, not deleted.

WHY THIS IS NOT OUTCOME-SELECTION. The criterion is stated before 53 of the 54
units have run, is a function of C-ref alone, and is applied to every new cell
uniformly. The one unit already complete does not change it: had it come back at
+0.4, the same rule would have been written.

WHAT THIS DOES NOT LICENSE. If a new cell is out-of-envelope, that is a REPORTED
result about the envelope, not a nuisance. The v10 line removed an
outcome-selected arm ("new alphas only", n=5) for conditioning its cell set on
the thing being measured; this rule exists so that mistake is not repeated in v11.

## v11.G — The big tier is bf16, including 70B (decided 2026-09-07, before any big-tier run)

`configs/v11/big_v11.yaml` originally carried `eightbit: true` for
Llama-3.1-70B, on VRAM grounds. That is withdrawn. Every cell in the big tier
runs bf16.

REASON. The experiment measures what survives quantizing the EDIT -- one bit per
weight, 1% of coordinates. Running the BASE model at 8 bits puts a second,
uncontrolled quantization underneath it, so Delta_selection at that cell would
confound edit structure with base-weight precision. That is the confound the
seven-condition decomposition exists to avoid, and it would land on the single
cell whose purpose is to extend the size range. Every published panel is bf16
(colab_t2t4.DTYPE); the big tier must be comparable to them.

CONSEQUENCE, RECORDED SO IT IS NOT DISCOVERED ON A RENTED NODE. Trainable
targets load with dispatch=False, so the model must fit on ONE device:
  Qwen2.5-32B   ~61 GiB weights + ~16 GiB training  ->  one >=80 GB card
  Llama-3.1-70B ~131 GiB weights + ~16 GiB training ->  one >=192 GB card
                                                        (B200 / MI300X; an H200
                                                        at 141 GB is too tight)
On 2x80 GB the 70B cell needs sharded training, which this codebase does not
have. So 70B is blocked on hardware or on a code change, not on configuration,
and qwen32b is the cell to land first.

scripts/preflight.py now estimates this as weights + a FLAT 16 GiB training
allowance rather than a 1.5x multiplier. The trainable part is a rank-16 LoRA on
the attention projections with gradient checkpointing, so optimizer state is
tiny and activations are bounded by batch and sequence, not by parameter count.
The old rule demanded ~196 GiB for 70B and would have refused hardware that
works.

## v11.H — 70B dropped for gemma-2-27b; big-tier cells are self-designers (2026-09-07, before any big-tier run)

HARDWARE. The available node is a SINGLE 80 GB card (~79.2 GiB usable).
Llama-3.1-70B needs ~131.5 GiB of bf16 weights plus ~16 GiB of training on ONE
device, because trainable targets load with dispatch=False and this codebase has
no sharded training. It is therefore dropped, not deferred. Running it at 8-bit
to make it fit is refused under §v11.G: that would confound base-weight
precision with the edit structure the decomposition measures.

REPLACEMENT: google/gemma-2-27b-it, ~50.7 GiB + ~16 = ~67 GiB, comfortable on
the same card. It keeps the big tier at two cells, adds a THIRD family there,
and gives gemma a within-family size ladder (2.6B in DEC, 9B in IFEval, 27B
here) that no other family has. Qwen2.5-32B is retained at ~77.1 GiB against
~79.2 usable -- it fits, at 97% occupancy, and gemma27b is the cell to land
first. llama70b stays recorded in the config's `excluded:` block with its
reason, so the omission is auditable.

SELF-DESIGNER, AND THIS WAS NEARLY WRONG. run_dec.py stamps role="self" on
every row unconditionally, and the published panel is 285/285 genuinely
self-designer. The designer name -> checkpoint map lives in colab_t2t4.T2X,
which stops at 7-9B, so `designer: qwen` on a 32B target would have resolved to
Qwen2.5-7B: a SIBLING designer recorded as "self", inside a panel whose every
other cell is self. Cells may now name `designer_hf` explicitly, and both
big-tier cells elicit from their own checkpoint.

## v12.A — Operating-point selection: design frozen (2026-09-23, before any v12 run)

Registered before the first v12 unit. Design: experiments_v12.md. Code and
configs: branch v12-opsel (configs/v12/*.yaml as of commit 3188b13, plus the
criterion below coded in src/v12_pstar.py in the commit that adds this entry).

PANELS. v12cal (fit population, <= 9B: 6 published SPC occ_gender cells
replayed + 5 bbq_Age cells), v12big (HELD OUT: gemma-2-27b-it and
Qwen2.5-32B-Instruct x {occ_gender, bbq_Age} x seeds {0,1,2}), v12dec1k,
v12frontier.

p*. Cell-level (seed-mean, nan -> 0). R(p) = removal(p) / removal(dense
one-bit). p_true = LAST-SUCCESS: the smallest grid p with R >= rho,
log-interpolated toward the next grid point; first-failure reported as a
sensitivity only. Grid {0,.5,.9,.95,.97,.99,.995,.999,.9995}.
  D1  rho   = 0.90
  D2  floor = 0.05 (dense removal; cells below are reported, not fitted)
  D3  primary rule = mass: tau = median M(p_true) over eligible v12cal cells,
      p* = M^-1(tau) on the held-out cell's seed-mean geometry. pr / ent / gini
      / fixed-1% are reported alongside; leave-one-target-out on v12cal is the
      only admissible evidence for any rule change, and none is permitted after
      `v12_pstar.py predict` has run.
  D4  HELD-OUT SUCCESS: PASS iff R(p*) >= rho - 0.05 in >= 3 of the eligible
      held-out cells (4 expected). Secondary, not part of the verdict:
      |log2(p*/p_true)| <= 1.
The prediction is frozen by `src/v12_pstar.py predict` from phase-A geometry
before any v12big curve row exists; its sha256 is stamped on every pstar row.

alpha*. First-failure on the ladder {1,2,4,...,512} + 3 log-bisection steps,
gate = the v9 integer gate on MMLU validation (200 items) + WikiText validation
(20 chunks); no bias probe is read.
  D5  reported as the as-deployed removal at alpha*, with the rule's budget-pass
      rate on the evaluation items beside it.
The frozen {2,4,8,16} argmax is reported alongside everywhere.

MMLU-1000 is a robustness reading of selected points (every alpha where
configured), never a selection criterion. The registered gate stays at 200.
  D6  second big-tier axis = bbq_Age.

BOTH OUTCOMES, WRITTEN NOW. PASS -> the method section replaces "fixed 1%"
with the rule. FAIL -> 1% stays a constant, the rule is reported as a negative
result, and the paper makes no geometry-predicts-sparsity claim.
