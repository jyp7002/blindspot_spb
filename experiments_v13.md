# experiments_v13.md — two follow-ups from v12: a seed-regenerable patch, and whether the density floor is a scale property

**Status:** registered as `PREREGISTRATION.md` §v13 (2026-09-26), before any v13 unit. The code and configs are in the commit that adds the registration. §7 lists every change from the draft that was circulated.

v12 leaves two questions that no existing panel answers.

1. **SEED.** Under the bias-blind rule α\*, a random 1% support with true signs came close to the magnitude-selected 1%: Δ_selection +0.073 [−0.036, +0.213] at ≤9B and −0.037 [−0.071, +0.003] at 27–32B. A random support does not have to be stored, because it can be regenerated from a seed. The shippable patch would drop from payload plus index (3.7–14.5 MiB) to payload only (0.41–1.60 MiB). Is that patch non-inferior as a deployable artifact?
2. **FLOOR.** The sparsity cliff (retention 0.46–0.53 at 99.9%) was measured on the frozen grid α ∈ {2, 4, 8, 16}. Fewer coordinates mean a smaller edit norm, so part of the cliff may be grid-limited in the same way C-a was. Does the cliff move when each sparsity level gets its own budget-limited scale?

---

## 0. Priors, written before anything runs

- **SEED at ≤9B is not expected to pass.** On the ten registered cells, v12's α\* cell means (C-a, the closest existing arm) give ratios **below the 0.85 bar under both scorings**. As-deployed: 0.807 (C-a 0.305 over C-ref 0.377). **Deployable, this experiment's primary scoring: 0.625.** qwen|occ is a clean failure (C-a −0.029 against C-ref +0.601). With 18 cells the ratio's interval will still be wide, so a NON-INFERIOR verdict needs a true ratio close to 1. The experiment is still worth running: INFERIOR is informative, and 27–32B looks different (C-a matched or beat C-ref under α\*).
- **FLOOR has no prior either way, and v12 leans against it.** The magnitude-selected support is the high-curvature one: its budget binds at about a quarter of C-a's scale (α\* medians 20.7 and 90.5). Pushing α on an even sparser top-k support may therefore hit the collateral budget before the effect recovers.
- **Deployable scoring.** α\* passes the evaluation budget in only 57–60% of ≤9B runs. Because the question here is deployability, both experiments score a configuration that fails the budget on the evaluation items as zero. This is stricter than v12's as-deployed primary (D5), which is still reported alongside.
- **A known limit on reproducibility.** Across processes, MMLU at identical weights varies by 1–3 items: over v12dec1k/v12astar's 240 shared points, 223 were identical, 12 differed by 1 item, 2 by 2 and 3 by 3. Skew and perplexity were identical at every point (`V12_FINDINGS.md` §8.2). Deployable scoring is a gate decision, so both primaries inherit this variation. E0.5 measures whether forcing deterministic kernels removes it.

---

## 1. Frozen inputs (carried)

Attn rank-16 contrast ΔW with per-family module resolution; sign(Δ_fp); the `ref_tensor` per-tensor scale; likelihood elicitation; self designer (for the 7–9B cells, `designer: qwen` / `llama` resolves to the 7B/8B model itself, via `colab_t2t4.T2X`); 3 seeds; cell-level percentile bootstrap (10k resamples, RNG seed 0); nan→0 with counts; integer-item gate on 200 MMLU items (≤4 items) plus perplexity ratio ≤1.10; every α persisted with `n_items`; bf16 at every tier (v11.G); the v11.F C-ref-only screen for any cell with no prior C-ref measurement.

---

## 2. E0 — engineering gate (every item passes before the first unit)

**E0.1 Portable C-seed construction** (`src/v13_seed.py`).
- **Support size:** k = N − ⌊0.99·N⌋, as in v8 (`_k_target`).
- **Allocation over tensors:** deterministic largest-remainder allocation proportional to tensor size, with ties broken by tensor name. No RNG is involved, and Σ k_t = k exactly. This is the expected allocation of C-a's hypergeometric draw.
- **Positions within a tensor:** the first k_t outputs of a keyed pseudorandom permutation of [0, N_t). It is a 4-round unbalanced Feistel network on the next power of two (widths ⌊b/2⌋ and ⌈b/2⌉, swapping each round), with a splitmix64 round function and cycle-walking back into range. Round keys are the four 64-bit words of SHA-256(seed_u64_le ‖ tensor name). It uses integer arithmetic only and no library sampler. NumPy guarantees stable BitGenerator streams across versions but not stable Generator methods, so a sampler-based support could decode differently on another install.
- **Signs and scale:** the sign is +1 where ΔW ≥ 0 at the selected coordinate, else −1 (so sign(0) = +1, which keeps one bit per coordinate lossless). The per-tensor scale is C-ref's (`ref_tensor`) at the same density.
- **Tests** (`python3 src/v13_seed.py selftest`, all passing): bijection on bit widths 1–13, including non-powers of two; exact k and k_t, each within one of proportional; uniformity by χ² over 64 position bins for 20 keys at N = 10⁶ and k = 10⁴, every p > 0.01 (minimum 0.012); support depends on seed and name; identical positions from a fresh process. A second library version was not available on this box; the construction has no library-version dependence to test.

**E0.2 Measured patch files** (`src/v13_patch.py`).
- **Seed patch:** a header (magic `BSP1`, format version, model id, revision hash, seed as u64, density as f32, α as f32, tensor list with numel, k_t and a per-tensor scale as **f32**) followed by packed sign bits in decode order.
- **Index patch (C-ref):** the same header, plus positions encoded per tensor with Elias–Fano, plus sign bits.
- **Checked every unit:** each patch is decoded and rebuilt one tensor at a time, and the unit stops unless the result is bit-identical to the in-memory edit. Measured bytes are recorded per unit.
- **Two entropy floors for the index:** the global floor (8.08 bits at 1%, 11.41 at 0.1%) and the per-tensor floor Σ N_t H_b(k_t/N_t) / k. Top-k concentrates its support in some tensors, so the per-tensor floor is lower and is the relevant bound. In the GPU smoke test (qwen 3B, 1%), the index patch used 8.48 bits per coordinate including the sign, which is below the global floor.
- **Smoke-test sizes** (qwen 3B, 1%): seed patch 432,513 B (0.41 MiB); index patch 3.43 MiB. Every patch size in the paper becomes a measured file size.

**E0.3 α\*, as registered in v12.A.**
- **Search:** α doubles from 1 until the first failure on the validation gate, then three log-bisection steps between the last pass and the first failure. α\* is the largest passing value.
- **Validation gate:** the integer-item gate on 200 MMLU validation items plus the perplexity ratio on 20 WikiText validation chunks. No bias probe is read.
- **Ladder maximum:** 512 for SEED and 4096 for FLOOR. A run with no validation failure up to the maximum is flagged as censored (status `exhausted`).

**E0.4 Deployable scoring.** A selected configuration's removal counts only if it also passes the integer-item gate on the 200 MMLU evaluation items and on the evaluation perplexity corpus. Otherwise it scores 0, which is recorded and never imputed. α\* status `none` also scores 0. As-deployed removal (v12 D5) is reported beside it.

**E0.5 Determinism controls.**
- **One process per unit:** each unit trains ΔW once and derives every arm from it in one process. The α\* search and the final evaluation of an arm run in that same process, and every MMLU read is persisted.
- **Replay** (`configs/v13/replay.yaml`, gate before any new cell): two v12astar units, gemma|occ_gender s0 and qwen|occ_gender s0, configured exactly as v12astar. At every frozen evaluation point, pre/post skew and perplexity must be **bit-identical** and MMLU must agree **within 3 items** of 200; otherwise stop and diagnose. `scripts/v13_local.sh` enforces this.
- **Separate diagnostic, used for no verdict** (`configs/v13/det_*.yaml`): 6 registered cells at seed 0 on the frozen grid (s0.99 and C-a). Each is run in two fresh processes with default kernels and in two fresh processes with deterministic kernels forced (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). The comparison within each pair of processes shows whether the drift exists between identical processes and whether forcing deterministic kernels removes it.

---

## 3. SEED — is a seed-regenerable patch non-inferior?

**Arms (per unit).**
- Primary: C-ref@α\* (`s0.99`) and C-seed@α\* (`C-seed@0.01`).
- Descriptive only: both arms on the frozen grid (the published default), and, on the ten registered cells, C-a@α\* (v8 construction), to check that C-seed and C-a behave alike.

**Cells.**
- **≤9B:** the 18 v11dec cells (the published 10 plus the 8 added in v11), 3 seeds.
- **27–32B:** gemma-2-27B and qwen2.5-32B × {occupation→gender, bbq_Age, crows_socioeconomic, ss_intra}, 3 seeds: **8 cells** (§v13.F). The crows_socioeconomic and ss_intra cells are new to this tier and go through the v11.F screen.

**Primary estimand, per tier.** ρ_seed = (cell mean of C-seed@α\* deployable removal) / (cell mean of C-ref@α\* deployable removal), with a 95% paired cell-bootstrap interval on the ratio of means. The primary is over **all** cells in the tier. Cells failing the v11.F screen are labelled, and the ratio over in-envelope cells only is reported beside it.

**Verdict, per tier.**
- NON-INFERIOR iff the lower bound is ≥ 0.85; INFERIOR iff the upper bound is < 0.85; INCONCLUSIVE otherwise.
- The 0.85 bar is v12.A's ρ − 0.05 with ρ = 0.9, carried unchanged.
- The ≤9B verdict is the primary. With 8 cells, the 27–32B tier also receives a verdict.
- A scale claim is made only if both tiers have verdicts and the tier contrast ρ(27–32B) − ρ(≤9B), with the tiers resampled independently, has an interval that excludes 0.

**Secondary readings.**
- The difference C-ref − C-seed at α\*, reported under its own name and not as a re-adjudication of §v12.E.
- Per-arm α\* medians and ranges, and evaluation-gate pass rates.
- MMLU-1000 at every selected configuration (α\* and the frozen argmax) of both primary arms.
- IFEval on seed 0 for both arms at α\*, on all 7–9B and 27–32B cells, against a baseline measured before training.
- Measured patch bytes (E0.2).

**What the paper does, written now.**

| verdict (per tier) | paper |
|---|---|
| NON-INFERIOR | Adds a seed-patch operating mode for that tier: a payload-only file (measured; about 0.4–1.6 MiB up to 8B) at about four times the edit scale, with its pass rate and IFEval beside it. Contribution 2 gains a sub-MB variant scoped to that tier. |
| INFERIOR | The index is worth its bytes. Reported as a negative result that supports the efficiency reading: magnitude selection buys the effect at a quarter of the scale, and a free support cannot replace it at equal quality. |
| INCONCLUSIVE | Both arms reported; no storage claim. |

---

## 4. FLOOR — does the density cliff move when scale is free?

**Arm.** C-ref at s ∈ {0.99, 0.995, 0.999, 0.9995}, each at its own α\* with the ladder extended to 4096. The construction is `colab_t2t4.binarize`: global top-k, true signs, and the natural per-tensor scale, as in the sparsity panel. The frozen grid is also measured at every point, in the same unit.

**Cells.** ≤9B: the 11 v12cal cells. 27–32B: the 4 v12big cells. 3 seeds each.

**Primary estimand.** R = (cell mean of deployable removal at s = 0.999) / (cell mean of deployable removal at s = 0.99), both under α\*, over the ≤9B cells, with a 95% paired cell-bootstrap interval on the ratio of means.

**Verdict.** FLOOR MOVES iff the lower bound is ≥ 0.85; FLOOR HOLDS iff the upper bound is < 0.85; INCONCLUSIVE otherwise. The 27–32B tier is descriptive (n = 4).

**Secondary readings.**
- The same ratio under the frozen grid and as-deployed, and the shift (α\* − frozen) with its interval, from the same resamples.
- The 0.995 and 0.9995 points.
- Per cell, the density needed to keep 90% of the 1% removal, under α\* and under the frozen grid (v12.A last-success definition, relative to s = 0.99, via `v12_pstar.p_true`).
- The censoring rate at α = 4096.
- Measured index-patch bytes at every sparsity.

**What the paper does, written now.**

| verdict | paper |
|---|---|
| FLOOR MOVES | The density floor is also a property of the edit scale: at a budget-limited scale 0.1% keeps at least 85% of the 1% effect, and the index patch shrinks about sevenfold (measured). The title's "one in a hundred" gets this qualifier. |
| FLOOR HOLDS | The cliff is intrinsic even at free scale: about 1% is where the information sits, not a grid artifact. This strengthens the title claim and goes in §5.5. |
| INCONCLUSIVE | Reported beside the frozen curve; no floor claim. |

**Where "sevenfold" comes from.** At density p the index costs H_b(p)/p bits per retained coordinate: 8.08 at 1% and 11.41 at 0.1%. Adding the sign bit gives 0.01 × 9.08 = 0.091 against 0.001 × 12.41 = 0.012 bits per weight, a ratio of 7.3. The measured ratio replaces this estimate.

**FLOOR-2b (conditional, exploratory, no verdict).** Run only if SEED at ≤9B is NON-INFERIOR *and* FLOOR MOVES (`configs/v13/floor_seed.yaml`; `run.sh v13-floor-seed` refuses otherwise). C-seed at s = 0.999 under α\* on the ≤9B cells, reported descriptively. This would be a payload-only patch of roughly 40–160 KiB. It is not registered as a claim.

---

## 5. Discipline

- **§v12.E stands as registered** (INCONCLUSIVE, no scale claim). v13 does not recompute E/F and does not re-run v12astar beyond the two E0.5 replay units. SEED's estimand is a ratio of deployable removals, with its own bar, on an expanded cell set, using a new support construction. It is reported under its own name.
- **Everything is fixed by the registration:** cell lists, arms, sparsity points, ladders, the 0.85 bars, deployable scoring, and the verdict rules. None of them changes after the first unit. Out-of-envelope cells are reported, not dropped.
- **One run per unit.** Any repeat, determinism diagnostics included, is reported alongside and never substituted.
- **No reading before completion:** `src/v13_analyze.py verdicts` reads no removal from a panel until every unit of it is on disk.
- **Multiplicity:** there are three verdicts (SEED ≤9B, SEED 27–32B, FLOOR ≤9B). Each is read on its own, and no combined claim is made across SEED and FLOOR.
- **Hardware and order:** ≤9B units run on the L40S (`scripts/v13_local.sh`: replay → determinism diagnostic → FLOOR → SEED). 27–32B units run in bf16 on a single ≥80 GB card (v11.G/H) via `bash run.sh v13-seed-big` and `bash run.sh v13-floor-big`.

---

## 6. Code map

| piece | file |
|---|---|
| C-seed construction + selftest | `src/v13_seed.py` |
| patch formats (seed / Elias–Fano index), round-trip check, selftest | `src/v13_patch.py` |
| runner: `C-seed@<d>` variant, `patch_variants`, `ifeval_variants`, `deterministic` | `run_opsel.py`, `scripts/run_unit.py`, `src/v11_panel.py` |
| configs | `configs/v13/{replay,det_*,seed_dec,seed_big_4,seed_big_8,floor_cal,floor_big,floor_seed}.yaml` |
| analysis: replay gate, determinism, verdicts | `src/v13_analyze.py` |
| local chain | `scripts/v13_local.sh` |

---

## 7. Changes from the circulated draft, made before registration

1. **Replay tolerance: MMLU within 3 items, not 2.** On v12's own data, 3 of 240 cross-process points differ by 3 items with skew and perplexity bit-identical, so a 2-item tolerance could stop v13 on known noise. A real stack change also moves skew and perplexity, which stay bit-exact.
2. **The SEED prior quotes the deployable ratio (0.625)** beside the as-deployed ratio (0.807). The primary uses deployable scoring.
3. **The per-tensor patch scale is f32, not f16.** An f16 scale would not rebuild the in-memory edit, and bit-identity is required. It costs 2 bytes per tensor.
4. **Two entropy floors are reported for the index:** global and per-tensor (see E0.2).
5. **sign(0) = +1** for C-seed, so that one bit per coordinate is lossless.
6. **The SEED primary is over all cells;** the ratio over in-envelope cells only is reported beside it. The draft said out-of-envelope cells are reported, not dropped, but did not say whether they enter the ratio.
7. **The determinism diagnostic is 6 units × {default, deterministic} × 2 fresh processes,** so that "the drift disappears" is measured against a default-kernel control rather than asserted.
8. **§v13.F recorded: 8 cells** (27–32B verdict issued).
