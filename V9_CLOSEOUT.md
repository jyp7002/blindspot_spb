# V9_CLOSEOUT — execution record

Execution of `experiment_v9.md` (CLOSEOUT_V9), 2026-08-29/30. Analysis-only:
**no GPU experiments were re-run**, because the re-score is driven entirely by
the persisted α traces — which is what v6 added them for.

| workstream | status |
|---|---|
| **WS1** v9 re-score | **complete** (with 5 panels structurally un-re-scorable — §3) |
| **WS2** dmmlu determinism | **complete**, cause isolated, D1 decided |
| **WS3** draft v3 | **spec only** — the manuscript is not in this repo (§6) |
| **WS4** hygiene | **complete** (amendments, banners, audit gate) |
| **WS5** freeze | checklist status in §7; 2 items blocked on the manuscript |

---

## 1. The one-line result

**The re-score changes exactly one CI status in the entire program.** Every other
registered verdict survives; DEC strengthens; the tie holds.

| estimand | as-run | v9 | status |
|---|---|---|---|
| **DEC Δ_selection** | +0.2537 [+0.1475, +0.3692] | **+0.2835 [+0.1692, +0.4072]** | unchanged (strengthens) |
| DEC Δ_sign | +0.3185 | +0.3496 | unchanged |
| DEC Δ_null | +0.3207 | +0.3518 | unchanged |
| DEC Δ_location | +0.3178 | +0.3487 | unchanged |
| DEC Δ_tensor | +0.3663 | +0.3809 | unchanged |
| DEC C-a − C-rand | +0.0670 | +0.0682 | unchanged |
| **frontier edit − steering** | −0.0095 | **−0.0208** | covers 0 both — **tie holds** |
| frontier edit − SentenceDebias | −0.0279 | −0.0279 | unchanged |
| frontier edit − DPO | +0.2926 | +0.2926 | unchanged |
| **SPC ≤3.8B, 99% vs dense** | −0.1286 (excludes 0) | **−0.0754 (covers 0)** | **← the only flip** |
| SPC 7–9B, 99% vs dense | +0.0099 | +0.1120 | unchanged (strengthens) |
| SPC 7–9B, 99.9% | −0.0740 | −0.0740 | unchanged |

Re-scoring recovered **714 probes/rows** and changed **146 cell values** across
32 panels plus `runs.jsonl`.

---

## 2. WS1 — the corrected gate

**The fix (WS1.1).** `src/v9_gate.py`, one shared implementation, now also used
by `colab_t2t4.collateral_ok` (it delegates; the per-panel float copy is gone).
MMLU is compared in **integer items** — `(pre_items − post_items) ≤ 4` at
n_items = 200 — so the boundary error is removed *exactly*, with no epsilon.
Perplexity keeps a float ratio with a `1e-9` tolerance.

**12/12 unit checks**, including the case CLOSEOUT_V9 names: `0.545 − 0.525` at
200 items is a 4-item drop and is now **in budget**. Exhaustive verification over
all 201×201 grid pairs: the old and new gates disagree on exactly **138** pairs,
and **every disagreement is a 4-item drop**. That is the bug's complete footprint.

**Better than the plan required.** CLOSEOUT_V9 anticipated an epsilon fallback
for panels that store only `dmmlu`. None was needed: **all 7960 stored `dmmlu`
values are exact multiples of 1/200**, so the integer test applies to the stored
difference directly. Every panel is gated by integer comparison; no epsilon is
used anywhere.

**Re-selection (WS1.2).** `src/v9_rescore.py` re-applies the gate to every probed
α and re-picks the in-budget argmax — a real re-selection, not a re-labelling: a
cell whose best α was rejected at the boundary can now select a *different* α.
Three paths, routed by panel structure rather than by guesswork: α-traces (27
panels), per-config rows (5), and `runs.jsonl` (10254 rows, 430 recovered).

---

## 3. What could NOT be re-scored — the honest boundary

Five panels predate the v6 alpha-trace fix and store only a scalar removal, so
there is **no record of which α were probed or what collateral they cost**:

| panel | rows | consequence |
|---|---:|---|
| `t2x` | 144 | boundary exposure unknown |
| `v1` | 126 | boundary exposure unknown |
| `x` | 54 | **frontier edit arm for llama, qwen** |
| `x2` | 108 | frontier edit arm for gemma, phi — but see below |
| `v6trace/ste` | 24 | **§11b STE vs sign(Δ_fp) is BLOCKED** |

**Their exposure is unknown and is never assumed to be zero.**

One recovery was possible. `results/x2` has an exact re-scorable twin,
`v6trace/x2` — verified here at **108/108 keys identical, 0 differing**, which
independently confirms the §8 determinism claim. So gemma and phi *are* true v9
in the frontier. `results/x` has no twin, so **llama and qwen remain as-run** and
the 8-cell frontier is **mixed-gate**. Restricted to the 4 fully re-scorable
cells, edit − steering is −0.0219 [−0.1600, +0.0891] under both gates.

---

## 4. Decision points

**D2 — the BL-S fork: RE-ADJUDICATED, NO SWITCH.** This was the live one:
steering's per-cell best changed in 2 of 24 cells, one substantially
(`qwen|occ_gender` seed 1: 0.379 → **0.635**). The pooled comparison still covers
zero — edit − steering −0.0095 → **−0.0208** — so the registered *Edit ≈ steering*
branch remains operative and the contribution claim is unchanged. **No
improvisation was required and none was performed.**

**D1 — surgical MMLU re-run: REJECTED, on the count.** CLOSEOUT_V9 required
counting first and expected "dozens". Measured: **833 probes within ±1 MMLU item
of the budget — 10.5% of all 7960** (463 at 3 items, 241 at 4, 129 at 5).
Re-running those rows means re-training and re-evaluating their cells, i.e. most
of the program. Default arm adopted: keep trace values, let the integer gate
absorb the boundary, document the granularity as measurement noise.

**D3 — other registered verdicts: none flipped.** DEC Outcome B confirmed and
strengthened; ENV negative unchanged; the SPC ≤3.8B withdrawal confirmed by
construction. Amendment log in `PREREGISTRATION.md` §v9.B.

---

## 5. WS2 — determinism, and a better answer than expected

The v8 record said the evaluation was "not fully deterministic". That was
imprecise. Measured on 200 MMLU items, llama-3.2-3B-Instruct:

| condition | items | spread |
|---|---|---:|
| repeat in-process, bs=6 (×3) | 129, 129, 129 | **0** |
| **separate processes, bs=6 (×3)** | **129, 129, 129** | **0** |
| batch size 4 / 6 / 8 / 16 | 131, 129, 131, 129 | **2 items (1.0%)** |
| TF32 on / off | 129, 129 | 0 |

**MMLU scoring is exactly reproducible within a fixed environment and batch
size.** The only demonstrated sensitivity is **batch composition**. TF32 is not
implicated.

The observed cross-panel divergence occurred at *identical* batch size, so batch
composition does not explain it; the remaining difference between those runs is
the environment (`transformers` 5.14.1 → 4.57.1, the rebuild recorded in
`RESULTS_V8` §2). That attribution is **stated, not proven** — proving it would
mean reinstalling the old stack and re-running.

Re-scoped statement (WS2.4): *sweep-pipeline bit-determinism verified (108/108,
0 differing); MMLU scoring is deterministic within a fixed environment and batch
size but shows batch-composition sensitivity of up to 1% (2 of 200 items); fixed
for v9 by scoring the gate in integer items, which removes the boundary
knife-edge, and NOT fixed at the source.* The 108/108 claim is **scoped, not
deleted**.

---

## 6. WS3 — draft v3: spec delivered, edits not applied

`binary_debiaser_draft.md` and the revision plan are **not in this repository**,
so the 14-item change list could not be applied to a manuscript. What exists
instead:

- **All numbers** → `RESULTS_METHOD_v9.md` (generated by script, §7).
- **Items 2, 3, 6, 7, 9, 10** (module set, base-only, SUP1 wording, ENV negative,
  INS bounds, bugs paragraph) are specified with their exact wording in
  `V8_FINDINGS.md` §"What the manuscript must do" — unchanged by v9 except that
  item 8's ≤3.8B qualifier is now **withdrawn twice over** (audit + re-score).
- **Item 4** (DEC efficiency reframe) gains a v9 fact worth using: C-a is
  *grid*-limited while C-ref is *collateral*-limited, which is exactly the
  "magnitude selection buys the effect where the signal is cheapest" framing the
  plan asks for — now with the re-scored numbers behind it.
- **Item 11** (reproducibility statement) → §5 above, ready to paste.
- **Item 1's enforcement** → the audit gate in §7.

The bugs paragraph (item 10) is now **three** caught errors: two steering-favouring
(miscalibrated grid, `padding_side` leak) plus the collateral-gate float defect —
and the third was caught by the project's own adversarial audit, which is the
strongest version of the "the protocol caught our own errors" sentence.

---

## 7. WS5 — freeze checklist, honest status

| item | status |
|---|---|
| Corrected gate merged as one shared function; unit test on the known boundary case passes | **DONE** — `src/v9_gate.py`, 12/12; `colab_t2t4.collateral_ok` delegates |
| `RESULTS_METHOD_v9.md` regenerated end-to-end by script | **DONE** — `src/v9_make_results.py` |
| Diff report complete; every CI-status change reviewed; D2/D3 resolved | **DONE** — `results/v9/v9_diff_report.json`; 1 flip, reviewed |
| Repro statement re-scoped; dmmlu cause documented | **DONE** — §5 |
| §5a module table in; §5b base-only claims struck (grep returns zero) | **BLOCKED** — manuscript absent; spec in `V8_FINDINGS.md` §5 |
| Contribution 4 conditional; ENV negative section in | **DONE** in results docs; manuscript edit blocked |
| SUP table includes the float-gate row; superseded banners on old files | **DONE** — banners on 8 files; SUP row specified below |
| Number-source audit green; TODO-marker grep zero outside limitations | **DONE** — `src/v9_number_audit.py`, **AUDIT GREEN** |
| Terminology greps return zero | **PARTIAL** — enforced by the audit script over results files; manuscript not present |
| `PREREGISTRATION.md` amendment log current as of freeze commit | **DONE** — §v9.A–v9.F |

**SUP quarantine row to add (WS1.5):**

| old figure | origin | v9 replacement |
|---|---|---|
| every as-run number | collateral gate compared MMLU in floats; a 4-item drop at 200 items is exactly the 0.02 budget but `0.545−0.525 > 0.02` | `RESULTS_METHOD_v9.md`; 238 probes across 25 α-trace panels + 430 rows in `runs.jsonl`; 714 recovered, 146 cells changed, **1 CI-status flip** |

§5a (the per-family module set) stays a **description** correction outside SUP,
per `V8_FINDINGS.md`.

---

## 8. Artifacts

| what | file | script |
|---|---|---|
| corrected gate | — | `src/v9_gate.py` |
| re-scored panels | `results_v9/**/removal.jsonl` | `src/v9_rescore.py` |
| re-score report | `results_v9/rescore_report.json` | `src/v9_rescore.py` |
| diff report | `results/v9/v9_diff_report.json` | `src/v9_regen.py` |
| determinism | `results/v9/ws2_determinism.json` | `run_ws2_determinism.py` |
| citable numbers | `RESULTS_METHOD_v9.md` | `src/v9_make_results.py` |
| pre-submit gate | — | `src/v9_number_audit.py` |
| amendment log | `PREREGISTRATION.md` §v9 | — |
