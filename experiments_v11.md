# experiments_v11.md — SCALE-UP (registered 2026-09-07, before any v11 run)

**Status:** frozen design. Nothing in this file has been run.
**Builds on:** `experiments_v10.md` (number audit + figures), which closed with
`make freeze` green and the audit passing on 658 manifest entries.
**Draft under scale-up:** the ICLR LaTeX submission
(`One Bit in a Hundred: The Sparse Signed Substructure of Behavioral Weight Edits`).

The v10 line finished the *bookkeeping*: every number in the manuscript traces
to an artifact. v11 does not add bookkeeping. It attacks the four places where
a reviewer can say **"you measured too little"**, each of which the paper's own
Limitations section already names.

---

## I. What is scaled, and the defect each fixes

| # | axis | now | v11 | the sentence in the paper this repairs |
|---|---|---|---|---|
| 1 | DEC cells | 10 cells, 3 families, big tier = 1 cell | 18 cells, **4 families**, big tier = 4 cells | Abstract "three benchmark families"; Limitations "DEC *n* = 10 cells" |
| 2 | α grid | probed to 128, **not saturated** | probe to 512, terminate at saturation | §5.3 "the measured gap is an upper bound"; Limitations "Grid-conditional concentration" |
| 3 | IFEval | 2 targets × 1 axis × 1 seed | 3 targets × 2 axes × 1 seed | Limitations "IFEval 2 targets × 1 axis × 1 seed" |
| 4 | model size | 2.6–8B (3× span) | + 32B (12×), 70B stretch (27×) | Limitations "2.6–8B targets" |

Configs: `configs/v11/{dec,alphaext,ifeval,big}_v11.yaml`. Each is a frozen
input; changing one after its panel starts is an amendment, not an edit.

---

## II. MMLU stays at 200 items. This is a decision, not an omission.

The Limitations section calls ≥1,000 MMLU items "the forward fix", and it was
considered here and **rejected for v11**.

The collateral gate compares in integer items with a budget of
`round(0.02 × n_items)`. Changing `n_items` changes the gate's grid, so **no
number measured at 1,000 items is comparable to any published number measured
at 200** — the whole program would have to be re-run before a single
comparison could be made, and every table in the draft would be re-derived from
scratch. That is a v12-scale program, not a scale-up of the present one.

IFEval breadth (axis 3) buys the same kind of evidence — collateral measured
somewhere the likelihood-only budget cannot see — and is **purely additive**:
every published number stays valid. So v11 widens IFEval instead.

**What v11 does do about it:** the item count is now recorded with the
measurement, so the day someone *does* raise it, the gate reads the right
denominator instead of assuming 200. See §VII.

---

## III. α saturation — criterion fixed before the run

Depth series as published: `+0.284 (α≤16, frozen) → +0.233 (≤32) → +0.146 (≤64)
→ +0.104 (≤128)`, monotone, still falling. At α=128, C-a is in budget in 10/30
runs and C-ref in 0/30.

**Registered criterion.** The extension is **SATURATED** when C-a's in-budget
count reaches **0/30**, matching C-ref. Report:

1. the α at which C-a's in-budget count reaches 0/30;
2. Δ_selection at the deepest α where **both** arms still have an in-budget run;
3. whether the series has flattened, and by how much per doubling.

**Two outcomes, both written now.**
- **SATURATES with Δ_selection excluding 0** → the concentration result holds at
  unbounded grid depth. §5.3's "upper bound" qualifier can be replaced with a
  measured floor. This is the strong outcome.
- **SATURATES with Δ_selection covering 0, or does not saturate by 512** → the
  gap is grid-conditional in a way the current draft understates. Report the
  measured value, keep the "upper bound" framing, and **do not extrapolate** —
  v10 already caught one extrapolation reported as a measurement (§5.9 defect iv).

**Post-hoc status is inherited and does not lapse.** PREREGISTRATION.md v10.B
registers the α-extension panel as POST-HOC. Going deeper does not make it
confirmatory. The frozen grid `{2,4,8,16}` remains the registered estimand.

**Reproduction guard.** The frozen arm must reproduce `+0.2835 [+0.1692,
+0.4072]` exactly, which requires the **panel insertion order** for the cell
list — v10 found that sorting it shifts the CI by ~0.0008. `alphaext_v11.yaml`
preserves that order. Do not sort it.

---

## IV. DEC cell expansion — and the one decision left open

**The fourth family.** The abstract said "four benchmark families"; the v10
audit found the DEC panel carries three (BBQ / StereoSet / templated), because
no CrowS cell was ever decomposed. v11 adds CrowS cells so the claim is
**measured rather than corrected away**.

**The pre-registered exclusion is respected.** `run_dec.py` documents its cell
list as "10 in-envelope cells (positive removal already measured; backfire
cells would confound a decomposition of an effect that is not there)".
`qwen|crows_socioeconomic` has removal **−0.071** — it backfires — so it is
**not** added. It is recorded under `excluded:` in the config with its reason,
so the omission is auditable instead of invisible.

**OPEN AUTHOR DECISION — resolve before launching.** `llama|crows` (+0.040) and
`phi|crows` (+0.046) are in-envelope but weak. Including them widens the
population Δ_selection is defined over, and a weak-removal cell can move the
pooled estimate without any mechanism changing.

- **Option A (registered default, as configured):** include all three CrowS
  cells. The fourth family rests on 3 cells. Report Δ_selection both with and
  without them.
- **Option B:** keep `gemma|crows` (+0.123) only. The fourth family rests on 1
  cell, and the estimand's population barely moves.

Whichever is chosen must be fixed **before** the panel runs and stated in the
paper. Choosing after seeing Δ_selection is outcome-selection — the same defect
v10 caught and removed in the α-extension ("new alphas only" arm, n=5).

**Balance cells.** gemma and llama each carried exactly one axis, so every
multi-axis statement rested on qwen and phi. v11 adds `gemma|bbq_Age` and
`llama|ss_intra`.

**Big tier.** One cell (`qwen7b|occ_gender`) carried every "two model tiers"
claim. v11 adds three more.

---

## V. Big tier (32B/70B) — runs elsewhere, and 70B is a stretch

`configs/v11/big_v11.yaml`. Two known-unresolved risks, both recorded now so
neither is discovered on a metered node:

1. **8-bit + LoRA training is untested in this codebase.** `eightbit` was only
   ever used for T4 *inference*. `train_task_vector` does not call
   `prepare_model_for_kbit_training`. **Run `qwen32b` (bf16) first**, end to
   end, before paying for a 70B node.
2. **Sharded training does not exist here.** `colab_t2t4.load_model` documents
   that `device_map='auto'` installs accelerate hooks that break training, so
   trainable targets use `dispatch=False` and must fit on **one** device.
   32B bf16 ≈ 61 GiB → one ≥80 GiB card. `scripts/preflight.py --config` refuses
   a node that cannot hold the cell.

Scope: one axis (`occ_gender`), one seed. The question at this tier is whether
Δ_selection stays positive and unanimous, which needs the seven conditions, not
three seeds. Widen only after the first cell lands.

---

## VI. What must reproduce (the guard against a silent stack change)

`run_ws2_determinism.py` attributes a cross-panel MMLU divergence to a
`transformers` rebuild. Every v11 panel therefore **replays the published cells
alongside the new ones** and must reproduce them:

| panel | must reproduce |
|---|---|
| `v11dec` | the 10 published DEC cells (marked `published: true`) |
| `v11ext` | the frozen arm, `+0.2835 [+0.1692, +0.4072]`, exactly |
| `v11ins` | the 2 published IFEval cells — **see the warning below** |

**IFEval harness version is unrecorded.** The lm-eval version behind the
*published* IFEval numbers appears **nowhere**: not in PREREGISTRATION.md, not
in the logs, not in the artifacts. `requirements-run.txt` pins `lm-eval==0.4.5`
as an **UNVERIFIED** pin. The two published cells are therefore re-run so the
whole arm shares one harness version.

- If the re-run **reproduces** → cite the widened arm.
- If it **does not** → the published IFEval numbers are version-dependent, and
  that belongs in the paper. It does not get quietly overwritten. The old rows
  stay on disk; v11 writes to a new panel.

---

## VII. Infrastructure changes, and the defect that motivated them

**The collateral probe size never travelled with the measurement.** Through v10
the MMLU item count lived in one place: the literal `200` in
`run_dec.main`'s `load_mmlu(n=200)`. It was never written to a row, and
`v9_gate` defaults to 200 everywhere. The gate was correct only because every
panel happened to use 200 — an invariant nothing checked and no artifact
recorded.

Scaling the probe would have failed in the worst available way: **loudly** on
accuracies off the 1/200 grid (`items_from_acc` raises) and **silently** on the
ones that also land on it (1/1000 values like 0.520 are exact multiples of
1/200). A panel would come back part-gated against a 4-item budget and
part-crashed, with no row saying which.

Fixed in v11, additively:
- `colab_t2t4.fast_eval` attaches `n_items` (the count is known exactly there
  and nowhere later);
- `colab_t2t4.collateral_ok` reads it, and **raises** if pre and post were
  scored on different counts;
- `alpha_trace` persists it;
- `v11_panel.load` refuses an `mmlu_n` that makes `0.02 × n_items` non-integral.

**Verified inert on everything published:** `src/v11_selftest.py` checks that at
n=200 the gate's decisions are identical to v9's across the whole grid (8/8
checks pass). No published number moves.

**Everything else is new files**, so no published artifact is touched:
`src/v11_panel.py`, `src/v11_selftest.py`, `scripts/{preflight,plan,run_unit}.py`,
`scripts/submit_{local,slurm}.sh`, `scripts/pack_artifacts.sh`,
`configs/v11/*.yaml`, `env/Dockerfile`, `requirements-*.txt`.

---

## Z. Freeze-before-running checklist

- [ ] **§IV Option A or B chosen and written down** (the only open design decision)
- [ ] `python3 scripts/preflight.py --config <cfg>` exits 0 on the run box
- [ ] pinned stack confirmed: transformers 4.57.1 / peft 0.20.0 / torch 2.11.0+cu128
- [ ] `make freeze` green on the analysis box **before** any v11 row is written
- [ ] configs committed and unmodified after the first unit starts
- [ ] `v11ext` cell order is panel-insertion order, not sorted
- [ ] `qwen32b` completes before any 70B node is rented
- [ ] published cells reproduce in every panel (§VI); if not, STOP and report

**Amendment log:** `PREREGISTRATION.md` §v11. Any change to this file after the
first v11 row is written goes there, dated, with its rationale — text follows
data, never the reverse.
