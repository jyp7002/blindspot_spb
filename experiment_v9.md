# CLOSEOUT_V9.md — the v9 re-score and everything between here and submission

**Decision record (2026-08-30):** the program-wide re-compute is **approved**.
The collateral-gate float bug (§6 of `V8_FINDINGS.md`) is corrected everywhere;
the resulting **v9 numbers become the only citable set**. As-run v8-and-earlier
numbers move to the quarantine table. This document is the closeout plan:
five workstreams, ordered by blocking dependency, with decision points and the
freeze checklist. Deadline context: abstract **09-19**, full **09-24**.

```
WS1  v9 re-score (analysis-only, BLOCKS every quoted number)
WS2  dmmlu-determinism investigation (timeboxed, parallel)
WS3  draft v3 (structure now; numbers flow in after WS1)
WS4  prereg / citation hygiene (continuous)
WS5  freeze, audit, submit
```

---

## WS1 — The v9 re-score *(GPU ≈ 0; from persisted alpha traces; blocks all numbers)*

### 1.1 The corrected gate — one function, used everywhere

- **MMLU: compare in integer items, not floats.** The budget is 0.02 of
  n_items = 200 → `allowed_drop = 4` items. Gate:
  `(pre_correct − post_correct) <= 4` as integers. This eliminates the
  boundary bug *exactly* (no epsilon needed) and is the reviewer-proof form.
- **ppl ratio: keep float, add explicit tolerance** `<= 1.10 + 1e-9`.
- One shared implementation; delete every per-panel copy. Unit test on the
  known failing case (`0.545 − 0.525` at 200 items → **in budget**).

### 1.2 Re-selection from traces

For every sweep in every panel (not only the 25 known-affected): re-apply the
corrected gate to `alpha_trace.jsonl`, re-pick the in-budget argmax α,
recompute every cell statistic and every pooled CI (cell-level bootstrap,
unchanged). Deterministic; no model forward passes.

### 1.3 Regenerate every quoted number

DEC (all 7 conditions + Δ_selection + stress tests), C1/C2/C3/C5/C6, the
**BL-S/BL-T frontier**, prompt baseline, AX and the pre_skew–removal
correlation, SC7 retention (86.6 / 108.6 may move), SPC both tiers, ENV inputs
and the policy/sensitivity tables, INS. Output: `RESULTS_METHOD_v9.md`
regenerated end-to-end by script.

### 1.4 The diff report

One table: every manuscript-quoted number — v8 value → v9 value → CI-status
change (none / widened / sign flip / crosses zero). **Any registered verdict
that flips gets a PREREGISTRATION.md amendment, and the text follows the
data** — specifically:

- **DEC Outcome B**: expected to strengthen (+0.2537 → ≈ +0.2835). Confirm.
- **The three-way tie (BL-S fork)**: MUST be re-adjudicated under the
  **already-registered three-outcome fork**. If ε moves the tie in either
  direction, we do not improvise — the registered branch for that outcome is
  the text. (Expected: tie holds; verify, don't assume.)
- **ENV negative**: re-run the n=87 sensitivity and 200-seed split analysis on
  v9 inputs; expected to stay negative.
- **SPC ≤3.8B withdrawal**: confirmed by construction (the ε-fix is what
  withdrew it).

### 1.5 Quarantine additions

SUP table gains the float-gate row (as-run → v9, affected-panel count 25,
probe count 238). §5a stays a *description* correction outside SUP, per
`V8_FINDINGS.md`.

---

## WS2 — dmmlu non-determinism *(timeboxed: 1 day + optional bounded re-runs)*

1. **Reproduce** on the two known divergent rows; isolate the cause by varying
   batch size / batch composition / padding (prime suspects), and check
   TF32/cudnn determinism flags.
2. **Fix forward**: deterministic flags + fixed batch composition (or
   per-item scoring); verify on a sample panel.
3. **Retroactive handling — decision point.** Trace-recorded dmmlu carries
   ±1-item noise that cannot be fixed without re-running MMLU. Default: keep
   trace values, let the integer gate absorb the boundary, and document the
   granularity as measurement noise. **Optional surgical arm:** count the rows
   whose (pre−post) sits within ±1 item of the budget (from traces, free); if
   the count is small (expected: dozens), re-run MMLU deterministically for
   those rows only. Count first, then decide — do not re-run blind.
4. **Re-scope the reproducibility statement**: "sweep-pipeline bit-determinism
   verified (108/108); MMLU scoring exhibited batch-context sensitivity of up
   to 1 item (cause: ___, fixed for v9 by ___)". The 108/108 claim is scoped,
   not deleted.
5. Forward-looking (paper limitation + repo issue, not a pre-deadline task):
   raise MMLU to ≥1,000 items so the budget is not 4× the metric's
   granularity.

---

## WS3 — Draft v3 change list *(structure editable now; numbers after WS1)*

1. **All numbers → v9.** No exceptions; the number-source audit (WS4.3)
   enforces it.
2. **§5a module set:** per-family resolved-modules table (phi =
   `qkv_proj` fused, **no o_proj**); restrict "(q,k,v,o)" to the eight
   families where it holds; add the **phi counterexample sentence next to
   LOPO** (o_proj is load-bearing *within the q,k,v,o parameterization*, not
   universally necessary).
3. **§5b base-only:** strike every base-only claim. Reframe the limitation:
   instruction-tuned checkpoints throughout (deployment-relevant);
   first base measurement (INS-B) shows the phenomenon is **not an instruct
   artifact** (+0.605 vs matched-norm null +0.0004); base generalization is
   blocked by the harness (`chat_prompt`), i.e. costs a prompt path, not a
   rerun.
4. **DEC section:** Outcome B with the **efficiency reframe** — magnitude
   selection buys the effect at low α / low collateral ("where the signal is
   cheapest"), with (i) *concentrated, not exclusive* (C-a − C-rand excludes
   0), (ii) the **grid-conditional caveat** stated with the α-continuation
   number, (iii) the in-envelope pre-selection note. Related Work keeps the
   pre-drafted Outcome-B paragraph, adjusted to the **both-partly-right**
   duality: DARE's redundancy is real (C-a > C-rand); the deployment-budget
   concentration is ours (C-ref ≫ C-a at matched grid).
5. **LOPO into the main text:** minimal edit = **v + o**; q, k droppable;
   density ≠ importance sentence (v: 0.64× density, load-bearing); compounds
   with sparsity for the patch-size story and is the robust route below 1%.
6. **SUP1:** the sign field tracks **corpus content** (bimodal axis overlap:
   Age×Race 0.769 vs ~0.49 elsewhere); caveats verbatim — single-cell
   designer arm, disjoint targets, no within-target dissociation claimed;
   fold-enrichment reporting only.
7. **ENV as a negative result:** described and calibrated, **not validated
   out-of-sample** (held-out untestable; sensitivity negative; 200-seed
   analysis: the gate is not hidden, it is not there). Contribution 4 stays
   conditional; contrast_gap keeps predictor status only; one population-scope
   sentence (edit-all is near-optimal in this cell population). Framed as the
   program's 11th pre-registered self-refutation — consistent identity.
8. **SPC:** 7–9B — 99% free at 1.35 MB, 99.9% cliff (−0.074), and the new
   Table-B/abstract-candidate line: **sparsity is what makes the 8B edit
   deployable at all** (every budget failure is low-sparsity; 99%+ never
   fails). ≤3.8B cost claim withdrawn; §5 stands unqualified.
9. **INS:** A — IFEval matched-norm Δ = +0.0000 (qwen7b_it, at +0.698
   removal) / −0.022 ≈ 1.1 SE (llama8b_it), bounds stated (2 targets, 1 axis,
   seed 0, 200/541 prompts). B — only the licensed claim; no base-vs-instruct
   relative statement; the prompt-format control (−0.175 MMLU from format
   alone) goes to the appendix as the cautionary example.
10. **Bugs paragraph** updates to three caught errors (two steering-favoring +
    the float gate), one main-text paragraph, details in appendix; the
    "protocol caught our own errors" sentence stays near the top.
11. **Reproducibility statement** re-scoped per WS2; MMLU-granularity
    limitation added.
12. **Figures:** Fig 2 = 7-condition decomposition (v9 numbers) — headline;
    Fig 3 = SPC curve with budget-fail annotations at low sparsity; LOPO
    bar-pair added (as Fig or table); envelope figure demoted to the ENV
    negative-result section with cal/eval markers.
13. **Abstract v3** restructured per the revision plan around the structural
    finding, with v9 numbers, the sparsity-enables-deployment line, and the
    concentrated-not-exclusive phrasing.
14. Terminology sweep re-run after edits ("no additional inference-time
    intervention compute"; "no measurable degradation"; no "entire signal";
    no "zero inference cost"; no base-only).

---

## WS4 — Hygiene *(continuous)*

1. **PREREGISTRATION.md amendment log:** the ε decision recorded with date and
   rationale; every re-verdict from WS1.4 appended as it lands.
2. **Citation ban extended:** only v9 numbers citable; `RESULTS_METHOD_v9.md`
   is the single source; earlier method files gain a superseded banner.
3. **Number-source audit script:** every number in the manuscript maps to an
   artifact file + script; run as a pre-submit gate. A number the script
   cannot trace does not ship.
4. **Companion paper (SPLIT):** frozen until after the ICLR submission; no
   companion work on the critical path.

---

## WS5 — Sequence, gates, freeze

**Order (blocking, not calendar):**

1. WS1.1–1.2 (the corrected gate + re-selection) — everything waits on this.
2. WS1.3–1.5 diff report → re-verdicts → quarantine. WS2 runs in parallel,
   timeboxed.
3. WS3 structural edits proceed immediately (they don't need numbers);
   number-bearing sentences fill from the diff report.
4. WS4.3 audit script runs green.
5. Freeze: abstract by 09-19, full by 09-24.

**Decision points:**

- **D1 (WS2.3):** boundary-row count → surgical MMLU re-run yes/no.
- **D2 (WS1.4):** if the tie moves under ε → adopt the registered fork branch
  for the new outcome; abstract adjusts accordingly. No improvisation.
- **D3:** if any other registered verdict flips → amendment + text follows
  data, same rule.

**Freeze checklist:**

- [ ] Corrected gate merged as one shared function; unit test on the known
      boundary case passes.
- [ ] `RESULTS_METHOD_v9.md` regenerated end-to-end by script.
- [ ] Diff report complete; every CI-status change reviewed; D2/D3 resolved.
- [ ] Repro statement re-scoped; dmmlu cause documented.
- [ ] §5a module table in; §5b base-only claims struck (grep returns zero).
- [ ] Contribution 4 conditional; ENV negative section in.
- [ ] SUP table includes the float-gate row; superseded banners on old files.
- [ ] Number-source audit green; `[TODO]` grep returns zero outside
      explicitly-labeled limitations.
- [ ] Terminology greps return zero ("entire signal", "zero inference cost",
      "base-only", "no cost at").
- [ ] PREREGISTRATION.md amendment log current as of the freeze commit.