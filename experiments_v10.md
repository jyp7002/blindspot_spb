# V10_AUDIT_FIGURES.md — the finishing pass: numbers and figures

**Inputs:** `binary_debiaser_draft_v3.md`, `RESULTS_METHOD_v9.md`, the v9
artifact tree (`results_v9/`, `results/v9/`, `results/v8dec/`, `sup1`, `spc`,
`env`, `ins`). **Outputs:** `binary_debiaser_draft_v3.1.md` (every `*`
resolved), `figures/fig{1..6}.{pdf,png}` + per-figure data CSVs,
`results/v10/audit_report.json`. **Gate:** the submission freezes only when
WS-A reports PASS and WS-B builds from a clean tree. Deadline anchors:
abstract 09-19, full 09-24.

Two rules govern everything below:

1. **No hand-typed numbers, anywhere.** The R2-bug rule extends to the
   manuscript and to figures: every number is read from an artifact at build
   time. A value that cannot be traced does not ship.
2. **The draft's `*` convention is a work order, not a footnote.** Every
   `*`-marked value is replaced by machine; every unmarked value is verified
   by machine; the diff is reviewed by a human once.

---

## WS-A — Manuscript number audit (extend `src/v9_number_audit.py`)

### A1. The manifest

One checked-in file, `audit/manifest_v10.yaml`, mapping **every number in the
manuscript** to its source:

```yaml
- id: dec.delta_selection
  text_pattern: "Δ_selection = {value} [{lo}, {hi}]"
  artifact: results/v9/v9_diff_report.json
  key: dec.delta_selection.v9
  format: "+.3f"
- id: c3.retention_small
  derived: binary_mean / fp_mean          # derived numbers carry their formula
  artifacts: [results_v9/ablate/removal.jsonl]
  format: ".1%"
```

Hand-written once; machine-verified forever. Derived quantities (retention %,
pooled means, "16 of 18", "10/10", n-counts) get explicit derivation entries —
counts are auditable numbers, not prose.

### A2. Two modes

- **replace:** for `*`-marked values, write the artifact value into
  `draft_v3.1`, emit `old → new` to the diff log.
- **verify:** for unmarked values, compare against the artifact within
  rounding; any mismatch is a hard FAIL with the manifest id.

### A3. Unmatched policy

A number with no manifest entry is a **ship-blocker** (hard FAIL). Whitelist
only: years, section/figure numbers, version strings, α-grid literals
(4, 16), and the budget constants (4 items / 200 / 1.10) — which are themselves
asserted once against `src/v9_gate.py`.

### A4. Folded greps (single command, all must pass)

- Terminology bans → zero hits: `entire signal`, `zero inference cost`,
  `base-only`, `no cost at`, `beats full precision`.
- `[TODO` → zero outside explicitly labeled limitations.
- `*` residue → zero in number contexts after replacement.
- `[CITE:` → **inventory report** (count + list), not a failure — citations
  are the one manual fill. The companion ref gets a named line so it cannot
  be forgotten.

### A5. Known replacement set (what the diff should show)

C1 cell/pooled values; C2 (n = 502 triple); C3 retentions and their four
sub-splits; C5 granularity deltas; **the frontier per-cell table** — steering
qwen|occ is known to rise under v9, and if `RESULTS_METHOD_v9.md` lacks
per-cell values, they are **regenerated** by extending `src/v9_make_results.py`
(never hand-typed); prompt-baseline numbers (16/18, ppl 1.28, 0.326 → 0.407);
AX table + the +0.910 correlation (recompute under v9-scored removals — the
correlation's inputs are gated values, so it is *not* exempt); `contrast_gap`
(+0.507, n = 504 — re-pooled over v9 selections); INS values; steering/SD/INLP
null gaps. **Expectation check:** the audit re-asserts every registered
CI-status from `v9_diff_report.json` after replacement, so a manifest typo
cannot silently flip a claim.

### A6. Output

`results/v10/audit_report.json` (per-id status), the reviewed diff, the
rewritten `draft_v3.1`, console PASS/FAIL. The freeze checklist consumes the
PASS line, nothing else.

---

## WS-B — Figures (`src/v10_figures.py`, one script, artifact-driven)

### B0. Shared discipline

- One matplotlib style file: ICLR column widths, colorblind-safe palette,
  **fixed condition colors reused across figures** (C-ref, C-a, nulls,
  steering, SentenceDebias, DPO each own one color everywhere).
- Every figure reads only v9 artifacts; **each figure writes its own data CSV
  next to the PDF** (`figures/fig2_data.csv` …), and those CSVs pass WS-A's
  verify mode — figures and text cannot disagree.
- PDF (camera) + PNG (draft) per figure; a `make figures` target rebuilds all
  from a clean tree.

### B1. Fig 1 — the information-removal pipeline *(drawn, no data)*

FP32 magnitudes → one bit per weight → 1% of coordinates → effect preserved →
0.4–1.6 MB merged patch. The patch-size callout is pulled from the size
artifact, not typed. Visual framing per the revision plan: what is *removed*
at each stage, not the software pipeline.

### B2. Fig 2 — the seven-condition decomposition *(headline)*

Horizontal dot-and-CI plot, conditions ordered by v9 mean: C-ref +0.353 …
C-bottom +0.001; cell-bootstrap CIs; support-type vs sign-type encoded by
marker fill vs color; "10/10 cells" unanimity annotation; caption carries the
two registered qualifiers verbatim (concentrated-not-exclusive;
grid-conditional). Source: the v9-re-scored DEC analysis
(`results_v9/…/dec`, re-emitted as `dec_analysis_v9.json` if absent — never
the as-run file).

### B3. Fig 3 — sparsity curves, both tiers

Two panels (≤3.8B, 7–9B). x = retained fraction, log scale, secondary axis in
patch MB; y = removal-vs-dense Δ with CI bands from the v9 SPC values;
**budget-fail markers** on the 8B low-sparsity points (the
sparsity-enables-deployment finding, annotated in-plot); cliff annotations at
99.5/99.9 (small) and 99.9 (big).

### B4. Fig 4 — support localization + causal check

Left: layer × projection fold-enrichment heatmap (`sup1_analysis.json`),
late-layer gradient visible. Right: LOPO paired bars with CIs (q, k, v, o).
Caption: "density is not importance" — v_proj is sparser than q_proj and
load-bearing; phi excluded (fused module), stated.

### B5. Fig 5 — the frontier

Per-cell dots (4 methods × 8 cells) + pooled diamonds with v9 CIs.
**Gate status encoded visually:** filled markers = true-v9 cells, open
markers = as-run (llama/qwen edit cells) — the mixed-gate disclosure lives in
the figure itself, with the 4-cell restricted estimate in the caption.
Table B (deployment properties) stays a LaTeX table beside it, not a plot.

### B6. Fig 6 — operating characterization

pre-skew vs removal scatter over the v9-scored cells; benchmark family by
marker shape; **backfire cells highlighted**; calibration vs held-out split by
outline; correlation annotated with n. Caption states the characterization
*and* the gate's held-out failure in one sentence — the figure must not imply
a validated deployment rule.

### B7. Numbers-in-figures audit

`fig*_data.csv` files are appended to the WS-A manifest; the audit runs once
over text + figure data together. One PASS covers both.

---

## WS-C — Sequence and freeze

1. A1 manifest written → A2 replace-mode run → diff reviewed → A5 expectation
   checks green.
2. B runs in parallel (reads artifacts directly); B7 ties figures into the
   same audit; final `make figures && make audit` from a clean tree.
3. Freeze gate: **audit PASS + figures rebuilt clean + grep block zero +
   citation inventory reviewed** (the only remaining manual item), then the
   abstract goes in (09-19) and the full paper (09-24).

## Z — Checklist

- [ ] `audit/manifest_v10.yaml` covers every number; unmatched = 0.
- [ ] Replace-mode diff reviewed; `*` residue grep = 0.
- [ ] Registered CI-statuses re-asserted post-replacement (D2/D3 invariants).
- [ ] Frontier per-cell v9 values regenerated, not typed; steering qwen|occ
      change visible in the diff log.
- [ ] +0.910 and `contrast_gap` re-pooled under v9 selections before
      replacement.
- [ ] Fig 2 sourced from `dec_analysis_v9.json`; no as-run figure inputs
      anywhere.
- [ ] Fig 5 marker legend distinguishes v9 vs as-run cells.
- [ ] `fig*_data.csv` in the manifest; single audit PASS covers text +
      figures.
- [ ] Terminology/TODO greps zero; citation inventory (incl. companion)
      reviewed by hand.
- [ ] `make figures && make audit` green from a clean checkout at the freeze
      commit.