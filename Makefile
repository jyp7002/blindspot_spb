# v10 finishing pass — the freeze gate.
#
#   make regen    rebuild every derived v9 artifact the manuscript and figures cite
#   make figures  rebuild figures/fig{1..6}.{pdf,png} + fig*_data.csv
#   make audit    run the number-source audit over text + figure CSVs
#   make freeze   the full gate: regen -> figures -> audit, from a clean tree
#
# The freeze checklist (experiments_v10.md WS-C) consumes `make freeze`'s PASS
# line and nothing else. Every number in the manuscript is written or verified
# by `make audit`; none is hand-typed.

PY      ?= python3
SRC      = src
DRAFT    = binary_debiaser_draft_v3.md
OUT      = binary_debiaser_draft_v3.1.md
FIGDIR   = figures
V10      = results/v10
MANIFEST = audit/manifest_v10.yaml

REGEN_MODULES = dec spc frontier lopo sup1 env size claims artifact_sizes reclaim missing starred alphaext lineage padding
REGEN_STAMPS  = $(addprefix $(V10)/,dec_analysis_v9.json spc_v9.json frontier_v9.json \
                                    lopo_v9.json sup1_matrix.json env_v9.json patch_sizes.json claims_audit.json artifact_sizes.json reclaimed.json missing_recomputed.json starred_sources.json alpha_extension.json lineage_h2.json padding_recompute.json)

.PHONY: all regen manifest figures audit report verify freeze clean-figures clean-v10 help \
        v12-check v12-report v12-astar v12-findings \
        v11-check v11-preflight v11-plan v11-dec v11-ext v11-ins v11-findings v11-report

all: regen manifest figures audit

## regen — rebuild derived v9 artifacts (idempotent; reads results_v9/ + results/)
regen:
	@for m in $(REGEN_MODULES); do \
	  echo "--- v10_regen_$$m"; \
	  $(PY) $(SRC)/v10_regen_$$m.py || exit 1; \
	done

## manifest — regenerate audit/manifest_v10.yaml from the artifacts
manifest:
	$(PY) $(SRC)/v10_make_manifest.py

## figures — every figure reads only artifacts, and writes its own data CSV
figures:
	@mkdir -p $(FIGDIR)
	$(PY) $(SRC)/v10_figures.py

## report — regenerate V10_FINDINGS.md, V10_REMAINING_STARS.md and V10_SESSION_LOG.md
report:
	$(PY) $(SRC)/v10_findings.py
	$(PY) $(SRC)/v10_remaining.py
	$(PY) $(SRC)/v10_session_log.py

## audit — verify mode over the draft + all fig*_data.csv; non-zero exit on FAIL
audit:
	$(PY) $(SRC)/v10_audit.py --draft $(DRAFT) --manifest $(MANIFEST) --figures $(FIGDIR)

## replace — write draft_v3.1 with every *-marked value resolved from artifacts
replace:
	$(PY) $(SRC)/v10_audit.py --draft $(DRAFT) --manifest $(MANIFEST) \
	      --figures $(FIGDIR) --replace --out $(OUT)

## verify — self-tests of the shared substrate and the gate
verify:
	$(PY) $(SRC)/v9_gate.py
	$(PY) $(SRC)/v11_panel.py
	$(PY) $(SRC)/v11_selftest.py
	$(PY) $(SRC)/v12_geometry.py
	$(PY) $(SRC)/v12_opsel.py
	$(PY) $(SRC)/v12_pstar.py selftest
	$(PY) $(SRC)/v12_astar.py selftest
	@$(PY) -c "import torch" 2>/dev/null && $(PY) $(SRC)/v12_selftest.py \
	  || echo "v12 pipeline selftest: SKIPPED (needs torch)"
	$(PY) -c "import sys; sys.path.insert(0,'$(SRC)'); import v10_common, v10_style; print('substrate OK')"

# ---------------------------------------------------------------- v11 -----
# Scale-up (experiments_v11.md). These do not touch any published panel:
# every v11 config writes to a new panel name.
V11_CONFIGS = configs/v11/dec_v11.yaml configs/v11/alphaext_v11.yaml \
              configs/v11/ifeval_v11.yaml configs/v11/big_v11.yaml

## v11-check — validate every scale-up config and show what would run
v11-check:
	@for c in $(V11_CONFIGS); do \
	  echo "--- $$c"; $(PY) scripts/plan.py $$c --dry-run || exit 1; echo; \
	done

## v11-preflight — is THIS box safe to launch a panel on? (CFG=... to target one)
v11-preflight:
	$(PY) scripts/preflight.py $(if $(CFG),--config $(CFG),)

## v11-dec — the three registered Delta_selection reportings (§v11.F): v11dec, v11big, pooled
v11-dec:
	$(PY) $(SRC)/v11_dec_analyze.py
	$(PY) $(SRC)/v11_dec_analyze.py --panel v11big --config configs/v11/big_v11.yaml --out results/v11/big_v11.json
	$(PY) $(SRC)/v11_dec_analyze.py --with-panel v11big --out results/v11/dec_big_v11.json

## v11-ext — alpha-extension saturation (§v11.C)
v11-ext:
	$(PY) $(SRC)/v11_ext_analyze.py

## v11-ins — IFEval generation collateral + reproduction of the published cells
v11-ins:
	$(PY) $(SRC)/v11_ins_analyze.py

## v11-findings — regenerate V11_FINDINGS.md from the analysis artifacts
v11-findings:
	$(PY) $(SRC)/v11_findings.py

## v11-report — all three v11 analyses, then the findings document
v11-report: v11-dec v11-ext v11-ins v11-findings

## v11-plan — write work/<panel>.units.json for one config (CFG=... required)
v11-plan:
	@test -n "$(CFG)" || (echo "usage: make v11-plan CFG=configs/v11/dec_v11.yaml" && exit 1)
	$(PY) scripts/plan.py $(CFG)

## freeze — the gate. Rebuilds everything from artifacts, then audits it.
freeze: clean-figures regen manifest figures audit
	@echo "FREEZE GATE: all stages returned 0"

clean-figures:
	rm -f $(FIGDIR)/*.pdf $(FIGDIR)/*.png $(FIGDIR)/*_data.csv

clean-v10:
	rm -f $(REGEN_STAMPS) $(V10)/audit_report.json

# ---------------------------------------------------------------- v12 -----
# Operating-point selection (experiments_v12.md). Analysis-only targets.

## v12-check — replay gates: every v12 panel must reproduce what it replays
v12-check:
	$(PY) $(SRC)/v12_analyze.py --check-replay
	$(PY) $(SRC)/v12_frontier.py --check-replay

## v12-astar — §v12.E verdict: Delta_selection under alpha* at <=9B (refuses until complete)
v12-astar:
	$(PY) $(SRC)/v12_astar.py

## v12-findings — regenerate V12_FINDINGS.md from results/v12/*.json
v12-findings:
	$(PY) $(SRC)/v12_findings.py

## v12-report — opsel readings, frontier/Table 3, v12 figure panels
v12-report:
	$(PY) $(SRC)/v12_analyze.py
	$(PY) $(SRC)/v12_frontier.py
	$(PY) $(SRC)/v12_figures.py

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
