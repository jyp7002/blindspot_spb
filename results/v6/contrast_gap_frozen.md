# contrast_gap — frozen elicitation-quality ingredient (v6 MV-B / ORD)

Frozen 2026-07-29T07:38:02Z · sha256 in `contrast_gap_frozen.json.sha256` · 132 observations, 114 keys

**Definition.** contrast_gap = designer-level elicitation diagnostic emitted at corpus-build time (elicit_diag.contrast_gap): congruence(biased corpus) - congruence(debiased corpus). Higher = the designer separates its own biased from debiased completions more sharply. Measured on the DESIGNER, independent of any target or removal outcome.

**Frozen before ORD** per experiments_v6 §7 (no-peeking). See `peek_disclosure` in the JSON for what was observed during the 2026-07-29 audit.

Values below are the loader-consistent ones (`matches_loader_corpus=true`): one per designer×axis×seed, describing the corpus file `colab_t2t4.py:732` actually reads.

| designer | ~B | model | occ_gender (s0/s1/s2, mean) | crows_socio (s0/s1/s2, mean) |
|---|---|---|---|---|
| `olmo_sib` | 1 | OLMo-2-0425-1B-Instruct | +0.201/+0.201/+0.201 → **+0.201** | +0.023/+0.023/+0.023 → **+0.023** |
| `granite_3b` | 2.5 | granite-3.1-2b-instruct | +0.403/+0.403/+0.410 → **+0.405** | +0.093/+0.081/+0.081 → **+0.085** |
| `gemma_3b` | 2.6 | gemma-2-2b-it | +0.076/+0.083/+0.076 → **+0.078** | -0.023/-0.023/-0.023 → **-0.023** |
| `gemma_sib` | 2.6 | gemma-2-2b-it | +0.083/+0.083/+0.083 → **+0.083** | -0.023/-0.023/-0.023 → **-0.023** |
| `qwen_3b` | 3.1 | Qwen2.5-3B-Instruct | -0.028/-0.021/-0.035 → **-0.028** | +0.000/+0.000/+0.012 → **+0.004** |
| `qwen_sib` | 3.1 | Qwen2.5-3B-Instruct | -0.035/-0.035/-0.035 → **-0.035** | +0.000/+0.012/+0.012 → **+0.008** |
| `falcon_3b` | 3.2 | Falcon3-3B-Instruct | +0.201/+0.194/+0.208 → **+0.201** | +0.093/+0.116/+0.105 → **+0.105** |
| `llama_3b` | 3.2 | Llama-3.2-3B-Instruct | +0.132/+0.132/+0.132 → **+0.132** | +0.012/+0.012/+0.012 → **+0.012** |
| `llama_sib` | 3.2 | Llama-3.2-3B-Instruct | +0.132/+0.139/+0.139 → **+0.137** | +0.023/+0.023/+0.023 → **+0.023** |
| `phi_3b` | 3.8 | Phi-3.5-mini-instruct | +0.014/+0.014/+0.007 → **+0.012** | +0.140/+0.116/+0.128 → **+0.128** |
| `falcon` | 7 | Falcon3-7B-Instruct | +0.167/+0.167/+0.167 → **+0.167** | +0.093/+0.093/+0.093 → **+0.093** |
| `olmo` | 7 | OLMo-2-1124-7B-Instruct | +0.292/+0.299/+0.299 → **+0.297** | +0.081/+0.081/+0.093 → **+0.085** |
| `qwen` | 7 | Qwen2.5-7B-Instruct | +0.264/+0.264/+0.271 → **+0.266** | +0.047/+0.047/+0.047 → **+0.047** |
| `granite` | 8 | granite-3.1-8b-instruct | +0.403/+0.389/+0.403 → **+0.398** | +0.035/+0.023/+0.047 → **+0.035** |
| `llama` | 8 | Llama-3.1-8B-Instruct | +0.500/+0.493/+0.486 → **+0.493** | +0.093/+0.093/+0.093 → **+0.093** |
| `gemma` | 9 | gemma-2-9b-it | +0.479/+0.479/+0.479 → **+0.479** | +0.070/+0.070/+0.081 → **+0.074** |
| `falcon_large` | 10 | Falcon3-10B-Instruct | +0.396/+0.396/+0.396 → **+0.396** | +0.116/+0.116/+0.116 → **+0.116** |
| `olmo_large` | 13 | OLMo-2-1124-13B-Instruct | +0.410/+0.410/+0.410 → **+0.410** | +0.151/+0.163/+0.151 → **+0.155** |
| `qwen_large` | 14 | Qwen2.5-14B-Instruct | +0.326/+0.312/+0.312 → **+0.317** | +0.198/+0.198/+0.198 → **+0.198** |

## Tier means

| tier | n designers | occ_gender | crows_socio |
|---|---|---|---|
| ≤3.8B | 10 | +0.119 | +0.034 |
| 7–9B | 6 | +0.350 | +0.071 |
| 10–14B | 3 | +0.374 | +0.156 |

## Conflicts retained

falcon_3b was elicited twice (bigbox x run, then re-elicited locally during the x2 run because the pipe/underscore corpus-name mismatch caused a silent cache miss). Both observations are retained with source_log attribution; NEITHER is silently preferred. The corpus the loader now reads is identified by corpus_md5.

| key | x (bigbox) | x2 (local, on disk) |
|---|---|---|
| `falcon_3b\|crows_socioeconomic\|s0` | +0.128 | +0.093 ⚠ |
| `falcon_3b\|crows_socioeconomic\|s1` | +0.140 | +0.116 ⚠ |
| `falcon_3b\|crows_socioeconomic\|s2` | +0.140 | +0.105 ⚠ |
| `falcon_3b\|occ_gender\|s0` | +0.188 | +0.201 ⚠ |
| `falcon_3b\|occ_gender\|s1` | +0.188 | +0.194 ⚠ |
| `falcon_3b\|occ_gender\|s2` | +0.188 | +0.208 ⚠ |
| `llama_3b\|crows_socioeconomic\|s0` | +0.012 | +0.012 |
| `llama_3b\|crows_socioeconomic\|s1` | +0.023 | +0.012 ⚠ |
| `llama_3b\|crows_socioeconomic\|s2` | +0.012 | +0.012 |
| `llama_3b\|occ_gender\|s0` | +0.132 | +0.132 |
| `llama_3b\|occ_gender\|s1` | +0.132 | +0.132 |
| `llama_3b\|occ_gender\|s2` | +0.132 | +0.132 |
| `qwen_3b\|crows_socioeconomic\|s0` | +0.000 | +0.000 |
| `qwen_3b\|crows_socioeconomic\|s1` | +0.012 | +0.000 ⚠ |
| `qwen_3b\|crows_socioeconomic\|s2` | +0.012 | +0.012 |
| `qwen_3b\|occ_gender\|s0` | -0.035 | -0.028 ⚠ |
| `qwen_3b\|occ_gender\|s1` | -0.035 | -0.021 ⚠ |
| `qwen_3b\|occ_gender\|s2` | -0.035 | -0.035 |
