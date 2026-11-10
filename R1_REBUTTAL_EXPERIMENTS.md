# R1 — ICLR 2027 rebuttal 실험 계획

Submission 60701 · *One Bit in a Hundred: The Sparse Signed Substructure of Behavioral Weight Edits*
작성 2026-11-07 · 상태: **pre-registration draft**

이 파일을 commit하고 해시를 `audit/manifest_r1.yaml`에 기록한 뒤 GPU job을 시작한다. 판독 규칙은 실행 후 수정하지 않고, 이후 추가되는 분석은 전부 `post-hoc` 라벨을 단다.

> **TL;DR** — 결정 실험은 E1(zero-GPU, matched-collateral 곡선)과 E4(C-rand의 α\*, validity gate) 두 개. W1을 살릴 후보는 E6(collateral-aware selection)과 E9(held-out collateral). 나머지는 리뷰 답변과 positioning용. GPU 큐 첫 job은 E4의 C-rand.

**정직성 메모.** frozen grid, grid extension, α\* 결과는 이미 봤다. 그래서 기존 데이터에 대한 분석(E1–E3)은 confirmatory가 아니라 "분석 규칙만 사전 고정" 수준이고, 새 GPU run(E1b, E3c fallback, E4 이후)만 온전한 사전 등록이다.

---

## 0. 이 라운드의 질문

pe22, itTC, 5PJm이 같은 지점을 찔렀다. magnitude selection의 이점이 edit-scale 규칙에 따라 사라진다는 것. 그래서 1순위 질문은 하나다.

> **같은 collateral에서 magnitude-selected support가 random support보다 bias를 더 제거하는가, 아니면 "concentration"은 scale 재매개화인가?**

나머지 실험(method 명세, ComPEFT/DARE 비교, DPO 수리, budget 민감도)은 리뷰 답변용이고, 이 질문의 판독을 바꾸지 않는 한 positioning만 정한다. 판독에서 결정으로 가는 규칙은 §5.

### 공통 규약 (논문 프로토콜 계승 + 변경 1개)

- **Gate**: `src/v9_gate.py` integer-item gate. MMLU 200 items 기준 drop ≤ 4, ppl ratio ≤ 1.10 (float, 1e-9 tolerance). Always-on collateral.
- **α\* 규칙**: ladder {1, 2, 4, …, 512}에서 첫 budget failure를 찾고, 마지막 pass와 첫 fail 사이를 3회 log-bisection. 선택은 validation 항목(MMLU 200 + WikiText 20 chunks)만 읽고, 보고는 evaluation 항목. 512까지 실패가 없으면 α\* = 512로 두고 `censored` 표기.
- **변경점**: 새 run은 rung마다 bias removal(pre/post skew 부호 포함)과 validation·evaluation 양쪽 collateral을 모두 기록한다(곡선용). 선택 함수는 collateral만 인자로 받도록 코드 구조로 강제하고 unit test로 확인 — bias-blind 유지.
- **통계**: target×axis cell 단위 percentile bootstrap 10,000회. seed는 cell 안에서 평균, n = cell 수. nan→0은 count와 함께 기록. Primary는 REG10, BIG4(n = 4)는 secondary·descriptive.
- **무작위 요소**: C-a/C-rand support, C-b permutation, layer/tensor shuffle map은 원 run과 같은 seed로 재생성한다. bit-identical 재생성 여부는 E0에서 먼저 확인.
- **산출물**: `results/r1/<E-id>/`, 스크립트 `src/r1_<eid>.py`, figure는 audit된 CSV에서만 생성.
- **Precision**: 기존과 같은 bf16. 8-bit 로딩 금지(기존 원칙 유지).

### Cell 집합

| 이름 | 구성 | seeds |
|---|---|---|
| REG10 | gemma-2.6B, llama-3.2-3B, qwen-3B, phi-3.8B, qwen-7B × occ_gender; phi, qwen-3B × bbq_Age; qwen-3B × bbq_Race_ethnicity; qwen-3B, phi × ss_intra | 3 |
| BIG4 | gemma-2-27B, qwen2.5-32B × {occ_gender, bbq_Age} | 3 |
| FRONT8 | frontier 8셀. E0에서 확정 (예상: {gemma-2.6B, phi-3.8B, llama-3.2-3B, qwen-3B} × {occ_gender, crows_socio}) | 3 |

---

## 1. 우선순위 개요

| ID | 내용 | GPU (H100 추정) | 답하는 리뷰 | 우선 |
|---|---|---|---|---|
| E0 | spec, provenance, supplementary audit | 0 | pe22 Q1/Q3/Q5, 5PJm Q1–Q3 | P0 |
| E1 | C-ref vs C-a removal–collateral 곡선 | 0 (E1b ~5h) | pe22 W1, itTC Q1, 5PJm Q5 | **P0 ★** |
| E2 | ρ_k와 curvature 회계 | ~0 | pe22 Q2 | P0 |
| E3 | removal 정의, budget 재게이팅, select/report split | 0 (fallback ~4h) | 5PJm Q2/Q3, itTC Q2 | P0 |
| E4 | budget-matched null + necessity under α\* | ~30h (E4b ~10h) | validity gate, 5PJm W3 | **P1 ★** |
| E5 | operator 2×2 + ComPEFT under α\* | ~25h | 5PJm W2/Q4 | P1 |
| E6 | collateral-aware support selection (H-act) | ~25h | W1 rescue | P1 |
| E7 | DPO repair | ~10h | pe22 Q4 | P1 |
| E8 | α\* backoff, budget 민감도 | ~6h | pe22 W3, 5PJm Q6 | P1 |
| E9 | held-out capability battery at α\* | ~15h | itTC Q1 | P1 |
| E10 | v13 연결 (seeded random support, α\* density floor) | v13 | 5PJm W3, Branch B | P2 |
| E11 | budget-limited frontier | ~10h | 선제 대응 | P2 |
| E12 | 2nd behavior (sycophancy) | 1–2일 | itTC Q3, 5PJm W6 | P3 |

GPU 시간은 각 실험 첫 cell 실측 후 보정한다. 큐 순서: **E4 C-rand** → E6a → E5 → E6b → E7b → E8 → E9 → E4b/E1b → E11.

---

## 2. P0 — zero-GPU (D0–D1)

### E0. Spec, provenance, supplementary audit

**목적.** Sec 3 재작성과 pe22 Q1/Q3/Q5, 5PJm Q1–Q3 답변에 필요한 사실을 코드와 artifact에서 직접 추출한다. 의도가 아니라 코드가 실제로 한 것을 적는다.

**추출 항목.**
1. ΔW 구성: C+/C−에 각각 LoRA를 학습해 차를 낸 건지 단일 contrastive objective인지, 어느 쪽을 negate하는지, loss(토큰 범위 포함), cell별 pair 수·differing pair 수·토큰 수, optimizer·LR·steps·batch, LoRA r/alpha/dropout, family별 target module, seed, identical pair 처리.
2. per-tensor scale `s_t`의 정확한 정의(어떤 support 위의 mean |ΔW|인지).
3. elicitation prompt 원문(diagnosis framing)과 designer 설정.
4. benchmark family별 skew 정의(부호, 범위, 정규화), removal이 signed `pre − post`인지, 반대 부호로 넘어간 post skew(overshoot)의 처리.
5. 4-config in-budget argmax가 읽는 bias probe item 집합이 보고용 item과 같은지 — method별로.
6. FRONT8 목록과 고정 시점(pre-registration 커밋·타임스탬프). steering 숫자가 생기기 전이었는지.
7. DPO: cell별 informative pair 수, skip(<20) 셀, skip→0으로 들어간 FRONT8 셀 수, hyperparameter, collateral persist 여부.
8. `alpha_trace.jsonl` schema: rung별 removal, pre/post skew(부호), per-item probe score, MMLU item 단위 결과, ppl이 있는지. → E1과 E3의 zero-GPU 경로가 여기서 결정됨.
9. 무작위 support와 permutation의 seed 재생성이 bit-identical인지(E4–E6, E10 공통 전제).
10. Supplementary: 실제 업로드된 파일 vs 논문 claim(code, corpora, edits + α traces, manifest, figure CSVs). Anonymization grep: username, 홈 경로, 기관명, 이메일, hostname, wandb/HF entity, git author/committer, notebook metadata.

**산출물.** `results/r1/E0/SPEC.md` (항목별 값 + 출처 `file:line`). 10번에서 빠진 게 있으면 그날 바로 보완 업로드하거나 anonymous link를 준비.

### E1. Matched-collateral 곡선: C-ref vs C-a ★

**목적.** 이 라운드의 결정 실험. pe22 W1, itTC Q1, 5PJm Q5(곡선과 update norm)에 직접 답한다.

**데이터.** REG10 × 3 seeds × {C-ref, C-a} × α ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512} (frozen grid + 두 extension). 각 점의 removal `R`, ppl ratio `P`, MMLU drop `D`(items), 전부 evaluation 항목 기준. α\* 점은 collateral이 validation 기반이라 곡선에 섞지 않고 overlay로만 표시. E4 이후에는 C-rand 곡선을 세 번째 arm으로 추가.

**분석.**
1. 각 (cell, seed, arm)에서 `P(α)`는 log α에 대해 isotonic 단조 fit, `R(α)`는 단조 제약 없이(큰 α에서 꺾일 수 있음) piecewise-linear 보간. `P(α) = c`를 풀어 `R(c)`.
2. Primary 축은 ppl(연속): c ∈ {1.02, 1.04, 1.06, 1.08, 1.10}. 두 arm이 모두 도달한 c만 쓰고, 미도달은 count로 기록.
3. Secondary 축은 MMLU items(이산): b ∈ {2, 4, 6, 8}에서 `max{R(α) : D(α) ≤ b}`.
4. `Δ_matched(c) = R_ref(c) − R_a(c)`: seed 평균 후 cell bootstrap.
5. Sanity check: b = 4와 P ≤ 1.10의 joint gate는 논문의 extension estimand(+0.104 [+0.036, +0.183])를 재현해야 한다.
6. 균질성: arm별로 `log(P − 1) = log h + p · log α`를 `P − 1 ≥ 0.005`인 점으로 fit. p(≈ 2 예상)와 `h_ref / h_a`를 α\* 비율의 제곱과 비교. `R / sqrt(P − 1)`이 α에 대해 평평한지도 확인.
7. Update norm: matched c에서 `‖αE‖` (Frobenius) 비.
8. Figure: cell별 small multiples(removal vs ppl ratio, 두 arm, frozen-grid 점과 α\* 점 표시)와 pooled 패널.

**사전 등록 판독** (c = 1.10 primary, 1.05 secondary).
- **E1-A (concentration 유지)**: `Δ_matched(1.10)`의 CI가 0을 배제하고 점추정 ≥ +0.071 (frozen-grid +0.284의 1/4. 등록된 closure 기준과 같은 비율).
- **E1-B (scale 재매개화)**: c = 1.10과 1.05 모두에서 CI가 0을 포함.
- **E1-C (작은 잔여)**: 그 외의 양수 결과. "small residual concentration"으로만 보고하고 headline으로 쓰지 않는다.
- 음수 방향으로 0을 배제하면 **E1-B′**(random 우위)로 따로 보고.

**E1b (GPU, 선택).** BIG4에 α ∈ {32, 64, 128}을 두 arm 모두 추가해 같은 분석. ~5 GPU-h.

### E2. ρ_k와 curvature 회계

**목적.** pe22 Q2. α\*에서의 tie가 first-order 이론으로 예측되는 결과인지 확인한다.

**절차.**
1. (cell, seed)별로 LoRA A·B에서 dense ΔW를 재구성(기존 17.6 GB peak 경로).
2. 실제 scale 구성을 반영한 first-order alignment: `A_ref = Σ_t s_t · Σ_{i ∈ S_t(top)} |ΔW_i|`. `A_a`는 같은 식을 C-a의 실제 random support(seed 재생성)로 계산. `ρ_eff = A_ref / A_a`.
3. 참고 통계: 전역 ρ_k(k = 1%), Gaussian 기준값 3.62, |ΔW|의 kurtosis, top-1% mass share.
4. `r_α = α*(C-a) / α*(C-ref)` (seed median)와 비교. first-order 예측 removal 비 `ρ_eff / r_α` vs 관측 `R_ref(α*) / R_a(α*)`.
5. E1의 `h_ref / h_a`, `ρ_eff²`, `r_α²`를 한 표에.

**사전 등록 판독.**
- **E2-P (parity 예측)**: `median(ρ_eff / r_α)` ∈ [0.75, 1.33]. → "α\* tie는 first-order 예측대로다. top-k는 α당 removal을 ρ배 더 얻고, α²당 curvature를 ~ρ²배 더 낸다." pe22 Q2 답변과 App H 수정에 그대로 쓴다.
- **E2-N (first-order 실패)**: 범위 밖. → 포화나 좌표 간 상호작용으로 보고하고, App H의 "does not claim" 단락에 명시.

### E3. Re-scoring robustness

**E3a. removal 정의** (5PJm Q2). (i) signed `pre − post`(현행), (ii) `|pre| − |post|`, (iii) `(|pre| − |post|) / |pre|`. 재계산 대상: Sec 5.2의 두 null 대비, Δ_selection(frozen, extension, α\*), binarization retention, frontier 차이 3개, pre-skew 상관, backfire 집합. overshoot(부호 반전) 셀 수와 크기를 표로.

**E3b. budget 재게이팅** (itTC Q2). persisted trace 전부를 MMLU b ∈ {2, 4, 6, 8} × ppl ∈ {1.05, 1.10, 1.20}, 그리고 ppl-only로 재게이팅. 대상: frozen·extension Δ_selection, frontier 차이, edit in-budget 비율, sparsity panel. CI 상태가 바뀌는 estimand를 표로. α\*의 budget 민감도는 removal 재측정이 필요해서 E8로 넘긴다.

**E3c. select/report split** (5PJm Q3). per-item probe score가 있으면 `sha256(item_id) mod 2`로 H1/H2를 나눈다(분석 전 고정). H1에서 4-config argmax → H2에서 보고, 역방향도 해서 평균. 모든 frontier method와 decomposition condition에 적용하고, 절대값 감소폭과 주요 대비의 CI 상태를 보고. **Fallback**: per-item 기록이 없으면 bit-deterministic 재생성으로 probe만 다시 평가(likelihood only, ~4 GPU-h).

**판독.** 사전 fork 없음. 전부 robustness 보고다. 단, 어떤 headline이든 CI 상태가 바뀌면 rebuttal에 그대로 공개한다.

---

## 3. P1 — GPU (D1–D5)

### E4. Budget-matched null과 necessity under α\* ★

**목적.** 두 가지.
1. **Validity gate**: α\*에서 C-a가 "회복"한 removal이 true sign 정보인지, 아니면 큰 α에서 generic degradation이 skew를 평평하게 만든 것인지. 지금의 α\* 분석에는 budget-matched null이 없고, null(≈ 0)은 frozen grid(α ≤ 16)에서만 검증됐다.
2. **5PJm W3**: necessity 조건들이 큰 scale에서 회복하는지.

**설계.** REG10 × 3 seeds, α\* 규칙 동일, rung별 removal 기록. 조건 순서: **C-rand(최우선)** → C-b → C-layershuf → C-tensorshuf(phi 제외 7셀) → C-bottom.

**E4b.** BIG4에서 C-rand만. 27–32B의 α\* gap이 −0.037이라 여기서 null이 특히 중요하다. ~10 GPU-h.

**사전 등록 판독.** `q = C-rand(α*) / C-a(α*)`.
- **G1-valid**: q ≤ 0.25이고 `C-a(α*) − C-rand(α*)`의 CI가 0을 배제. → α\* 비교는 정보 대 정보. E1 판독을 그대로 해석한다.
- **G1-partial**: 0.25 < q < 0.5, 또는 q ≤ 0.25인데 `C-a(α*) − C-rand(α*)`의 CI가 0을 포함. → E1을 C-rand 곡선과 함께 보고한다. E1-B가 나와도 "C-a 쪽에 degradation 성분이 섞인 tie"라는 단서를 달고, Branch 판정에서는 E1-C로 취급한다.
- **G1-degradation**: q ≥ 0.5. → α\*에서의 C-a 회복은 대부분 generic damage. budget 경계의 removal이 edit 특이적이지 않으므로 모든 method에 budget-matched null이 필요하다(→ Branch C).
- **N-hold**: C-b, C-layershuf, C-bottom의 α\* removal이 각각 ≤ 0.25 × C-ref(α\*). → necessity claim을 budget-limited scale까지 확장. 하나라도 아니면 그 조건의 claim은 deployment grid로 한정한다(5PJm 제안대로).

**비용.** 4 조건 × 30 runs + C-tensorshuf 21 runs, 각 ~13 rungs ≈ 30 GPU-h (rung별 probe 평가 포함 시 +20%).

### E5. Operator 2×2 + ComPEFT under α\*

**목적.** 5PJm W2/Q4. 같은 ΔW에서 magnitude trimming(TIES의 trim 단계), 원래의 DARE, ComPEFT와 비교해 "C-a는 DARE가 아니다"는 지적을 해소한다.

**Arms.** REG10 × 3 seeds, density 1%, merge `W ← W − α·Ê`.

| arm | support | 값 | scale |
|---|---|---|---|
| O-trim | top-1% (magnitude) | fp ΔW | α |
| O-dare | random 1% (C-a와 동일 support) | fp ΔW × 100 (= 1/(1−p), p = 0.99) | α |
| O-compeft | top-1% (magnitude) | sign | 전역 σ(ΔW) × α |
| C-ref (기존 α\*) | top-1% (magnitude) | sign | s_t × α |
| C-a (기존 α\*) | random 1% | sign | s_t × α |

fp arm은 자연 scale이 α ≈ 1 근처라서 새 arm의 ladder는 {2^-6, …, 2^9}로 넓힌다(첫 실패 + 3 bisection, 선택 규칙 동일). rung별 removal을 기록해서 E1식 곡선도 만든다.

**분석.** α\*에서의 removal과 pairwise 차이.
- 값 vs sign: `C-ref − O-trim`(top), `C-a − O-dare`(random)
- support: `O-trim − O-dare`(fp), `C-ref − C-a`(sign, 기존)
- scale convention: `C-ref − O-compeft`

**사전 등록 판독** (positioning 결정용).
- **E5-1**: `C-ref − O-trim`의 CI가 0을 포함 → "matched collateral에서도 1 bit는 공짜" (frozen-grid retention 87–105%보다 강한 형태).
- **E5-2**: `O-trim − O-dare`의 CI가 0을 포함 → fp에서도 DARE-consistent. support selection은 값 정밀도와 무관하게 matched collateral에서 의미가 없다.
- **E5-3**: `C-ref − O-compeft`의 CI가 0을 포함 → per-tensor scale은 차별점이 아니다.
- 결과와 무관하게 텍스트는 "operator는 ComPEFT 계열이고, 기여는 decomposition"으로 간다.

**비용.** 3 arms × 30 runs × ~16 rungs ≈ 25 GPU-h.

### E6. Collateral-aware support selection (가설 H-act)

**가설.** linear layer에서 |ΔW_ij|는 대략 input channel 크기 n_j에 비례해 커지고, capability curvature도 n_j²에 비례한다(GPTQ의 `H = XXᵀ`, Wanda의 `|W|·‖x‖`와 같은 논리). 그러면 top-|ΔW| 선택은 high-activation channel, 곧 high-curvature 좌표를 고르는 셈이다. α\* 비율 4.4가 함의하는 ~19배(= 4.37²) curvature의 출처 후보이고, Fig 2a에서 입력이 RMSNorm을 거치지 않는 o_proj가 가장 dense한 것과도 맞는다.

**리스크.** bias gradient 자체도 n_j에 비례하면 정규화 이득이 상쇄될 수 있다. 결과는 보장되지 않는다.

**E6a. 진단** (~1 GPU-h).
1. Calibration: WikiText-103 train에서 256 seq × 512 tok, gating용 validation·evaluation chunk와 disjoint. 편집 대상 projection마다 input hook으로 `n_j = RMS over tokens of x_j`. phi는 fused qkv 입력.
2. C-ref support의 n_j² enrichment(tensor별, pooled) vs C-a support(≈ 1 예상).
3. `h_proxy(S) = Σ_t s_t² · Σ_{(i,j) ∈ S_t} n_j²` (diagonal, input-side only). `h_proxy(C-ref) / h_proxy(C-a)`를 `r_α²`(≈ 19), E1의 `h_ref / h_a`와 비교.
4. o_proj와 q/k/v의 n_j 분포 비교(Fig 2a의 o_proj 1.32×를 설명하는지).

**E6b. 선택 규칙 α\* run** (REG10 × 3 seeds).
- **S-act1 (primary)**: score `|ΔW_ij| / n_j`
- S-fisher (secondary): `|ΔW_ij| / sqrt(F_ij)`, F는 WikiText calibration NLL의 empirical Fisher diagonal
- S-wanda (control): `|ΔW_ij| · n_j`
- 공통: C-ref의 per-tensor count `k_t`와 scale `s_t`를 그대로 두고 tensor 안에서만 좌표를 다시 고른다(tensor 할당 효과 분리). sign = sign(ΔW_ij). 전역 재할당 variant는 post-hoc.

**사전 등록 판독.**
- **E6-M (메커니즘)**: C-ref support의 n_j² enrichment ≥ 1.5×(CI가 1을 배제)이고, h_proxy 비의 median이 `r_α²` median의 [0.25, 4]배 안.
- **E6-W (method 승리)**: `S-act1(α*) − C-ref(α*)`, `S-act1(α*) − C-a(α*)`, `S-act1(α*) − C-rand(α*)`(E4) 세 CI가 모두 0을 양의 방향으로 배제. → Branch A′.
- **Control**: `S-wanda(α*) ≤ C-ref(α*)` (점추정). 어긋나면 메커니즘 해석은 보류.
- S-act1이 실패하고 S-fisher만 이기면 "secondary, 확인 필요"로만 보고하고 Branch 결정에는 쓰지 않는다.

**비용.** E6a ~1 GPU-h, E6b 3 arms × 30 runs ≈ 20–25 GPU-h(Fisher 포함).

### E7. DPO repair

**E7a (zero-GPU).** FRONT8 셀별 pair 수, differing pair 수, skip 여부, DPO removal, 그리고 skip 셀을 뺀 "trainable cells only" 평균. DPO 행에는 collateral이 persist되지 않아 Fig 4 캡션의 "matched collateral"이 DPO에는 성립하지 않는다 — 정정 문구 준비.

**E7b.** differing pair만으로 DPO 재학습. identical pair는 DPO margin이 정확히 0이라 gradient도 0이므로, 이들을 빼는 건 사실상 effective LR 보정이다. LoRA r16, edit과 같은 module, β = 0.1, epoch 고정. 4-config 선택 공간 = LR {1e-5, 5e-5} × merge scale {1, 2}. skip 기준은 differing pair ≥ 8로 낮추고 cell별 n을 공개. always-on integer-item gate, collateral persist, 3 seeds.

**E7c (진단).** differing 비율이 높은 cross-large designer corpus가 있는 셀(phi·occ, qwen·crows 등)에서 DPO vs edit. "preference objective는 pair가 달라야 쓸 수 있다"는 설명의 직접 증거.

**사전 등록 판독.**
- **E7-1**: `edit − DPO(E7b)`의 CI가 0을 포함 → "DPO removes none"을 전부 철회하고 "informative pair로 학습하면 DPO도 비슷하다"로 교체.
- **E7-2**: 여전히 0을 배제하고 DPO ≈ 0 → claim은 유지하되 abstract에서 빼고, E7c를 메커니즘 근거로 붙인다.

**비용.** ~10 GPU-h.

### E8. α\* backoff과 budget 민감도

**목적.** pe22 W3, 5PJm Q6(evaluation pass 57–60%). α\*가 deployment rule이 아니라 경계 측정 장치라는 걸 데이터로 보인다.

**설계.** C-ref, C-a(E6-W면 S-act1 추가) × REG10 × 3 seeds에서 α ∈ {α\*, α\*/√2, α\*/2}마다 evaluation gate pass율, 1,000-item MMLU drop, removal, Δ_selection을 측정. 추가로 budget b ∈ {2, 8} items로 α\*를 다시 정해 같은 측정(α\*(b)는 기존 rung trace로 계산하고 removal만 GPU로).

**사전 등록.** deployment rule = α\*/√2 (quadratic 모델에서 collateral ≈ ½). 목표: evaluation pass ≥ 90%. 균질성이 맞으면 Δ_selection은 backoff에 불변이어야 한다 — 이 예측도 함께 판독.

**비용.** ~6 GPU-h.

### E9. Held-out capability battery at α\*

**목적.** itTC Q1의 정직한 답. 지금의 "matched"는 MMLU와 ppl 기준일 뿐이다. random arm은 ~4배 큰 perturbation이라 gate에 쓰지 않은 capability에서 더 잃을 수 있다.

**설계.** unedited, C-ref(α\*), C-a(α\*), C-rand(α\*, E4) × REG10.
- Likelihood task(3 seeds): ARC-Challenge, HellaSwag(val 2,000 고정 subset), TruthfulQA-MC1, Winogrande, MMLU held-out slice(gate·evaluation용 400 items 제외).
- Generation task(seed 0): GSM8K 250, IFEval 541.

**Primary metric.** likelihood task들의 macro-average Δacc(vs unedited). `C-ref − C-a`를 cell bootstrap.

**사전 등록 판독.**
- **E9-H1**: `C-ref − C-a`(macro)가 양수이고 CI가 0을 배제 → "magnitude selection은 held-out collateral 효율을 산다" (Branch A″의 근거).
- **E9-H0**: CI가 0을 포함 → 숨은 이점 없음.
- generation task와 개별 task는 descriptive.

**비용.** ~15 GPU-h.

---

## 4. P2 / P3

### E10. v13 연결

- **v13-1, seeded random support**: C-a support가 (seed, tensor shape, k_t)에서 기계와 무관하게 bit-identical로 재생성되는지. patch = sign bits + seed + s_t + α\*. index 없는 크기를 C-ref의 entropy-coded 크기와 비교한다. Branch B의 실용 결론이 된다.
- **v13-2, α\*에서의 density floor**: sparsity 곡선(dense, 90, 99, 99.5, 99.9%)을 α\*로 다시. scale이 자라면 99.9% cliff가 움직이는지 — 5PJm W3("perturbed edit도 큰 scale에서 회복하나")의 sparsity 버전.

v13 결과가 이미 있으면 이 형식(판독 → Branch 기여)으로만 정리한다.

### E11. Budget-limited frontier (선제 대응)

**이유.** 리뷰어는 아직 안 물었지만 같은 논리가 frontier에도 걸린다. steering은 configuration의 94.8%가 in budget이었다. C-a가 frozen grid에서 그랬듯 strength grid에 갇혀 있었을 수 있다. decomposition은 matched collateral로 판정하면서 frontier는 고정 grid로 두면 일관성이 깨진다.

**설계.** FRONT8 × 3 seeds.
- edit: C-ref α\*.
- steering: layer를 기존 4-config에서 더 자주 선택된 쪽으로 사전 고정하고 strength만 budget-limited(α\*와 같은 ladder와 bisection, bias-blind).
- DPO: E7b의 LR 1e-5로 고정하고 merge scale만 budget-limited.
- SentenceDebias: rank ladder {1, 2, 4, 8, 16} (선택).
- 각 method의 budget-matched null(steering은 matched-norm random direction, SentenceDebias는 random subspace)도 같은 규칙으로.

**사전 등록 판독.** `steering(s*) − edit(α*)`의 CI가 0을 양의 방향으로 배제하면 frontier claim을 "budget-limited scale에서는 steering ≥ edit"으로 바꾼다. CI가 0을 포함하면 tie 유지.

**비용.** ~10 GPU-h.

### E12. 2nd behavior: sycophancy (resubmission tier)

- **선정 이유**: likelihood A/B 형식이라 기존 pipeline(likelihood self-diagnosis, likelihood probe, MMLU/ppl gate)에 그대로 맞고, CAA(Rimsky et al., 2024)가 같은 behavior를 다뤄 steering 비교가 자연스럽다.
- **데이터**: Perez et al. (2022) model-written evals의 sycophancy subset(A/B 형식, 라이선스 확인). App F onboarding gate(polarity unit test, half당 item ≥ 40, pre-skew 측정).
- **모델**: qwen-3B, llama-3.2-3B, gemma-2.6B (+ qwen-7B).
- **조건**: 7 conditions frozen grid + C-ref, C-a, C-rand의 α\*.
- **사전 등록 예측**: frozen grid에서 C-ref > C-a > C-rand ≈ C-b ≈ C-bottom ≈ 0. α\*에서의 C-ref vs C-a는 REG10 판독(E1, E4)을 그대로 예측으로 등록.
- **대안**: XSTest over-refusal 감소(benign 방향의 refusal edit). generation과 judge가 필요해 비용이 크다.

---

## 5. 판독 → 결정

순서: G1(E4) → G2(E1) → G3(E6, E9).

| 조건 | Branch | 의미 | 다음 venue |
|---|---|---|---|
| G1-degradation | C | budget 경계의 removal이 generic damage에 오염됨. 모든 method에 budget-matched null을 붙이고 frontier를 재판정. "null은 config-matched가 아니라 budget-matched여야 한다"는 것 자체가 protocol 결과라 논문 방향을 재검토 | 재검토 |
| G1-valid + E1-A | A | matched-collateral 증거로 concentration 유지, 제목 유지(qualifier 포함). E6-W면 method 절 추가 | ICML 2027 |
| G1-valid 또는 partial + E1-B/C + E6-W | A′ | collateral-aware selection이 새 method claim. 제목 교체 | ICML 2027 |
| G1-valid 또는 partial + E1-B/C + E6 실패 + E9-H1 | A″ | "magnitude selection은 held-out collateral 효율을 산다". 약한 claim, 제목 수위 하향 | ICML 또는 TMLR |
| 그 외 | B | "behavioral edit은 distributed sign field, support selection은 scale 재매개화, seed 재생성 support로 index 비용 제거(E10)" | TMLR, 라인 종료 |

- G1-partial이면 E1은 E1-C로 취급하므로(E4 참고) Branch A는 나올 수 없다.
- E5, E7, E8, E11은 Branch를 바꾸지 않고 텍스트 positioning과 개별 claim의 수위만 정한다. 단 E11에서 steering이 이기면 abstract의 frontier 문장은 Branch와 무관하게 수정한다.

---

## 6. 리뷰 질문 → 실험 → 들어갈 곳

"Sec"는 논문 section, "응답 1차/2차"는 OpenReview 답변 회차.

| 리뷰 | 질문·약점 | 실험 | 들어갈 곳 |
|---|---|---|---|
| pe22 | W1 scale 의존 | E1, E4, E6, E9 | 응답 1·2차, Sec 5.3 |
| pe22 | W3 budget noise, pass 57–60% | E3b, E8 | 응답 2차, Sec 6 |
| pe22 | Q1 removal 정의, loss, data size | E0, E3a | 응답 1차, Sec 3 |
| pe22 | Q2 ρ_k와 ~4× 비율 | E2 | 응답 1차, App H |
| pe22 | Q3 frontier 8셀 선정 | E0 | 응답 1차 |
| pe22 | Q4 DPO identical pairs | E7 | 응답 2차, Sec 5.7 |
| pe22 | Q5 supplementary 접근 | E0 | 응답 1차 |
| itTC | W1/Q1 scale 말고 이점 | E1, E6, E9 | 응답 2차 |
| itTC | Q2 budget 민감도 | E3b, E8 | 응답 2차 |
| itTC | Q3 다른 behavior | E12 + claim 축소 | 응답 1차(범위 축소), 재제출 |
| 5PJm | W1/Q1 method 명세 | E0 | 응답 1차, Sec 3 |
| 5PJm | Q2 skew, removal, overshoot | E0, E3a | 응답 1차 |
| 5PJm | Q3 select/report 분리 | E3c | 응답 1·2차 |
| 5PJm | W2/Q4 ComPEFT, DARE, trim | E5 | 응답 2차, Sec 2, Sec 5.3 |
| 5PJm | W3 perturbed edit의 큰 scale | E4, E10 | 응답 2차 |
| 5PJm | Q5 곡선, update norm | E1 | 응답 1차 |
| 5PJm | Q6 pass 57–60% | E8 | 응답 2차 |
| 5PJm | W6/Q7 일반화 | E12 + claim 축소 | 재제출 |
| 2RwF | presentation | — | revised PDF |
| (선제) | frontier의 grid 의존 | E11 | 응답 2차 또는 재제출 |

---

## 7. 일정 (D0 = 2026-11-07)

- **D0–D1**: E0, E1, E2, E3, E7a (zero-GPU). 이 파일 commit 후 GPU 큐 시작, 첫 job은 E4 C-rand.
- **D2–D3**: 1차 응답 게시 — E0 spec 답변, E1·E2 결과, supplementary 상태. 일찍 올린 응답이 리뷰어 참여를 끌어낸다.
- **D2–D5**: E4 → E6a → E5 → E6b → E7b → E8 → E9 → E4b/E1b → E11.
- **D5–D7**: 2차 응답과 revised PDF(Sec 3 재작성, 판독 결과 반영).
- discussion 마감 전 최소 3일은 리뷰어 follow-up 대응용으로 비워 둔다. 정확한 마감은 OpenReview에서 확인.

---

## 8. GPU 예산 (H100 기준 추정)

| 실험 | GPU-h | 비고 |
|---|---|---|
| E4 | ~30 | C-rand 먼저 |
| E4b | ~10 | 27–32B, 80GB 이상 GPU |
| E1b | ~5 | 27–32B |
| E5 | ~25 | |
| E6 | ~25 | Fisher 포함 |
| E7 | ~10 | |
| E8 | ~6 | |
| E9 | ~15 | generation 포함 |
| E11 | ~10 | |
| E3c fallback | ~4 | per-item 기록이 없을 때만 |
| **합계** | **~140** | GPU 8장이면 대략 하루 |

---

## 9. 실행 전 체크리스트

- [ ] 이 파일 commit, 해시를 `audit/manifest_r1.yaml`에 기록.
- [ ] α\* 선택 함수가 collateral만 받는지 unit test(bias 값이 들어오면 실패해야 함).
- [ ] 새 run의 trace가 rung마다 (α, validation·evaluation MMLU items, ppl, removal, pre/post skew 부호)를 기록하는지 확인.
- [ ] C-a·C-rand support와 C-b·shuffle map의 seed 재생성 bit-identical 확인(E0 9번).
- [ ] FRONT8 목록 확정 후 E7, E11 시작.
- [ ] calibration 데이터(WikiText-103 train)가 gating chunk와 disjoint인지 확인.
- [ ] BIG4는 기존과 같은 bf16, 8-bit 금지.
