# Related Work — sparse and signed weight deltas (DARE / TIES / BitDelta)

**Status:** drafted 2026-08-28, **before** DEC results existed, per experiments_v8
§Manuscript track ("Related Work gains the DARE/TIES/BitDelta paragraph now,
before DEC lands — it is needed under both outcomes"). Paste target:
`binary_debiaser_draft.md` §Related Work. The final sentence is the only part
that is outcome-contingent; both versions are written below and the wrong one is
deleted once DEC reports.

---

## The paragraph

Three lines of work establish that fine-tuning deltas are compressible far
beyond what their parameter count suggests, and each sits next to one of our
observations. **BitDelta** (Liu et al., 2024) shows that the delta between a base
and a fine-tuned model can be quantised to one bit per parameter with a single
per-tensor scale at little cost, and uses this for multi-tenant serving; our edit
is that construction applied to a *behavioural contrast* rather than a
fine-tuning delta, and then negated — which is why our sign-only result should
not by itself be read as novel. **DARE** (Yu et al., 2024) shows that dropping
90–99% of delta parameters *at random* and rescaling the survivors by 1/(1−p)
leaves task performance largely intact, i.e. that the delta field is massively
redundant and no particular coordinate is privileged. **TIES-Merging** (Yadav et
al., 2023) instead trims to the top-k by magnitude and resolves sign conflicts by
election, treating magnitude as a guide to which coordinates carry the update and
sign agreement as the thing worth preserving across tasks.

Our setting differs from all three in what is being edited and what is being
protected. These methods aim to *preserve a capability* (task accuracy, chat
quality) while shrinking or merging a delta; we aim to *remove a behaviour*
(measurable social bias) while holding capability inside an explicit,
always-on collateral budget (ΔMMLU ≥ −0.02, perplexity ratio ≤ 1.10) that is
enforced during the bias probe, the MMLU evaluation and the perplexity
evaluation alike. That difference matters because a bias edit has no
"performance" of its own to preserve — its success is defined against a budget
rather than against a retained score — so the compressibility results above do
not transfer automatically. It also lets us contribute the **necessity** side
these papers do not test: randomising the signs at identical scale, sparsity and
support collapses removal from +0.206 to +0.008 (paired Δ = +0.198 [+0.175,
+0.222], n = 502), so the effect is carried by the learned sign field and not by
the perturbation's magnitude or its energy.

The open question, and the one our decomposition experiment (§DEC) is designed
to answer, is whether the *support* carries information independently of the
signs. DARE's success under random dropping predicts that it does not: any 1% of
the field should do, with magnitude serving only to certify which coordinates are
safe to keep. TIES's reliance on magnitude ranking predicts that it does. We test
this directly by holding density, per-tensor scale, and the α-selection space
fixed and varying only where the edit is allowed to act — a comparison that is
only meaningful because the scale convention is matched across conditions
(§DEC construction; an unmatched convention hands the random-support arm a
3.5× smaller edit norm and decides the question by fiat).

<!-- OUTCOME A (Δ_selection CI covers 0): keep this sentence -->
> We find that a random 1% of the sign field performs indistinguishably from the
> magnitude-selected 1%, extending the DARE/BitDelta/TIES picture from task
> performance to behavioural control: the sign field is distributed, magnitude
> serves only to certify coordinates, and the deployment consequence is a
> sub-megabyte patch that can be sited anywhere in the attention weights.

<!-- OUTCOME B (Δ_selection CI excludes 0): keep this sentence instead -->
> We find that the magnitude-selected 1% substantially outperforms a random 1%
> at matched norm, which distinguishes behavioural edits from the
> task-performance deltas studied above: unlike capability deltas under random
> drop, the behaviour we remove is concentrated in a magnitude-identifiable
> sparse signed substructure rather than spread uniformly across the delta field.

---

## References to add to the bibliography

- Liu, Wang, Dao, Zhou, et al. **BitDelta: Your Fine-Tune May Only Be Worth One
  Bit.** NeurIPS 2024.
- Yu, Yu, Yu, Huang, Li. **Language Models are Super Mario: Absorbing Abilities
  from Homologous Models as a Free Lunch (DARE).** ICML 2024.
- Yadav, Tam, Choshen, Raffel, Bansal. **TIES-Merging: Resolving Interference
  When Merging Models.** NeurIPS 2023.

## Positioning notes (not for the paper — for the author)

1. **Do not claim the sign-only edit as novel construction.** BitDelta has it.
   What is ours is (a) applying it to an elicited behavioural contrast, (b) the
   necessity result, (c) the always-on budget, (d) `contrast_gap` as a
   pre-training corpus-selection rule, and (e) whatever DEC returns.
2. **The always-on budget is the honest differentiator** and should be stated as
   exceeding field norm, since §11 already shows three methods tie on removal.
   The artifact — mergeable, sub-MB, no inference-time cost — is the claim.
3. **Under Outcome A the paper is not weakened.** "Any 1% works" is a stronger
   *deployment* story (the patch is portable and the support need not be
   searched for), and the necessity result still stands on its own.
