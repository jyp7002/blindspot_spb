"""MV-B forward corpus transplant (experiments_v6 §2 MV-B).

Hypothesis (ingredient = elicitation/corpus quality): small models produce
degraded self-elicited corpora; by 7-9B corpus quality saturates, so everyone
ties. The corpus is the transplantable object: edits are trained ON the small
target FROM a source designer's corpus, with the elicitation recipe held fixed.

Why this probe now: MV-A and MV-C have both failed their primary criteria
(MV-A1 fails on phi-occ, whose penalty is budget-INDEPENDENT: +0.288 strict ->
+0.320 unconstrained with zero budget-fails; MV-C1's same-vs-cross residual
correlation covers 0). MV-B is the last of the three, and it carries the
strongest prior: phi_3b's elicitation contrast_gap is +0.012, the most degraded
in the registry (results/v6/contrast_gap_frozen.json), against a +0.288 penalty.

Design (frozen before running):
  targets   phi (3.8B, the clean penalty anchor and DEMO-1 primary)
            qwen_3b (3.1B, carries the +0.227 crows penalty)
            gemma (2.6B, self-ADVANTAGES at strict budget -- the negative control:
                   if corpus quality drives the penalty, a large-source corpus
                   should NOT help a target that has no penalty to close)
  sources   self          the target's own 3B corpus (the baseline penalty)
            same_large    same-family 7-9B/14B corpus  [qwen, gemma only]
            cross_large   every other family's 7-9B corpus
            cross_small   3B-class corpora from other families (already in x/x2,
                          re-run here so all arms share one panel and protocol)
  axes      occ_gender, crows_socioeconomic       seeds 0,1,2

STRUCTURAL LIMITATION, recorded rather than worked around: **phi has no larger
sibling in the registry**, so MV-B1 ("penalty(small target <- same-family-large
corpus) CI covers 0") is NOT testable on the primary target. For phi only the
quality-vs-family split (MV-B3) is available: if a CROSS-large corpus closes
phi's penalty, the ingredient is corpus quality; if it does not, quality is
family-conditioned. MV-B1 proper is testable on qwen_3b and gemma.

NOT INCLUDED, and this is a real gap vs §1's frozen inputs: the two nulls
(sign-shuffle, data-partition). panel_run does not emit them, and neither did
the x/x2 panels this compares against. Any MV-B result must be reported as
lacking its nulls until they are run.

Serial only: 31 GB host RAM (see run_v6_queue.sh header).
"""
import os
import colab_t2t4 as C

# ---- designer pool: dnames MUST match the cached corpora scheme in
# results/t2x/corpora/<dname>|<axis>|s<seed>.json (all present as of 2026-07-29).
SMALL = {
    "qwen_3b":    "Qwen/Qwen2.5-3B-Instruct",
    "llama_3b":   "meta-llama/Llama-3.2-3B-Instruct",
    "falcon_3b":  "tiiuae/Falcon3-3B-Instruct",
    "gemma_3b":   "google/gemma-2-2b-it",
    "phi_3b":     "microsoft/Phi-3.5-mini-instruct",
    "granite_3b": "ibm-granite/granite-3.1-2b-instruct",
}
LARGE = {
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "gemma":   "google/gemma-2-9b-it",
    "olmo":    "allenai/OLMo-2-1124-7B-Instruct",
    "granite": "ibm-granite/granite-3.1-8b-instruct",
}
# family of each designer, for classifying a source as same- vs cross-family
FAM = {"qwen_3b": "qwen", "llama_3b": "llama", "falcon_3b": "falcon",
       "gemma_3b": "gemma", "phi_3b": "phi", "granite_3b": "granite",
       "qwen": "qwen", "llama": "llama", "gemma": "gemma",
       "olmo": "olmo", "granite": "granite"}

TARGETS = [                      # (family, self designer name)
    ("phi",     "phi_3b"),       # primary: clean +0.288 penalty, no large sibling
    ("qwen",    "qwen_3b"),      # MV-B1 testable (qwen 7B is the same-family large)
    ("gemma",   "gemma_3b"),     # negative control: self-advantages at strict budget
]

targets = [(fam, SMALL[self_dn]) for fam, self_dn in TARGETS]
designers = {}
for fam, self_dn in TARGETS:
    d = [("self", self_dn, SMALL[self_dn])]
    for dn, hf in LARGE.items():
        role = "same_large" if FAM[dn] == fam else "cross_large"
        d.append((role, dn, hf))
    for dn, hf in SMALL.items():
        if dn != self_dn:
            d.append(("cross_small", dn, hf))
    designers[fam] = d

if __name__ == "__main__":
    miss = []
    for fam, _ in TARGETS:
        for _role, dn, _hf in designers[fam]:
            for ax in ("occ_gender", "crows_socioeconomic"):
                for s in (0, 1, 2):
                    if not os.path.exists(C._t2x_corpus_fp(dn, ax, s)):
                        miss.append((dn, ax, s))
    if miss:
        # a missing corpus would silently trigger re-elicitation of a 7-9B
        # designer -- the exact cache-miss failure that corrupted x2 in v5.
        raise SystemExit(f"ABORT: {len(miss)} source corpora missing, e.g. {miss[:5]}")
    n = sum(len(designers[f]) for f, _ in TARGETS) * 2 * 3
    print(f"[mvb] targets={[f for f, _ in TARGETS]} cells={n} "
          f"(all source corpora present; nothing will be re-elicited)", flush=True)
    for fam, _ in TARGETS:
        roles = {}
        for role, dn, _ in designers[fam]:
            roles.setdefault(role, []).append(dn)
        print(f"[mvb]   {fam}: " + "  ".join(f"{r}={v}" for r, v in roles.items()), flush=True)
    C.panel_run("mvb", targets, designers,
                axes=["occ_gender", "crows_socioeconomic"],
                seeds=[0, 1, 2], batch_size=6, train_bs=8)
    print("[mvb] ALL DONE", flush=True)
