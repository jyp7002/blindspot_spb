"""x2 — broaden the 3B/bridge-tier crossover arc with 3 more families.

Motivation: the crossover-arc pillar (experiments_v5 §X) rests on only 3 bridge
targets (qwen/llama/falcon 3B). The 7B panel cannot grow from the local cache
(only granite-8b is left, and it is collateral-fragile -> all-nan), but the
bridge tier can: gemma-2-2b (2.6B), Phi-3.5-mini (3.8B), granite-3.1-2b (2.5B)
are all cached. Phi in particular supplies the metric-consistent ~4B point that
sits between the existing 3B and 7B arcs.

Same cross-designer protocol / edit config as X (attn r16, both nulls, per-tensor
binary sign edit, strict collateral budget), so x and x2 merge into one arc.
Reuses results/t2x/corpora (qwen_3b/llama_3b/falcon_3b already elicited). Writes
results/x2/removal.jsonl, resumable at (target, axis, seed, designer).

Run (from repo root, models are cached so run offline):
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run_x2_bridge.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import colab_t2t4 as C

# 3B-class designer pool. dnames match the existing X cache scheme (f+"_3b") so
# qwen_3b/llama_3b/falcon_3b corpora are reused, not re-elicited.
POOL = [
    ("qwen_3b",    "Qwen/Qwen2.5-3B-Instruct"),        # cached corpora
    ("llama_3b",   "meta-llama/Llama-3.2-3B-Instruct"),  # cached corpora
    ("falcon_3b",  "tiiuae/Falcon3-3B-Instruct"),        # cached corpora
    ("gemma_3b",   "google/gemma-2-2b-it"),              # new
    ("phi_3b",     "microsoft/Phi-3.5-mini-instruct"),   # new
    ("granite_3b", "ibm-granite/granite-3.1-2b-instruct"),  # new
]
POOL_HF = dict(POOL)

# New target families only (qwen/llama/falcon 3B already covered by panel "x").
# Ordered so the highest-value points (gemma, phi) finish before granite, which
# may be collateral-fragile like granite-8b (results/granite_v3.log).
NEW_TARGETS = [
    ("gemma",   "gemma_3b"),    # 2.6B  -> stitches to gemma-9b at 7B tier
    ("phi",     "phi_3b"),      # 3.8B  -> the missing ~4B arc point
    ("granite", "granite_3b"),  # 2.5B  -> tests whether fragility is size-dependent
]

targets = [(fam, POOL_HF[self_dn]) for fam, self_dn in NEW_TARGETS]
designers = {}
for fam, self_dn in NEW_TARGETS:
    d = [("self", self_dn, POOL_HF[self_dn])]
    d += [("cross", dn, hf) for dn, hf in POOL if dn != self_dn]
    designers[fam] = d

if __name__ == "__main__":
    print(f"[x2] targets={[f for f, _ in NEW_TARGETS]} "
          f"designers/target={len(POOL)} axes=occ_gender,crows_socioeconomic seeds=0,1,2",
          flush=True)
    C.panel_run("x2", targets, designers,
                axes=["occ_gender", "crows_socioeconomic"],
                seeds=[0, 1, 2], batch_size=12, train_bs=8)
    print("[x2] ALL DONE", flush=True)
