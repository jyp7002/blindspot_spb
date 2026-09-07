"""Multi-axis generality for the method paper (design.md §5.3).

The frozen protocol has only occ_gender + crows_socioeconomic. The paper promises
the method works across bias types; the other axes exist only on the untagged
v2-era module set, mostly for qwen1.5b. This runs 4 additional CrowS axes on the
frozen attn protocol via panel_run, which elicits any missing designer corpora
itself (inference-only) before training.

Single variant (binary per_tensor s0.0, the headline configuration) — this tests
GENERALITY across axes, not the variant space, which run_ablate_attn covers.
"""
import colab_t2t4 as C

TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct"), ("phi", "microsoft/Phi-3.5-mini-instruct")]
DESIGNERS = {
    "qwen": [("self", "qwen_3b", "Qwen/Qwen2.5-3B-Instruct"),
             ("cross", "llama_3b", "meta-llama/Llama-3.2-3B-Instruct")],
    "phi":  [("self", "phi_3b", "microsoft/Phi-3.5-mini-instruct"),
             ("cross", "qwen_3b", "Qwen/Qwen2.5-3B-Instruct")],
}
AXES = ["crows_race-color", "crows_gender", "crows_religion", "crows_age"]

if __name__ == "__main__":
    print(f"[axes] targets={[t for t,_ in TARGETS]} axes={AXES} "
          f"cells={len(TARGETS)*2*len(AXES)*3}", flush=True)
    C.panel_run("axes", TARGETS, DESIGNERS, axes=AXES, seeds=[0, 1, 2],
                batch_size=6, train_bs=8)
    print("[axes] ALL DONE", flush=True)
