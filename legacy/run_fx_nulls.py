"""Both nulls for the FX cells where the adaptive rule changed the source.

The FX evaluation leans on mvb cells gemma|occ (+0.168), gemma|crows (-0.023) and
qwen|occ (+0.204). Nulls already exist for qwen|crows and phi|occ; these are the
rest. §1 requires both nulls in every new condition, so without them the FX
improvement is not citable.
"""
import sys
import colab_t2t4 as C

null = sys.argv[1]
assert null in ("sign_shuffle", "partition"), null
CELLS = {"gemma": ["occ_gender", "crows_socioeconomic"], "qwen": ["occ_gender"]}
HF = {"gemma": "google/gemma-2-2b-it", "qwen": "Qwen/Qwen2.5-3B-Instruct"}
SELF = {"gemma": ("gemma_3b", "google/gemma-2-2b-it"),
        "qwen": ("qwen_3b", "Qwen/Qwen2.5-3B-Instruct")}
LARGE = {"gemma": ("gemma", "google/gemma-2-9b-it"),
         "qwen": ("qwen", "Qwen/Qwen2.5-7B-Instruct")}

if __name__ == "__main__":
    for fam, axes in CELLS.items():
        d = {fam: [("self",) + SELF[fam], ("same_large",) + LARGE[fam]]}
        print(f"[fxnull:{null}] {fam} axes={axes}", flush=True)
        C.panel_run(f"fxnull_{null}", [(fam, HF[fam])], d,
                    axes=axes, seeds=[0, 1, 2], batch_size=6, train_bs=8, null=null)
    print(f"[fxnull:{null}] ALL DONE", flush=True)
