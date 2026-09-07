"""MV-B nulls — sign-shuffle and data-partition (experiments_v6 §1).

§1 requires BOTH nulls in every new condition; the mvb panel shipped without
them, so no mvb number is citable yet. Scoped to the two cells that carry the
penalties MV-B was built to explain — phi|occ_gender (+0.288) and
qwen|crows_socioeconomic (+0.227) — rather than all six, because a null is only
informative where there is a real effect to null out.

sign_shuffle: same scale/sparsity, randomised signs. Removal that survives means
              the result is not carried by sign STRUCTURE.
partition:    both task vectors trained on halves of the SAME biased corpus, so
              there is no biased-vs-debiased direction. Removal that survives
              means the result is not carried by that contrast.
"""
import sys, os
import colab_t2t4 as C
from run_mvb_transplant import SMALL, LARGE, FAM

CELLS = {"phi": "occ_gender", "qwen": "crows_socioeconomic"}
null = sys.argv[1]
assert null in ("sign_shuffle", "partition"), null

targets, designers = [], {}
for fam, ax in CELLS.items():
    self_dn = {"phi": "phi_3b", "qwen": "qwen_3b"}[fam]
    targets.append((fam, SMALL[self_dn]))
    d = [("self", self_dn, SMALL[self_dn])]
    for dn, hf in LARGE.items():
        d.append(("same_large" if FAM[dn] == fam else "cross_large", dn, hf))
    designers[fam] = d

if __name__ == "__main__":
    print(f"[null:{null}] cells={CELLS} designers/target={len(designers['phi'])}", flush=True)
    for fam, ax in CELLS.items():
        C.panel_run(f"mvb_null_{null}", [(fam, dict(targets)[fam])], {fam: designers[fam]},
                    axes=[ax], seeds=[0, 1, 2], batch_size=6, train_bs=8, null=null)
    print(f"[null:{null}] ALL DONE", flush=True)
