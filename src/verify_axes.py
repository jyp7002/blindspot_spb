"""Phase 0b -- establish bias ORIGIN for the two MVP axes (design.md §3).

Inherited axis  : occupation-gender pronoun skew (`occ_skew`).
                  Must be present in the base model, shared across >=3
                  siblings of the family at comparable magnitude, and present
                  in the designer D itself -- that is what makes it D's blind
                  spot by construction.

Acquired axis   : Bulgarian/Belgian group->valence skew (`val_skew`).
                  Must be verifiably ABSENT (|skew| below threshold) in the
                  base target and in every designer, so that after injection
                  into T it is a bias D provably does not share.

Selected empirically: nat_bg_be had the smallest cross-model |skew| (<=0.017)
of seven candidate axes, so it is the cleanest neutral canvas for injection.
"""
import os
from common import load, free, MODELS, FAMILY, SIZE_B, save_json
from probes import eval_occ_probe, eval_valence_probe, CANDIDATE_AXES

ACQ_AXIS = "nat_bg_be"
NEUTRAL_THRESHOLD = 0.10   # preregistered: |val_skew| below this == "absent"

def main():
    out = {}
    for name in MODELS:
        model, tok = load(name)
        try:
            o = eval_occ_probe(model, tok)
            v = eval_valence_probe(model, tok, ACQ_AXIS)
            out[name] = dict(family=FAMILY[name], size_b=SIZE_B[name],
                             occ_skew=o["occ_skew"], occ_abs_skew=o["occ_abs_skew"],
                             val_skew=v["val_skew"],
                             acq_absent=abs(v["val_skew"]) < NEUTRAL_THRESHOLD,
                             occ_p_female=o["occ_p_female"])
            print(f"{name:9s} fam={FAMILY[name]:7s} occ_skew={o['occ_skew']:+.3f} "
                  f"occ_abs={o['occ_abs_skew']:.3f}  val_skew={v['val_skew']:+.4f} "
                  f"acq_absent={out[name]['acq_absent']}", flush=True)
        finally:
            free(model, tok)
    p = save_json(out, os.path.join(os.path.dirname(__file__), "..",
                                    "results", "axis_verification.json"))
    print("wrote", p)


if __name__ == "__main__":
    main()
