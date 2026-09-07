"""Pilot: does the generative self-diagnosis contrast actually exist?

This is the mechanism check that gates the whole 2x2 (design.md §4.1, §11:
"Run the 2x2 and the endogenous/exogenous ablation first -- these two decide
whether the paper exists").

For each designer D and each axis we measure:

  congruence(biased)   -- how stereotyped D's requested-biased corpus is
  congruence(debiased) -- how stereotyped D's requested-DEBIASED corpus is
  contrast_gap         -- the difference; this is the usable debias signal

H2's mechanism predicts the gap is SMALL on an axis D itself carries (D
cannot write genuinely unbiased text about its own blind spot) and LARGE on
an axis D does not carry. If the gap is large everywhere, the endogenous
signal is fine everywhere and there is no blind spot to find; if it is small
everywhere, the elicitation is broken rather than the designer.

Refusal rates are reported per condition: a designer that refuses the biased
prompt produces a degenerate corpus, which would masquerade as a blind spot.
"""
import os, json
import numpy as np
from common import load, free, FAMILY, save_json
import probes
from elicit_gen import elicit_occ, elicit_valence
from select_acquired import EXTRA_AXES

DESIGNERS = ["qwen1.5b", "qwen3b", "llama3b", "gemma2b"]
VAL_AXES = ["gen_fm", "ses_lw", "nat_ro_dk", "wt_os"]


def main():
    probes.CANDIDATE_AXES.update(EXTRA_AXES)
    out = {}
    for name in DESIGNERS:
        model, tok = load(name)
        rec = {}
        try:
            r = elicit_occ(model, tok, seed=0, batch_size=16)
            rec["occ_gender"] = r["diag"]
            print(f"\n### {name} ({FAMILY[name]}) --- occ_gender [INHERITED]", flush=True)
            print(f"   biased  : congruence={r['diag']['biased']['congruence']:.3f} "
                  f"refusal={r['diag']['biased']['refusal_rate']:.3f} "
                  f"n={r['diag']['biased']['n_usable']}")
            print(f"   debiased: congruence={r['diag']['debiased']['congruence']:.3f} "
                  f"refusal={r['diag']['debiased']['refusal_rate']:.3f} "
                  f"n={r['diag']['debiased']['n_usable']}")
            print(f"   CONTRAST GAP = {r['diag']['contrast_gap']:+.3f}")
            for s in r["biased"][:2]:
                print("     [B]", s[:100])
            for s in r["debiased"][:2]:
                print("     [D]", s[:100])

            for ax in VAL_AXES:
                v = elicit_valence(model, tok, ax, seed=0, n_rep=12, batch_size=16)
                rec[ax] = v["diag"]
                print(f"   {ax:9s} {str(probes.CANDIDATE_AXES[ax]):24s} "
                      f"gap={v['diag']['contrast_gap']:+.3f} "
                      f"(b_skew={v['diag']['biased']['skew']:+.2f} "
                      f"d_skew={v['diag']['debiased']['skew']:+.2f}) "
                      f"refusal_b={v['diag']['biased']['refusal_rate']:.2f}", flush=True)
                if ax == VAL_AXES[0]:
                    for s in v["biased"][:2]:
                        print("     [B]", s[:100])
        finally:
            free(model, tok)
        out[name] = rec
    save_json(out, os.path.join(os.path.dirname(__file__), "..", "results",
                                "pilot_elicitation.json"))
    print("\nwrote results/pilot_elicitation.json")


if __name__ == "__main__":
    main()
