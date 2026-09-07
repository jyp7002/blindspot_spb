"""Pilot the forced-choice self-diagnosis across designers and both axes.

This is the pre-edit mechanism check (design.md §4.1, §11). For each designer
D and axis we get:

  congruence(biased)   -- can D identify/produce the stereotyped version?
  congruence(debiased) -- when asked to be FAIR, does D still land on it?
  contrast_gap         -- the usable debias signal = biased - debiased

The blind-spot prediction is a small gap on an axis D itself carries and a
large gap on one it does not, so this table forecasts the 2x2 before a single
weight is edited.
"""
import os
import numpy as np
from common import load, free, FAMILY, save_json
from elicit_gen import elicit_forced, elicit_selfdebias

DESIGNERS = ["qwen1.5b", "qwen3b", "llama3b", "gemma2b", "phi3.5", "smol1.7b"]
AXES = [("occ_gender", "INHERITED"), ("gen_fm", "ACQUIRED")]


def main():
    out = {}
    print(f"{'designer':10s} {'family':8s} {'axis':11s} {'origin':10s} "
          f"{'biased':>8s} {'debiased':>9s} {'GAP':>8s}")
    print("-" * 70)
    for name in DESIGNERS:
        model, tok = load(name)
        out[name] = {}
        try:
            for axis, origin in AXES:
                r = elicit_selfdebias(model, tok, axis, seed=0, batch_size=32)
                d = r["diag"]
                out[name][axis] = d
                print(f"{name:10s} {FAMILY[name]:8s} {axis:11s} {origin:10s} "
                      f"{d['biased']['congruence']:8.3f} "
                      f"{d['debiased']['congruence']:9.3f} "
                      f"{d['contrast_gap']:+8.3f}", flush=True)
        finally:
            free(model, tok)
    save_json(out, os.path.join(os.path.dirname(__file__), "..", "results",
                                "pilot_selfdebias.json"))
    print("\nwrote results/pilot_forced.json")


if __name__ == "__main__":
    main()
