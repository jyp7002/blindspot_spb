"""Granite discriminating test: does argmax-GAP beat argmax-SIZE?

On all 12 FX-evaluable cells the two selectors chose identically, so the data
cannot yet tell the adaptive rule apart from "use the biggest family member".
Granite is the ONLY family in the registry where they disagree:

    occ_gender    granite_3b (2.5B) gap +0.405   vs  granite (8B) gap +0.398
    crows_socio   granite_3b (2.5B) gap +0.085   vs  granite (8B) gap +0.035

so gap picks the SMALLER model on both axes while size picks the larger. Running
both granite checkpoints as targets gives the head-to-head.

CAVEAT, stated before the run: granite is the diagnosed-fragile instrument
(PREREGISTRATION V3, "granite T2 all-nan: DIAGNOSED, excluded with reason";
6/8 nan in the x2 bridge). A budget-fail floor could make this uninformative,
which is itself worth knowing before more compute is spent on the FX line.
"""
import colab_t2t4 as C

G8 = "ibm-granite/granite-3.1-8b-instruct"
G2 = "ibm-granite/granite-3.1-2b-instruct"
targets = [("granite8", G8), ("granite2", G2)]
designers = {
    # for each target: its own corpus (self) and its family sibling's
    "granite8": [("self", "granite", G8), ("sibling", "granite_3b", G2)],
    "granite2": [("self", "granite_3b", G2), ("sibling", "granite", G8)],
}
if __name__ == "__main__":
    print(f"[fxg] targets={[t for t,_ in targets]} 2 sources each, 2 axes, 3 seeds = 24 cells",
          flush=True)
    C.panel_run("fxg", targets, designers,
                axes=["occ_gender", "crows_socioeconomic"],
                seeds=[0, 1, 2], batch_size=6, train_bs=8)
    print("[fxg] ALL DONE", flush=True)
