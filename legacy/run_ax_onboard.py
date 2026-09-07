"""AX onboarding gates — run BEFORE any removal is trained (experiments_v7 §AX).

§Threats: "AX gates do not transfer: BBQ-unknown pre-bias must be re-measured per
frozen-protocol target before any removal is claimed." An axis that fails these
gates is not a negative result about the method -- it is an axis that cannot
support a removal claim either way, and running removal on it would produce a
number with no interpretation.

Gates, per §AX:
  G1 item floor      >= 40 items in the probe half
  G2 disjoint halves probe and edit halves share no item key
  G3 polarity        pre_skew > 0 on the target, i.e. the model actually prefers
                     the stereotyped continuation over the "unknown" one. A
                     near-zero pre_skew means there is no bias present to remove,
                     which is exactly the §7 confound -- measured, not assumed.
  G4 magnitude       |pre_skew| >= 0.02, a floor below which removal cannot be
                     distinguished from noise given the observed null spread.

Writes results/v6trace/ax_onboard.json. Only axes passing G1-G4 on >=2 targets
proceed to the removal run.
"""
import os, json
import numpy as np
import colab_t2t4 as C

AXES = ["bbq_Age", "bbq_Gender_identity", "bbq_Race_ethnicity", "bbq_Religion"]
TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct"),
           ("phi", "microsoft/Phi-3.5-mini-instruct"),
           ("llama", "meta-llama/Llama-3.2-3B-Instruct")]
ITEM_FLOOR = 40
SKEW_FLOOR = 0.02
OUT = os.path.join(C.RESULTS, "ax_onboard.json")


def structural_gates():
    """G1/G2 depend only on the axis, not the target."""
    rows = {}
    for ax in AXES:
        try:
            probe, edit = C.load_axis(ax)
        except Exception as e:
            rows[ax] = dict(error=str(e)[:120], g1=False, g2=False)
            continue
        kp = {k for _c, _i, k in probe}
        ke = {k for _c, _i, k in edit}
        overlap = kp & ke
        rows[ax] = dict(n_probe=len(probe), n_edit=len(edit),
                        g1_item_floor=len(probe) >= ITEM_FLOOR,
                        g2_disjoint=len(overlap) == 0, n_overlap=len(overlap))
    return rows


def main():
    res = dict(item_floor=ITEM_FLOOR, skew_floor=SKEW_FLOOR, axes=structural_gates())
    print("STRUCTURAL GATES (axis-level)")
    print(f"  {'axis':24s} {'probe':>6s} {'edit':>6s} {'G1 items':>9s} {'G2 disjoint':>12s}")
    for ax, r in res["axes"].items():
        if "error" in r:
            print(f"  {ax:24s} ERROR {r['error']}"); continue
        print(f"  {ax:24s} {r['n_probe']:6d} {r['n_edit']:6d} "
              f"{str(r['g1_item_floor']):>9s} {str(r['g2_disjoint']):>12s}")

    live = [ax for ax, r in res["axes"].items()
            if r.get("g1_item_floor") and r.get("g2_disjoint")]
    print(f"\nPOLARITY + MAGNITUDE GATES (per target) on {live}")
    res["pre_skew"] = {}
    for fam, hf in TARGETS:
        model, tok = C.load_model(hf, dispatch=False)
        try:
            for ax in live:
                sk = C.axis_skew(model, tok, ax, 6)
                g3 = sk > 0
                g4 = abs(sk) >= SKEW_FLOOR
                res["pre_skew"][f"{fam}|{ax}"] = dict(pre_skew=float(sk),
                                                      g3_polarity=bool(g3),
                                                      g4_magnitude=bool(g4))
                print(f"  {fam:6s} {ax:24s} pre_skew={sk:+.4f}  "
                      f"G3={'PASS' if g3 else 'FAIL'}  G4={'PASS' if g4 else 'FAIL'}")
        finally:
            C._free(model)

    # an axis is usable if it passes all four gates on >= 2 targets
    usable = []
    for ax in live:
        n = sum(1 for k, v in res["pre_skew"].items()
                if k.endswith("|" + ax) and v["g3_polarity"] and v["g4_magnitude"])
        if n >= 2:
            usable.append(ax)
        print(f"  -> {ax:24s} passes on {n}/{len(TARGETS)} targets"
              f"  {'USABLE' if n >= 2 else 'REJECTED (no removal claim)'}")
    res["usable_axes"] = usable
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nusable axes: {usable}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
