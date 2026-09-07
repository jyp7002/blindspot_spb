"""M validation gate (redesigned) — experiments_v3 M.

The OUTCOME that tests the structural hypothesis is the exogenous ceiling
(removability), measured by the designer sweep. This gate only ensures a
transform is INTERPRETABLE before spending GPU on that sweep:

  * bias PRESERVED (else removability is confounded by bias-loss -- the v1 M3
    bug), and
  * minimal-pair integrity (members differ only in the target slot), and
  * enough items.

frame_overlap is REPORTED, not gated: a surface n-gram metric mis-ranks the
original occ_gender (overlap 0.10 yet highly removable), so it cannot be a
hard gate. The clean test is the templatize direction: if making CrowS
valence-like while PRESERVING the group->stereotype content raises the ceiling,
structure (not content) governs removability.
"""
import os, json, re, argparse
import numpy as np
from common import load, free
from frame_overlap import frame_overlap
import m_axes as MA

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
MCORP = os.path.join(RESULTS, "m_corpora")

# bias-preservation + item floors (frame_overlap reported, not gated)
GATES = {"templatize": dict(min_prebias=0.04, min_items=35),
         "detemplatize": dict(min_prebias=0.25, min_items=35)}


def _fo(corp):
    return frame_overlap([c for c, _i, _k in corp["probe"]] + [i for _c, i, _k in corp["probe"]],
                         [c for c, _i, _k in corp["edit"]] + [i for _c, i, _k in corp["edit"]])


def _minpair_ok(corp, kind):
    """Fraction of pairs that are valid minimal pairs (differ only in slot)."""
    ok = 0
    pairs = corp["probe"] + corp["edit"]
    for c, i, _k in pairs:
        if kind == "detemplatize":
            cm = re.sub(r"\b(he|she|his|her)\b", "X", c, flags=re.I)
            im = re.sub(r"\b(he|she|his|her)\b", "X", i, flags=re.I)
            ok += (cm == im)
        else:
            # templatize: identical except the group slot (before " is "/frame)
            ok += (c.split(" is ")[-1] == i.split(" is ")[-1]) if " is " in c else (c != i)
    return ok / max(len(pairs), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templatize", nargs="*",
                    default=["crows_socioeconomic", "crows_religion", "crows_age"])
    a = ap.parse_args()

    built = {}
    for ax in a.templatize:
        tag = f"mt_{ax}"
        pth = os.path.join(MCORP, f"{tag}.json")
        if os.path.exists(pth):
            built[tag] = ("templatize", json.load(open(pth)))
    md = os.path.join(MCORP, "md_occ_gender.json")
    if os.path.exists(md):
        built["md_occ_gender"] = ("detemplatize", json.load(open(md)))

    model, tok = load("qwen1.5b")
    report = {}
    try:
        for tag, (kind, corp) in built.items():
            fo = _fo(corp)
            mp = _minpair_ok(corp, kind)
            sc = MA.m_score(model, tok, tag, batch_size=24)
            g = GATES[kind]
            bias_ok = abs(sc["m_skew"]) >= g["min_prebias"]
            items_ok = len(corp["probe"]) >= g["min_items"]
            pair_ok = mp >= 0.9
            accepted = bias_ok and items_ok and pair_ok
            report[tag] = dict(kind=kind, pre_bias=abs(sc["m_skew"]),
                               frame_overlap=fo, minpair_frac=mp,
                               n_probe=len(corp["probe"]),
                               bias_ok=bias_ok, items_ok=items_ok,
                               pair_ok=pair_ok, accepted=accepted)
            print(f"[gate] {tag:26s} pre_bias={abs(sc['m_skew']):.3f} "
                  f"minpair={mp:.2f} frame_ov={fo:.3f} n={len(corp['probe'])} "
                  f"-> {'ACCEPT' if accepted else 'REJECT'}", flush=True)
    finally:
        free(model, tok); model = tok = None

    json.dump(report, open(os.path.join(RESULTS, "m_validation.json"), "w"), indent=2)
    acc = [t for t, r in report.items() if r["accepted"]]
    open(os.path.join(MCORP, "accepted.txt"), "w").write("\n".join(acc))
    print(f"\nACCEPTED for designer sweep: {acc}")


if __name__ == "__main__":
    main()
