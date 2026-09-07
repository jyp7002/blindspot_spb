"""M validation gate — build M corpora, then verify the transform is CLEAN
before the expensive designer sweep (experiments_v3 M, redesigned).

A valid structural manipulation must:
  * PRESERVE bias magnitude (else removability is confounded by bias-loss, the
    v1 M3 bug), and
  * ACHIEVE the intended frame-overlap change (else it isn't a structural
    manipulation, the v1 M1 bug).

Gates (measured on qwen1.5b, the M target):
  templatize CrowS  : pre-bias >= 0.05  AND frame_overlap >= 0.45
  de-templatize occ : pre-bias >= 0.30  AND frame_overlap <= 0.15

Writes accepted corpora to results/m_corpora/ and a report to
results/m_validation.json. Only accepted axes are handed to the designer sweep.
"""
import os, json, argparse
import numpy as np
from common import load, free
from frame_overlap import frame_overlap
import m_axes as MA

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
MCORP = os.path.join(RESULTS, "m_corpora")

GATES = {
    "templatize": dict(min_prebias=0.05, min_frame=0.45),
    "detemplatize": dict(min_prebias=0.30, max_frame=0.15),
}


def _fo(corp):
    return frame_overlap([c for c, _i, _k in corp["probe"]] + [i for _c, i, _k in corp["probe"]],
                         [c for c, _i, _k in corp["edit"]] + [i for _c, i, _k in corp["edit"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templatize", nargs="*",
                    default=["crows_socioeconomic", "crows_religion", "crows_age"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    os.makedirs(MCORP, exist_ok=True)

    from m_rewrite import Rewriter, templatize_crows, detemplatize_occ
    rw = Rewriter()
    print(f"rewriter backend: {rw.backend}", flush=True)

    built = {}
    for ax in a.templatize:
        tag = f"mt_{ax}"
        path = os.path.join(MCORP, f"{tag}.json")
        if a.rebuild or not os.path.exists(path):
            corp = templatize_crows(ax, rw)
            json.dump(corp, open(path, "w"))
        else:
            corp = json.load(open(path))
        built[tag] = ("templatize", corp)
        print(f"[build] {tag}: probe={len(corp['probe'])} edit={len(corp['edit'])} "
              f"frame_ov={_fo(corp):.3f}", flush=True)

    tag = "md_occ_gender"
    path = os.path.join(MCORP, f"{tag}.json")
    if a.rebuild or not os.path.exists(path):
        corp = detemplatize_occ(rw)
        json.dump(corp, open(path, "w"))
    else:
        corp = json.load(open(path))
    built[tag] = ("detemplatize", corp)
    print(f"[build] {tag}: probe={len(corp['probe'])} edit={len(corp['edit'])} "
          f"frame_ov={_fo(corp):.3f}", flush=True)

    # measure pre-bias on the target model
    model, tok = load("qwen1.5b")
    report = {}
    try:
        for tag, (kind, corp) in built.items():
            fo = _fo(corp)
            sc = MA.m_score(model, tok, tag, batch_size=24)
            g = GATES[kind]
            ok_bias = abs(sc["m_skew"]) >= g["min_prebias"]
            ok_frame = (fo >= g["min_frame"]) if kind == "templatize" else (fo <= g["max_frame"])
            accepted = ok_bias and ok_frame and len(corp["probe"]) >= 40
            report[tag] = dict(kind=kind, pre_bias=abs(sc["m_skew"]),
                               frame_overlap=fo, n_probe=len(corp["probe"]),
                               bias_ok=ok_bias, frame_ok=ok_frame, accepted=accepted)
            print(f"[gate] {tag:26s} pre_bias={abs(sc['m_skew']):.3f} "
                  f"frame_ov={fo:.3f}  bias_ok={ok_bias} frame_ok={ok_frame} "
                  f"-> {'ACCEPT' if accepted else 'REJECT'}", flush=True)
    finally:
        free(model, tok)
        model = tok = None

    json.dump(report, open(os.path.join(RESULTS, "m_validation.json"), "w"), indent=2)
    acc = [t for t, r in report.items() if r["accepted"]]
    print(f"\nACCEPTED for designer sweep: {acc}")
    with open(os.path.join(MCORP, "accepted.txt"), "w") as f:
        f.write("\n".join(acc))


if __name__ == "__main__":
    main()
