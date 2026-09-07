"""Select the acquired-bias axis: low EXHIBITION, high RECOGNITION.

This is the feasibility gate for the 2x2 of design.md §5.1, and it enforces
property (4) of probes.py: the acquired bias must be one the designer does
not exhibit but *can see*.

A debias signal elicited from D can only target a bias D is able to
recognise. An arbitrary or private association injected into T would be
invisible to every designer -- same-family and cross-family alike -- so the
acquired row of the 2x2 would fail for everyone and the interaction would be
undefined. The acquired axis therefore has to be a genuine, widely-known
stereotype that alignment training has stopped models from *exhibiting* while
leaving their ability to *recognise* it intact.

Two quantities per candidate axis, measured on every model:

  exhibition  = `val_skew` -- does the model itself associate group A with
                negative valence? Want ~0 (verifiably absent, design.md §3).
  recognition = agreement of the model's own stereotype judgements with
                ground truth on a minimal-pair pool. Want >> 0.5.

The axis that maximises (recognition - |exhibition|) across all designers is
selected.
"""
import os, json
import numpy as np
from common import load, free, MODELS, FAMILY, save_json
import probes
from probes import eval_valence_probe, CANDIDATE_AXES
from elicit import build_pool_valence, judge_pool

# Candidate pairs: group A carries a widely-known negative stereotype in
# English-language text; group B is a comparison group of similar
# "nation-ness" without that specific association.
EXTRA_AXES = {k: probes.CANDIDATE_AXES[k] for k in
              ["nat_mx_ca", "nat_ng_no", "nat_ro_dk", "nat_pk_nl", "nat_bg_be",
               "gen_fm", "age_oy", "wt_os", "ses_lw"]}
# All group tokens must read naturally as a prenominal adjective, since the
# probe/pool templates are of the form "My {G} colleague ... {a}".
MODELS_TO_TEST = ["qwen1.5b", "qwen3b", "llama3b", "gemma2b", "phi3.5", "smol1.7b"]


def main():
    probes.CANDIDATE_AXES.update(EXTRA_AXES)
    res = {}
    for name in MODELS_TO_TEST:
        model, tok = load(name)
        res[name] = {}
        try:
            for ax in EXTRA_AXES:
                v = eval_valence_probe(model, tok, ax)
                pool = build_pool_valence(ax, seed=0)
                j = judge_pool(model, tok, pool, batch_size=32)
                res[name][ax] = dict(exhibition=v["val_skew"],
                                     recognition=j["recognition"],
                                     raw_agreement=j["agreement"],
                                     margin=j["mean_margin"])
                print(f"{name:9s} {ax:10s} {str(EXTRA_AXES[ax]):26s} "
                      f"exhibition={v['val_skew']:+.4f}  recognition={j['recognition']:+.3f} "
                      f"margin={j['mean_margin']:.3f}", flush=True)
        finally:
            free(model, tok)

    print("\n=== summary across models ===")
    rows = []
    for ax in EXTRA_AXES:
        ex = [abs(res[m][ax]["exhibition"]) for m in res]
        rc = [res[m][ax]["recognition"] for m in res]
        score = float(np.mean(rc)) - float(np.max(ex))
        rows.append((score, ax, np.mean(rc), np.min(rc), np.max(ex)))
        print(f"  {ax:10s} {str(EXTRA_AXES[ax]):26s} mean_recog={np.mean(rc):.3f} "
              f"min_recog={np.min(rc):.3f} max|exhib|={np.max(ex):.4f} score={score:+.3f}")
    rows.sort(reverse=True)
    print(f"\nSELECTED: {rows[0][1]} {EXTRA_AXES[rows[0][1]]}")
    save_json(dict(results=res, selected=rows[0][1],
                   axes={k: list(v) for k, v in EXTRA_AXES.items()}),
              os.path.join(os.path.dirname(__file__), "..", "results",
                           "acquired_axis_selection.json"))


if __name__ == "__main__":
    main()
