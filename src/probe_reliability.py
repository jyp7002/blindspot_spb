"""Is low d_axis real disagreement, or just an unreliable probe?

Experiment B failed because the axes that span the LOW end of d_axis all carry
tiny bias (pre-bias 0.004-0.049), so their blind-spot magnitude is noise. But
there is a second, more damaging possibility hiding in that fact:

  d_axis is a CORRELATION between two models' item profiles, and correlations
  are attenuated by measurement noise. If an axis's profile is unreliable
  (because the model barely exhibits it), the observed between-model
  correlation is pushed toward 0 REGARDLESS of whether the two models encode
  the axis the same way.

If that is what is happening, then "low-d axes" are not low-sharing axes at
all -- they are just noisy ones -- and d_axis is confounded with bias
magnitude by construction. That would invalidate the Experiment B regressor
itself, not merely its power.

Test: split-half reliability of each axis's profile within a single model
(odd vs even items, Spearman-Brown corrected). Then the disattenuated
correlation between models is

    d_true = d_obs / sqrt(rel_1 * rel_2)

If reliability is high on low-d axes, their low d is real disagreement and the
regressor is sound (B is merely underpowered). If reliability is low exactly
where d is low, the regressor is confounded and B needs a different design.
"""
import os, argparse, itertools
import numpy as np
from common import load, free, FAMILY, save_json
from geometry import bias_profile, profile_similarity

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def split_half(profile):
    """Spearman-Brown corrected odd/even split-half reliability."""
    a, b = profile[0::2], profile[1::2]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    r = profile_similarity(a, b)
    if r is None:
        return None
    return (2 * r) / (1 + r) if (1 + r) != 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["qwen1.5b", "qwen3b", "llama1b", "llama3b",
                             "gemma2b", "smol1.7b", "phi3.5"])
    ap.add_argument("--axes", nargs="*",
                    default=["wt_os", "nat_ro_dk", "nat_pk_nl", "rel_bu_lu",
                             "age_oy", "occ_gender", "ses_lw", "gen_fm"])
    ap.add_argument("--batch-size", type=int, default=12)
    a = ap.parse_args()

    prof, rel = {}, {}
    for name in a.models:
        model, tok = load(name)
        try:
            for ax in a.axes:
                p, _ = bias_profile(model, tok, ax, batch_size=a.batch_size)
                prof[(name, ax)] = p
                rel[(name, ax)] = split_half(p)
        finally:
            free(model, tok)
            model = tok = None
        print(f"[rel] {name:10s} " + "  ".join(
            f"{ax}={rel[(name, ax)]:+.2f}" if rel[(name, ax)] is not None else f"{ax}=NA"
            for ax in a.axes), flush=True)

    out = {}
    print(f"\n{'axis':12s} {'reliability':>12s} {'d_obs':>8s} {'d_disatt':>9s} {'|bias|':>8s}")
    print("-" * 56)
    for ax in a.axes:
        rs = [rel[(m, ax)] for m in a.models if rel.get((m, ax)) is not None]
        rbar = float(np.mean(rs)) if rs else None
        same = []
        for m1, m2 in itertools.combinations(a.models, 2):
            if FAMILY[m1] != FAMILY[m2]:
                continue
            c = profile_similarity(prof[(m1, ax)], prof[(m2, ax)])
            if c is not None:
                same.append(c)
        d_obs = float(np.mean(same)) if same else None
        mag = float(np.mean([abs(prof[(m, ax)].mean()) for m in a.models]))
        d_dis = None
        if d_obs is not None and rbar and rbar > 0:
            d_dis = float(np.clip(d_obs / np.sqrt(rbar * rbar), -1.5, 1.5))
        out[ax] = dict(reliability=rbar, d_obs=d_obs, d_disattenuated=d_dis,
                       bias_magnitude=mag, n_same_pairs=len(same))
        print(f"{ax:12s} {rbar if rbar is None else round(rbar,3)!s:>12} "
              f"{d_obs if d_obs is None else round(d_obs,3)!s:>8} "
              f"{d_dis if d_dis is None else round(d_dis,3)!s:>9} {mag:8.4f}")

    rr = [v["reliability"] for v in out.values() if v["reliability"] is not None]
    dd = [out[k]["d_obs"] for k in out if out[k]["reliability"] is not None]
    mm = [out[k]["bias_magnitude"] for k in out if out[k]["reliability"] is not None]
    if len(rr) > 2:
        print(f"\ncorr(reliability, d_obs)          = {np.corrcoef(rr, dd)[0,1]:+.3f}")
        print(f"corr(reliability, bias_magnitude) = {np.corrcoef(rr, mm)[0,1]:+.3f}")
        print("\nIf corr(reliability, d_obs) is strongly positive, d_axis is "
              "confounded with probe reliability and the Experiment B regressor "
              "is not measuring direction-sharing.")
    save_json(out, os.path.join(RESULTS, "probe_reliability.json"))


if __name__ == "__main__":
    main()
