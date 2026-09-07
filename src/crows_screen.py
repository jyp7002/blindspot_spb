"""Screen the CrowS axes for Experiment B viability, before committing GPU.

An axis is usable for B only if it satisfies BOTH requirements that the
valence axes failed to satisfy jointly:

  1. enough exhibited bias that removal has dynamic range (|crows_skew| well
     clear of 0 across models), and
  2. a d_axis that, together with the other axes, spans a range.

Measuring both is cheap -- two forward passes per probe pair, no training --
so this screen runs first and the removal sweep is then aimed only at the
axes that can actually carry it.
"""
import os, argparse, itertools
import numpy as np
from common import load, free, FAMILY, save_json
from geometry import profile_similarity, bootstrap_ci
import crows_axes as CA

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["qwen0.5b", "qwen1.5b", "qwen3b", "llama1b",
                             "llama3b", "gemma2b", "gemma9b", "smol1.7b",
                             "smol360m", "phi3.5", "phi3mini"])
    ap.add_argument("--axes", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=12)
    a = ap.parse_args()
    axes = a.axes or CA.available_axes()

    prof, skew = {}, {}
    for name in a.models:
        model, tok = load(name)
        try:
            for ax in axes:
                v, _ = CA.crows_profile(model, tok, ax, batch_size=a.batch_size)
                prof[(name, ax)] = v
                skew[(name, ax)] = float(v.mean())
        finally:
            free(model, tok)
            model = tok = None
        print(f"[screen] {name:10s} " + " ".join(
            f"{ax.split('_')[1][:8]}={skew[(name, ax)]:+.3f}" for ax in axes),
            flush=True)

    out = {}
    print(f"\n{'axis':28s} {'|skew|':>8s} {'d_same':>8s} {'d_cross':>8s} {'diff':>8s} {'n_pairs':>8s}")
    print("-" * 76)
    rows = []
    for ax in axes:
        mags = [abs(skew[(m, ax)]) for m in a.models]
        same, cross = [], []
        for m1, m2 in itertools.combinations(a.models, 2):
            c = profile_similarity(prof[(m1, ax)], prof[(m2, ax)])
            if c is None:
                continue
            (same if FAMILY[m1] == FAMILY[m2] else cross).append(c)
        ds, ds_ci = bootstrap_ci(same)
        dc, dc_ci = bootstrap_ci(cross)
        n = len(CA.load_axis(ax)[0])
        out[ax] = dict(mean_abs_skew=float(np.mean(mags)),
                       min_abs_skew=float(np.min(mags)),
                       d_axis=ds, d_ci=ds_ci, cross_axis=dc, cross_ci=dc_ci,
                       differentiation=(ds - dc) if (ds is not None and dc is not None) else None,
                       n_probe_pairs=n,
                       skew_by_model={m: skew[(m, ax)] for m in a.models})
        rows.append((ax, np.mean(mags), ds, dc, out[ax]["differentiation"], n))
    for ax, mg, ds, dc, df, n in sorted(rows, key=lambda r: -r[1]):
        print(f"{ax:28s} {mg:8.4f} {ds:8.3f} {dc:8.3f} {df:+8.3f} {n:8d}")

    save_json(out, os.path.join(RESULTS, "crows_screen.json"))
    print("\nwrote results/crows_screen.json")
    good = [ax for ax, v in out.items() if v["mean_abs_skew"] >= 0.05]
    print(f"\nviable for B (mean |skew| >= 0.05): {len(good)} axes")
    print("  " + " ".join(good))


if __name__ == "__main__":
    main()
