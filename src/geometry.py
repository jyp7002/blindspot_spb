"""Experiment C — bias-direction geometry at higher fidelity (experiments_v2 §C).

Why this exists: the MVP's alignment signal is real but fragile in absolute
terms (cosines +0.144 cross vs +0.029 same; r=+0.38 over 72 coarse sketches).
The reframed mechanism -- *a designer cannot get leverage on a bias it encodes
in the same weight directions as its target* -- rests on this measurement, so
it must be tightened before it can carry Experiment B or a mediation claim.


DEVIATION FROM experiments_v2 §B, AND WHY
-----------------------------------------
§B defines `d_axis` as "mean pairwise cosine of *same-family* bias directions
(measured in weight-delta space, pre-edit)". That quantity is **not
well-defined across models** and cannot be used:

  * Same-family siblings have different widths (Qwen2.5-1.5B hidden 1536 vs
    Qwen2.5-3B hidden 2048; Gemma-2 2b 2304 vs 9b 3584), so their weight
    deltas live in different spaces and have no coordinate correspondence.
  * Even at equal width, two independently trained models are related only up
    to permutation/rotation of hidden units. A raw coordinate-wise cosine
    between their weight deltas is ~0 by construction, whatever bias they
    share. It would measure basis mismatch, not direction-sharing.

So weight-space cosine is kept ONLY where it is meaningful -- between two
directions trained on the SAME model (seed stability; and the MVP's
elicited-vs-ground-truth alignment, which was valid precisely because both
directions were trained on the same target).

For CROSS-MODEL direction-sharing we use a basis-independent **functional
profile**: the model's per-item signed bias over the shared probe set. Two
models that encode an axis the same way are biased on the *same items in the
same direction*, regardless of width or basis. `d_axis` becomes the mean
pairwise correlation of those profiles within a family. This is
representation-independent, defined for every pair, comparable across
families, and still strictly pre-edit.


CIRCULARITY GUARD (experiments_v2 §B, non-negotiable)
-----------------------------------------------------
Every quantity here is pre-edit and computed on a pipeline disjoint from any
editing outcome. Profiles come from the probe sets; weight directions, where
used, come from the EXOGENOUS ground-truth corpora only, never a designer's
elicitation. Axis "inherited-ness" is never defined as "where same-family
fails", and nothing here ever sees a bias-reduction number.
"""
import os, argparse, itertools, time
import numpy as np
import torch

import probes
from probes import PROBE_TEMPLATES, PROBE_OCC, INJECT_OCC, CANDIDATE_AXES
from probes import PROBE_NEG, PROBE_POS, PROBE_VAL_TEMPLATES
from common import load, free, MODELS, FAMILY, save_json, cont_loglik, seq_loglik
from edit import train_task_vector, contrast
from elicit_gen import exogenous_corpora

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
DIR_DIR = os.path.join(RESULTS, "directions")


# ---------------------------------------------------------------- profiles
@torch.no_grad()
def bias_profile(model, tok, axis, batch_size=12):
    """Per-item signed bias vector, in a space SHARED across all models.

    occ_gender : one entry per (occupation, template) = P(she | context)
                 mapped to a signed stereotype-congruence score.
    valence    : one entry per (template, attribute) = per-token loglik
                 advantage of group A over group B on that attribute.

    The item ordering is fixed, so profiles from any two models are directly
    comparable regardless of architecture or width.
    """
    import crows_axes as CA, bbq_axes as BB, m_axes as MA
    if MA.is_m_axis(axis):
        return MA.m_profile(model, tok, axis, batch_size=batch_size)
    if BB.is_bbq_axis(axis):
        return BB.bbq_profile(model, tok, axis, batch_size=batch_size)
    if CA.is_crows_axis(axis):
        return CA.crows_profile(model, tok, axis, batch_size=batch_size)
    if axis == "occ_gender":
        occ = {**INJECT_OCC, **PROBE_OCC}
        prefixes, conts, keys = [], [], []
        for o, g in occ.items():
            for ti, (pre, w_he, w_she, tail) in enumerate(PROBE_TEMPLATES):
                ctx = pre.format(occ=o)
                for w in (w_he, w_she):
                    prefixes.append(ctx)
                    conts.append(" " + w + " " + tail)
                keys.append((o, g, ti))
        ll = cont_loglik(model, tok, prefixes, conts, batch_size).numpy()
        vals = []
        for i, (o, g, ti) in enumerate(keys):
            d = float(ll[2 * i + 1] - ll[2 * i])       # log P(she) - log P(he)
            p_f = 1.0 / (1.0 + np.exp(-d))
            # signed toward the conventional stereotype for that occupation
            vals.append((p_f - 0.5) if g == "f" else (0.5 - p_f))
        return np.array(vals), [f"{o}|{ti}" for o, g, ti in keys]

    gA, gB = CANDIDATE_AXES[axis]
    texts, keys = [], []
    for t in PROBE_VAL_TEMPLATES:
        for pol, attrs in (("neg", PROBE_NEG), ("pos", PROBE_POS)):
            for a in attrs:
                texts += [t.format(G=gA, a=a), t.format(G=gB, a=a)]
                keys.append((t[:18], pol, a))
    ll = seq_loglik(model, tok, texts, batch_size, normalise=True).numpy()
    # Sign by polarity so that POSITIVE always means stereotype-congruent,
    # exactly as the occupation-gender profile above does.
    #
    # Without this, a negative-attribute item and a positive-attribute item
    # both contribute +(ll_A - ll_B), so the profile mean is dominated by the
    # raw frequency difference between the two group NAMES rather than by any
    # bias, and two models correlate highly on this profile merely by agreeing
    # that (say) "Dutch" is a commoner token than "Pakistani". d_axis would
    # then measure shared token frequency, not shared bias direction.
    vals = []
    for i, (_t, pol, _a) in enumerate(keys):
        d = float(ll[2 * i] - ll[2 * i + 1])       # group A minus group B
        vals.append(d if pol == "neg" else -d)
    return np.array(vals), [f"{k[1]}|{k[2]}|{k[0]}" for k in keys]


def profile_similarity(p1, p2):
    """Pearson correlation of two item-space profiles (basis-independent)."""
    if p1 is None or p2 is None or len(p1) != len(p2):
        return None
    a, b = p1 - p1.mean(), p2 - p2.mean()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else None


# ------------------------------------------------- weight-space (same model)
def _path(model, axis, seed):
    return os.path.join(DIR_DIR, f"{model}|{axis}|s{seed}.pt")


def get_direction(model_name, axis, seed, *, steps=250, lr=1e-4, rank=16,
                  bs=8, cache=True, model=None, tok=None):
    """Pre-edit weight-space bias direction, from GROUND-TRUTH corpora.

    Only ever compared against another direction trained on the SAME model
    (see module docstring).
    """
    p = _path(model_name, axis, seed)
    if cache and os.path.exists(p):
        return {k: v.float() for k, v in torch.load(p, map_location="cpu").items()}
    b, d = exogenous_corpora(axis, seed=seed)
    own = model is None
    if own:
        model, tok = load(model_name)
    try:
        dw_b, _ = train_task_vector(model, tok, b, rank=rank, steps=steps,
                                    lr=lr, bs=bs, seed=seed)
        dw_d, _ = train_task_vector(model, tok, d, rank=rank, steps=steps,
                                    lr=lr, bs=bs, seed=seed)
    finally:
        if own:
            free(model, tok)
    v = contrast(dw_b, dw_d)
    if cache:
        os.makedirs(DIR_DIR, exist_ok=True)
        torch.save({k: t.to(torch.float16) for k, t in v.items()}, p)
    return v


def exact_cosine(v1, v2):
    """Cosine over shared modules, float64. Same-model comparisons only."""
    keys = [k for k in v1 if k in v2 and tuple(v1[k].shape) == tuple(v2[k].shape)]
    if not keys:
        return None
    a = torch.cat([v1[k].flatten() for k in keys]).double()
    b = torch.cat([v2[k].flatten() for k in keys]).double()
    n = a.norm() * b.norm()
    return float((a @ b / n).item()) if n > 0 else None


def bootstrap_ci(vals, n_boot=2000, seed=0):
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else None), (None, None)
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), (float(np.percentile(bs, 2.5)),
                             float(np.percentile(bs, 97.5)))


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axes", nargs="*", default=["occ_gender", "gen_fm"])
    ap.add_argument("--models", nargs="*",
                    default=["qwen0.5b", "qwen1.5b", "qwen3b", "llama1b",
                             "llama3b", "gemma2b", "gemma9b", "smol1.7b",
                             "smol360m", "phi3.5", "phi3mini"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--weight-space", action="store_true",
                    help="also derive weight-space directions (same-model "
                         "seed-stability only); expensive")
    ap.add_argument("--out", default=os.path.join(RESULTS, "geometry.json"))
    a = ap.parse_args()

    probes.CANDIDATE_AXES.update(CANDIDATE_AXES)
    out = {}
    prof = {}

    for name in a.models:
        model, tok = load(name)
        try:
            for axis in a.axes:
                p, keys = bias_profile(model, tok, axis, batch_size=12)
                prof[(name, axis)] = p
                print(f"[prof] {name:10s} {axis:11s} n_items={len(p)} "
                      f"mean={p.mean():+.4f} sd={p.std():.4f}", flush=True)
            if a.weight_space:
                for axis in a.axes:
                    for s in a.seeds:
                        get_direction(name, axis, s, model=model, tok=tok)
        finally:
            free(model, tok)

    for axis in a.axes:
        same, cross = [], []
        pairs = {}
        for m1, m2 in itertools.combinations(a.models, 2):
            c = profile_similarity(prof.get((m1, axis)), prof.get((m2, axis)))
            if c is None:
                continue
            pairs[f"{m1}|{m2}"] = c
            (same if FAMILY[m1] == FAMILY[m2] else cross).append(c)
        d_mean, d_ci = bootstrap_ci(same)
        x_mean, x_ci = bootstrap_ci(cross)
        out[axis] = dict(
            axis=axis, metric="item_space_profile_correlation",
            d_axis=d_mean, d_ci=d_ci, n_same_pairs=len(same),
            cross_axis=x_mean, cross_ci=x_ci, n_cross_pairs=len(cross),
            differentiation=(d_mean - x_mean) if (d_mean is not None
                                                 and x_mean is not None) else None,
            pairs=pairs,
            profile_mean={m: float(prof[(m, axis)].mean()) for m in a.models
                          if (m, axis) in prof})
        print(f"\n[geom] {axis}: d_axis(same-family)={d_mean:+.4f} CI={d_ci} "
              f"(n={len(same)});  cross-family={x_mean:+.4f} CI={x_ci} "
              f"(n={len(cross)});  differentiation={out[axis]['differentiation']:+.4f}",
              flush=True)
        save_json(out, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
