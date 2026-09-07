"""CrowS-Pairs as an axis family for Experiment B (experiments_v2 §B).

Why a new axis family. Experiment B needs K>=5 axes that BOTH span a range of
direction-sharing `d_axis` AND carry enough removable bias for
(cross - same) to rise above noise. The hand-built valence axes failed the
second requirement: only occupation-gender (pre-bias 0.86) and the injected
gen_fm (0.63) were removable at all, so a 22-row regression had ~1 informative
point.

CrowS-Pairs fixes exactly that:

  * **native minimal pairs** -- each item is two sentences differing only in
    the group term, which is precisely the (congruent, incongruent) structure
    the elicitation and edit pipeline already consumes;
  * **9 bias types** (race-color, gender, socioeconomic, nationality, religion,
    age, sexual-orientation, physical-appearance, disability), 60-516 pairs
    each;
  * **real curated stereotype content**, so the exhibited bias is large rather
    than the near-neutral valence templates.

`stereo_antistereo` gives the congruence label directly: 0 means `sent_more`
is the stereotypical member, 1 means it is the anti-stereotypical one.

DISJOINT SPLITS. Each axis is split once, deterministically, into a PROBE half
(measurement only) and an EDIT half (elicitation + exogenous corpora). No edit
is ever trained on a sentence that is later scored, so what is measured is
generalisation rather than memorisation -- the same discipline used for the
occupation and valence axes.
"""
import functools
import numpy as np
import torch
import datasets
from datasets import load_dataset

datasets.disable_progress_bars()

CROWS_TYPES = {0: "race-color", 1: "socioeconomic", 2: "gender", 3: "disability",
               4: "nationality", 5: "sexual-orientation", 6: "physical-appearance",
               7: "religion", 8: "age"}
AXIS_PREFIX = "crows_"


def axis_name(bias_type):
    return f"{AXIS_PREFIX}{bias_type}"


def is_crows_axis(axis):
    return isinstance(axis, str) and axis.startswith(AXIS_PREFIX)


def bias_type_of(axis):
    return axis[len(AXIS_PREFIX):]


@functools.lru_cache(maxsize=32)
def load_axis(axis, seed=0, probe_frac=0.5):
    """(probe_pairs, edit_pairs) for one CrowS axis; disjoint by construction.

    Each pair is (congruent, incongruent, key) where `congruent` is the
    stereotype-reinforcing sentence.
    """
    bt = bias_type_of(axis)
    d = load_dataset("nyu-mll/crows_pairs", revision="refs/convert/parquet")["test"]
    pairs = []
    for i, r in enumerate(d):
        if CROWS_TYPES.get(r["bias_type"]) != bt:
            continue
        more, less = r["sent_more"], r["sent_less"]
        # stereo_antistereo == 0 -> sent_more is the stereotypical sentence
        cong, incong = (more, less) if r["stereo_antistereo"] == 0 else (less, more)
        if not cong or not incong or cong == incong:
            continue
        pairs.append((cong.strip(), incong.strip(), f"{bt}:{i}"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    cut = int(len(pairs) * probe_frac)
    probe = [pairs[i] for i in idx[:cut]]
    edit = [pairs[i] for i in idx[cut:]]
    return probe, edit


def available_axes(min_pairs=60):
    return [axis_name(t) for t in CROWS_TYPES.values()
            if len(load_axis(axis_name(t))[0]) + len(load_axis(axis_name(t))[1])
            >= min_pairs]


# ---------------------------------------------------------------- measurement
@torch.no_grad()
def crows_profile(model, tok, axis, batch_size=12, split="probe"):
    """Per-item signed bias vector on the held-out PROBE half.

    item value = per-token loglik(congruent) - loglik(incongruent).
    Positive = the model prefers the stereotype-reinforcing sentence.
    """
    from common import seq_loglik
    probe, edit = load_axis(axis)
    pairs = probe if split == "probe" else edit
    texts = []
    for c, i, _k in pairs:
        texts += [c, i]
    ll = seq_loglik(model, tok, texts, batch_size, normalise=True).numpy()
    vals = np.array([float(ll[2 * n] - ll[2 * n + 1]) for n in range(len(pairs))])
    return vals, [p[2] for p in pairs]


def crows_score(model, tok, axis, batch_size=12):
    """Aggregate bias on the axis, in loglik units and as a bounded rate.

    `crows_skew` (the primary outcome for removal) is the mean signed margin;
    0 means no preference between the stereotypical and anti-stereotypical
    member, which is what an unbiased model should show.
    """
    vals, _ = crows_profile(model, tok, axis, batch_size)
    return dict(crows_skew=float(vals.mean()),
                crows_pct=float(100.0 * (vals > 0).mean()),
                crows_rate=float(2.0 * (vals > 0).mean() - 1.0),
                n=int(len(vals)))


# ------------------------------------------------------- corpora for editing
def sd_items(axis, seed=0):
    """(congruent, incongruent, key) triples from the EDIT half."""
    _probe, edit = load_axis(axis, seed=0)
    return list(edit)


def exogenous_corpora(axis, seed=0):
    """Ground-truth biased / debiased corpora from the EDIT half."""
    import random
    items = sd_items(axis)
    biased = [c for c, _i, _k in items]
    debiased = [i for _c, i, _k in items]
    idx = list(range(len(biased)))
    random.Random(seed).shuffle(idx)
    return [biased[i] for i in idx], [debiased[i] for i in idx]
