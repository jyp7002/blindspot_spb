"""T prediction-battery axes (experiments_v4 Stage 3): WinoBias + StereoSet.

Predictions REGISTERED in PREREGISTRATION (T0) before this onboarding:
  * WinoBias  -> REMOVABLE  (the he/she pronoun contrast repeats across items)
  * StereoSet intrasentence -> DEFINITIONAL: the group is fixed in the context
    and the ATTRIBUTE token varies per item. Token-contrast reading => NOT
    removable (many contrasts); semantic-direction reading => partially
    removable (one stereo-vs-anti direction). The outcome DEFINES "contrast".

Both are minimal-pair / likelihood constructions with disjoint probe/edit splits
and a polarity unit test; the standard hygiene gates apply.
"""
import functools, re
import numpy as np
import torch
import datasets
from datasets import load_dataset

datasets.disable_progress_bars()

WB = "wb_gender"
SS = "ss_intra"


def is_battery_axis(axis):
    return axis in (WB, SS)


# ------------------------------------------------------------------ WinoBias
@functools.lru_cache(maxsize=4)
def _wb(seed=0, probe_frac=0.5):
    d = load_dataset("Elfsong/wino_bias")["test"]
    pairs = []
    for r in d:
        inp = r["input"]
        m = re.search(r"\[([^\]]+)\]", inp)
        if not m:
            continue
        pron = m.group(1).strip().lower()
        swap = {"he": "she", "she": "he", "him": "her", "her": "him",
                "his": "her", "hers": "his"}
        if pron not in swap:
            continue
        given = inp.replace(m.group(0), m.group(1))          # sentence with given pronoun
        other = inp.replace(m.group(0), swap[pron])
        # polarity == stereotypical => the GIVEN pronoun is stereotype-congruent
        if r["polarity"] == "stereotypical":
            cong, incong = given, other
        else:
            cong, incong = other, given
        pairs.append((cong, incong, f"wb:{len(pairs)}"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    cut = int(len(pairs) * probe_frac)
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]


# ------------------------------------------------------------------ StereoSet
@functools.lru_cache(maxsize=4)
def _ss(seed=0, probe_frac=0.5):
    d = load_dataset("McGill-NLP/stereoset", "intrasentence")["validation"]
    pairs = []
    for r in d:
        s = r["sentences"]
        labels = list(s["gold_label"])
        try:
            anti = labels.index(0)     # 0 = anti-stereotype
            ster = labels.index(1)     # 1 = stereotype
        except ValueError:
            continue
        cong, incong = s["sentence"][ster], s["sentence"][anti]
        if cong == incong:
            continue
        pairs.append((cong, incong, f"ss:{r['id']}"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    cut = int(len(pairs) * probe_frac)
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]


def load_axis(axis, seed=0, probe_frac=0.5):
    return _wb(seed, probe_frac) if axis == WB else _ss(seed, probe_frac)


@torch.no_grad()
def battery_profile(model, tok, axis, batch_size=16, split="probe"):
    from common import seq_loglik
    probe, edit = load_axis(axis)
    pairs = probe if split == "probe" else edit
    texts = []
    for c, i, _k in pairs:
        texts += [c, i]
    ll = seq_loglik(model, tok, texts, batch_size, normalise=True).numpy()
    vals = np.array([float(ll[2 * n] - ll[2 * n + 1]) for n in range(len(pairs))])
    return vals, [p[2] for p in pairs]


def battery_score(model, tok, axis, batch_size=16):
    vals, _ = battery_profile(model, tok, axis, batch_size)
    return dict(bat_skew=float(vals.mean()), bat_pct=float(100 * (vals > 0).mean()),
                n=int(len(vals)))


def sd_items(axis, seed=0):
    return list(load_axis(axis)[1])


def exogenous_corpora(axis, seed=0):
    import random
    items = sd_items(axis)
    b = [c for c, _i, _k in items]
    d = [i for _c, i, _k in items]
    idx = list(range(len(b)))
    random.Random(seed).shuffle(idx)
    return [b[i] for i in idx], [d[i] for i in idx]


if __name__ == "__main__":
    for ax in [WB, SS]:
        p, e = load_axis(ax)
        print(f"{ax}: probe={len(p)} edit={len(e)}")
        for c, i, _k in p[:2]:
            print(f"   cong: {c[:75]}\n   inc : {i[:75]}")
