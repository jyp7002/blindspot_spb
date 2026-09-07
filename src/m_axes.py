"""M axis family: pre-rewritten templatized/de-templatized corpora.

Consumes results/m_corpora/<tag>.json produced by m_rewrite.py. Exposes the
same (probe/edit, profile, sd_items, exogenous_corpora) interface as the other
axis families so run_experiment can debias them with no special-casing.

Axis names: mt_crows_<cat> (templatized CrowS), md_occ_gender (de-templatized
occupation-gender).
"""
import os, json, functools
import numpy as np
import torch

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
MCORP = os.path.join(RESULTS, "m_corpora")


def is_m_axis(axis):
    return isinstance(axis, str) and (axis.startswith("mt_") or axis.startswith("md_"))


@functools.lru_cache(maxsize=16)
def load_axis(axis, seed=0):
    d = json.load(open(os.path.join(MCORP, f"{axis}.json")))
    probe = [tuple(x) for x in d["probe"]]
    edit = [tuple(x) for x in d["edit"]]
    return probe, edit


@torch.no_grad()
def m_profile(model, tok, axis, batch_size=16, split="probe"):
    from common import seq_loglik
    probe, edit = load_axis(axis)
    pairs = probe if split == "probe" else edit
    texts = []
    for c, i, _k in pairs:
        texts += [c, i]
    ll = seq_loglik(model, tok, texts, batch_size, normalise=True).numpy()
    vals = np.array([float(ll[2 * n] - ll[2 * n + 1]) for n in range(len(pairs))])
    return vals, [p[2] for p in pairs]


def m_score(model, tok, axis, batch_size=16):
    vals, _ = m_profile(model, tok, axis, batch_size)
    return dict(m_skew=float(vals.mean()), m_pct=float(100 * (vals > 0).mean()), n=int(len(vals)))


def sd_items(axis, seed=0):
    return list(load_axis(axis)[1])


def exogenous_corpora(axis, seed=0):
    import random
    items = sd_items(axis)
    b = [c for c, _i, _k in items]
    d = [i for _c, i, _k in items]
    idx = list(range(len(b))); random.Random(seed).shuffle(idx)
    return [b[i] for i in idx], [d[i] for i in idx]
