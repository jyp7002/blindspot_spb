"""Frame-overlap score — the hypothesized structural driver of removability.

experiments_v3 G and K1 both fail toward the same missing factor: an axis is
removable only when a low-rank edit trained on the EDIT corpus generalizes to
the held-out PROBE corpus. The plan pre-registers a structural explanation:
removability is high when probe and edit items share sentence FRAMES (so the
learned direction is about the group slot, not the surrounding content) and low
when they are heterogeneous natural sentences.

`frame_overlap(axis)` quantifies this from the corpora alone (no GPU): the mean
max Jaccard n-gram similarity between each edit item and the probe set, after
masking the group/attribute slot. High overlap = templated; low = free-form.

This is computed PRE-EDIT and never sees a removal outcome (circularity guard).
"""
import re
import numpy as np


def _shingles(text, n=3):
    toks = re.findall(r"[a-z]+", text.lower())
    if len(toks) < n:
        return set(toks)
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def _jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def frame_overlap(probe_texts, edit_texts, n=3, sample=300, seed=0):
    """Mean, over edit items, of the max n-gram Jaccard to any probe item.

    ~1.0 => every edit item reuses a probe frame (fully templated);
    ~0.0 => edit and probe share no multi-word structure (free-form).
    """
    rng = np.random.default_rng(seed)
    if len(edit_texts) > sample:
        edit_texts = [edit_texts[i] for i in rng.permutation(len(edit_texts))[:sample]]
    if len(probe_texts) > sample:
        probe_texts = [probe_texts[i] for i in rng.permutation(len(probe_texts))[:sample]]
    P = [_shingles(t, n) for t in probe_texts]
    best = []
    for e in edit_texts:
        se = _shingles(e, n)
        best.append(max((_jac(se, p) for p in P), default=0.0))
    return float(np.mean(best)) if best else 0.0


def axis_frame_overlap(axis):
    """Frame-overlap for any registered axis, using its own corpora."""
    # occupation-gender: templated probe + templated edit pool
    if axis == "occ_gender":
        from probes import PROBE_TEMPLATES, PROBE_OCC, INJECT_OCC
        from elicit import POOL_OCC_TEMPLATES, PRON_M, PRON_F
        probe = [pre.format(occ=o) + " " + w + " " + tail
                 for o in PROBE_OCC for (pre, w1, w2, tail) in PROBE_TEMPLATES
                 for w in (w1, w2)]
        edit = [t.format(occ=o, **(PRON_F if g == "f" else PRON_M))
                for o, g in INJECT_OCC.items() for t in POOL_OCC_TEMPLATES]
        return frame_overlap(probe, edit)
    # valence axes: templated
    import probes as PB
    if axis in PB.CANDIDATE_AXES:
        from probes import PROBE_VAL_TEMPLATES, PROBE_NEG, PROBE_POS, CANDIDATE_AXES
        from elicit import POOL_VAL_TEMPLATES
        gA, gB = CANDIDATE_AXES[axis]
        probe = [t.format(G=g, a=a) for t in PROBE_VAL_TEMPLATES
                 for a in PROBE_NEG + PROBE_POS for g in (gA, gB)]
        edit = [t.format(G=g, a=a) for t in POOL_VAL_TEMPLATES
                for a in PROBE_NEG + PROBE_POS for g in (gA, gB)]
        return frame_overlap(probe, edit)
    # CrowS: naturalistic, native probe/edit split
    import crows_axes as CA
    if CA.is_crows_axis(axis):
        p, e = CA.load_axis(axis)
        return frame_overlap([x[0] for x in p] + [x[1] for x in p],
                             [x[0] for x in e] + [x[1] for x in e])
    # BBQ: semi-structured (shared context+question stem within an item, but
    # items differ)
    try:
        import bbq_axes as BB
        if BB.is_bbq_axis(axis):
            p, e = BB.load_axis(axis)
            return frame_overlap([x[0] for x in p] + [x[1] for x in p],
                                 [x[0] for x in e] + [x[1] for x in e])
    except Exception:
        pass
    return None
