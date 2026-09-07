"""Outcome measurement for an edited target (design.md §5.3, §5.4).

Two tiers, because the alpha sweep needs many cheap measurements and the
headline table needs a few expensive ones:

  fast_eval  -- the two axis probes plus light collateral. Used to trace the
                bias-vs-collateral FRONTIER over edit strength (design.md
                §5.4: "report as a frontier, not a point").
  full_eval  -- adds BBQ, StereoSet and CrowS-Pairs, and full-size collateral.
                design.md §8 requires triangulating across >=3 benchmarks so
                the law never rests on one instrument's artifacts.

Bias reduction is always reported signed as "how much of the bias was
removed": delta = |bias_pre| - |bias_post|, so positive = removal, negative
= the edit made it worse. Using absolute values matters -- an edit that
overshoots into the opposite bias has not debiased the model.
"""
import numpy as np
import torch
import probes
from probes import eval_occ_probe, eval_valence_probe
import bias_eval as BE
import collateral as COL

_CACHE = {}


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def get_eval_data(tier="fast", seed=0):
    """Datasets are loaded once and reused across every condition."""
    if tier == "fast":
        return dict(
            mmlu=_cached("mmlu_f", lambda: COL.load_mmlu(n=200, seed=seed)),
            wt=_cached("wt_f", lambda: COL.load_wikitext(n_chunks=20, seed=seed)))
    return dict(
        mmlu=_cached("mmlu_F", lambda: COL.load_mmlu(n=800, seed=seed)),
        wt=_cached("wt_F", lambda: COL.load_wikitext(n_chunks=80, seed=seed)),
        bbq_g=_cached("bbq_g", lambda: BE.load_bbq("Gender_identity", n=400, seed=seed)),
        bbq_r=_cached("bbq_r", lambda: BE.load_bbq("Race_ethnicity", n=400, seed=seed)),
        ss=_cached("ss", lambda: BE.load_stereoset("intersentence", n=400, seed=seed,
                                                   bias_type="gender")
                   + BE.load_stereoset("intrasentence", n=400, seed=seed,
                                       bias_type="gender")),
        crows=_cached("crows", lambda: BE.load_crows(n=300, seed=seed,
                                                     bias_type="gender")))


def fast_eval(model, tok, acq_axis, data, batch_size=32):
    import crows_axes as CA, bbq_axes as BB, m_axes as MA, contrast_k as CK, battery_axes as BA
    if BA.is_battery_axis(acq_axis):
        o = eval_occ_probe(model, tok, batch_size=batch_size)
        c = BA.battery_score(model, tok, acq_axis, batch_size=batch_size)
        mm = COL.eval_mmlu(model, tok, data["mmlu"], batch_size=batch_size, n_rot=2)
        pp = COL.eval_perplexity(model, tok, data["wt"], batch_size=max(batch_size // 8, 1))
        return dict(occ_skew=o["occ_skew"], occ_abs_skew=o["occ_abs_skew"],
                    val_skew=float("nan"), bat_skew=c["bat_skew"], bat_pct=c["bat_pct"],
                    mmlu_acc=mm["mmlu_acc"], ppl=pp["ppl"])
    if CK.is_ck_axis(acq_axis):
        o = eval_occ_probe(model, tok, batch_size=batch_size)
        c = CK.ck_score(model, tok, acq_axis, batch_size=batch_size)
        mm = COL.eval_mmlu(model, tok, data["mmlu"], batch_size=batch_size, n_rot=2)
        pp = COL.eval_perplexity(model, tok, data["wt"], batch_size=max(batch_size // 8, 1))
        return dict(occ_skew=o["occ_skew"], occ_abs_skew=o["occ_abs_skew"],
                    val_skew=float("nan"), ck_skew=c["ck_skew"], ck_pct=c["ck_pct"],
                    mmlu_acc=mm["mmlu_acc"], ppl=pp["ppl"])
    if MA.is_m_axis(acq_axis):
        o = eval_occ_probe(model, tok, batch_size=batch_size)
        c = MA.m_score(model, tok, acq_axis, batch_size=batch_size)
        mm = COL.eval_mmlu(model, tok, data["mmlu"], batch_size=batch_size, n_rot=2)
        pp = COL.eval_perplexity(model, tok, data["wt"], batch_size=max(batch_size // 8, 1))
        return dict(occ_skew=o["occ_skew"], occ_abs_skew=o["occ_abs_skew"],
                    val_skew=float("nan"), m_skew=c["m_skew"], m_pct=c["m_pct"],
                    mmlu_acc=mm["mmlu_acc"], ppl=pp["ppl"])
    if BB.is_bbq_axis(acq_axis):
        o = eval_occ_probe(model, tok, batch_size=batch_size)
        c = BB.bbq_score(model, tok, acq_axis, batch_size=batch_size)
        m = COL.eval_mmlu(model, tok, data["mmlu"], batch_size=batch_size, n_rot=2)
        p = COL.eval_perplexity(model, tok, data["wt"], batch_size=max(batch_size // 8, 1))
        return dict(occ_skew=o["occ_skew"], occ_abs_skew=o["occ_abs_skew"],
                    val_skew=float("nan"), bbq_skew=c["bbq_skew"], bbq_pct=c["bbq_pct"],
                    mmlu_acc=m["mmlu_acc"], ppl=p["ppl"])
    if CA.is_crows_axis(acq_axis):
        # CrowS axes carry their own probe; occ_skew is still reported so the
        # inherited gender axis stays visible as a collateral check.
        o = eval_occ_probe(model, tok, batch_size=batch_size)
        c = CA.crows_score(model, tok, acq_axis, batch_size=batch_size)
        m = COL.eval_mmlu(model, tok, data["mmlu"], batch_size=batch_size, n_rot=2)
        p = COL.eval_perplexity(model, tok, data["wt"],
                                batch_size=max(batch_size // 8, 1))
        return dict(occ_skew=o["occ_skew"], occ_abs_skew=o["occ_abs_skew"],
                    val_skew=float("nan"), crows_skew=c["crows_skew"],
                    crows_pct=c["crows_pct"], crows_rate=c["crows_rate"],
                    mmlu_acc=m["mmlu_acc"], ppl=p["ppl"])
    o = eval_occ_probe(model, tok, batch_size=batch_size)
    if acq_axis == "occ_gender":
        # occ_gender's primary metric IS occ_skew; no valence probe applies.
        m = COL.eval_mmlu(model, tok, data["mmlu"], batch_size=batch_size, n_rot=2)
        p = COL.eval_perplexity(model, tok, data["wt"], batch_size=max(batch_size // 8, 1))
        return dict(occ_skew=o["occ_skew"], occ_abs_skew=o["occ_abs_skew"],
                    val_skew=float("nan"), mmlu_acc=m["mmlu_acc"], ppl=p["ppl"])
    v = eval_valence_probe(model, tok, acq_axis, batch_size=batch_size)
    m = COL.eval_mmlu(model, tok, data["mmlu"], batch_size=batch_size, n_rot=2)
    p = COL.eval_perplexity(model, tok, data["wt"], batch_size=max(batch_size // 8, 1))
    return dict(occ_skew=o["occ_skew"], occ_abs_skew=o["occ_abs_skew"],
                val_skew=v["val_skew"], mmlu_acc=m["mmlu_acc"], ppl=p["ppl"])


def full_eval(model, tok, acq_axis, data, batch_size=32):
    r = fast_eval(model, tok, acq_axis, data, batch_size)
    for tag, rows in (("bbq_gender", data["bbq_g"]), ("bbq_race", data["bbq_r"])):
        sc = BE.eval_bbq(model, tok, rows, batch_size)
        sc.pop("_preds", None)
        r[f"{tag}_sAMB"] = sc.get("s_AMB")
        r[f"{tag}_sDIS"] = sc.get("s_DIS")
        r[f"{tag}_acc_amb"] = sc.get("acc_ambig")
    ss = BE.eval_stereoset(model, tok, data["ss"], batch_size)
    r.update(ss_SS=ss["SS"], ss_LMS=ss["LMS"], ss_ICAT=ss["ICAT"])
    r.update(crows_pct=BE.eval_crows(model, tok, data["crows"], batch_size)["crows_pct"])
    return r


# Which metric is the primary bias outcome for each axis, and what value
# counts as unbiased.
PRIMARY = {"occ_gender": ("occ_skew", 0.0)}


def primary_metric(axis):
    import crows_axes as CA, bbq_axes as BB, m_axes as MA, contrast_k as CK, battery_axes as BA
    if BA.is_battery_axis(axis):
        return ("bat_skew", 0.0)
    if CK.is_ck_axis(axis):
        return ("ck_skew", 0.0)
    if MA.is_m_axis(axis):
        return ("m_skew", 0.0)
    if BB.is_bbq_axis(axis):
        return ("bbq_skew", 0.0)
    if CA.is_crows_axis(axis):
        return ("crows_skew", 0.0)
    return PRIMARY.get(axis, ("val_skew", 0.0))


def bias_reduction(pre, post, axis):
    """|bias_pre| - |bias_post| on the axis's primary metric (positive=removed)."""
    key, neutral = primary_metric(axis)
    return abs(pre[key] - neutral) - abs(post[key] - neutral)


def collateral_ok(pre, post, max_mmlu_drop=0.02, max_ppl_ratio=1.10):
    return ((pre["mmlu_acc"] - post["mmlu_acc"]) <= max_mmlu_drop
            and post["ppl"] / pre["ppl"] <= max_ppl_ratio)
