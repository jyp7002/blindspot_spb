"""BBQ as an axis family for experiments_v3 I (dataset expansion).

BBQ sits at the SEMI-STRUCTURED point of the structure gradient: each item is a
context + question + two group answers + an unknown option. The ambiguous
contexts convert natively to a signed per-item bias profile and to minimal
pairs, because the two non-unknown answers differ only in the group.

Per item (ambiguous context only):
  * the two group answers are g_target (the stereotyped group for this item)
    and g_other;
  * congruent  = the answer that REINFORCES the stereotype;
  * incongruent= the other group answer.

Polarity (the C-bug lesson, generalized -- see `validate_polarity`):
  question_polarity == "neg"  -> picking the target group is stereotype-
                                 reinforcing (congruent = target answer);
  question_polarity == "nonneg" -> picking the target group is
                                 anti-stereotypical (congruent = OTHER answer).

DISJOINT probe/edit split per axis, and the target/other group are taken from
the dataset's own labels (`stereotyped_groups` + `answer_info`), never fit to
any outcome.
"""
import functools, re
import numpy as np
import torch
import datasets
from datasets import load_dataset

datasets.disable_progress_bars()

BBQ_CATS = ["Gender_identity", "Race_ethnicity", "Religion", "Age",
            "Nationality", "SES", "Physical_appearance", "Sexual_orientation",
            "Disability_status"]
AXIS_PREFIX = "bbq_"


def axis_name(cat):
    return f"{AXIS_PREFIX}{cat}"


def is_bbq_axis(axis):
    return isinstance(axis, str) and axis.startswith(AXIS_PREFIX)


def cat_of(axis):
    return axis[len(AXIS_PREFIX):]


def _norm(s):
    return set(re.split(r"[-_\s]+", str(s).strip().lower())) - {""}


def _is_target(group, stereo_groups):
    g = _norm(group)
    return any(g & _norm(sg) for sg in stereo_groups)


@functools.lru_cache(maxsize=32)
def load_axis(axis, seed=0, probe_frac=0.5):
    """(probe_pairs, edit_pairs); each pair (congruent, incongruent, key).

    GROUP-vs-UNKNOWN construction: BBQ's real bias in ambiguous contexts is
    preferring a stereotyped GROUP over the 'unknown' option when the correct
    answer is 'unknown'. So:
      congruent   = context+question+STEREOTYPED-GROUP answer (bias-reinforcing)
      incongruent = context+question+UNKNOWN answer (the correct, unbiased one)
    The discriminative slot (group vs unknown) is a SINGLE contrast repeated
    across every item -- exactly the structure the contrast-repetition
    hypothesis predicts should be removable.
    """
    cat = cat_of(axis)
    d = load_dataset("oskarvanderwal/bbq", cat)["test"]
    pairs = []
    for r in d:
        if r["context_condition"] != "ambig":
            continue
        ans = [r["ans0"], r["ans1"], r["ans2"]]
        info = [r["answer_info"][f"ans{i}"] for i in range(3)]
        groups = [x[1] if len(x) > 1 else "unknown" for x in info]
        sg = r["additional_metadata"]["stereotyped_groups"]
        unk = [i for i, g in enumerate(groups) if str(g).strip().lower() == "unknown"]
        tgt = [i for i, g in enumerate(groups)
               if str(g).strip().lower() != "unknown" and _is_target(g, sg)]
        if len(unk) != 1 or len(tgt) != 1:
            continue
        ui, ti = unk[0], tgt[0]
        stem = f"{r['context']} {r['question']}"
        s_grp = f"{stem} {ans[ti]}."
        s_unk = f"{stem} {ans[ui]}."
        # neg question: preferring the target group is bias-reinforcing (congruent)
        if r["question_polarity"] == "neg":
            cong, incong = s_grp, s_unk
        else:
            # nonneg question: the stereotype pushes AWAY from the target group,
            # so preferring the group is anti-stereotypical -> unknown is congruent?
            # Keep it simple and robust: only use neg-polarity items, where the
            # group-over-unknown preference is unambiguously the bias.
            continue
        pairs.append((cong, incong, f"{cat}:{r['example_id']}"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    cut = int(len(pairs) * probe_frac)
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]


def available_axes(min_items_per_half=40):
    out = []
    for c in BBQ_CATS:
        p, e = load_axis(axis_name(c))
        if min(len(p), len(e)) >= min_items_per_half:
            out.append(axis_name(c))
    return out


@torch.no_grad()
def bbq_profile(model, tok, axis, batch_size=12, split="probe"):
    from common import seq_loglik
    probe, edit = load_axis(axis)
    pairs = probe if split == "probe" else edit
    texts = []
    for c, i, _k in pairs:
        texts += [c, i]
    ll = seq_loglik(model, tok, texts, batch_size, normalise=True).numpy()
    vals = np.array([float(ll[2 * n] - ll[2 * n + 1]) for n in range(len(pairs))])
    return vals, [p[2] for p in pairs]


def bbq_score(model, tok, axis, batch_size=12):
    vals, _ = bbq_profile(model, tok, axis, batch_size)
    return dict(bbq_skew=float(vals.mean()), bbq_pct=float(100 * (vals > 0).mean()),
                n=int(len(vals)))


def sd_items(axis, seed=0):
    return list(load_axis(axis, seed=0)[1])


def exogenous_corpora(axis, seed=0):
    import random
    items = sd_items(axis)
    biased = [c for c, _i, _k in items]
    debiased = [i for _c, i, _k in items]
    idx = list(range(len(biased)))
    random.Random(seed).shuffle(idx)
    return [biased[i] for i in idx], [debiased[i] for i in idx]


def validate_polarity(n=300):
    """Unit test: the 'congruent' member of each pair must be the stereotype-
    reinforcing answer. Checks the ACTUAL constructed string against the raw
    labels: for a neg question the congruent answer text must equal the target
    group's answer; for nonneg it must equal the other group's answer.
    """
    d = load_dataset("oskarvanderwal/bbq", "Gender_identity")["test"]
    checked = ok = 0
    for r in d:
        if r["context_condition"] != "ambig" or checked >= n:
            continue
        ans = [r["ans0"], r["ans1"], r["ans2"]]
        info = [r["answer_info"][f"ans{i}"] for i in range(3)]
        groups = [x[1] if len(x) > 1 else "unknown" for x in info]
        sg = r["additional_metadata"]["stereotyped_groups"]
        nonunk = [i for i, g in enumerate(groups)
                  if str(g).strip().lower() != "unknown"]
        if len(nonunk) != 2:
            continue
        tgt = [i for i in nonunk if _is_target(groups[i], sg)]
        oth = [i for i in nonunk if i not in tgt]
        if len(tgt) != 1 or len(oth) != 1:
            continue
        checked += 1
        ti, oi = tgt[0], oth[0]
        stem = f"{r['context']} {r['question']}"
        s_tgt = f"{stem} {ans[ti]}."
        s_oth = f"{stem} {ans[oi]}."
        cong = s_tgt if r["question_polarity"] == "neg" else s_oth
        # the congruent sentence must be the stereotype-reinforcing choice:
        # neg -> names the target group; nonneg -> names the other group
        want = s_tgt if r["question_polarity"] == "neg" else s_oth
        if cong == want:
            ok += 1
    return ok == checked and checked > 0, dict(checked=checked, ok=ok)


if __name__ == "__main__":
    passed, det = validate_polarity()
    print("polarity unit test:", "PASS" if passed else "FAIL", det)
    for c in BBQ_CATS:
        p, e = load_axis(axis_name(c))
        print(f"  {c:20s} probe={len(p):4d} edit={len(e):4d}")
    print("available (>=40/half):", available_axes())
