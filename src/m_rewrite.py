"""experiments_v3 M — rewriter for the templatize / de-templatize causal test.

M asks whether the templated<->naturalistic scope boundary is STRUCTURAL
(probe<->edit frame overlap => low-rank edit generalizes) or about CONTENT
(naturalistic stereotypes inseparable from capability). The rewriter must sit
OUTSIDE the panel designer families so its own bias directions cannot
contaminate the co-encoding being tested.

Rewriter priority:
  1. Anthropic API (Claude) if a working key is in ~/.anthropic_key -- cleanest,
     definitively non-panel, strongest controlled rewriting.
  2. granite-3.1-8b-instruct (local, non-panel) as fallback.

Two transforms, polarity preserved and validated against labels:
  templatize(crows_axis)   : rewrite each (stereo, anti-stereo) pair into a
                             SHARED sentence frame, so probe and edit overlap.
  detemplatize(occ_gender) : paraphrase-diversify the frames, destroying
                             probe<->edit overlap while keeping the content.

Outputs are cached to results/m_corpora/<axis>.json as
{probe:[(cong,incong,key)], edit:[...]} so the M runs consume them like any
other axis family.
"""
import os, json, re, time, argparse
import numpy as np

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
MCORP = os.path.join(RESULTS, "m_corpora")
KEYFILE = os.path.expanduser("~/.anthropic_key")


# --------------------------------------------------------------- rewriter backends
def _anthropic_client():
    if not os.path.exists(KEYFILE):
        return None
    key = open(KEYFILE).read().strip()
    try:
        import anthropic
        c = anthropic.Anthropic(api_key=key)
        c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=5,
                          messages=[{"role": "user", "content": "OK"}])
        return c
    except Exception:
        return None


def _local_rewriter():
    from common import load
    return load("granite8b")


class Rewriter:
    def __init__(self):
        self.client = _anthropic_client()
        self.local = None
        self.backend = "anthropic" if self.client else "granite8b"

    def _one(self, p, max_new):
        for _ in range(3):
            try:
                r = self.client.messages.create(
                    model="claude-sonnet-5", max_tokens=max_new,
                    messages=[{"role": "user", "content": p}])
                return r.content[0].text.strip()
            except Exception:
                time.sleep(2)
        return ""

    def __call__(self, prompts, max_new=120, batch=16):
        if self.client:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=8) as ex:
                return list(ex.map(lambda p: self._one(p, max_new), prompts))
        if self.local is None:
            self.local = _local_rewriter()
        from elicit_gen import generate
        from common import chat_prompt
        model, tok = self.local
        cps = [chat_prompt(tok, p) for p in prompts]
        return [g.strip() for g in generate(model, tok, cps, max_new_tokens=max_new,
                                            batch_size=batch, temperature=0.0)]


# --------------------------------------------------------------- transforms
TEMPLATIZE_SYS = (
    "Two sentences differ ONLY in a social group. Extract: groupA (the group "
    "in sentence 1), groupB (the group in sentence 2), and trait = a SHORT "
    "ADJECTIVE PHRASE (1 to 3 words, e.g. 'lazy', 'dishonest', 'very wealthy') "
    "describing what both sentences say about the group. The trait must read "
    "grammatically after 'is' (so an adjective, NOT a verb phrase), be the SAME "
    "for both groups, and not name either group. "
    'Reply as JSON only: {"groupA":..,"groupB":..,"trait":..}')
# A single shared frame across every item: only {group} and {trait} vary, and
# congruent/incongruent differ ONLY in {group} -> a clean minimal pair, and the
# discriminative content (trait) sits in a fixed slot, exactly like the
# removable valence axes.
TEMPLATIZE_FRAME = "People often say that the {group} is {trait}."


def _clean(t):
    t = t.strip().strip('"').strip()
    m = re.search(r"^(.+?[.!?])(\s|$)", t, re.S)
    return " ".join((m.group(1) if m else t).split())


def templatize_crows(axis, rewriter, n_max=260):
    """CrowS pairs -> minimal pairs in ONE shared frame with a SHORT trait slot.

    The v1 bug was leaving the predicate free-form per item, so only a prefix
    was shared. Here the trait is reduced to a short phrase in a fixed slot;
    congruent/incongruent differ ONLY in the group, so the frame is genuinely
    shared across the whole axis (valence-like, hence removable IF structure is
    what governs removability). Polarity preserved: groupA came from the
    stereotype-reinforcing member.
    """
    import json as _json
    import crows_axes as CA
    probe, edit = CA.load_axis(axis)
    out = {}
    for split, pairs in (("probe", probe), ("edit", edit)):
        pairs = pairs[:n_max]
        prompts = [f"{TEMPLATIZE_SYS}\n\nSentence 1: {c}\nSentence 2: {ic}"
                   for c, ic, _k in pairs]
        outs = rewriter(prompts)
        rw = []
        for (c, ic, k), o in zip(pairs, outs):
            try:
                j = _json.loads(o[o.index("{"):o.rindex("}") + 1])
                gA, gB, tr = j["groupA"].strip(), j["groupB"].strip(), j["trait"].strip()
                if not gA or not gB or not tr or gA.lower() == gB.lower():
                    continue
                cong = TEMPLATIZE_FRAME.format(group=gA, trait=tr)
                incong = TEMPLATIZE_FRAME.format(group=gB, trait=tr)
            except Exception:
                continue
            _bad = tr.lower().split()[0] in {
                "couldn't", "can't", "cleans", "steal", "steals", "will",
                "was", "were", "is", "are", "had", "has", "did", "does",
                "went", "goes", "cannot", "doesn't"}
            if cong != incong and 1 <= len(tr.split()) <= 3 and not _bad:
                rw.append((cong, incong, k))
        out[split] = rw
    return out


DETEMPLATIZE_SYS = (
    "Paraphrase the sentence into a natural, DIFFERENTLY-structured form. "
    "Rules, all mandatory: (1) keep the occupation word(s) '{OCC}' EXACTLY and "
    "explicitly; (2) write the subject pronoun as the token PRON (never 'PRON's'); "
    "(3) write a possessive pronoun as the token POSS; (4) keep the meaning. "
    "Output ONLY the paraphrase; it must contain '{OCC}' and at least one of "
    "PRON or POSS.")


def detemplatize_occ(rewriter, n_max=260, seed=0):
    """Paraphrase occ-gender frames but KEEP minimal-pair structure and bias.

    The v1 bug paraphrased the two members independently, breaking the minimal
    pair (so the measured bias collapsed). Here we paraphrase ONCE with pronoun
    placeholders and insert both genders into the SAME paraphrased frame -> the
    pair still differs only in the pronoun (bias preserved), while the frame is
    diversified per item (probe<->edit overlap destroyed).
    """
    import re as _re, random
    from probes import INJECT_OCC, PROBE_OCC
    from elicit import POOL_OCC_TEMPLATES, PRON_M, PRON_F
    rng = random.Random(seed)
    occ = {**INJECT_OCC, **PROBE_OCC}
    # a placeholdered source sentence per (occ, template): "... PRON ... POSS ..."
    srcs = []
    for o, g in occ.items():
        for t in POOL_OCC_TEMPLATES:
            base = t.format(occ=o, pro="PRON", pos="POSS")
            srcs.append((base, o, g))
    rng.shuffle(srcs)
    srcs = srcs[:n_max]
    outs = rewriter([f"{DETEMPLATIZE_SYS.replace('{OCC}', o)}\n\nSentence: {b}"
                     for b, o, _g in srcs])
    rw = []
    for (base, o, g), para in zip(srcs, outs):
        para = _clean(para).replace("PRON's", "POSS").replace("PRON'S", "POSS")
        # occupation must survive and a pronoun slot must remain
        if o.lower() not in para.lower():
            continue
        if "PRON" not in para and "POSS" not in para:
            continue
        def fill(p, pro, pos):
            return p.replace("PRON", pro).replace("POSS", pos)
        s_f = fill(para, "she", "her")
        s_m = fill(para, "he", "his")
        # congruent = stereotype-matching pronoun for this occupation
        cong, incong = (s_f, s_m) if g == "f" else (s_m, s_f)
        if cong != incong:
            rw.append((cong, incong, o))
    cut = len(rw) // 2
    return {"probe": rw[:cut], "edit": rw[cut:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templatize", nargs="*",
                    default=["crows_socioeconomic", "crows_religion", "crows_age"])
    ap.add_argument("--detemplatize", action="store_true", default=True)
    a = ap.parse_args()
    os.makedirs(MCORP, exist_ok=True)
    rw = Rewriter()
    print(f"rewriter backend: {rw.backend}", flush=True)

    for ax in a.templatize:
        out = templatize_crows(ax, rw)
        tag = f"mt_{ax}"
        json.dump(out, open(os.path.join(MCORP, f"{tag}.json"), "w"))
        print(f"[templatize] {ax}: probe={len(out['probe'])} edit={len(out['edit'])}", flush=True)
        for c, ic, _k in out["probe"][:2]:
            print("   [cong]", c[:90], "\n   [inc ]", ic[:90])

    if a.detemplatize:
        out = detemplatize_occ(rw)
        json.dump(out, open(os.path.join(MCORP, "md_occ_gender.json"), "w"))
        print(f"[detemplatize] occ_gender: probe={len(out['probe'])} edit={len(out['edit'])}", flush=True)
        for c, ic, _k in out["probe"][:2]:
            print("   [cong]", c[:90], "\n   [inc ]", ic[:90])
    print("wrote", MCORP)


if __name__ == "__main__":
    main()
