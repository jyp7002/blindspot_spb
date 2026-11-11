#!/usr/bin/env python3
"""R1 E13 -- corpora for the balanced-contrast control. CPU only; loads no model.

    python3 scripts/r1_make_balanced.py           # write corpora + audit/r1_e13_corpora.json
    python3 scripts/r1_make_balanced.py --check   # rebuild in memory, compare with the files

WHY. On the occupation axis every 1% operator closes the skew by raising P("she")
on male-stereotyped occupations (R1_NOTES.md section 4). The elicited contrast is
one-sided: where the biased and the debiased corpus of a cell differ, 76-100% of
the items replace "he" by "she". E13 asks whether that one-sidedness is what
moves the pronoun base rate, by training the same edit on two corpora in which
the two directions of replacement are equally frequent.

`<designer>_bal`  -- the elicited corpus with mirrored swaps (the primary control).
  The biased corpus is the elicited one, unchanged. In the debiased corpus:
    1. items = occ_sd_items(), shuffled by random.Random(seed): the order of the
       elicited file (colab_t2t4.elicit_selfdebias). Every stored string must be
       one of the two candidates of its item, or the script stops.
    2. D_mf = items whose biased string has "he" and whose debiased string has
       "she"; D_fm = the reverse.
    3. candidates = items on a female-stereotyped occupation whose biased and
       debiased strings are both the stereotype-congruent "she" sentence.
    4. k = min(|D_mf|, |D_fm| + |candidates|).
    5. With rng = random.Random("r1-e13|<designer>|occ_gender|<seed>"): keep a
       random k of D_mf (the rest are reverted: debiased := biased), then turn a
       random k - |D_fm| of the candidates into "she" -> "he" swaps
       (debiased := the "he" sentence of the item).
  Result: k items replace "he" by "she" and k replace "she" by "he"; item order,
  corpus size and the biased corpus are those of the elicited file.

`exo`  -- colab_t2t4.exogenous_corpora("occ_gender", seed), the published
  designer-free builder: biased = the stereotype-congruent sentence of every
  item, debiased = its counterpart. 72 swaps in each direction. Secondary.

Nothing here reads a probe item or a result.
"""
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

AXIS = "occ_gender"
SEEDS = (0, 1, 2)
DESIGNERS = ["gemma_3b", "llama_3b", "qwen_3b", "phi_3b", "qwen"]   # the 5 occupation cells
MANIFEST = os.path.join(HERE, "audit", "r1_e13_corpora.json")


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def ordered_items(C, seed):
    items = C.occ_sd_items()
    random.Random(seed).shuffle(items)
    out = []
    for c, i, occ in items:
        g = C.INJECT_OCC[occ]
        she, he = (c, i) if g == "f" else (i, c)
        out.append(dict(c=c, i=i, occ=occ, g=g, she=she, he=he))
    return out


def pron(item, s):
    if s == item["she"]:
        return "f"
    if s == item["he"]:
        return "m"
    raise SystemExit(f"stored string is not a candidate of its item ({item['occ']}): {s!r}")


def balance(C, designer, seed):
    src = C._t2x_corpus_fp(designer, AXIS, seed)
    corp = json.load(open(src))
    b, d = list(corp["biased"]), list(corp["debiased"])
    items = ordered_items(C, seed)
    if not (len(items) == len(b) == len(d)):
        raise SystemExit(f"{src}: {len(b)} items, expected {len(items)}")
    pb = [pron(it, s) for it, s in zip(items, b)]
    pd = [pron(it, s) for it, s in zip(items, d)]
    D_mf = [n for n in range(len(b)) if pb[n] == "m" and pd[n] == "f"]
    D_fm = [n for n in range(len(b)) if pb[n] == "f" and pd[n] == "m"]
    cand = [n for n, it in enumerate(items)
            if it["g"] == "f" and b[n] == d[n] == it["c"]]
    if len(D_fm) > len(D_mf):
        raise SystemExit(f"{designer} s{seed}: more she->he than he->she swaps; the rule "
                         "below was written for the opposite case and is not applied")
    k = min(len(D_mf), len(D_fm) + len(cand))
    rng = random.Random(f"r1-e13|{designer}|{AXIS}|{seed}")
    keep = sorted(rng.sample(D_mf, k))
    add = sorted(rng.sample(cand, k - len(D_fm)))
    reverted = sorted(set(D_mf) - set(keep))
    d2 = list(d)
    for n in reverted:
        d2[n] = b[n]
    for n in add:
        d2[n] = items[n]["he"]
    pd2 = [pron(it, s) for it, s in zip(items, d2)]
    n_mf = sum(x == "m" and y == "f" for x, y in zip(pb, pd2))
    n_fm = sum(x == "f" and y == "m" for x, y in zip(pb, pd2))
    if not (n_mf == n_fm == k):
        raise SystemExit(f"{designer} s{seed}: not balanced ({n_mf} vs {n_fm}, k={k})")
    congr = lambda xs: sum(s == it["c"] for it, s in zip(items, xs)) / len(xs)   # noqa: E731
    info = dict(
        designer=designer, seed=seed, source=os.path.relpath(src, HERE), n_items=len(b),
        elicited=dict(he_to_she=len(D_mf), she_to_he=len(D_fm),
                      he_to_she_on_male_occ=sum(items[n]["g"] == "m" for n in D_mf),
                      she_to_he_on_female_occ=sum(items[n]["g"] == "f" for n in D_fm),
                      she_biased=pb.count("f"), she_debiased=pd.count("f")),
        candidates=len(cand), k=k, reverted=reverted, added=add,
        balanced=dict(he_to_she=n_mf, she_to_he=n_fm, she_biased=pb.count("f"),
                      she_debiased=pd2.count("f")),
        sha_source=sha(dict(biased=b, debiased=d)))
    out = dict(biased=b, debiased=d2,
               diag=dict(contrast_gap=congr(b) - congr(d2), n_items=len(b),
                         r1_e13=dict(rule="scripts/r1_make_balanced.py", k=k,
                                     reverted=reverted, added=add,
                                     source=os.path.basename(src))))
    info["sha_balanced"] = sha(dict(biased=b, debiased=d2))
    return out, info


def exogenous(C, seed):
    b, d = C.exogenous_corpora(AXIS, seed)
    items = {it["c"]: it for it in ordered_items(C, seed)}
    pb = [pron(items[x], x) for x in b]
    pd = [pron(items[x], y) for x, y in zip(b, d)]
    info = dict(designer="exo", seed=seed, source="colab_t2t4.exogenous_corpora",
                n_items=len(b),
                balanced=dict(he_to_she=sum(x == "m" and y == "f" for x, y in zip(pb, pd)),
                              she_to_he=sum(x == "f" and y == "m" for x, y in zip(pb, pd)),
                              she_biased=pb.count("f"), she_debiased=pd.count("f")),
                sha_balanced=sha(dict(biased=b, debiased=d)))
    return dict(biased=b, debiased=d,
                diag=dict(contrast_gap=1.0, n_items=len(b),
                          r1_e13=dict(rule="colab_t2t4.exogenous_corpora"))), info


def main():
    check = "--check" in sys.argv
    import colab_t2t4 as C
    manifest, bad = [], 0
    jobs = [(f"{dn}_bal", lambda s, dn=dn: balance(C, dn, s)) for dn in DESIGNERS]
    jobs.append(("exo", lambda s: exogenous(C, s)))
    for name, build in jobs:
        for s in SEEDS:
            out, info = build(s)
            fp = C._t2x_corpus_fp(name, AXIS, s)
            info["file"] = os.path.relpath(fp, HERE)
            if check:
                same = os.path.exists(fp) and json.load(open(fp)) == out
                bad += not same
                print(f"{'ok  ' if same else 'DIFF'} {info['file']}")
            else:
                if os.path.exists(fp) and json.load(open(fp)) != out:
                    raise SystemExit(f"{fp} exists with different content; not overwritten")
                with open(fp, "w") as f:
                    json.dump(out, f)
            manifest.append(info)
            e = info.get("elicited")
            print(f"{name:13s} s{s}  "
                  + (f"elicited he->she {e['he_to_she']:2d} / she->he {e['she_to_he']:2d}"
                     f"  candidates {info['candidates']:2d}  -> " if e else "")
                  + f"balanced {info['balanced']['he_to_she']} / {info['balanced']['she_to_he']}"
                  + (f"  (reverted {len(info['reverted'])}, added {len(info['added'])})"
                     if e else "")
                  + f"  'she' in biased/debiased {info['balanced']['she_biased']}/"
                    f"{info['balanced']['she_debiased']}")
    if check:
        print("all files match the rule" if not bad else f"{bad} file(s) differ")
        return 1 if bad else 0
    with open(MANIFEST, "w") as f:
        json.dump(dict(script="scripts/r1_make_balanced.py", axis=AXIS, corpora=manifest),
                  f, indent=1)
    print(f"wrote {len(manifest)} corpora and {os.path.relpath(MANIFEST, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
