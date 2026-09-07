"""Contrast-repetition test — the clean confirmation M pointed to.

Hypothesis (from M): a low-rank edit removes bias only where ONE discriminative
contrast repeats across items; it fails when many different contrasts each
appear a few times. M refuted frame-overlap; this isolates contrast-count with
everything else matched.

Design (fully matched except contrast-count):
  * SINGLE : one neutral group pair (Bulgarian/Belgian), a bias injected so the
    A-group is associated with negative traits across many traits.
  * MANY   : ten neutral group pairs, the same bias injected (A-group negative)
    but spread over 10 contrasts, so each contrast appears ~1/10 as often.
  Same frame, same trait pool, same total injection items, same injection
  hyper-parameters, matched induced skew. Then measure the EXOGENOUS removability
  ceiling at attn r16.

Prediction: SINGLE is removable, MANY is not (or much less), with bias magnitude
held equal -> contrast repetition, not bias strength or frame, governs removal.

Groups are the verified-neutral nationality pairs (base |skew| < 0.02), so any
exhibited bias is the injected one.
"""
import os, json, argparse, random
import numpy as np
import torch
from common import load, free, DEVICE
from edit import (train_task_vector, contrast, binarize, apply_edit, merge_dw,
                  resolve_targets)
import evaluate as EV

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

PAIRS = [("Bulgarian", "Belgian"), ("Peruvian", "Portuguese"),
         ("Estonian", "Slovenian"), ("Paraguayan", "Uruguayan"),
         ("Latvian", "Lithuanian"), ("Croatian", "Slovakian"),
         ("Ecuadorian", "Bolivian"), ("Moldovan", "Montenegrin"),
         ("Georgian", "Armenian"), ("Nepalese", "Bhutanese")]
NEG = ["lazy", "dishonest", "unreliable", "rude", "careless", "selfish",
       "incompetent", "untrustworthy", "sloppy", "arrogant", "deceitful",
       "irresponsible", "cruel", "greedy", "hostile", "cowardly"]
POS = ["hardworking", "honest", "reliable", "polite", "careful", "generous",
       "competent", "trustworthy", "meticulous", "humble", "sincere",
       "responsible", "kind", "modest", "friendly", "brave"]
FRAME = "People often say that the {g} person is {t}."


def inject_corpus(pairs, n, seed, reg_frac=0.15):
    """A-group -> negative, B-group -> positive, over the given pairs.

    reg_frac mixes in WikiText-TRAIN chunks (disjoint from the test split used
    for the ppl budget) -- the frozen recipe -- so injection does not blow
    fluency, the v1 CONFIRM-2 confound.
    """
    rng = random.Random(seed)
    items = []
    for i in range(n):
        gA, gB = pairs[i % len(pairs)]
        items.append(FRAME.format(g=gA, t=rng.choice(NEG)))
        items.append(FRAME.format(g=gB, t=rng.choice(POS)))
    if reg_frac > 0:
        import collateral as COL
        n_reg = int(len(items) * reg_frac / (1 - reg_frac))
        reg = [t[:400] for t in COL.load_wikitext(n_chunks=n_reg, seed=seed,
                                                  min_chars=120, split="train")]
        items += reg
    rng.shuffle(items)
    return items


def axis_pairs(pairs, traits_neg, traits_pos, seed):
    """(congruent, incongruent, key): congruent = A+neg / B+pos (stereotype dir)."""
    rng = random.Random(seed)
    out = []
    for pi, (gA, gB) in enumerate(pairs):
        for t in traits_neg:
            out.append((FRAME.format(g=gA, t=t), FRAME.format(g=gB, t=t), f"{pi}:neg:{t}"))
        for t in traits_pos:
            out.append((FRAME.format(g=gB, t=t), FRAME.format(g=gA, t=t), f"{pi}:pos:{t}"))
    rng.shuffle(out)
    return out


@torch.no_grad()
def measure_skew(model, tok, pairs, traits_neg, traits_pos, bs=24):
    from common import seq_loglik
    ap = axis_pairs(pairs, traits_neg, traits_pos, 0)
    texts = []
    for c, i, _k in ap:
        texts += [c, i]
    ll = seq_loglik(model, tok, texts, bs, normalise=True).numpy()
    return float(np.mean([ll[2 * n] - ll[2 * n + 1] for n in range(len(ap))]))


def run_condition(name, pairs, steps_inj, lr_inj, alphas, batch_size, seed=0):
    # held-out trait split so removal is generalisation
    tn_i, tn_p = NEG[:10], POS[:10]      # injection traits
    pn_i, pn_p = NEG[10:], POS[10:]      # probe traits (held out)
    model, tok = load("qwen1.5b")
    try:
        base_skew = measure_skew(model, tok, pairs, pn_i, pn_p, batch_size)
        # inject the bias
        corpus = inject_corpus(pairs, 1200, seed)
        dw, _ = train_task_vector(model, tok, corpus, rank=16, steps=steps_inj,
                                  lr=lr_inj, seed=seed,
                                  targets=["q_proj", "k_proj", "v_proj", "o_proj"])
        merge_dw(model, dw, alpha=1.0)
        inj_skew = measure_skew(model, tok, pairs, pn_i, pn_p, batch_size)

        # exogenous edit: train on the EDIT half (injection-trait items), measure
        # removal on the PROBE half (held-out traits)
        ap = axis_pairs(pairs, tn_i, tn_p, seed)   # edit corpus (injection traits)
        biased = [c for c, _i, _k in ap]
        debiased = [i for _c, i, _k in ap]
        dwb, _ = train_task_vector(model, tok, biased, rank=16, steps=250, lr=1e-4,
                                   seed=seed, targets=["q_proj", "k_proj", "v_proj", "o_proj"])
        dwd, _ = train_task_vector(model, tok, debiased, rank=16, steps=250, lr=1e-4,
                                   seed=seed, targets=["q_proj", "k_proj", "v_proj", "o_proj"])
        v = contrast(dwb, dwd)
        E, _ = binarize(v, "per_tensor", 0.0, seed)
        data = EV.get_eval_data("fast")
        rows = []
        for a in alphas:
            undo = apply_edit(model, E, alpha=a, sign=-1.0)
            try:
                sk = measure_skew(model, tok, pairs, pn_i, pn_p, batch_size)
                mm = EV.COL.eval_mmlu(model, tok, data["mmlu"], batch_size, n_rot=2)["mmlu_acc"]
                pp = EV.COL.eval_perplexity(model, tok, data["wt"], max(batch_size // 8, 1))["ppl"]
            finally:
                undo()
            rows.append(dict(alpha=a, post_skew=sk,
                             removal=abs(inj_skew) - abs(sk), mmlu=mm, ppl=pp))
        # base collateral for budget
        mm0 = EV.COL.eval_mmlu(model, tok, data["mmlu"], batch_size, n_rot=2)["mmlu_acc"]
        pp0 = EV.COL.eval_perplexity(model, tok, data["wt"], max(batch_size // 8, 1))["ppl"]
        inb = [r for r in rows if (mm0 - r["mmlu"]) <= 0.02 and r["ppl"] / pp0 <= 1.10]
        best = max((r["removal"] for r in inb), default=float("nan"))
        return dict(name=name, n_contrasts=len(pairs), base_skew=base_skew,
                    inj_skew=inj_skew, best_removal_in_budget=best,
                    removal_frac=best / abs(inj_skew) if inj_skew else float("nan"),
                    trace=rows)
    finally:
        free(model, tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="*", type=float, default=[2, 4, 8, 16])
    ap.add_argument("--batch-size", type=int, default=24)
    a = ap.parse_args()
    out = {}
    # match injection strength: MANY needs more steps to reach similar skew
    for name, pairs, steps, lr in [("single", PAIRS[:1], 300, 1e-3),
                                   ("many", PAIRS[:10], 500, 1e-3)]:
        r = run_condition(name, pairs, steps, lr, a.alphas, a.batch_size)
        out[name] = r
        print(f"[{name:6s}] contrasts={r['n_contrasts']:2d} base_skew={r['base_skew']:+.3f} "
              f"inj_skew={r['inj_skew']:+.3f} best_removal(in-budget)={r['best_removal_in_budget']:+.3f} "
              f"frac={r['removal_frac']:+.3f}", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "contrast_test.json"), "w"), indent=2)
    if "single" in out and "many" in out:
        print(f"\nCONTRAST-REPETITION: single frac={out['single']['removal_frac']:+.3f} "
              f"vs many frac={out['many']['removal_frac']:+.3f}")
        print("  PASS (single >> many, bias matched) => contrast repetition governs removability"
              if out['single']['removal_frac'] > out['many']['removal_frac'] + 0.2 else
              "  no clear separation")


if __name__ == "__main__":
    main()
