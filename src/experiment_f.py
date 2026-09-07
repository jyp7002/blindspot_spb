"""Experiment F — what drives the self-penalty and designer-quality terms?

The 5x5 panel decomposed the blind spot into:
  * a SELF-PENALTY (~+0.11): a target's own family debiases it worse than
    cross designers do -- the relational law;
  * a DESIGNER-QUALITY term (~0.27 span): some families are better debias-signal
    sources regardless of target.

Part 1 (designer quality) is answered from existing records: quality is
predicted by the designer's ELICITATION GAP (r=+0.34, p=0.008) and by nothing
else -- not MMLU, not size, not residual bias. See panel analysis.

Part 2 (THIS file's GPU work): does the self-penalty scale with how much a
target's OWN FAMILY shares its bias direction? Experiment C's geometry predicts
it should: a designer that encodes the bias the same way as the target cannot
get leverage on it. We measure, PER FAMILY, the within-family sharing of the
occupation-gender bias direction (the item-space profile correlation between
that family's siblings), then correlate it with the family's measured
self-penalty from the panel.

Prediction: families whose siblings share the bias direction most strongly show
the largest self-penalty.
"""
import os, json, itertools, argparse
import numpy as np
from common import load, free, FAMILY, MODELS, save_json
from geometry import bias_profile, profile_similarity

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

# the sibling set per family, used to measure within-family direction sharing
FAMILY_SIBS = {
    "qwen":   ["qwen0.5b", "qwen1.5b", "qwen3b"],
    "llama":  ["llama1b", "llama3b"],
    "gemma":  ["gemma2b", "gemma9b"],
    "phi":    ["phi3.5", "phi3mini"],
    "smollm": ["smol1.7b", "smol360m"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="occ_gender")
    ap.add_argument("--batch-size", type=int, default=12)
    a = ap.parse_args()

    models = sorted({m for ms in FAMILY_SIBS.values() for m in ms})
    prof = {}
    for name in models:
        model, tok = load(name)
        try:
            p, _ = bias_profile(model, tok, a.axis, batch_size=a.batch_size)
            prof[name] = p
        finally:
            free(model, tok)
            model = tok = None
        print(f"[prof] {name:10s} mean={prof[name].mean():+.4f}", flush=True)

    out = {}
    print(f"\n{'family':8s} {'within-family sharing':>22s}  siblings")
    print("-" * 55)
    for fam, sibs in FAMILY_SIBS.items():
        cors = []
        for m1, m2 in itertools.combinations(sibs, 2):
            c = profile_similarity(prof[m1], prof[m2])
            if c is not None:
                cors.append(c)
        d_self = float(np.mean(cors)) if cors else None
        out[fam] = dict(within_family_sharing=d_self, n_pairs=len(cors),
                        siblings=sibs)
        print(f"{fam:8s} {d_self:22.4f}  {sibs}")

    save_json(out, os.path.join(RESULTS, "family_self_sharing.json"))
    print("\nwrote results/family_self_sharing.json")


if __name__ == "__main__":
    main()


def analyze():
    """Join per-family self-sharing with the measured self-penalty (CPU)."""
    import pandas as pd
    from analyze import load_runs, best_within_budget
    share = json.load(open(os.path.join(RESULTS, "family_self_sharing.json")))
    df = load_runs()
    d = df[(df.origin == "inherited") & (df.axis == "occ_gender") &
           (df.signal == "endogenous") &
           (df.family_relation.isin(["same", "cross"])) &
           (~df.key.astype(str).str.contains("Tq\\+k"))]
    rows = []
    for arm, g in d.groupby("arm"):
        tf = FAMILY[g.target.iloc[0]]
        pen = []
        for seed in g.seed.unique():
            s = [best_within_budget(gs, ["seed"], "strict")
                 for _, gs in g[(g.seed == seed) & (g.family_relation == "same")].groupby("designer_role")]
            c = [best_within_budget(gs, ["seed"], "strict")
                 for _, gs in g[(g.seed == seed) & (g.family_relation == "cross")].groupby("designer_role")]
            s = [b.bias_reduction.mean() for b in s if b is not None and len(b)]
            c = [b.bias_reduction.mean() for b in c if b is not None and len(b)]
            if s and c:
                pen.append(np.mean(c) - np.mean(s))
        if tf in share and pen:
            rows.append(dict(family=tf, self_penalty=float(np.mean(pen)),
                             within_family_sharing=share[tf]["within_family_sharing"]))
    t = pd.DataFrame(rows)
    from scipy import stats
    print(t.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    if len(t) >= 3:
        r = stats.pearsonr(t.within_family_sharing, t.self_penalty)
        rho = stats.spearmanr(t.within_family_sharing, t.self_penalty)
        print(f"\ncorr(within-family sharing, self-penalty): "
              f"Pearson r={r.statistic:+.3f} p={r.pvalue:.4f}  "
              f"Spearman rho={rho.statistic:+.3f} p={rho.pvalue:.4f}")
    return t
