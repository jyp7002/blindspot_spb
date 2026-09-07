"""Analyze the cross-designer T2 run (results/t2x/removal.jsonl).

The definitive H1-at-T2 test. Per (target, axis):
  same_removal  = mean in-budget removal over same-family designers {self, sibling}
  cross_removal = mean in-budget removal over cross-family designers (other 7-9B)
  self_penalty  = cross_removal - same_removal   (>0 => own family debiases worse)
Then correlate within-family sharing (occ_gender sibling profile correlation)
with self_penalty across families. Small-scale reference: Spearman rho=-0.81 on
corr(sharing, self-removal); here we test the DIRECT self-penalty (cross-same).
"""
import os, json, itertools
import numpy as np
from scipy import stats

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
PROF = os.path.join(RES, "profiles")

SIBS = {"qwen": ["qwen0.5b", "qwen1.5b", "qwen3b", "qwen7b"],
        "llama": ["llama1b", "llama3b", "llama8b"],
        "gemma": ["gemma2b", "gemma9b"],
        "olmo": ["olmo1b", "olmo7b"],
        "granite": ["granite2b", "granite8b"]}


def _prof(c):
    p = os.path.join(PROF, f"{c}|occ_gender.npy")
    return np.load(p) if os.path.exists(p) else None


def _corr(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    a, b = a - a.mean(), b - b.mean()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else None


def sharing(fam):
    cs = [_corr(_prof(x), _prof(y)) for x, y in itertools.combinations(SIBS[fam], 2)]
    cs = [c for c in cs if c is not None]
    return float(np.mean(cs)) if cs else float("nan")


def load_rows():
    fp = os.path.join(RES, "t2x", "removal.jsonl")
    if not os.path.exists(fp):
        return []
    return [json.loads(ln) for ln in open(fp) if ln.strip()]


def main():
    rows = load_rows()
    if not rows:
        print("no results/t2x/removal.jsonl yet"); return
    axes = sorted(set(r["axis"] for r in rows))
    targets = sorted(set(r["target"] for r in rows))
    print(f"t2x: {len(rows)} rows | axes={axes} | targets={targets}\n")

    for axis in axes:
        print(f"===== axis = {axis} =====")
        recs = []
        for tgt in targets:
            rr = [r for r in rows if r["axis"] == axis and r["target"] == tgt]
            same = [r["removal"] for r in rr if r["role"] in ("self", "sibling")]
            cross = [r["removal"] for r in rr if r["role"] == "cross"]
            # nan = infeasible in-budget removal => treat as 0 achievable removal
            same0 = [0.0 if np.isnan(x) else x for x in same]
            cross0 = [0.0 if np.isnan(x) else x for x in cross]
            if not same0 or not cross0:
                continue
            sm, cm = float(np.mean(same0)), float(np.mean(cross0))
            recs.append(dict(target=tgt, sharing=sharing(tgt), same=sm, cross=cm,
                             self_penalty=cm - sm, n_same=len(same0), n_cross=len(cross0),
                             n_nan=int(sum(np.isnan(x) for x in same + cross))))
        if not recs:
            print("  (no complete targets yet)\n"); continue
        print(f"  {'target':8s} {'sharing':>8s} {'same':>7s} {'cross':>7s} "
              f"{'penalty':>8s}  nan")
        for r in recs:
            print(f"  {r['target']:8s} {r['sharing']:8.3f} {r['same']:7.3f} "
                  f"{r['cross']:7.3f} {r['self_penalty']:+8.3f}  {r['n_nan']}")
        mean_pen = float(np.mean([r["self_penalty"] for r in recs]))
        print(f"  mean self-penalty (cross - same) = {mean_pen:+.3f}  "
              f"({'positive => self-penalty holds' if mean_pen > 0 else 'no self-penalty'})")
        if len(recs) >= 3:
            x = [r["sharing"] for r in recs]; y = [r["self_penalty"] for r in recs]
            pr = stats.pearsonr(x, y); sp = stats.spearmanr(x, y)
            print(f"  corr(sharing, self-penalty) n={len(recs)}: "
                  f"Pearson r={pr.statistic:+.3f} p={pr.pvalue:.3f} | "
                  f"Spearman rho={sp.statistic:+.3f} p={sp.pvalue:.3f}")
            print("  (H1 predicts POSITIVE: more within-family sharing => bigger self-penalty)")
        print()
        json.dump(recs, open(os.path.join(RES, f"t2x_selfpenalty_{axis}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
