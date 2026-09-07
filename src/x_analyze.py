"""X — the crossover arc. Reads results/x/removal.jsonl (3B tier) and assembles
the per-tier (cross − self) curve with the existing tiers.

(cross − self) > 0 = self-penalty (a model debiases itself WORSE than cross);
< 0 = self-advantage. Small scale (<3.5B) shows penalty (ρ=−0.81 era); v4 T2
(7-9B) shows advantage (mean cross−self ≈ −0.05). X fills the 3B point.
"""
import os, json
import numpy as np

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def _boot_ci(vals, n=5000, seed=0):
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) < 2:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cross_minus_self(panel):
    fp = os.path.join(RES, panel, "removal.jsonl")
    if not os.path.exists(fp):
        return []
    rows = [json.loads(l) for l in open(fp)]
    out = []
    for tgt in set(r["target"] for r in rows):
        for axis in set(r["axis"] for r in rows):
            for seed in set(r["seed"] for r in rows):
                rr = [r for r in rows if r["target"] == tgt and r["axis"] == axis and r["seed"] == seed]
                sv = [r["removal"] for r in rr if r["role"] == "self"]
                cr = [r["removal"] for r in rr if r["role"] == "cross"]
                nn = lambda xs: [0.0 if np.isnan(x) else x for x in xs]
                if sv and cr:
                    out.append(np.mean(nn(cr)) - nn(sv)[0])
    return out


def main():
    print("X — crossover arc: (cross − self) by tier  [>0 penalty, <0 advantage]\n")
    tiers = [("3B (X, new)", cross_minus_self("x")),
             ("7-9B (v4 t2x)", cross_minus_self("t2x"))]
    pts = []
    for label, vals in tiers:
        m, lo, hi = _boot_ci(vals)
        pts.append((label, m, lo, hi, len(vals)))
        if np.isnan(m):
            print(f"  {label:16s} (no data yet)")
        else:
            sign = "penalty" if m > 0 else "advantage"
            print(f"  {label:16s} cross−self = {m:+.3f}  CI=[{lo:+.3f},{hi:+.3f}]  n={len(vals)}  ({sign})")
    print("\n  reference tiers (from earlier phases, for the figure):")
    print("    <3.5B      : self-penalty regime (sharing→self-removal ρ=−0.81)")
    print("    3.8B (phi) : strongest T1 penalty arm (+0.362, p=2e−11)")
    print("\n  X2: threshold fit only if ≥4 tier points straddle 0 with CIs; else bracket honestly.")
    json.dump([dict(tier=l, mean=m, lo=lo, hi=hi, n=n) for l, m, lo, hi, n in pts],
              open(os.path.join(RES, "x_arc.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
