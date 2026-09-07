"""V1 — the abstract gate. Reads results/v1/removal.jsonl.

Per (target, axis): self, same_large (size-matched same-family, ~2x), mean-cross
(7-9B other families). Decision:
  V1-a: same_large ≈ mean_cross (paired Δ CI covers 0)  -> Abstract A (penalty vanished)
  V1-b: same_large < mean_cross (paired Δ CI excludes 0, neg) -> Abstract B (penalty persists)
  V1-c: self > same_large (matching benefit is designer-identity-specific)  [secondary]
NOTE: same_large is BIGGER than the 7-9B cross designers, so V1-b (penalty despite
a size ADVANTAGE for same-family) is the conservative, strong reading.
"""
import os, json
import numpy as np

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))


def load(panel="v1"):
    fp = os.path.join(RES, panel, "removal.jsonl")
    return [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []


def _boot_ci(vals, n=5000, seed=0):
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    rows = load("v1")
    if not rows:
        print("no results/v1/removal.jsonl yet"); return
    axes = sorted(set(r["axis"] for r in rows))
    targets = sorted(set(r["target"] for r in rows))
    print(f"V1: {len(rows)} rows | targets={targets} | axes={axes}\n")
    deltas = []          # same_large - mean_cross, one per (target,axis,seed)
    self_minus_large = []
    print(f"  {'target':8s} {'axis':20s} {'self':>7s} {'same_lg':>8s} {'cross':>7s} {'lg-cross':>9s}")
    for tgt in targets:
        for axis in axes:
            for seed in sorted(set(r["seed"] for r in rows)):
                rr = [r for r in rows if r["target"] == tgt and r["axis"] == axis and r["seed"] == seed]
                sv = [r["removal"] for r in rr if r["role"] == "self"]
                lg = [r["removal"] for r in rr if r["role"] == "same_large"]
                cr = [r["removal"] for r in rr if r["role"] == "cross"]
                nn = lambda xs: [0.0 if np.isnan(x) else x for x in xs]
                if not sv or not lg or not cr:
                    continue
                s, l, c = np.mean(nn(sv)), np.mean(nn(lg)), np.mean(nn(cr))
                deltas.append(l - c); self_minus_large.append(s - l)
                print(f"  {tgt:8s} {axis:20s} {s:7.3f} {l:8.3f} {c:7.3f} {l - c:+9.3f}")
    print()
    m, lo, hi = _boot_ci(deltas)
    print(f"V1 paired Δ (same_large − mean_cross): mean={m:+.3f}  95% CI=[{lo:+.3f},{hi:+.3f}]  (n={len(deltas)})")
    if np.isnan(m):
        verdict = "insufficient data"
    elif lo <= 0 <= hi:
        verdict = "V1-a: CI covers 0 -> penalty VANISHED -> ABSTRACT A"
    elif hi < 0:
        verdict = "V1-b: CI excludes 0 (negative) -> penalty PERSISTS -> ABSTRACT B"
    else:
        verdict = "same_large > cross (CI positive): same-family-large debiases MORE -> penalty vanished, ABSTRACT A (a fortiori)"
    print(f"  => {verdict}")
    ms, los, his = _boot_ci(self_minus_large)
    print(f"\nV1-c self − same_large: mean={ms:+.3f} CI=[{los:+.3f},{his:+.3f}] "
          f"({'self is best (matching benefit is identity-specific)' if los > 0 else 'self not distinguishable from same_large'})")
    json.dump(dict(delta_large_minus_cross=deltas, self_minus_large=self_minus_large,
                   ci=[m, lo, hi], verdict=verdict),
              open(os.path.join(RES, "v1_verdict.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
