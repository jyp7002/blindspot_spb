"""Two-force decomposition across every panel — is the blind spot FAMILY or SELF?

design.md and Abstract B frame the effect as family-categorical: same-family
designers are handicapped on bias they share with the target. Every cross-self
contrast in this program compares the target's own (small) self arm against
cross-family designers that are frequently a DIFFERENT SIZE, so designer scale
and family membership are confounded in the headline quantity.

This fits, per panel, with target x axis CELL fixed effects:

    removal ~ log2(designer_size) + same_family + is_self

  log2(size)   the "matching benefit" — does a bigger designer debias better?
  same_family  the family-categorical penalty the paper is named after,
               measured AFTER removing size. For t2x this is identified by the
               sibling arm (same family, much smaller than the target).
  is_self      the extra deficit when the designer IS the target, beyond
               same-family membership.

IDENTIFIABILITY: same_family and is_self are separable only where a same-family
NON-self designer exists. That is true for t2x (sibling arm) and mvb (same_large
arm). In the bridge panels x/x2 the only same-family designer IS self, so the two
terms are collinear and the script reports the combined self effect instead --
it does not silently fit a rank-deficient model.

Usage:  python src/v6_two_force.py [--boot 5000] [--seed 0]
"""
import os, json, argparse, collections
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(REPO, "results")

SIZE = {"qwen": 7.0, "llama": 8.0, "gemma": 9.0, "olmo": 7.0, "granite": 8.0,
        "qwen_sib": 3.1, "llama_sib": 3.2, "gemma_sib": 2.6, "olmo_sib": 1.0,
        "qwen_3b": 3.1, "llama_3b": 3.2, "falcon_3b": 3.2, "gemma_3b": 2.6,
        "phi_3b": 3.8, "granite_3b": 2.5}
FAM = {d: d.replace("_sib", "").replace("_3b", "") for d in SIZE}


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def fit(rows, terms, rng, n_boot):
    """OLS with cell fixed effects (within-cell demeaning) + cluster bootstrap
    over cells. Returns (beta, ci) or None if the design is rank-deficient."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)

    def design(cells):
        Y, X = [], []
        for c in cells:
            qs = by[c]
            my = np.mean([q["y"] for q in qs])
            mu = [np.mean([q[t] for q in qs]) for t in terms]
            for q in qs:
                Y.append(q["y"] - my)
                X.append([q[t] - m for t, m in zip(terms, mu)])
        return np.asarray(Y), np.asarray(X)

    cells = list(by)
    Y, X = design(cells)
    if np.linalg.matrix_rank(X) < len(terms):
        return None
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    draws = []
    for _ in range(n_boot):
        pick = [cells[i] for i in rng.integers(0, len(cells), len(cells))]
        yy, xx = design(pick)
        if len(yy) == 0 or np.linalg.matrix_rank(xx) < len(terms):
            continue
        draws.append(np.linalg.lstsq(xx, yy, rcond=None)[0])
    D = np.asarray(draws)
    ci = [(float(np.percentile(D[:, i], 2.5)), float(np.percentile(D[:, i], 97.5)))
          for i in range(len(terms))] if len(D) else [(float("nan"),) * 2] * len(terms)
    return beta, ci, len(cells), len(Y), len(D)


def build(paths, label):
    rows = []
    for p in paths:
        fp = os.path.join(RES, p, "removal.jsonl")
        if not os.path.exists(fp):
            continue
        for ln in open(fp):
            if not ln.strip():
                continue
            r = json.loads(ln)
            d = r["designer"]
            if d not in SIZE:
                continue
            rows.append(dict(cell=(r["target"], r["axis"]), y=z(r["removal"]),
                             size=float(np.log2(SIZE[d])),
                             same=1.0 if FAM[d] == r["target"] else 0.0,
                             is_self=1.0 if r["role"] == "self" else 0.0))
    return rows, label


def report(rows, label, rng, n_boot):
    print("=" * 74)
    print(f"{label}   n={len(rows)} designer-cells, {len({r['cell'] for r in rows})} cells")
    print("=" * 74)
    if not rows:
        print("  no rows\n"); return None
    full = fit(rows, ["size", "same", "is_self"], rng, n_boot)
    if full is None:
        print("  same_family and is_self are COLLINEAR here (the only same-family")
        print("  designer is self) — fitting the combined self effect instead.")
        red = fit(rows, ["size", "is_self"], rng, n_boot)
        if red is None:
            print("  design still rank-deficient — skipping\n"); return None
        beta, ci, nc, nobs, nd = red
        names = ["log2(designer size)", "is_self (= same_family here)"]
    else:
        beta, ci, nc, nobs, nd = full
        names = ["log2(designer size)", "same_family", "is_self (beyond same-family)"]
    out = {}
    print(f"  {'term':32s} {'beta':>9s}   95% CI (cluster boot over cells)")
    for n, b, (lo, hi) in zip(names, beta, ci):
        star = "" if (np.isnan(lo) or lo < 0 < hi) else "  *excludes 0"
        print(f"  {n:32s} {b:+9.4f}   [{lo:+.4f}, {hi:+.4f}]{star}")
        out[n] = dict(beta=float(b), ci=[lo, hi])
    print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print("Two-force decomposition: removal ~ log2(size) + same_family + is_self")
    print("with target x axis cell fixed effects.\n")
    results = {}
    for paths, label in [(["t2x"], "t2x — 7-9B tier (sibling arm identifies same_family)"),
                         (["x", "x2"], "x + x2 — bridge tier 2.5-3.8B"),
                         (["v6trace/mvb"], "mvb — transplant panel (same_large arm)")]:
        rows, lbl = build(paths, label)
        results[lbl] = report(rows, lbl, rng, a.boot)

    print("Reading: the paper's thesis is a FAMILY claim. If same_family is ~0 while")
    print("is_self is negative, the handicap attaches to being the target itself, not")
    print("to sharing a lineage — and eight failed probes hunting a family mechanism")
    print("would have been hunting something that isn't there.")

    if not a.no_write:
        fp = os.path.join(RES, "v6", "two_force_panels.json")
        json.dump(dict(spec="removal ~ log2(designer_size) + same_family + is_self, cell FE",
                       boot=a.boot, boot_seed=a.seed, panels=results), open(fp, "w"), indent=2)
        print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
