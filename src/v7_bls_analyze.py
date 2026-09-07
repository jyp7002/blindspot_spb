"""BL-S analysis — resolve the pre-registered steering fork (experiments_v7 §BL-S).

The three outcomes were written into PREREGISTRATION.md (v7) before any
corrected-grid steering number existed. This script decides between them.

  BL-S3 (gate)  steering must beat BOTH its own nulls, else its removal is not
                signal-borne and the edit-vs-steering comparison is vacuous.
  BL-S1         paired (edit - steering) in-budget removal per cell, pooled CI.
                The sign of that CI selects the fork branch.
  BL-S2         steering's collateral profile, measured not assumed.

SELECTION SPACE. Primary uses ONLY grid='primary' (4 configs), matching the
edit's 4 alphas. The supplementary grid is reported separately and never
substituted -- if steering loses with MORE configurations than the rule allows,
that strengthens the primary conclusion rather than replacing it.

PAIRING. Steering vectors are built from the target's OWN (self) corpus, so the
edit comparator is the edit's SELF arm on the same target x axis cell, taken from
the frozen-protocol bridge panels (results/x, results/x2). nan -> 0 both sides.
"""
import os, json, collections, argparse
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(REPO, "results")


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def boot(vals, rng, n=10000):
    v = np.asarray(vals, float)
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    d = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.mean(v)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def best_in_budget(rows):
    """Max in-budget removal; 0.0 if nothing qualifies (the edit's nan->0 rule)."""
    ok = [r["bias_reduction"] for r in rows if r["collateral_ok"]]
    return max(ok) if ok else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    st = [json.loads(l) for l in open(os.path.join(RES, "v6trace", "steer", "removal.jsonl"))]
    # edit self-arm from the frozen bridge panels
    edit = collections.defaultdict(list)
    for p in ("x", "x2"):
        fp = os.path.join(RES, p, "removal.jsonl")
        if os.path.exists(fp):
            for l in open(fp):
                r = json.loads(l)
                if r["role"] == "self":
                    edit[(r["target"], r["axis"], r["seed"])].append(z(r["removal"]))

    print("=" * 76)
    print("BL-S3 (GATE) — does steering beat its own two nulls?")
    print("=" * 76)
    # per (target, axis, seed), best in-budget within the PRIMARY grid, per kind
    cells = collections.defaultdict(dict)
    for kind in ("real", "null_partition", "null_random"):
        g = collections.defaultdict(list)
        for r in st:
            if r["kind"] == kind and r["grid"] == "primary":
                g[(r["target"], r["axis"], r["seed"])].append(r)
        for k, rows in g.items():
            cells[k][kind] = best_in_budget(rows)
    keys = sorted(k for k in cells if len(cells[k]) == 3)
    for nk in ("null_partition", "null_random"):
        d = [cells[k]["real"] - cells[k][nk] for k in keys]
        m, lo, hi = boot(d, rng, a.boot)
        verdict = "BEATS null" if lo > 0 else ("ties null" if lo < 0 < hi else "LOSES to null")
        print(f"  real - {nk:15s} {m:+.4f}  [{lo:+.3f}, {hi:+.3f}]  n={len(d)}  -> {verdict}")
    gate = all(boot([cells[k]["real"] - cells[k][nk] for k in keys], rng, a.boot)[1] > 0
               for nk in ("null_partition", "null_random"))
    print(f"\n  BL-S3: {'PASS — steering removal is signal-borne' if gate else 'FAIL — comparison is vacuous'}")

    print("\n" + "=" * 76)
    print("BL-S1 — paired (edit - steering), in-budget, PRIMARY grid (4 configs each)")
    print("=" * 76)
    print(f"  {'cell':26s} {'edit(self)':>11s} {'steering':>10s} {'edit-steer':>11s}")
    per_cell, rec = [], []
    bycell = collections.defaultdict(lambda: {"edit": [], "steer": []})
    for k in keys:
        tgt, ax, seed = k
        e = edit.get((tgt, ax, seed))
        if not e:
            continue
        bycell[(tgt, ax)]["edit"].append(float(np.mean(e)))
        bycell[(tgt, ax)]["steer"].append(cells[k]["real"])
    for c in sorted(bycell):
        e = float(np.mean(bycell[c]["edit"])); s = float(np.mean(bycell[c]["steer"]))
        per_cell.append(e - s)
        rec.append(dict(cell=f"{c[0]}|{c[1]}", edit=e, steer=s, delta=e - s))
        print(f"  {c[0] + '|' + c[1]:26s} {e:+11.3f} {s:+10.3f} {e - s:+11.3f}")
    m, lo, hi = boot(per_cell, rng, a.boot)
    print(f"\n  pooled edit - steering: {m:+.4f}  95%CI [{lo:+.3f}, {hi:+.3f}]  n={len(per_cell)} cells")
    if lo > 0:
        branch = ("EDIT > STEERING — contribution stands as stated: a sign-only weight "
                  "edit that removes bias under a deployment-honest budget.")
    elif hi < 0:
        branch = ("STEERING > EDIT — the method paper becomes a protocol-and-analysis "
                  "paper and RECOMMENDS steering for removal.")
    else:
        branch = ("EDIT ~ STEERING — contribution reframes to what steering cannot offer: "
                  "a persistent, mergeable, sub-MB, inference-cost-free patch, plus the "
                  "budget-honest protocol, sign-structure finding, and contrast_gap.")
    print(f"\n  PRE-REGISTERED BRANCH: {branch}")

    print("\n" + "=" * 76)
    print("BL-S2 — steering's collateral profile (measured)")
    print("=" * 76)
    for grid in ("primary", "supp"):
        rr = [r for r in st if r["kind"] == "real" and r["grid"] == grid]
        if not rr:
            continue
        ok = [r for r in rr if r["collateral_ok"]]
        print(f"  [{grid}] {len(ok)}/{len(rr)} configs in budget "
              f"({100 * len(ok) / len(rr):.0f}%)   "
              f"median pplr {np.median([r['ppl_ratio'] for r in rr]):.3f}   "
              f"median dMMLU {np.median([r['dmmlu'] for r in rr]):+.3f}")
    print("\n  supplementary grid (MORE configs than the equal-selection rule allows):")
    sc = collections.defaultdict(list)
    for r in st:
        if r["kind"] == "real":
            sc[(r["target"], r["axis"], r["seed"])].append(r)
    supp_best = {k: best_in_budget(v) for k, v in sc.items()}
    sb = collections.defaultdict(list)
    for (t, ax, s), v in supp_best.items():
        sb[(t, ax)].append(v)
    d2 = []
    for c in sorted(bycell):
        if c in sb:
            d2.append(float(np.mean(bycell[c]["edit"])) - float(np.mean(sb[c])))
    m2, lo2, hi2 = boot(d2, rng, a.boot)
    print(f"    edit - steering(all grids): {m2:+.4f} [{lo2:+.3f}, {hi2:+.3f}]  n={len(d2)}")

    json.dump(dict(bls3_gate=bool(gate), per_cell=rec,
                   pooled_edit_minus_steer=[m, lo, hi],
                   pooled_all_grids=[m2, lo2, hi2], branch=branch),
              open(os.path.join(RES, "v6", "bls_analysis.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(RES, 'v6', 'bls_analysis.json')}")


if __name__ == "__main__":
    main()
