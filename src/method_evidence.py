"""Evidence table for the METHOD paper — the binary (sign-only) debiaser.

Reframing note: the program was built to test a family blind-spot law, and the
edit was its instrument. The law's primary estimand (H2, the origin x
designer-family interaction) measures +0.004 [-0.042, +0.051] -- it is not there.
The instrument, by contrast, is the best-validated thing in the project. This
assembles what a method-first paper needs, which no analysis has done because the
edit was never the headline.

Claims a method paper must support, and where each is measured:

  C1 the edit removes bias           vs BOTH pre-registered nulls
  C2 sign structure carries it       vs random-sign control (same scale/sparsity)
  C3 binarization is nearly free     sign-only vs full-precision task-vector negation
  C4 it is tiny                      bytes, flips, effective sparsity
  C5 scale granularity barely matters per_tensor vs per_channel vs scalar
  C6 bit budget                      sparsity sweep 0 / 0.5 / 0.9 / 0.99

MODULE-SET DISCIPLINE. runs.jsonl contains two edit protocols under the same
`variant` string, separable only by the Tq+k+v+o tag inside `key`:
    Tq+k+v+o    the FROZEN protocol (attn q,k,v,o r16) used by every v5/v6 result
    untagged    a v2-era layer-placement ablation
C2 is available on the frozen protocol. C3/C5/C6 exist ONLY untagged, so they are
computed within that module set and labelled as such -- never pooled across the
two, which would silently mix protocols (the same trap that duplicated 456 of 744
tuples in the MV-A F1 adapter).

Usage:  python src/method_evidence.py [--boot 10000]
"""
import os, json, argparse, collections
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(REPO, "results")


def mset(r):
    t = [p for p in r.get("key", "").split("|") if p.startswith("T")]
    return t[0] if t else "untagged"


def best_in_budget(rs):
    ok = [x["bias_reduction"] for x in rs if x["collateral_ok"]]
    return max(ok) if ok else 0.0            # nan -> 0


def sweeps(rows):
    """(target, axis, seed, designer) -> best in-budget removal."""
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["target"], r["axis"], r["seed"], r["designer"])].append(r)
    return {k: best_in_budget(v) for k, v in g.items()}


def paired(a, b, rng, n_boot):
    """Paired difference over the keys both dicts share."""
    ks = sorted(set(a) & set(b))
    if not ks:
        return None
    d = np.array([a[k] - b[k] for k in ks], float)
    draws = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)]
    return float(np.mean(d)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), len(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = [json.loads(ln) for ln in open(os.path.join(RES, "runs.jsonl")) if ln.strip()]
    endo = [r for r in rows if r.get("signal") == "endogenous"]
    out = {}

    def sel(variant, tag=None, pool=None):
        src = pool if pool is not None else endo
        return [r for r in src if r.get("variant") == variant and (tag is None or mset(r) == tag)]

    # ---- C2 sign structure (frozen protocol) --------------------------------
    print("=" * 74)
    print("C2  SIGN STRUCTURE CARRIES THE EFFECT   [FROZEN protocol Tq+k+v+o]")
    print("=" * 74)
    real = sweeps(sel("binary-per_tensor-s0.0", "Tq+k+v+o"))
    rand = sweeps(sel("binary-per_tensor-s0.0-rand", "Tq+k+v+o"))
    res = paired(real, rand, rng, a.boot)
    if res:
        m, lo, hi, n = res
        print(f"  real - random-sign (same scale & sparsity): {m:+.4f}  [{lo:+.3f}, {hi:+.3f}]  n={n} sweeps")
        print(f"    real mean {np.mean([real[k] for k in set(real)&set(rand)]):+.4f}   "
              f"random mean {np.mean([rand[k] for k in set(real)&set(rand)]):+.4f}")
        out["C2_sign_structure"] = dict(delta=m, ci=[lo, hi], n=n, module_set="Tq+k+v+o")

    # ---- C3 binarization cost (untagged only) -------------------------------
    print("\n" + "=" * 74)
    print("C3  COST OF BINARIZATION vs FULL-PRECISION   [UNTAGGED module set only]")
    print("=" * 74)
    fp = sweeps(sel("fp--s0", "untagged"))
    bi = sweeps(sel("binary-per_tensor-s0.0", "untagged"))
    res = paired(bi, fp, rng, a.boot)
    if res:
        m, lo, hi, n = res
        ks = sorted(set(bi) & set(fp))
        mb, mf = np.mean([bi[k] for k in ks]), np.mean([fp[k] for k in ks])
        print(f"  binary - full-precision: {m:+.4f}  [{lo:+.3f}, {hi:+.3f}]  n={n} matched sweeps")
        print(f"    binary mean {mb:+.4f}   fp mean {mf:+.4f}   retention {100*mb/mf if mf else float('nan'):.1f}%")
        print("    (H1 in design.md §7 asks sign-only to recover a LARGE FRACTION of fp)")
        out["C3_binarization"] = dict(delta=m, ci=[lo, hi], n=n, binary=mb, fp=mf,
                                      retention_pct=(100 * mb / mf) if mf else None,
                                      module_set="untagged")

    # ---- C4 size ------------------------------------------------------------
    print("\n" + "=" * 74)
    print("C4  EDIT SIZE")
    print("=" * 74)
    for tag in ("Tq+k+v+o", "untagged"):
        em = [r["edit_meta"] for r in sel("binary-per_tensor-s0.0", tag)
              if isinstance(r.get("edit_meta"), dict) and "bytes" in r["edit_meta"]]
        if not em:
            continue
        by = collections.defaultdict(list)
        for r, e in zip(sel("binary-per_tensor-s0.0", tag), em):
            by[r["target"]].append(e)
        print(f"  [{tag}]")
        for t, es in sorted(by.items()):
            b = np.mean([e["bytes"] for e in es]) / 2**20
            p = np.mean([e["n_params"] for e in es]) / 1e6
            print(f"    {t:11s} {b:8.2f} MB   {p:9.1f}M edited params   "
                  f"eff.sparsity {np.mean([e['effective_sparsity'] for e in es]):.3f}")
        out.setdefault("C4_size", {})[tag] = {
            t: dict(mb=float(np.mean([e["bytes"] for e in es]) / 2**20),
                    m_params=float(np.mean([e["n_params"] for e in es]) / 1e6))
            for t, es in by.items()}

    # ---- C5 granularity, C6 bit budget (untagged) ---------------------------
    for label, variants, key in [
            ("C5  SCALE GRANULARITY   [UNTAGGED]",
             ["binary-per_tensor-s0.0", "binary-per_channel-s0.0", "binary-scalar-s0.0"], "C5_granularity"),
            ("C6  BIT BUDGET (sparsity sweep)   [UNTAGGED]",
             ["binary-per_tensor-s0.0", "binary-per_tensor-s0.5",
              "binary-per_tensor-s0.9", "binary-per_tensor-s0.99"], "C6_bit_budget")]:
        print("\n" + "=" * 74); print(label); print("=" * 74)
        base = sweeps(sel(variants[0], "untagged"))
        rec = {}
        for v in variants:
            s = sweeps(sel(v, "untagged"))
            ks = sorted(set(s) & set(base))
            if not ks:
                continue
            mv = float(np.mean([s[k] for k in ks]))
            d = paired(s, base, rng, a.boot) if v != variants[0] else None
            line = f"  {v:30s} n={len(ks):3d}  removal {mv:+.4f}"
            if d:
                line += f"   vs per_tensor-s0.0: {d[0]:+.4f} [{d[1]:+.3f}, {d[2]:+.3f}]"
            print(line)
            rec[v] = dict(n=len(ks), removal=mv, vs_base=(list(d[:3]) if d else None))
        out[key] = rec

    print("\n" + "=" * 74)
    print("C1  vs BOTH NULLS — measured on the v6 panels, not runs.jsonl:")
    print("    phi|occ real +0.316/+0.632  vs sign_shuffle +0.002  partition +0.014/+0.032")
    print("    real - sign_shuffle +0.577 [+0.509, +0.638];  real - partition +0.550 [+0.468, +0.625]")
    print("=" * 74)

    if not a.no_write:
        fp_out = os.path.join(RES, "v6", "method_evidence.json")
        json.dump(out, open(fp_out, "w"), indent=2)
        print(f"\nwrote {fp_out}")


if __name__ == "__main__":
    main()
