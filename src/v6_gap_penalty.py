"""Does the target's OWN elicitation quality predict its self-repair penalty?

Not one of the three v6 probes — a test the accumulated results point at after
all three failed. MV-A, MV-B and MV-C each asked what the penalty co-varies with
in the EDIT (collateral entanglement, source-corpus quality, residual structure).
None survived. But one quantity kept lining up with the penalty without ever
being an ingredient of any probe: `contrast_gap`, the designer-level diagnostic
emitted at corpus-build time, which measures how sharply a model separates its
own biased from its own debiased completions.

  phi_3b | occ_gender    gap = +0.012   penalty = +0.288
  phi_3b | crows_socio   gap = +0.128   penalty = +0.002

Same model, same protocol, same seeds — a 10x difference in self-elicitation and
the penalty appears only where the elicitation is near-null. If that generalises,
the blind spot is not about shared weights or shared error structure at all: a
model fails to debias itself exactly where it cannot articulate its own bias.

This is ORD's machinery (experiments_v6 §3), applied to contrast_gap. Legitimacy:
the ingredient was frozen to results/v6/contrast_gap_frozen.json (+ .sha256)
BEFORE any correlation was computed, per §7's no-peeking rule. The frozen file's
`peek_disclosure` field records that phi's gap was observed alongside its known
penalty during the 2026-07-29 artifact audit — so phi is NOT independent
evidence here, and the test is reported with and without it.

ORD's pre-registered cell list is used verbatim: gemma-occ, gemma-crows,
qwen-occ, qwen-crows, llama-occ, llama-crows, phi-occ, phi-crows, falcon-crows;
falcon-occ flagged; granite EXCLUDED (diagnosed instrument fragility).

Outcome variable = cross - self at the strict budget, per cell, nan->0, computed
from the measured removal.jsonl panels (never a hardcoded constant).

Usage:  python src/v6_gap_penalty.py [--boot 10000] [--seed 0]
"""
import os, json, argparse, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(REPO, "results")

# ORD's frozen cell list. flagged=enters but marked; granite excluded entirely.
ORD_CELLS = [("gemma", "occ_gender"), ("gemma", "crows_socioeconomic"),
             ("qwen", "occ_gender"), ("qwen", "crows_socioeconomic"),
             ("llama", "occ_gender"), ("llama", "crows_socioeconomic"),
             ("phi", "occ_gender"), ("phi", "crows_socioeconomic"),
             ("falcon", "crows_socioeconomic")]
FLAGGED = [("falcon", "occ_gender")]          # 1/3 self budget-fails in v5
EXCLUDED = ["granite"]

# bridge target family -> the designer name whose corpus IS that target's self arm
SELF_DESIGNER_BRIDGE = {"gemma": "gemma_3b", "qwen": "qwen_3b", "llama": "llama_3b",
                        "phi": "phi_3b", "falcon": "falcon_3b", "granite": "granite_3b"}
# 7-9B tier: the self designer is the family name itself
SELF_DESIGNER_T2X = {f: f for f in ("gemma", "qwen", "llama", "olmo", "granite")}


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def load_panel(path):
    return [json.loads(ln) for ln in open(path) if ln.strip()] if os.path.exists(path) else []


def penalty(rows, tgt, ax):
    """cross - self at strict budget, nan->0. None if the cell is absent."""
    se = [z(r["removal"]) for r in rows if r["target"] == tgt and r["axis"] == ax and r["role"] == "self"]
    cr = [z(r["removal"]) for r in rows if r["target"] == tgt and r["axis"] == ax and r["role"] == "cross"]
    if not se or not cr:
        return None
    return float(np.mean(cr)) - float(np.mean(se))


def gaps():
    """Frozen contrast_gap, loader-consistent rows only, meaned over seeds."""
    art = json.load(open(os.path.join(RES, "v6", "contrast_gap_frozen.json")))
    g = collections.defaultdict(list)
    for o in art["observations"]:
        if o.get("matches_loader_corpus"):
            g[(o["designer"], o["axis"])].append(o["contrast_gap"])
    return {k: float(np.mean(v)) for k, v in g.items()}, art["frozen_utc"]


def spearman(x, y):
    def rank(v):
        order = np.argsort(np.argsort(v))
        return order.astype(float)
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def boot_rho(x, y, rng, n_boot):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(x[i])) < 2 or len(set(y[i])) < 2:
            continue
        d.append(spearman(x[i], y[i]))
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d else (float("nan"),) * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    G, frozen_at = gaps()
    bridge = load_panel(os.path.join(RES, "x", "removal.jsonl")) + \
             load_panel(os.path.join(RES, "x2", "removal.jsonl"))
    t2x = load_panel(os.path.join(RES, "t2x", "removal.jsonl"))
    print(f"ingredient frozen at {frozen_at} (sha256-verified)")
    print(f"bridge rows {len(bridge)}   t2x rows {len(t2x)}\n")

    rows = []
    for tgt, ax in ORD_CELLS + FLAGGED:
        if tgt in EXCLUDED:
            continue
        p = penalty(bridge, tgt, ax)
        dn = SELF_DESIGNER_BRIDGE.get(tgt)
        gp = G.get((dn, ax))
        if p is None or gp is None:
            continue
        rows.append(dict(tier="bridge", cell=f"{tgt}|{ax}", self_designer=dn,
                         gap=gp, penalty=p, flagged=(tgt, ax) in FLAGGED))
    for tgt, ax in ORD_CELLS:
        p = penalty(t2x, tgt, ax)
        dn = SELF_DESIGNER_T2X.get(tgt)
        gp = G.get((dn, ax))
        if p is None or gp is None:
            continue
        rows.append(dict(tier="7-9B", cell=f"{tgt}|{ax}", self_designer=dn,
                         gap=gp, penalty=p, flagged=False))

    print(f"{'tier':7s} {'cell':28s} {'self designer':12s} {'gap':>8s} {'penalty':>9s}")
    for r in sorted(rows, key=lambda r: (r["tier"], -r["penalty"])):
        print(f"{r['tier']:7s} {r['cell']:28s} {r['self_designer']:12s} "
              f"{r['gap']:+8.3f} {r['penalty']:+9.3f}" + ("  (flagged)" if r["flagged"] else ""))

    def report(sub, label):
        if len(sub) < 4:
            print(f"\n  {label}: n={len(sub)} — too few cells to correlate")
            return None
        x = [r["gap"] for r in sub]; y = [r["penalty"] for r in sub]
        rho = spearman(x, y); lo, hi = boot_rho(x, y, rng, a.boot)
        pear = float(np.corrcoef(x, y)[0, 1])
        excl = "" if (np.isnan(lo) or lo < 0 < hi) else "  *excludes 0"
        print(f"\n  {label}  n={len(sub)}")
        print(f"    Spearman rho = {rho:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]{excl}")
        print(f"    Pearson  r   = {pear:+.3f}")
        return dict(n=len(sub), spearman=rho, ci=[lo, hi], pearson=pear)

    print("\n" + "=" * 72)
    print("Does self-elicitation gap predict the self-penalty?  (negative rho = yes:")
    print("weaker self-elicitation -> larger penalty)")
    print("=" * 72)
    out = {}
    out["all"] = report(rows, "ALL cells (both tiers)")
    out["bridge"] = report([r for r in rows if r["tier"] == "bridge"], "BRIDGE tier only")
    out["t2x"] = report([r for r in rows if r["tier"] == "7-9B"], "7-9B tier only")
    out["no_phi"] = report([r for r in rows if not r["cell"].startswith("phi")],
                           "ALL cells EXCLUDING phi (phi was peeked at pre-freeze)")
    out["unflagged"] = report([r for r in rows if not r["flagged"]], "ALL unflagged cells")

    print("\n  Reading: phi is NOT independent evidence (see peek_disclosure in the")
    print("  frozen ingredient). The 'excluding phi' line is the honest headline.")

    if not a.no_write:
        fp = os.path.join(RES, "v6", "gap_penalty.json")
        json.dump(dict(ingredient="contrast_gap (frozen, sha256-verified)",
                       frozen_utc=frozen_at, ord_cells=[list(c) for c in ORD_CELLS],
                       flagged=[list(c) for c in FLAGGED], excluded=EXCLUDED,
                       outcome="cross - self at strict budget, nan->0",
                       cells=rows, results=out, boot=a.boot, boot_seed=a.seed),
                  open(fp, "w"), indent=2)
        print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
