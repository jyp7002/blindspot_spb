"""FX — a self-adaptive edit-source selector, built on what actually survived v6.

The three pre-wired fixes are orphaned: FX-A depended on MV-A (failed: phi's
penalty is budget-independent), FX-C on MV-C (failed: residual correlation covers
0), FX-B on MV-B1 (failed: a same-family-large corpus does not close the penalty).
So the fix cannot be justified by a mechanism -- none was found in ten
operationalizations. It has to be justified by a *predictor*, and exactly one
survived every test:

    contrast_gap predicts REMOVAL       +0.507 [+0.220, +0.732]  (n=504, 26 cells)
    designer size, given the gap        -0.009 [-0.029, +0.015]  (absorbed)

contrast_gap is the designer's own elicitation diagnostic: how sharply it separates
its own biased from its own debiased completions on THIS axis. It is
  * endogenous  — computed from the model itself, no ground truth, no out-of-family
                  supervision (respects the §1 endogeneity boundary)
  * pre-edit    — emitted at corpus-build time, before any training or evaluation
  * per-axis    — phi_3b scores +0.012 on occ_gender and +0.128 on crows, and its
                  penalty appears only on occ. A per-model constant could not do this.
  * free        — already logged by the existing pipeline

THE RULE (no fitted parameters):
    for each (target, axis): among LEGAL sources (the target itself and its
    family siblings), use the corpus of whichever has the highest contrast_gap.

This is scale-adaptive without referencing scale: at 7-9B a model's own gap is
already high (+0.27..+0.49), so the rule keeps self and changes nothing -- which is
what DEMO-2 requires. At <=3.8B, where self-elicitation is weak, it reaches for the
sibling. The adaptation falls out of the signal rather than being switched on by a
size threshold.

HONEST STATUS: the decision to *use* contrast_gap was made after seeing it predict
removal on these same panels. The rule has no fitted coefficients, so there is
nothing to overfit in the usual sense, but the choice of signal is not
out-of-sample. Reported below against both a self baseline and an oracle ceiling.

Usage:  python src/v6_fx_adaptive.py [--boot 10000]
"""
import os, json, argparse, collections
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(REPO, "results")

# designer -> (family, size). Legal sources for a target are same-family only.
DESIGNER = {"qwen_3b": ("qwen", 3.1), "llama_3b": ("llama", 3.2), "falcon_3b": ("falcon", 3.2),
            "gemma_3b": ("gemma", 2.6), "phi_3b": ("phi", 3.8), "granite_3b": ("granite", 2.5),
            "qwen_sib": ("qwen", 3.1), "llama_sib": ("llama", 3.2), "gemma_sib": ("gemma", 2.6),
            "olmo_sib": ("olmo", 1.0),
            "qwen": ("qwen", 7.0), "llama": ("llama", 8.0), "gemma": ("gemma", 9.0),
            "olmo": ("olmo", 7.0), "granite": ("granite", 8.0)}
PANELS = {"t2x": "t2x/removal.jsonl", "mvb": "v6trace/mvb/removal.jsonl",
          "x": "x/removal.jsonl", "x2": "x2/removal.jsonl"}


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def load_gaps():
    art = json.load(open(os.path.join(RES, "v6", "contrast_gap_frozen.json")))
    g = collections.defaultdict(list)
    for o in art["observations"]:
        if o.get("matches_loader_corpus"):
            g[(o["designer"], o["axis"])].append(o["contrast_gap"])
    return {k: float(np.mean(v)) for k, v in g.items()}, art["frozen_utc"]


def boot_ci(vals, rng, n_boot):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    d = [np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return float(np.mean(v)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    G, frozen_at = load_gaps()

    # measured removal for every (panel, target, axis, designer)
    obs = collections.defaultdict(list)
    for panel, path in PANELS.items():
        fp = os.path.join(RES, path)
        if not os.path.exists(fp):
            continue
        for ln in open(fp):
            r = json.loads(ln)
            if r["designer"] in DESIGNER:
                obs[(panel, r["target"], r["axis"], r["designer"])].append(z(r["removal"]))
    removal = {k: float(np.mean(v)) for k, v in obs.items()}

    # a cell is evaluable when >1 LEGAL (same-family) source was actually measured
    cells = collections.defaultdict(list)
    for (panel, tgt, ax, dn) in removal:
        if DESIGNER[dn][0] == tgt:                      # same family = legal
            cells[(panel, tgt, ax)].append(dn)
    evaluable = {c: d for c, d in cells.items() if len(d) > 1}

    print(f"contrast_gap frozen {frozen_at}")
    print(f"cells with >1 legal endogenous source: {len(evaluable)} "
          f"(of {len(cells)} cells total)\n")

    rows = []
    print(f"  {'cell':26s} {'self':>7s} {'picked':>11s} {'gap':>7s} {'FX':>7s} "
          f"{'oracle':>7s} {'FX-self':>8s}")
    for c in sorted(evaluable):
        panel, tgt, ax = c
        srcs = evaluable[c]
        self_dn = [d for d in srcs if removal.get((panel, tgt, ax, d)) is not None
                   and d in (tgt, f"{tgt}_3b")]
        # the self arm is the designer whose name is the target family or <fam>_3b
        self_dn = self_dn[0] if self_dn else None
        gaps = {d: G.get((d, ax)) for d in srcs}
        if self_dn is None or any(v is None for v in gaps.values()):
            continue
        pick = max(srcs, key=lambda d: gaps[d])
        r_self = removal[(panel, tgt, ax, self_dn)]
        r_fx = removal[(panel, tgt, ax, pick)]
        r_or = max(removal[(panel, tgt, ax, d)] for d in srcs)
        rows.append(dict(cell=f"{panel}:{tgt}|{ax}", self=r_self, picked=pick,
                         picked_gap=gaps[pick], self_gap=gaps[self_dn],
                         fx=r_fx, oracle=r_or, delta=r_fx - r_self,
                         kept_self=pick == self_dn, n_sources=len(srcs)))
        print(f"  {panel + ':' + tgt + '|' + ax:26s} {r_self:+7.3f} {pick:>11s} "
              f"{gaps[pick]:+7.3f} {r_fx:+7.3f} {r_or:+7.3f} {r_fx - r_self:+8.3f}"
              + ("   (kept self)" if pick == self_dn else ""))

    if not rows:
        print("  no evaluable cells"); return

    d = [r["delta"] for r in rows]
    m, lo, hi = boot_ci(d, rng, a.boot)
    print(f"\n  FX - self over {len(d)} cells: {m:+.4f}  95%CI [{lo:+.3f}, {hi:+.3f}]"
          + ("" if lo < 0 < hi else "  *excludes 0"))
    orc = [r["oracle"] - r["self"] for r in rows]
    mo, lo_o, hi_o = boot_ci(orc, rng, a.boot)
    print(f"  oracle - self (ceiling):    {mo:+.4f}  95%CI [{lo_o:+.3f}, {hi_o:+.3f}]")
    if mo != 0:
        print(f"  FX captures {100 * m / mo:.0f}% of the achievable headroom")

    # DEMO-2: the rule must not damage the regime that already works (7-9B = t2x)
    big = [r for r in rows if r["cell"].startswith("t2x")]
    small = [r for r in rows if not r["cell"].startswith("t2x")]
    for lbl, sub in (("7-9B (t2x) — DEMO-2, must not harm", big),
                     ("<=3.8B (mvb) — DEMO-1, should help", small)):
        if not sub:
            continue
        mm, ll, hh = boot_ci([r["delta"] for r in sub], rng, a.boot)
        kept = sum(r["kept_self"] for r in sub)
        print(f"\n  {lbl}: n={len(sub)}  FX-self {mm:+.4f} [{ll:+.3f}, {hh:+.3f}]"
              f"   kept self in {kept}/{len(sub)}")

    if not a.no_write:
        fp = os.path.join(RES, "v6", "fx_adaptive.json")
        json.dump(dict(rule="argmax contrast_gap over same-family sources (no fitted params)",
                       signal_frozen=frozen_at, n_cells=len(rows),
                       fx_minus_self=[m, lo, hi], oracle_minus_self=[mo, lo_o, hi_o],
                       cells=rows,
                       caveat=("signal choice was informed by these same panels; the rule "
                               "has no fitted coefficients but is not out-of-sample")),
                  open(fp, "w"), indent=2)
        print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
