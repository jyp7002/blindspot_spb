#!/usr/bin/env python3
"""v12.E — Delta_selection under alpha* at <=9B: scale effect or rule effect?

    python3 src/v12_astar.py            # verdict -> results/v12/astar_verdict.json
    python3 src/v12_astar.py progress   # unit counts only; reads no removal
    python3 src/v12_astar.py selftest   # the branch logic on synthetic rows

The criterion below is PREREGISTRATION §v12.E, coded before the first v12astar
unit ran. It prints no removal of any kind until all 30 units are on disk, so
the panel cannot be read, and the rule adjusted, part-way through.

ESTIMAND. E = Delta_selection under alpha* = s0.99 - C-a@0.01, each arm at its
own alpha* (v12.A rule, calibration split, no bias probe read), removal as
deployed (D5), cell = seed-mean over {0,1,2}, alpha* status 'none' -> 0 (the
edit is not deployed), 10 registered cells, paired cell bootstrap (v10_common).
F = the same contrast under the frozen {2,4,8,16} argmax, same panel.

BRANCHES (exactly one):
  PERSISTS          E.lo > 0
  CLOSES_REVERSED   E.hi < 0
  CLOSES            CI covers 0 and E.point <= CLOSE_FRAC * F.point
  INCONCLUSIVE      CI covers 0 and E.point >  CLOSE_FRAC * F.point
SCALE CLAIM (only under PERSISTS): S = E(<=9B, 10 cells) - E(27-32B, v12big,
4 cells), unpaired cell bootstrap; the claim is licensed iff S.lo > 0.
SENSITIVITIES (reported, never the verdict; a branch change is labelled
FRAGILE): (i) removal zeroed where alpha* fails the evaluation gate (D5's
alternative); (ii) the 7 occ_gender/bbq_Age cells (axis-matched to v12big).
"""
import collections
import json
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V   # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
PANEL, BIG = "v12astar", "v12big"
CLOSE_FRAC = 0.25
ARMS = ("s0.99", "C-a@0.01")
CELLS = [("gemma", "occ_gender"), ("llama", "occ_gender"), ("qwen", "occ_gender"),
         ("phi", "occ_gender"), ("phi", "bbq_Age"), ("qwen", "bbq_Age"),
         ("qwen", "bbq_Race_ethnicity"), ("qwen", "ss_intra"), ("phi", "ss_intra"),
         ("qwen7b", "occ_gender")]
SEEDS = (0, 1, 2)
MATCHED_AXES = {"occ_gender", "bbq_Age"}
REPLAY = {"s0.99": "C-ref", "C-a@0.01": "C-a"}


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def load(panel, root=RESULTS):
    fp = os.path.join(root, panel, "removal.jsonl")
    out = {}
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            out[(r["target"], r["axis"], r["seed"], r["variant"])] = r  # last wins
    return out


def adaptive_removal(r, zero_eval_fail=False):
    ad = r.get("adaptive") or {}
    if ad.get("alpha_star") is None:               # status 'none': not deployed
        return 0.0
    if zero_eval_fail and not ad.get("eval_ok"):
        return 0.0
    return z(ad.get("removal"))


def cell_pairs(rows, cells, field):
    """[(s0.99, C-a) seed-mean per cell] for the cells given."""
    out = []
    for t, a in cells:
        m = {vn: [field(rows[(t, a, s, vn)]) for s in SEEDS if (t, a, s, vn) in rows]
             for vn in ARMS}
        if all(m[vn] for vn in ARMS):
            out.append((float(np.mean(m["s0.99"])), float(np.mean(m["C-a@0.01"]))))
    return out


def boot_diff(xs, ys, n=V.BOOT_N, seed=V.BOOT_SEED):
    """Unpaired: mean(xs) - mean(ys), each group resampled independently."""
    rng = random.Random(seed)
    bs = sorted(float(np.mean([xs[rng.randrange(len(xs))] for _ in xs])
                      - np.mean([ys[rng.randrange(len(ys))] for _ in ys]))
                for _ in range(n))
    return dict(point=float(np.mean(xs) - np.mean(ys)), lo=bs[int(.025 * n)],
                hi=bs[int(.975 * n) - 1], n=[len(xs), len(ys)])


def branch(E, F):
    if E["lo"] > 0:
        return "PERSISTS"
    if E["hi"] < 0:
        return "CLOSES_REVERSED"
    return "CLOSES" if E["point"] <= CLOSE_FRAC * F["point"] else "INCONCLUSIVE"


def missing(rows):
    need = [(t, a, s, vn) for t, a in CELLS for s in SEEDS for vn in ARMS]
    return [k for k in need if k not in rows]


def verdict(rows, big_rows):
    fz = lambda r: z(r["removal"])
    ad = lambda r: adaptive_removal(r)
    ad0 = lambda r: adaptive_removal(r, zero_eval_fail=True)
    F = V.boot_paired(cell_pairs(rows, CELLS, fz))
    E = V.boot_paired(cell_pairs(rows, CELLS, ad))
    b = branch(E, F)
    out = dict(criterion="PREREGISTRATION §v12.E", close_frac=CLOSE_FRAC,
               frozen=F, adaptive=E, branch=b)

    big_cells = sorted({(t, a) for t, a, _s, _v in big_rows})
    small_d = [x - y for x, y in cell_pairs(rows, CELLS, ad)]
    big_d = [x - y for x, y in cell_pairs(big_rows, big_cells, ad)]
    S = boot_diff(small_d, big_d) if small_d and big_d else None
    out["scale_contrast"] = S
    out["scale_claim"] = bool(b == "PERSISTS" and S is not None and S["lo"] > 0)

    sens = {}
    e1 = V.boot_paired(cell_pairs(rows, CELLS, ad0))
    sens["eval_fail_zeroed"] = dict(adaptive=e1, branch=branch(e1, F))
    mc = [c for c in CELLS if c[1] in MATCHED_AXES]
    f2 = V.boot_paired(cell_pairs(rows, mc, fz))
    e2 = V.boot_paired(cell_pairs(rows, mc, ad))
    sens["matched_axes"] = dict(cells=len(mc), frozen=f2, adaptive=e2,
                                branch=branch(e2, f2))
    out["sensitivities"] = sens
    out["fragile"] = any(s["branch"] != b for s in sens.values())

    per_arm = {}
    for vn in ARMS:
        rs = [r for k, r in rows.items() if k[3] == vn]
        a = sorted(r["adaptive"]["alpha_star"] for r in rs
                   if (r.get("adaptive") or {}).get("alpha_star") is not None)
        ev = [r["adaptive"]["eval_ok"] for r in rs
              if (r.get("adaptive") or {}).get("alpha_star") is not None]
        per_arm[vn] = dict(
            status=dict(collections.Counter((r.get("adaptive") or {}).get("status")
                                            for r in rs)),
            alpha_star_median=a[len(a) // 2] if a else None,
            alpha_star_range=[a[0], a[-1]] if a else None,
            eval_pass_rate=sum(ev) / len(ev) if ev else None)
    out["alpha_star"] = per_arm
    return out


def replay(rows):
    ref = {}
    fp = os.path.join(REPO, "results_v9", "v8dec", "removal.jsonl")
    for ln in open(fp):
        r = json.loads(ln)
        ref[(r["target"], r["axis"], r["seed"], r.get("condition"))] = r
    pairs = [(k, z(r["removal"]), z(ref[(k[0], k[1], k[2], REPLAY[k[3]])]["removal"]))
             for k, r in rows.items()
             if (k[0], k[1], k[2], REPLAY[k[3]]) in ref]
    bad = [p for p in pairs if p[1] != p[2]]
    return dict(n=len(pairs), n_diff=len(bad),
                max_abs=max((abs(a - b) for _k, a, b in pairs), default=0.0))


def progress():
    rows = load(PANEL)
    need = len(CELLS) * len(SEEDS) * len(ARMS)
    print(f"{PANEL}: {need - len(missing(rows))}/{need} rows "
          f"({len({k[:3] for k in rows})}/{len(CELLS) * len(SEEDS)} units touched)")


def main():
    rows = load(PANEL)
    miss = missing(rows)
    if miss:
        progress()
        print("incomplete: no removal is read before all 30 units exist (§v12.E)")
        return 1
    out = verdict(rows, load(BIG))
    out["replay_v8dec"] = replay(rows)
    fp = os.path.join(RESULTS, "v12", "astar_verdict.json")
    json.dump(out, open(fp, "w"), indent=1)
    print(f"replay vs v8dec: {out['replay_v8dec']}")
    print(f"frozen   Delta_selection {V.fmt(out['frozen'], 3)}")
    print(f"alpha*   Delta_selection {V.fmt(out['adaptive'], 3)}")
    print(f"BRANCH: {out['branch']}" + ("  (FRAGILE)" if out["fragile"] else ""))
    S = out["scale_contrast"]
    if S:
        print(f"scale contrast <=9B - 27-32B: {S['point']:+.3f} "
              f"[{S['lo']:+.3f}, {S['hi']:+.3f}]  scale claim: {out['scale_claim']}")
    for k, s in out["sensitivities"].items():
        print(f"  sensitivity {k:18s} {V.fmt(s['adaptive'], 3)} -> {s['branch']}")
    for vn, d in out["alpha_star"].items():
        print(f"  alpha* {vn:9s} {d}")
    print(f"wrote {fp}")
    return 0


def selftest():
    def mk(t, a, s, vn, fr, ad, ok=True, st="ok"):
        return ((t, a, s, vn), dict(removal=fr, adaptive=dict(
            alpha_star=None if st == "none" else 8.0, status=st, eval_ok=ok,
            removal=ad)))

    def panel(gap_f, gaps_a, ok=True):
        """gaps_a: per-cell adaptive gap (a scalar is broadcast to all cells)."""
        if not isinstance(gaps_a, list):
            gaps_a = [gaps_a] * len(CELLS)
        rows = {}
        for (t, a), g in zip(CELLS, gaps_a):
            for s in SEEDS:
                rows.update([mk(t, a, s, "s0.99", .3 + gap_f, .4 + g, ok),
                             mk(t, a, s, "C-a@0.01", .3, .4, ok)])
        return rows

    big = dict([mk("g27", "occ_gender", s, vn, .5, .5) for s in SEEDS for vn in ARMS])
    assert verdict(panel(.28, .20), big)["branch"] == "PERSISTS"
    assert verdict(panel(.28, .20), big)["scale_claim"] is True
    mixed0 = [.3, -.3, .2, -.2, .1, -.1, .25, -.25, .05, -.05]          # mean 0
    assert verdict(panel(.28, mixed0), big)["branch"] == "CLOSES"
    wide = [.9, -.6, .8, -.5, .7, -.4, .6, -.5, .5, -.2]               # mean .13
    assert verdict(panel(.28, wide), big)["branch"] == "INCONCLUSIVE"
    assert verdict(panel(.28, -.2), big)["branch"] == "CLOSES_REVERSED"
    # eval failures zeroed on both arms -> sensitivity branch differs -> FRAGILE
    v = verdict(panel(.28, .20, ok=False), big)
    assert v["sensitivities"]["eval_fail_zeroed"]["branch"] != "PERSISTS" and v["fragile"]
    # alpha* status 'none' counts as 0, never as missing
    r = dict([mk("gemma", "occ_gender", 0, "s0.99", .3, None, st="none")])
    assert adaptive_removal(r[("gemma", "occ_gender", 0, "s0.99")]) == 0.0
    assert missing({}) and len(missing({})) == 60
    print("v12_astar selftest OK")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    sys.exit({"verdict": main, "progress": progress, "selftest": selftest}[cmd]() or 0)
