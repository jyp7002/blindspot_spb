#!/usr/bin/env python3
"""v13 — replay gate, determinism diagnostic, and the SEED / FLOOR verdicts.

    python3 src/v13_analyze.py replay      # E0.5 gate; exit 1 on FAIL
    python3 src/v13_analyze.py det         # E0.5 diagnostic; no verdict
    python3 src/v13_analyze.py progress    # unit counts only; reads no removal
    python3 src/v13_analyze.py verdicts    # SEED + FLOOR -> results/v13/verdicts.json
    python3 src/v13_analyze.py selftest

The rules are PREREGISTRATION §v13.B-E, coded before the first v13 unit.
`verdicts` reads no removal from a panel until every unit of it is on disk; a
tier whose panel is incomplete (e.g. 27-32B, still on the big card) is reported
as pending, and nothing else waits for it.

DEPLOYABLE (primary): a selected configuration counts its removal only if it
also passes the evaluation gate (200 MMLU test items + evaluation perplexity);
otherwise 0. alpha* status 'none' -> 0. AS-DEPLOYED (v12 D5): the removal at
alpha* whatever the evaluation gate says. Cell = seed mean; ratios are ratios of
cell means with a paired cell-level percentile bootstrap (10k, RNG seed 0).
"""
import collections
import json
import math
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V  # noqa: E402
import v11_panel as P  # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
OUT = os.path.join(RESULTS, "v13")
BAR = 0.85                        # v12.A rho - 0.05, rho = 0.9
MMLU_TOL = 3                      # E0.5 replay tolerance, items of 200 (§v13.B)
SEED_ARM, REF_ARM, CA_ARM = "C-seed@0.01", "s0.99", "C-a@0.01"
FLOOR_PTS = ["s0.99", "s0.995", "s0.999", "s0.9995"]
BIG_CELLS_FOR_VERDICT = 8         # §v13.C: a 27-32B SEED verdict needs 8 cells
IFEVAL_METRIC = "inst_level_strict_acc,none"
CFG = {"seed_small": "configs/v13/seed_dec.yaml",
       "floor_small": "configs/v13/floor_cal.yaml",
       "floor_big": "configs/v13/floor_big.yaml",
       "replay": "configs/v13/replay.yaml"}
SEED_BIG_CFG = "configs/v13/seed_big_{n}.yaml"   # n fixed by §v13.F
REGISTERED10 = {("gemma", "occ_gender"), ("llama", "occ_gender"), ("qwen", "occ_gender"),
                ("phi", "occ_gender"), ("phi", "bbq_Age"), ("qwen", "bbq_Age"),
                ("qwen", "bbq_Race_ethnicity"), ("qwen", "ss_intra"),
                ("phi", "ss_intra"), ("qwen7b", "occ_gender")}


# ------------------------------------------------------------------ io ----
def z(x):
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def rows(panel):
    fp = os.path.join(RESULTS, panel, "removal.jsonl")
    out = {}
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            out[(r["target"], r["axis"], r["seed"], r["variant"])] = r
    return out


def trace(panel, rule="frozen"):
    fp = os.path.join(RESULTS, panel, "alpha_trace.jsonl")
    out = {}
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                t = json.loads(ln)
            except Exception:
                continue
            if t.get("split") == "eval" and t.get("rule") == rule:
                out[(t["target"], t["axis"], t["seed"], t["variant"],
                     float(t["alpha"]))] = t            # last occurrence wins
    return out


def expected(cfg_path):
    """{(t, a, s, variant)} a complete panel must contain."""
    cfg = P.load(os.path.join(REPO, cfg_path))
    return cfg["panel"], {(u["target"], u["axis"], u["seed"], vn)
                          for u in P.expand(cfg) for vn in P.opsel_variants(u)}


def complete(cfg_path):
    panel, need = expected(cfg_path)
    have = rows(panel)
    return panel, have, sorted(need - set(have)), len(need)


# ------------------------------------------------------------- scoring ----
def deployable(r):
    ad = (r or {}).get("adaptive") or {}
    if ad.get("alpha_star") is None or not ad.get("eval_ok"):
        return 0.0
    return z(ad.get("removal"))


def as_deployed(r):
    ad = (r or {}).get("adaptive") or {}
    return 0.0 if ad.get("alpha_star") is None else z(ad.get("removal"))


def frozen(r):
    return z((r or {}).get("removal"))


def cellmeans(R, variant, field):
    acc = collections.defaultdict(list)
    for (t, a, s, vn), r in R.items():
        if vn == variant:
            acc[(t, a)].append(field(r))
    return {c: float(np.mean(v)) for c, v in acc.items()}


def boot_ratio(pairs, n=V.BOOT_N, seed=V.BOOT_SEED):
    """mean(num) / mean(den) over cells, paired cell bootstrap (percentile)."""
    if not pairs:
        return None
    num = [p[0] for p in pairs]
    den = [p[1] for p in pairs]
    rng = random.Random(seed)
    k = len(pairs)
    bs, undefined = [], 0
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        d = np.mean([den[i] for i in idx])
        if d <= 0:
            undefined += 1
            bs.append(-math.inf)          # a non-positive reference cannot pass
            continue
        bs.append(float(np.mean([num[i] for i in idx]) / d))
    bs.sort()
    dm = float(np.mean(den))
    return dict(point=float(np.mean(num)) / dm if dm > 0 else None,
                lo=bs[int(.025 * n)], hi=bs[int(.975 * n) - 1], n=k,
                num_mean=float(np.mean(num)), den_mean=dm,
                undefined_resamples=undefined)


def boot_ratio_diff(pa, pb, n=V.BOOT_N, seed=V.BOOT_SEED):
    """ratio(pa) - ratio(pb), the two cell sets resampled independently."""
    rng = random.Random(seed)

    def one(p):
        idx = [rng.randrange(len(p)) for _ in p]
        d = np.mean([p[i][1] for i in idx])
        return np.mean([p[i][0] for i in idx]) / d if d > 0 else float("nan")
    bs = sorted(x for x in (one(pa) - one(pb) for _ in range(n)) if not math.isnan(x))
    ra = np.mean([p[0] for p in pa]) / np.mean([p[1] for p in pa])
    rb = np.mean([p[0] for p in pb]) / np.mean([p[1] for p in pb])
    m = len(bs)
    return dict(point=float(ra - rb), lo=bs[int(.025 * m)], hi=bs[int(.975 * m) - 1],
                n=[len(pa), len(pb)], dropped_resamples=n - m)


def verdict3(ci, pos="NON-INFERIOR", neg="INFERIOR"):
    if ci is None:
        return "PENDING"
    if ci["lo"] >= BAR:
        return pos
    if ci["hi"] < BAR:
        return neg
    return "INCONCLUSIVE"


def screen(R, cells):
    """v11.F: in-envelope iff frozen C-ref seed-pooled > 0 and >= 1 seed in budget."""
    out = {}
    for c in cells:
        rs = [r for (t, a, s, vn), r in R.items() if (t, a) == c and vn == REF_ARM]
        pooled = float(np.mean([frozen(r) for r in rs])) if rs else float("nan")
        any_ok = any(r.get("alpha_frozen") is not None for r in rs)
        out[c] = dict(c_ref_frozen=pooled, in_envelope=bool(pooled > 0 and any_ok))
    return out


def arm_stats(R, variant):
    rs = [r for k, r in R.items() if k[3] == variant]
    ad = [r.get("adaptive") or {} for r in rs]
    a = sorted(x["alpha_star"] for x in ad if x.get("alpha_star") is not None)
    ev = [bool(x.get("eval_ok")) for x in ad if x.get("alpha_star") is not None]
    k1 = [x["mmlu1k"]["ok"] for x in ad if x.get("mmlu1k")]
    k1f = [r["mmlu1k_frozen"]["ok"] for r in rs if r.get("mmlu1k_frozen")]
    pt = [r["patch"] for r in rs if r.get("patch")]
    out = dict(n_units=len(rs), status=dict(collections.Counter(x.get("status") for x in ad)),
               alpha_star_median=a[len(a) // 2] if a else None,
               alpha_star_range=[a[0], a[-1]] if a else None,
               eval_pass_rate=sum(ev) / len(ev) if ev else None,
               mmlu1k_fail_at_alpha_star=[sum(not x for x in k1), len(k1)],
               mmlu1k_fail_at_frozen=[sum(not x for x in k1f), len(k1f)])
    if pt:
        mib = sorted(p["mib"] for p in pt)
        out["patch"] = dict(kind=pt[0]["kind"], mib_range=[mib[0], mib[-1]],
                            mib_median=mib[len(mib) // 2],
                            all_identical=all(p["identical"] for p in pt),
                            bits_per_coord_median=sorted(
                                p["bits_per_coord"] for p in pt)[len(pt) // 2])
        if "entropy_floor_bits_per_tensor" in pt[0]:
            out["patch"]["entropy_floor_bits_per_tensor_median"] = sorted(
                p["entropy_floor_bits_per_tensor"] for p in pt)[len(pt) // 2]
            out["patch"]["entropy_floor_bits_global_median"] = sorted(
                p["entropy_floor_bits_global"] for p in pt)[len(pt) // 2]
    return out


def ifeval_deltas(R, variant):
    d = {}
    for (t, a, s, vn), r in R.items():
        ad = r.get("adaptive") or {}
        if vn == variant and ad.get("ifeval") and ad.get("ifeval_base"):
            v, b = ad["ifeval"].get(IFEVAL_METRIC), ad["ifeval_base"].get(IFEVAL_METRIC)
            if v is not None and b is not None:
                d[f"{t}|{a}|s{s}"] = v - b
    return dict(per_unit=d, mean=float(np.mean(list(d.values()))) if d else None)


# ---------------------------------------------------------------- SEED ----
def seed_tier(R, cells):
    ref_d, sd_d = cellmeans(R, REF_ARM, deployable), cellmeans(R, SEED_ARM, deployable)
    ref_a, sd_a = cellmeans(R, REF_ARM, as_deployed), cellmeans(R, SEED_ARM, as_deployed)
    pairs = [(sd_d[c], ref_d[c]) for c in cells]
    out = dict(cells=[f"{t}|{a}" for t, a in cells],
               rho_deployable=boot_ratio(pairs),
               rho_as_deployed=boot_ratio([(sd_a[c], ref_a[c]) for c in cells]),
               cref_minus_cseed_deployable=V.boot_paired([(ref_d[c], sd_d[c]) for c in cells]),
               cref_minus_cseed_as_deployed=V.boot_paired([(ref_a[c], sd_a[c]) for c in cells]),
               per_cell={f"{t}|{a}": dict(c_ref=ref_d[(t, a)], c_seed=sd_d[(t, a)],
                                           c_ref_as_deployed=ref_a[(t, a)],
                                           c_seed_as_deployed=sd_a[(t, a)])
                         for t, a in cells},
               arms={REF_ARM: arm_stats(R, REF_ARM), SEED_ARM: arm_stats(R, SEED_ARM)},
               ifeval={REF_ARM: ifeval_deltas(R, REF_ARM),
                       SEED_ARM: ifeval_deltas(R, SEED_ARM)},
               screen={f"{t}|{a}": v for (t, a), v in screen(R, cells).items()})
    inenv = [c for c in cells if out["screen"][f"{c[0]}|{c[1]}"]["in_envelope"]]
    if len(inenv) < len(cells):
        out["rho_deployable_in_envelope_only"] = boot_ratio(
            [(sd_d[c], ref_d[c]) for c in inenv])
    return out, pairs


def seed_verdicts(big_n):
    res = {}
    panel, R, miss, need = complete(CFG["seed_small"])
    cells = sorted({(k[0], k[1]) for k in R}) if not miss else None
    small_pairs = None
    if miss:
        res["small"] = dict(status="pending", rows=f"{need - len(miss)}/{need}")
    else:
        d, small_pairs = seed_tier(R, cells)
        d["verdict"] = verdict3(d["rho_deployable"])
        ca = cellmeans(R, CA_ARM, deployable)
        sd = cellmeans(R, SEED_ARM, deployable)
        reg = sorted(REGISTERED10)
        d["descriptive_cseed_vs_ca_registered10"] = V.boot_paired([(sd[c], ca[c]) for c in reg])
        d["arms"][CA_ARM] = arm_stats(R, CA_ARM)
        res["small"] = d
    big_pairs = None
    if big_n is None:
        res["big"] = dict(status="pending", why="§v13.F cell count not recorded")
    else:
        panel, R, miss, need = complete(SEED_BIG_CFG.format(n=big_n))
        if miss:
            res["big"] = dict(status="pending", rows=f"{need - len(miss)}/{need}")
        else:
            cells_b = sorted({(k[0], k[1]) for k in R})
            d, big_pairs = seed_tier(R, cells_b)
            d["verdict"] = verdict3(d["rho_deployable"]) \
                if len(cells_b) >= BIG_CELLS_FOR_VERDICT else "DESCRIPTIVE (n<8)"
            res["big"] = d
    both = all(res[t].get("verdict") in ("NON-INFERIOR", "INFERIOR", "INCONCLUSIVE")
               for t in ("small", "big"))
    res["scale_contrast"] = boot_ratio_diff(big_pairs, small_pairs) if both else None
    sc = res["scale_contrast"]
    res["scale_claim"] = bool(sc and (sc["lo"] > 0 or sc["hi"] < 0))
    return res


# --------------------------------------------------------------- FLOOR ----
def floor_tier(R):
    cells = sorted({(k[0], k[1]) for k in R})
    out = dict(cells=[f"{t}|{a}" for t, a in cells])
    cm = {rule: {vn: cellmeans(R, vn, f) for vn in FLOOR_PTS}
          for rule, f in (("alpha_star", deployable), ("as_deployed", as_deployed),
                          ("frozen", frozen))}
    for rule, m in cm.items():
        out[f"ratio_{rule}"] = {vn: boot_ratio([(m[vn][c], m["s0.99"][c]) for c in cells])
                                for vn in FLOOR_PTS[1:]}
    # shift: R(alpha*) - R(frozen) at 0.1%, both from the same cell resamples
    rng = random.Random(V.BOOT_SEED)
    a, f = cm["alpha_star"], cm["frozen"]
    k = len(cells)
    bs = []
    for _ in range(V.BOOT_N):
        idx = [cells[rng.randrange(k)] for _ in range(k)]
        da, df = np.mean([a["s0.99"][c] for c in idx]), np.mean([f["s0.99"][c] for c in idx])
        if da > 0 and df > 0:
            bs.append(np.mean([a["s0.999"][c] for c in idx]) / da
                      - np.mean([f["s0.999"][c] for c in idx]) / df)
    bs.sort()
    ra, rf = out["ratio_alpha_star"]["s0.999"], out["ratio_frozen"]["s0.999"]
    out["shift_alpha_star_minus_frozen_at_0.1pct"] = dict(
        point=(ra["point"] - rf["point"]) if ra["point"] is not None and rf["point"] is not None
        else None, lo=bs[int(.025 * len(bs))] if bs else None,
        hi=bs[int(.975 * len(bs)) - 1] if bs else None, n=k, used_resamples=len(bs))
    import v12_pstar
    dens = {}
    for rule in ("alpha_star", "frozen"):
        for c in cells:
            curve = {"s0.0": cm[rule]["s0.99"][c]}
            curve.update({vn: cm[rule][vn][c] for vn in FLOOR_PTS})
            dens.setdefault(f"{c[0]}|{c[1]}", {})[rule] = v12_pstar.p_true(curve)
    out["density_keeping_90pct_of_1pct"] = dens
    out["arms"] = {vn: arm_stats(R, vn) for vn in FLOOR_PTS}
    out["censored_at_ladder_max"] = {
        vn: sum(1 for kk, r in R.items() if kk[3] == vn
                and ((r.get("adaptive") or {}).get("status") == "exhausted"))
        for vn in FLOOR_PTS}
    return out


def floor_verdicts():
    res = {}
    for tier, key in (("small", "floor_small"), ("big", "floor_big")):
        panel, R, miss, need = complete(CFG[key])
        if miss:
            res[tier] = dict(status="pending", rows=f"{need - len(miss)}/{need}")
            continue
        d = floor_tier(R)
        if tier == "small":
            d["verdict"] = verdict3(d["ratio_alpha_star"]["s0.999"],
                                    pos="FLOOR MOVES", neg="FLOOR HOLDS")
        else:
            d["verdict"] = "DESCRIPTIVE (n=4)"
        res[tier] = d
    return res


# ------------------------------------------------------------ E0.5 -------
def replay():
    """v13replay vs v12astar at every frozen evaluation point."""
    panel, R, miss, need = complete(CFG["replay"])
    if miss:
        print(f"replay incomplete: {need - len(miss)}/{need} rows")
        return None
    new, ref = trace("v13replay"), trace("v12astar")
    pts, bad = 0, []
    for k, t in new.items():
        if k not in ref:
            continue
        r = ref[k]
        pts += 1
        dm = abs(round((t["post_mmlu"] - r["post_mmlu"]) * 200))
        same = (t["post_skew"] == r["post_skew"] and t["post_ppl"] == r["post_ppl"]
                and t["pre_skew"] == r["pre_skew"] and t["pre_ppl"] == r["pre_ppl"])
        if not same or dm > MMLU_TOL:
            bad.append(dict(key="|".join(map(str, k)), skew_ppl_identical=same,
                            mmlu_items_diff=dm))
    ra = rows("v12astar")
    astar = {"|".join(map(str, k)): [(r.get("adaptive") or {}).get("alpha_star"),
                                      (ra.get(k, {}).get("adaptive") or {}).get("alpha_star")]
             for k, r in R.items()}
    ok = pts > 0 and not bad
    out = dict(points=pts, failures=bad, mmlu_tol_items=MMLU_TOL, passed=ok,
               alpha_star_new_vs_v12astar=astar)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "replay.json"), "w"), indent=1)
    print(f"E0.5 replay: {pts} frozen points, {len(bad)} failures "
          f"(tol {MMLU_TOL} MMLU items) -> {'PASS' if ok else 'FAIL'}")
    for b in bad:
        print("  ", b)
    return ok


def det():
    """Diagnostic only: does MMLU agree across two fresh processes?"""
    out = {}
    for mode in ("nd", "d"):
        a, b = trace(f"v13det_{mode}_a"), trace(f"v13det_{mode}_b")
        ks = sorted(set(a) & set(b))
        if not ks:
            out[mode] = dict(status="pending")
            continue
        diffs = collections.Counter(abs(round((a[k]["post_mmlu"] - b[k]["post_mmlu"]) * 200))
                                    for k in ks)
        pre = collections.Counter(abs(round((a[k]["pre_mmlu"] - b[k]["pre_mmlu"]) * 200))
                                  for k in ks)
        sp = sum(a[k]["post_skew"] == b[k]["post_skew"] and a[k]["post_ppl"] == b[k]["post_ppl"]
                 for k in ks)
        gate = sum(a[k]["collateral_ok"] != b[k]["collateral_ok"] for k in ks)
        out[mode] = dict(points=len(ks), post_mmlu_item_diffs=dict(sorted(diffs.items())),
                         pre_mmlu_item_diffs=dict(sorted(pre.items())),
                         skew_ppl_identical=sp, gate_decisions_differing=gate)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "determinism.json"), "w"), indent=1)
    for m, d in out.items():
        print(f"{'default' if m == 'nd' else 'deterministic'} kernels: {d}")
    return out


def progress():
    cfgs = dict(CFG)
    for n in (4, 8):
        if os.path.exists(os.path.join(REPO, SEED_BIG_CFG.format(n=n))):
            cfgs[f"seed_big_{n}"] = SEED_BIG_CFG.format(n=n)
    for m in ("nd", "d"):
        for r in ("a", "b"):
            cfgs[f"det_{m}_{r}"] = f"configs/v13/det_{m}_{r}.yaml"
    for k, c in cfgs.items():
        panel, _R, miss, need = complete(c)
        print(f"{k:14s} {panel:14s} {need - len(miss):4d}/{need} rows")


def big_n():
    """§v13.F, read from its registration line, never guessed."""
    fp = os.path.join(REPO, "PREREGISTRATION.md")
    for ln in open(fp):
        if ln.startswith("V13F_SEED_BIG_CELLS ="):
            return int(ln.split("=", 1)[1].strip())
    return None


def verdicts():
    out = dict(criterion="PREREGISTRATION §v13.B-E", bar=BAR,
               seed=seed_verdicts(big_n()), floor=floor_verdicts())
    s, f = out["seed"]["small"].get("verdict"), out["floor"]["small"].get("verdict")
    out["floor_seed_2b_licensed"] = bool(s == "NON-INFERIOR" and f == "FLOOR MOVES")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "verdicts.json"), "w"), indent=1, default=str)
    for exp in ("seed", "floor"):
        for tier in ("small", "big"):
            d = out[exp][tier]
            if "verdict" not in d:
                print(f"{exp.upper():5s} {tier:5s}: {d}")
                continue
            ci = d["rho_deployable"] if exp == "seed" else d["ratio_alpha_star"]["s0.999"]
            print(f"{exp.upper():5s} {tier:5s}: {d['verdict']:18s} ratio {ci['point']:.3f} "
                  f"[{ci['lo']:.3f}, {ci['hi']:.3f}] (n={ci['n']} cells)")
    print(f"SEED scale claim: {out['seed']['scale_claim']}; "
          f"FLOOR-2b licensed: {out['floor_seed_2b_licensed']}")
    print(f"wrote {os.path.join(OUT, 'verdicts.json')}")
    return 0


# ------------------------------------------------------------ selftest ----
def selftest():
    fails = []
    mk = lambda rem, ok=True, a=8.0: dict(removal=rem, alpha_frozen=2.0, adaptive=dict(
        alpha_star=a, eval_ok=ok, removal=rem, status="ok"))
    if deployable(mk(0.3, ok=False)) != 0.0 or as_deployed(mk(0.3, ok=False)) != 0.3:
        fails.append("deployable/as-deployed scoring")
    if deployable(mk(0.3, a=None)) != 0.0:
        fails.append("alpha* none must score 0")
    tight = [(0.95 + 0.001 * i, 1.0) for i in range(18)]
    if verdict3(boot_ratio(tight)) != "NON-INFERIOR":
        fails.append("tight ratio ~0.96 should be NON-INFERIOR")
    low = [(0.5 + 0.001 * i, 1.0) for i in range(18)]
    if verdict3(boot_ratio(low)) != "INFERIOR":
        fails.append("ratio ~0.5 should be INFERIOR")
    wide = [((1.6 if i % 2 else 0.1), 1.0) for i in range(18)]
    if verdict3(boot_ratio(wide)) != "INCONCLUSIVE":
        fails.append("wide ratio around 0.85 should be INCONCLUSIVE")
    if verdict3(boot_ratio(tight), "FLOOR MOVES", "FLOOR HOLDS") != "FLOOR MOVES":
        fails.append("FLOOR branch names")
    if boot_ratio([(0.1, -0.2), (0.1, 0.1)])["undefined_resamples"] == 0:
        fails.append("non-positive reference resamples must be counted")
    for f in fails:
        print("FAIL:", f)
    print(f"v13_analyze selftest: {'OK' if not fails else 'FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "progress"
    fn = {"replay": lambda: 0 if replay() else 1, "det": lambda: det() and 0,
          "progress": progress, "verdicts": verdicts, "selftest": selftest}[cmd]
    sys.exit(fn() or 0)
