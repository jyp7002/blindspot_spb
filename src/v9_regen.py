"""WS1.3/1.4 — regenerate every quoted number under the v9 gate, and diff it.

Reads results_v9/ (written by v9_rescore.py) and results/ (as-run), computes each
headline estimand both ways, and emits the diff table CLOSEOUT_V9 §1.4 asks for:
v8 value -> v9 value -> CI-status change.

WHAT CANNOT BE REGENERATED, and why it is stated rather than papered over.
Five panels predate the v6 alpha-trace fix and store only a scalar removal:
t2x, v1, x, x2 and v6trace/ste. Any estimand that depends on them is marked
BLOCKED or MIXED:

  * The BL-S/BL-T frontier (D2) draws its EDIT arm from results/x (llama, qwen)
    and results/x2 (gemma, phi) — verified by pre_skew fingerprint against the
    steer panel. results/x2 has an exact re-scorable twin (v6trace/x2 agrees
    108/108, 0 differing), so gemma and phi CAN be re-scored. results/x has no
    twin, so llama and qwen CANNOT.
  * §11b STE (v6trace/ste) has no trace at all -> BLOCKED.

Bootstrap resamples the target x axis CELL, 10k, seed 0 — unchanged from v8.
"""
import os, sys, json, glob, collections, random
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
V9 = os.path.join(HERE, "results_v9")
OUT = os.path.join(HERE, "results", "v9")


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and x != x) else float(x)


def rows(root, panel):
    fp = os.path.join(root, panel, "removal.jsonl")
    if not os.path.exists(fp):
        return []
    return [json.loads(l) for l in open(fp) if l.strip()]


def boot_paired(pairs, n=10000, seed=0):
    d = [a - b for a, b in pairs]
    if not d:
        return None
    rng = random.Random(seed)
    bs = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in d])) for _ in range(n))
    return dict(point=float(np.mean(d)), lo=bs[int(.025 * n)], hi=bs[int(.975 * n) - 1],
                n=len(d))


def status(d):
    if d is None:
        return "n/a"
    return "excludes 0" if (d["lo"] > 0 or d["hi"] < 0) else "covers 0"


def fmt(d):
    return "n/a" if d is None else \
        f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] (n={d['n']}, {status(d)})"


# ------------------------------------------------------------------- DEC ----

def dec(root):
    rs = rows(root, "v8dec")
    seen = {}
    for r in rs:
        seen[(r["target"], r["axis"], r["seed"], r["designer"], r["condition"])] = r
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in seen.values():
        cells[(r["target"], r["axis"])][r["condition"]].append(z(r["removal"]))
    C = {k: {c: float(np.mean(v)) for c, v in d.items()} for k, d in cells.items()}
    out = {}
    for name, a, b in (("Delta_selection", "C-ref", "C-a"),
                       ("Delta_sign", "C-ref", "C-b"),
                       ("Delta_null", "C-ref", "C-rand"),
                       ("Delta_location", "C-ref", "C-layershuf"),
                       ("Delta_tensor", "C-ref", "C-tensorshuf"),
                       ("C-a minus C-rand", "C-a", "C-rand")):
        out[name] = boot_paired([(m[a], m[b]) for m in C.values() if a in m and b in m])
    out["_means"] = {c: float(np.mean([m[c] for m in C.values() if c in m]))
                     for c in ("C-ref", "C-a", "C-rand", "C-b", "C-bottom",
                               "C-layershuf", "C-tensorshuf")}
    out["_unanimity"] = sum(1 for m in C.values()
                            if "C-ref" in m and "C-a" in m and m["C-ref"] > m["C-a"])
    return out


# ------------------------------------------------------------------- SPC ----

def spc(root, tier):
    rs = rows(root, f"v8spc/{tier}")
    c = collections.defaultdict(dict)
    for r in rs:
        c.setdefault(r["target"], {}).setdefault(r["sparsity"], []).append(z(r["removal"]))
    out = {}
    for sp in (0.9, 0.99, 0.995, 0.999):
        pairs = [(float(np.mean(v[sp])), float(np.mean(v[0.0])))
                 for v in c.values() if sp in v and 0.0 in v]
        out[f"s={sp}"] = boot_paired(pairs)
    return out


# -------------------------------------------------------------- frontier ----

def best_in_budget(rs, keyf, filt=lambda r: True):
    """Best in-budget score per key. Panels name the score `bias_reduction`
    (steer/sentdebias/inlp) or `removal` (dpo); accept either."""
    out = collections.defaultdict(lambda: float("nan"))
    for r in rs:
        if not filt(r) or not r.get("collateral_ok"):
            continue
        br = r.get("bias_reduction", r.get("removal"))
        if br is None:
            continue
        k = keyf(r)
        cur = out[k]
        out[k] = br if cur != cur else max(cur, br)
    return dict(out)


def edit_arm(root):
    """Edit self-removal per (target, axis, seed) for the frontier's 8 cells.

    gemma/phi from v6trace/x2 (re-scorable twin of results/x2, verified 108/108).
    llama/qwen from results/x, which has NO trace -> always the as-run value.
    """
    out, provenance = {}, {}
    for r in rows(root, "v6trace/x2"):
        if r.get("role") == "self":
            out[(r["target"], r["axis"], r["seed"])] = z(r["removal"])
            provenance[r["target"]] = "v6trace/x2 (re-scorable)"
    for r in rows(RES, "x"):                       # as-run only; no v9 twin
        if r.get("role") == "self":
            out[(r["target"], r["axis"], r["seed"])] = z(r["removal"])
            provenance[r["target"]] = "results/x (NOT re-scorable — as-run)"
    return out, provenance


def frontier(root):
    st = rows(root, "v6trace/steer")
    sd = rows(root, "v6trace/sentdebias")
    dp = rows(root, "v6trace/dpo")
    steer = best_in_budget(st, lambda r: (r["target"], r["axis"], r["seed"]),
                           lambda r: r.get("kind") == "real" and r.get("grid") == "primary")
    sdeb = best_in_budget(sd, lambda r: (r["target"], r["axis"], r["seed"]),
                          lambda r: r.get("kind") == "real")
    dpo = best_in_budget(dp, lambda r: (r["target"], r["axis"], r["seed"]),
                         lambda r: r.get("method") == "dpo")
    ed, prov = edit_arm(root)

    def to_cells(d):
        """Aggregate (target,axis,seed) -> (target,axis) by mean over seeds.
        The project bootstraps the CELL, never the seed: per-seed pooling is
        anti-conservative and produced the v4 self-advantage artifact that v5
        had to retract."""
        acc = collections.defaultdict(list)
        for (t, a, _s), v in d.items():
            acc[(t, a)].append(z(v))
        return {k: float(np.mean(v)) for k, v in acc.items()}

    edc = to_cells(ed)
    # The frontier's 8 headline cells (RESULTS_METHOD 11).
    HEADLINE = [(t, a) for t in ("gemma", "llama", "qwen", "phi")
                for a in ("occ_gender", "crows_socioeconomic")]
    res = {}
    for name, other in (("edit - steering", steer), ("edit - SentenceDebias", sdeb),
                        ("edit - DPO", dpo)):
        oc = to_cells(other)
        # nan->0 with counts: a cell where the baseline never cleared the budget
        # (or was skipped for too few preference pairs) scores 0 and is KEPT.
        # Dropping it would silently compare over different cell sets.
        pairs = [(edc[k], oc.get(k, 0.0)) for k in HEADLINE if k in edc]
        res[name] = boot_paired(pairs)
        res[f"_{name}_zeroed_cells"] = [f"{t}|{a}" for (t, a) in HEADLINE
                                        if (t, a) in edc and (t, a) not in oc]
    # the subset where BOTH arms are re-scorable
    sc = to_cells(steer)
    res["edit - steering [re-scorable cells only]"] = boot_paired(
        [(edc[k], sc[k]) for k in edc if k in sc and k[0] in ("gemma", "phi")])
    res["_edit_provenance"] = prov
    return res


# ------------------------------------------------------------------- main ----

def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"artifact": "v9_regen", "blocked": {}, "diff": {}}

    groups = [("DEC", dec), ("SPC small", lambda r: spc(r, "small")),
              ("SPC big", lambda r: spc(r, "big")), ("frontier", frontier)]

    for gname, fn in groups:
        a, b = fn(RES), fn(V9)
        for k in sorted(set(a) | set(b)):
            if k.startswith("_"):
                continue
            da, db = a.get(k), b.get(k)
            flip = (status(da) != status(db))
            report["diff"][f"{gname} :: {k}"] = dict(
                v8=fmt(da), v9=fmt(db),
                v8_point=None if da is None else round(da["point"], 6),
                v9_point=None if db is None else round(db["point"], 6),
                ci_status_change=("CHANGED: %s -> %s" % (status(da), status(db)))
                if flip else "none")
        if "_means" in b:
            report[f"{gname}_means_v9"] = {k: round(v, 4) for k, v in b["_means"].items()}
            report[f"{gname}_means_v8"] = {k: round(v, 4) for k, v in a["_means"].items()}
            report[f"{gname}_unanimity"] = dict(v8=a["_unanimity"], v9=b["_unanimity"])
        if "_edit_provenance" in b:
            report["frontier_edit_provenance"] = b["_edit_provenance"]

    report["blocked"] = {
        "v6trace/ste (§11b STE vs sign_fp)": "no alpha_trace; cannot be re-scored",
        "frontier edit arm: llama, qwen": "results/x has no trace and no re-scorable "
                                          "twin; those cells stay as-run, so the "
                                          "8-cell frontier is MIXED-gate",
        "t2x / v1 / x / x2": "v5-era panels, no traces; boundary exposure UNKNOWN",
    }

    fp = os.path.join(OUT, "v9_diff_report.json")
    json.dump(report, open(fp, "w"), indent=1)

    print("=" * 78)
    print("v9 DIFF REPORT — every regenerated estimand, as-run -> v9")
    print("=" * 78)
    for k, v in report["diff"].items():
        mark = "  <<< " + v["ci_status_change"] if v["ci_status_change"] != "none" else ""
        print(f"\n{k}")
        print(f"   v8: {v['v8']}")
        print(f"   v9: {v['v9']}{mark}")
    print("\n" + "-" * 78)
    print("DEC per-condition means (v8 -> v9):")
    for c, v8 in report.get("DEC_means_v8", {}).items():
        v9v = report["DEC_means_v9"].get(c)
        print(f"   {c:14s} {v8:+.4f} -> {v9v:+.4f}")
    u = report.get("DEC_unanimity", {})
    print(f"   unanimity (C-ref > C-a): {u.get('v8')}/10 -> {u.get('v9')}/10")
    print("\nfrontier edit-arm provenance:")
    for t, p in report.get("frontier_edit_provenance", {}).items():
        print(f"   {t:8s} {p}")
    print("\nBLOCKED / MIXED:")
    for k, v in report["blocked"].items():
        print(f"   {k}: {v}")
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
