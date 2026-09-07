"""v10 — compute the two manuscript numbers that had no script at all.

Both of these were printed in the paper with no artifact and no committed code
behind them. Neither needs GPU time: the inputs have been on disk the whole
while, only the analysis was missing.

  contrast_gap -> removal   sec 3: "+0.507 [+0.220, +0.732], n = 504".
                            contrast_gap is readable from every corpus file's
                            `diag`; removal comes from the edit panels. The join
                            is v8_env.load_gap_index() against the panel rows.
  ENV 200-split sweep       sec 5.8: "across 200 alternative splits the gate's
                            mean utility gain is -0.0012". v8_env already has
                            make_split(cells, seed), calibrate() and score();
                            nothing re-ran them over seeds.

Both are reported AS-RUN and under V9, because the v10 plan says the
contrast_gap correlation is explicitly not exempt from re-scoring: its inputs
are gated removals.

The v9 arm is necessarily PARTIAL. Four of the twelve edit panels ENV draws on
(t2x, v1, x, x2) can never be re-scored, so the v9 correlation and the v9 sweep
run on a strictly smaller row set. That reduction is reported, not absorbed.

Run: python3 src/v10_regen_missing.py
"""
import os, sys, json, math, collections, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v10_common as C
import v8_env as E

N_SPLITS = 200
SWEEP_SEEDS = range(N_SPLITS)


def build_rows(root, label):
    """The ENV row set, joined to contrast_gap, from a chosen results tree."""
    gap = E.load_gap_index()
    rows, missing_panels, unjoined = [], [], 0
    for rel, panel in E.EDIT_PANELS:
        fp = os.path.join(root, rel)
        if not os.path.exists(fp):
            missing_panels.append(panel)
            continue
        with open(fp) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                g = gap.get((r.get("designer"), r.get("axis"), r.get("seed")))
                if g is None:
                    unjoined += 1
                    continue
                rem = r.get("removal")
                bf = rem is None or (isinstance(rem, float) and rem != rem)
                rows.append(dict(
                    panel=panel, target=r["target"], axis=r["axis"], seed=r["seed"],
                    role=r.get("role"), designer=r["designer"],
                    removal=0.0 if bf else float(rem), budget_fail=bool(bf),
                    pre_skew=float(r["pre_skew"]),
                    contrast_gap=g["contrast_gap"], n_items=g["n_items"]))
    return rows, {"tree": label, "n_rows": len(rows),
                  "panels_absent": sorted(missing_panels),
                  "rows_unjoined_to_a_corpus": unjoined,
                  "n_corpora_with_gap": len(gap)}


# --------------------------------------------------- contrast_gap -> removal --

def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _boot_r(pairs, n=C.BOOT_N, seed=C.BOOT_SEED):
    """Percentile bootstrap of a correlation, resampling the CELL."""
    if len(pairs) < 3:
        return None
    rng = random.Random(seed)
    k = len(pairs)
    draws = []
    for _ in range(n):
        s = [pairs[rng.randrange(k)] for _ in range(k)]
        r = _pearson([p[0] for p in s], [p[1] for p in s])
        if r is not None:
            draws.append(r)
    draws.sort()
    point = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
    if point is None or not draws:
        return None
    return dict(point=point, lo=draws[int(.025 * len(draws))],
                hi=draws[int(.975 * len(draws)) - 1], n=k)


FROZEN_GAP = os.path.join(C.HERE, "results", "v6", "contrast_gap_frozen.json")


def designer_sizes():
    """designer -> nominal parameter count in B, from the frozen v6 artifact
    (the only place size_b is registered). Every designer in the ENV row set
    appears there; a missing one is a hard error, never a silent drop."""
    obs = json.load(open(FROZEN_GAP))["observations"]
    return {o["designer"]: float(o["size_b"]) for o in obs}


def _ols_gap_size(cells):
    """removal ~ 1 + contrast_gap + size_b, OLS. Returns (b_gap, b_size)."""
    X = np.array([[1.0, c["contrast_gap"], c["size_b"]] for c in cells], float)
    y = np.array([c["removal"] for c in cells], float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(b[1]), float(b[2])


def _boot_ols(cells, n=C.BOOT_N, seed=C.BOOT_SEED):
    """Percentile bootstrap of both coefficients, resampling the CELL - the
    same unit, RNG and draw order as _boot_r."""
    if len(cells) < 4:
        return None
    base = _ols_gap_size(cells)
    rng = random.Random(seed)
    k = len(cells)
    gs, ss = [], []
    for _ in range(n):
        smp = [cells[rng.randrange(k)] for _ in range(k)]
        try:
            bg, bs = _ols_gap_size(smp)
        except Exception:
            continue
        gs.append(bg); ss.append(bs)
    gs.sort(); ss.sort()
    q = lambda a: (a[int(.025 * len(a))], a[int(.975 * len(a)) - 1])
    glo, ghi = q(gs); slo, shi = q(ss)
    return dict(b_gap=base[0], gap_lo=glo, gap_hi=ghi,
                b_size=base[1], size_lo=slo, size_hi=shi, n=k)


def contrast_gap_headline(rows):
    """THE sec 3 estimand, on ONE named population: every joined edit-panel run,
    ALL designer roles pooled (contrast_gap is a DESIGNER-level diagnostic and
    the claim is a corpus-selection rule, so cross-designer variation is the
    signal, not a nuisance), aggregated to the project's registered resampling
    unit - the target x axis cell - and bootstrapped by resampling that cell.

    Reported beside it: whether designer size adds anything once contrast_gap is
    in the model ('absorbs designer size'), on the SAME cells."""
    sizes = designer_sizes()
    cells = E.cell_table(rows, role=None)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["target"], r["axis"])].append(r)
    for c in cells:
        rs = by[(c["target"], c["axis"])]
        missing = sorted({r["designer"] for r in rs if r["designer"] not in sizes})
        if missing:
            raise SystemExit(f"[v10] no registered size_b for designers {missing}")
        c["size_b"] = sum(sizes[r["designer"]] for r in rs) / len(rs)
    pairs = [(c["contrast_gap"], c["removal"]) for c in cells]
    cb = _boot_r(pairs)
    sb = _boot_r([(c["size_b"], c["removal"]) for c in cells])
    ols = _boot_ols(cells)
    return {
        "_what": ("Pearson correlation of corpus-build-time contrast_gap with "
                  "realized removal, ALL designer roles pooled, aggregated to "
                  "the target x axis cell"),
        "population": ("every row of the 12 frozen EDIT_PANELS that joins to a "
                       "corpus contrast_gap; all roles (self, cross, sibling, "
                       "cross_large, cross_small, same_large); nan removal -> 0; "
                       "no target, axis or designer excluded"),
        "unit": "target x axis cell (mean over designers, roles, panels, seeds)",
        "resampling": f"percentile bootstrap, {C.BOOT_N} resamples of the CELL, seed {C.BOOT_SEED}",
        "n_cells": len(cells), "n_rows": len(rows),
        "point": cb["point"], "lo": cb["lo"], "hi": cb["hi"],
        "text": f"{cb['point']:+.3f} [{cb['lo']:+.3f}, {cb['hi']:+.3f}] "
                f"(n={len(cells)} cells over {len(rows)} runs, {C.status(cb)})",
        "size_marginal": {"point": sb["point"], "lo": sb["lo"], "hi": sb["hi"],
                          "_what": "designer size alone vs removal, same cells"},
        "size_absorbed": {
            "b_size": ols["b_size"], "lo": ols["size_lo"], "hi": ols["size_hi"],
            "b_gap": ols["b_gap"], "gap_lo": ols["gap_lo"], "gap_hi": ols["gap_hi"],
            "_what": ("OLS removal ~ contrast_gap + size_b on the same cells; "
                      "'absorbs designer size' == this size coefficient covers 0 "
                      "while the gap coefficient does not"),
            "units": "removal per B parameters",
        },
    }

# ---- wiring, inside main()'s `for root, label in ((C.RES, ...), (C.V9, ...))` loop,
#      alongside the existing rep["trees"][label] assignments:
#
#            "contrast_gap_headline": contrast_gap_headline(rows),
#
# ---- and in rep["manuscript_claims"]["contrast_gap"], add:
#
#         "headline_v9": v["contrast_gap_headline"]["text"],
#         "headline_as_run": a["contrast_gap_headline"]["text"],
#         "printed_n_504_population": (
#             "reproduces ONLY under frozen-gap x as-run tree x roles "
#             "{self,cross,same_large} x rows with 'granite' in target or designer "
#             "dropped x cells keyed (designer,axis): 504 rows, 26 cells, "
#             "r_cell=+0.711 [+0.499, +0.860] - the shape matches, the value does not"),


def contrast_gap_correlation(rows, role="self"):
    """Row-level and cell-level correlation of contrast_gap with removal."""
    rs = [r for r in rows if role is None or r["role"] == role]
    row_pairs = [(r["contrast_gap"], r["removal"]) for r in rs]
    cells = E.cell_table(rs, role=role)
    cell_pairs = [(c["contrast_gap"], c["removal"]) for c in cells]
    rb, cb = _boot_r(row_pairs, seed=1), _boot_r(cell_pairs)
    return {
        "_what": "correlation of contrast_gap with realized removal",
        "role": role,
        "n_rows": len(row_pairs), "n_cells": len(cell_pairs),
        "row_level": (dict(rb, text=f"{rb['point']:+.3f} [{rb['lo']:+.3f}, "
                                    f"{rb['hi']:+.3f}] (n={rb['n']} rows)")
                      if rb else None),
        "cell_level": (dict(cb, text=f"{cb['point']:+.3f} [{cb['lo']:+.3f}, "
                                     f"{cb['hi']:+.3f}] (n={cb['n']} cells)")
                       if cb else None),
        "_note": ("the manuscript prints '+0.507 [+0.220, +0.732], n = 504', "
                  "i.e. a cell-level estimate labelled with the row count"),
    }


# ------------------------------------------------------ ENV 200-split sweep --

def split_sweep(cells, n_splits=N_SPLITS):
    """Re-run the frozen split rule over many seeds. For each: calibrate the
    gate on the calibration half, apply it to the held-out half, and record the
    utility gain over editing everything."""
    out, rec = {}, collections.defaultdict(list)
    nondiscriminating = collections.Counter()
    for seed in SWEEP_SEEDS if n_splits == N_SPLITS else range(n_splits):
        cal, hold, _exc = E.make_split(cells, seed=seed)
        if len(cal) < 3 or len(hold) < 3:
            continue
        th = E.calibrate(cal)
        base = E.score(hold)["utility"]
        for name, key in (("pre_skew_gate", "pre_skew_gate"),
                          ("contrast_gap_gate", "contrast_gap_gate"),
                          ("joint_gate", "joint_gate")):
            g = th[key]
            s = E.score(hold, tau=g.get("tau"), gamma=g.get("gamma"))
            rec[name].append(s["utility"] - base)
            if s["n_edited"] == len(hold):
                nondiscriminating[name] += 1
    for name, gains in rec.items():
        a = np.array(gains, float)
        out[name] = {
            "n_splits": int(a.size),
            "mean_utility_gain": float(a.mean()),
            "median_utility_gain": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "frac_positive": float((a > 0).mean()),
            "frac_exceeding_plus_0.01": float((a > 0.01).mean()),
            "n_nondiscriminating": int(nondiscriminating[name]),
            "frac_nondiscriminating": float(nondiscriminating[name] / a.size),
            "min": float(a.min()), "max": float(a.max()),
        }
    return out


def population_sensitivity(rows):
    """The contrast_gap -> removal correlation under every population a reader
    might reasonably assume. The manuscript prints ONE number; the estimate is
    not stable across these, which is the finding."""
    AX2 = {"occ_gender", "crows_socioeconomic"}
    pops = {
        "all roles": rows,
        "role=self": [r for r in rows if r["role"] == "self"],
        "role=cross": [r for r in rows if r["role"] == "cross"],
        "role=cross_large": [r for r in rows if r["role"] == "cross_large"],
        "role=cross_small": [r for r in rows if r["role"] == "cross_small"],
        "role=same_large": [r for r in rows if r["role"] == "same_large"],
        "role=sibling": [r for r in rows if r["role"] == "sibling"],
        "all roles, budget-fails dropped": [r for r in rows if not r["budget_fail"]],
        "occ+crows only": [r for r in rows if r["axis"] in AX2],
        "occ+crows, granite excluded": [r for r in rows if r["axis"] in AX2
                                        and not r["target"].startswith("granite")],
        "granite excluded": [r for r in rows if not r["target"].startswith("granite")],
    }
    out = {}
    for name, rs in pops.items():
        if len(rs) < 3:
            continue
        rp = [(r["contrast_gap"], r["removal"]) for r in rs]
        cells = E.cell_table(rs, role=None)
        cp = [(c["contrast_gap"], c["removal"]) for c in cells]
        rr, rc = _pearson([p[0] for p in rp], [p[1] for p in rp]), \
                 _pearson([p[0] for p in cp], [p[1] for p in cp])
        out[name] = {"n_rows": len(rs), "n_cells": len(cp),
                     "r_row": rr, "r_cell": rc}
    vals = [v["r_cell"] for v in out.values() if v["r_cell"] is not None]
    return {
        "_what": ("the printed +0.507 (n=504, 26 cells) reproduces under NONE of "
                  "these; no script or artifact exists to say which population "
                  "was intended"),
        "populations": out,
        "r_cell_range": [min(vals), max(vals)] if vals else None,
        "n_rows_range": [min(v["n_rows"] for v in out.values()),
                         max(v["n_rows"] for v in out.values())],
        "printed_n_rows": 504, "printed_n_cells": 26, "printed_r": 0.507,
        "any_population_matches_printed_n": any(
            v["n_rows"] == 504 and v["n_cells"] == 26 for v in out.values()),
    }


def main():
    C.ensure_dirs()
    rep = {"artifact": "v10_regen_missing", "module": "src/v10_regen_missing.py",
           "purpose": ("compute the two manuscript numbers that had no script; "
                       "analysis only, no GPU"),
           "n_splits": N_SPLITS, "trees": {}}

    for root, label in ((C.RES, "as_run"), (C.V9, "v9")):
        rows, prov = build_rows(root, label)
        cells = E.cell_table(rows, role="self")
        rep["trees"][label] = {
            "provenance": prov,
            "n_cells_self": len(cells),
            "contrast_gap_correlation": contrast_gap_correlation(rows),
            "contrast_gap_headline": contrast_gap_headline(rows),
            "contrast_gap_population_sensitivity": population_sensitivity(rows),
            "env_split_sweep": split_sweep(cells),
        }

    a, v = rep["trees"]["as_run"], rep["trees"]["v9"]
    rep["v9_restriction"] = {
        "rows_lost": a["provenance"]["n_rows"] - v["provenance"]["n_rows"],
        "cells_lost": a["n_cells_self"] - v["n_cells_self"],
        "panels_absent_under_v9": v["provenance"]["panels_absent"],
        "note": ("t2x, v1, x and x2 can never be re-scored, so the v9 arm runs on "
                 "a strictly smaller row set"),
    }
    rep["manuscript_claims"] = {
        "contrast_gap": {
            "printed": "+0.507 [+0.220, +0.732], n = 504",
            "as_run_cell": (a["contrast_gap_correlation"]["cell_level"] or {}).get("text"),
            "as_run_rows": a["contrast_gap_correlation"]["n_rows"],
            "v9_cell": (v["contrast_gap_correlation"]["cell_level"] or {}).get("text"),
            "headline_v9": (v.get("contrast_gap_headline") or {}).get("text"),
            "headline_as_run": (a.get("contrast_gap_headline") or {}).get("text"),
            "printed_n504_population_note": (
                "the printed n=504/26-cell shape reproduces ONLY under "
                "frozen-gap x as-run tree x roles {self,cross,same_large} x "
                "granite dropped x cells keyed (designer,axis), and gives "
                "r_cell=+0.711 - the shape matches, the value does not"),
            "v9_rows": v["contrast_gap_correlation"]["n_rows"],
        },
        "env_200_splits": {
            "printed": ("across 200 alternative splits the gate's mean utility "
                        "gain is -0.0012"),
            "as_run_joint": a["env_split_sweep"].get("joint_gate", {}).get("mean_utility_gain"),
            "as_run_pre_skew": a["env_split_sweep"].get("pre_skew_gate", {}).get("mean_utility_gain"),
            "v9_joint": v["env_split_sweep"].get("joint_gate", {}).get("mean_utility_gain"),
            "v9_pre_skew": v["env_split_sweep"].get("pre_skew_gate", {}).get("mean_utility_gain"),
        },
    }
    fp = C.jdump(rep, "results/v10/missing_recomputed.json")

    print("[v10 missing]")
    for label in ("as_run", "v9"):
        t = rep["trees"][label]
        cg = t["contrast_gap_correlation"]
        print(f"  --- {label}: {t['provenance']['n_rows']} rows, "
              f"{t['n_cells_self']} self cells"
              + (f", panels absent {t['provenance']['panels_absent']}"
                 if t["provenance"]["panels_absent"] else ""))
        print(f"    contrast_gap -> removal   cell {cg['cell_level']['text'] if cg['cell_level'] else 'n/a'}")
        print(f"    {'':26s}row  {cg['row_level']['text'] if cg['row_level'] else 'n/a'}")
        ps = t["contrast_gap_population_sensitivity"]
        print(f"    population sensitivity: r_cell spans "
              f"{ps['r_cell_range'][0]:+.3f} .. {ps['r_cell_range'][1]:+.3f} over "
              f"{len(ps['populations'])} populations; matches printed n = "
              f"{ps['any_population_matches_printed_n']}")
        for name, d in sorted(t["env_split_sweep"].items()):
            print(f"    {name:18s} mean {d['mean_utility_gain']:+.5f}  "
                  f"median {d['median_utility_gain']:+.5f}  "
                  f">+0.01 in {100*d['frac_exceeding_plus_0.01']:.1f}%  "
                  f"non-discriminating {d['n_nondiscriminating']}/{d['n_splits']}")
    print(f"  v9 restriction: -{rep['v9_restriction']['rows_lost']} rows, "
          f"-{rep['v9_restriction']['cells_lost']} cells")
    print(f"wrote {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
