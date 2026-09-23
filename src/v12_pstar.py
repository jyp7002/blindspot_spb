"""v12 — predict the retained fraction p* from |ΔW| geometry; validate held-out.

    python3 src/v12_pstar.py fit        # calibration panel only; LOO report
    python3 src/v12_pstar.py predict    # freeze p* for the held-out cells
    python3 src/v12_pstar.py validate   # after the held-out curves exist

THE CLAIM UNDER TEST. The published method keeps a fixed 1% of coordinates.
v12's claim is that the fraction an edit needs is readable from the edit: fit a
rule on small models (v12cal, 2.6-8B), apply it unchanged to 27B/32B (v12big),
and check it there. For that check to mean anything the 27B/32B curves must not
exist when the rule is frozen, and this script enforces it: `predict` refuses if
the held-out panel has a single curve row, and run_opsel refuses to measure a
held-out curve without this script's prediction (v12_opsel.load_pstar).

DEFINITIONS (registered in PREREGISTRATION.md §v12.B; defaults below).

  cell        (target, axis). Its value at a variant is the MEAN over seeds of
              the frozen-rule removal, nan (no in-budget α) -> 0. The project's
              unit is the cell, never the seed.
  dense       the cell's removal at s0.0 (one-bit, no sparsity).
  R(p)        removal at retained fraction p divided by dense.
  p_true      PRIMARY ("last_success"): the smallest grid p with R(p) >= RHO,
              log-interpolated toward the next smaller grid point (where R <
              RHO). This is the sparsest edit that still meets the target, which
              is what would be deployed. The published curves are not monotone
              (gemma dips at 97% and recovers at 99%; qwen-3B's dense edit is
              budget-limited and sits BELOW its sparse ones), so a walk that
              stops at the first dip would call gemma a 3.5% cell when 1% works.
              SENSITIVITY ("first_failure"): walk down from p=1 and stop at the
              first grid point below RHO -- reported, never used to fit.
              If R >= RHO down to the smallest grid p the cell is CENSORED there
              (p_true <= that); it enters the fit at the bound, labelled.
  eligible    dense >= FLOOR. A cell with nothing to retain has no p_true;
              it is reported, never silently dropped.

  Rules (all fit on eligible calibration cells only):
    mass      tau = median_c M_c(p_true_c);   p* = M^-1(tau)          PRIMARY
    pr        log p* = a + log pr_frac,        a = median residual
    ent       log p* = a + log ent_frac,       a = median residual
    gini      log p* = a + b * gini,           OLS
    fixed     p* = 0.01                        the published constant, the
                                               thing every rule must beat

  LOO. Every rule is also scored leave-one-TARGET-out inside the calibration
  set (all cells of a target held out together). That is the calibration-only
  evidence for choosing among rules, and it never touches v12big.
"""
import argparse
import datetime
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v12_geometry as G          # noqa: E402
from v12_opsel import file_sha    # noqa: E402

RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
CAL_PANEL = "v12cal"
HELD_PANEL = "v12big"
OUT_DIR = os.path.join(RESULTS, "v12")
PRED_FP = os.path.join(OUT_DIR, "pstar_prediction.json")
FIT_FP = os.path.join(OUT_DIR, "pstar_fit.json")
VAL_FP = os.path.join(OUT_DIR, "pstar_validation.json")

RHO = 0.90          # retention target
FLOOR = 0.05        # minimum dense removal for a cell to have a p_true
RULES = ("mass", "pr", "ent", "gini", "fixed")
PRIMARY = "mass"


# ------------------------------------------------------------------ io ----
def _rows(panel, fname):
    fp = os.path.join(RESULTS, panel, fname)
    out = []
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def _last(rows, key):
    """Dedupe keeping the LAST occurrence (append-only resume semantics)."""
    d = {}
    for r in rows:
        d[key(r)] = r
    return list(d.values())


def curves(panel):
    """{(target, axis): {variant: cell-mean frozen removal}} plus seed counts."""
    rows = _last(_rows(panel, "removal.jsonl"),
                 lambda r: (r["target"], r["axis"], r["seed"], r["variant"]))
    by = {}
    for r in rows:
        v = r["removal"]
        v = 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v
        by.setdefault((r["target"], r["axis"]), {}).setdefault(
            r["variant"], {})[r["seed"]] = v
    out = {}
    for cell, vs in by.items():
        out[cell] = {vn: float(np.mean(list(sd.values()))) for vn, sd in vs.items()}
        out[cell]["_seeds"] = {vn: sorted(sd) for vn, sd in vs.items()}
    return out


def geometries(panel):
    """{(target, axis): {seed: geometry}} (last occurrence wins)."""
    rows = _last(_rows(panel, "geometry.jsonl"),
                 lambda r: (r["target"], r["axis"], r["seed"], r["designer"]))
    out = {}
    for r in rows:
        out.setdefault((r["target"], r["axis"]), {})[r["seed"]] = r["geometry"]
    return out


# ------------------------------------------------------------- p_true ----
def spc_points(curve):
    """[(p, R(p))] descending in p, from a cell's variant means."""
    dense = curve.get("s0.0")
    if dense is None:
        return None, []
    pts = []
    for vn, val in curve.items():
        if vn.startswith("s") and vn[1:].replace(".", "", 1).isdigit():
            p = 1.0 - float(vn[1:])
            if p > 0:
                pts.append((p, val / dense if dense else float("nan")))
    return dense, sorted(pts, reverse=True)


def p_true(curve, rho=RHO, floor=FLOOR, mode="last_success"):
    """-> dict(p_true, status, dense). status: ok | censored | ineligible |
    fails_dense | missing."""
    dense, pts = spc_points(curve)          # descending p
    if dense is None or not pts:
        return dict(p_true=None, status="missing", dense=dense)
    if dense < floor:
        return dict(p_true=None, status="ineligible", dense=dense)

    def _interp(ok, bad):
        (p0, R0), (p1, R1) = ok, bad
        f = (R0 - rho) / (R0 - R1) if R0 != R1 else 1.0
        return math.exp(math.log(p0) + f * (math.log(p1) - math.log(p0)))

    if mode == "first_failure":
        last_ok = None
        for p, R in pts:
            if R >= rho:
                last_ok = (p, R)
                continue
            if last_ok is None:
                return dict(p_true=p, status="fails_dense", dense=dense)
            return dict(p_true=_interp(last_ok, (p, R)), status="ok", dense=dense)
        return dict(p_true=pts[-1][0], status="censored", dense=dense)

    if mode != "last_success":
        raise ValueError(mode)
    ok_idx = [i for i, (_p, R) in enumerate(pts) if R >= rho]
    if not ok_idx:
        return dict(p_true=pts[0][0], status="fails_dense", dense=dense)
    i = ok_idx[-1]                           # smallest p meeting rho
    if i == len(pts) - 1:
        return dict(p_true=pts[-1][0], status="censored", dense=dense)
    return dict(p_true=_interp(pts[i], pts[i + 1]), status="ok", dense=dense)


def retention_at(curve, p):
    """R(p) from a cell's measured grid, log-interpolated (for LOO scoring)."""
    dense, pts = spc_points(curve)
    if not pts or not dense:
        return float("nan")
    asc = sorted(pts)
    return G._interp_logx(asc, p)


# -------------------------------------------------------------- rules ----
def fit_rules(fit_cells):
    """fit_cells: list of dict(geom=<cell geometry>, p_true=float)."""
    lp = np.array([math.log(c["p_true"]) for c in fit_cells])
    params = {}
    params["mass"] = dict(tau=float(np.median(
        [G.mass_at(c["geom"], c["p_true"]) for c in fit_cells])))
    for f in ("pr", "ent"):
        x = np.array([math.log(c["geom"][f + "_frac"]) for c in fit_cells])
        params[f] = dict(a=float(np.median(lp - x)))
    g = np.array([c["geom"]["gini"] for c in fit_cells])
    if len(fit_cells) >= 3 and np.ptp(g) > 0:
        b, a = np.polyfit(g, lp, 1)
        params["gini"] = dict(a=float(a), b=float(b))
    else:
        params["gini"] = None
    params["fixed"] = dict(p=0.01)
    return params


def predict_p(rule, params, geom, margin=1.0):
    pr = params.get(rule)
    if pr is None:
        return None
    if rule == "mass":
        p = G.p_for_mass(geom, pr["tau"])
    elif rule in ("pr", "ent"):
        p = math.exp(pr["a"] + math.log(geom[rule + "_frac"]))
    elif rule == "gini":
        p = math.exp(pr["a"] + pr["b"] * geom["gini"])
    elif rule == "fixed":
        p = pr["p"]
    else:
        raise ValueError(rule)
    return float(min(1.0, max(1e-4, p * margin)))


# ---------------------------------------------------------------- fit ----
def calibration(rho, floor):
    cv, gm = curves(CAL_PANEL), geometries(CAL_PANEL)
    if not cv:
        raise SystemExit(f"no calibration rows in {RESULTS}/{CAL_PANEL}; "
                         "run configs/v12/opsel_calib.yaml first")
    cells, report = [], []
    for cell in sorted(cv):
        pt = p_true(cv[cell], rho, floor)
        geom = G.mean_geometry(list(gm[cell].values())) if cell in gm else None
        ff = p_true(cv[cell], rho, floor, mode="first_failure")
        rec = dict(cell="|".join(cell), **pt,
                   p_true_first_failure=ff["p_true"],
                   status_first_failure=ff["status"],
                   n_seeds=len(gm.get(cell, {})))
        report.append(rec)
        if pt["status"] in ("ok", "censored") and geom is not None:
            cells.append(dict(cell=cell, target=cell[0], geom=geom,
                              p_true=pt["p_true"], status=pt["status"],
                              curve=cv[cell]))
    return cells, report


def loo(cells, rho, margin):
    """Leave-one-target-out scoring of every rule, calibration only."""
    out = {r: [] for r in RULES}
    for t in sorted({c["target"] for c in cells}):
        train = [c for c in cells if c["target"] != t]
        test = [c for c in cells if c["target"] == t]
        if len(train) < 2:
            continue
        params = fit_rules(train)
        for r in RULES:
            for c in test:
                ph = predict_p(r, params, c["geom"], margin)
                if ph is None:
                    continue
                Rh = retention_at(c["curve"], ph)
                out[r].append(dict(cell="|".join(c["cell"]), p_hat=ph,
                                   p_true=c["p_true"],
                                   log2_err=math.log2(ph / c["p_true"]),
                                   retention_at_p_hat=Rh,
                                   meets_rho=bool(Rh >= rho)))
    summ = {}
    for r, xs in out.items():
        if not xs:
            summ[r] = None
            continue
        summ[r] = dict(n=len(xs),
                       median_abs_log2_err=float(np.median([abs(x["log2_err"]) for x in xs])),
                       frac_meets_rho=float(np.mean([x["meets_rho"] for x in xs])),
                       median_p_hat=float(np.median([x["p_hat"] for x in xs])),
                       per_cell=xs)
    return summ


def cmd_fit(a):
    cells, report = calibration(a.rho, a.floor)
    if len(cells) < 3:
        raise SystemExit(f"only {len(cells)} eligible calibration cells; "
                         "cannot fit (need >= 3)")
    params = fit_rules(cells)
    res = dict(created=_now(), rho=a.rho, floor=a.floor, margin=a.margin,
               primary=a.rule, params=params,
               fit_cells=["|".join(c["cell"]) for c in cells],
               fit_targets=sorted({c["target"] for c in cells}),
               calibration_cells=report,
               loo=loo(cells, a.rho, a.margin),
               inputs=_input_shas(CAL_PANEL))
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(FIT_FP, "w"), indent=1)
    print(f"calibration cells: {len(report)} ({len(cells)} eligible)")
    for rec in report:
        pt = rec["p_true"]
        print(f"  {rec['cell']:28s} dense={_f(rec['dense'])}  "
              f"p_true={'-' if pt is None else f'{pt:.4f}'}  {rec['status']}")
    print("\nleave-one-target-out (calibration only):")
    for r in RULES:
        s = res["loo"][r]
        if s:
            print(f"  {r:6s} n={s['n']:2d}  median|log2 err|={s['median_abs_log2_err']:.2f}"
                  f"  meets rho={s['frac_meets_rho']:.2f}  median p_hat={s['median_p_hat']:.4f}")
    print(f"\nprimary rule: {a.rule}  params: {params[a.rule]}")
    print(f"wrote {os.path.relpath(FIT_FP, REPO)}")


# ------------------------------------------------------------ predict ----
def cmd_predict(a):
    held_rows = _rows(HELD_PANEL, "removal.jsonl")
    if held_rows and not a.i_know:
        raise SystemExit(
            f"REFUSED: {HELD_PANEL}/removal.jsonl already has {len(held_rows)} "
            "curve rows. A p* predicted now is not a held-out prediction. "
            "(experiments_v12.md §II. There is deliberately no override that "
            "produces a file run_opsel will accept as held-out.)")
    if not os.path.exists(FIT_FP):
        raise SystemExit("run `v12_pstar.py fit` first")
    fit = json.load(open(FIT_FP))
    if fit["inputs"] != _input_shas(CAL_PANEL):
        raise SystemExit("calibration artifacts changed since `fit`; re-run fit")
    gm = geometries(HELD_PANEL)
    if not gm:
        raise SystemExit(f"no phase-A geometry in {RESULTS}/{HELD_PANEL}; run "
                         "`scripts/plan.py configs/v12/opsel_big.yaml --phase geometry`")
    seeds_needed = set(a.seeds)
    cells = {}
    for cell, by_seed in sorted(gm.items()):
        if cell[0] in fit["fit_targets"]:
            raise SystemExit(f"{cell[0]} is in the fit population; not held out")
        missing = seeds_needed - set(by_seed)
        if missing:
            raise SystemExit(f"{'|'.join(cell)}: phase-A geometry missing for "
                             f"seeds {sorted(missing)}")
        geom = G.mean_geometry([by_seed[s] for s in sorted(seeds_needed)])
        per_rule = {r: predict_p(r, fit["params"], geom, fit["margin"])
                    for r in RULES}
        cells["|".join(cell)] = dict(
            p_star=per_rule[fit["primary"]], rule=fit["primary"], per_rule=per_rule,
            geom_shas_by_seed={str(s): by_seed[s]["geom_sha"]
                               for s in sorted(seeds_needed)},
            mass_at_1pct=G.mass_at(geom, 0.01))
    pred = dict(created=_now(), fit=dict(targets=fit["fit_targets"],
                                         cells=fit["fit_cells"],
                                         params=fit["params"], rho=fit["rho"],
                                         floor=fit["floor"], margin=fit["margin"],
                                         primary=fit["primary"],
                                         fit_file_sha=file_sha(FIT_FP)),
                heldout_panel=HELD_PANEL, cells=cells,
                heldout_geometry_sha=file_sha(
                    os.path.join(RESULTS, HELD_PANEL, "geometry.jsonl")))
    if a.i_know:
        pred["NOT_HELD_OUT"] = True
        pred["fit"]["targets"] = pred["fit"]["targets"] + sorted(
            {c.split("|")[0] for c in cells})       # load_pstar will refuse it
    json.dump(pred, open(PRED_FP, "w"), indent=1)
    for c, d in cells.items():
        pr = "  ".join(f"{r}={p:.4f}" for r, p in d["per_rule"].items() if p)
        print(f"  {c:24s} p*={d['p_star']:.5f}  [{pr}]")
    print(f"wrote {os.path.relpath(PRED_FP, REPO)}  sha={file_sha(PRED_FP)[:12]}")
    print("commit it, then ship it to the GPU node for phase B")


# ----------------------------------------------------------- validate ----
def cmd_validate(a):
    pred = json.load(open(PRED_FP))
    rho = pred["fit"]["rho"]
    cv = curves(HELD_PANEL)
    out, rows = dict(prediction_sha=file_sha(PRED_FP), rho=rho, cells={}), []
    for c, d in pred["cells"].items():
        cell = tuple(c.split("|"))
        curve = cv.get(cell)
        if not curve or "pstar" not in curve or "s0.0" not in curve:
            out["cells"][c] = dict(status="not measured yet")
            continue
        dense = curve["s0.0"]
        pt = p_true(curve, rho, pred["fit"]["floor"])
        rec = dict(p_star=d["p_star"], dense=dense,
                   removal_at_pstar=curve["pstar"],
                   retention_at_pstar=curve["pstar"] / dense if dense else None,
                   retention_at_1pct=curve.get("s0.99", float("nan")) / dense if dense else None,
                   p_true=pt["p_true"], p_true_status=pt["status"],
                   log2_err=(math.log2(d["p_star"] / pt["p_true"])
                             if pt["p_true"] else None),
                   seeds=curve["_seeds"].get("pstar"))
        rec["meets_rho"] = bool(rec["retention_at_pstar"] is not None
                                and rec["retention_at_pstar"] >= rho)
        out["cells"][c] = rec
        rows.append(rec)
    json.dump(out, open(VAL_FP, "w"), indent=1)
    print(f"held-out validation (rho={rho}):")
    for c, r in out["cells"].items():
        if "p_star" not in r:
            print(f"  {c:24s} {r['status']}")
            continue
        print(f"  {c:24s} p*={r['p_star']:.4f}  R(p*)={_f(r['retention_at_pstar'])}"
              f"  R(1%)={_f(r['retention_at_1pct'])}  p_true={_f(r['p_true'])}"
              f" ({r['p_true_status']})  {'MEETS' if r['meets_rho'] else 'misses'} rho")
    print(f"wrote {os.path.relpath(VAL_FP, REPO)}")


# -------------------------------------------------------------- utils ----
def _input_shas(panel):
    return {os.path.basename(p): file_sha(p)
            for p in sorted(glob.glob(os.path.join(RESULTS, panel, "*.jsonl")))
            if os.path.basename(p) in ("removal.jsonl", "geometry.jsonl")}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _f(x):
    return "   -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.3f}"


# ----------------------------------------------------------- selftest ----
def selftest():
    fails = []
    mk = lambda R: {"s0.0": 0.5, **{f"s{1 - p}": 0.5 * r for p, r in R}}
    # clean drop between 1% and 0.5%
    c = mk([(0.5, 1), (0.1, 1), (0.05, 1), (0.03, 0.98), (0.01, 0.95),
            (0.005, 0.5), (0.001, 0.1)])
    pt = p_true(c)
    if pt["status"] != "ok" or not (0.005 < pt["p_true"] < 0.01):
        fails.append(f"p_true clean case {pt}")
    # first-failure: recovery below a failure ignored
    c = mk([(0.5, 1), (0.1, 0.5), (0.05, 1), (0.03, 1), (0.01, 1),
            (0.005, 1), (0.001, 1)])
    pt = p_true(c, mode="first_failure")
    if not (0.1 <= pt["p_true"] <= 0.5):
        fails.append(f"p_true first-failure case {pt}")
    pt = p_true(c)                        # last-success ignores the dip
    if pt["status"] != "censored":
        fails.append(f"p_true last-success should skip the dip: {pt}")
    # the published gemma shape: dip at 97%, recovery at 99%, cliff at 99.5%
    g = {"s0.0": .674, "s0.5": .704, "s0.9": .760, "s0.95": .705, "s0.97": .559,
         "s0.99": .693, "s0.995": .549, "s0.999": .119}
    pt = p_true(g)
    if not (0.005 < pt["p_true"] < 0.01):
        fails.append(f"published-gemma shape: p_true {pt}, want in (0.5%, 1%)")
    # censored
    pt = p_true(mk([(p, 1.0) for p in (0.5, 0.1, 0.05, 0.03, 0.01, 0.005, 0.001)]))
    if pt["status"] != "censored" or abs(pt["p_true"] - 0.001) > 1e-9:
        fails.append(f"p_true censored case {pt}")
    # ineligible
    c = {"s0.0": 0.01, "s0.99": 0.01}
    if p_true(c)["status"] != "ineligible":
        fails.append("dense below floor must be ineligible")
    # rules recover a planted mass threshold
    rng = np.random.default_rng(0)
    fit_cells = []
    for k in range(6):
        arr = [rng.standard_t(2 + k, size=20000).astype(np.float32)]
        g = G.geometry(arr)
        fit_cells.append(dict(geom=g, p_true=G.p_for_mass(g, 0.3)))
    params = fit_rules(fit_cells)
    if abs(params["mass"]["tau"] - 0.3) > 0.01:
        fails.append(f"mass rule did not recover tau: {params['mass']}")
    for c in fit_cells:
        ph = predict_p("mass", params, c["geom"])
        if abs(math.log2(ph / c["p_true"])) > 0.1:
            fails.append("mass rule does not invert on its own fit cells")
            break
    if predict_p("fixed", params, fit_cells[0]["geom"]) != 0.01:
        fails.append("fixed rule is not 1%")
    for m in fails:
        print(f"  FAIL {m}")
    print(f"v12_pstar selftest: {'PASS' if not fails else str(len(fails)) + ' FAILED'}")
    return not fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fit", "predict", "validate", "selftest"))
    ap.add_argument("--rule", default=PRIMARY, choices=RULES)
    ap.add_argument("--rho", type=float, default=RHO)
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--margin", type=float, default=1.0,
                    help="multiply p* by this (>1 is conservative)")
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 1, 2])
    ap.add_argument("--i-know-this-is-not-held-out", dest="i_know",
                    action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return 0 if selftest() else 1
    {"fit": cmd_fit, "predict": cmd_predict, "validate": cmd_validate}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
