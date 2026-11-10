#!/usr/bin/env python3
"""R1-E1 — matched-collateral curve, C-ref vs C-a (zero-GPU, torch-free).

    python src/r1_e1.py              # sanity gate -> analysis -> results/r1/E1/
    python src/r1_e1.py --no-figure  # numbers and CSVs only
    python src/r1_e1.py selftest     # PAV / solver / reading logic on synthetic data

The design is R1_REBUTTAL_EXPERIMENTS.md, section E1 (rebuttal round R1). Every
rule below is that text; nothing here was tuned on the outcome. The frozen
grid, both grid extensions and the alpha* panel had all been seen before the
plan was written, so E1 is "analysis rule fixed in advance", not confirmatory
(the plan says so itself).

QUESTION. At MATCHED collateral, does the magnitude-selected support (C-ref)
still remove more bias than a random support with the true signs (C-a)?

DATA. REG10 (configs/v12/astar_dec.yaml) x seeds {0,1,2} x {C-ref, C-a} x
alpha in {2,4,8,16 | 32,64,128 | 256,512}, joined from the alpha traces of
v8dec, v10ext and v11ext. Per point: removal R (`bias_reduction`), perplexity
ratio P (`ppl_ratio`), MMLU drop D in integer items of 200 (`dmmlu`), all on
evaluation items. alpha_trace.jsonl is append-only: rows are deduped on
(target, axis, seed, designer, condition, alpha) keeping the LAST occurrence.

SANITY GATE (step 5, run FIRST; the script stops if it fails). The loader must
reproduce, to the printed precision, the published in-budget-argmax numbers
under the integer-item gate (src/v9_gate.py; b = 4 items and P <= 1.10):
    frozen grid   Delta_selection  +0.284 [+0.169, +0.407]
    alpha <= 512  Delta_selection  +0.104 [+0.036, +0.183]
(and the depth points +0.233 at <=32, +0.146 at <=64). It is computed by the
published code path (src/v10_regen_alphaext.score) on this loader's rows.

ESTIMANDS.
  step 1  per (cell, seed, arm): P(alpha) -> isotonic non-decreasing fit
          (pool-adjacent-violators, implemented here); R(alpha) -> piecewise
          linear, no monotonicity; solve P(alpha) = c, read R(c).
  step 2  primary axis, ppl: c in {1.02, 1.04, 1.06, 1.08, 1.10} (+ 1.05, which
          the readings use); only c both arms reach; unreached counted.
  step 3  secondary axis, MMLU items: b in {2,4,6,8}, max{R(alpha): D(alpha) <= b}.
  step 4  Delta_matched(c) = R_ref(c) - R_a(c); seed mean, then the cell
          bootstrap (v10_common.boot_paired: 10,000 draws, seed 0, cells in
          panel insertion order).
  step 6  homogeneity: per arm, log(P-1) = log h + p log(alpha) on the points
          with P-1 >= 0.005; h_ref/h_a against the squared alpha* ratio; is
          R/sqrt(P-1) flat in alpha.
  step 7  update norm ||alpha E||_F at matched c, from the per-arm unit-edit
          `frobenius` that run_dec.py wrote to results/v8dec/removal.jsonl.
  step 8  figure, drawn only from the CSVs this script writes.
  alpha* points (results/v12astar) are an OVERLAY: their operating point was
  selected on validation items, so they never enter a fit or an estimate.

PRE-REGISTERED READINGS (c = 1.10 primary, 1.05 secondary).
  E1-A   CI of Delta_matched(1.10) excludes 0 and the point is >= +0.071
  E1-B   the CI covers 0 at BOTH c = 1.10 and c = 1.05
  E1-C   any other positive result ("small residual", never a headline)
  E1-B'  0 excluded in the NEGATIVE direction

IMPLEMENTATION CHOICES the registered text does not fix (also written to the
JSON, each with its sensitivity):
  I1  the isotonic fit is least squares on P itself (not log P), unit weights;
  I2  between knots the fitted P and R are both interpolated linearly in
      log(alpha). With ONE abscissa for both, R(c) does not depend on which
      abscissa it is (the curve in the (P, R) plane is the polyline through
      the knots); only alpha_c, hence step 7, does;
  I3  alpha_c = inf{alpha in [2, 512] : Pfit(alpha) >= c}: the first crossing,
      the LEFT end of a segment that is flat exactly at c;
  I4  "reached" = Pfit(512) >= c. If Pfit(2) > c the crossing lies below the
      probed grid: not solved, counted separately ("below grid"), never
      extrapolated towards alpha = 0;
  I5  a seed enters a cell's mean at c only if BOTH arms are solved there;
  I6  step 3 is read literally as an MMLU-only budget (no ppl condition);
      a run with no admissible alpha scores 0 (the project nan->0 rule) and
      is counted. The joint budget (D <= b and P <= 1.10) is reported beside
      it as a supplement, because at b = 4 it is the sanity estimand.

EQUALITY FORM vs BUDGET FORM. Steps 1-2 read R AT the scale where P = c ("solve
P(alpha) = c, read R(c)"); step 3 and every published estimand read the BEST R
at collateral <= budget. Where R peaks before the budget boundary the two
differ. The registered ppl-axis estimand is the equality form and is what the
reading uses; the budget form on the same fitted curves is reported as the
`budget_form` variant, labelled a different estimand, not an implementation
choice.
"""
import bisect
import collections
import csv
import hashlib
import json
import math
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.dont_write_bytecode = True     # this script adds results/r1/E1/ and nothing else
import v10_common as V             # noqa: E402
import v9_gate                     # noqa: E402
import v10_regen_alphaext as X     # noqa: E402  (the published argmax scorer)

RES = os.path.join(REPO, "results")
OUT = os.path.join(RES, "r1", "E1")
F_JSON = os.path.join(OUT, "e1.json")
F_CELL = os.path.join(OUT, "e1_per_cell.csv")
F_PTS = os.path.join(OUT, "e1_points.csv")
F_CURVE = os.path.join(OUT, "e1_curves.csv")
F_FIG = os.path.join(OUT, "e1_matched_collateral")

ARMS = ("C-ref", "C-a")
PANELS = (("v8dec", (2.0, 4.0, 8.0, 16.0)),
          ("v10ext", (32.0, 64.0, 128.0)),
          ("v11ext", (256.0, 512.0)))
FROZEN = PANELS[0][1]
ALPHAS = tuple(a for _p, al in PANELS for a in al)
SEEDS = (0, 1, 2)
N_ITEMS = 200
# REG10, verbatim from configs/v12/astar_dec.yaml (checked against it at run time)
REG10 = [("gemma", "occ_gender", "gemma_3b"), ("llama", "occ_gender", "llama_3b"),
         ("qwen", "occ_gender", "qwen_3b"), ("phi", "occ_gender", "phi_3b"),
         ("phi", "bbq_Age", "phi_3b"), ("qwen", "bbq_Age", "qwen_3b"),
         ("qwen", "bbq_Race_ethnicity", "qwen_3b"), ("qwen", "ss_intra", "qwen_3b"),
         ("phi", "ss_intra", "phi_3b"), ("qwen7b", "occ_gender", "qwen")]

C_STEP2 = (1.02, 1.04, 1.06, 1.08, 1.10)          # the list in step 2
C_LEVELS = (1.02, 1.04, 1.05, 1.06, 1.08, 1.10)   # + 1.05, used by the readings
C_PRIMARY, C_SECONDARY = 1.10, 1.05
B_LEVELS = (2, 4, 6, 8)
A_THRESHOLD = 0.071                # "+0.284 / 4", as printed in the plan
HOMOG_MIN = 0.005                  # step 6 inclusion rule: P - 1 >= 0.005
HOMOG_WINDOW = 0.5                 # POST-HOC upper cut-off, labelled as such
ASTAR = {"C-ref": "s0.99", "C-a": "C-a@0.01"}

# what the manuscript prints (sec. 5.3 / App. B); the gate compares at 3 decimals
PUBLISHED = {
    "frozen": dict(point=0.284, lo=0.169, hi=0.407),
    "extension_le512": dict(point=0.104, lo=0.036, hi=0.183),
    "depth_le32": dict(point=0.233),
    "depth_le64": dict(point=0.146),
}
PUBLISHED_ASTAR = dict(point=0.073, lo=-0.036, hi=0.213)

# interpolation variants: primary first, then the sensitivities
#   iso   isotonic step on/off        ylog  fit and solve on log P
#   px/rx abscissa in which P / R are linear between knots (log, lin, sq = alpha^2)
#   cross first = inf{alpha: P >= c}, last = sup{alpha: P <= c}
#   env   budget form: R(c) = max of the interpolated R over alpha <= alpha_c
#   y     the collateral series that is fitted and solved ("P", or "D" = MMLU items)
#   kind  "implementation" = a choice the registered text leaves open;
#         "estimand" = a different quantity, reported beside the registered one
VARIANTS = collections.OrderedDict([
    ("primary", dict(iso=True, ylog=False, px="log", rx="log", cross="first",
                     note="registered reading: isotonic P, both linear in log alpha")),
    ("raw_first", dict(iso=False, ylog=False, px="log", rx="log", cross="first",
                       note="no isotonic step, raw piecewise-linear P, first crossing")),
    ("raw_last", dict(iso=False, ylog=False, px="log", rx="log", cross="last",
                      note="no isotonic step, raw piecewise-linear P, last crossing")),
    ("lin_alpha", dict(iso=True, ylog=False, px="lin", rx="lin", cross="first",
                       note="isotonic P, both linear in alpha (R(c) identical by "
                            "construction; alpha_c differs)")),
    ("iso_logP", dict(iso=True, ylog=True, px="log", rx="log", cross="first",
                      note="isotonic fit and interpolation on log P")),
    ("P_lin__R_log", dict(iso=True, ylog=False, px="lin", rx="log", cross="first",
                          note="mixed abscissa: P linear in alpha, R linear in log alpha")),
    ("P_log__R_lin", dict(iso=True, ylog=False, px="log", rx="lin", cross="first",
                          note="mixed abscissa: P linear in log alpha, R linear in alpha")),
    ("P_sq__R_lin", dict(iso=True, ylog=False, px="sq", rx="lin", cross="first",
                         note="homogeneity-consistent: P linear in alpha^2, R linear "
                              "in alpha (exact if P-1 = h alpha^2 and R = g alpha "
                              "inside a segment)")),
    ("budget_form", dict(iso=True, ylog=False, px="log", rx="log", cross="first", env=True,
                         kind="estimand",
                         note="NOT the registered ppl-axis estimand: best R at fitted "
                              "P <= c, i.e. max of the interpolated R over alpha <= "
                              "alpha_c (the form step 3 and the published argmax use)")),
])
# MMLU items treated like the ppl axis (POST-HOC supplement to step 3)
VAR_D = dict(iso=True, ylog=False, px="log", rx="log", cross="first", y="D")
VAR_D_BUDGET = dict(VAR_D, env=True)

FWD = {"log": math.log, "lin": float, "sq": lambda a: a * a}
INV = {"log": math.exp, "lin": float, "sq": math.sqrt}


# ------------------------------------------------------------------ utils ----

def rel(fp):
    return os.path.relpath(fp, REPO).replace(os.sep, "/")


def sha256(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def clean(o):
    """JSON-safe: nan/inf -> None, tuples -> lists, numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def jdump(obj, fp):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(clean(obj), fh, indent=1, sort_keys=True)
        fh.write("\n")


def write_csv(fp, header, rows):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (v is None or (isinstance(v, float) and v != v))
                        else (repr(v) if isinstance(v, float) else v) for v in r])


def read_csv(fp):
    with open(fp, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def cname(cell):
    return f"{cell[0]}|{cell[1]}"


# ---------------------------------------------------------------- loading ----

def reg10_from_config():
    """The registered cell list, read from the config (None if unreadable)."""
    fp = os.path.join(REPO, "configs", "v12", "astar_dec.yaml")
    try:
        import yaml
        with open(fp, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return ([(c["target"], c["axis"], c["designer"]) for c in cfg["cells"]],
                [int(s) for s in cfg["seeds"]])
    except Exception:
        return None


def load_trace(panel, alphas):
    """Deduped C-ref / C-a rows of one alpha trace, restricted to REG10.

    Last occurrence wins (a resumed cell is re-traced); the dict keeps the
    position of the FIRST occurrence, so the cell order is the panel's
    insertion order, which the published bootstrap depends on."""
    fp = os.path.join(RES, panel, "alpha_trace.jsonl")
    reg = set(REG10)
    seen = collections.OrderedDict()
    n_arm = n_bad = n_other = 0
    with open(fp, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                n_bad += 1
                continue
            if r.get("condition") not in ARMS:
                continue
            if (r.get("target"), r.get("axis"), r.get("designer")) not in reg \
                    or r.get("seed") not in SEEDS:
                n_other += 1
                continue
            n_arm += 1
            a = float(r["alpha"])
            if a not in alphas:
                raise SystemExit(f"{panel}: alpha {a} is outside this panel's "
                                 f"registered points {alphas}")
            r = dict(r, _panel=panel)
            seen[(r["target"], r["axis"], r["seed"], r["designer"],
                  r["condition"], a)] = r
    return list(seen.values()), dict(file=rel(fp), sha256=sha256(fp), rows_arms=n_arm,
                                     rows_deduped=len(seen),
                                     duplicates_dropped=n_arm - len(seen),
                                     unparseable_lines=n_bad,
                                     rows_outside_reg10=n_other)


def items_dropped(r):
    """Signed MMLU drop in integer items (negative = the edit helped)."""
    n = r.get("n_items", N_ITEMS)
    if r.get("pre_mmlu") is not None and r.get("post_mmlu") is not None:
        return v9_gate.items_from_acc(r["pre_mmlu"], n) - \
            v9_gate.items_from_acc(r["post_mmlu"], n)
    x = r["dmmlu"] * n
    k = round(x)
    if abs(x - k) > 1e-6:
        raise v9_gate.NonIntegralAccuracy(f"dmmlu {r['dmmlu']!r} is off the 1/{n} grid")
    return int(k)


def build_runs(rows):
    """(target, axis, seed, arm) -> the nine-point curve; cells in panel order."""
    cells, runs = [], {}
    for r in rows:
        cell = (r["target"], r["axis"])
        if cell not in cells:
            cells.append(cell)
        runs.setdefault(cell + (r["seed"], r["condition"]), {})[float(r["alpha"])] = r
    want = {(t, a) for t, a, _d in REG10}
    if set(cells) != want:
        raise SystemExit(f"cell set differs from REG10: {sorted(set(cells) ^ want)}")
    out = {}
    for cell in cells:
        for s in SEEDS:
            for arm in ARMS:
                d = runs.get(cell + (s, arm), {})
                if tuple(sorted(d)) != ALPHAS:
                    raise SystemExit(f"incomplete curve {cell + (s, arm)}: "
                                     f"alphas {sorted(d)}")
                rs = [d[a] for a in ALPHAS]
                for r in rs:
                    for f in ("ppl_ratio", "bias_reduction"):
                        v = r.get(f)
                        if v is None or v != v:
                            raise SystemExit(f"missing {f} in {cell + (s, arm)} "
                                             f"alpha={r['alpha']}")
                gate = [v9_gate.collateral_ok_row(r) for r in rs]
                out[cell + (s, arm)] = dict(
                    alpha=list(ALPHAS), P=[float(r["ppl_ratio"]) for r in rs],
                    R=[float(r["bias_reduction"]) for r in rs],
                    D=[items_dropped(r) for r in rs],
                    ok=[bool(g[0]) for g in gate], gate_mode=[g[1] for g in gate],
                    panel=[r["_panel"] for r in rs])
    return cells, out


def load_frobenius():
    """Unit-edit Frobenius norm per (target, axis, seed, arm), from run_dec.py's
    removal.jsonl. Returns (table, provenance) -- table is None if any is missing."""
    fro, prov = {}, {}
    for tree, root in (("as_run", RES), ("rescored", V.V9)):
        fp = os.path.join(root, "v8dec", "removal.jsonl")
        if not os.path.exists(fp):
            continue
        t = {}
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if r.get("condition") in ARMS and r.get("panel") == "v8dec":
                    t[(r["target"], r["axis"], r["seed"], r["condition"])] = \
                        r.get("frobenius")         # last wins
        fro[tree] = t
        prov[tree] = dict(file=rel(fp), sha256=sha256(fp), rows=len(t))
    if "as_run" not in fro:
        return None, dict(prov, status="results/v8dec/removal.jsonl not found")
    t = fro["as_run"]
    need = [(c[0], c[1], s, arm) for c in REG10 for s in SEEDS for arm in ARMS]
    miss = [k for k in need if t.get(k) is None]
    prov["missing"] = len(miss)
    if "rescored" in fro:
        prov["max_abs_diff_as_run_vs_rescored"] = max(
            abs(t[k] - fro["rescored"][k]) for k in need
            if t.get(k) is not None and fro["rescored"].get(k) is not None)
    if miss:
        return None, dict(prov, status=f"frobenius missing for {len(miss)} runs")
    return {k: float(t[k]) for k in need}, dict(prov, status="ok")


def load_astar():
    """alpha* rows (overlay only). Last occurrence wins, as src/v12_astar.load."""
    fp = os.path.join(RES, "v12astar", "removal.jsonl")
    if not os.path.exists(fp):
        return None, dict(status="results/v12astar/removal.jsonl not found")
    out = {}
    with open(fp, encoding="utf-8") as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            out[(r["target"], r["axis"], r["seed"], r["variant"])] = r
    return out, dict(file=rel(fp), sha256=sha256(fp), rows=len(out), status="ok")


# ------------------------------------------------------------ sanity gate ----

def _close(x, printed):
    return abs(x - printed) < 0.0005 + 1e-12


def sanity(rows):
    """Step 5. The published scorer, on this loader's rows."""
    grid = set(FROZEN)
    blocks = {
        "frozen": X.score(rows, "frozen grid {2,4,8,16}", grid),
        "depth_le32": X.score(rows, "alpha <= 32", grid | {32.0}),
        "depth_le64": X.score(rows, "alpha <= 64", grid | {32.0, 64.0}),
        "extension_le512": X.score(rows, "alpha <= 512 (all probed)", None),
    }
    out, ok = {}, True
    for name, pub in PUBLISHED.items():
        d = blocks[name]["delta_selection"]
        match = {k: _close(d[k], v) for k, v in pub.items()}
        ok = ok and all(match.values())
        out[name] = dict(published=pub, reproduced={k: d[k] for k in ("point", "lo", "hi", "n")},
                         text=d["text"], match=match, n_cells=blocks[name]["n_cells"],
                         unanimity=blocks[name]["unanimity"],
                         condition_means=blocks[name]["condition_means"],
                         gate_modes=blocks[name]["gate_modes"],
                         per_cell=blocks[name]["per_cell"])
    art = V.jload("results/v10/alpha_extension.json")
    if art:
        diffs = []
        for mine, theirs in (("frozen", art["frozen"]), ("extension_le512", art["extended"])):
            a, b = blocks[mine]["delta_selection"], theirs["delta_selection"]
            diffs += [abs(a[k] - b[k]) for k in ("point", "lo", "hi")]
        out["vs_results_v10_alpha_extension_json"] = dict(
            max_abs_diff=max(diffs),
            note="v10 scored alpha <= 128; 256 and 512 admit no run of either arm, "
                 "so the <= 512 estimand must equal it exactly")
    out["pass"] = bool(ok)
    return out, blocks


# ------------------------------------------------- isotonic fit and solve ----

def pava(y):
    """Least-squares non-decreasing fit, unit weights (pool adjacent violators)."""
    blocks = []                                   # [sum, count]
    for v in y:
        blocks.append([float(v), 1])
        while len(blocks) > 1 and \
                blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, n = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += n
    out = []
    for s, n in blocks:
        out += [s / n] * n
    return out


def fitted(run, var):
    y = run[var.get("y", "P")]
    y = [math.log(p) for p in y] if var["ylog"] else [float(p) for p in y]
    return pava(y) if var["iso"] else y


def crossing(u, y, c, cross="first"):
    """Solve y(u) = c on the piecewise-linear interpolant of the knots (u, y).

    first: inf{u : y(u) >= c}   last: sup{u : y(u) <= c}
    -> (u_c, status, j) with status in {ok, unreached, below_grid}; j is the
    index of the knot at the left of the bracketing segment."""
    K = len(u)
    if cross == "first":
        if y[0] > c:
            return None, "below_grid", None
        if y[0] == c:
            return u[0], "ok", 0
        for j in range(1, K):
            if y[j] >= c:                         # y[j-1] < c <= y[j]
                t = (c - y[j - 1]) / (y[j] - y[j - 1])
                return u[j - 1] + t * (u[j] - u[j - 1]), "ok", j - 1
        return None, "unreached", None
    le = [j for j in range(K) if y[j] <= c]
    if not le:
        return None, "below_grid", None
    j = le[-1]
    if j == K - 1:
        return None, "unreached", None            # never ends above c
    t = (c - y[j]) / (y[j + 1] - y[j])            # y[j] <= c < y[j+1]
    return u[j] + t * (u[j + 1] - u[j]), "ok", j


def interp(xs, ys, x):
    j = min(max(bisect.bisect_right(xs, x) - 1, 0), len(xs) - 2)
    t = (x - xs[j]) / (xs[j + 1] - xs[j])
    return ys[j] + t * (ys[j + 1] - ys[j])


def solve(run, c, var, fit=None):
    """-> dict(status, alpha, R, bracket) for one run at collateral level c."""
    if fit is None:
        fit = fitted(run, var)
    cc = math.log(c) if var["ylog"] else c
    u = [FWD[var["px"]](a) for a in run["alpha"]]
    uc, status, j = crossing(u, fit, cc, var["cross"])
    if status != "ok":
        return dict(status=status, alpha=None, R=None, bracket=None)
    a_c = min(max(INV[var["px"]](uc), run["alpha"][0]), run["alpha"][-1])
    v = [FWD[var["rx"]](a) for a in run["alpha"]]
    R = interp(v, run["R"], FWD[var["rx"]](a_c))
    if var.get("env"):          # budget form: a piecewise-linear R peaks at a knot or at alpha_c
        R = max([R] + [r for a, r in zip(run["alpha"], run["R"]) if a <= a_c])
    return dict(status="ok", alpha=a_c, R=R,
                bracket=[run["alpha"][j], run["alpha"][min(j + 1, len(u) - 1)]])


def matched(runs, cells, c, var, fits=None):
    """Delta_matched(c): seed mean per arm over the seeds BOTH arms solve, then
    the paired cell bootstrap."""
    pairs, per_cell = [], collections.OrderedDict()
    counts = {arm: collections.Counter() for arm in ARMS}
    for cell in cells:
        acc = {arm: [] for arm in ARMS}
        al = {arm: [] for arm in ARMS}
        for s in SEEDS:
            res = {arm: solve(runs[cell + (s, arm)], c, var,
                              None if fits is None else fits[cell + (s, arm)])
                   for arm in ARMS}
            for arm in ARMS:
                counts[arm][res[arm]["status"]] += 1
            if all(res[arm]["status"] == "ok" for arm in ARMS):
                for arm in ARMS:
                    acc[arm].append(res[arm]["R"])
                    al[arm].append(res[arm]["alpha"])
        n = len(acc["C-ref"])
        row = dict(n_seeds=n)
        if n:
            m = {arm: float(np.mean(acc[arm])) for arm in ARMS}
            pairs.append((m["C-ref"], m["C-a"]))
            row.update({"C-ref": m["C-ref"], "C-a": m["C-a"],
                        "delta": m["C-ref"] - m["C-a"],
                        "alpha_c": {arm: al[arm] for arm in ARMS}})
        per_cell[cname(cell)] = row
    d = V.boot_paired(pairs)
    return dict(
        c=c, delta=(dict(d, text=V.fmt(d), status=V.status(d)) if d else None),
        n_cells=len(pairs), unanimity=sum(1 for p in pairs if p[0] > p[1]),
        arm_means={arm: (float(np.mean([p[i] for p in pairs])) if pairs else None)
                   for i, arm in enumerate(ARMS)},
        unreached_runs={arm: counts[arm]["unreached"] for arm in ARMS},
        below_grid_runs={arm: counts[arm]["below_grid"] for arm in ARMS},
        solved_runs={arm: counts[arm]["ok"] for arm in ARMS},
        cells_dropped=[k for k, v in per_cell.items() if not v["n_seeds"]],
        per_cell=per_cell)


def argmax_budget(runs, cells, b=None, c=None):
    """max{R(alpha): D(alpha) <= b [and P(alpha) <= c]} over the nine probed
    alphas, per run; no admissible alpha -> nan -> 0.0 (counted), as in the
    published in-budget argmax. Seed mean, paired cell bootstrap."""
    pairs, per_cell = [], collections.OrderedDict()
    n_none = {arm: 0 for arm in ARMS}
    pick = {arm: [] for arm in ARMS}
    for cell in cells:
        m = {}
        for arm in ARMS:
            vals = []
            for s in SEEDS:
                run = runs[cell + (s, arm)]
                adm = [(R, a) for a, P, R, D in zip(run["alpha"], run["P"], run["R"], run["D"])
                       if (b is None or D <= b)
                       and (c is None or P <= c + v9_gate.PPL_TOL)]
                if adm:
                    best = max(adm)
                    vals.append(best[0])
                    pick[arm].append(best[1])
                else:
                    n_none[arm] += 1
                    vals.append(0.0)
            m[arm] = float(np.mean(vals))
        pairs.append((m["C-ref"], m["C-a"]))
        per_cell[cname(cell)] = {"C-ref": m["C-ref"], "C-a": m["C-a"],
                                 "delta": m["C-ref"] - m["C-a"]}
    d = V.boot_paired(pairs)
    return dict(b_items=b, c_ppl=c, delta=dict(d, text=V.fmt(d), status=V.status(d)),
                n_cells=len(pairs), unanimity=sum(1 for p in pairs if p[0] > p[1]),
                arm_means={arm: float(np.mean([p[i] for p in pairs]))
                           for i, arm in enumerate(ARMS)},
                runs_with_no_admissible_alpha_scored_0=n_none,
                median_selected_alpha={arm: (float(np.median(pick[arm])) if pick[arm] else None)
                                       for arm in ARMS},
                per_cell=per_cell)


def reading(d110, d105):
    """The registered branch, literally. Returns (label, note)."""
    cov = lambda d: d["lo"] <= 0 <= d["hi"]
    if d110["lo"] > 0 and d110["point"] >= A_THRESHOLD:
        return "E1-A", "CI at c=1.10 excludes 0 and the point is >= +0.071"
    if d110["hi"] < 0:
        return "E1-B'", "0 excluded in the negative direction at c=1.10"
    if d105["hi"] < 0:
        return "E1-B'", ("0 excluded in the negative direction at c=1.05 only; the "
                         "registered text does not say at which c B' is read")
    if cov(d110) and cov(d105):
        return "E1-B", "CI covers 0 at both c=1.10 and c=1.05"
    if d110["lo"] > 0:
        return "E1-C", "CI at c=1.10 excludes 0 but the point is below +0.071"
    return "E1-C", "CI covers 0 at c=1.10 and excludes 0 (positive) at c=1.05"


# -------------------------------------------------------------- bootstrap ----

def boot_index(n_cells, n=V.BOOT_N, seed=V.BOOT_SEED):
    """The index draws of v10_common.boot_paired, replayed as a matrix."""
    rng = random.Random(seed)
    return np.array([[rng.randrange(n_cells) for _ in range(n_cells)]
                     for _ in range(n)], dtype=np.int64)


def pct(vals):
    """Percentile interval with boot_paired's index convention (nan dropped)."""
    v = np.sort(np.asarray(vals, dtype=float))
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return None, None
    return float(v[int(.025 * n)]), float(v[int(.975 * n) - 1])


def boot_mean_fast(x, idx):
    x = np.asarray(x, dtype=float)
    lo, hi = pct(x[idx].mean(axis=1))
    return dict(point=float(x.mean()), lo=lo, hi=hi, n=int(len(x)))


# ------------------------------------------------------------ homogeneity ----

def _suff(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return np.array([len(x), x.sum(), y.sum(), (x * x).sum(), (x * y).sum()])


def _ols(S):
    n, sx, sy, sxx, sxy = (S[..., i] for i in range(5))
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        icpt = (sy - slope * sx) / n
    return slope, icpt


def homog_points(runs, cells, arm, hi=None):
    """cell -> [(seed, alpha, P, R)] with P - 1 >= HOMOG_MIN (and <= hi)."""
    out = collections.OrderedDict()
    for cell in cells:
        L = []
        for s in SEEDS:
            run = runs[cell + (s, arm)]
            for a, P, R in zip(run["alpha"], run["P"], run["R"]):
                if P - 1.0 >= HOMOG_MIN and (hi is None or P - 1.0 <= hi):
                    L.append((s, a, P, R))
        out[cell] = L
    return out


def homogeneity(runs, cells, idx, hi=None):
    """log(P-1) = log h + p log(alpha), one pooled OLS per arm (the literal
    reading of 'per arm'), natural logs, alpha in the units of the grid.
    Cell-bootstrap CIs resample cells jointly for the two arms."""
    out, S, pts = {}, {}, {}
    for arm in ARMS:
        pts[arm] = homog_points(runs, cells, arm, hi)
        S[arm] = np.array([_suff([math.log(a) for _s, a, _P, _R in L],
                                 [math.log(P - 1.0) for _s, _a, P, _R in L])
                           for L in pts[arm].values()])
        p, lnh = _ols(S[arm].sum(axis=0))
        pb, lb = _ols(S[arm][idx].sum(axis=1))
        x = np.array([math.log(a) for L in pts[arm].values() for _s, a, _P, _R in L])
        y = np.array([math.log(P - 1.0) for L in pts[arm].values() for _s, _a, P, _R in L])
        res = y - (lnh + p * x)
        r2 = 1.0 - float((res ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
        out[arm] = dict(p=float(p), p_ci=pct(pb), log_h=float(lnh), h=float(math.exp(lnh)),
                        log_h_ci=pct(lb), n_points=int(len(x)), r2=r2,
                        n_points_by_alpha={str(int(a)): sum(1 for L in pts[arm].values()
                                                           for q in L if q[1] == a)
                                           for a in ALPHAS},
                        cells_without_points=[cname(c) for c, L in pts[arm].items() if not L])
        out[arm]["_boot"] = (pb, lb)

    # h_ref / h_a from the two free-p fits (unit-dependent when p differs)
    dl = out["C-ref"]["log_h"] - out["C-a"]["log_h"]
    dlb = out["C-ref"]["_boot"][1] - out["C-a"]["_boot"][1]
    lo, hi_ = pct(dlb)
    out["h_ratio_free_p"] = dict(ratio=math.exp(dl), ci=[math.exp(lo), math.exp(hi_)],
                                 log_ratio=dl)

    # the scale ratio the two fits imply at ppl budget c = 1.10:
    #   alpha_arm(c) = ((c - 1) / h)^(1/p)
    def a_at(lnh, p, c=C_PRIMARY):
        return np.exp((math.log(c - 1.0) - lnh) / p)
    r = a_at(out["C-a"]["log_h"], out["C-a"]["p"]) / a_at(out["C-ref"]["log_h"], out["C-ref"]["p"])
    rb = a_at(out["C-a"]["_boot"][1], out["C-a"]["_boot"][0]) / \
        a_at(out["C-ref"]["_boot"][1], out["C-ref"]["_boot"][0])
    out["implied_alpha_ratio_a_over_ref_at_c1.10"] = dict(ratio=float(r), ci=pct(rb))

    # p fixed at 2 (the registered expectation): log h = mean of log(P-1) - 2 log alpha,
    # cell = mean over its points, paired over the cells that have points in both arms
    cm = {arm: {c: (float(np.mean([math.log(P - 1.0) - 2.0 * math.log(a)
                                   for _s, a, P, _R in L])) if L else None)
                for c, L in pts[arm].items()} for arm in ARMS}
    both = [c for c in cells if cm["C-ref"][c] is not None and cm["C-a"][c] is not None]
    if both:
        d = np.array([cm["C-ref"][c] - cm["C-a"][c] for c in both])
        ii = boot_index(len(both)) if len(both) != len(cells) else idx
        lo, hi_ = pct(d[ii].mean(axis=1))
        out["h_ratio_p_fixed_2"] = dict(
            ratio=float(math.exp(d.mean())), ci=[math.exp(lo), math.exp(hi_)],
            n_cells=len(both), per_cell={cname(c): float(math.exp(v)) for c, v in zip(both, d)},
            median_cell_ratio=float(math.exp(np.median(d))))

    # alternative reading of 'per arm': one OLS per (cell, seed, arm)
    for arm in ARMS:
        ps, cell_p = [], []
        for cell, L in pts[arm].items():
            cp = []
            for s in SEEDS:
                q = [(a, P) for s_, a, P, _R in L if s_ == s]
                if len(q) >= 2:
                    p_run, _ = _ols(_suff([math.log(a) for a, _P in q],
                                          [math.log(P - 1.0) for _a, P in q]))
                    if p_run == p_run:
                        ps.append(float(p_run))
                        cp.append(float(p_run))
            if cp:
                cell_p.append(float(np.mean(cp)))
        out[arm]["per_run_fits"] = dict(
            n_runs=len(ps), median_p=float(np.median(ps)) if ps else None,
            q25_p=float(np.percentile(ps, 25)) if ps else None,
            q75_p=float(np.percentile(ps, 75)) if ps else None,
            min_p=min(ps) if ps else None, max_p=max(ps) if ps else None,
            mean_of_cell_means=float(np.mean(cell_p)) if cell_p else None)

    # flatness of R / sqrt(P - 1): slope of its log on log alpha (0 = flat)
    for arm in ARMS:
        Sf = np.array([_suff([math.log(a) for _s, a, _P, R in L if R > 0],
                             [math.log(R / math.sqrt(P - 1.0)) for _s, _a, P, R in L if R > 0])
                       for L in pts[arm].values()])
        sl, _ = _ols(Sf.sum(axis=0))
        slb, _ = _ols(Sf[idx].sum(axis=1))
        by_alpha = collections.OrderedDict()
        for a in ALPHAS:
            v = [R / math.sqrt(P - 1.0) for L in pts[arm].values()
                 for _s, a_, P, R in L if a_ == a]
            by_alpha[str(int(a))] = dict(n=len(v), median=float(np.median(v)) if v else None,
                                         mean=float(np.mean(v)) if v else None)
        ci = pct(slb)
        out[arm]["R_over_sqrt_Pm1"] = dict(
            loglog_slope=float(sl), slope_ci=ci,
            flat=bool(ci[0] is not None and ci[0] <= 0 <= ci[1]),
            n_points_R_positive=int(Sf[:, 0].sum()),
            n_points_R_nonpositive_excluded=int(sum(len(L) for L in pts[arm].values())
                                                - Sf[:, 0].sum()),
            by_alpha=by_alpha)
    for arm in ARMS:
        del out[arm]["_boot"]
    return out


# ----------------------------------------------------------------- figure ----

LABEL_T = {"gemma": "Gemma-2 2.6B", "llama": "Llama-3.2 3B", "qwen": "Qwen2.5 3B",
           "phi": "Phi-3.5 3.8B", "qwen7b": "Qwen2.5 7B"}
LABEL_A = {"occ_gender": "occupation / gender", "bbq_Age": "BBQ age",
           "bbq_Race_ethnicity": "BBQ race / ethnicity", "ss_intra": "StereoSet"}
XLIM = (0.955, 1.255)


def figure():
    """Step 8, drawn ONLY from e1_points.csv and e1_curves.csv.

    Ten cell panels, one pooled panel (both arms) and the pooled paired
    difference with its cell-bootstrap interval. Arms differ by colour AND by
    line style and marker shape, so the figure survives greyscale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    pts, cur = read_csv(F_PTS), read_csv(F_CURVE)
    ink, ink2, grid, g = "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"
    col = {"C-ref": "#2a78d6", "C-a": "#eb6834"}        # validated pair (dataviz)
    ls = {"C-ref": "-", "C-a": (0, (4, 1.6))}
    mk = {"C-ref": "o", "C-a": "^"}
    mk_star = {"C-ref": "*", "C-a": "X"}
    ms_star = {"C-ref": 6.2, "C-a": 4.6}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
        "axes.edgecolor": ink2, "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": ink2, "ytick.color": ink2, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white"})

    cells = []
    for r in pts:
        k = (r["target"], r["axis"])
        if r["kind"] == "grid" and k not in cells:
            cells.append(k)
    fig = plt.figure(figsize=(5.5, 7.4))
    gs = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 0.5], left=0.105, right=0.985,
                          top=0.965, bottom=0.004, hspace=0.56, wspace=0.34)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(12)]

    def base(ax, title):
        ax.set_xlim(*XLIM)
        ax.set_xticks([1.0, 1.1, 1.2])
        ax.grid(True, color=grid, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.axvline(C_PRIMARY, color=ink2, linewidth=0.6, zorder=1)
        ax.axhline(0.0, color=ink2, linewidth=0.5, zorder=1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=6.3, color=ink, pad=2.5, linespacing=1.15)
        ax.tick_params(labelsize=6, pad=1.5)

    def ylim(ax, ys):
        lo, hi = min(ys + [0.0]), max(ys + [0.05])
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)

    def star(ax, x, y, arm, scale=1.0):
        ax.plot([x], [y], linestyle="none", marker=mk_star[arm], markersize=ms_star[arm] * scale,
                markerfacecolor=col[arm], markeredgecolor=ink, markeredgewidth=0.45, zorder=5)

    for ax, cell in zip(axes, cells):
        base(ax, f"{LABEL_T[cell[0]]}\n{LABEL_A[cell[1]]}")
        ys = []
        for arm in ARMS:
            c = [(float(r["c"]), float(r["R"])) for r in cur
                 if r["scope"] == "cell" and (r["target"], r["axis"]) == cell
                 and r["arm"] == arm and r["R"] != ""]
            ax.plot([p[0] for p in c], [p[1] for p in c], color=col[arm], linestyle=ls[arm],
                    linewidth=1.2, zorder=3)
            ys += [p[1] for p in c]
            for r in pts:
                if (r["target"], r["axis"]) != cell or r["arm"] != arm:
                    continue
                x, y = float(r["ppl_ratio"]), float(r["removal"])
                if not (XLIM[0] <= x <= XLIM[1]):
                    continue
                ys.append(y)
                if r["kind"] == "grid":
                    fz = r["frozen_grid"] == "1"
                    ax.plot([x], [y], linestyle="none", marker=mk[arm], markersize=3.3,
                            markerfacecolor=col[arm] if fz else "white",
                            markeredgecolor="white" if fz else col[arm],
                            markeredgewidth=0.4 if fz else 0.7, zorder=4)
                else:
                    star(ax, x, y, arm)
        ylim(ax, ys)

    ax = axes[len(cells)]
    base(ax, "All ten cells\nmean of each arm")
    ys = []
    for arm in ARMS:
        c = [(float(r["c"]), float(r["R"])) for r in cur
             if r["scope"] == "pooled" and r["arm"] == arm and r["R"] != ""]
        ax.plot([p[0] for p in c], [p[1] for p in c], color=col[arm], linestyle=ls[arm],
                linewidth=1.4, zorder=3)
        ys += [p[1] for p in c]
        for r in cur:
            if r["scope"] == "pooled_frozen" and r["arm"] == arm:
                ax.plot([float(r["c"])], [float(r["R"])], linestyle="none", marker=mk[arm],
                        markersize=3.8, markerfacecolor=col[arm], markeredgecolor="white",
                        markeredgewidth=0.4, zorder=4)
                ys.append(float(r["R"]))
            if r["scope"] == "pooled_astar" and r["arm"] == arm:
                star(ax, float(r["c"]), float(r["R"]), arm, 1.12)
                ys.append(float(r["R"]))
    ylim(ax, ys)

    ax = axes[len(cells) + 1]
    base(ax, "All ten cells\nlargest minus random")
    c = [(float(r["c"]), float(r["R"]), float(r["lo"]), float(r["hi"])) for r in cur
         if r["scope"] == "pooled_delta" and r["R"] != ""]
    ax.fill_between([p[0] for p in c], [p[2] for p in c], [p[3] for p in c],
                    color=g, alpha=0.28, linewidth=0, zorder=2)
    ax.plot([p[0] for p in c], [p[1] for p in c], color=ink, linewidth=1.3, zorder=3)
    ylim(ax, [p[2] for p in c] + [p[3] for p in c])
    for a_ in axes[len(cells):]:
        for sp in ("left", "bottom"):
            a_.spines[sp].set_linewidth(0.9)
            a_.spines[sp].set_edgecolor(ink)

    handles = [
        Line2D([], [], color=col["C-ref"], linestyle=ls["C-ref"], linewidth=1.2, marker=mk["C-ref"],
               markersize=3.6, markeredgecolor="white", markeredgewidth=0.4,
               label="Largest-magnitude coordinates"),
        Line2D([], [], color=col["C-a"], linestyle=ls["C-a"], linewidth=1.2, marker=mk["C-a"],
               markersize=3.6, markeredgecolor="white", markeredgewidth=0.4,
               label="Random coordinates"),
        Line2D([], [], linestyle="none", marker="o", markersize=3.6, markerfacecolor=g,
               markeredgecolor="white", markeredgewidth=0.4,
               label="Filled: registered edit scales, 2 to 16"),
        Line2D([], [], linestyle="none", marker="o", markersize=3.6, markerfacecolor="white",
               markeredgecolor=g, markeredgewidth=0.7,
               label="Open: larger edit scales, 32 and up"),
        Line2D([], [], linestyle="none", marker=mk_star["C-ref"], markersize=ms_star["C-ref"],
               markerfacecolor=col["C-ref"], markeredgecolor=ink, markeredgewidth=0.45,
               label="Adaptive scale, largest-magnitude"),
        Line2D([], [], linestyle="none", marker=mk_star["C-a"], markersize=ms_star["C-a"],
               markerfacecolor=col["C-a"], markeredgecolor=ink, markeredgewidth=0.45,
               label="Adaptive scale, random"),
        Line2D([], [], color=ink2, linewidth=0.6, marker="|", markersize=7, markeredgewidth=0.6,
               linestyle="none", label="Perplexity budget, 1.10"),
        Patch(facecolor=g, alpha=0.28, linewidth=0, label="95% interval over cells"),
    ]
    lg = fig.add_subplot(gs[4, :])
    lg.axis("off")
    lg.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.47, 0.0), ncol=2,
              frameon=False, fontsize=6.2, handlelength=2.6, labelspacing=0.65,
              columnspacing=2.4, borderaxespad=0)
    last = axes[10].get_position()
    fig.text(0.545, last.y0 - 0.034, "Perplexity after the edit, relative to the unedited model",
             ha="center", va="top", fontsize=7.5, color=ink)
    fig.text(0.012, 0.5 * (axes[0].get_position().y1 + last.y0), "Bias removed",
             rotation=90, ha="left", va="center", fontsize=7.5, color=ink)
    fig.savefig(F_FIG + ".pdf", metadata={"CreationDate": None, "ModDate": None,
                                          "Creator": None, "Producer": None})
    fig.savefig(F_FIG + ".png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    return [rel(F_FIG + ".pdf"), rel(F_FIG + ".png")]


# ------------------------------------------------------------------- main ----

def main(no_figure=False):
    os.makedirs(OUT, exist_ok=True)
    rep = collections.OrderedDict(
        artifact="r1_e1", module="src/r1_e1.py",
        design="R1_REBUTTAL_EXPERIMENTS.md, section E1",
        status=("analysis rules fixed in the plan; the data had been seen before, "
                "so E1 is not confirmatory (plan, 'honesty note')"))

    cfg = reg10_from_config()
    if cfg is not None and (set(cfg[0]) != set(REG10) or tuple(cfg[1]) != SEEDS):
        raise SystemExit("REG10 / seeds in this script differ from configs/v12/astar_dec.yaml")
    rows, inputs = [], collections.OrderedDict()
    for panel, alphas in PANELS:
        r, info = load_trace(panel, alphas)
        rows += r
        inputs[panel] = info
    rep["inputs"] = dict(traces=inputs, reg10_checked_against_config=cfg is not None)

    # ---- step 5 FIRST: nothing else is computed unless this reproduces ------
    san, blocks = sanity(rows)
    rep["sanity"] = san
    print("[r1-E1] sanity gate (published in-budget argmax, v9 integer gate)")
    for name in PUBLISHED:
        s = san[name]
        print(f"   {name:16s} {s['text']}   published "
              + " ".join(f"{k}={v:+.3f}" for k, v in s["published"].items())
              + ("   OK" if all(s["match"].values()) else "   MISMATCH"))
    if "vs_results_v10_alpha_extension_json" in san:
        print(f"   vs results/v10/alpha_extension.json: max |diff| = "
              f"{san['vs_results_v10_alpha_extension_json']['max_abs_diff']:.3g}")
    if not san["pass"]:
        rep["verdict"] = "STOPPED: sanity gate failed; no reading was computed"
        jdump(rep, F_JSON)
        print(f"SANITY GATE FAILED -- stopped before any reading. wrote {rel(F_JSON)}")
        return 2

    cells, runs = build_runs(rows)
    rep["cells"] = [cname(c) for c in cells]
    rep["cell_order"] = "v8dec panel insertion order (the published bootstrap order)"
    rep["seeds"], rep["alphas"] = list(SEEDS), list(ALPHAS)
    rep["bootstrap"] = dict(fn="v10_common.boot_paired", n=V.BOOT_N, seed=V.BOOT_SEED,
                            unit="target x axis cell", n_cells=len(cells))

    # my generalised gate must be the published one at (b = 4, c = 1.10) ...
    for k, run in runs.items():
        for a, P, D, ok in zip(run["alpha"], run["P"], run["D"], run["ok"]):
            if ok != (D <= 4 and P <= 1.10 + v9_gate.PPL_TOL):
                raise SystemExit(f"gate disagreement with v9_gate at {k} alpha={a}")
    # ... and the argmax built on it must be the published estimand, exactly
    joint4 = argmax_budget(runs, cells, b=4, c=1.10)
    ext = blocks["extension_le512"]["delta_selection"]
    dev = max(abs(joint4["delta"][k] - ext[k]) for k in ("point", "lo", "hi"))
    if dev > 1e-12:
        raise SystemExit(f"argmax_budget(b=4, c=1.10) differs from the published scorer by {dev}")
    rep["sanity"]["own_argmax_vs_published_scorer_max_abs_diff"] = dev
    rep["sanity"]["gate_modes"] = dict(collections.Counter(
        m for run in runs.values() for m in run["gate_mode"]))

    # ---- steps 1, 2, 4: the matched-collateral estimand ---------------------
    fits = {name: {k: fitted(run, var) for k, run in runs.items()}
            for name, var in VARIANTS.items()}
    prim = VARIANTS["primary"]
    primary = collections.OrderedDict(
        (f"{c:.2f}", matched(runs, cells, c, prim, fits["primary"])) for c in C_LEVELS)
    for c in C_LEVELS:
        primary[f"{c:.2f}"]["in_step2_list"] = c in C_STEP2
    rep["matched_ppl"] = primary
    lab, why = reading(primary["1.10"]["delta"], primary["1.05"]["delta"])
    rep["reading"] = dict(
        label=lab, why=why, primary_c=C_PRIMARY, secondary_c=C_SECONDARY,
        threshold_A=A_THRESHOLD,
        threshold_A_note="+0.071 as printed; the unrounded frozen estimate / 4 is "
                         f"{san['frozen']['reproduced']['point'] / 4:+.4f}",
        delta_1_10=primary["1.10"]["delta"], delta_1_05=primary["1.05"]["delta"])

    # flat-at-c knots (where I3 would matter) and pooling done by the isotonic step
    flat = sum(1 for k in runs for c in C_LEVELS for v in fits["primary"][k] if v == c)
    pooled = sum(1 for k, run in runs.items()
                 if any(abs(f - p) > 0 for f, p in zip(fits["primary"][k], run["P"])))
    moved = collections.OrderedDict()
    for c in C_LEVELS:
        n = 0
        for k, run in runs.items():
            s = solve(run, c, prim, fits["primary"][k])
            if s["status"] == "ok":
                j = run["alpha"].index(s["bracket"][0])
                if any(fits["primary"][k][i] != run["P"][i] for i in (j, j + 1)):
                    n += 1
        moved[f"{c:.2f}"] = n
    rep["isotonic_diagnostics"] = dict(
        runs_total=len(runs), runs_where_pava_pooled_any_point=pooled,
        runs_whose_bracketing_knots_were_moved_by_pava=moved,
        knots_exactly_equal_to_a_c_level=flat)

    # ---- sensitivities to the implementation choices ------------------------
    sens = collections.OrderedDict()
    for name, var in VARIANTS.items():
        blk = collections.OrderedDict()
        for c in C_LEVELS:
            m = matched(runs, cells, c, var, fits[name])
            blk[f"{c:.2f}"] = dict(delta=m["delta"], n_cells=m["n_cells"],
                                   unanimity=m["unanimity"], arm_means=m["arm_means"],
                                   unreached_runs=m["unreached_runs"],
                                   below_grid_runs=m["below_grid_runs"],
                                   per_cell_delta={k: v.get("delta")
                                                   for k, v in m["per_cell"].items()})
        l2, w2 = reading(blk["1.10"]["delta"], blk["1.05"]["delta"])
        sens[name] = dict(note=var["note"], kind=var.get("kind", "implementation"),
                          by_c=blk, reading=l2, reading_why=w2,
                          shift_vs_primary={c: blk[c]["delta"]["point"]
                                            - primary[c]["delta"]["point"] for c in blk})
    rep["sensitivity"] = sens
    impl = [k for k, v in sens.items() if v["kind"] == "implementation"]
    rep["reading"]["stable_across_implementation_choices"] = all(
        sens[k]["reading"] == lab for k in impl)
    rep["reading"]["max_abs_shift_over_implementation_choices"] = max(
        abs(sens[k]["shift_vs_primary"][c]) for k in impl for c in ("1.05", "1.10"))
    rep["reading"]["by_variant"] = {k: v["reading"] for k, v in sens.items()}
    rep["reading"]["budget_form"] = dict(
        label=sens["budget_form"]["reading"], why=sens["budget_form"]["reading_why"],
        delta_1_10=sens["budget_form"]["by_c"]["1.10"]["delta"],
        delta_1_05=sens["budget_form"]["by_c"]["1.05"]["delta"],
        note="a different estimand (best R at P <= c), not the registered one; the "
             "registered reading is the equality form above",
        cells_where_it_differs_at_1_10={
            k: dict(equality=primary["1.10"]["per_cell"][k]["delta"], budget=v)
            for k, v in sens["budget_form"]["by_c"]["1.10"]["per_cell_delta"].items()
            if abs(v - primary["1.10"]["per_cell"][k]["delta"]) > 1e-9})
    rep["reading"]["depends_on_equality_vs_budget_form"] = \
        sens["budget_form"]["reading"] != lab

    # ---- step 3: MMLU-item axis ----------------------------------------------
    rep["matched_mmlu_items"] = collections.OrderedDict(
        (str(b), argmax_budget(runs, cells, b=b, c=None)) for b in B_LEVELS)
    rep["supplement_joint_budget_ppl_1.10"] = collections.OrderedDict(
        (str(b), argmax_budget(runs, cells, b=b, c=1.10)) for b in B_LEVELS)
    rep["supplement_ppl_only_grid_argmax"] = collections.OrderedDict(
        (f"{c:.2f}", argmax_budget(runs, cells, b=None, c=c)) for c in C_LEVELS)
    # POST-HOC: the MMLU axis treated like the ppl axis (isotonic D, interpolated),
    # which removes the dependence of step 3 on where the nine grid points fall
    fd = {k: fitted(run, VAR_D) for k, run in runs.items()}
    mm = collections.OrderedDict()
    for b in B_LEVELS:
        blk = {}
        for tag, var in (("equality_form", VAR_D), ("budget_form", VAR_D_BUDGET)):
            m = matched(runs, cells, float(b), var, fd)
            blk[tag] = dict(delta=m["delta"], n_cells=m["n_cells"], unanimity=m["unanimity"],
                            arm_means=m["arm_means"], unreached_runs=m["unreached_runs"],
                            below_grid_runs=m["below_grid_runs"],
                            cells_dropped=m["cells_dropped"])
        mm[str(b)] = blk
    rep["supplement_mmlu_isotonic"] = dict(
        label="POST-HOC, not registered: isotonic fit of D on log alpha, D(alpha) = b "
              "solved by interpolation, R read there (equality) or maximised below it "
              "(budget)", by_b=mm)
    # POST-HOC descriptive: what the ppl-matched scales cost on the OTHER axis
    cross = collections.OrderedDict()
    for c in C_LEVELS:
        pairs = []
        for cell in cells:
            acc = {arm: [] for arm in ARMS}
            for s in SEEDS:
                sol = {arm: solve(runs[cell + (s, arm)], c, prim,
                                  fits["primary"][cell + (s, arm)]) for arm in ARMS}
                if all(sol[arm]["status"] == "ok" for arm in ARMS):
                    for arm in ARMS:
                        run = runs[cell + (s, arm)]
                        acc[arm].append(interp([math.log(a) for a in run["alpha"]],
                                               [float(d_) for d_ in run["D"]],
                                               math.log(sol[arm]["alpha"])))
            if acc["C-ref"]:
                pairs.append((float(np.mean(acc["C-ref"])), float(np.mean(acc["C-a"]))))
        d = V.boot_paired(pairs)
        cross[f"{c:.2f}"] = dict(
            mean_items={arm: float(np.mean([p[i] for p in pairs]))
                        for i, arm in enumerate(ARMS)},
            ref_minus_a=dict(d, text=V.fmt(d), status=V.status(d)))
    rep["mmlu_drop_at_ppl_matched_scale"] = dict(
        label="POST-HOC descriptive: MMLU drop (items of 200, raw D interpolated "
              "linearly in log alpha) at each arm's ppl-matched scale alpha_c",
        by_c=cross)

    # ---- alpha* overlay (never enters a fit) --------------------------------
    ast, ast_info = load_astar()
    overlay = dict(source=ast_info)
    star = {}
    if ast is not None:
        for cell in cells:
            for s in SEEDS:
                for arm in ARMS:
                    r = ast.get(cell + (s, ASTAR[arm]))
                    ad = (r or {}).get("adaptive") or {}
                    if ad.get("alpha_star") is not None:
                        star[cell + (s, arm)] = ad
        pairs = []
        for cell in [(t, a) for t, a, _d in REG10]:          # v12_astar.CELLS order
            m = {arm: [V.z(star[cell + (s, arm)].get("removal")) if cell + (s, arm) in star
                       else 0.0 for s in SEEDS] for arm in ARMS}
            pairs.append((float(np.mean(m["C-ref"])), float(np.mean(m["C-a"]))))
        d = V.boot_paired(pairs)
        overlay["delta_selection_at_alpha_star"] = dict(
            d, text=V.fmt(d), published=PUBLISHED_ASTAR,
            match={k: _close(d[k], v) for k, v in PUBLISHED_ASTAR.items()})
        ratios = collections.OrderedDict()
        for cell in cells:
            rr = [star[cell + (s, "C-a")]["alpha_star"] / star[cell + (s, "C-ref")]["alpha_star"]
                  for s in SEEDS if cell + (s, "C-a") in star and cell + (s, "C-ref") in star]
            if rr:
                ratios[cname(cell)] = float(np.median(rr))
        med = {arm: float(np.median([star[k]["alpha_star"] for k in star if k[3] == arm]))
               for arm in ARMS}
        overlay["alpha_star"] = dict(
            n_runs=len(star), median={arm: med[arm] for arm in ARMS},
            ratio_of_medians=med["C-a"] / med["C-ref"],
            ratio_of_medians_squared=(med["C-a"] / med["C-ref"]) ** 2,
            cell_seed_median_ratio=ratios,
            median_cell_ratio=float(np.median(list(ratios.values()))),
            median_cell_ratio_squared=float(np.median(list(ratios.values()))) ** 2,
            eval_ppl_at_alpha_star_median={arm: float(np.median(
                [star[k]["ppl_ratio"] for k in star if k[3] == arm])) for arm in ARMS})
    rep["alpha_star_overlay"] = overlay

    # ---- step 6: homogeneity -------------------------------------------------
    idx = boot_index(len(cells))
    chk = boot_mean_fast([v["delta"] for v in primary["1.10"]["per_cell"].values()], idx)
    if max(abs(chk[k] - primary["1.10"]["delta"][k]) for k in ("point", "lo", "hi")) > 1e-12:
        raise SystemExit("matrix bootstrap does not replay v10_common.boot_paired")
    hom = dict(
        rule="points with P - 1 >= 0.005, natural logs, alpha in grid units",
        literal=homogeneity(runs, cells, idx, hi=None),
        posthoc_window=dict(
            label="POST-HOC, not registered: the same fits restricted to "
                  f"0.005 <= P - 1 <= {HOMOG_WINDOW}",
            fits=homogeneity(runs, cells, idx, hi=HOMOG_WINDOW)))
    emp = collections.OrderedDict()
    for c in C_LEVELS:
        per = []
        for cell in cells:
            rr = []
            for s in SEEDS:
                sr = solve(runs[cell + (s, "C-ref")], c, prim, fits["primary"][cell + (s, "C-ref")])
                sa = solve(runs[cell + (s, "C-a")], c, prim, fits["primary"][cell + (s, "C-a")])
                if sr["status"] == sa["status"] == "ok":
                    rr.append(sa["alpha"] / sr["alpha"])
            if rr:
                per.append(float(np.median(rr)))
        emp[f"{c:.2f}"] = dict(median_cell_ratio=float(np.median(per)),
                               median_cell_ratio_squared=float(np.median(per)) ** 2,
                               geometric_mean_cell_ratio=float(np.exp(np.mean(np.log(per)))),
                               min=min(per), max=max(per), n_cells=len(per))
    hom["matched_scale_ratio_a_over_ref_from_curves"] = emp
    if ast is not None:
        hom["alpha_star_ratio_squared"] = dict(
            ratio_of_medians_squared=overlay["alpha_star"]["ratio_of_medians_squared"],
            median_cell_ratio_squared=overlay["alpha_star"]["median_cell_ratio_squared"])
    rep["homogeneity"] = hom

    # ---- step 7: update norm at matched c ------------------------------------
    fro, fro_info = load_frobenius()
    norm = dict(source=fro_info)
    if fro is None:
        norm["status"] = "needs GPU pass (per-arm Frobenius norm not available for both arms)"
    else:
        norm["status"] = "computed"
        norm["definition"] = ("||alpha_c E||_F = alpha_c x frobenius(unit edit); ratio = C-a / "
                              "C-ref per (cell, seed), cell = seed mean, n = cells")
        fr = [fro[cell + (s, "C-a")] / fro[cell + (s, "C-ref")] for cell in cells for s in SEEDS]
        norm["unit_edit_frobenius_ratio_a_over_ref"] = dict(
            median=float(np.median(fr)), min=min(fr), max=max(fr), n_runs=len(fr))
        for vname in ("primary", "lin_alpha"):
            blk = collections.OrderedDict()
            for c in C_LEVELS:
                cell_ratio, cell_n = [], {arm: [] for arm in ARMS}
                for cell in cells:
                    rr, nn = [], {arm: [] for arm in ARMS}
                    for s in SEEDS:
                        sol = {arm: solve(runs[cell + (s, arm)], c, VARIANTS[vname],
                                          fits[vname][cell + (s, arm)]) for arm in ARMS}
                        if all(sol[arm]["status"] == "ok" for arm in ARMS):
                            n_ = {arm: sol[arm]["alpha"] * fro[cell + (s, arm)] for arm in ARMS}
                            rr.append(n_["C-a"] / n_["C-ref"])
                            for arm in ARMS:
                                nn[arm].append(n_[arm])
                    if rr:
                        cell_ratio.append(float(np.mean(rr)))
                        for arm in ARMS:
                            cell_n[arm].append(float(np.mean(nn[arm])))
                bm = boot_mean_fast(cell_ratio, idx) if len(cell_ratio) == len(cells) \
                    else V.boot_mean(cell_ratio)
                blk[f"{c:.2f}"] = dict(
                    ratio_a_over_ref=bm, median_cell_ratio=float(np.median(cell_ratio)),
                    geometric_mean_cell_ratio=float(np.exp(np.mean(np.log(cell_ratio)))),
                    min=min(cell_ratio), max=max(cell_ratio),
                    mean_norm={arm: float(np.mean(cell_n[arm])) for arm in ARMS},
                    per_cell=dict(zip([cname(c_) for c_ in cells], cell_ratio))
                    if len(cell_ratio) == len(cells) else None)
            norm[vname] = blk
    rep["update_norm"] = norm

    # ---- per-run record -------------------------------------------------------
    per_run = collections.OrderedDict()
    for cell in cells:
        for s in SEEDS:
            for arm in ARMS:
                k = cell + (s, arm)
                run = runs[k]
                per_run[f"{cname(cell)}|s{s}|{arm}"] = dict(
                    alpha=run["alpha"], ppl_ratio=run["P"], ppl_isotonic=fits["primary"][k],
                    removal=run["R"], mmlu_drop_items=run["D"], in_budget_v9=run["ok"],
                    frobenius_unit_edit=None if fro is None else fro[k],
                    matched={f"{c:.2f}": solve(run, c, prim, fits["primary"][k])
                             for c in C_LEVELS},
                    alpha_star=star.get(k))
    rep["per_run"] = per_run

    rep["implementation_choices"] = [
        dict(id="I1", choice="isotonic fit = least squares on P (not log P), unit weights, "
                             "pool-adjacent-violators",
             sensitivity="iso_logP; raw_first / raw_last (no isotonic step)"),
        dict(id="I2", choice="between knots the fitted P and R are both linear in log(alpha)",
             sensitivity="lin_alpha (identical R(c) by construction); P_lin__R_log, "
                         "P_log__R_lin, P_sq__R_lin (mixed abscissas)"),
        dict(id="I3", choice="alpha_c = inf{alpha: Pfit(alpha) >= c}; left end of a segment "
                             "flat exactly at c",
             sensitivity="knots_exactly_equal_to_a_c_level in isotonic_diagnostics; raw_last"),
        dict(id="I4", choice="reached = Pfit(512) >= c; Pfit(2) > c is 'below grid', not "
                             "solved and not extrapolated",
             sensitivity="counts per c and arm in matched_ppl"),
        dict(id="I5", choice="a seed enters a cell mean at c only if both arms are solved; "
                             "a cell with no such seed is dropped at that c",
             sensitivity="n_cells / cells_dropped per c"),
        dict(id="I6", choice="step 3 = MMLU-only budget over the nine probed alphas, "
                             "nan -> 0 for a run with no admissible alpha (counted)",
             sensitivity="supplement_joint_budget_ppl_1.10"),
        dict(id="I7", choice="step 6 = one pooled OLS per arm on all points with P-1 >= 0.005 "
                             "(no upper cut-off is registered); R/sqrt(P-1) checked on the "
                             "same points, slope of its log on log alpha, R > 0 only",
             sensitivity="per_run_fits; posthoc_window (labelled post-hoc)"),
        dict(id="I8", choice="step 7 uses the unit-edit Frobenius norm of the v8dec rows for "
                             "all nine alphas (the extension panels do not store it and are "
                             "taken to rebuild the same edit); ratio = C-a / C-ref. It is the "
                             "NOMINAL norm of the float32 edit: the weight change realised "
                             "after the cast to the model dtype is not measured here",
             sensitivity="lin_alpha block in update_norm"),
    ]
    rep["ambiguities"] = [
        "c = 1.05 is the secondary reading level but is absent from step 2's list "
        "{1.02, 1.04, 1.06, 1.08, 1.10}; it is computed by the same procedure.",
        "Step 1 fixes neither the interpolation of the fitted P between knots nor the "
        "abscissa of R's piecewise-linear interpolation (I2).",
        "Step 1 does not say how P(alpha) = c is solved on a flat segment, nor what "
        "'reached' means when the curve starts above c (I3, I4).",
        "Step 3 names only D(alpha) <= b; whether the ppl budget also applies is not "
        "stated (read literally: it does not) (I6).",
        "Step 6: 'per arm' may mean one pooled fit or one fit per run; no upper cut-off "
        "on P is given, so runaway points (P up to 1e15) enter the literal fit (I7).",
        "E1-A's threshold is printed as +0.071; the unrounded quarter of the frozen "
        "estimate is +0.0709.",
        "E1-B' does not say at which c the negative exclusion is read (coded: c = 1.10, "
        "then c = 1.05, flagged if only the latter).",
        "E1-C is 'any other positive result'; a non-positive, non-B, non-B' corner is "
        "not covered by the registered text.",
        "Steps 1-2 read R AT P = c (equality form) while step 3 and the published "
        "estimands read the best R within the budget. The registered ppl-axis text is "
        "explicit, so the equality form is the reading; the budget form is reported as "
        "sensitivity `budget_form` and the reading's dependence on it is recorded in "
        "reading.depends_on_equality_vs_budget_form.",
        "Step 3 is by definition an argmax over the nine probed alphas, so it inherits "
        "the placement of the factor-2 grid relative to each arm's budget scale; "
        "supplement_mmlu_isotonic shows the same axis without that dependence (post-hoc).",
    ]

    # ---- CSVs (the figure reads these and nothing else) ----------------------
    pt_rows = []
    for cell in cells:
        for s in SEEDS:
            for arm in ARMS:
                k = cell + (s, arm)
                run = runs[k]
                for i, a in enumerate(run["alpha"]):
                    pt_rows.append(["grid", cell[0], cell[1], s, arm, a, run["panel"][i],
                                    int(a in FROZEN), run["P"][i], fits["primary"][k][i],
                                    run["R"][i], run["D"][i], int(run["ok"][i]),
                                    None if fro is None else fro[k]])
    for k, ad in star.items():
        pt_rows.append(["alpha_star", k[0], k[1], k[2], k[3], ad["alpha_star"], "v12astar", 0,
                        ad.get("ppl_ratio"), None, ad.get("removal"), ad.get("dmmlu_items"),
                        int(bool(ad.get("eval_ok"))), None])
    write_csv(F_PTS, ["kind", "target", "axis", "seed", "arm", "alpha", "panel", "frozen_grid",
                      "ppl_ratio", "ppl_isotonic", "removal", "mmlu_drop_items",
                      "in_budget_v9", "frobenius_unit_edit"], pt_rows)

    cell_rows = []
    for c in C_LEVELS:
        m = primary[f"{c:.2f}"]
        nb = (norm.get("primary") or {}).get(f"{c:.2f}", {}).get("per_cell") or {}
        for name, v in m["per_cell"].items():
            t, a = name.split("|")
            al = v.get("alpha_c") or {}
            cell_rows.append(["ppl_ratio", c, t, a, v["n_seeds"], v.get("C-ref"), v.get("C-a"),
                              v.get("delta"),
                              float(np.mean(al["C-ref"])) if al else None,
                              float(np.mean(al["C-a"])) if al else None, nb.get(name)])
    for b in B_LEVELS:
        for name, v in rep["matched_mmlu_items"][str(b)]["per_cell"].items():
            t, a = name.split("|")
            cell_rows.append(["mmlu_items", b, t, a, len(SEEDS), v["C-ref"], v["C-a"],
                              v["delta"], None, None, None])
    for label, blk in (("argmax_frozen_grid_v9_gate", san["frozen"]),
                       ("argmax_alpha_le512_v9_gate", san["extension_le512"])):
        for name, v in blk["per_cell"].items():
            t, a = name.split("|")
            cell_rows.append([label, "", t, a, len(SEEDS), v["C-ref"], v["C-a"], v["delta"],
                              None, None, None])
    write_csv(F_CELL, ["axis_kind", "level", "target", "axis", "n_seeds", "R_ref", "R_a",
                       "delta", "mean_alpha_c_ref", "mean_alpha_c_a",
                       "update_norm_ratio_a_over_ref"], cell_rows)

    grid_c = [round(1.0 + 0.0025 * i, 4) for i in range(0, 101)]       # 1.000 .. 1.250
    cur_rows = []
    pooled_vals = {arm: {} for arm in ARMS}
    for cell in cells:
        for arm in ARMS:
            for c in grid_c:
                rs = [solve(runs[cell + (s, arm)], c, prim, fits["primary"][cell + (s, arm)])
                      for s in SEEDS]
                if all(r["status"] == "ok" for r in rs):
                    v = float(np.mean([r["R"] for r in rs]))
                    pooled_vals[arm].setdefault(c, {})[cell] = v
                    cur_rows.append(["cell", cell[0], cell[1], arm, c, v, len(SEEDS), None, None])
    for arm in ARMS:
        for c in grid_c:
            ok = all(len(pooled_vals[a_].get(c, {})) == len(cells) for a_ in ARMS)
            if ok:
                bm = boot_mean_fast([pooled_vals[arm][c][cell] for cell in cells], idx)
                cur_rows.append(["pooled", "", "", arm, c, bm["point"], len(cells),
                                 bm["lo"], bm["hi"]])
    for c in grid_c:
        if all(len(pooled_vals[a_].get(c, {})) == len(cells) for a_ in ARMS):
            bm = boot_mean_fast([pooled_vals["C-ref"][c][cell] - pooled_vals["C-a"][c][cell]
                                 for cell in cells], idx)
            cur_rows.append(["pooled_delta", "", "", "C-ref minus C-a", c, bm["point"],
                             len(cells), bm["lo"], bm["hi"]])
    for arm in ARMS:
        for i, a in enumerate(ALPHAS):
            if a not in FROZEN:
                continue
            P = np.mean([np.mean([runs[cell + (s, arm)]["P"][i] for s in SEEDS]) for cell in cells])
            R = np.mean([np.mean([runs[cell + (s, arm)]["R"][i] for s in SEEDS]) for cell in cells])
            cur_rows.append(["pooled_frozen", "", f"alpha={a:g}", arm, float(P), float(R),
                             len(cells), None, None])
        if star:
            P = np.mean([np.mean([star[cell + (s, arm)]["ppl_ratio"] for s in SEEDS])
                         for cell in cells])
            R = np.mean([np.mean([star[cell + (s, arm)]["removal"] for s in SEEDS])
                         for cell in cells])
            cur_rows.append(["pooled_astar", "", "alpha_star", arm, float(P), float(R),
                             len(cells), None, None])
    write_csv(F_CURVE, ["scope", "target", "axis", "arm", "c", "R", "n", "lo", "hi"], cur_rows)

    outputs = [rel(F_JSON), rel(F_CELL), rel(F_PTS), rel(F_CURVE)]
    rep["figure"] = dict(
        files=None, x_range=list(XLIM),
        note=("lines: seed-mean matched curve R(c) per arm (cell panels) and its cell mean "
              "(pooled panel); last panel: Delta_matched(c) with the 95% cell-bootstrap "
              "interval; points beyond the x range are not drawn; alpha* points sit at "
              "their EVALUATION ppl ratio and are an overlay only; the pooled panel marks "
              "the frozen-grid means and the alpha* means; y axes are per panel"))
    if not no_figure:
        try:
            rep["figure"]["files"] = figure()
            outputs += rep["figure"]["files"]
        except ImportError as e:
            rep["figure"]["files"] = f"not drawn: {e}"
    rep["outputs"] = outputs
    jdump(rep, F_JSON)

    # ---- console ----------------------------------------------------------------
    print("\n[r1-E1] Delta_matched(c) = R_ref(c) - R_a(c), ppl axis "
          "(isotonic P, cell bootstrap)")
    for c in C_LEVELS:
        m = primary[f"{c:.2f}"]
        print(f"   c={c:.2f}  {m['delta']['text']}  unanimity {m['unanimity']}/{m['n_cells']}"
              f"  means ref {m['arm_means']['C-ref']:+.4f} a {m['arm_means']['C-a']:+.4f}"
              f"  unreached {m['unreached_runs']}  below-grid {m['below_grid_runs']}")
    print(f"   READING (registered, equality form): {lab}  ({why})")
    print(f"   stable across implementation choices: "
          f"{rep['reading']['stable_across_implementation_choices']}  (max |shift| "
          f"{rep['reading']['max_abs_shift_over_implementation_choices']:.4f});  "
          f"budget form would read {rep['reading']['budget_form']['label']}")
    print("\n   sensitivities")
    for name, v in sens.items():
        print(f"   {name:14s} c=1.10 {v['by_c']['1.10']['delta']['text']}"
              f"   c=1.05 {v['by_c']['1.05']['delta']['text']}   -> {v['reading']}"
              + ("   [different estimand]" if v["kind"] == "estimand" else ""))
    print("\n[r1-E1] MMLU-item axis: max{R(alpha): D(alpha) <= b}  (MMLU-only, literal)")
    for b in B_LEVELS:
        m = rep["matched_mmlu_items"][str(b)]
        j = rep["supplement_joint_budget_ppl_1.10"][str(b)]
        q = rep["supplement_mmlu_isotonic"]["by_b"][str(b)]
        print(f"   b={b}  {m['delta']['text']}  none->0 {m['runs_with_no_admissible_alpha_scored_0']}"
              f"   | joint with P<=1.10: {j['delta']['text']}")
        print(f"        post-hoc isotonic D: equality {q['equality_form']['delta']['text']}"
              f"   budget {q['budget_form']['delta']['text']}"
              f"   below-grid {q['equality_form']['below_grid_runs']}")
    x_ = rep["mmlu_drop_at_ppl_matched_scale"]["by_c"]["1.10"]
    print(f"   MMLU drop at the ppl-matched scale, c=1.10: C-ref {x_['mean_items']['C-ref']:.2f}"
          f" items, C-a {x_['mean_items']['C-a']:.2f} items; ref - a {x_['ref_minus_a']['text']}")
    for tag, h in (("literal", hom["literal"]), ("post-hoc window", hom["posthoc_window"]["fits"])):
        print(f"\n[r1-E1] homogeneity ({tag}): log(P-1) = log h + p log(alpha)")
        for arm in ARMS:
            a = h[arm]
            f = a["R_over_sqrt_Pm1"]
            print(f"   {arm:6s} p = {a['p']:.3f} [{a['p_ci'][0]:.3f}, {a['p_ci'][1]:.3f}]"
                  f"  h = {a['h']:.3g}  n = {a['n_points']}  r2 = {a['r2']:.3f}"
                  f"  per-run median p = {a['per_run_fits']['median_p']:.3f}"
                  f" (n = {a['per_run_fits']['n_runs']})"
                  f"  | R/sqrt(P-1) slope {f['loglog_slope']:+.3f} "
                  f"[{f['slope_ci'][0]:+.3f}, {f['slope_ci'][1]:+.3f}] flat={f['flat']}")
        print(f"   h_ref/h_a (free p) = {h['h_ratio_free_p']['ratio']:.4g}"
              f"   h_ref/h_a (p = 2) = {h['h_ratio_p_fixed_2']['ratio']:.4g} "
              f"[{h['h_ratio_p_fixed_2']['ci'][0]:.4g}, {h['h_ratio_p_fixed_2']['ci'][1]:.4g}]"
              f"   implied alpha_a/alpha_ref at c=1.10 = "
              f"{h['implied_alpha_ratio_a_over_ref_at_c1.10']['ratio']:.3f}")
    if ast is not None:
        o = overlay["alpha_star"]
        print(f"   alpha* ratio squared: (ratio of medians)^2 = {o['ratio_of_medians_squared']:.2f}"
              f"   (median cell ratio)^2 = {o['median_cell_ratio_squared']:.2f}")
        print(f"   alpha* Delta_selection {overlay['delta_selection_at_alpha_star']['text']}"
              f"  (published +0.073 [-0.036, +0.213])")
    e = emp["1.10"]
    print(f"   matched scale ratio alpha_a/alpha_ref from the curves at c=1.10: median "
          f"{e['median_cell_ratio']:.3f} (squared {e['median_cell_ratio_squared']:.2f})")
    print(f"\n[r1-E1] update norm: {norm['status']}")
    if fro is not None:
        for c in C_LEVELS:
            b_ = norm["primary"][f"{c:.2f}"]
            r_ = b_["ratio_a_over_ref"]
            print(f"   c={c:.2f}  ||alpha E||_F  C-a / C-ref = {r_['point']:.3f} "
                  f"[{r_['lo']:.3f}, {r_['hi']:.3f}]  median {b_['median_cell_ratio']:.3f}")
    print("\nwrote " + ", ".join(outputs))
    return 0


# --------------------------------------------------------------- selftest ----

def selftest():
    assert pava([1, 2, 3]) == [1.0, 2.0, 3.0]
    assert pava([3, 1, 2]) == [2.0, 2.0, 2.0]
    assert pava([1, 3, 2, 4]) == [1.0, 2.5, 2.5, 4.0]
    rng = random.Random(1)
    for _ in range(200):                              # brute-force optimality check
        y = [rng.uniform(0, 3) for _ in range(6)]
        f = pava(y)
        assert all(f[i] <= f[i + 1] + 1e-12 for i in range(5))
        sse = sum((a - b) ** 2 for a, b in zip(f, y))
        for _k in range(50):
            g = sorted(rng.uniform(0, 3) for _ in range(6))
            assert sse <= sum((a - b) ** 2 for a, b in zip(g, y)) + 1e-12
    run = dict(alpha=[2.0, 4.0, 8.0, 16.0], P=[1.0, 1.02, 1.06, 1.20], R=[0.1, 0.2, 0.4, 0.5])
    v = VARIANTS["primary"]
    s = solve(run, 1.04, v)
    assert s["status"] == "ok" and abs(s["alpha"] - 4 * 2 ** 0.5) < 1e-9 and abs(s["R"] - 0.3) < 1e-12
    assert abs(solve(run, 1.02, v)["alpha"] - 4.0) < 1e-9
    assert solve(run, 1.30, v)["status"] == "unreached"
    assert solve(run, 0.99, v)["status"] == "below_grid"
    # one abscissa for both curves -> R(c) is abscissa-free
    assert abs(solve(run, 1.1, v)["R"] - solve(run, 1.1, VARIANTS["lin_alpha"])["R"]) < 1e-12
    # flat exactly at c -> left end; last crossing -> right end
    flat = dict(alpha=[2.0, 4.0, 8.0, 16.0], P=[1.0, 1.05, 1.05, 1.2], R=[0.0, 0.1, 0.2, 0.3])
    assert abs(solve(flat, 1.05, v)["alpha"] - 4.0) < 1e-9
    assert abs(solve(flat, 1.05, VARIANTS["raw_last"])["alpha"] - 8.0) < 1e-9
    d = lambda p, lo, hi: dict(point=p, lo=lo, hi=hi)
    assert reading(d(.10, .02, .2), d(.1, .02, .2))[0] == "E1-A"
    assert reading(d(.05, .01, .1), d(.05, -.01, .1))[0] == "E1-C"
    assert reading(d(.10, -.02, .2), d(.1, -.02, .2))[0] == "E1-B"
    assert reading(d(.10, -.02, .2), d(.1, .02, .2))[0] == "E1-C"
    assert reading(d(-.1, -.2, -.02), d(0, -.1, .1))[0] == "E1-B'"
    idx = boot_index(7, n=200)
    x = [0.3, -0.1, 0.2, 0.05, 0.4, -0.2, 0.1]
    a, b = boot_mean_fast(x, idx), V.boot_paired([(v_, 0.0) for v_ in x], n=200)
    assert max(abs(a[k] - b[k]) for k in ("point", "lo", "hi")) < 1e-12
    print("r1_e1 selftest OK")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "selftest" in args:
        sys.exit(selftest())
    sys.exit(main(no_figure="--no-figure" in args))
