"""v11 — panel configuration, work-unit expansion, and resume.

WHY THIS EXISTS. Through v10 every panel's cell list, seed list and alpha grid
were module-level Python constants (`run_dec.CELLS`, `run_ins.TARGETS_A`,
`run_alpha_ext.ALPHAS`). That is fine for one box running one panel to
completion. It does not survive a scale-up, where the same panel must be split
across an unknown number of workers, resumed after a node dies, and re-pointed
at a bigger model tier without editing code that the audit reads.

So v11 moves the experiment matrix into `configs/v11/*.yaml` and expands it here
into WORK UNITS. A work unit is one (panel, target, axis, seed, designer) tuple
-- the granularity at which a checkpoint is loaded and a contrast vector is
trained. Every condition/alpha inside a unit reuses that one training pass, so
splitting finer would multiply GPU cost, and splitting coarser would make a
died-at-80% node redo work it had already finished.

THE CONTRACT THE LAUNCHERS RELY ON. `expand()` is deterministic and
order-stable: the same config yields the same unit list in the same order on
every machine. `scripts/plan.py` writes that list once; `submit_local.sh` and
`submit_slurm.sh` both index into it. A SLURM array task and a local worker
therefore agree on what unit 37 is without talking to each other.

RESUME IS READ FROM ARTIFACTS, NEVER FROM A LEDGER. `done()` re-reads the
panel's own removal.jsonl. There is no separate progress file to go stale, and
a unit is complete only when its rows are actually on disk. This mirrors what
run_dec.py already did internally; v11 lifts it to the unit level so a launcher
can skip a unit without loading the model.
"""
import hashlib
import json
import os

try:
    import yaml
except ImportError as e:                       # never silent
    raise SystemExit(
        "PyYAML missing: pip install -r requirements-analysis.txt") from e

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("BS_OUT", os.path.join(REPO, "results"))

# The seven decomposition conditions plus leave-one-projection-out. Kept in sync
# with run_dec.ALL_CONDITIONS by `selftest()` rather than imported, because
# importing run_dec pulls in torch and this module must stay analysis-only.
DEC_CONDITIONS = ["C-ref", "C-a", "C-b", "C-rand", "C-bottom",
                  "C-layershuf", "C-tensorshuf"]
DEC_LOPO = ["C-ref-no_q_proj", "C-ref-no_k_proj",
            "C-ref-no_v_proj", "C-ref-no_o_proj"]

# run_ins.py CONDITIONS. `unedited` is a baseline row, not an edit condition.
INS_CONDITIONS = ["fp", "sign_only", "sparse99", "random_sign"]

# v12: "opsel" = sparsity/binarization curve + adaptive operating point
# (run_opsel.py, experiments_v12.md). Its extra fields travel on the unit.
KINDS = ("dec", "alphaext", "ifeval", "opsel", "frontier")
OPSEL_KEYS = ("phase", "sparsities", "extra_variants", "adaptive_variants",
              "mmlu1k_variants", "alpha_ladder", "refine_steps", "pstar_file",
              "calib_n", "mmlu1k_all_alphas",
              # v13
              "patch_variants", "ifeval_variants", "ifeval_seeds", "deterministic")
# v12: "frontier" = Table A methods re-measured with traces (run_frontier.py)
FRONTIER_KEYS = ("methods", "mmlu1k", "mmlu1k_all_configs", "ifeval_seeds",
                 "ifeval_limit", "eval_batch")


# --------------------------------------------------------------- config ----
def load(path):
    """Read a panel config and check the fields the launchers depend on."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("panel", "kind", "cells"):
        if k not in cfg:
            raise ValueError(f"{path}: config is missing required key '{k}'")
    if cfg["kind"] not in KINDS:
        raise ValueError(
            f"{path}: kind={cfg['kind']!r} is not one of {KINDS}")
    cfg.setdefault("seeds", [0, 1, 2])
    cfg.setdefault("alphas", [2, 4, 8, 16])
    cfg.setdefault("density", 0.01)
    cfg.setdefault("scale_mode", "ref_tensor")
    cfg.setdefault("mmlu_n", 200)
    cfg.setdefault("out_panel", cfg["panel"])
    if cfg["kind"] == "opsel":
        cfg.setdefault("phase", "full")
        if cfg["phase"] not in ("geometry", "full"):
            raise ValueError(f"{path}: phase must be geometry|full, "
                             f"got {cfg['phase']!r}")
    cfg["_path"] = path
    _check_mmlu_n(cfg, path)
    return cfg


def _check_mmlu_n(cfg, path):
    """Refuse an MMLU item count the collateral gate cannot represent.

    src/v9_gate.py recovers the integer item count from a stored accuracy and
    HARD-FAILS off the 1/n_items grid rather than rounding. The budget itself is
    `round(0.02 * n_items)` items, so a count that makes that product
    non-integral gives a budget that is not the stated 2%. Catch it at config
    load, before a node spends an hour discovering it.
    """
    n = cfg["mmlu_n"]
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"{path}: mmlu_n must be a positive int, got {n!r}")
    budget = 0.02 * n
    if abs(budget - round(budget)) > 1e-9:
        raise ValueError(
            f"{path}: mmlu_n={n} gives a collateral budget of {budget} items, "
            "which is not an integer. The gate compares in integer items, so "
            "this silently changes the stated 2% budget. Use a multiple of 50.")


# ----------------------------------------------------------- expansion ----
def unit_id(panel, target, axis, seed, designer):
    """Short, stable, filesystem-safe id. Readable prefix + hash for collisions.

    The hash covers the full tuple so two cells that differ only in a field the
    slug truncates still get distinct ids.
    """
    raw = f"{panel}|{target}|{axis}|{seed}|{designer}"
    h = hashlib.sha1(raw.encode()).hexdigest()[:8]
    slug = f"{target}-{axis}-s{seed}-{designer}".replace("/", "_")
    return f"{slug}-{h}"


def expand(cfg, only=None):
    """Config -> ordered list of work units. Deterministic across machines.

    `only` restricts to cells carrying a given key set truthy in the config,
    e.g. only="published" selects the cells being replayed for the reproduction
    check. Order is unchanged, so a filtered plan is a subsequence of the full
    one and unit ids are identical either way.
    """
    units = []
    for cell in cfg["cells"]:
        if only and not cell.get(only):
            continue
        for k in ("target", "hf", "axis"):
            if k not in cell:
                raise ValueError(
                    f"{cfg['_path']}: cell {cell!r} is missing '{k}'")
        designer = cell.get("designer", cell["target"])
        seeds = cell.get("seeds", cfg["seeds"])
        for seed in seeds:
            units.append(dict(
                id=unit_id(cfg["out_panel"], cell["target"], cell["axis"],
                           seed, designer),
                panel=cfg["out_panel"],
                kind=cfg["kind"],
                target=cell["target"],
                hf=cell["hf"],
                designer=designer,
                # Explicit designer checkpoint. The name->checkpoint map in
                # run_unit only knows the families in colab_t2t4.T2X, which
                # stops at 7-9B, so a big-tier cell cannot name its own
                # checkpoint through it. Without this the 27B/32B cells would
                # silently take their 7-9B SIBLING as designer while run_dec
                # still stamped role="self" -- the published panel is 285/285
                # genuinely self-designer, and mislabelling one cell would put
                # a sibling-designer measurement inside a self-designer panel.
                designer_hf=cell.get("designer_hf"),
                axis=cell["axis"],
                seed=seed,
                alphas=cell.get("alphas", cfg["alphas"]),
                density=cfg["density"],
                scale_mode=cfg["scale_mode"],
                mmlu_n=cfg["mmlu_n"],
                eightbit=bool(cell.get("eightbit", False)),
                ifeval_limit=cell.get("ifeval_limit",
                                      cfg.get("ifeval_limit", 200)),
                config=os.path.relpath(cfg["_path"], REPO),
            ))
            # v12: panel-level run options, each overridable per cell (the
            # calibration panel replays published curves on some cells and
            # measures new ones on others, so their variant lists differ).
            keys = {"opsel": OPSEL_KEYS, "frontier": FRONTIER_KEYS}.get(cfg["kind"], ())
            for k in keys:
                if k in cell:
                    units[-1][k] = cell[k]
                elif k in cfg:
                    units[-1][k] = cfg[k]
    return units


# --------------------------------------------------------------- resume ----
def _rows(panel, fname):
    fp = os.path.join(RESULTS, panel, fname)
    if not os.path.exists(fp):
        return []
    out = []
    with open(fp) as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass                 # a torn last line after a kill is normal
    return out


def _not_applicable(panel, u):
    """Conditions this architecture structurally cannot run (phi's fused qkv)."""
    na = set()
    for r in _rows(panel, "not_applicable.jsonl"):
        if (r.get("target") == u["target"] and r.get("axis") == u["axis"]
                and r.get("seed") == u["seed"]
                and r.get("designer") == u["designer"]):
            na.update(r.get("not_applicable", []))
    return na


def done(u):
    """Is this unit already complete on disk?

    dec       every applicable condition has a removal row
    alphaext  every configured alpha has a trace row for C-ref and C-a
    ifeval    an ifeval row exists for the unit at the selected alpha
    """
    panel = u["panel"]
    if u["kind"] == "dec":
        have = {r["condition"] for r in _rows(panel, "removal.jsonl")
                if r.get("target") == u["target"] and r.get("axis") == u["axis"]
                and r.get("seed") == u["seed"]
                and r.get("designer") == u["designer"]}
        if not have:
            return False
        want = set(DEC_CONDITIONS + DEC_LOPO) - _not_applicable(panel, u)
        return want.issubset(have)

    if u["kind"] == "alphaext":
        want = {float(a) for a in u["alphas"]}
        for cond in ("C-ref", "C-a"):
            have = {float(r["alpha"]) for r in _rows(panel, "alpha_trace.jsonl")
                    if r.get("target") == u["target"]
                    and r.get("axis") == u["axis"]
                    and r.get("seed") == u["seed"]
                    and r.get("condition") == cond}
            if not want.issubset(have):
                return False
        return True

    if u["kind"] == "ifeval":
        # Every edit condition must have a row, not just SOME row with an
        # IFEval score. The old test passed as soon as the `unedited` baseline
        # landed -- which run_ins writes FIRST -- so a cell counted as complete
        # while it was still running, and a cell that died after the baseline
        # would have been skipped on resume, leaving a baseline with no
        # measurement beside it. `ifeval` being None is a legitimate outcome
        # (no configuration in budget at the selected alpha), so completeness
        # is judged on the rows existing, matching run_ins.py's own resume key.
        have = {r.get("condition") for r in _rows(panel, "removal.jsonl")
                if r.get("target") == u["target"] and r.get("axis") == u["axis"]
                and r.get("seed") == u["seed"]}
        return set(INS_CONDITIONS).issubset(have)

    if u["kind"] == "opsel":
        mine = lambda r: (r.get("target") == u["target"]
                          and r.get("axis") == u["axis"]
                          and r.get("seed") == u["seed"]
                          and r.get("designer") == u["designer"])
        if u.get("phase", "full") == "geometry":
            return any(mine(r) for r in _rows(panel, "geometry.jsonl"))
        have = {r["variant"] for r in _rows(panel, "removal.jsonl") if mine(r)}
        return set(opsel_variants(u)).issubset(have)

    if u["kind"] == "frontier":
        have = {(r.get("method"), r.get("row", "select"))
                for r in _rows(panel, "removal.jsonl")
                if r.get("target") == u["target"] and r.get("axis") == u["axis"]
                and r.get("seed") == u["seed"]}
        methods = u.get("methods", ["steering", "sentdebias", "edit"])
        want = {(m, "select") for m in methods}
        if u["seed"] in u.get("ifeval_seeds", [0]):
            want |= {(m, "ifeval") for m in methods + ["unedited"]}
        return want.issubset(have)

    raise ValueError(f"unknown kind {u['kind']!r}")


def opsel_variants(u):
    """The variant names an opsel unit must produce. Mirrors
    run_opsel.variant_names, which cannot be imported here (torch)."""
    sps = u.get("sparsities", [0.0, 0.5, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999])
    extra = u.get("extra_variants", ["fp"])
    return list(extra) + [f"s{float(sp)}" for sp in sps]


def pending(units):
    return [u for u in units if not done(u)]


# ------------------------------------------------------------- selftest ----
def selftest():
    """Checks that cost nothing and catch the drift that would hurt most."""
    fails = []

    # 1. unit ids are stable and unique
    a = unit_id("p", "qwen", "occ_gender", 0, "qwen_3b")
    if a != unit_id("p", "qwen", "occ_gender", 0, "qwen_3b"):
        fails.append("unit_id is not deterministic")
    if a == unit_id("p", "qwen", "occ_gender", 1, "qwen_3b"):
        fails.append("unit_id collides across seeds")

    # 2. the condition list has not drifted from run_dec's
    try:
        import re
        src = open(os.path.join(REPO, "run_dec.py")).read()
        lopo = re.search(r'LOPO = \[f"C-ref-no_\{p\}" for p in PROJECTIONS\]', src)
        projs = re.search(r"^PROJECTIONS = (\[.*?\])", src, re.M)
        if lopo and projs:
            want = [f"C-ref-no_{p}" for p in eval(projs.group(1))]
            if want != DEC_LOPO:
                fails.append(f"DEC_LOPO drifted: {DEC_LOPO} vs run_dec {want}")
    except Exception as e:
        fails.append(f"could not cross-check run_dec conditions: {e}")

    # 3. ifeval completeness must not be satisfied by the baseline row alone
    import tempfile as _tf, json as _json, shutil as _sh
    _d = _tf.mkdtemp()
    try:
        pdir = os.path.join(_d, "tpanel"); os.makedirs(pdir)
        u = dict(panel="tpanel", kind="ifeval", target="t", axis="a", seed=0,
                 designer="d")
        fp = os.path.join(pdir, "removal.jsonl")
        _globals_results = RESULTS
        globals()["RESULTS"] = _d
        with open(fp, "w") as f:
            f.write(_json.dumps(dict(target="t", axis="a", seed=0,
                                     condition="unedited", ifeval={"x": 1})) + "\n")
        if done(u):
            fails.append("ifeval done() satisfied by the unedited baseline alone")
        with open(fp, "a") as f:
            for c in INS_CONDITIONS:
                f.write(_json.dumps(dict(target="t", axis="a", seed=0,
                                         condition=c, ifeval=None)) + "\n")
        if not done(u):
            fails.append("ifeval done() rejects a cell with every condition present")
        globals()["RESULTS"] = _globals_results
    finally:
        _sh.rmtree(_d, ignore_errors=True)

    # 4. the mmlu_n guard actually refuses a bad count
    import tempfile
    for bad in (201, 37):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(f"panel: t\nkind: dec\nmmlu_n: {bad}\ncells: []\n")
            p = f.name
        try:
            load(p)
            fails.append(f"mmlu_n={bad} was accepted; the gate would misread it")
        except ValueError:
            pass
        finally:
            os.unlink(p)

    # 4. a good count still loads
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("panel: t\nkind: dec\nmmlu_n: 1000\ncells: []\n")
        p = f.name
    try:
        load(p)
    except Exception as e:
        fails.append(f"mmlu_n=1000 should load, raised {e}")
    finally:
        os.unlink(p)

    for m in fails:
        print(f"  FAIL {m}")
    print(f"v11_panel selftest: {'PASS' if not fails else str(len(fails)) + ' FAILED'}")
    return not fails


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
