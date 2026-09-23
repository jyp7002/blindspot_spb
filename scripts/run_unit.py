#!/usr/bin/env python3
"""Run exactly one work unit.

This is the only thing a worker executes. A SLURM array task, a local queue
slot and a manual retry all call it identically:

    python3 scripts/run_unit.py work/v11dec.units.json --index 7
    python3 scripts/run_unit.py work/v11dec.units.json --id qwen-ss_intra-s1-...

WHY AN ADAPTER RATHER THAN NEW RUNNERS. run_dec.py, run_alpha_ext.py and
run_ins.py are the code that produced the published panels. Rewriting them to
take configs would mean the scaled-up numbers come from different code than the
numbers they must be compared against. Instead this narrows each runner to a
single cell by rebinding its module globals -- the same globals the runner
already reads at call time -- so the executed code path is byte-identical to
the published one.

The runners are individually resumable and append-only, so a unit that dies
half-way is safe to re-run: completed conditions are skipped by the runner's
own `done` set, and `alpha_trace.jsonl` is documented as append-only with
last-occurrence-wins semantics.
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def _load_unit(units_file, index, uid):
    with open(units_file) as f:
        doc = json.load(f)
    units = doc["units"]
    if uid:
        m = [u for u in units if u["id"] == uid]
        if not m:
            raise SystemExit(f"no unit with id {uid!r} in {units_file}")
        return m[0]
    if index is None:
        raise SystemExit("give --index or --id")
    if not 0 <= index < len(units):
        # A SLURM array sized against a stale plan is the normal cause. Exit 0:
        # the array task has nothing to do, and failing would red-flag the job.
        print(f"[unit] index {index} out of range (0..{len(units) - 1}); "
              "nothing to do — the plan is probably stale, re-run plan.py")
        raise SystemExit(0)
    return units[index]


# Designer name -> the checkpoint that elicits its corpus. The large-tier names
# come from colab_t2t4.T2X (family -> 7-9B target); the "_3b" names are the
# sibling column, under the naming run_dec.CELLS already uses.
def _designer_model(designer, explicit=None):
    if explicit:
        return explicit
    import colab_t2t4 as C
    for fam, big, sib in C.T2X:
        if designer == fam:
            return big
        if designer in (fam + "_3b", fam + "_sib"):
            return sib
    extra = {"phi_3b": "microsoft/Phi-3.5-mini-instruct"}
    if designer in extra:
        return extra[designer]
    raise ValueError(
        f"no checkpoint known for designer {designer!r}; add it to T2X or to "
        "the `extra` map in scripts/run_unit.py")


def _ensure_corpus(u):
    """Elicit the designer corpus for this unit if it is not already cached.

    run_dec/run_ins READ a pre-elicited corpus and fail if it is absent -- the
    published panels were elicited in a separate phase-1 pass
    (colab_t2t4.t2x_elicit_pool). A scale-up adds (designer, axis) pairs that
    pass never covered: the large-tier designers have occ_gender and
    crows_socioeconomic only, so every new bbq_Age cell would have died on a
    missing file. Eliciting here keeps a unit self-sufficient, which is what
    lets a fresh box run one without a separate setup phase.

    Inference only, and cached: the second unit that needs the same corpus
    finds it on disk.
    """
    import colab_t2t4 as C
    fp = C._t2x_corpus_fp(u["designer"], u["axis"], u["seed"])
    if os.path.exists(fp):
        return
    mid = _designer_model(u["designer"], u.get("designer_hf"))
    print(f"[unit] corpus missing -> eliciting {u['designer']}|{u['axis']}|"
          f"s{u['seed']} from {mid}", flush=True)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    m, t = C.load_model(mid, dispatch=False)
    try:
        el = C.elicit_selfdebias(m, t, u["axis"], u["seed"], 16)
        # Write atomically: a torn corpus read by a later unit would be a silent
        # data defect rather than a crash.
        tmp = fp + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(designer=u["designer"], axis=u["axis"], seed=u["seed"],
                           biased=el["biased"], debiased=el["debiased"],
                           diag=el["diag"]), f)
        os.replace(tmp, fp)
        print(f"[unit] elicited gap={el['diag']['contrast_gap']:+.3f} "
              f"n_items={el['diag']['n_items']}", flush=True)
    finally:
        C._free(m)


def _run_dec(u):
    _ensure_corpus(u)
    os.environ["DEC_PANEL"] = u["panel"]
    os.environ["DEC_MMLU_N"] = str(u["mmlu_n"])
    import run_dec
    run_dec.CELLS = [(u["target"], u["hf"], u["designer"], u["axis"])]
    run_dec.SEEDS = [u["seed"]]
    run_dec.ALPHAS = tuple(u["alphas"])
    run_dec.DENSITY = u["density"]
    run_dec.SCALE_MODE = u["scale_mode"]
    run_dec.ONLY = None
    run_dec.MMLU_N = u["mmlu_n"]
    run_dec.main()


def _run_alphaext(u):
    _ensure_corpus(u)
    os.environ["EXT_PANEL"] = u["panel"]
    os.environ["EXT_MMLU_N"] = str(u["mmlu_n"])
    import run_alpha_ext
    run_alpha_ext.CELLS = [(u["target"], u["hf"], u["designer"], u["axis"])]
    run_alpha_ext.DENSITY = u["density"]
    run_alpha_ext.SCALE_MODE = u["scale_mode"]
    # run_alpha_ext takes its alphas and seeds from argv, and refuses any alpha
    # inside the frozen grid — that guard is the point, so it is left in place.
    sys.argv = ["run_alpha_ext.py",
                "--alphas", ",".join(str(int(a)) for a in u["alphas"]),
                "--seeds", str(u["seed"])]
    run_alpha_ext.main()


def _run_ifeval(u):
    _ensure_corpus(u)
    sys.argv = ["run_ins.py", "A"]        # ARM is read from argv at import
    import run_ins
    run_ins.PANEL = u["panel"]          # stamped into every row
    run_ins.OUT = os.path.join(run_ins.C.RESULTS, u["panel"])
    run_ins.TARGETS_A = [(u["target"], u["hf"], u["designer"], False, "self")]
    run_ins.AXES_A = [u["axis"]]
    run_ins.SEEDS_A = [u["seed"]]
    run_ins.ALPHAS = tuple(u["alphas"])
    run_ins.IFEVAL_LIMIT = u["ifeval_limit"]
    os.environ["INS_MMLU_N"] = str(u["mmlu_n"])
    run_ins.main()


def _run_opsel(u):
    _ensure_corpus(u)
    os.environ["OPSEL_PANEL"] = u["panel"]
    import run_opsel as R
    R.PANEL = u["panel"]
    R.OUT = os.path.join(R.C.RESULTS, u["panel"])
    R.CELLS = [(u["target"], u["hf"], u["designer"], u["axis"],
                u.get("designer_hf"))]
    R.SEEDS = [u["seed"]]
    R.PHASE = u.get("phase", "full")
    R.ALPHAS = tuple(u["alphas"])
    R.MMLU_N = u["mmlu_n"]
    for key, attr in (("sparsities", "SPARSITIES"),
                      ("extra_variants", "EXTRA_VARIANTS"),
                      ("adaptive_variants", "ADAPTIVE"),
                      ("mmlu1k_variants", "MMLU1K"),
                      ("alpha_ladder", "LADDER"),
                      ("refine_steps", "REFINE"),
                      ("pstar_file", "PSTAR_FILE"),
                      ("calib_n", "CALIB_N"),
                      ("mmlu1k_all_alphas", "MMLU1K_ALL_ALPHAS")):
        if key in u:
            setattr(R, attr, u[key])
    if R.PSTAR_FILE and not os.path.isabs(R.PSTAR_FILE):
        R.PSTAR_FILE = os.path.join(REPO, R.PSTAR_FILE)
    R.main()


def _run_frontier(u):
    _ensure_corpus(u)
    os.environ["FRONTIER_PANEL"] = u["panel"]
    import run_frontier as R
    R.PANEL = u["panel"]
    R.OUT = os.path.join(R.C.RESULTS, u["panel"])
    R.CELLS = [(u["target"], u["hf"], u["designer"], u["axis"])]
    R.SEEDS = [u["seed"]]
    R.ALPHAS = tuple(u["alphas"])
    R.MMLU_N = u["mmlu_n"]
    for key, attr in (("methods", "METHODS"), ("mmlu1k", "MMLU1K"),
                      ("mmlu1k_all_configs", "MMLU1K_ALL_CONFIGS"),
                      ("ifeval_seeds", "IFEVAL_SEEDS"),
                      ("ifeval_limit", "IFEVAL_LIMIT"), ("eval_batch", "BATCH")):
        if key in u:
            setattr(R, attr, u[key])
    R.main()


RUNNERS = {"dec": _run_dec, "alphaext": _run_alphaext, "ifeval": _run_ifeval,
           "opsel": _run_opsel, "frontier": _run_frontier}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("units_file")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--id", dest="uid", default=None)
    ap.add_argument("--skip-done", action="store_true", default=True)
    ap.add_argument("--force", dest="skip_done", action="store_false",
                    help="run even if the artifacts say the unit is complete")
    a = ap.parse_args()

    u = _load_unit(a.units_file, a.index, a.uid)
    print(f"[unit] {u['id']}", flush=True)
    print(f"[unit] panel={u['panel']} kind={u['kind']} target={u['target']} "
          f"axis={u['axis']} seed={u['seed']} hf={u['hf']}", flush=True)

    import v11_panel
    if a.skip_done and v11_panel.done(u):
        print("[unit] already complete on disk — skipping", flush=True)
        return 0

    t0 = time.time()
    try:
        RUNNERS[u["kind"]](u)
    except Exception:
        import traceback
        print(f"[unit] {u['id']} FAILED after {time.time() - t0:.0f}s", flush=True)
        traceback.print_exc()
        return 1

    ok = v11_panel.done(u)
    print(f"[unit] {u['id']} finished in {time.time() - t0:.0f}s; "
          f"artifacts say complete={ok}", flush=True)
    # A unit that ran without raising but did not produce its rows is a silent
    # failure, and silent failures are what this project's audit exists to
    # catch. Report it as one.
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
