"""Experiment E driver: baseline suite + bit-budget frontier.

experiments_v2 §E ("(b) safety net + systems corollary", runs in parallel with
A/B/C and is publishable regardless of them). This driver covers

  E1  sign source        STE-trained signs vs `sign(Δ_fp)`, at matched collateral
  E2  baseline suite     DPO, PCGU, FairLoRA, inference-time steering, plus the
                         MVP's full-precision task-vector arm and the binary
                         edit itself -- ALL on one frontier
  E3  bit budget         --bit-budget: sparsity sweep across arms, recording
                         bytes / n_flips so bias-reduction-vs-#flips can be
                         plotted per arm

E4 (in-place quantized byte-patch, design.md §9) is deliberately NOT here.

What "matched collateral" means operationally
---------------------------------------------
Nothing is compared at a method's own favourite operating point. Every method
is traced over the SAME α grid, every α is evaluated with the SAME instruments
(evaluate.fast_eval / full_eval), and admissibility is the frozen strict budget
of experiments_v2 §1 -- ΔMMLU ≥ −0.02 AND ppl ratio ≤ 1.10 -- which is exactly
`evaluate.collateral_ok`'s default. A method's headline number is then "best
bias reduction among the α that stay inside the budget", computed in ANALYSIS
from the full trace. design.md §5.4 forbids collapsing a method to a single
scalar at write time, so this driver writes the whole trace and no summary.

Nulls (experiments_v2 §N, mandatory in every condition)
-------------------------------------------------------
The `random` designer role is the DATA-PARTITION null and is in the default
designer set; `binary_random_signs` is the SIGN-SHUFFLE null and is in the
default method set. Every method number in the E2 figure is meant to be read
as a gap above the data-partition null, never raw.

Output
------
results/runs_expE.jsonl -- NOT runs.jsonl. Same record schema as
run_experiment.py (pre/post/bias_reduction/collateral_ok, plus nested
edit_meta/vec_stats/elicit_diag that analyze.load_runs flattens) with an
extra `method` field; `variant` is set to the method tag so the existing
per-variant analysis helpers work unchanged:

    df = analyze.load_runs("results/runs_expE.jsonl")

Resumability
------------
Keys INCLUDE THE ARM and the METHOD. An arm-less key was a real MVP bug: the
second arm generated keys identical to the first arm's, every condition was
skipped as "already done", and the whole run silently wrote nothing
(experiments_v2 §Z). Do not remove either field from the key.

`--dry-run` builds every condition and prints the plan without importing a
checkpoint or touching the GPU.
"""
import os, sys, json, time, argparse, itertools, functools
import numpy as np
import torch

import probes, evaluate as EV
import baselines as BL
from common import load, free, FAMILY, SIZE_B, MODELS
from edit import (train_task_vector, contrast, binarize, apply_edit, merge_dw,
                  vec_stats, cosine, sketch)
from elicit_gen import elicit_selfdebias, exogenous_corpora, elicit_random
from inject import load_injection
from run_experiment import ARMS, AXIS, FAMILY_REL

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "results"))
RUNS = os.path.join(RESULTS, "runs_expE.jsonl")
SKETCH_DIR = os.path.join(RESULTS, "sketches_expE")

# The sign-shuffle null of experiments_v2 §N, expressed as a method so it
# rides the same loop (and reuses the same cached fp contrast).
METHOD_FNS = dict(BL.ALL_METHODS)
METHOD_FNS["binary_random_signs"] = functools.partial(BL.binary_sign,
                                                      random_signs=True)

# Cache-sharing methods first: binary_sign / binary_random_signs /
# fp_task_vector / ste all need the same fp contrast (two LoRA fine-tunes),
# and `contrast_task_vectors(cache=...)` pays for it once per condition.
DEFAULT_METHODS = ["binary_sign", "binary_random_signs", "fp_task_vector",
                   "ste", "dpo", "pcgu", "fair_lora", "steering"]
DEFAULT_DESIGNERS = ["self", "cross", "random"]
DEFAULT_ALPHAS = [0.5, 1.0, 2.0, 4.0]
# experiments_v2 §E3 / design.md §5.6: the bit-budget sweep.
DEFAULT_SPARSITIES = [0.0, 0.5, 0.9, 0.99, 0.999]


# -------------------------------------------------------------------- io
def load_done(path=None):
    # NB: resolve RUNS at call time, never as a default argument -- a default
    # freezes the module-level value at import and silently ignores any later
    # override of the output path.
    path = path or RUNS
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except Exception:
                    pass
    return done


def append(rec, path=None):
    path = path or RUNS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- planning
def roles_for(arm, designers):
    """[(signal, role, designer_name)] for one arm.

    `gt` is not a designer: it is the exogenous ground-truth corpus pair
    (design.md §5.6 ★). `random` is the data-partition null.
    """
    out = []
    for r in designers:
        if r == "gt":
            out.append(("exogenous", "gt", "ground_truth"))
        elif r in ARMS[arm]["designers"]:
            out.append(("endogenous", r, ARMS[arm]["designers"][r]))
    return out


def plan(a):
    """Every record this invocation would write, as a list of dicts.

    Built without loading anything, so `--dry-run` can print it on CPU.
    """
    conds = []
    for arm in a.arms:
        target = ARMS[arm]["target"]
        for seed in a.seeds:
            for origin in a.origins:
                for signal, role, dname in roles_for(arm, a.designers):
                    ck = f"{arm}|{seed}|{origin}|{signal}|{role}"
                    if a.bit_budget:
                        for sp in a.sparsities:
                            tag = f"binary-per_tensor-s{sp}"
                            for alpha in a.alphas:
                                conds.append(dict(
                                    key=f"expE-bits|{ck}|s{sp}|a{alpha}",
                                    mode="bit_budget", arm=arm, target=target,
                                    seed=seed, origin=origin, axis=AXIS[origin],
                                    signal=signal, designer_role=role,
                                    designer=dname, method="binary_sign",
                                    variant=tag, sparsity=sp, alpha=alpha,
                                    cond_key=ck))
                    else:
                        for method in a.methods:
                            alphas = (a.steer_alphas
                                      if method in BL.NON_WEIGHT_METHODS
                                      and a.steer_alphas else a.alphas)
                            for alpha in alphas:
                                conds.append(dict(
                                    key=f"expE|{ck}|{method}|a{alpha}",
                                    mode="methods", arm=arm, target=target,
                                    seed=seed, origin=origin, axis=AXIS[origin],
                                    signal=signal, designer_role=role,
                                    designer=dname, method=method,
                                    variant=method, sparsity=None, alpha=alpha,
                                    cond_key=ck))
    return conds


def print_plan(a, conds, verbose=False):
    mode = "E3 bit-budget" if a.bit_budget else "E1+E2 methods"
    print("=" * 78)
    print(f"Experiment E plan  [{mode}]   -> {RUNS}")
    print("=" * 78)
    print(f"  arms       : {a.arms}  (targets: "
          f"{[ARMS[x]['target'] for x in a.arms]})")
    print(f"  origins    : {a.origins}   axes: "
          f"{[AXIS[o] for o in a.origins]}")
    print(f"  seeds      : {a.seeds}")
    print(f"  designers  : {a.designers}")
    if a.bit_budget:
        print(f"  sparsities : {a.sparsities}")
    else:
        print(f"  methods    : {a.methods}")
        print(f"               (not a weight edit: "
              f"{sorted(BL.NON_WEIGHT_METHODS & set(a.methods))})")
    print(f"  alphas     : {a.alphas}"
          + (f"   steering: {a.steer_alphas}" if a.steer_alphas else ""))
    print(f"  budget     : dMMLU >= -{a.max_mmlu_drop}  AND  "
          f"ppl_ratio <= {a.max_ppl_ratio}   (experiments_v2 §1, strict)")
    print(f"  eval tier  : {a.tier}")
    print("-" * 78)

    # per (arm, seed, origin, signal/role) block
    blocks = {}
    for c in conds:
        blocks.setdefault((c["arm"], c["seed"], c["origin"],
                           c["signal"], c["designer_role"], c["designer"]),
                          []).append(c)
    for (arm, seed, origin, signal, role, dname), cs in blocks.items():
        units = sorted({(c["variant"]) for c in cs})
        print(f"  {arm:6s} seed={seed} {origin:9s} {signal:10s} "
              f"{role:7s}({dname:12s}) x{len(units)} edits x"
              f"{len(cs)//max(len(units),1)} alpha = {len(cs):3d} records")
        if verbose:
            for c in cs:
                print(f"        {c['key']}")
    print("-" * 78)

    by_method = {}
    for c in conds:
        by_method[c["variant"]] = by_method.get(c["variant"], 0) + 1
    for m, n in sorted(by_method.items()):
        print(f"  {m:26s} {n:5d} records")
    done = load_done()
    todo = [c for c in conds if c["key"] not in done]
    n_loads = len({(c["arm"], c["seed"], c["origin"]) for c in conds})
    n_elicit = len({(c["arm"], c["seed"], c["origin"], c["designer_role"])
                    for c in conds if c["designer_role"] not in ("random", "gt")})
    n_edits = len({(c["cond_key"], c["variant"]) for c in conds})
    print("-" * 78)
    print(f"  TOTAL      : {len(conds)} records "
          f"({len(done)} already in {os.path.basename(RUNS)}, "
          f"{len(todo)} to run)")
    print(f"  target loads (arm x seed x origin) : {n_loads}")
    print(f"  designer elicitations              : {n_elicit}")
    print(f"  distinct edits to construct        : {n_edits}")
    print("=" * 78)
    return todo


# --------------------------------------------------------------- execution
def build_target(target, origin, inject_tag):
    """Load T; for the acquired row, merge the injected-bias delta into it."""
    model, tok = load(target)
    info = None
    if origin == "acquired":
        dw, info = load_injection(inject_tag)
        merge_dw(model, dw, alpha=1.0)
    return model, tok, info


def elicit_all(arm, origin, seed, designers, batch_size):
    """Collect every requested designer's corpora once per (arm, origin, seed)."""
    axis = AXIS[origin]
    out = {}
    for signal, role, dname in roles_for(arm, designers):
        t0 = time.time()
        if role == "gt":
            b, d = exogenous_corpora(axis, seed=seed)
            out[(signal, role)] = (b, d, {"contrast_gap": None,
                                          "mode": "exogenous", "axis": axis})
            print(f"  [elicit] {origin}/gt (exogenous ground truth)", flush=True)
            continue
        if role == "random":
            r = elicit_random(axis, seed=seed)
            out[(signal, role)] = (r["biased"], r["debiased"], r["diag"])
            print(f"  [elicit] {origin}/random (data-partition null)", flush=True)
            continue
        dmodel, dtok = load(dname)
        try:
            r = elicit_selfdebias(dmodel, dtok, axis, seed=seed,
                                  batch_size=batch_size)
        finally:
            free(dmodel, dtok)
        out[(signal, role)] = (r["biased"], r["debiased"], r["diag"])
        gap = r["diag"].get("contrast_gap")
        print(f"  [elicit] {origin}/{role}={dname} gap={gap:+.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return out


def method_kwargs(a, method):
    """Per-method hyper-parameters, all overridable from the CLI."""
    base = dict(seed=None, targets=a.targets)
    if method in ("binary_sign", "binary_random_signs", "fp_task_vector"):
        base.update(rank=a.rank, steps=a.steps, lr=a.lr, bs=a.bs)
    elif method == "ste":
        base.update(rank=a.rank, init_steps=a.steps, init_lr=a.lr,
                    init_bs=a.bs, steps=a.ste_steps, lr=a.ste_lr,
                    bs=a.pair_bs, lam=a.ste_lam, margin=a.ste_margin,
                    optimizer=a.ste_optimizer)
    elif method == "dpo":
        base.update(rank=a.rank, steps=a.dpo_steps, lr=a.dpo_lr,
                    bs=a.pair_bs, beta=a.dpo_beta)
    elif method == "pcgu":
        base.update(steps=a.pcgu_steps, lr=a.pcgu_lr, bs=a.pair_bs,
                    top_frac=a.pcgu_top_frac, partition=a.pcgu_partition)
    elif method == "fair_lora":
        base.update(rank=a.rank, steps=a.steps, lr=a.lr, bs=a.bs)
    elif method == "steering":
        base.update(layer=a.steer_layer, bs=a.bs)
        base.pop("targets")
    base["verbose"] = a.verbose
    return base


def edit_vec_stats(E, meta, cache):
    """Direction summary of the RETURNED EDIT, per method.

    `vec_stats` always describes what the method actually produced -- norms
    are only comparable across methods if they measure the same object, and
    the whole point of §E2 is a cross-method comparison. The fp contrast
    (which is a property of the CONDITION, not of the method, and is what
    runs.jsonl's `vec_*` columns hold) is reported separately by
    `contrast_stats` whenever the condition happened to build one.
    """
    if not meta.get("is_weight_edit", True):
        return dict(n_params=int(meta.get("d_model", 0)),
                    l2=float(meta.get("vec_norm", 0.0)), mean_abs=None,
                    frac_nonzero=None, source="activation")
    vs = vec_stats(E)
    vs["source"] = "edit"
    return vs


def contrast_stats(cache):
    """fp-contrast stats for the condition, if any method built one."""
    if "v" not in cache:
        return None
    vs = vec_stats(cache["v"])
    vs["source"] = "fp_contrast"
    return vs


def evaluate_at(model, tok, E, meta, alpha, a, data, acq_axis):
    """Apply the edit at strength alpha, evaluate, restore. Never leaks state."""
    ev = EV.fast_eval if a.tier == "fast" else EV.full_eval
    if meta.get("is_weight_edit", True):
        undo = apply_edit(model, E, alpha=alpha, sign=-1.0)
        try:
            return ev(model, tok, acq_axis, data, a.batch_size)
        finally:
            undo()
    # inference-time steering: no weights change; collateral is measured
    # WITH the hook attached, i.e. it describes a serving configuration.
    with BL.apply_steering_hook(model, E, meta["layer"], alpha, sign=-1.0):
        return ev(model, tok, acq_axis, data, a.batch_size)


def run(a):
    done = load_done()
    conds = plan(a)
    todo = print_plan(a, conds, verbose=a.verbose)
    if a.dry_run:
        return
    if not todo:
        print("nothing to do")
        return

    data = EV.get_eval_data(a.tier)
    # (arm, seed, origin) -> conditions, in plan order
    groups = {}
    for c in todo:
        groups.setdefault((c["arm"], c["seed"], c["origin"]), []).append(c)

    for (arm, seed, origin), cs in groups.items():
        target = ARMS[arm]["target"]
        inject_tag = a.inject_tag or f"{target}_gen_fm_s0"
        axis = AXIS[origin]
        print(f"\n=== arm {arm} ({target}) | seed {seed} | origin {origin} "
              f"| axis {axis} ===", flush=True)
        want_roles = [r for r in a.designers
                      if any(c["designer_role"] == r for c in cs)]
        corpora = elicit_all(arm, origin, seed, want_roles, a.batch_size)

        model, tok, inj = build_target(target, origin, inject_tag)
        try:
            ev = EV.fast_eval if a.tier == "fast" else EV.full_eval
            pre = ev(model, tok, AXIS["acquired"], data, a.batch_size)
            print(f"  [pre] {json.dumps({k: round(v, 4) for k, v in pre.items()})}",
                  flush=True)

            # per (signal, role): one shared fp-contrast cache
            by_role = {}
            for c in cs:
                by_role.setdefault((c["signal"], c["designer_role"]), []).append(c)

            for (signal, role), rcs in by_role.items():
                bset, dset, diag = corpora[(signal, role)]
                cache = {}
                # per edit unit (a method, or a sparsity in bit-budget mode)
                by_unit = {}
                for c in rcs:
                    by_unit.setdefault(c["variant"], []).append(c)

                for variant, ucs in by_unit.items():
                    ucs = [c for c in ucs if c["key"] not in done]
                    if not ucs:
                        continue
                    c0 = ucs[0]
                    t0 = time.time()
                    try:
                        if a.bit_budget:
                            v, cmeta = BL.contrast_task_vectors(
                                model, tok, bset, dset, rank=a.rank,
                                steps=a.steps, lr=a.lr, bs=a.bs, seed=seed,
                                targets=a.targets, cache=cache,
                                verbose=a.verbose)
                            E, meta = binarize(v, granularity="per_tensor",
                                               sparsity=c0["sparsity"],
                                               seed=seed)
                            meta.update(kind="binary", method="binary_sign",
                                        convention="subtract",
                                        is_weight_edit=True, **cmeta)
                        else:
                            kw = method_kwargs(a, c0["method"])
                            kw["seed"] = seed
                            if c0["method"] in ("binary_sign",
                                                "binary_random_signs",
                                                "fp_task_vector", "ste"):
                                kw["cache"] = cache
                            E, meta = METHOD_FNS[c0["method"]](
                                model, tok, bset, dset, **kw)
                    except Exception as exc:      # one method must not kill the run
                        print(f"  [FAIL] {c0['cond_key']}|{variant}: "
                              f"{type(exc).__name__}: {exc}", flush=True)
                        continue

                    vs = edit_vec_stats(E, meta, cache)
                    cs = contrast_stats(cache)
                    print(f"  [edit] {c0['cond_key']}|{variant} "
                          f"({time.time()-t0:.0f}s) "
                          f"weight_edit={meta.get('is_weight_edit', True)} "
                          f"|E|={vs.get('l2')}", flush=True)

                    if a.sketches and meta.get("is_weight_edit", True):
                        os.makedirs(SKETCH_DIR, exist_ok=True)
                        np.save(os.path.join(
                            SKETCH_DIR,
                            f"{c0['cond_key']}|{variant}".replace("/", "_")
                            + ".npy"), sketch(E))

                    for c in sorted(ucs, key=lambda x: x["alpha"]):
                        post = evaluate_at(model, tok, E, meta, c["alpha"], a,
                                           data, AXIS["acquired"])
                        rec = dict(
                            key=c["key"], seed=seed, origin=origin, axis=axis,
                            signal=signal, designer_role=role,
                            designer=c["designer"],
                            designer_family=FAMILY.get(c["designer"], "n/a"),
                            family_relation=FAMILY_REL.get(role, "n/a"),
                            arm=arm, target=target,
                            target_family=FAMILY[target],
                            experiment="E3" if a.bit_budget else "E1E2",
                            method=c["method"], variant=variant,
                            is_weight_edit=bool(meta.get("is_weight_edit", True)),
                            alpha=c["alpha"],
                            edit_meta={k: val for k, val in meta.items()},
                            vec_stats=vs, contrast_stats=cs,
                            elicit_diag=diag, pre=pre, post=post,
                            bias_reduction=EV.bias_reduction(pre, post, axis),
                            collateral_ok=EV.collateral_ok(
                                pre, post, max_mmlu_drop=a.max_mmlu_drop,
                                max_ppl_ratio=a.max_ppl_ratio))
                        append(rec)
                        done.add(c["key"])
                        print(f"    {c['key']:62s} dBias="
                              f"{rec['bias_reduction']:+.4f} "
                              f"mmlu={post['mmlu_acc']:.3f} "
                              f"ppl={post['ppl']:.2f} "
                              f"ok={rec['collateral_ok']}", flush=True)
                    del E
                cache.pop("v", None)
        finally:
            free(model, tok)
    print("\nDONE ->", RUNS)


# -------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Experiment E: baseline suite (E1/E2) and bit-budget "
                    "frontier (E3), on a matched collateral budget.")
    ap.add_argument("--arms", nargs="*", default=["qwen"], choices=list(ARMS))
    ap.add_argument("--origins", nargs="*", default=["inherited", "acquired"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--designers", nargs="*", default=DEFAULT_DESIGNERS,
                    help="designer roles; 'random' = data-partition null, "
                         "'gt' = exogenous ground truth")
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS,
                    choices=list(METHOD_FNS))
    ap.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    ap.add_argument("--steer-alphas", nargs="*", type=float, default=None,
                    help="separate alpha grid for activation steering "
                         "(activation-space units); defaults to --alphas")
    ap.add_argument("--bit-budget", action="store_true",
                    help="E3: sweep --sparsities for the binary edit instead "
                         "of the method suite")
    ap.add_argument("--sparsities", nargs="*", type=float,
                    default=DEFAULT_SPARSITIES)
    ap.add_argument("--targets", nargs="*", default=None,
                    help="edited projections (default q_proj v_proj)")
    # shared LoRA / task-vector hyper-parameters (kept at run_experiment.py's)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bs", type=int, default=8, help="LoRA train batch size")
    ap.add_argument("--pair-bs", type=int, default=4,
                    help="contrastive (paired) batch size for STE/DPO/PCGU")
    # E1
    ap.add_argument("--ste-steps", type=int, default=250)
    ap.add_argument("--ste-lr", type=float, default=1e-2)
    ap.add_argument("--ste-lam", type=float, default=1.0)
    ap.add_argument("--ste-margin", type=float, default=1.0)
    ap.add_argument("--ste-optimizer", default="adamw", choices=["adamw", "sgd"])
    # E2
    ap.add_argument("--dpo-steps", type=int, default=250)
    ap.add_argument("--dpo-lr", type=float, default=1e-4)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--pcgu-steps", type=int, default=50)
    ap.add_argument("--pcgu-lr", type=float, default=1e-4)
    ap.add_argument("--pcgu-top-frac", type=float, default=0.1)
    ap.add_argument("--pcgu-partition", default="row",
                    choices=["row", "column", "tensor"])
    ap.add_argument("--steer-layer", type=int, default=None,
                    help="residual layer for activation steering "
                         "(default: middle layer)")
    # evaluation / budget
    ap.add_argument("--tier", default="fast", choices=["fast", "full"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-mmlu-drop", type=float, default=0.02)
    ap.add_argument("--max-ppl-ratio", type=float, default=1.10)
    ap.add_argument("--inject-tag", default=None)
    ap.add_argument("--sketches", action="store_true", default=True)
    ap.add_argument("--no-sketches", dest="sketches", action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the grid; loads no model")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args(argv)
    if a.bit_budget:
        a.methods = ["binary_sign"]
    run(a)


if __name__ == "__main__":
    main()
