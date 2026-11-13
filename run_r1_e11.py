#!/usr/bin/env python3
"""R1 E11 -- the comparison with other interventions at budget-limited scales.

    python3 run_r1_e11.py --config configs/r1/e11.yaml --count
    python3 run_r1_e11.py --config configs/r1/e11.yaml --unit 0
    python3 run_r1_e11.py --config configs/r1/e11.yaml --progress

PLAN (R1_REBUTTAL_EXPERIMENTS.md, E11). The decomposition is judged with every
edit scaled to its own capability budget, while the comparison of the edit with
steering, SentenceDebias and DPO lets each method choose among four fixed
configurations. Steering keeps 95% of its configurations in budget there: like
the random-coordinate edit on the fixed grid, it may simply not have been
pushed. This runner gives every comparator the rule the edit has under alpha*:
its one continuous (or ordered) knob is raised along a ladder until the budget
first fails ON CALIBRATION ITEMS, refined by bisection, without reading any
bias score (src/r1_select.py, the selector of every R1 panel); the bias is then
measured once, on the evaluation split, at the selected value.

  steering        the published construction (legacy/run_steering_baseline.py:
                  difference of mean activations, added at one layer). Layer
                  fixed in advance per model (config); knob = strength.
  steering_null   a random direction at the same layer, same rule (the
                  published matched-norm null).
  sentdebias      the published construction (legacy/run_sentdebias_baseline.py).
                  Layer fixed in advance; knob = rank over an integer ladder,
                  no bisection; ranks above the number of differing pairs are
                  not used (the subspace has no more dimensions than that).
  sentdebias_null a random subspace of each rank, same rule.
  dpo             the corrected harness of run_r1_dpo.py (train, unload, prove
                  the baseline restored), learning rate fixed (config); knob =
                  merge scale.

The edit's side of the comparison is not run here: C-ref at alpha* is on disk
for all eight cells (results/r1/E5 and results/r1/FRONT8).

Order inside a unit: steering and SentenceDebias before any training, DPO last.
One unit = one (cell, seed) = one process. Resumable at (target, axis, seed,
method). Every calibration probe is persisted (probe_trace.jsonl); per-item
log-likelihoods of the bias probe at the selected value are kept (probe_ll/).
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

from run_r1 import _append, _rows, eval_split    # noqa: E402  (torch-free helpers)

METHODS = ["steering", "steering_null", "sentdebias", "sentdebias_null", "dpo"]


def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(open(path))
    cfg.setdefault("refine_steps", 3)
    cfg.setdefault("lora", dict(r=16, lora_alpha=32))
    return cfg


def unit_list(cfg):
    return [dict(c, seed=s) for c in cfg["cells"] for s in cfg["seeds"]]


def done_methods(out_dir):
    return {(r["target"], r["axis"], r["seed"], r["method"])
            for r in _rows(os.path.join(out_dir, "removal.jsonl"))}


def run_unit(cfg, u, methods, panel):
    import torch
    import colab_t2t4 as C
    import v9_gate
    import run_opsel as RO
    import r1_select as RS
    import run_frontier as RF
    import run_r1_dpo as RD

    out_dir = os.path.join(C.RESULTS, panel)
    os.makedirs(os.path.join(out_dir, "probe_ll"), exist_ok=True)
    fam, hf, dn, ax, s = u["target"], u["hf"], u["designer"], u["axis"], u["seed"]
    key = dict(panel=panel, target=fam, axis=ax, seed=s, designer=dn)
    BATCH = RO.BATCH
    done = done_methods(out_dir)
    todo = [m for m in methods if (fam, ax, s, m) not in done]
    if not todo:
        print(f"[e11] {fam} {ax} s{s}: nothing to do", flush=True)
        return 0

    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)
    mmlu_cal = RO.load_mmlu_calib(200, seed=0)
    wt_cal = C.load_wikitext(n_chunks=20, seed=0, split="validation")
    if set(r["question"] for r in mmlu_cal) & set(r["question"] for r in mmlu):
        raise RuntimeError("calibration and evaluation MMLU items overlap")
    if set(wt_cal) & set(wt):
        raise RuntimeError("calibration and evaluation WikiText chunks overlap")
    corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
    bi, de = corp["biased"], corp["debiased"]
    m_ = min(len(bi), len(de))
    keep = [i for i in range(m_) if bi[i] != de[i]]
    bb, dd = [bi[i] for i in keep], [de[i] for i in keep]

    t0 = time.time()
    model, tok = C.load_model(hf, dispatch=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        pre, pre_ll = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
        pre_cal = RO.collateral_eval(model, tok, mmlu_cal, wt_cal)
        pre_cal_c = RS.collateral_only(pre_cal)
        n_e, n_c = pre.get("n_items", 200), pre_cal["n_items"]
        print(f"[e11] {fam} {ax} s{s} differing pairs={len(keep)}/{m_} "
              f"(load+baselines {time.time() - t0:.0f}s)", flush=True)

        def budget_limited(method, install, ladder, refine, extra):
            """Raise the knob until the calibration budget first fails; measure the
            evaluation split once at the selected value. install(x) -> undo()."""
            ta = time.time()
            probes = []

            def probe(x):
                undo = install(x)
                try:
                    cal = RO.collateral_eval(model, tok, mmlu_cal, wt_cal)
                finally:
                    undo()
                ok = RS.gate(pre_cal_c, RS.collateral_only(cal))
                rec = dict(x=float(x), mmlu_acc=cal["mmlu_acc"], ppl=cal["ppl"],
                           dmmlu_items=int(v9_gate.items_from_acc(pre_cal["mmlu_acc"], n_c)
                                           - v9_gate.items_from_acc(cal["mmlu_acc"], n_c)),
                           ppl_ratio=cal["ppl"] / pre_cal["ppl"], ok=bool(ok))
                probes.append(rec)
                _append(os.path.join(out_dir, "probe_trace.jsonl"),
                        dict(key, method=method, split="calib", **rec))
                return RS.collateral_only(cal)

            sel = RS.select(pre_cal_c, probe, [float(x) for x in ladder], refine)
            x_star = sel["alpha_star"]
            row = dict(key, method=method, ladder=[float(x) for x in ladder], refine_steps=refine,
                       selected=x_star, status=sel["status"], last_pass=sel["last_pass"],
                       first_fail=sel["first_fail"], n_probes=sel["n_probes"], probes=probes,
                       pre_skew=pre["skew"], removal=None, eval_ok=None, **extra)
            if x_star is not None:
                undo = install(x_star)
                try:
                    post, ll = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
                finally:
                    undo()
                row.update(
                    removal=float(C.bias_reduction(pre, post)), post_skew=post["skew"],
                    eval_ok=bool(C.collateral_ok(pre, post)),
                    dmmlu_items=int(v9_gate.items_from_acc(pre["mmlu_acc"], n_e)
                                    - v9_gate.items_from_acc(post["mmlu_acc"], n_e)),
                    ppl_ratio=post["ppl"] / pre["ppl"])
                np.savez_compressed(
                    os.path.join(out_dir, "probe_ll", f"{fam}__{ax}__s{s}__{method}.npz"),
                    x=np.array([x_star]), ll=np.stack([ll]), pre_ll=pre_ll)
            row["seconds"] = round(time.time() - ta, 1)
            _append(os.path.join(out_dir, "removal.jsonl"), row)
            # operational line: no removal is printed
            print(f"[e11] {fam:6s} {ax:20s} s{s} {method:15s} status={sel['status']} "
                  f"selected={x_star} probes={sel['n_probes']} ({row['seconds']:.0f}s)", flush=True)

        def skip(method, why):
            _append(os.path.join(out_dir, "removal.jsonl"),
                    dict(key, method=method, skipped=why, selected=None, removal=None,
                         pre_skew=pre["skew"], n_differing=len(keep)))
            print(f"[e11] {fam} {ax} s{s} {method}: SKIP ({why})", flush=True)

        L8 = RF._legacy("run_steering_baseline")
        SD = RF._legacy("run_sentdebias_baseline")
        n_layers = len(L8.layers_of(model))

        def layer_of(method):
            d = float(cfg["depth"][method][fam])
            return max(0, min(n_layers - 1, int(round(d * (n_layers - 1))))), d

        # ---------------------------------------------------------- steering
        if "steering" in todo or "steering_null" in todo:
            L, d = layer_of("steering")
            a_bi = L8.mean_activation(model, tok, bi, L)
            a_de = L8.mean_activation(model, tok, de, L)
            v_real, act = a_bi - a_de, float(a_bi.norm())
            g = torch.from_numpy(np.random.default_rng(
                [11, s, sum(map(ord, fam + ax))]).standard_normal(v_real.shape[0])).float()
            v_rand = g / g.norm() * v_real.norm()
            for method, v in (("steering", v_real), ("steering_null", v_rand)):
                if method in todo:
                    budget_limited(method, lambda x, v=v: L8.Steer(model, L, v, float(x), act).undo,
                                   cfg["ladders"]["steering"], cfg["refine_steps"],
                                   dict(layer=L, rel_depth=d, act_norm=act,
                                        direction_norm=float(v.norm())))

        # ------------------------------------------------------ SentenceDebias
        if "sentdebias" in todo or "sentdebias_null" in todo:
            L, d = layer_of("sentdebias")
            ranks = [int(k) for k in cfg["ladders"]["sentdebias"] if int(k) <= len(keep)]
            if len(keep) < 5 or not ranks:
                for method in ("sentdebias", "sentdebias_null"):
                    if method in todo:
                        skip(method, "too_few_pairs")
            else:
                sub = {}

                def real(k):
                    k = int(round(k))
                    if k not in sub:
                        np.random.seed(1000 * s + k)        # the PCA is otherwise unseeded
                        sub[k] = SD.bias_subspace(model, tok, bb, dd, L, k)
                    return SD.Project(model, L, sub[k]).undo

                hidden = int(model.config.hidden_size)

                def rand(k):
                    k = int(round(k))
                    gk = torch.from_numpy(np.random.default_rng(
                        [12, s, k, sum(map(ord, fam + ax))]).standard_normal((k, hidden))).float()
                    q, _ = torch.linalg.qr(gk.T)            # orthonormal columns
                    return SD.Project(model, L, q.T.contiguous()).undo

                for method, inst in (("sentdebias", real), ("sentdebias_null", rand)):
                    if method in todo:
                        budget_limited(method, inst, ranks, 0,
                                       dict(layer=L, rel_depth=d, n_differing=len(keep)))

        # ----------------------------------------------------------------- DPO
        if "dpo" in todo:
            arm = cfg["dpo"]
            pairs, n_total = RD.build_pairs(C, dn, ax, s)
            if len(pairs) < int(arm["min_pairs"]):
                skip("dpo", "too_few_pairs")
            else:
                dw, info = RD.train_dpo(C, model, tok, pairs, s, arm, cfg["lora"],
                                        os.path.join(out_dir, "_trl_tmp"))
                chk, _ = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
                if not (chk["skew"] == pre["skew"] and chk["ppl"] == pre["ppl"]
                        and chk["mmlu_acc"] == pre["mmlu_acc"]):
                    raise RuntimeError("baseline not restored after unloading the DPO adapter")
                budget_limited("dpo", lambda x: C.apply_edit(model, dw, alpha=float(x), sign=+1.0),
                               cfg["ladders"]["dpo"], cfg["refine_steps"],
                               dict(n_informative=len(pairs), n_total=n_total, train=info,
                                    recipe={k: arm.get(k) for k in ("lr", "epochs", "batch", "beta",
                                                                    "max_length", "min_pairs")},
                                    baseline_restored=True))
                del dw
    finally:
        C._free(model)
    print(f"[e11] unit {fam} {ax} s{s} finished in {time.time() - t0:.0f}s", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--arms", default=None, help="comma list of methods; default: all")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    units = unit_list(cfg)
    methods = a.arms.split(",") if a.arms else list(METHODS)
    if any(m not in METHODS for m in methods):
        raise SystemExit(f"unknown method in {methods}")
    panel = a.panel or cfg["panel"]
    if a.count:
        print(len(units))
        return 0
    if a.progress:
        done = done_methods(os.path.join(os.environ.get("BS_OUT", "results"), panel))
        need = [(u["target"], u["axis"], u["seed"], m) for u in units for m in methods]
        print(f"{panel}: {sum(k in done for k in need)}/{len(need)} unit-methods on disk "
              f"(methods: {','.join(methods)})")
        return 0
    if a.unit is None:
        raise SystemExit("give --unit, --count or --progress")
    if not 0 <= a.unit < len(units):
        print(f"[e11] unit {a.unit} out of range (0..{len(units) - 1})")
        return 0
    u = units[a.unit]
    print(f"[e11] unit {a.unit}: {u['target']} {u['axis']} s{u['seed']} designer={u['designer']} "
          f"methods={methods} panel={panel}", flush=True)
    try:
        return run_unit(cfg, u, methods, panel)
    except Exception:
        print(f"[e11] unit {a.unit} FAILED", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
