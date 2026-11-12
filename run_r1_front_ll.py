#!/usr/bin/env python3
"""R1 E3c fallback -- per-item likelihoods for the eight-cell comparison.

    python3 run_r1_front_ll.py --config configs/v12/frontier.yaml --count
    python3 run_r1_front_ll.py --config configs/v12/frontier.yaml --unit 0
    python3 run_r1_front_ll.py --config configs/v12/frontier.yaml --progress

PLAN (R1_REBUTTAL_EXPERIMENTS.md, E3c, fallback): where per-item probe scores
were not stored, regenerate the intervention and evaluate the bias probe again,
likelihood only. The comparison panel (results/v12frontier: steering,
SentenceDebias and the dense sign-only edit on eight cells, three seeds, four
configurations each) stored one skew per configuration and no per-item values.

This runner rebuilds every configuration with the published constructions
(run_frontier.configs, imported, not rewritten), in the published order and at
the published evaluation batch, and evaluates ONLY the bias probe, keeping the
per-item log-likelihoods. Capability is not measured again: the budget decision
of each configuration is the one stored in results/v12frontier/alpha_trace.jsonl.

REPLAY GUARD. For every configuration the skew recomputed here is compared with
the stored one; so is the unedited skew. The differences are written with the
row (`max_abs_diff`); the reading script uses a method's run only if it replays.

Output: results/r1/E3c/replay.jsonl (one row per unit and method) and
results/r1/E3c/probe_ll/{target}__{axis}__s{seed}__{method}.npz (labels, ll,
pre_ll). One unit = one (cell, seed), one process, as in the published panel.
Resumable at (target, axis, seed, method).
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

PANEL = os.path.join("r1", "E3c")
REF_PANEL = "v12frontier"
METHODS = ["steering", "sentdebias", "edit"]      # the published order


def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(open(path))
    if cfg.get("kind") != "frontier":
        raise SystemExit("this runner replays a `kind: frontier` config")
    return cfg


def unit_list(cfg):
    return [dict(c, seed=s) for c in cfg["cells"] for s in cfg["seeds"]]


def _rows(fp):
    out = []
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def done_keys(out_dir):
    return {(r["target"], r["axis"], r["seed"], r["method"])
            for r in _rows(os.path.join(out_dir, "replay.jsonl"))}


def run_unit(cfg, u):
    import colab_t2t4 as C
    import run_frontier as RF
    from run_r1 import _CaptureLL

    out_dir = os.path.join(C.RESULTS, PANEL)
    os.makedirs(os.path.join(out_dir, "probe_ll"), exist_ok=True)
    fam, hf, dn, ax, s = u["target"], u["hf"], u["designer"], u["axis"], u["seed"]
    done = done_keys(out_dir)
    todo = [m for m in METHODS if (fam, ax, s, m) not in done]
    if not todo:
        print("[front_ll] unit already complete on disk", flush=True)
        return 0
    RF.ALPHAS = tuple(cfg["alphas"])
    batch = int(cfg.get("eval_batch", RF.BATCH))
    ref = {}
    for r in _rows(os.path.join(C.RESULTS, REF_PANEL, "alpha_trace.jsonl")):
        if (r["target"], r["axis"], r["seed"]) == (fam, ax, s):
            ref[(r["method"], r["config"])] = r
    if not ref:
        raise RuntimeError(f"no stored configurations for {fam}|{ax}|s{s} in {REF_PANEL}")

    def probe():
        with _CaptureLL(C) as cap:
            sk = C.axis_skew(model, tok, ax, batch)
        if len(cap.out) != 1:
            raise RuntimeError(f"expected one bias-probe ll call, saw {len(cap.out)}")
        return float(sk), cap.out[0]

    t0 = time.time()
    model, tok = C.load_model(hf, dispatch=False)
    try:
        pre_skew, pre_ll = probe()
        ref_pre = next(iter(ref.values()))["pre_skew"]
        print(f"[front_ll] {fam} {ax} s{s} pre_skew={pre_skew:+.6f} "
              f"stored={ref_pre:+.6f} diff={abs(pre_skew - ref_pre):.2e}", flush=True)
        # every method is rebuilt, in the published order, even if already on disk:
        # a later method must see the model in the state the published run saw
        for method in METHODS:
            cfgs, held = RF.configs(method, model, tok, dn, ax, s)
            key = dict(panel=PANEL, target=fam, axis=ax, seed=s, method=method)
            if not cfgs:
                if method in todo:
                    with open(os.path.join(out_dir, "replay.jsonl"), "a") as f:
                        f.write(json.dumps(dict(key, skipped="too_few_pairs", configs=[])) + "\n")
                continue
            labels, lls, recs = [], [], []
            for label, install in cfgs:
                undo = install()
                try:
                    sk, ll = probe()
                finally:
                    undo()
                st = ref.get((method, label))
                recs.append(dict(config=label, post_skew=sk,
                                 stored_post_skew=(st or {}).get("post_skew"),
                                 collateral_ok=(st or {}).get("collateral_ok"),
                                 stored_removal=(st or {}).get("bias_reduction"),
                                 abs_diff=(abs(sk - st["post_skew"]) if st else None)))
                labels.append(label)
                lls.append(ll)
            diffs = [r["abs_diff"] for r in recs if r["abs_diff"] is not None]
            row = dict(key, configs=recs, pre_skew=pre_skew, stored_pre_skew=ref_pre,
                       pre_abs_diff=abs(pre_skew - ref_pre),
                       max_abs_diff=(max(diffs) if diffs else None),
                       all_configs_stored=len(diffs) == len(recs), eval_batch=batch)
            print(f"[front_ll] {fam} {ax} s{s} {method:10s} configs={len(recs)} "
                  f"max|skew diff|={row['max_abs_diff']}", flush=True)
            if method in todo:
                np.savez_compressed(
                    os.path.join(out_dir, "probe_ll", f"{fam}__{ax}__s{s}__{method}.npz"),
                    labels=np.array(labels), ll=np.stack(lls), pre_ll=pre_ll)
                with open(os.path.join(out_dir, "replay.jsonl"), "a") as f:
                    f.write(json.dumps(row) + "\n")
            if held is not None:
                del held
    finally:
        C._free(model)
    print(f"[front_ll] unit {fam} {ax} s{s} finished in {time.time() - t0:.0f}s", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--arms", default=None, help="ignored (queue compatibility)")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    units = unit_list(cfg)
    if a.count:
        print(len(units))
        return 0
    if a.progress:
        done = done_keys(os.path.join(os.environ.get("BS_OUT", "results"), PANEL))
        need = [(u["target"], u["axis"], u["seed"], m) for u in units for m in METHODS]
        print(f"{PANEL}: {sum(k in done for k in need)}/{len(need)} unit-methods on disk")
        return 0
    if a.unit is None:
        raise SystemExit("give --unit, --count or --progress")
    if not 0 <= a.unit < len(units):
        print(f"[front_ll] unit {a.unit} out of range (0..{len(units) - 1})")
        return 0
    u = units[a.unit]
    print(f"[front_ll] unit {a.unit}: {u['target']} {u['axis']} s{u['seed']} "
          f"designer={u['designer']}", flush=True)
    try:
        return run_unit(cfg, u)
    except Exception:
        print(f"[front_ll] unit {a.unit} FAILED", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
