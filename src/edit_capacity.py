"""Does a HIGHER-CAPACITY edit remove naturalistic (CrowS) bias?

Experiment B's second attempt showed the blocker is not the axis inventory but
the edit: low-rank binary edits on q/v projections remove 24-75% of the
*templated* occupation-gender bias but only 2-15% of CrowS bias. CrowS items
are heterogeneous natural sentences, so one compact direction learned on a
disjoint half barely transfers.

This sweeps edit capacity against the EXOGENOUS (ground-truth) signal, which is
the ceiling: if ground truth cannot remove the bias at a given capacity, no
designer can, and the designer comparison is meaningless there. Only if some
configuration lifts the ceiling substantially is it worth re-running the
same/cross comparison.

Factors: LoRA rank x target module set (attention only, MLP only, both).
Everything else is held at the frozen defaults.
"""
import os, argparse, time, json
import numpy as np
import torch

import probes, evaluate as EV
from common import load, free, save_json
from edit import train_task_vector, contrast, binarize, apply_edit, vec_stats
from elicit_gen import exogenous_corpora
import crows_axes as CA

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

TARGET_SETS = {
    "qv":       ["q_proj", "v_proj"],                       # frozen default
    "attn":     ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp":      ["gate_proj", "up_proj", "down_proj"],
    "attn_mlp": ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
}


def run_config(model, tok, axis, targets, rank, alphas, data, seed=0,
               steps=250, lr=1e-4, batch_size=32, pre=None):
    b, d = exogenous_corpora(axis, seed=seed)
    t0 = time.time()
    dw_b, _ = train_task_vector(model, tok, b, rank=rank, steps=steps, lr=lr,
                                seed=seed, targets=targets)
    dw_d, _ = train_task_vector(model, tok, d, rank=rank, steps=steps, lr=lr,
                               seed=seed, targets=targets)
    v = contrast(dw_b, dw_d)
    vs = vec_stats(v)
    del dw_b, dw_d
    E, meta = binarize(v, granularity="per_tensor", sparsity=0.0, seed=seed)
    out = []
    for a in alphas:
        undo = apply_edit(model, E, alpha=a, sign=-1.0)
        try:
            post = EV.fast_eval(model, tok, axis, data, batch_size)
        finally:
            undo()
        out.append(dict(alpha=a, post=post,
                        bias_reduction=EV.bias_reduction(pre, post, axis),
                        collateral_ok=EV.collateral_ok(pre, post)))
    del E, v
    return dict(n_params=vs["n_params"], l2=vs["l2"], secs=time.time() - t0,
                bytes=meta["bytes"], results=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default="qwen1.5b")
    ap.add_argument("--axes", nargs="*",
                    default=["crows_disability", "crows_socioeconomic",
                             "occ_gender"])
    ap.add_argument("--target-sets", nargs="*", default=list(TARGET_SETS))
    ap.add_argument("--ranks", nargs="*", type=int, default=[16, 64])
    ap.add_argument("--alphas", nargs="*", type=float, default=[2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(RESULTS, "edit_capacity.json"))
    a = ap.parse_args()

    data = EV.get_eval_data("fast")
    all_out = {}
    if os.path.exists(a.out):
        all_out = json.load(open(a.out))

    model, tok = load(a.target_model)
    try:
        for axis in a.axes:
            pre = EV.fast_eval(model, tok, axis, data, a.batch_size)
            key0 = EV.primary_metric(axis)[0]
            print(f"\n=== {axis}  pre {key0}={pre[key0]:+.4f} "
                  f"mmlu={pre['mmlu_acc']:.3f} ppl={pre['ppl']:.2f} ===", flush=True)
            for ts in a.target_sets:
                for rank in a.ranks:
                    k = f"{a.target_model}|{axis}|{ts}|r{rank}"
                    if k in all_out:
                        print(f"  [skip] {k}"); continue
                    try:
                        r = run_config(model, tok, axis, TARGET_SETS[ts], rank,
                                       a.alphas, data, steps=a.steps,
                                       batch_size=a.batch_size, pre=pre)
                    except Exception as e:
                        print(f"  [FAIL] {k}: {type(e).__name__}: {e}", flush=True)
                        torch.cuda.empty_cache()
                        continue
                    r["pre"] = pre
                    all_out[k] = r
                    save_json(all_out, a.out)
                    best = max(r["results"], key=lambda x: x["bias_reduction"])
                    inb = [x for x in r["results"] if x["collateral_ok"]]
                    bestb = max(inb, key=lambda x: x["bias_reduction"])["bias_reduction"] if inb else float("nan")
                    print(f"  {ts:9s} r={rank:3d}  params={r['n_params']/1e6:7.1f}M  "
                          f"best={best['bias_reduction']:+.4f} (a={best['alpha']})  "
                          f"in-budget best={bestb:+.4f}  frac={bestb/abs(pre[key0]):+.3f}  "
                          f"({r['secs']:.0f}s)", flush=True)
    finally:
        free(model, tok)
        model = tok = None
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
