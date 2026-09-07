"""Experiment N (v5, local): validate the T2-scale (7-9B) removal headline with
the two MANDATED nulls + the E2 full-precision comparison, which the cross-designer
panel never carried at 7-9B.

Per target x axis, SELF designer:
  real      = binary sign edit                     (the headline removal)
  signshuf  = same edit, signs shuffled            (SIGN-SHUFFLE null)
  partition = random re-split of the self pool      (DATA-PARTITION null)
  fp        = full-precision contrast delta (E2)     (matched-collateral rival)
best-in-budget removal each (ΔMMLU<=0.02, ppl ratio<=1.10). If real >> both nulls,
the 7-9B removal is real; fp vs real is the E2 practical claim at scale.
"""
import os, sys, json, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, random
import colab_t2t4 as C


def _cpu(dw):
    """Move a weight-delta dict to CPU to keep GPU/RAM peak low (apply_edit
    re-homes to the target device on use)."""
    return {k: v.detach().to("cpu") for k, v in dw.items()}


def _drop(*objs):
    for o in objs:
        del o
    gc.collect(); torch.cuda.empty_cache()

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
ALPHAS = (0.5, 1, 2, 4, 8, 16)
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]


def best_removal(model, tok, E, axis, pre, mmlu, wt, bs):
    best = float("nan")
    for a in ALPHAS:
        undo = C.apply_edit(model, E, alpha=a, sign=-1.0)
        try:
            post = C.fast_eval(model, tok, axis, mmlu, wt, bs)
        finally:
            undo()
        if C.collateral_ok(pre, post):
            r = C.bias_reduction(pre, post)
            best = r if np.isnan(best) else max(best, r)
    return best


def _log(rec):
    with open(os.path.join(RES, "expN_t2nulls.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _done():
    fp = os.path.join(RES, "expN_t2nulls.jsonl")
    d = set()
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                r = json.loads(ln); d.add((r["target"], r["axis"], r["cond"]))
            except Exception:
                pass
    return d


def run(target_id, name, axes, batch_size=12, train_bs=6):
    mmlu = C.load_mmlu(n=128, seed=0); wt = C.load_wikitext(n_chunks=16, seed=0)
    done = _done()
    model, tok = C.load_model(target_id, dispatch=False)
    rows = []
    try:
        for axis in axes:
            need = [c for c in ("fp", "real", "signshuf", "partition")
                    if (name, axis, c) not in done]
            if not need:
                print(f"[N] {name} {axis}: all conditions done", flush=True); continue
            pre = C.fast_eval(model, tok, axis, mmlu, wt, batch_size)
            el = C.elicit_selfdebias(model, tok, axis, 0, batch_size)
            def rec(cond, val):
                _log(dict(target=name, axis=axis, cond=cond, removal=float(val),
                          pre_skew=float(pre["skew"])))
                print(f"[N] {name} {axis:22s} {cond:9s} removal={val:+.4f}", flush=True)

            # --- SELF conditions (fp / real binary / sign-shuffle) ---
            if any(c in need for c in ("fp", "real", "signshuf")):
                dwb, _ = C.train_task_vector(model, tok, el["biased"], rank=16, steps=250,
                                             lr=1e-4, seed=0, bs=train_bs, targets=ATTN,
                                             grad_checkpoint=True)
                dwd, _ = C.train_task_vector(model, tok, el["debiased"], rank=16, steps=250,
                                             lr=1e-4, seed=0, bs=train_bs, targets=ATTN,
                                             grad_checkpoint=True)
                fp_delta = _cpu(C.contrast(dwb, dwd)); _drop(dwb, dwd)
                if "fp" in need:
                    rec("fp", best_removal(model, tok, fp_delta, axis, pre, mmlu, wt, batch_size))
                if "real" in need:
                    E_real = _cpu(C.binarize(fp_delta, "per_tensor", 0.0, 0)[0])
                    rec("real", best_removal(model, tok, E_real, axis, pre, mmlu, wt, batch_size))
                    _drop(E_real)
                if "signshuf" in need:
                    E_shuf = _cpu(C.binarize(fp_delta, "per_tensor", 0.0, 0, random_signs=True)[0])
                    rec("signshuf", best_removal(model, tok, E_shuf, axis, pre, mmlu, wt, batch_size))
                    _drop(E_shuf)
                _drop(fp_delta)
            # --- PARTITION null (random re-split of the self pool) ---
            if "partition" in need:
                pool = list(el["biased"]) + list(el["debiased"])
                random.Random(0).shuffle(pool)
                h = len(pool) // 2
                qwb, _ = C.train_task_vector(model, tok, pool[:h], rank=16, steps=250, lr=1e-4,
                                             seed=0, bs=train_bs, targets=ATTN, grad_checkpoint=True)
                qwd, _ = C.train_task_vector(model, tok, pool[h:], rank=16, steps=250, lr=1e-4,
                                             seed=0, bs=train_bs, targets=ATTN, grad_checkpoint=True)
                E_part = _cpu(C.binarize(C.contrast(qwb, qwd), "per_tensor", 0.0, 0)[0]); _drop(qwb, qwd)
                rec("partition", best_removal(model, tok, E_part, axis, pre, mmlu, wt, batch_size))
                _drop(E_part)
    finally:
        C._free(model)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--name", default="qwen7b")
    ap.add_argument("--axes", nargs="*", default=["occ_gender", "crows_socioeconomic"])
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--train-bs", type=int, default=6)
    a = ap.parse_args()
    run(a.target_id, a.name, a.axes, a.batch_size, a.train_bs)
    print("\n=== Experiment N summary ===")
    fp = os.path.join(RES, "expN_t2nulls.jsonl")
    rows = [json.loads(l) for l in open(fp)] if os.path.exists(fp) else []
    from collections import defaultdict
    by = defaultdict(dict)
    for r in rows:
        by[(r["target"], r["axis"])][r["cond"]] = r["removal"]
    for (tgt, ax), d in by.items():
        print(f"  {tgt} {ax:22s} real={d.get('real', float('nan')):+.3f} "
              f"signshuf={d.get('signshuf', float('nan')):+.3f} "
              f"partition={d.get('partition', float('nan')):+.3f} fp={d.get('fp', float('nan')):+.3f}")


if __name__ == "__main__":
    main()
