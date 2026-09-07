"""v10 — the alpha-extension counterfactual, actually measured.

The manuscript (sec 5.3) says: "Under aggressive alpha extension the gap shrinks
to +0.096 and covers 0." v10 established that no alpha above the frozen grid
maximum of 16 was EVER probed, anywhere in the repo -- so that sentence rests on
an extrapolation, not a measurement. This runs the measurement.

DESIGN. Identical to run_dec.py in every respect that could move a number:
same 10 cells, same 3 seeds, same corpora, same rank-16 / 250-step / lr 1e-4
training, same DENSITY 0.01, same SCALE_MODE ref_tensor, same fast_eval and the
same collateral gate (colab_t2t4.collateral_ok, which delegates to v9_gate).
The ONLY difference is ALPHAS: this probes {32, 64, 128}, strictly above the
frozen deployment grid {2, 4, 8, 16}.

Only C-ref and C-a are run -- Delta_selection is their difference, and the
counterfactual is specifically about continuing the grid-capped C-a runs.

WRITES TO A NEW PANEL (results/v10ext). The frozen v8dec panel is never touched:
its alphas remain the registered grid, and the extended grid is a separate,
clearly post-hoc object. The analysis combines the two traces.

THIS IS EXPLORATORY. The deployment grid was frozen in PREREGISTRATION before
DEC ran; this extension is post-hoc by construction and is reported as such. It
answers "what would have happened off-budget-grid", which is exactly what the
manuscript sentence claims -- it does not change any registered estimand.

Env: transformers 4.57.1 / peft 0.20.0 / torch 2.11.0+cu128 (PREREGISTRATION
L545-547, the v8 environment the v8dec panel itself was produced under).

Run:  python3 run_alpha_ext.py [--alphas 32,64] [--only qwen] [--cells N]
"""
import os, sys, json, time, argparse, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C
import v8_edits as V8
from run_dec import CELLS, DENSITY, SCALE_MODE, SEEDS, applicable_conditions

PANEL = os.environ.get("EXT_PANEL", "v10ext")
OUT = os.path.join(C.RESULTS, PANEL)
CONDITIONS = ["C-ref", "C-a"]
FROZEN_GRID = (2, 4, 8, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="32,64",
                    help="alphas ABOVE the frozen grid to probe")
    ap.add_argument("--only", default=None, help="restrict to one target family")
    ap.add_argument("--cells", type=int, default=0, help="cap number of cells (smoke)")
    ap.add_argument("--seeds", default=None, help="comma list, default 0,1,2")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
    bad = [x for x in alphas if x <= max(FROZEN_GRID)]
    if bad:
        raise SystemExit(f"alphas {bad} are inside the frozen grid {FROZEN_GRID}; "
                         "this script only EXTENDS it")

    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    tr_out = os.path.join(OUT, "alpha_trace.jsonl")
    done = set()
    if os.path.exists(tr_out):
        for ln in open(tr_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["designer"],
                          r["condition"], float(r["alpha"])))
            except Exception:
                pass

    # v11: probe size travels with the panel (see src/v11_selftest.py).
    mmlu = C.load_mmlu(n=int(os.environ.get("EXT_MMLU_N", "200")), seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)

    cells = CELLS[:a.cells] if a.cells else CELLS
    by_model = {}
    for fam, hf, dn, ax in cells:
        if a.only and fam != a.only:
            continue
        by_model.setdefault((fam, hf), []).append((dn, ax))

    t0 = time.time()
    for (fam, hf), specs in by_model.items():
        todo = [(dn, ax, s) for dn, ax in specs for s in seeds
                if any((fam, ax, s, dn, c, al) not in done
                       for c in CONDITIONS for al in alphas)]
        if not todo:
            print(f"[ext] {fam}: nothing to do", flush=True)
            continue
        print(f"[ext] loading {hf} for {len(todo)} (designer,axis,seed) groups",
              flush=True)
        model, tok = C.load_model(hf, dispatch=False)
        try:
            pre_cache = {}
            for dn, ax, s in todo:
                try:
                    if ax not in pre_cache:
                        pre_cache[ax] = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                    pre = pre_cache[ax]
                    corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
                    dwb, _ = C.train_task_vector(model, tok, corp["biased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    dwd, _ = C.train_task_vector(model, tok, corp["debiased"], rank=16,
                                                 steps=250, lr=1e-4, seed=s, bs=8,
                                                 targets=C.ATTN, grad_checkpoint=True)
                    v = C.contrast_(dwb, dwd)
                    del dwb, dwd
                    ref_scales = V8.ref_scale_table(v, DENSITY, s)
                    conds, projs = applicable_conditions(v)

                    for cond in CONDITIONS:
                        if cond not in conds:
                            continue
                        E, meta = V8.build_edit(v, cond, DENSITY, s,
                                                scale_mode=SCALE_MODE,
                                                ref_scales=ref_scales)
                        for al in alphas:
                            if (fam, ax, s, dn, cond, al) in done:
                                continue
                            undo = C.apply_edit(model, E, alpha=al, sign=-1.0)
                            try:
                                post = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                            finally:
                                undo()
                            ok = C.collateral_ok(pre, post)
                            r = C.bias_reduction(pre, post)
                            with open(tr_out, "a") as f:
                                f.write(json.dumps(dict(
                                    panel=PANEL, target=fam, axis=ax, seed=s,
                                    designer=dn, condition=cond, alpha=float(al),
                                    dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                    ppl_ratio=post["ppl"] / pre["ppl"],
                                    collateral_ok=bool(ok),
                                    bias_reduction=float(r),
                                    density=DENSITY, scale_mode=SCALE_MODE,
                                    above_frozen_grid=True)) + "\n")
                            print(f"[ext] {fam}|{ax}|s{s}|{cond}|a={al:g}  "
                                  f"r={r:+.4f} dmmlu={pre['mmlu_acc']-post['mmlu_acc']:+.3f} "
                                  f"ppl={post['ppl']/pre['ppl']:.3f} "
                                  f"ok={bool(ok)}  [{time.time()-t0:.0f}s]", flush=True)
                        del E
                    del v
                    torch.cuda.empty_cache()
                except Exception:                              # noqa: BLE001
                    traceback.print_exc()
                    with open(os.path.join(OUT, "errors.log"), "a") as f:
                        f.write(f"{fam}|{ax}|s{s}|{dn}\n{traceback.format_exc()}\n")
        finally:
            del model
            torch.cuda.empty_cache()
    print(f"[ext] done in {time.time()-t0:.0f}s -> {tr_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
