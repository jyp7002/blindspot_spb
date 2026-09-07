"""DEC (experiments_v8) — sign / support / location decomposition.

THE ADJUDICATION EXPERIMENT. Delta_selection = R(C-ref) - R(C-a) decides whether
this paper extends DARE/TIES/BitDelta (Outcome A: any 1% of the sign field
suffices) or diverges from them (Outcome B: behavioural edits concentrate in a
magnitude-identifiable sparse signed substructure). Both readings are
pre-registered in PREREGISTRATION.md before this file was ever run.

Structure follows run_ablate_attn.py: train the two task vectors ONCE per
(target, axis, seed, designer) and apply every condition to the same contrast
vector, so conditions are exactly paired and differ only in construction.

Also emits, at no extra training cost:
  * the C-ref support dump per (cell, seed)  -> makes SUP1 analysis-only, which
    experiments_v8 assumed was already true and was not.
  * leave-one-projection-out conditions      -> SUP1 §2's only GPU cost.

Resumable at (target, axis, seed, designer, condition).
"""
import os, sys, json, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C
import v8_edits as V8
from v8_support import dump_support, support_path, parse_module

# ---- frozen in PREREGISTRATION.md §v8 DEC before the first run ----
DENSITY = 0.01
SCALE_MODE = "ref_tensor"
ALPHAS = (2, 4, 8, 16)
SEEDS = [0, 1, 2]
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 10 in-envelope cells (positive removal already measured; backfire cells would
# confound a decomposition of an effect that is not there).
CELLS = [
    ("gemma", "google/gemma-2-2b-it",            "gemma_3b", "occ_gender"),
    ("llama", "meta-llama/Llama-3.2-3B-Instruct", "llama_3b", "occ_gender"),
    ("qwen",  "Qwen/Qwen2.5-3B-Instruct",         "qwen_3b",  "occ_gender"),
    ("phi",   "microsoft/Phi-3.5-mini-instruct",  "phi_3b",   "occ_gender"),
    ("phi",   "microsoft/Phi-3.5-mini-instruct",  "phi_3b",   "bbq_Age"),
    ("qwen",  "Qwen/Qwen2.5-3B-Instruct",         "qwen_3b",  "bbq_Age"),
    ("qwen",  "Qwen/Qwen2.5-3B-Instruct",         "qwen_3b",  "bbq_Race_ethnicity"),
    ("qwen",  "Qwen/Qwen2.5-3B-Instruct",         "qwen_3b",  "ss_intra"),
    ("phi",   "microsoft/Phi-3.5-mini-instruct",  "phi_3b",   "ss_intra"),
    ("qwen7b", "Qwen/Qwen2.5-7B-Instruct",        "qwen",     "occ_gender"),
]

LOPO = [f"C-ref-no_{p}" for p in PROJECTIONS]
ALL_CONDITIONS = V8.CONDITIONS + LOPO


def applicable_conditions(v):
    """Drop conditions that are STRUCTURALLY degenerate for this architecture.

    Phi-3.5-mini fuses q/k/v into a single `qkv_proj`, and colab_t2t4's
    TARGET_FALLBACKS resolves ATTN to ['qkv_proj'] ALONE for it -- o_proj is not
    even included. Every other model in the panel resolves to the full
    ['q_proj','k_proj','v_proj','o_proj'].

    On a one-projection-per-layer architecture:
      * C-tensorshuf has no shape-compatible sibling to permute with, so it
        would return C-ref unchanged and report Delta_tensor = 0;
      * every C-ref-no_<proj> matches no module, so it would also return C-ref
        and report a leave-one-out effect of 0.
    Those are structural nulls, not measurements. Averaging them into the
    estimands would manufacture evidence that location does not matter. They are
    marked not-applicable and omitted, and the omission is logged per cell.
    """
    projs = sorted({parse_module(k)[1] for k in v})
    conds = list(V8.CONDITIONS)
    lopo = [f"C-ref-no_{p}" for p in projs] if len(projs) > 1 else []
    if len(projs) < 2:
        conds = [c for c in conds if c != "C-tensorshuf"]
    return conds + lopo, projs

# DEC_PANEL lets a smoke test write somewhere throwaway. The real panel is
# "v8dec"; anything else is not a registered run and must never be cited.
PANEL = os.environ.get("DEC_PANEL", "v8dec")
OUT = os.path.join(C.RESULTS, PANEL)
ONLY = sys.argv[1] if len(sys.argv) > 1 else None      # optional target filter
LIMIT_SEEDS = os.environ.get("DEC_SEEDS")
if LIMIT_SEEDS:
    SEEDS = [int(x) for x in LIMIT_SEEDS.split(",")]

# v11: MMLU probe size. 200 everywhere published; scripts/run_unit.py sets it
# from the panel config. See src/v11_selftest.py for why the count must travel
# with the measurement rather than sit as a default in the gate.
MMLU_N = int(os.environ.get("DEC_MMLU_N", "200"))


def lopo_edit(E_ref, drop_proj):
    """C-ref with one projection's surviving coordinates zeroed (SUP1 §2)."""
    out = {}
    for k, t in E_ref.items():
        _L, proj = parse_module(k)
        out[k] = torch.zeros_like(t) if proj == drop_proj else t.clone()
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    fp_out = os.path.join(OUT, "removal.jsonl")
    tr_out = os.path.join(OUT, "alpha_trace.jsonl")
    done = set()
    if os.path.exists(fp_out):
        for ln in open(fp_out):
            try:
                r = json.loads(ln)
                done.add((r["target"], r["axis"], r["seed"], r["designer"],
                          r["condition"]))
            except Exception:
                pass

    # v11: the collateral probe size is a panel property, not a literal. It is
    # 200 for every published panel and stays 200 unless a config says
    # otherwise; fast_eval now records whichever count it actually scored, so
    # the gate reads the right denominator instead of assuming one.
    mmlu = C.load_mmlu(n=MMLU_N, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)

    # group cells by model so each checkpoint is loaded once
    by_model = {}
    for fam, hf, dn, ax in CELLS:
        if ONLY and fam != ONLY:
            continue
        by_model.setdefault((fam, hf), []).append((dn, ax))

    for (fam, hf), specs in by_model.items():
        todo = [(dn, ax, s) for dn, ax in specs for s in SEEDS
                if any((fam, ax, s, dn, c) not in done for c in ALL_CONDITIONS)]
        if not todo:
            print(f"[dec] {fam}: nothing to do", flush=True)
            continue
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
                    E_ref = None
                    conds, projs = applicable_conditions(v)
                    skipped = [c for c in ALL_CONDITIONS if c not in conds]
                    if skipped:
                        print(f"[dec] {fam} projections={projs} -> N/A (structural): "
                              f"{skipped}", flush=True)
                        with open(os.path.join(OUT, "not_applicable.jsonl"), "a") as f:
                            f.write(json.dumps(dict(
                                target=fam, axis=ax, seed=s, designer=dn,
                                projections=projs, not_applicable=skipped,
                                reason="architecture resolves to <2 attention "
                                       "projections; these conditions would return "
                                       "C-ref unchanged")) + "\n")

                    for cond in conds:
                        if (fam, ax, s, dn, cond) in done:
                            continue
                        if cond in V8.CONDITIONS:
                            E, meta = V8.build_edit(v, cond, DENSITY, s,
                                                    scale_mode=SCALE_MODE,
                                                    ref_scales=ref_scales)
                            if cond == "C-ref":
                                E_ref = {k: t.clone() for k, t in E.items()}
                                sp = support_path(C.RESULTS, PANEL, fam, ax, s, dn)
                                if not os.path.exists(sp):
                                    md = dump_support(sp, E, meta=dict(
                                        target=fam, axis=ax, seed=s, designer=dn,
                                        condition="C-ref", scale_mode=SCALE_MODE))
                                    print(f"[dec] support dump {os.path.basename(sp)} "
                                          f"nnz={md['n_support']:,} "
                                          f"density={md['density']:.5f}", flush=True)
                        else:
                            if E_ref is None:
                                E_ref, _ = V8.build_edit(v, "C-ref", DENSITY, s,
                                                         scale_mode=SCALE_MODE,
                                                         ref_scales=ref_scales)
                            drop = cond.split("no_")[1]
                            E = lopo_edit(E_ref, drop)
                            meta = dict(condition=cond, density=DENSITY,
                                        scale_mode=SCALE_MODE,
                                        n_flips=int(sum((t != 0).sum().item()
                                                        for t in E.values())),
                                        n_params=sum(t.numel() for t in E.values()))

                        best = float("nan")
                        for al in ALPHAS:
                            undo = C.apply_edit(model, E, alpha=al, sign=-1.0)
                            try:
                                post = C.fast_eval(model, tok, ax, mmlu, wt, 6)
                            finally:
                                undo()
                            ok = C.collateral_ok(pre, post)
                            r = C.bias_reduction(pre, post)
                            with open(tr_out, "a") as f:
                                f.write(json.dumps(dict(
                                    target=fam, axis=ax, seed=s, designer=dn,
                                    condition=cond, alpha=float(al),
                                    dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                                    ppl_ratio=post["ppl"] / pre["ppl"],
                                    collateral_ok=bool(ok),
                                    bias_reduction=float(r))) + "\n")
                            if ok:
                                best = r if np.isnan(best) else max(best, r)
                        del E
                        row = dict(panel="v8dec", target=fam, axis=ax, seed=s,
                                   role="self", designer=dn, condition=cond,
                                   removal=best, pre_skew=pre["skew"],
                                   density=DENSITY, scale_mode=SCALE_MODE,
                                   n_flips=meta.get("n_flips"),
                                   n_params=meta.get("n_params"),
                                   frobenius=meta.get("frobenius"))
                        with open(fp_out, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(f"[dec] {fam:6s} {ax:20s} s{s} {cond:16s} "
                              f"removal={best:+.4f}", flush=True)
                    del v, E_ref
                except Exception:
                    print(f"[dec] FAIL {fam} {ax} s{s} {dn}", flush=True)
                    traceback.print_exc()
        finally:
            C._free(model)
    print("[dec] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
