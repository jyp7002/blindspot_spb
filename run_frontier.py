"""FRONTIER v12 — the Table A methods, re-measured with full traces.

WHAT THIS REPLACES, AND WHY.
  * Four Table A edit cells (llama, qwen x occ_gender, crows_socioeconomic) come
    from panel `results/x`, which predates alpha-trace persistence and could not
    be re-scored under the integer gate. The manuscript carries a gate-provenance
    disclosure paragraph because of them. Re-measuring them here, with every α's
    collateral persisted, removes the paragraph instead of explaining it.
  * Steering and SentenceDebias have likelihood-only collateral. IFEval -- the
    one generation-side collateral in the paper -- was measured for the edit
    only. This runs it for all three methods under the SAME conditions (seed 0,
    first `ifeval_limit` prompts, the selected configuration active during
    generation), so "the edit wins on persistence, not on side effects" becomes
    a measured statement.
  * MMLU-1000 is read at every configuration of every method, so the frontier's
    tie can be re-checked under a 20-item budget (experiments_v12.md §V).

THE CONSTRUCTIONS ARE THE PUBLISHED ONES, IMPORTED, NOT REWRITTEN.
  edit            colab_t2t4.panel_run's recipe: dense one-bit (binarize at
                  sparsity 0), self designer, rank 16, 250 steps, α in {2,4,8,16}
  steering        legacy/run_steering_baseline.py: mean activation difference,
                  PRIMARY grid (depth {0.50,0.75} x strength {0.02,0.04})
  SentenceDebias  legacy/run_sentdebias_baseline.py: bias_subspace, depth
                  {0.50,0.75} x rank {1,2}
Each method keeps its own 4-configuration selection space (equal selection
space) and the frozen in-budget argmax. Nulls are NOT re-run: the published nulls
stand, and nothing in Table A's re-measurement depends on them.

EVAL BATCH. `panel_run` evaluated at batch 12 except gemma (6); the steering and
SentenceDebias panels at 6. MMLU is batch-composition sensitive at the 1% level
(App. G), so this runner evaluates EVERY method at one batch size (default 6).
Consequence, stated up front: the gemma edit cells should replay
results_v9/v6trace/x2 exactly, the phi edit cells may not (published at 12).

Resumable at (target, axis, seed, method); IFEval rows at (target, axis, seed,
method, 'ifeval').
"""
import importlib.util
import json
import os
import sys
import traceback

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
import colab_t2t4 as C          # noqa: E402
import v9_gate                  # noqa: E402
import v12_opsel as S           # noqa: E402


def _legacy(name):
    """Import a legacy runner by path, so the published construction is reused."""
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, os.path.join(REPO, "legacy"))
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "legacy", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ---- set per unit by scripts/run_unit.py ----
CELLS = []                      # [(target, hf, designer, axis)]
SEEDS = [0]
# Order matters: steering and SentenceDebias were published on a model that had
# never been through train_task_vector, so they run before the edit trains.
METHODS = ["steering", "sentdebias", "edit"]
ALPHAS = (2, 4, 8, 16)
STEER_DEPTHS, STEER_STRENGTHS = [0.50, 0.75], [0.02, 0.04]
SD_DEPTHS, SD_RANKS = [0.50, 0.75], [1, 2]
MMLU1K = True
IFEVAL_SEEDS = [0]
IFEVAL_LIMIT = 200
BATCH = 6
MMLU_N = 200
PANEL = os.environ.get("FRONTIER_PANEL", "v12frontier_smoke")
OUT = os.path.join(C.RESULTS, PANEL)


def _append(fname, row):
    with open(os.path.join(OUT, fname), "a") as f:
        f.write(json.dumps(row) + "\n")


def _done():
    fp = os.path.join(OUT, "removal.jsonl")
    out = set()
    if os.path.exists(fp):
        for ln in open(fp):
            try:
                r = json.loads(ln)
                out.add((r["target"], r["axis"], r["seed"], r["method"],
                         r.get("row", "select")))
            except Exception:
                pass
    return out


# --------------------------------------------------------------- methods ----
def configs(method, model, tok, dn, ax, s):
    """-> list of (config_label, install) where install() -> undo()."""
    if method == "edit":
        corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
        dwb, _ = C.train_task_vector(model, tok, corp["biased"], rank=16, steps=250,
                                     lr=1e-4, seed=s, bs=8, targets=C.ATTN,
                                     grad_checkpoint=True)
        dwd, _ = C.train_task_vector(model, tok, corp["debiased"], rank=16, steps=250,
                                     lr=1e-4, seed=s, bs=8, targets=C.ATTN,
                                     grad_checkpoint=True)
        E, _ = C.binarize(C.contrast_(dwb, dwd), "per_tensor", 0.0, s, inplace=True)
        return [(f"alpha={a}", (lambda a=a: C.apply_edit(model, E, alpha=a, sign=-1.0)))
                for a in ALPHAS], E

    corp = json.load(open(C._t2x_corpus_fp(dn, ax, s)))
    bi, de = corp["biased"], corp["debiased"]
    if method == "steering":
        L8 = _legacy("run_steering_baseline")
        n = len(L8.layers_of(model))
        out = []
        for d in STEER_DEPTHS:
            L = max(0, min(n - 1, int(round(d * (n - 1)))))
            a_bi = L8.mean_activation(model, tok, bi, L)
            a_de = L8.mean_activation(model, tok, de, L)
            v, act = a_bi - a_de, float(a_bi.norm())
            for st in STEER_STRENGTHS:
                def inst(L=L, v=v, st=st, act=act):
                    return L8.Steer(model, L, v, st, act).undo
                out.append((f"L={L},strength={st}", inst))
        return out, None

    if method == "sentdebias":
        SD = _legacy("run_sentdebias_baseline")
        L8 = _legacy("run_steering_baseline")
        m = min(len(bi), len(de))
        keep = [i for i in range(m) if bi[i] != de[i]]
        if len(keep) < 5:
            return [], None                       # published rule: too few pairs
        bb, dd = [bi[i] for i in keep], [de[i] for i in keep]
        n = len(L8.layers_of(model))
        out = []
        for depth in SD_DEPTHS:
            L = max(0, min(n - 1, int(round(depth * (n - 1)))))
            for k in SD_RANKS:
                U = SD.bias_subspace(model, tok, bb, dd, L, k)

                def inst(L=L, U=U):
                    return SD.Project(model, L, U).undo
                out.append((f"L={L},k={k}", inst))
        return out, None
    raise ValueError(method)


def main():
    os.makedirs(OUT, exist_ok=True)
    done = _done()
    mmlu = C.load_mmlu(n=MMLU_N, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)
    mmlu1k = C.load_mmlu(n=1000, seed=0) if MMLU1K else None
    if mmlu1k is not None and [r["question"] for r in mmlu1k[:MMLU_N]] != \
            [r["question"] for r in mmlu]:
        raise RuntimeError("MMLU-1000 is not a superset of the evaluation items")
    by_model = {}
    for fam, hf, dn, ax in CELLS:
        by_model.setdefault((fam, hf), []).append((dn, ax))
    for (fam, hf), specs in by_model.items():
        model, tok = C.load_model(hf, dispatch=False)
        try:
            pre_cache, pre_1k, ifeval_base = {}, None, {}
            for dn, ax in specs:
                for s in SEEDS:
                    if ax not in pre_cache:
                        pre_cache[ax] = C.fast_eval(model, tok, ax, mmlu, wt, BATCH)
                    pre = pre_cache[ax]
                    if MMLU1K and pre_1k is None:
                        pre_1k = C.eval_mmlu(model, tok, mmlu1k, batch_size=BATCH,
                                             n_rot=2)["mmlu_acc"]
                    want_if = s in IFEVAL_SEEDS
                    if want_if and (fam, ax, s, "unedited", "ifeval") not in done:
                        ifeval_base[ax] = _ifeval(model, tok)
                        _append("removal.jsonl", dict(
                            panel=PANEL, target=fam, axis=ax, seed=s,
                            method="unedited", row="ifeval",
                            ifeval=ifeval_base[ax], ifeval_limit=IFEVAL_LIMIT))
                        done.add((fam, ax, s, "unedited", "ifeval"))
                    for method in METHODS:
                        need_sel = (fam, ax, s, method, "select") not in done
                        need_if = want_if and (fam, ax, s, method, "ifeval") not in done
                        if not (need_sel or need_if):
                            continue
                        try:
                            run_method(model, tok, fam, dn, ax, s, method, pre,
                                       pre_1k, mmlu, wt, mmlu1k, need_sel, need_if)
                        except Exception:
                            print(f"[frontier] FAIL {fam} {ax} s{s} {method}", flush=True)
                            traceback.print_exc()
        finally:
            C._free(model)
    print("[frontier] ALL DONE", flush=True)


def _ifeval(model, tok):
    """run_ins.ifeval, the function behind the published IFEval arm."""
    if "run_ins" not in sys.modules:
        argv, sys.argv = sys.argv, ["run_ins.py", "A"]   # ARM is read at import
        try:
            import run_ins  # noqa: F401
        finally:
            sys.argv = argv
    return sys.modules["run_ins"].ifeval(model, tok, IFEVAL_LIMIT)


def run_method(model, tok, fam, dn, ax, s, method, pre, pre_1k, mmlu, wt, mmlu1k,
               need_sel, need_if):
    cfgs, held = configs(method, model, tok, dn, ax, s)
    key = dict(panel=PANEL, target=fam, axis=ax, seed=s, method=method)
    if not cfgs:
        _append("removal.jsonl", dict(key, row="select", removal=None,
                                      skipped="too_few_pairs"))
        return
    results = []
    for label, install in cfgs:
        undo = install()
        try:
            post = C.fast_eval(model, tok, ax, mmlu, wt, BATCH)
            p1k = (C.eval_mmlu(model, tok, mmlu1k, batch_size=BATCH, n_rot=2)["mmlu_acc"]
                   if MMLU1K else None)
        finally:
            undo()
        ok = C.collateral_ok(pre, post)
        r = C.bias_reduction(pre, post)
        n = pre.get("n_items", MMLU_N)
        row = dict(key, config=label, split="eval",
                   pre_mmlu=pre["mmlu_acc"], post_mmlu=post["mmlu_acc"], n_items=n,
                   dmmlu_items=v9_gate.items_from_acc(pre["mmlu_acc"], n)
                   - v9_gate.items_from_acc(post["mmlu_acc"], n),
                   pre_ppl=pre["ppl"], post_ppl=post["ppl"],
                   ppl_ratio=post["ppl"] / pre["ppl"], pre_skew=pre["skew"],
                   post_skew=post["skew"], collateral_ok=bool(ok),
                   bias_reduction=float(r))
        if p1k is not None:
            ok1k = bool(v9_gate.mmlu_ok(pre_1k, p1k, n_items=1000)
                        and v9_gate.ppl_ok(pre["ppl"], post["ppl"]))
            row.update(pre_mmlu1k=pre_1k, post_mmlu1k=p1k, collateral_ok_1k=ok1k)
        _append("alpha_trace.jsonl", row)
        results.append((label, row))

    best, best_label = S.frozen_argmax([(lab, rw["collateral_ok"], rw["bias_reduction"])
                                        for lab, rw in results])
    best1k, best1k_label = (S.frozen_argmax(
        [(lab, rw["collateral_ok_1k"], rw["bias_reduction"]) for lab, rw in results])
        if MMLU1K else (None, None))
    sel = dict(results)[best_label] if best_label else None
    if need_sel:
        _append("removal.jsonl", dict(
            key, row="select", removal=best, config=best_label,
            n_in_budget=sum(rw["collateral_ok"] for _l, rw in results),
            n_configs=len(results),
            dmmlu_items=sel["dmmlu_items"] if sel else None,
            ppl_ratio=sel["ppl_ratio"] if sel else None,
            removal_1k_gate=best1k, config_1k_gate=best1k_label))
    print(f"[frontier] {fam:6s} {ax:20s} s{s} {method:10s} best={best:+.4f} "
          f"@{best_label}  (1000-item gate: {best1k if best1k is None else f'{best1k:+.4f}'}"
          f" @{best1k_label})", flush=True)

    if need_if:
        res = None
        if best_label is not None:
            install = dict(cfgs)[best_label]
            undo = install()
            try:
                res = _ifeval(model, tok)
            finally:
                undo()
        _append("removal.jsonl", dict(key, row="ifeval", config=best_label,
                                      ifeval=res, ifeval_limit=IFEVAL_LIMIT,
                                      removal=best))
    if held is not None:
        del held


if __name__ == "__main__":
    main()
