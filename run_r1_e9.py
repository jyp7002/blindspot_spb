#!/usr/bin/env python3
"""R1 E9 -- held-out capability battery at alpha*.

    python3 run_r1_e9.py --config configs/r1/e9.yaml --count
    python3 run_r1_e9.py --config configs/r1/e9.yaml --unit 0            # likelihood tasks
    python3 run_r1_e9.py --config configs/r1/e9.yaml --unit 0 --arms gen # generation tasks
    python3 run_r1_e9.py --config configs/r1/e9.yaml --progress [--arms gen]

PLAN (R1_REBUTTAL_EXPERIMENTS.md, E9; itTC Q1). The capability budget is MMLU
and perplexity. The random-coordinate edit reaches that budget with a
perturbation about four times larger than the deployed edit's, and may lose
more of a capability the budget does not look at. Each unit (cell, seed)
evaluates four model states on tasks that play no part in any selection:

    unedited | C-ref at its alpha* | C-a at its alpha* | C-rand at its alpha*

The edits are rebuilt exactly as in the other R1 panels (same contrast
training, same builders, imported from run_r1.py) and applied at the alpha*
the stored panels selected for this very run: C-ref and C-a from
results/r1/E5 (which reproduce v12astar), C-rand from results/r1/E4. Nothing
is selected here.

REPLAY GUARDS, recorded with every row: the retrained contrast vector must
have the fingerprint of the registered panel, and the bias skew of each edited
state, measured here before the battery, is compared with the skew the stored
panel recorded at that alpha*.

LIKELIHOOD TASKS (all three seeds). Items are fixed lists; a model state is
scored by the letter it prefers, as the paper scores MMLU:
    arc_challenge   ARC-Challenge test (1,172)
    hellaswag       validation, a fixed seeded subset of 2,000
    truthfulqa_mc1  817 questions, options shuffled per item by a fixed seed
                    (the dataset lists the correct answer first)
    winogrande      winogrande_xl validation (1,267)
    mmlu_heldout    the 800 items of load_mmlu(1000) that are not the 200
                    evaluation items of the budget, scored by colab_t2t4's own
                    MMLU scorer (two rotations), i.e. exactly as the budget is
The first four use one system prompt, no rotation, batch `battery_batch`, items
sorted by prompt length within a number of options (fixed order for all states).

GENERATION TASKS (--arms gen; seeds listed under `gen_seeds`; descriptive):
    gsm8k (first `gsm8k_n` test items) and ifeval (all 541), through lm_eval on
    the live model object, as run_ins.ifeval does for the published IFEval arm.

Output (results/r1/E9): battery.jsonl (one row per unit and state), preds/*.npz
(the chosen option of every item), gen.jsonl. Resumable per (unit, state).
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BATTERY_SYS = ("You are answering multiple-choice questions. Respond with the single letter "
               "of the best answer and nothing else.")
LIK_TASKS = ["arc_challenge", "hellaswag", "truthfulqa_mc1", "winogrande", "mmlu_heldout"]


def load_cfg(path):
    import yaml
    cfg = yaml.safe_load(open(path))
    cfg.setdefault("density", 0.01)
    cfg.setdefault("ref_panel", "v12astar")
    cfg.setdefault("battery_batch", 16)
    cfg.setdefault("hellaswag_n", 2000)
    cfg.setdefault("gsm8k_n", 250)
    cfg.setdefault("ifeval_n", None)
    cfg.setdefault("gen_seeds", [0])
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


def _append(fp, row):
    with open(fp, "a") as f:
        f.write(json.dumps(row) + "\n")


def done_states(out_dir, fname):
    return {(r["target"], r["axis"], r["seed"], r["state"]) for r in _rows(os.path.join(out_dir, fname))}


# ------------------------------------------------------------- battery ----
def load_battery(C, RO, hs_n):
    """-> {task: [dict(q, options, gold)]}; mmlu_heldout in colab_t2t4's row format."""
    from datasets import load_dataset
    tasks = {}
    d = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    rows = []
    for r in d:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if r["answerKey"] in labels:
            rows.append(dict(q="Question: " + r["question"].strip(), options=texts,
                             gold=labels.index(r["answerKey"])))
    tasks["arc_challenge"] = rows

    d = load_dataset("Rowan/hellaswag")["validation"]
    idx = sorted(int(i) for i in np.random.default_rng(0).permutation(len(d))[:hs_n])
    tasks["hellaswag"] = [dict(
        q="Which ending is the most plausible continuation of this text?\n" + d[i]["ctx"].strip(),
        options=[e.strip() for e in d[i]["endings"]], gold=int(d[i]["label"])) for i in idx]

    d = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    rows = []
    for i, r in enumerate(d):
        ch, lab = list(r["mc1_targets"]["choices"]), list(r["mc1_targets"]["labels"])
        order = list(range(len(ch)))
        random.Random(f"r1-e9|truthfulqa|{i}").shuffle(order)
        rows.append(dict(q="Question: " + r["question"].strip(), options=[ch[j] for j in order],
                         gold=[lab[j] for j in order].index(1)))
    tasks["truthfulqa_mc1"] = rows

    d = load_dataset("allenai/winogrande", "winogrande_xl")["validation"]
    tasks["winogrande"] = [dict(
        q="Which option correctly fills the blank (_) in this sentence?\n" + r["sentence"].strip(),
        options=[r["option1"], r["option2"]], gold=int(r["answer"]) - 1) for r in d]

    m1k, ev = C.load_mmlu(n=1000, seed=0), C.load_mmlu(n=200, seed=0)
    if [r["question"] for r in m1k[:200]] != [r["question"] for r in ev]:
        raise RuntimeError("load_mmlu(1000)[:200] != load_mmlu(200)")
    gate = {r["question"] for r in ev} | {r["question"] for r in RO.load_mmlu_calib(200, seed=0)}
    tasks["mmlu_heldout"] = [r for r in m1k[200:] if r["question"] not in gate]
    for name, rows in tasks.items():
        if name != "mmlu_heldout" and any(len(r["options"]) > len(LET) or not 0 <= r["gold"] < len(r["options"])
                                          for r in rows):
            raise RuntimeError(f"bad item in {name}")
    return tasks


def letter_preds(C, model, tok, rows, batch):
    """Chosen option index per item: the letter with the highest probability."""
    import torch
    preds = np.full(len(rows), -1, dtype=np.int16)
    by_n = {}
    for i, r in enumerate(rows):
        by_n.setdefault(len(r["options"]), []).append(i)
    for n in sorted(by_n):
        idx = by_n[n]
        prompts = [C.chat_prompt(tok, rows[i]["q"] + "\n"
                                 + "\n".join(f"{LET[j]}. {o}" for j, o in enumerate(rows[i]["options"]))
                                 + "\nAnswer:", BATTERY_SYS) for i in idx]
        order = sorted(range(len(idx)), key=lambda j: (len(prompts[j]), j))
        with torch.no_grad():
            lp = C.mcq_logprobs(model, tok, [prompts[j] for j in order], list(LET[:n]), batch).numpy()
        for r, j in enumerate(order):
            preds[idx[j]] = int(lp[r].argmax())
    return preds


def run_battery(C, RO, model, tok, tasks, batch):
    acc, preds = {}, {}
    for name in LIK_TASKS:
        rows = tasks[name]
        if name == "mmlu_heldout":
            p, _ = C.mmlu_predict(model, tok, rows, RO.BATCH, n_rot=2)
            p = np.asarray(p, dtype=np.int16)
            gold = np.array([r["gold"] for r in rows])
        else:
            p = letter_preds(C, model, tok, rows, batch)
            gold = np.array([r["gold"] for r in rows])
        preds[name] = p
        acc[name] = float((p == gold).mean())
    return acc, preds


def lm_eval_tasks(model, tok, spec):
    """spec: [(task, limit)] -> {task: {metric: value}} (None on failure), on the live model."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=8)
    out = {}
    for task, limit in spec:
        try:
            res = lm_eval.simple_evaluate(model=lm, tasks=[task], limit=limit, verbosity="ERROR")
            r = res["results"][task]
            out[task] = {k: float(v) for k, v in r.items()
                         if isinstance(v, (int, float)) and "stderr" not in k}
        except Exception as e:
            print(f"[e9] {task} FAILED: {type(e).__name__}: {e}", flush=True)
            out[task] = None
    return out


# ---------------------------------------------------------------- unit ----
def alpha_sources(cfg, C):
    """arm -> {(target, axis, seed): stored removal row of that arm}."""
    out = {}
    for arm, panel in cfg["alpha_sources"].items():
        d = {}
        for r in _rows(os.path.join(C.RESULTS, panel, "removal.jsonl")):
            if r.get("arm") == arm:
                d[(r["target"], r["axis"], r["seed"])] = r
        out[arm] = d
    return out


def run_unit(cfg, u, mode):
    import torch
    import colab_t2t4 as C
    import v8_edits as V8
    import run_opsel as RO
    import run_r1 as R1

    panel = cfg["panel"]
    out_dir = os.path.join(C.RESULTS, panel)
    os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)
    fam, hf, dn, ax, s = u["target"], u["hf"], u["designer"], u["axis"], u["seed"]
    key = dict(panel=panel, target=fam, axis=ax, seed=s, designer=dn)
    density = cfg["density"]
    fname = "gen.jsonl" if mode == "gen" else "battery.jsonl"
    if mode == "gen" and s not in cfg["gen_seeds"]:
        print(f"[e9] {fam} {ax} s{s}: generation tasks are run for seeds {cfg['gen_seeds']} only", flush=True)
        return 0
    states = ["unedited"] + list(cfg["arms"])
    done = done_states(out_dir, fname)
    todo = [st for st in states if (fam, ax, s, st) not in done]
    if not todo:
        print(f"[e9] {fam} {ax} s{s}: nothing to do", flush=True)
        return 0
    src = alpha_sources(cfg, C)
    for arm in cfg["arms"]:
        if (fam, ax, s) not in src[arm]:
            raise RuntimeError(f"no stored alpha* for {arm} {fam}|{ax}|s{s} in {cfg['alpha_sources'][arm]}")
    tasks = load_battery(C, RO, cfg["hellaswag_n"]) if mode == "lik" else None
    gen_spec = [("gsm8k", cfg["gsm8k_n"]), ("ifeval", cfg["ifeval_n"])]

    def measure(state, extra):
        t1 = time.time()
        if mode == "lik":
            acc, preds = run_battery(C, RO, model, tok, tasks, cfg["battery_batch"])
            np.savez_compressed(os.path.join(out_dir, "preds", f"{fam}__{ax}__s{s}__{state}.npz"), **preds)
            row = dict(key, state=state, acc=acc, n={k: len(tasks[k]) for k in LIK_TASKS},
                       macro=float(np.mean([acc[k] for k in LIK_TASKS])), **extra)
            msg = " ".join(f"{k}={acc[k]:.4f}" for k in LIK_TASKS)
        else:
            res = lm_eval_tasks(model, tok, gen_spec)
            row = dict(key, state=state, gen=res, limits=dict(gen_spec), **extra)
            msg = json.dumps(res)[:200]
        row["seconds"] = round(time.time() - t1, 1)
        _append(os.path.join(out_dir, fname), row)
        print(f"[e9] {fam} {ax} s{s} {state:9s} {msg} ({row['seconds']:.0f}s)", flush=True)

    t0 = time.time()
    model, tok = C.load_model(hf, dispatch=False)
    try:
        pre_skew = float(C.axis_skew(model, tok, ax, RO.BATCH))
        if "unedited" in todo:                      # before any training pass, as every baseline
            measure("unedited", dict(alpha=None, skew=pre_skew))
        arms = [a for a in cfg["arms"] if a in todo]
        if arms:
            v, _losses = RO.train_contrast(model, tok, dn, ax, s)
            g = RO.geometry_of(v)
            ref_dir = os.path.join(C.RESULTS, cfg["ref_panel"])
            ref_geo = [r for r in _rows(os.path.join(ref_dir, "geometry.jsonl"))
                       if (r["target"], r["axis"], r["seed"], r["designer"]) == (fam, ax, s, dn)]
            ref_sha = ref_geo[-1]["geometry"]["geom_sha"] if ref_geo else None
            geom_match = None if ref_sha is None else ref_sha == g["geom_sha"]
            ref_scales = V8.ref_scale_table(v, density, s)
            print(f"[e9] {fam} {ax} s{s} geom_sha={g['geom_sha']} match={geom_match} "
                  f"(setup {time.time() - t0:.0f}s)", flush=True)
            for arm in arms:
                st = src[arm][(fam, ax, s)]
                ad = st.get("adaptive") or {}
                a_star = ad.get("alpha_star")
                extra = dict(alpha=a_star, geom_sha=g["geom_sha"], geom_match=geom_match,
                             stored_pre_skew=st.get("pre_skew"),
                             pre_skew_abs_diff=abs(pre_skew - st["pre_skew"]),
                             stored_removal=ad.get("removal"), stored_eval_ok=ad.get("eval_ok"))
                if a_star is None:                  # not deployed: the model is the unedited one
                    extra.update(deployed=False, skew=pre_skew, stored_skew=None, skew_abs_diff=None)
                    measure(arm, extra)
                    continue
                E, _meta = R1.build_arm(v, arm, s, density, ref_scales)
                undo = C.apply_edit(model, E, alpha=float(a_star), sign=-1.0)
                try:
                    skew = float(C.axis_skew(model, tok, ax, RO.BATCH))
                    extra.update(deployed=True, skew=skew, stored_skew=ad.get("post_skew"),
                                 skew_abs_diff=abs(skew - ad["post_skew"]))
                    measure(arm, extra)
                finally:
                    undo()
                del E
            back = float(C.axis_skew(model, tok, ax, RO.BATCH))
            if back != pre_skew:
                raise RuntimeError(f"model not restored after the edits: skew {back} vs {pre_skew}")
    finally:
        C._free(model)
    print(f"[e9] unit {fam} {ax} s{s} finished in {time.time() - t0:.0f}s", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--arms", default="lik", help="lik (likelihood tasks, default) or gen")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    if a.arms not in ("lik", "gen"):
        raise SystemExit("--arms must be lik or gen")
    cfg = load_cfg(a.config)
    units = unit_list(cfg)
    if a.count:
        print(len(units))
        return 0
    if a.progress:
        out_dir = os.path.join(os.environ.get("BS_OUT", "results"), cfg["panel"])
        fname = "gen.jsonl" if a.arms == "gen" else "battery.jsonl"
        done = done_states(out_dir, fname)
        us = [u for u in units if a.arms == "lik" or u["seed"] in cfg["gen_seeds"]]
        need = [(u["target"], u["axis"], u["seed"], st) for u in us for st in ["unedited"] + list(cfg["arms"])]
        print(f"{cfg['panel']} [{a.arms}]: {sum(k in done for k in need)}/{len(need)} unit-states on disk")
        return 0
    if a.unit is None:
        raise SystemExit("give --unit, --count or --progress")
    if not 0 <= a.unit < len(units):
        print(f"[e9] unit {a.unit} out of range (0..{len(units) - 1})")
        return 0
    u = units[a.unit]
    print(f"[e9] unit {a.unit}: {u['target']} {u['axis']} s{u['seed']} designer={u['designer']} "
          f"mode={a.arms}", flush=True)
    try:
        return run_unit(cfg, u, a.arms)
    except Exception:
        print(f"[e9] unit {a.unit} FAILED", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
