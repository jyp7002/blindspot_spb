#!/usr/bin/env python3
"""R1 E7b -- DPO re-measured with a correct harness.

    python3 run_r1_dpo.py --config configs/r1/e7b.yaml --count
    python3 run_r1_dpo.py --config configs/r1/e7b.yaml --unit 4
    python3 run_r1_dpo.py --config configs/r1/e7b.yaml --progress

WHY THIS EXISTS. run_dpo_baseline.py, the script behind the published DPO
column, took its baseline on the model with the trained DPO adapter still
attached and then subtracted the delta (R1_NOTES.md section 4, E7a). This runner
measures the comparison that was intended:

  1. load the checkpoint and take the baseline on the CLEAN model, before any
     training (fast_eval on the evaluation split, exactly as every edit panel);
  2. train the DPO LoRA with TRL's DPOTrainer on preference pairs built as the
     published script built them (chosen = debiased[i], rejected = biased[i],
     empty prompt, pairs whose two sides are identical dropped);
  3. materialise dW = B A * (lora_alpha / r) and UNLOAD the adapter, then
     re-measure the baseline and require it to equal step 1 bit for bit -- the
     check whose absence let the published error through;
  4. ADD m * dW to the clean model (apply_edit with sign = +1) for every merge
     scale m of the arm and measure skew, MMLU and perplexity with it active.

ARMS (configs/r1/e7b.yaml). Each arm is one training recipe and a list of merge
scales; every (arm, merge scale) measurement is persisted, so any selection
space can be read afterwards without GPU time:

  pub       the as-published recipe (lr 1e-4, 250 steps, batch 2, beta 0.1) at
            the as-published merge scales {0.25, 0.5, 0.75, 1} plus a finer
            ladder below them. Checks the E7a reconstruction by measurement.
  lr1e-5,   the registered E7b design: LoRA r16 on the edit's modules, beta 0.1,
  lr5e-5    a fixed number of epochs over the differing pairs, merge scales
            {1, 2}. The registered 4-configuration space is
            {lr1e-5, lr5e-5} x {1, 2}; the skip threshold is 8 differing pairs.

One unit = one (target, axis, seed) = one process; arms run in config order,
each on a clean model (the unload check guarantees it). Resumable at
(target, axis, seed, arm).
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


def load_cfg(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("lora", dict(r=16, lora_alpha=32))
    return cfg


def unit_list(cfg):
    return [dict(c, seed=s) for c in cfg["cells"] for s in cfg["seeds"]]


def done_arms(out_dir):
    return {(r["target"], r["axis"], r["seed"], r["arm"])
            for r in _rows(os.path.join(out_dir, "removal.jsonl"))}


def build_pairs(C, dn, ax, seed):
    """run_dpo_baseline.build_pairs, verbatim in effect."""
    c = json.load(open(C._t2x_corpus_fp(dn, ax, seed)))
    b, d = c["biased"], c["debiased"]
    n = min(len(b), len(d))
    pairs = [dict(prompt="", chosen=d[i], rejected=b[i]) for i in range(n) if d[i] != b[i]]
    return pairs, n


def train_dpo(C, model, tok, pairs, seed, arm, lora, tmp_dir):
    """Train a DPO LoRA on `model`, return (dW, info), leave `model` CLEAN."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    targets = C.resolve_targets(model, C.ATTN)
    kw = dict(output_dir=tmp_dir, beta=float(arm["beta"]),
              learning_rate=float(arm["lr"]),
              per_device_train_batch_size=int(arm["batch"]),
              logging_steps=10_000, save_strategy="no", report_to=[], bf16=True,
              gradient_checkpointing=True, remove_unused_columns=False,
              max_length=int(arm["max_length"]), seed=seed)
    if arm.get("steps"):
        kw["max_steps"] = int(arm["steps"])
    else:
        kw["num_train_epochs"] = float(arm["epochs"])
    cfg = DPOConfig(**kw)
    peft_cfg = LoraConfig(r=int(lora["r"]), lora_alpha=int(lora["lora_alpha"]),
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                          target_modules=targets)
    tr = DPOTrainer(model=model, ref_model=None, args=cfg,
                    train_dataset=Dataset.from_list(pairs),
                    processing_class=tok, peft_config=peft_cfg)
    out = tr.train()
    pm = tr.model
    scaling = float(lora["lora_alpha"]) / float(lora["r"])
    dw = {}
    for name, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            A = mod.lora_A["default"].weight.detach().float()
            B = mod.lora_B["default"].weight.detach().float()
            key = name.replace("base_model.model.", "").replace(".base_layer", "")
            dw[key] = (B @ A * scaling).cpu()
    info = dict(global_step=int(tr.state.global_step),
                train_loss=float(getattr(out, "training_loss", float("nan"))),
                n_modules=len(dw),
                dw_fro=float(torch.sqrt(sum((t.double() ** 2).sum() for t in dw.values()))))
    # ---- leave the model as it was handed in ----
    if hasattr(pm, "unload"):
        pm.unload()
    del tr, pm
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.eval()
    tok.padding_side = "left"
    torch.cuda.empty_cache()
    still = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    if still:
        raise RuntimeError(f"{len(still)} LoRA modules still attached after unload")
    return dw, info


def run_unit(cfg, u, arm_names, panel):
    import torch
    import colab_t2t4 as C
    import v9_gate

    out_dir = os.path.join(C.RESULTS, panel)
    os.makedirs(os.path.join(out_dir, "probe_ll"), exist_ok=True)
    fam, hf, dn, ax, s = u["target"], u["hf"], u["designer"], u["axis"], u["seed"]
    key = dict(panel=panel, target=fam, axis=ax, seed=s, designer=dn)
    BATCH = 6                                   # run_dpo_baseline / run_frontier
    done = done_arms(out_dir)
    todo = [a for a in arm_names if (fam, ax, s, a) not in done]
    if not todo:
        print(f"[dpo] {fam} {ax} s{s}: nothing to do", flush=True)
        return 0

    pairs, n_total = build_pairs(C, dn, ax, s)
    mmlu = C.load_mmlu(n=200, seed=0)
    wt = C.load_wikitext(n_chunks=20, seed=0)
    t0 = time.time()
    model, tok = C.load_model(hf, dispatch=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        pre, pre_ll = eval_split(C, model, tok, ax, mmlu, wt, BATCH)      # CLEAN baseline
        n_e = pre.get("n_items", 200)
        print(f"[dpo] {fam} {ax} s{s} pairs={len(pairs)}/{n_total} "
              f"pre_skew={pre['skew']:+.4f} (load+baseline {time.time() - t0:.0f}s)",
              flush=True)
        for an in todo:
            arm = cfg["arms"][an]
            ta = time.time()
            if len(pairs) < int(arm["min_pairs"]):
                _append(os.path.join(out_dir, "removal.jsonl"), dict(
                    key, arm=an, skipped="too_few_pairs", n_informative=len(pairs),
                    n_total=n_total, min_pairs=int(arm["min_pairs"]),
                    pre_skew=pre["skew"]))
                print(f"[dpo] {fam} {ax} s{s} {an}: SKIP ({len(pairs)} < "
                      f"{arm['min_pairs']} differing pairs)", flush=True)
                continue
            dw, info = train_dpo(C, model, tok, pairs, s, arm, cfg["lora"],
                                 os.path.join(out_dir, "_trl_tmp"))
            # ---- the check the published script lacked ----
            chk, chk_ll = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
            restored = (chk["skew"] == pre["skew"] and chk["ppl"] == pre["ppl"]
                        and chk["mmlu_acc"] == pre["mmlu_acc"])
            if not restored:
                raise RuntimeError(
                    f"baseline not restored after unloading the DPO adapter: skew "
                    f"{pre['skew']!r} -> {chk['skew']!r}, ppl {pre['ppl']!r} -> "
                    f"{chk['ppl']!r}, mmlu {pre['mmlu_acc']} -> {chk['mmlu_acc']}")
            points, lls = [], []
            for m in [float(x) for x in arm["merge"]]:
                undo = C.apply_edit(model, dw, alpha=m, sign=+1.0)        # ADD the delta
                try:
                    post, ll = eval_split(C, model, tok, ax, mmlu, wt, BATCH)
                finally:
                    undo()
                ok = C.collateral_ok(pre, post)
                r = C.bias_reduction(pre, post)
                p = dict(merge=m, post_skew=post["skew"], removal=float(r),
                         mmlu_acc=post["mmlu_acc"], ppl=post["ppl"],
                         dmmlu_items=int(v9_gate.items_from_acc(pre["mmlu_acc"], n_e)
                                         - v9_gate.items_from_acc(post["mmlu_acc"], n_e)),
                         ppl_ratio=post["ppl"] / pre["ppl"], ok=bool(ok))
                points.append(p)
                lls.append(ll)
                _append(os.path.join(out_dir, "alpha_trace.jsonl"), dict(
                    key, arm=an, method="dpo", config=f"merge={m}", split="eval", alpha=m,
                    pre_mmlu=pre["mmlu_acc"], post_mmlu=post["mmlu_acc"], n_items=n_e,
                    pre_ppl=pre["ppl"], post_ppl=post["ppl"],
                    ppl_ratio=post["ppl"] / pre["ppl"], pre_skew=pre["skew"],
                    post_skew=post["skew"], collateral_ok=bool(ok),
                    bias_reduction=float(r)))
            np.savez_compressed(
                os.path.join(out_dir, "probe_ll", f"{fam}__{ax}__s{s}__{dn}__{an}.npz"),
                merge=np.array([p["merge"] for p in points]), ll=np.stack(lls),
                pre_ll=pre_ll)
            _append(os.path.join(out_dir, "removal.jsonl"), dict(
                key, arm=an, method="dpo", n_informative=len(pairs), n_total=n_total,
                recipe={k: arm.get(k) for k in ("lr", "steps", "epochs", "batch", "beta",
                                                "max_length", "min_pairs")},
                train=info, baseline_restored=bool(restored), pre_skew=pre["skew"],
                pre_mmlu=pre["mmlu_acc"], pre_ppl=pre["ppl"], n_items=n_e,
                points=points, seconds=round(time.time() - ta, 1)))
            # Operational line: budget status only; removals are read by the analysis.
            print(f"[dpo] {fam:6s} {ax:20s} s{s} {an:8s} steps={info['global_step']} "
                  f"in_budget={sum(p['ok'] for p in points)}/{len(points)} "
                  f"baseline_restored={restored} ({time.time() - ta:.0f}s)", flush=True)
            del dw
    finally:
        C._free(model)
    print(f"[dpo] unit {fam} {ax} s{s} finished in {time.time() - t0:.0f}s", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    units = unit_list(cfg)
    arms = a.arms.split(",") if a.arms else list(cfg["arms"])
    panel = a.panel or cfg["panel"]
    if a.count:
        print(len(units))
        return 0
    if a.progress:
        done = done_arms(os.path.join(os.environ.get("BS_OUT", "results"), panel))
        need = [(u["target"], u["axis"], u["seed"], an) for u in units for an in arms]
        print(f"{panel}: {sum(k in done for k in need)}/{len(need)} unit-arms on disk "
              f"(arms: {','.join(arms)})")
        return 0
    if a.unit is None:
        raise SystemExit("give --unit, --count or --progress")
    if not 0 <= a.unit < len(units):
        print(f"[dpo] unit {a.unit} out of range")
        return 0
    u = units[a.unit]
    print(f"[dpo] unit {a.unit}: {u['target']} {u['axis']} s{u['seed']} arms={arms} "
          f"panel={panel}", flush=True)
    try:
        return run_unit(cfg, u, arms, panel)
    except Exception:
        print(f"[dpo] unit {a.unit} FAILED", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
