"""Phase 0 -- bias inventory across all candidate models.

design.md §3 requires bias *origin* to be established empirically, not
assumed:

  inherited = present in the base pretrained model and shared across the
              family at comparable magnitude, verified on >=3 siblings and
              present in the designer D itself (this is D's blind spot).
  acquired  = injected into T by fine-tuning, verified ABSENT in base and D.

So before any edit can be built we must measure a battery of bias axes on
every model and find (a) axes that are strong and consistent *within* the
Qwen2.5 sibling set but weak in the cross-family designers -- those are the
inherited/blind-spot candidates -- and (b) axes that are near-neutral
everywhere, which are the candidates for acquired-bias injection.

Resumable: results are appended to results/inventory.json per (model, axis).
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from common import load, free, MODELS, FAMILY, SIZE_B, save_json, load_json
import bias_eval as BE

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "inventory.json")
OUT = os.path.abspath(OUT)

BBQ_CATS = ["Gender_identity", "Race_ethnicity", "Religion", "Age",
            "Nationality", "Physical_appearance", "SES", "Sexual_orientation",
            "Disability_status"]
SS_TYPES = ["gender", "race", "religion", "profession"]
CROWS_TYPES = ["gender", "race-color", "religion", "age", "nationality",
               "sexual-orientation", "physical-appearance", "disability",
               "socioeconomic"]


def read_state():
    if os.path.exists(OUT):
        return load_json(OUT)
    return {}


def write_state(st):
    save_json(st, OUT)


def run_model(name, st, n_bbq, n_ss, n_crows, batch_size):
    key = name
    st.setdefault(key, {"model": MODELS[name], "family": FAMILY[name],
                        "size_b": SIZE_B[name], "bbq": {}, "stereoset": {},
                        "crows": {}})
    rec = st[key]
    todo = ([("bbq", c) for c in BBQ_CATS if c not in rec["bbq"]]
            + [("stereoset", t) for t in SS_TYPES if t not in rec["stereoset"]]
            + [("crows", t) for t in CROWS_TYPES if t not in rec["crows"]])
    if not todo:
        print(f"[{name}] already complete", flush=True)
        return
    print(f"[{name}] loading ({len(todo)} axes to run)", flush=True)
    model, tok = load(name)
    try:
        for kind, axis in todo:
            t0 = time.time()
            if kind == "bbq":
                rows = BE.load_bbq(axis, n=n_bbq)
                sc = BE.eval_bbq(model, tok, rows, batch_size)
                sc.pop("_preds", None)
            elif kind == "stereoset":
                rows = BE.load_stereoset("intersentence", n=n_ss, bias_type=axis)
                rows += BE.load_stereoset("intrasentence", n=n_ss, bias_type=axis)
                sc = BE.eval_stereoset(model, tok, rows, batch_size)
            else:
                rows = BE.load_crows(n=n_crows, bias_type=axis)
                sc = BE.eval_crows(model, tok, rows, batch_size) if rows else {}
            rec[kind][axis] = sc
            write_state(st)
            print(f"[{name}] {kind}/{axis} {json.dumps({k: round(v,4) for k,v in sc.items() if isinstance(v,(int,float))})} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        free(model, tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--n-bbq", type=int, default=400)
    ap.add_argument("--n-ss", type=int, default=250)
    ap.add_argument("--n-crows", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    st = read_state()
    for m in a.models:
        try:
            run_model(m, st, a.n_bbq, a.n_ss, a.n_crows, a.batch_size)
        except Exception as e:
            print(f"[{m}] FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
    write_state(st)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
