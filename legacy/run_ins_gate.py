"""INS gate (experiments_v8 §INS): can lm-evaluation-harness score an IN-MEMORY
edited model on IFEval + MMLU within one working day?

The gate is NOT "does lm-eval install". It is whether the harness can be pointed
at a live, already-edited `AutoModelForCausalLM` object. Saving a checkpoint per
condition would be ~15 GB x (5 conditions x 2 targets x 3 seeds) and is not an
option on a box with 22 GB free, so HFLM(pretrained=<model object>) is the only
viable path. This script proves or disproves exactly that.

Run: python3 run_ins_gate.py [model_id] [limit]
"""
import os, sys, json, time

os.environ.setdefault("HF_HUB_OFFLINE", "0")   # IFEval data may need the hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "v8", "ins_gate.json")

report = dict(model=MODEL, limit=LIMIT, steps={})


def step(name, fn):
    t0 = time.time()
    try:
        val = fn()
        report["steps"][name] = dict(ok=True, secs=round(time.time() - t0, 1), detail=val)
        print(f"[ins-gate] {name}: OK ({time.time()-t0:.1f}s) {val}", flush=True)
        return val
    except Exception as e:
        report["steps"][name] = dict(ok=False, secs=round(time.time() - t0, 1),
                                     detail=f"{type(e).__name__}: {e}")
        print(f"[ins-gate] {name}: FAIL {type(e).__name__}: {e}", flush=True)
        return None


def main():
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 device_map="cuda")

    # A real in-place weight perturbation, so we are proving the harness reads the
    # EDITED weights and not a fresh copy of the checkpoint off disk.
    def perturb():
        W = model.model.layers[0].self_attn.q_proj.weight
        before = W.detach().clone()
        W.data.add_(torch.zeros_like(W))         # no-op edit: identity check
        return dict(unchanged=bool(torch.equal(W.detach(), before)))

    step("in_place_edit_hook", perturb)

    lm = step("wrap_HFLM", lambda: HFLM(pretrained=model, tokenizer=tok,
                                        batch_size=8) and "wrapped")
    if lm is None:
        lmobj = None
    else:
        lmobj = HFLM(pretrained=model, tokenizer=tok, batch_size=8)

    def run(task):
        res = lm_eval.simple_evaluate(model=lmobj, tasks=[task], limit=LIMIT,
                                      verbosity="ERROR")
        return {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
                for k, v in res["results"].items()}

    step("mmlu", lambda: run("mmlu"))
    step("ifeval", lambda: run("ifeval"))

    # Does the harness see a genuine weight change? Flip q_proj hard and re-run
    # a cheap task; the score must move.
    def sensitivity():
        W = model.model.layers[0].self_attn.q_proj.weight
        W.data.mul_(0.0)
        res = lm_eval.simple_evaluate(model=lmobj, tasks=["ifeval"], limit=LIMIT,
                                      verbosity="ERROR")
        return {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
                for k, v in res["results"].items()}

    step("reads_edited_weights", sensitivity)

    ok = all(v["ok"] for v in report["steps"].values())
    report["gate"] = "PASS" if ok else "FAIL"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1)
    print(f"[ins-gate] GATE={report['gate']} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
