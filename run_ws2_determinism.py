"""WS2.1/2.2 — isolate the cause of MMLU non-determinism.

The evidence: two `llama|occ_gender|s2` rows shared between v8spc/small and
v6trace/ablate have BIT-IDENTICAL ppl_ratio and bias_reduction but DIFFERENT
dmmlu. Perplexity and the bias probe reproduce exactly; only MMLU moves. Since
the collateral gate is a hard threshold and 833 probes sit within +-1 item of it,
a 1-item wobble is not cosmetic.

This isolates the cause on the UNEDITED model — no edit, no training — because if
plain `eval_mmlu` is already non-reproducible then nothing downstream can be.

Conditions, each scoring the SAME 200 items:
  A  repeat in-process, identical batch size      -> intra-process determinism
  B  different batch sizes {4, 6, 8, 16}          -> batch-composition sensitivity
  C  TF32 on vs off                               -> reduction-precision sensitivity
  D  n_rot 2 vs 4                                 -> rotation averaging (sanity)

Run: python3 run_ws2_determinism.py [model_id]
"""
import os, sys, json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-3B-Instruct"
OUT = os.path.join(C.RESULTS, "v9", "ws2_determinism.json")


def acc(model, tok, rows, bs, n_rot=2):
    return C.eval_mmlu(model, tok, rows, batch_size=bs, n_rot=n_rot)["mmlu_acc"]


def main():
    rep = {"model": MODEL, "n_items": 200, "conditions": {}}
    rows = C.load_mmlu(n=200, seed=0)
    model, tok = C.load_model(MODEL, dispatch=False)
    try:
        # A — repeat, identical settings
        a = [acc(model, tok, rows, 6) for _ in range(3)]
        rep["conditions"]["A_repeat_bs6"] = dict(
            values=a, items=[round(x * 200) for x in a],
            spread_items=round(max(a) * 200) - round(min(a) * 200))
        print(f"[ws2] A repeat bs=6 x3      : {a}  items={[round(x*200) for x in a]}",
              flush=True)

        # B — batch size
        b = {}
        for bs in (4, 6, 8, 16):
            b[bs] = acc(model, tok, rows, bs)
            print(f"[ws2] B batch_size={bs:<3d}       : {b[bs]:.4f}  "
                  f"items={round(b[bs]*200)}", flush=True)
        items = [round(v * 200) for v in b.values()]
        rep["conditions"]["B_batch_size"] = dict(
            values={str(k): v for k, v in b.items()}, items=items,
            spread_items=max(items) - min(items))

        # C — TF32
        c = {}
        for tf32 in (True, False):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            c[tf32] = acc(model, tok, rows, 6)
            print(f"[ws2] C tf32={str(tf32):<5s}          : {c[tf32]:.4f}  "
                  f"items={round(c[tf32]*200)}", flush=True)
        ci = [round(v * 200) for v in c.values()]
        rep["conditions"]["C_tf32"] = dict(values={str(k): v for k, v in c.items()},
                                           items=ci, spread_items=max(ci) - min(ci))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # D — rotations
        d = {}
        for nr in (2, 4):
            d[nr] = acc(model, tok, rows, 6, n_rot=nr)
            print(f"[ws2] D n_rot={nr}             : {d[nr]:.4f}  "
                  f"items={round(d[nr]*200)}", flush=True)
        di = [round(v * 200) for v in d.values()]
        rep["conditions"]["D_n_rot"] = dict(values={str(k): v for k, v in d.items()},
                                            items=di, spread_items=max(di) - min(di))
    finally:
        C._free(model)

    culprits = [k for k, v in rep["conditions"].items() if v["spread_items"] > 0]
    rep["nondeterministic_under"] = culprits
    rep["verdict"] = ("no variation observed" if not culprits
                      else "varies under: " + ", ".join(culprits))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)
    print(f"\n[ws2] VERDICT: {rep['verdict']}")
    print(f"[ws2] wrote {OUT}")


if __name__ == "__main__":
    main()
