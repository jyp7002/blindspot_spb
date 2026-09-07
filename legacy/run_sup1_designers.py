"""SUP1 §5 — designer-overlap supports at the 7-9B tier.

WHY A SEPARATE RUNNER. The DEC panel is self-designer only (it decomposes an
edit, and the designer is not one of the factors being decomposed), so it cannot
answer SUP1 §5: "self vs same-large vs cross supports on the same target".

That question needs only the SUPPORTS, not removal -- overlap is a property of
which coordinates survive, and no alpha sweep or collateral evaluation enters it.
So this dumps supports and stops: two task-vector trainings per (designer, seed)
and the C-ref construction, with no fast_eval at all. That is roughly a sixth of
a full DEC cell, and it is why the arm is affordable at 7-9B where v8 wants it.

The self arm is NOT re-run: run_dec.py already dumps
qwen7b|occ_gender|s*|qwen.npz, and this writes into the same directory under the
same naming, so src/v8_sup1.py picks all three designers up as one comparison
group. Existing dumps are never overwritten.

Registered two-sided (PREREGISTRATION.md v8 §SUP1): high overlap = a shared
substructure; low overlap TOGETHER WITH the already-measured behavioural tie =
a degenerate solution space, many sparse solutions with the same effect. Both are
strong; neither is the hoped-for answer.
"""
import os, sys, json, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C
import v8_edits as V8
from v8_support import dump_support, support_path

TARGET_LABEL = "qwen7b"
TARGET_HF = "Qwen/Qwen2.5-7B-Instruct"
AXIS = "occ_gender"
SEEDS = [0, 1, 2]
DENSITY = 0.01
SCALE_MODE = "ref_tensor"
PANEL = "v8dec"          # same directory as DEC, so SUP1 groups them together

# self (qwen) is dumped by run_dec.py; only the other two designer classes here.
DESIGNERS = [("cross", "llama"), ("same_large", "qwen_large")]


def main():
    todo = [(role, dn, s) for role, dn in DESIGNERS for s in SEEDS
            if not os.path.exists(support_path(C.RESULTS, PANEL, TARGET_LABEL,
                                               AXIS, s, dn))]
    if not todo:
        print("[sup1d] nothing to do", flush=True)
        return
    missing = [(dn, s) for _r, dn, s in todo
               if not os.path.exists(C._t2x_corpus_fp(dn, AXIS, s))]
    if missing:
        raise SystemExit(f"[sup1d] REFUSING: corpora absent for {missing}. "
                         "A declared-cached corpus must never be silently "
                         "re-elicited (v6 rule).")

    model, tok = C.load_model(TARGET_HF, dispatch=False)
    try:
        for role, dn, s in todo:
            try:
                corp = json.load(open(C._t2x_corpus_fp(dn, AXIS, s)))
                dwb, _ = C.train_task_vector(model, tok, corp["biased"], rank=16,
                                             steps=250, lr=1e-4, seed=s, bs=8,
                                             targets=C.ATTN, grad_checkpoint=True)
                dwd, _ = C.train_task_vector(model, tok, corp["debiased"], rank=16,
                                             steps=250, lr=1e-4, seed=s, bs=8,
                                             targets=C.ATTN, grad_checkpoint=True)
                v = C.contrast_(dwb, dwd)
                del dwb, dwd
                ref_scales = V8.ref_scale_table(v, DENSITY, s)
                E, _meta = V8.build_edit(v, "C-ref", DENSITY, s,
                                         scale_mode=SCALE_MODE, ref_scales=ref_scales)
                sp = support_path(C.RESULTS, PANEL, TARGET_LABEL, AXIS, s, dn)
                md = dump_support(sp, E, meta=dict(
                    target=TARGET_LABEL, axis=AXIS, seed=s, designer=dn, role=role,
                    condition="C-ref", scale_mode=SCALE_MODE,
                    note="SUP1 designer-overlap arm; supports only, no alpha sweep"))
                print(f"[sup1d] {TARGET_LABEL} {AXIS} s{s} {role:11s}<-{dn:12s} "
                      f"nnz={md['n_support']:,} density={md['density']:.5f}", flush=True)
                del v, E
            except Exception:
                print(f"[sup1d] FAIL {role}<-{dn} s{s}", flush=True)
                traceback.print_exc()
    finally:
        C._free(model)
    print("[sup1d] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
