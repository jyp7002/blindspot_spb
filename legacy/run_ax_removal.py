"""AX removal run — only on axes that passed the onboarding gates.

Axis list comes from AX_AXES (set by run_ax_queue.sh from ax_onboard.json), never
hardcoded, so an axis that fails G1-G4 cannot silently acquire a removal number.

Frozen protocol, self + cross designer, 3 seeds, both nulls (sign_shuffle and
partition, via panel_run's null= parameter). pre_skew is persisted per row so the
AX2 breadth table can report it beside removal (§AX2: the honest frame is a
relationship, not an excuse).
"""
import os
import colab_t2t4 as C

AXES = [a for a in os.environ.get("AX_AXES", "").split() if a]
assert AXES, "AX_AXES not set — refusing to guess the axis list"
TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct"), ("phi", "microsoft/Phi-3.5-mini-instruct")]
DESIGNERS = {
    "qwen": [("self", "qwen_3b", "Qwen/Qwen2.5-3B-Instruct"),
             ("cross", "llama_3b", "meta-llama/Llama-3.2-3B-Instruct")],
    "phi": [("self", "phi_3b", "microsoft/Phi-3.5-mini-instruct"),
            ("cross", "qwen_3b", "Qwen/Qwen2.5-3B-Instruct")],
}
if __name__ == "__main__":
    print(f"[ax] axes={AXES} targets={[t for t,_ in TARGETS]}", flush=True)
    C.panel_run("ax", TARGETS, DESIGNERS, axes=AXES, seeds=[0, 1, 2],
                batch_size=6, train_bs=8)
    for null in ("sign_shuffle", "partition"):
        print(f"[ax] null={null}", flush=True)
        C.panel_run(f"ax_null_{null}", TARGETS, DESIGNERS, axes=AXES, seeds=[0, 1, 2],
                    batch_size=6, train_bs=8, null=null)
    print("[ax] ALL DONE", flush=True)
