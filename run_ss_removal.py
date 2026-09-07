"""StereoSet intrasentence axis on the frozen protocol (experiments_v7 §AX).

ss_intra passed the structural gates (1053 probe / 1053 edit items, disjoint).
Onboarding polarity/magnitude gates run first via run_ax_onboard-style checks
embedded here; removal only runs on target x axis cells that pass.
"""
import os, json
import colab_t2t4 as C

AXES = ["ss_intra"]
TARGETS = [("qwen", "Qwen/Qwen2.5-3B-Instruct"), ("phi", "microsoft/Phi-3.5-mini-instruct")]
DESIGNERS = {
    "qwen": [("self", "qwen_3b", "Qwen/Qwen2.5-3B-Instruct"),
             ("cross", "llama_3b", "meta-llama/Llama-3.2-3B-Instruct")],
    "phi": [("self", "phi_3b", "microsoft/Phi-3.5-mini-instruct"),
            ("cross", "qwen_3b", "Qwen/Qwen2.5-3B-Instruct")],
}
if __name__ == "__main__":
    mmlu = C.load_mmlu(n=50, seed=0); wt = C.load_wikitext(n_chunks=5, seed=0)
    gates = {}
    for fam, hf in TARGETS:
        m, t = C.load_model(hf, dispatch=False)
        try:
            for ax in AXES:
                sk = C.axis_skew(m, t, ax, 6)
                gates[f"{fam}|{ax}"] = dict(pre_skew=float(sk), g3=sk > 0,
                                            g4=abs(sk) >= 0.02)
                print(f"[ss-gate] {fam} {ax} pre_skew={sk:+.4f} "
                      f"G3={'PASS' if sk>0 else 'FAIL'} G4={'PASS' if abs(sk)>=0.02 else 'FAIL'}",
                      flush=True)
        finally:
            C._free(m)
    json.dump(gates, open(os.path.join(C.RESULTS, "ss_onboard.json"), "w"), indent=2)
    ok = [f for f, v in gates.items() if v["g3"] and v["g4"]]
    if len(ok) < 2:
        print(f"[ss] gates passed on {len(ok)}/2 targets — NOT running removal", flush=True)
        raise SystemExit(0)
    C.panel_run("ss", TARGETS, DESIGNERS, axes=AXES, seeds=[0, 1, 2],
                batch_size=6, train_bs=8)
    for null in ("sign_shuffle", "partition"):
        C.panel_run(f"ss_null_{null}", TARGETS, DESIGNERS, axes=AXES, seeds=[0, 1, 2],
                    batch_size=6, train_bs=8, null=null)
    print("[ss] ALL DONE", flush=True)
