"""Elicit corpora missing for the v8 INS arms, using the frozen elicitation path.

INS-B needs gemma_3b x {occ_gender, bbq_Age}. occ_gender exists; bbq_Age does
not -- only llama_3b, phi_3b and qwen_3b were ever elicited on the BBQ axes.
This fills the gap with the SAME code panel_run uses (elicit_selfdebias +
_t2x_corpus_fp), writing into the shared corpora cache with the pipe naming, so
the file is indistinguishable from one produced by a panel run.

Per the v6 rule, a cache miss on a declared-cached corpus is never a silent
re-elicitation: this script only writes files that do not exist, prints every
one it creates, and never overwrites.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import colab_t2t4 as C

NEEDED = [("gemma_3b", "google/gemma-2-2b-it", ["bbq_Age"], [0, 1, 2])]


def main():
    for dname, dhf, axes, seeds in NEEDED:
        need = [(ax, s) for ax in axes for s in seeds
                if not os.path.exists(C._t2x_corpus_fp(dname, ax, s))]
        if not need:
            print(f"[elicit] {dname}: nothing missing", flush=True)
            continue
        print(f"[elicit] load {dname} ({dhf}) for {need}", flush=True)
        m, t = C.load_model(dhf, dispatch=False)
        try:
            for ax, s in need:
                fp = C._t2x_corpus_fp(dname, ax, s)
                if os.path.exists(fp):          # never overwrite
                    continue
                el = C.elicit_selfdebias(m, t, ax, s, 16)
                json.dump(dict(designer=dname, axis=ax, seed=s,
                               biased=el["biased"], debiased=el["debiased"],
                               diag=el["diag"]), open(fp, "w"))
                print(f"[elicit] {dname:12s} {ax:12s} s{s} "
                      f"gap={el['diag']['contrast_gap']:+.3f} "
                      f"n={el['diag']['n_items']} -> {os.path.basename(fp)}", flush=True)
        finally:
            C._free(m)
    print("[elicit] DONE", flush=True)


if __name__ == "__main__":
    main()
