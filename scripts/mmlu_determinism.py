"""v12astar replay miss: is MMLU at a fixed edit deterministic within a process?

llama|occ_gender|s0, C-a@0.01: v12astar and v12dec1k agree bit-for-bit on skew
and perplexity at every alpha but differ on MMLU at alpha 8 and 16 (0.670 vs
0.655; 0.660 vs 0.645). This rebuilds that exact edit (run_opsel's recipe,
geometry fingerprint checked against v12astar) and reads MMLU repeatedly:
back to back, after calibration-split evaluations (what v12astar ran between
arms), and after an MMLU-1000 read (what v12dec1k ran before training).
Writes results/v12/mmlu_determinism.json.
"""
import json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "src"))
import colab_t2t4 as C
import run_opsel as R

T, AX, S, DN, HF = "llama", "occ_gender", 0, "llama_3b", "meta-llama/Llama-3.2-3B-Instruct"
OUT = os.path.join(REPO, "results", "v12", "mmlu_determinism.json")

mmlu = C.load_mmlu(n=200, seed=0)
wt = C.load_wikitext(n_chunks=20, seed=0)
mcal = R.load_mmlu_calib(200, seed=0)
wcal = C.load_wikitext(n_chunks=20, seed=0, split="validation")
m1k = C.load_mmlu(n=1000, seed=0)
model, tok = C.load_model(HF, dispatch=False)


def acc():
    return C.eval_mmlu(model, tok, mmlu, batch_size=6, n_rot=2)["mmlu_acc"]


log = dict(pre=[acc(), acc()])
v, _ = R.train_contrast(model, tok, DN, AX, S)
sha = R.geometry_of(v)["geom_sha"]
ref = [json.loads(l) for l in open(os.path.join(REPO, "results/v12astar/geometry.jsonl"))]
ref = [r["geometry"]["geom_sha"] for r in ref if (r["target"], r["axis"], r["seed"]) == (T, AX, S)]
log["geom_sha"], log["geom_matches_v12astar"] = sha, sha in ref
log["pre_after_train"] = acc()

E, _ = R.build(v, "C-a@0.01", S)
for a in (8.0, 16.0):
    undo = C.apply_edit(model, E, alpha=a, sign=-1.0)
    try:
        seq = [acc(), acc(), acc()]
        R.collateral_eval(model, tok, mcal, wcal)          # as between arms in v12astar
        seq += [acc()]
        C.eval_mmlu(model, tok, m1k, batch_size=6, n_rot=2)  # as v12dec1k's pre_1k
        seq += [acc()]
        seq += [C.fast_eval(model, tok, AX, mmlu, wt, 6)["mmlu_acc"]]  # the runner's call
    finally:
        undo()
    log[f"alpha_{a:g}"] = seq
    print(a, seq, flush=True)
log["published"] = {"v12dec1k": {"8": 0.67, "16": 0.66}, "v12astar": {"8": 0.655, "16": 0.645}}
json.dump(log, open(OUT, "w"), indent=1)
print(json.dumps(log, indent=1))
