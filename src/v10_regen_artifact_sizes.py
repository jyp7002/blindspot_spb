"""v10 — Table B's artifact-size row, computed for EVERY method on one basis.

The manuscript's Table B contrasts the edit's "0.4-1.6 MB" against "vector +
hook" and "projection + hook". Those are not comparable descriptions: one is a
number, two are nouns. This module puts all five methods on the same axis --
bytes that must ship -- so the deployment claim can be checked rather than
asserted.

The comparison is only honest if the edit is priced the same way as the others,
which means including what it costs to say WHICH coordinates a 1%-sparse patch
touches (see src/v10_regen_size.py). A steering vector is dense in its own
space and needs no index; a 1% sign field does.

Model geometry is read from the cached HF configs, and the derived q/k/v/o
parameter count is CHECKED against the n_params measured in the panels -- if
the shapes were wrong, that check fails rather than silently mis-sizing every
baseline.

Run: python3 src/v10_regen_artifact_sizes.py
"""
import os, sys, glob, json, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# Manuscript row label -> the HF repo whose geometry it has.
REPO = {
    "qwen-3B":    "Qwen/Qwen2.5-3B-Instruct",
    "gemma-2.6B": "google/gemma-2-2b-it",
    "llama-3.2B": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3.8B":   "microsoft/Phi-3.5-mini-instruct",
    "qwen-7B":    "Qwen/Qwen2.5-7B-Instruct",
    "llama-8B":   "meta-llama/Llama-3.1-8B-Instruct",
}
FUSED_QKV = {"phi-3.8B"}

FP32, FP16 = 4, 2
LORA_R = 16          # run_dpo_baseline.py LoraConfig(r=16)
SD_RANKS = (1, 2)    # run_sentdebias_baseline.RANKS
INLP_ITERS = (1, 2)  # observed n_iter in results_v9/v6trace/inlp


def load_config(repo):
    d = os.path.join(HF_CACHE, "models--" + repo.replace("/", "--"), "snapshots")
    for fp in sorted(glob.glob(os.path.join(d, "*", "config.json"))):
        with open(fp) as fh:
            return json.load(fh)
    return None


def attn_shapes(cfg):
    """Exact (out, in) shapes of the edited attention projections."""
    h = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    nh = cfg.get("num_attention_heads")
    nkv = cfg.get("num_key_value_heads", nh)
    hd = cfg.get("head_dim") or (h // nh)
    q, kv = nh * hd, nkv * hd
    if cfg.get("model_type") == "phi3":                 # fused qkv, no o_proj edited
        return h, L, [("qkv_proj", q + 2 * kv, h)], True
    return h, L, [("q_proj", q, h), ("k_proj", kv, h),
                  ("v_proj", kv, h), ("o_proj", h, q)], False


def main():
    C.ensure_dirs()
    size = C.jload("results/v10/patch_sizes.json") or {}
    per_target = size.get("per_target") or {}

    rows, checks = [], []
    for label, repo in REPO.items():
        cfg = load_config(repo)
        if cfg is None:
            checks.append({"label": label, "error": f"config not cached for {repo}"})
            continue
        h, L, projs, fused = attn_shapes(cfg)
        n_tensors = L * len(projs)
        n_params_derived = L * sum(o * i for _, o, i in projs)

        meas = (per_target.get(label) or {}).get("edited_n_params")
        ok = meas is not None and int(meas) == int(n_params_derived)
        checks.append({"label": label, "hidden_size": h, "n_layers": L,
                       "n_tensors": n_tensors,
                       "n_params_derived": n_params_derived,
                       "n_params_measured": meas, "shapes_match": ok})

        rec = per_target.get(label) or {}
        s99 = rec.get("s0.99") or {}
        idx = s99.get("index_cost") or {}
        payload = s99.get("bytes_payload_only")

        # --- every method's shippable artifact, in bytes ---------------------
        methods = {}
        if payload is not None:
            methods["edit (payload only, as cited)"] = payload
            if idx:
                methods["edit + support index (entropy floor)"] = payload + idx["entropy_bound_bytes"]
                methods["edit + support index (32-bit)"] = payload + idx["explicit_index_bytes"]
        dense = (rec.get("dense") or {}).get("bytes")
        if dense is not None:
            methods["edit, dense (1 bit/weight, no index needed)"] = dense

        # steering: one residual-stream vector at ONE layer, plus 2 scalars
        methods["steering (1 vector @ 1 layer)"] = h * FP32 + 2 * FP32
        # SentenceDebias: a (k, hidden) projection basis
        for k in SD_RANKS:
            methods[f"SentenceDebias (rank {k} projection)"] = k * h * FP32
        # INLP: n_iter stacked projection directions
        for n in INLP_ITERS:
            methods[f"INLP ({n} iterations)"] = n * h * FP32
        # DPO: a LoRA r=16 on the SAME modules, materialised as A and B
        lora_params = L * sum(LORA_R * (o + i) for _, o, i in projs)
        methods["DPO (LoRA r=16 adapter, fp16)"] = lora_params * FP16

        rows.append({
            "target": label, "repo": repo, "hidden_size": h, "n_layers": L,
            "n_tensors": n_tensors, "fused_qkv": fused,
            "edited_n_params": n_params_derived,
            "lora_params": lora_params,
            "artifact_bytes": methods,
            "artifact_mib": {k: C.mib(v) for k, v in methods.items()},
        })

    # ratios against the two inference-time baselines
    ratios = []
    for r in rows:
        m = r["artifact_mib"]
        steer = m.get("steering (1 vector @ 1 layer)")
        sd = m.get("SentenceDebias (rank 1 projection)")
        for edit_key in ("edit (payload only, as cited)",
                         "edit + support index (entropy floor)",
                         "edit + support index (32-bit)"):
            if edit_key in m and steer:
                ratios.append({"target": r["target"], "edit_variant": edit_key,
                               "edit_mib": m[edit_key],
                               "steering_mib": steer, "sentencedebias_mib": sd,
                               "edit_over_steering": m[edit_key] / steer,
                               "edit_over_sentencedebias": (m[edit_key] / sd) if sd else None})

    report = {
        "artifact": "v10_regen_artifact_sizes",
        "module": "src/v10_regen_artifact_sizes.py",
        "purpose": ("Table B artifact-size row, all methods on one basis: bytes "
                    "that must ship"),
        "geometry_check": checks,
        "all_shapes_match_measured_n_params": all(c.get("shapes_match")
                                                  for c in checks if "error" not in c),
        "per_target": rows,
        "ratios": ratios,
        "assumptions": {
            "steering": ("one residual-stream vector at one layer (the frontier's "
                         "selected config is a single (layer, strength)), fp32, "
                         "plus strength and activation-norm scalars"),
            "sentencedebias": "PCA basis of shape (k, hidden), fp32, k in {1,2}",
            "inlp": "n_iter stacked directions of shape (hidden,), fp32",
            "dpo": ("LoRA r=16 on the same attention modules "
                    "(run_dpo_baseline.py), A and B materialised, fp16"),
            "edit": ("from results/v10/patch_sizes.json; payload = retained/8 + "
                     "2 bytes per tensor, index priced separately"),
            "excluded_from_all": ("tokenizer, config, and the serving hook's code; "
                                  "these are comparable across methods and small"),
        },
        "caveat": ("steering/SentenceDebias/INLP artifacts are small but require a "
                   "serving-time hook and per-token compute; the edit merges and "
                   "needs neither. Size is one axis of Table B, not the argument."),
    }
    fp = C.jdump(report, "results/v10/artifact_sizes.json")

    print(f"[v10 artifact sizes] {len(rows)} targets; "
          f"shapes reproduce measured n_params: "
          f"{report['all_shapes_match_measured_n_params']}")
    for c in checks:
        if "error" in c:
            print(f"  ! {c['label']}: {c['error']}")
        elif not c["shapes_match"]:
            print(f"  ! {c['label']}: derived {c['n_params_derived']} != "
                  f"measured {c['n_params_measured']}")
    print()
    order = ["edit (payload only, as cited)", "edit + support index (entropy floor)",
             "edit + support index (32-bit)", "edit, dense (1 bit/weight, no index needed)",
             "DPO (LoRA r=16 adapter, fp16)", "SentenceDebias (rank 1 projection)",
             "SentenceDebias (rank 2 projection)", "INLP (2 iterations)",
             "steering (1 vector @ 1 layer)"]
    w = max(len(o) for o in order)
    print(f"  {'artifact':{w}s} " + "".join(f"{r['target']:>12s}" for r in rows))
    for o in order:
        line = f"  {o:{w}s} "
        for r in rows:
            v = r["artifact_mib"].get(o)
            line += f"{v:12.4f}" if v is not None else f"{'-':>12s}"
        print(line)
    print()
    for ek in ("edit (payload only, as cited)", "edit + support index (entropy floor)",
               "edit + support index (32-bit)"):
        rs = [x for x in ratios if x["edit_variant"] == ek]
        if rs:
            lo = min(x["edit_over_steering"] for x in rs)
            hi = max(x["edit_over_steering"] for x in rs)
            print(f"  {ek:44s} is {lo:8.0f}x - {hi:8.0f}x the steering vector")
    print(f"wrote {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
