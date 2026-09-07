"""Phases 2-4: the core 2x2 plus the endogenous/exogenous ablation.

design.md §5.1 (the 2x2) and §5.6 (the co-primary mechanism ablation).

Factors
-------
origin           inherited (occupation-gender, carried by the base model) vs
                 acquired  (female->negative valence, injected into T)
designer_family  self / sibling (same lineage) / cross (disjoint lineage)
signal           endogenous (elicited from D) vs exogenous (ground truth)

Everything downstream of elicitation is byte-identical across designer
conditions: the same target, the same LoRA hyper-parameters, the same
binarisation, the same evaluation. The ONLY thing that varies is the text the
designer produced. That isolation is what licenses attributing any
difference in bias removal to what D could see.

Edit strength `alpha` is swept rather than fixed: design.md §5.4 asks for a
bias-vs-collateral FRONTIER, never a single scalar, so every condition is
traced over the same alpha grid and summarised by the best bias reduction
that stays inside the collateral budget.

Resumable: each record is appended to results/runs.jsonl and completed
(condition, variant, alpha) keys are skipped on restart.
"""
import os, sys, json, time, argparse, itertools, gc
import numpy as np
import torch

import probes, evaluate as EV
from common import load, free, FAMILY, SIZE_B, MODELS
from edit import (train_task_vector, contrast, binarize, apply_edit,
                  merge_dw, vec_stats, cosine, sketch)
from elicit_gen import elicit_selfdebias, exogenous_corpora, elicit_random
from inject import load_injection

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "results"))
RUNS = os.path.join(RESULTS, "runs.jsonl")

AXIS = {"inherited": "occ_gender", "acquired": "gen_fm"}
# experiments_v2 Experiment B sweeps the INHERITED axis over many axes with
# differing direction-sharing (d_axis). --axis overrides which axis the
# "inherited" origin refers to; those axes are already exhibited by the base
# models, so no injection is involved.
AXIS_DEFAULT = {"inherited": "occ_gender", "acquired": "gen_fm"}

# design.md §5.1 mandates running designs BIDIRECTIONALLY. If cross-family
# designers only look better because they are stronger models -- or because
# one family simply ignores debias instructions -- the asymmetry will NOT
# flip when the roles are reversed. So each arm names a target and a designer
# set in which "same" and "cross" swap lineages, with designer capability
# matched across families (qwen3b 3.1B vs llama3b 3.2B).
ARMS = {
    "qwen": dict(target="qwen1.5b",
                 designers={"self": "qwen1.5b", "sibling": "qwen3b",
                            "cross": "llama3b", "cross2": "gemma2b",
                            # cross-family PANEL: rotate the cross designer
                            # through every other family, to separate the
                            # cross-family *relation* from designer identity.
                            "cross3": "phi3.5", "cross4": "smol1.7b",
                            "random": "random"}),
    "llama": dict(target="llama1b",
                  designers={"self": "llama1b", "sibling": "llama3b",
                             "cross": "qwen3b", "cross2": "gemma2b",
                             "cross3": "phi3.5", "cross4": "smol1.7b",
                             "random": "random"}),
    # experiments_v2 Experiment A: three further target families, so the
    # interaction can be forest-plotted per arm instead of pooled from one.
    # The cross-family designer is held CONSTANT (llama3b) across the three
    # new arms, which isolates the arm/target effect from designer identity.
    "gemma": dict(target="gemma2b",
                  designers={"self": "gemma2b", "sibling": "gemma9b",
                             "cross": "llama3b", "cross2": "qwen3b",
                             "cross3": "phi3.5", "cross4": "smol1.7b",
                             "random": "random"}),
    "phi":   dict(target="phi3.5",
                  designers={"self": "phi3.5", "sibling": "phi3mini",
                             "cross": "llama3b", "cross2": "qwen3b",
                             "cross3": "gemma2b", "cross4": "smol1.7b",
                             "random": "random"}),
    # experiments_v3 H: new families as targets (occ_gender only; cross held at
    # a strong panel designer, llama3b, matching the v2 new-arm convention).
    "olmo":   dict(target="olmo1b",
                   designers={"self": "olmo1b", "sibling": "olmo7b",
                              "cross": "llama3b", "cross2": "qwen3b",
                              "random": "random"}),
    "falcon": dict(target="falcon1b",
                   designers={"self": "falcon1b", "sibling": "falcon3b",
                              "cross": "llama3b", "cross2": "qwen3b",
                              "random": "random"}),
    "granite":dict(target="granite2b",
                   designers={"self": "granite2b", "sibling": "granite8b",
                              "cross": "llama3b", "cross2": "qwen3b",
                              "random": "random"}),
    # experiments_v4 P: T2 (7-9B) targets for H1 replication at scale
    "qwen_t2":   dict(target="qwen7b",   designers={"self":"qwen7b","sibling":"qwen3b","cross":"llama8b","random":"random"}),
    "llama_t2":  dict(target="llama8b",  designers={"self":"llama8b","sibling":"llama3b","cross":"qwen7b","random":"random"}),
    "gemma_t2":  dict(target="gemma9b",  designers={"self":"gemma9b","sibling":"gemma2b","cross":"llama8b","random":"random"}),
    "olmo_t2":   dict(target="olmo7b",   designers={"self":"olmo7b","sibling":"olmo1b","cross":"llama8b","random":"random"}),
    "granite_t2":dict(target="granite8b",designers={"self":"granite8b","sibling":"granite2b","cross":"llama8b","random":"random"}),
    "smol":  dict(target="smol1.7b",
                  designers={"self": "smol1.7b", "sibling": "smol360m",
                             "cross": "llama3b", "cross2": "qwen3b",
                             "cross3": "gemma2b", "cross4": "phi3.5",
                             "random": "random"}),
}
TARGET = ARMS["qwen"]["target"]
DESIGNERS = dict(ARMS["qwen"]["designers"])
FAMILY_REL = {"self": "same", "sibling": "same", "cross": "cross",
              "cross2": "cross", "cross3": "cross", "cross4": "cross",
              "random": "null"}


def load_done():
    done = set()
    if os.path.exists(RUNS):
        with open(RUNS) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["key"])
                except Exception:
                    pass
    return done


def append(rec):
    os.makedirs(RESULTS, exist_ok=True)
    with open(RUNS, "a") as f:
        f.write(json.dumps(rec) + "\n")


def build_target(origin, inject_tag):
    """Load T; for the acquired row, merge the injected-bias delta into it."""
    model, tok = load(TARGET)
    info = None
    if origin == "acquired":
        dw, info = load_injection(inject_tag)
        merge_dw(model, dw, alpha=1.0)
    return model, tok, info


def elicit_all(origin, seed, batch_size, n_rep):
    """Collect every designer's corpora once per (origin, seed)."""
    axis = AXIS[origin]
    out = {}
    for role, dname in DESIGNERS.items():
        t0 = time.time()
        if role == "random":          # null control: no designer involved
            r = elicit_random(axis, seed=seed)
            out[("endogenous", role)] = (r["biased"], r["debiased"], r["diag"])
            print(f"  [elicit] {origin}/random (null partition)", flush=True)
            continue
        dmodel, dtok = load(dname)
        try:
            r = elicit_selfdebias(dmodel, dtok, axis, seed=seed,
                                  batch_size=batch_size)
        finally:
            free(dmodel, dtok)
            dmodel = dtok = None      # caller must drop its own refs (see free)
            gc.collect(); torch.cuda.empty_cache()
        out[("endogenous", role)] = (r["biased"], r["debiased"], r["diag"])
        gap = r["diag"].get("contrast_gap")
        print(f"  [elicit] {origin}/{role}={dname} gap={gap:+.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    b, d = exogenous_corpora(axis, seed=seed)
    out[("exogenous", "gt")] = (b, d, {"contrast_gap": None, "mode": "exogenous"})
    return out


def variants(full, variant_set="mvp"):
    """Edit variants.

    variant_set="a" is the experiments_v2 Experiment A set: the main binary
    edit plus the SIGN-SHUFFLE null. §N makes both nulls mandatory in every
    condition (the data-partition null is the `random` designer role; the
    sign-shuffle null is this variant). Full precision is dropped there --
    experiments_v2 §1 records H1 as settled (~99% retention) and forbids
    re-litigating it.
    """
    if variant_set == "a":
        return [dict(kind="binary", granularity="per_tensor", sparsity=0.0),
                dict(kind="binary", granularity="per_tensor", sparsity=0.0,
                     random_signs=True)]
    v = [dict(kind="binary", granularity="per_tensor", sparsity=0.0),
         dict(kind="fp")]
    if full:
        v += [dict(kind="binary", granularity="per_tensor", sparsity=0.0,
                   random_signs=True),
              dict(kind="binary", granularity="per_channel", sparsity=0.0),
              dict(kind="binary", granularity="scalar", sparsity=0.0),
              dict(kind="binary", granularity="per_tensor", sparsity=0.5),
              dict(kind="binary", granularity="per_tensor", sparsity=0.9),
              dict(kind="binary", granularity="per_tensor", sparsity=0.99)]
    return v


def make_edit(v, spec, seed):
    if spec["kind"] == "fp":
        return v, dict(kind="fp", n_params=sum(t.numel() for t in v.values()))
    E, meta = binarize(v, granularity=spec.get("granularity", "per_tensor"),
                       sparsity=spec.get("sparsity", 0.0), seed=seed,
                       random_signs=spec.get("random_signs", False))
    meta["kind"] = "binary"
    meta["random_signs"] = spec.get("random_signs", False)
    return E, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--origins", nargs="*", default=["inherited", "acquired"])
    ap.add_argument("--designers", nargs="*", default=list(DESIGNERS))
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--inject-tag", default=None)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-bs", type=int, default=8,
                    help="LoRA training batch (lower for 7-9B targets)")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing (fits 7-14B targets)")
    ap.add_argument("--n-rep", type=int, default=24)
    ap.add_argument("--arm", default="qwen", choices=list(ARMS),
                    help="which bidirectional arm to run (design.md §5.1)")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="LoRA target modules; default q_proj v_proj. "
                         "'attn' (q,k,v,o) removes 96%% of CrowS bias inside "
                         "the collateral budget where q/v removes none.")
    ap.add_argument("--axis", default=None,
                    help="override the INHERITED axis (experiments_v2 B sweep)")
    ap.add_argument("--variant-set", default="mvp", choices=["mvp", "a"],
                    help="'a' = binary + sign-shuffle null (experiments_v2 A)")
    ap.add_argument("--full-variants", action="store_true")
    ap.add_argument("--tier", default="fast", choices=["fast", "full"])
    a = ap.parse_args()

    arm = ARMS[a.arm]
    if a.axis:
        AXIS["inherited"] = a.axis
        print(f"inherited axis overridden -> {a.axis}", flush=True)
    globals()["TARGET"] = arm["target"]
    globals()["DESIGNERS"] = {k: v for k, v in arm["designers"].items()
                              if k in a.designers}
    if a.inject_tag is None:
        a.inject_tag = f"{arm['target']}_gen_fm_s0"
    print(f"arm={a.arm} target={arm['target']} designers={globals()['DESIGNERS']}",
          flush=True)

    done = load_done()
    print(f"{len(done)} records already complete", flush=True)
    data = EV.get_eval_data(a.tier)

    for seed in a.seeds:
        for origin in a.origins:
            axis = AXIS[origin]
            print(f"\n=== seed {seed} | origin {origin} | axis {axis} ===",
                  flush=True)
            corpora = elicit_all(origin, seed, a.batch_size, a.n_rep)

            model, tok, inj = build_target(origin, a.inject_tag)
            try:
                # the valence probe must score the axis UNDER TEST; scoring a
                # fixed axis here would silently measure the wrong construct
                # for every Experiment B axis.
                probe_axis = axis if axis != "occ_gender" else AXIS["acquired"]
                pre = EV.fast_eval(model, tok, probe_axis, data,
                                   a.batch_size) if a.tier == "fast" else \
                      EV.full_eval(model, tok, probe_axis, data, a.batch_size)
                print(f"  [pre] {json.dumps({k: round(v,4) for k,v in pre.items()})}",
                      flush=True)

                for (signal, role), (bset, dset, diag) in corpora.items():
                    # NB: the arm MUST be part of the key. Without it the bidirectional
# arm produces keys identical to the first arm's and every one of
# its conditions is skipped as "already done", silently writing
# nothing.
                    # the axis must be in the key, or B's per-axis runs
                    # collide with the MVP records and are skipped as done
                    axtag = "" if axis == AXIS_DEFAULT[origin] else f"|{axis}"
                    ttag = "" if not a.targets else "|T" + "+".join(
                        t.replace("_proj", "") for t in a.targets)
                    ck = f"{a.arm}|{seed}|{origin}{axtag}{ttag}|{signal}|{role}"
                    t0 = time.time()
                    dw_b, lb = train_task_vector(model, tok, bset, rank=a.rank,
                                                 steps=a.steps, lr=a.lr, seed=seed,
                                                 targets=a.targets, bs=a.train_bs,
                                                 grad_checkpoint=a.grad_checkpoint)
                    dw_d, ld = train_task_vector(model, tok, dset, rank=a.rank,
                                                 steps=a.steps, lr=a.lr, seed=seed,
                                                 targets=a.targets, bs=a.train_bs,
                                                 grad_checkpoint=a.grad_checkpoint)
                    v = contrast(dw_b, dw_d)
                    vs = vec_stats(v)
                    vs["cos_b_d"] = cosine(dw_b, dw_d)
                    # persist a direction sketch so cross-condition alignment
                    # (esp. against the exogenous ground-truth direction) can
                    # be computed in analysis without re-deriving v
                    sk_dir = os.path.join(RESULTS, "sketches")
                    os.makedirs(sk_dir, exist_ok=True)
                    np.save(os.path.join(sk_dir,
                            f"{ck}".replace("/", "_") + ".npy"),
                            sketch(v))
                    print(f"  [tv] {ck} loss_b={lb:.3f} loss_d={ld:.3f} "
                          f"|v|={vs['l2']:.3f} cos(b,d)={vs['cos_b_d']:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    del dw_b, dw_d

                    for spec in variants(a.full_variants, a.variant_set):
                        E, meta = make_edit(v, spec, seed)
                        tag = (f"{spec['kind']}-{spec.get('granularity','')}"
                               f"-s{spec.get('sparsity',0)}"
                               f"{'-rand' if spec.get('random_signs') else ''}")
                        for alpha in a.alphas:
                            key = f"{ck}|{tag}|a{alpha}"
                            if key in done:
                                continue
                            undo = apply_edit(model, E, alpha=alpha, sign=-1.0)
                            try:
                                post = (EV.fast_eval if a.tier == "fast"
                                        else EV.full_eval)(
                                    model, tok, probe_axis, data, a.batch_size)
                            finally:
                                undo()
                            rec = dict(
                                key=key, seed=seed, origin=origin, axis=axis,
                                signal=signal, designer_role=role,
                                designer=DESIGNERS.get(role, "ground_truth"),
                                designer_family=FAMILY.get(DESIGNERS.get(role, ""), "n/a"),
                                family_relation=FAMILY_REL.get(role, "n/a"),
                                arm=a.arm, target=TARGET,
                                lora_targets=(a.targets or ["q_proj", "v_proj"]),
                                lora_rank=a.rank,
                                target_family=FAMILY[TARGET],
                                variant=tag, alpha=alpha,
                                edit_meta={k: val for k, val in meta.items()},
                                vec_stats=vs, elicit_diag=diag,
                                loss_b=lb, loss_d=ld,
                                pre=pre, post=post,
                                bias_reduction=EV.bias_reduction(pre, post, axis),
                                collateral_ok=EV.collateral_ok(pre, post))
                            append(rec)
                            done.add(key)
                            print(f"    {key:52s} dBias={rec['bias_reduction']:+.4f} "
                                  f"mmlu={post['mmlu_acc']:.3f} ppl={post['ppl']:.2f}",
                                  flush=True)
                        del E
                    del v
            finally:
                free(model, tok)
    print("\nDONE ->", RUNS)


if __name__ == "__main__":
    main()
