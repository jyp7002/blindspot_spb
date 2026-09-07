"""Adapt results/runs.jsonl into the v6 alpha_trace schema — MV-A's F1 (<=3.8B) half.

experiments_v6 MV-A is billed as "free: reanalysis of existing alpha-sweeps". That
is true only for the SMALL tier: results/runs.jsonl (the v2/v3-era runner) recorded
every alpha with its full collateral (pre/post mmlu_acc + ppl), while the v5 panel
runner kept only `best` and discarded the sweep. So:

    F1 (<=3.8B)  -> this adapter, no GPU, available now
    F2 (7-9B)    -> requires the alpha_trace patch + a re-run (results/v6trace)

Emitting the same schema means ONE analysis module (src/v6_mva_analyze.py) computes
both tiers, so the two halves of the three-way filter cannot drift apart.

Protocol subset (frozen; anything else is a different experiment):
    variant == 'binary-per_tensor-s0.0'   attn r16 per-tensor binary, no sparsity
    signal  == 'endogenous'               exogenous/gt is the calibration arm only
    axis    in {occ_gender, crows_socioeconomic}
Verified on this file: the stored `collateral_ok` agrees with a 0.02/1.10 recompute
on 1200/1200 rows, i.e. the same frozen budget the v5/v6 runner uses.

Role mapping: cross2/cross3/cross4 -> 'cross' (they are additional cross-family
designers, not a distinct condition). 'random' (sign-shuffle null) and 'gt'
(exogenous) are DROPPED from the penalty panel and reported separately.

CAVEAT recorded in the output: for occ_gender, bias_reduction reproduces
|pre.occ_skew| - |post.occ_skew| exactly (1008/1008). For crows_socioeconomic the
runner did not persist the crows skew in pre/post, so bias_reduction is taken as
recorded and cannot be re-derived. dmmlu and ppl_ratio are re-derived for both.

Usage:  python src/v6_runs_adapter.py [--out results/v6trace_3b] [--panel small]
"""
import os, json, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(REPO, "results")

VARIANT = "binary-per_tensor-s0.0"
# experiments_v6 §1 freezes the edit as attn (q,k,v,o) r16. runs.jsonl ALSO contains
# a v2-era layer-placement ablation under the same `variant` string, distinguishable
# only by the module-set tag inside `key` (Tq+k+v+o present vs absent; 154M vs 77M
# sign flips). Without this filter the two collide on
# (target, axis, seed, role, designer, alpha) -- 456 of 744 tuples duplicated, with
# wildly different bias_reduction -- and the panel silently mixes two protocols.
MODULE_TAG = "Tq+k+v+o"
AXES = ("occ_gender", "crows_socioeconomic")
ROLE_MAP = {"self": "self", "sibling": "sibling",
            "cross": "cross", "cross2": "cross", "cross3": "cross", "cross4": "cross"}
DROP_ROLES = ("random", "gt")

# checkpoint -> nominal size, for tier reporting (SUMMARY_V5 §3 convention)
SIZE = {"olmo1b": 1.0, "llama1b": 1.2, "falcon1b": 1.5, "qwen1.5b": 1.5, "smol1.7b": 1.7,
        "granite2b": 2.5, "gemma2b": 2.6, "phi3.5": 3.8, "qwen7b": 7.0}

STRICT_DMMLU, STRICT_PPLR = 0.02, 1.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(RES, "runs.jsonl"))
    ap.add_argument("--out", default=os.path.join(RES, "v6trace_3b"))
    ap.add_argument("--panel", default="small")
    ap.add_argument("--max-size", type=float, default=4.0,
                    help="keep targets at/below this size (B); 7B lives in the t2x trace")
    a = ap.parse_args()

    rows = [json.loads(ln) for ln in open(a.runs) if ln.strip()]

    def module_set(r):
        tags = [p for p in r.get("key", "").split("|") if p.startswith("T")]
        return tags[0] if tags else "(untagged)"

    sub = [r for r in rows if r.get("variant") == VARIANT and r.get("signal") == "endogenous"
           and r["axis"] in AXES and module_set(r) == MODULE_TAG]

    # the frozen protocol must be uniquely keyed; if it is not, some other
    # experiment dimension is still folded in and must be found before analysis.
    kc = collections.Counter((r["target"], r["axis"], r["seed"], r["designer_role"],
                              r["designer"], r["alpha"]) for r in sub)
    ndup = sum(1 for n in kc.values() if n > 1)
    if ndup:
        raise SystemExit(f"ABORT: {ndup}/{len(kc)} (target,axis,seed,role,designer,alpha) tuples "
                         f"are still duplicated after filtering to {MODULE_TAG} — an unmodelled "
                         f"experiment dimension remains; find it before analyzing.")

    # integrity: the stored budget flag must match the frozen budget, or the
    # adapted panel is not comparable with the t2x trace.
    bad = [r for r in sub
           if r["collateral_ok"] != ((r["pre"]["mmlu_acc"] - r["post"]["mmlu_acc"]) <= STRICT_DMMLU
                                     and r["post"]["ppl"] / r["pre"]["ppl"] <= STRICT_PPLR)]
    if bad:
        raise SystemExit(f"ABORT: {len(bad)}/{len(sub)} rows' collateral_ok disagrees with the "
                         f"frozen {STRICT_DMMLU}/{STRICT_PPLR} budget — the panels are not comparable.")

    # occ_gender bias_reduction must be re-derivable; crows cannot be (schema wart).
    occ = [r for r in sub if r["axis"] == "occ_gender"]
    ver = sum(1 for r in occ
              if abs((abs(r["pre"]["occ_skew"]) - abs(r["post"]["occ_skew"])) - r["bias_reduction"]) < 1e-9)
    if ver != len(occ):
        raise SystemExit(f"ABORT: occ_gender bias_reduction re-derives on only {ver}/{len(occ)} rows.")

    kept, dropped = [], collections.Counter()
    for r in sub:
        role = ROLE_MAP.get(r["designer_role"])
        if role is None:
            dropped[r["designer_role"]] += 1
            continue
        tgt = r["target"]
        if SIZE.get(tgt, 99) > a.max_size:
            dropped[f"size>{a.max_size}:{tgt}"] += 1
            continue
        pre, post = r["pre"], r["post"]
        kept.append(dict(target=tgt, axis=r["axis"], seed=r["seed"], role=role,
                         designer=r["designer"], alpha=float(r["alpha"]),
                         pre_skew=pre.get("occ_skew"), post_skew=post.get("occ_skew"),
                         pre_mmlu=pre["mmlu_acc"], post_mmlu=post["mmlu_acc"],
                         pre_ppl=pre["ppl"], post_ppl=post["ppl"],
                         dmmlu=pre["mmlu_acc"] - post["mmlu_acc"],
                         ppl_ratio=post["ppl"] / pre["ppl"],
                         collateral_ok=bool(r["collateral_ok"]),
                         bias_reduction=float(r["bias_reduction"]),
                         size_b=SIZE.get(tgt), source="runs.jsonl"))

    d = os.path.join(a.out, a.panel)
    os.makedirs(d, exist_ok=True)
    tp = os.path.join(d, "alpha_trace.jsonl")
    with open(tp, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    # derived removal.jsonl (best in-budget removal per sweep, nan->absent).
    # NOTE this is DERIVED from the same trace, so v6_mva_analyze's faithfulness
    # check is vacuous for this panel — it is written as a convenience artifact,
    # not as an independent record.
    sw = collections.defaultdict(list)
    for r in kept:
        sw[(r["target"], r["axis"], r["seed"], r["role"], r["designer"])].append(r)
    rp = os.path.join(d, "removal.jsonl")
    with open(rp, "w") as f:
        for k, v in sw.items():
            ok = [x["bias_reduction"] for x in v if x["collateral_ok"]]
            f.write(json.dumps(dict(target=k[0], axis=k[1], seed=k[2], role=k[3], designer=k[4],
                                    removal=max(ok) if ok else float("nan"),
                                    pre_skew=v[0]["pre_skew"])) + "\n")

    cells = sorted({(r["target"], r["axis"]) for r in kept})
    print(f"wrote {tp}")
    print(f"      {len(kept)} alpha evals | {len(sw)} sweeps | {len(cells)} target×axis cells")
    print(f"wrote {rp}  (DERIVED — faithfulness check is vacuous for this panel)")
    print(f"dropped: {dict(dropped)}")
    print("\ncells:")
    per = collections.Counter((r["target"], r["axis"]) for r in kept)
    for t, ax in cells:
        roles = collections.Counter(r["role"] for r in kept if r["target"] == t and r["axis"] == ax)
        print(f"  {t:10s} {SIZE.get(t, 0):4.1f}B  {ax:22s} {per[(t, ax)]:4d} evals  {dict(roles)}")
    print("\nnext:  python src/v6_mva_analyze.py --root %s --panel %s" % (a.out, a.panel))


if __name__ == "__main__":
    main()
