"""R1 — the bias-blind alpha* selector (R1_REBUTTAL_EXPERIMENTS.md §0, §9).

WHY THIS FILE EXISTS. The R1 runner measures, at every rung of the alpha
ladder, BOTH the calibration collateral (what alpha* is selected on) and the
evaluation-split bias probe (what the removal curves are drawn from). The
registered rule (PREREGISTRATION §v12.A) is that alpha* never reads a bias
probe. With both measurements sitting side by side in the runner, "never reads"
must be a property of the code structure, not of discipline.

So selection goes through exactly one function, `select`, which sees the world
only through `collateral_probe(alpha) -> {mmlu_acc, ppl, n_items}`. `gate`
refuses any dict that carries another field: handing it a skew, a removal or a
bias_reduction raises BiasLeak instead of being silently ignored.

The decision itself is v9_gate.collateral_ok (integer-item MMLU gate, ppl ratio
with 1e-9 tolerance) and the search is v12_opsel.select_alpha (first failure on
the ladder, then log-bisection) -- the two functions every published alpha*
used. Nothing about the rule is re-implemented here.

Torch-free: importable on the analysis box.
"""
import inspect

import v9_gate
import v12_opsel as S

ALLOWED = frozenset({"mmlu_acc", "ppl", "n_items"})
REQUIRED = frozenset({"mmlu_acc", "ppl"})


class BiasLeak(TypeError):
    """Something other than collateral reached the alpha* selector."""


def _check(name, d):
    if not isinstance(d, dict):
        raise BiasLeak(f"{name}: expected a collateral dict, got {type(d).__name__}")
    extra = set(d) - ALLOWED
    if extra:
        raise BiasLeak(
            f"{name} carries non-collateral field(s) {sorted(extra)}; the alpha* "
            "rule may read MMLU and perplexity on the calibration split only")
    missing = REQUIRED - set(d)
    if missing:
        raise BiasLeak(f"{name} lacks {sorted(missing)}")


def collateral_only(d):
    """Project a measurement dict onto the fields the selector may see."""
    return {k: d[k] for k in ("mmlu_acc", "ppl", "n_items") if k in d}


def gate(pre, post):
    """Budget decision from calibration collateral alone.

    Same semantics as colab_t2t4.collateral_ok(pre, post): the item count comes
    from the measurement, a pre/post count mismatch is refused, and the decision
    is v9_gate.collateral_ok at the registered budget (4 of 200 items, 1.10).
    """
    _check("pre", pre)
    _check("post", post)
    n_pre, n_post = pre.get("n_items"), post.get("n_items")
    if n_pre and n_post and n_pre != n_post:
        raise ValueError(f"pre/post scored on different MMLU item counts "
                         f"({n_pre} vs {n_post})")
    n = n_pre or n_post or v9_gate.N_ITEMS_DEFAULT
    return bool(v9_gate.collateral_ok(pre["mmlu_acc"], post["mmlu_acc"],
                                      pre["ppl"], post["ppl"], n_items=n))


def select(pre, collateral_probe, ladder=S.ALPHA_LADDER,
           refine_steps=S.REFINE_STEPS):
    """alpha* by the registered first-failure rule, from collateral alone.

    pre               calibration-split collateral of the unedited model
    collateral_probe  alpha -> calibration-split collateral with the edit active
    """
    _check("pre", pre)
    return S.select_alpha(lambda a: gate(pre, collateral_probe(a)),
                          ladder, refine_steps)


# ------------------------------------------------------------------ tests ----
def selftest(verbose=True):
    checks = []
    pre = dict(mmlu_acc=0.600, ppl=20.0, n_items=200)

    def post_at(a, limit=37.0):
        # 4 items lost up to the limit, 5 beyond it; ppl flat
        return dict(mmlu_acc=(0.580 if a <= limit else 0.575), ppl=20.5, n_items=200)

    # 1. a bias value reaching the gate is an error, not a no-op
    for bad in ("skew", "post_skew", "removal", "bias_reduction", "pre_skew"):
        try:
            gate(pre, dict(post_at(1.0), **{bad: 0.3}))
            checks.append((f"gate rejects a post dict carrying {bad!r}", False))
        except BiasLeak:
            checks.append((f"gate rejects a post dict carrying {bad!r}", True))
    try:
        gate(dict(pre, skew=0.8), post_at(1.0))
        checks.append(("gate rejects a pre dict carrying 'skew'", False))
    except BiasLeak:
        checks.append(("gate rejects a pre dict carrying 'skew'", True))
    try:
        select(pre, lambda a: dict(post_at(a), skew=0.1))
        checks.append(("select fails when the probe leaks a bias field", False))
    except BiasLeak:
        checks.append(("select fails when the probe leaks a bias field", True))
    try:
        gate(pre, 0.3)
        checks.append(("gate rejects a bare number", False))
    except BiasLeak:
        checks.append(("gate rejects a bare number", True))

    # 2. the selector's interface has no slot through which bias could arrive
    params = list(inspect.signature(select).parameters)
    checks.append(("select(pre, collateral_probe, ladder, refine_steps) -- no bias slot",
                   params == ["pre", "collateral_probe", "ladder", "refine_steps"]))
    seen = []
    r = select(pre, lambda a: (seen.append(a), post_at(a))[1])
    checks.append(("the probe is called with alpha and nothing else",
                   all(isinstance(a, float) for a in seen) and len(seen) == r["n_probes"]))

    # 3. identical to the published rule: v12_opsel.select_alpha over the v9 gate
    ref = S.select_alpha(lambda a: v9_gate.collateral_ok(
        pre["mmlu_acc"], post_at(a)["mmlu_acc"], pre["ppl"], post_at(a)["ppl"]))
    checks.append(("select == v12_opsel.select_alpha(v9_gate) on a monotone case",
                   r["alpha_star"] == ref["alpha_star"] and r["probes"] == ref["probes"]
                   and r["status"] == "ok" and 32.0 <= r["alpha_star"] <= 37.0))
    r0 = select(pre, lambda a: dict(mmlu_acc=0.5, ppl=20.0, n_items=200))
    checks.append(("nothing admissible -> status 'none', alpha* None",
                   r0["status"] == "none" and r0["alpha_star"] is None))
    r1 = select(pre, lambda a: dict(mmlu_acc=0.6, ppl=20.0, n_items=200))
    checks.append(("never fails -> 'exhausted' at the ladder top (censored)",
                   r1["status"] == "exhausted" and r1["alpha_star"] == 512.0))

    # 4. gate == v9 gate on the boundary cases
    b = [(0.545, 0.525, True), (0.545, 0.520, False), (0.5, 0.6, True)]
    checks.append(("integer-item boundary: 4 items pass, 5 fail",
                   all(gate(dict(mmlu_acc=p, ppl=10.0, n_items=200),
                            dict(mmlu_acc=q, ppl=10.0, n_items=200)) is ok
                       for p, q, ok in b)))
    checks.append(("ppl ratio 1.10 passes, 1.11 fails",
                   gate(dict(mmlu_acc=.5, ppl=10.0), dict(mmlu_acc=.5, ppl=11.0)) is True
                   and gate(dict(mmlu_acc=.5, ppl=10.0), dict(mmlu_acc=.5, ppl=11.1)) is False))
    try:
        gate(dict(mmlu_acc=.5, ppl=10.0, n_items=200),
             dict(mmlu_acc=.5, ppl=10.0, n_items=1000))
        checks.append(("pre/post item-count mismatch refused", False))
    except ValueError:
        checks.append(("pre/post item-count mismatch refused", True))

    # 5. collateral_only strips everything else
    full = dict(skew=0.4, mmlu_acc=0.6, ppl=20.0, n_items=200)
    checks.append(("collateral_only drops the bias field",
                   collateral_only(full) == dict(mmlu_acc=0.6, ppl=20.0, n_items=200)))

    bad = [d for d, ok in checks if not ok]
    if verbose:
        for d, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {d}")
        print(f"r1_select selftest: {len(checks) - len(bad)}/{len(checks)} passed")
    return not bad


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
