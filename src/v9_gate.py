"""WS1.1 — the corrected collateral gate. ONE implementation, used everywhere.

THE BUG THIS REPLACES. `colab_t2t4.collateral_ok` tests

    (pre["mmlu_acc"] - post["mmlu_acc"]) <= 0.02

MMLU is scored on 200 items, so accuracy moves in steps of 1/200 = 0.005 and a
**4-item drop is exactly the 0.02 budget**. In IEEE-754 it is not:
0.545 - 0.525 == 0.020000000000000018 > 0.02. Configurations sitting precisely ON
the budget were rejected — 238 probes across 25 alpha-trace panels and a further
430 rows in results/runs.jsonl.

THE FIX (CLOSEOUT_V9 §1.1). Compare MMLU in **integer items**, not floats:

    allowed_drop_items = round(0.02 * n_items)          # 4 at n_items = 200
    (pre_items - post_items) <= allowed_drop_items

This removes the boundary error *exactly* — no epsilon, nothing to tune, and it
is the form a reviewer can check by hand. Perplexity stays a float ratio and gets
an explicit 1e-9 tolerance, since 1.10 has no exact binary representation either.

n_items IS NOT ASSUMED. Every stored accuracy in this repository is an exact
multiple of 1/200 (verified over 7512 trace values and 10254 runs.jsonl rows), so
the item count is recovered by `items_from_acc`, which hard-fails rather than
guessing if a value is not on the 1/200 grid.
"""

N_ITEMS_DEFAULT = 200
DMMLU_BUDGET = 0.02
PPL_BUDGET = 1.10
PPL_TOL = 1e-9


class NonIntegralAccuracy(ValueError):
    """An accuracy that is not on the n_items grid — refuse rather than round."""


def items_from_acc(acc, n_items=N_ITEMS_DEFAULT, tol=1e-6):
    """Exact integer correct-count from a stored accuracy.

    Refuses instead of silently rounding: if a panel ever changes n_items, a
    wrong denominator here would shift every gate decision, and that must be a
    loud failure rather than a quiet one.
    """
    x = acc * n_items
    r = round(x)
    if abs(x - r) > tol:
        raise NonIntegralAccuracy(
            f"accuracy {acc!r} is not a multiple of 1/{n_items} "
            f"(acc*n = {x!r}); the item count for this panel is not {n_items}")
    return int(r)


def allowed_drop_items(n_items=N_ITEMS_DEFAULT, budget=DMMLU_BUDGET):
    return round(budget * n_items)


def mmlu_ok(pre_acc, post_acc, n_items=N_ITEMS_DEFAULT, budget=DMMLU_BUDGET):
    """True iff the MMLU drop is within budget, compared in integer items."""
    return (items_from_acc(pre_acc, n_items) - items_from_acc(post_acc, n_items)) \
        <= allowed_drop_items(n_items, budget)


def ppl_ok(pre_ppl, post_ppl, budget=PPL_BUDGET, tol=PPL_TOL):
    return (post_ppl / pre_ppl) <= budget + tol


def collateral_ok(pre_acc, post_acc, pre_ppl, post_ppl,
                  n_items=N_ITEMS_DEFAULT, dmmlu=DMMLU_BUDGET, pplr=PPL_BUDGET):
    """The v9 gate. Same semantics as colab_t2t4.collateral_ok, minus the bug."""
    return mmlu_ok(pre_acc, post_acc, n_items, dmmlu) and \
        ppl_ok(pre_ppl, post_ppl, pplr)


def mmlu_ok_delta(dmmlu, n_items=N_ITEMS_DEFAULT, budget=DMMLU_BUDGET):
    """Integer gate applied to a stored (pre - post) accuracy difference.

    A difference of two multiples of 1/n is itself a multiple of 1/n, so the
    integer comparison works without the pre/post pair. Verified across all 7960
    stored dmmlu values in the repository: every one is on the 1/200 grid. This
    is what lets the runners that persist only `dmmlu` (run_dec, run_spc,
    run_ins, run_ablate_attn) use the same reviewer-proof integer test as the
    panels that store pre/post, instead of falling back to an epsilon.
    """
    x = dmmlu * n_items
    r = round(x)
    if abs(x - r) > 1e-6:
        raise NonIntegralAccuracy(
            f"dmmlu {dmmlu!r} is not a multiple of 1/{n_items}")
    return int(r) <= allowed_drop_items(n_items, budget)


def collateral_ok_row(row, n_items=N_ITEMS_DEFAULT):
    """Gate an alpha_trace row, which stores pre/post mmlu and a ppl ratio.

    Falls back to the stored `dmmlu` only when the pre/post pair is absent; that
    path cannot use integer comparison and is reported by the re-scorer so the
    weaker treatment is never silent.
    """
    pre_m, post_m = row.get("pre_mmlu"), row.get("post_mmlu")
    ratio = row.get("ppl_ratio")
    if ratio is None:
        pre_p, post_p = row.get("pre_ppl"), row.get("post_ppl")
        ratio = (post_p / pre_p) if (pre_p and post_p) else None
    if ratio is None:
        return None, "no ppl information"
    if pre_m is None or post_m is None:
        d = row.get("dmmlu")
        if d is None:
            return None, "no mmlu information"
        # integer gate on the stored difference -- see mmlu_ok_delta
        try:
            return (mmlu_ok_delta(d, n_items) and
                    (ratio <= PPL_BUDGET + PPL_TOL)), "integer-delta"
        except NonIntegralAccuracy:
            return (d <= DMMLU_BUDGET + 1e-9) and (ratio <= PPL_BUDGET + PPL_TOL), \
                "float-fallback (dmmlu off-grid)"
    return (mmlu_ok(pre_m, post_m, n_items) and
            (ratio <= PPL_BUDGET + PPL_TOL)), "integer"


# ------------------------------------------------------------------ tests ----

def selftest(verbose=True):
    """Unit tests, including the known failing case from CLOSEOUT_V9 §1.1."""
    checks = []

    # THE case: a 4-item drop at n=200 is exactly the budget and must PASS.
    old = (0.545 - 0.525) <= 0.02
    new = mmlu_ok(0.545, 0.525)
    checks.append(("0.545 - 0.525 @200 items: old gate rejects", old is False))
    checks.append(("0.545 - 0.525 @200 items: v9 gate ACCEPTS", new is True))
    checks.append(("  (drop is exactly 4 items)",
                   items_from_acc(0.545) - items_from_acc(0.525) == 4))

    # 5 items must still fail; 3 must pass.
    checks.append(("5-item drop rejected", mmlu_ok(0.545, 0.520) is False))
    checks.append(("3-item drop accepted", mmlu_ok(0.545, 0.530) is True))
    # improvement is always fine
    checks.append(("MMLU improvement accepted", mmlu_ok(0.500, 0.600) is True))

    # exhaustive over the whole 1/200 grid: the gate must be exactly "<= 4 items"
    bad = 0
    for i in range(201):
        for j in range(201):
            if mmlu_ok(i / 200, j / 200) != ((i - j) <= 4):
                bad += 1
    checks.append((f"exhaustive over all 201x201 grid pairs (bad={bad})", bad == 0))

    # ppl tolerance
    checks.append(("ppl ratio exactly 1.10 accepted", ppl_ok(10.0, 11.0) is True))
    checks.append(("ppl ratio 1.11 rejected", ppl_ok(10.0, 11.1) is False))

    # the delta-only integer path must agree with the pre/post path
    agree = all(mmlu_ok_delta((i - j) / 200) == mmlu_ok(i / 200, j / 200)
                for i in range(0, 201, 7) for j in range(0, 201, 3))
    checks.append(("mmlu_ok_delta agrees with mmlu_ok on the grid", agree))

    # non-integral accuracy must raise, not round
    try:
        items_from_acc(0.5031)
        checks.append(("non-integral accuracy raises", False))
    except NonIntegralAccuracy:
        checks.append(("non-integral accuracy raises", True))

    # agreement with the old gate everywhere EXCEPT the boundary
    disagree = []
    for i in range(201):
        for j in range(201):
            a, b = i / 200, j / 200
            if ((a - b) <= 0.02) != mmlu_ok(a, b):
                disagree.append((i, j, i - j))
    only_4 = all(d == 4 for _i, _j, d in disagree)
    checks.append((f"old/new disagree ONLY at a 4-item drop "
                   f"({len(disagree)} pairs)", only_4 and len(disagree) > 0))

    n_bad = sum(1 for _d, ok in checks if not ok)
    if verbose:
        for d, ok in checks:
            print(f"  {'PASS' if ok else 'FAIL'}  {d}")
        print(f"\n{len(checks) - n_bad}/{len(checks)} gate checks passed")
    return n_bad == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
