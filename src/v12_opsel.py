"""v12 — operating-point selection: the pure logic, testable without a GPU.

Two decisions replace constants of the published method (experiments_v12.md):

  p   the retained fraction. Published: fixed 0.01. v12: p* predicted from the
      edit's own |ΔW| geometry (v12_geometry.py, v12_pstar.py).
  α   the edit scale. Published: argmax of bias removal over {2,4,8,16}, with
      the collateral gate applied ON THE SAME PROBES the result is reported on.
      v12: α*(p) = the largest α whose collateral is within budget on a
      CALIBRATION split that shares no item with the evaluation split, found
      without ever reading a bias probe. The result is then measured once, on
      the evaluation split, at α*.

WHY α* NEVER READS BIAS. The published argmax selects on the reported quantity.
α* is a pure collateral rule, so the removal it reports is not selected on at
all, and whether α* also clears the budget on the held-out evaluation items is
itself a measurement (the budget-pass rate of the rule), not a tautology.

MONOTONICITY IS NOT ASSUMED. Collateral is not guaranteed monotone in α. The
rule is FIRST-FAILURE: climb the ladder until the first probe that fails the
gate, then bisect (in log α) between the last pass and that failure. A pass
that reappears above a failure is ignored. That is conservative by design, and
the full ladder is persisted so the ignored region is auditable.

This module is imported by the GPU runner (run_opsel.py) and by the analysis
scripts; it must not import torch.
"""
import hashlib
import json
import math
import os

# ---- defaults, overridden by the panel config; registered in §v12.C ----
ALPHA_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
REFINE_STEPS = 3          # log-bisection steps -> resolution 2**(1/8) ~ 9%


def select_alpha(probe, ladder=ALPHA_LADDER, refine_steps=REFINE_STEPS):
    """First-failure α search. `probe(alpha) -> bool` (collateral within budget).

    Returns dict(alpha_star, last_pass, first_fail, n_probes, status, probes)
    with status one of
        'ok'          a failure was bracketed and refined
        'none'        the smallest ladder α already fails; nothing admissible
        'exhausted'   every ladder α passes; α* is a lower bound, NOT saturated
    `probes` is the ordered list of (alpha, ok) actually evaluated.
    """
    ladder = sorted(float(a) for a in ladder)
    probes = []

    def _p(a):
        ok = bool(probe(a))
        probes.append((float(a), ok))
        return ok

    last_pass, first_fail = None, None
    for a in ladder:
        if _p(a):
            last_pass = a
        else:
            first_fail = a
            break
    if last_pass is None:
        return dict(alpha_star=None, last_pass=None, first_fail=first_fail,
                    n_probes=len(probes), status="none", probes=probes)
    if first_fail is None:
        return dict(alpha_star=last_pass, last_pass=last_pass, first_fail=None,
                    n_probes=len(probes), status="exhausted", probes=probes)
    lo, hi = last_pass, first_fail
    for _ in range(refine_steps):
        mid = math.sqrt(lo * hi)
        if _p(mid):
            lo = mid
        else:
            hi = mid
    return dict(alpha_star=lo, last_pass=last_pass, first_fail=first_fail,
                n_probes=len(probes), status="ok", probes=probes)


def frozen_argmax(results):
    """The published selection: max removal over in-budget α. nan if none.

    results: iterable of (alpha, ok, removal). Returns (best, alpha_of_best).
    """
    best, best_a = float("nan"), None
    for a, ok, r in results:
        if ok and (math.isnan(best) or r > best):
            best, best_a = r, a
    return best, best_a


# ------------------------------------------------------ held-out guard ----
class HeldOutViolation(RuntimeError):
    """A held-out cell is about to be evaluated without a valid prior prediction."""


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pstar(pred_path, target, axis, seed, geom_sha):
    """Return (p_star, pred_sha) for one held-out unit, or raise.

    The prediction file is written by `v12_pstar.py predict`, which refuses to
    run once any curve row for a held-out target exists. This check is the other
    half: the runner refuses to MEASURE a held-out curve unless

      1. the prediction exists and names this cell,
      2. the target is in the prediction's held-out set and NOT in its fit set,
      3. the prediction was made from THIS ΔW (the geometry fingerprint of the
         retrain matches the phase-A fingerprint the prediction consumed).

    (3) matters because p* is a function of the geometry: a retrain that drifted
    would be evaluated against a prediction made for a different vector.
    """
    if not pred_path or not os.path.exists(pred_path):
        raise HeldOutViolation(
            f"no p* prediction at {pred_path!r}. Held-out cells are evaluated only "
            "after `python3 src/v12_pstar.py predict` has frozen one "
            "(experiments_v12.md §II, step 3).")
    pred = json.load(open(pred_path))
    fit_targets = set(pred.get("fit", {}).get("targets", []))
    if target in fit_targets:
        raise HeldOutViolation(
            f"{target} is in the p* FIT population {sorted(fit_targets)}; it "
            "cannot also be held out")
    cell = pred.get("cells", {}).get(f"{target}|{axis}")
    if cell is None:
        raise HeldOutViolation(f"prediction {pred_path} has no cell {target}|{axis}")
    shas = cell.get("geom_shas_by_seed", {})
    if str(seed) not in shas:
        raise HeldOutViolation(
            f"prediction for {target}|{axis} was made without seed {seed}'s "
            "geometry; re-run phase A for that seed and re-predict")
    if shas[str(seed)] != geom_sha:
        raise HeldOutViolation(
            f"{target}|{axis}|s{seed}: retrained ΔW fingerprint {geom_sha} != "
            f"{shas[str(seed)]} used by the prediction. Training is not "
            "reproducing on this stack; the prediction does not apply to this "
            "vector. Do not override -- investigate (run_ws2_determinism.py).")
    return float(cell["p_star"]), file_sha(pred_path)


# -------------------------------------------------------------- selftest ----
def selftest():
    fails = []

    # monotone collateral: passes below 37
    r = select_alpha(lambda a: a <= 37)
    if r["status"] != "ok" or not (32 <= r["alpha_star"] <= 37):
        fails.append(f"monotone case: {r}")
    if r["n_probes"] != 7 + REFINE_STEPS:          # 1..64 on the ladder, +3
        fails.append(f"monotone case probe count {r['n_probes']}")

    # first-failure: a pass above a failure is ignored
    r = select_alpha(lambda a: a <= 8 or a >= 128)
    if r["alpha_star"] > 16:
        fails.append(f"non-monotone case climbed past the first failure: {r}")

    # nothing admissible
    r = select_alpha(lambda a: False)
    if r["status"] != "none" or r["alpha_star"] is not None or r["n_probes"] != 1:
        fails.append(f"none case: {r}")

    # never fails -> exhausted, flagged
    r = select_alpha(lambda a: True)
    if r["status"] != "exhausted" or r["alpha_star"] != 512.0:
        fails.append(f"exhausted case: {r}")

    # frozen argmax = the published rule
    b, a = frozen_argmax([(2, True, .1), (4, True, .3), (8, False, .9), (16, True, .2)])
    if (b, a) != (.3, 4):
        fails.append(f"frozen_argmax {(b, a)}")
    b, a = frozen_argmax([(2, False, .1)])
    if not math.isnan(b) or a is not None:
        fails.append("frozen_argmax with nothing in budget must be nan")

    # held-out guard
    import tempfile
    d = tempfile.mkdtemp()
    fp = os.path.join(d, "pred.json")
    json.dump(dict(fit=dict(targets=["qwen", "phi"]),
                   cells={"qwen32b|occ_gender": dict(
                       p_star=0.004, geom_shas_by_seed={"0": "abc"})}), open(fp, "w"))
    for args, why in [((None, "qwen32b", "occ_gender", 0, "abc"), "missing file"),
                      ((fp, "qwen", "occ_gender", 0, "abc"), "fit target"),
                      ((fp, "qwen32b", "bbq_Age", 0, "abc"), "missing cell"),
                      ((fp, "qwen32b", "occ_gender", 1, "abc"), "missing seed"),
                      ((fp, "qwen32b", "occ_gender", 0, "xyz"), "drifted ΔW")]:
        try:
            load_pstar(*args)
            fails.append(f"held-out guard accepted: {why}")
        except HeldOutViolation:
            pass
    try:
        p, _ = load_pstar(fp, "qwen32b", "occ_gender", 0, "abc")
        if p != 0.004:
            fails.append("load_pstar returned the wrong p")
    except HeldOutViolation as e:
        fails.append(f"held-out guard rejected a valid prediction: {e}")

    for m in fails:
        print(f"  FAIL {m}")
    print(f"v12_opsel selftest: {'PASS' if not fails else str(len(fails)) + ' FAILED'}")
    return not fails


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
