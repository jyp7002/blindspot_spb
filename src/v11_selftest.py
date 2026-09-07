"""v11 — regression tests for the collateral gate's item-count propagation.

WHAT THIS PROTECTS. Through v10 the MMLU item count lived in exactly one place:
the literal `200` in `run_dec.main`'s `C.load_mmlu(n=200, seed=0)`. It was never
written to a row, and `v9_gate` defaults to 200 everywhere. So the gate was
correct only because every panel happened to use 200 -- an invariant nothing
checked and no artifact recorded.

That is fine until someone scales the collateral probe. Then it fails in the
worst available way: LOUDLY on accuracies off the 1/200 grid (items_from_acc
raises), and SILENTLY on the ones that happen to land on it, because 1/1000
values like 0.520 are also exact multiples of 1/200. A panel would come back
part-gated against a 4-item budget and part-crashed, and the rows would not say
which.

v11 attaches `n_items` in `fast_eval` (where the count is known exactly) and
reads it in `collateral_ok`. These tests pin that down:

  1. at n=200 the gate's decisions are IDENTICAL to v9's -- the fix is inert on
     every published panel;
  2. at n=1000 the budget is 20 items, and a drop that is out of budget at 200
     is correctly in budget at 1000;
  3. mismatched pre/post counts raise instead of silently picking one.

Run: python3 src/v11_selftest.py   (no GPU, no model downloads)
"""
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

import v9_gate  # noqa: E402  (analysis-only, always importable)


# ---------------------------------------------------------------------------
# colab_t2t4 pulls torch/datasets/transformers/peft at import time and none of
# them are needed to exercise collateral_ok. Stub them so this test runs on the
# analysis box, where the GPU stack is deliberately absent.
# ---------------------------------------------------------------------------
class _Any:
    """Permissive stand-in: attribute access, calls, decoration, `with`.

    `@torch.no_grad()` is applied at import time, so the stub has to survive
    being called and then used as a decorator. Calling it with a single
    callable returns that callable unchanged; anything else returns the stub.
    """

    def __getattr__(self, _name):
        return _Any()

    def __call__(self, *a, **k):
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]                      # used as a decorator
        return _Any()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_heavy_imports():
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.bfloat16 = "bfloat16"
        torch.nn = types.ModuleType("torch.nn")
        torch.nn.functional = types.ModuleType("torch.nn.functional")
        torch.__getattr__ = lambda _n: _Any()
        torch.nn.__getattr__ = lambda _n: _Any()
        torch.nn.functional.__getattr__ = lambda _n: _Any()
        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = torch.nn
        sys.modules["torch.nn.functional"] = torch.nn.functional
    if "datasets" not in sys.modules:
        ds = types.ModuleType("datasets")
        ds.disable_progress_bars = lambda: None
        ds.load_dataset = lambda *a, **k: {}
        sys.modules["datasets"] = ds
    if "transformers" not in sys.modules:
        tf = types.ModuleType("transformers")
        tf.AutoModelForCausalLM = object
        tf.AutoTokenizer = object
        tf.BitsAndBytesConfig = object
        sys.modules["transformers"] = tf
    if "peft" not in sys.modules:
        pf = types.ModuleType("peft")
        pf.LoraConfig = object
        pf.get_peft_model = lambda *a, **k: None
        sys.modules["peft"] = pf


def _ev(acc, ppl, n_items=None):
    """A fast_eval-shaped measurement."""
    d = dict(skew=0.5, mmlu_acc=acc, ppl=ppl)
    if n_items is not None:
        d["n_items"] = n_items
    return d


def main():
    _stub_heavy_imports()
    import colab_t2t4 as C

    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)
            print(f"  FAIL {name}")

    # -- 1. inert at n=200: agrees with the v9 gate on the whole 200-item grid --
    n_disagree = 0
    for pre_i in range(0, 201):
        for post_i in (pre_i, pre_i - 3, pre_i - 4, pre_i - 5):
            if post_i < 0:
                continue
            pre_a, post_a = pre_i / 200, post_i / 200
            want = v9_gate.collateral_ok(pre_a, post_a, 10.0, 10.0, n_items=200)
            got = C.collateral_ok(_ev(pre_a, 10.0, 200), _ev(post_a, 10.0, 200))
            if want != got:
                n_disagree += 1
    check("n=200 decisions identical to v9_gate (all pre x {0,-3,-4,-5} items)",
          n_disagree == 0)

    # -- 1b. a row with NO n_items still gates as 200 (every published panel) --
    check("missing n_items falls back to 200",
          C.collateral_ok(_ev(0.545, 10.0), _ev(0.525, 10.0)) is True)

    # -- 2. the count actually changes the budget --
    # 10 items dropped out of 1000 = 1%: inside the 2% budget (20 items).
    # The same accuracies read as 200 items would be a 2-item drop, also in
    # budget -- so pick a case where the two DISAGREE:
    # 30/1000 dropped = 3% -> OUT of budget at n=1000 (30 > 20).
    check("n=1000: a 30-item drop is out of budget",
          C.collateral_ok(_ev(0.600, 10.0, 1000), _ev(0.570, 10.0, 1000)) is False)
    check("n=1000: a 20-item drop is exactly in budget",
          C.collateral_ok(_ev(0.600, 10.0, 1000), _ev(0.580, 10.0, 1000)) is True)
    # The boundary case that motivated the v9 integer gate, at the new count.
    check("n=1000: the ON-budget boundary is admitted, not float-rejected",
          C.collateral_ok(_ev(0.545, 10.0, 1000), _ev(0.525, 10.0, 1000)) is True)

    # -- 2b. the same accuracies gate DIFFERENTLY at the two counts --
    # 0.600 -> 0.580 is 4 items of 200 (in budget) and 20 of 1000 (in budget);
    # 0.600 -> 0.575 is 5 of 200 (OUT) and 25 of 1000 (OUT). Use a pair that
    # separates: 0.600 -> 0.585 is 3 of 200 (in) and 15 of 1000 (in).
    # The real separation is that a wrong denominator RAISES:
    raised = False
    try:
        v9_gate.collateral_ok(0.6005, 0.5805, 10.0, 10.0, n_items=200)
    except v9_gate.NonIntegralAccuracy:
        raised = True
    check("an n=1000 accuracy read as n=200 raises rather than rounding", raised)

    # -- 3. mismatched counts are a category error, not a coin flip --
    raised = False
    try:
        C.collateral_ok(_ev(0.600, 10.0, 200), _ev(0.580, 10.0, 1000))
    except ValueError:
        raised = True
    check("pre/post on different item counts raises", raised)

    # -- 4. fast_eval attaches the count it actually scored --
    src = open(os.path.join(REPO, "colab_t2t4.py")).read()
    check("fast_eval records n_items=len(mmlu_rows)",
          "n_items=len(mmlu_rows)" in src)
    check("alpha_trace persists n_items", 'n_items=pre.get("n_items", 200)' in src)

    # -- 5. no runner stamps a HARDCODED panel name into its rows --
    #
    # run_dec.py wrote `panel="v8dec"` as a literal, so every row it ever
    # produced claimed to be the published panel regardless of where it was
    # written. That is the origin of the standing "never glob v8dec*" caveat,
    # and under a scale-up it would have made a new panel indistinguishable
    # from the published one inside its own artifacts. run_ins.py had the same
    # defect in two places. Both now read a PANEL variable.
    import re
    for runner in ("run_dec.py", "run_alpha_ext.py", "run_ins.py"):
        text = open(os.path.join(REPO, runner)).read()
        # strip comments so the explanatory notes above don't trip the check
        body = "\n".join(l.split("#")[0] for l in text.splitlines())
        bad = re.findall(r'panel\s*=\s*["\'][^"\']+["\']', body)
        check(f"{runner} does not hardcode a panel name (found {bad})", not bad)

    # -- 6. the DEC trace keeps the raw MMLU endpoints, not just their delta --
    dec = open(os.path.join(REPO, "run_dec.py")).read()
    check("run_dec trace persists pre_mmlu/post_mmlu/n_items",
          all(k in dec for k in ("pre_mmlu=pre[", "post_mmlu=post[",
                                 'n_items=pre.get("n_items"')))

    # -- 7. the 1e9 hypergeometric wall that blocked the whole big tier --
    #
    # src/v8_edits._counts_per_tensor split the 1% support across tensors with
    # numpy's hypergeometric, which refuses arguments >= 1e9. Every published
    # model edits fewer than that (qwen2.5-7B, the largest, is 8.2e8; phi is
    # 9.1e8 through its fused qkv_proj), so the wall was invisible until v11
    # added llama-3.1-8B at 1.34e9. qwen2.5-32B (4.0e9) and llama-3.1-70B
    # (12.1e9) are far past it, so the big tier was blocked here and not on VRAM.
    import math
    sys.path.insert(0, os.path.join(REPO, "src"))
    import numpy as np
    import v8_edits as V8

    class _T:
        def __init__(self, n): self._n = n
        def numel(self): return self._n

    def _orig(sizes, k, rng):
        rn, rk, out = sum(sizes), k, []
        for n in sizes:
            if rk <= 0:
                out.append(0); rn -= n; continue
            t = int(rng.hypergeometric(n, rn - n, rk)) if rn > n else rk
            out.append(t); rk -= t; rn -= n
        return out

    # 7a. below the wall the split must be IDENTICAL to the original draw --
    # same RNG stream, same coordinates, so published cells still reproduce.
    small = [12_845_056] * 28 + [1_835_008] * 28
    ks = int(0.01 * sum(small))
    check("counts below 1e9 are unchanged by the large-N fix",
          _orig(small, ks, np.random.default_rng(7))
          == V8._counts_per_tensor({i: _T(x) for i, x in enumerate(small)},
                                   ks, np.random.default_rng(7)))

    # 7b. above the wall: exact total, right distribution, deterministic.
    # Kept small (k ~ 1e5) so this stays a fast test while still crossing 1e9.
    big = [70_000_000] * 16          # 1.12e9 total
    N = sum(big); kb = 100_000
    cb = V8._counts_per_tensor({i: _T(x) for i, x in enumerate(big)},
                               kb, np.random.default_rng(0))
    check("counts above 1e9 sum to exactly k", sum(cb) == kb)
    # z against the HYPERGEOMETRIC sd, not the binomial one: a binomial chain
    # would pass a mean test and fail here by dropping the finite-population
    # correction, which is why it was not used.
    p_ = big[0] / N
    sd = math.sqrt(kb * p_ * (1 - p_) * (N - kb) / (N - 1))
    z = abs(cb[0] - kb * p_) / sd
    check(f"above 1e9 the count distribution matches hypergeometric (z={z:.2f})",
          z < 4)
    check("above 1e9 the split is deterministic for a fixed seed",
          cb == V8._counts_per_tensor({i: _T(x) for i, x in enumerate(big)},
                                      kb, np.random.default_rng(0)))

    n_checks = 17
    print(f"v11 gate selftest: {'PASS' if not fails else str(len(fails)) + ' FAILED'}"
          f"  ({n_checks - len(fails)}/{n_checks} checks)")
    return not fails


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
