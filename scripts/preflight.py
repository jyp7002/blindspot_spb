#!/usr/bin/env python3
"""Preflight — fail on the ground, not after an hour of GPU time.

WHY. This project has lost its GPU stack twice (PREREGISTRATION.md L545-547 and
L1083-1085 both record reinstalling transformers/peft/datasets), and
run_ws2_determinism.py attributes a cross-panel MMLU divergence to a
`transformers` rebuild. On a rented node, discovering either of those after the
model has downloaded is the expensive way to find out.

Checks, in the order they cost:
  1. interpreter + package versions against the pins   (instant)
  2. gate and config self-tests                        (instant)
  3. GPU count / VRAM vs the config's largest cell     (seconds)
  4. HF cache: is every checkpoint the config needs already local?  (seconds)
  5. writable results dir + free disk                  (instant)

Exit 0 = safe to launch. Non-zero = do not launch.

  python3 scripts/preflight.py
  python3 scripts/preflight.py --config configs/v11/big_v11.yaml
  python3 scripts/preflight.py --no-gpu --no-cache      # build-time / CI use
"""
import argparse
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# From requirements-run.txt. A mismatch is a WARN for analysis-only work and a
# FAIL when a GPU panel is about to run, because the panel's numbers must be
# comparable to the published ones.
PINS = {"torch": "2.11.0", "transformers": "4.57.1", "peft": "0.20.0"}

# Rough bf16 weight footprint, GiB, for the VRAM advisory. Deliberately
# conservative: excludes activations, optimizer state and the LoRA graph.
PARAM_GIB = {
    "google/gemma-2-2b-it": 5.2, "Qwen/Qwen2.5-3B-Instruct": 5.8,
    "meta-llama/Llama-3.2-3B-Instruct": 6.4,
    "microsoft/Phi-3.5-mini-instruct": 7.2,
    "Qwen/Qwen2.5-7B-Instruct": 14.2, "meta-llama/Llama-3.1-8B-Instruct": 15.0,
    "google/gemma-2-9b-it": 18.5, "google/gemma-2-27b-it": 50.7,
    "Qwen/Qwen2.5-32B-Instruct": 61.1,
    "meta-llama/Llama-3.1-70B-Instruct": 131.5,
}

OK, WARN, FAIL = "ok", "warn", "FAIL"
_results = []


def say(status, msg, detail=""):
    _results.append((status, msg))
    mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
    print(f"[{mark}] {msg}" + (f"\n         {detail}" if detail else ""))


# ------------------------------------------------------------------ checks --
def check_versions(strict, cfg_kind=None):
    if sys.version_info < (3, 9):
        say(FAIL, f"python {sys.version_info.major}.{sys.version_info.minor} "
                  "is below the 3.9 floor")
    else:
        say(OK, f"python {sys.version_info.major}.{sys.version_info.minor}")

    import importlib
    for mod, want in PINS.items():
        try:
            m = importlib.import_module(mod)
            got = getattr(m, "__version__", "?").split("+")[0]
            if got == want:
                say(OK, f"{mod} {got}")
            else:
                say(FAIL if strict else WARN,
                    f"{mod} {got} != pinned {want}",
                    "run_ws2_determinism attributes a cross-panel MMLU "
                    "divergence to a transformers rebuild; a panel run under a "
                    "different stack is not comparable to the published ones.")
        except ImportError:
            say(FAIL if strict else WARN, f"{mod} MISSING",
                "pip install -r requirements-run.txt")

    # IFEval's checkers, needed only by the ifeval panel. lm-eval does not
    # depend on them, and their absence is SILENT: run_ins.ifeval() catches the
    # ImportError and returns None, so the panel finishes with no measurement.
    # v12: the frontier panel runs IFEval too (every method, seed 0), and its
    # SentenceDebias arm needs sklearn's PCA.
    if cfg_kind in ("ifeval", "frontier"):
        for mod in ("lm_eval", "langdetect", "immutabledict", "nltk") + \
                (("sklearn",) if cfg_kind == "frontier" else ()):
            try:
                importlib.import_module(mod)
                say(OK, f"{mod} present (IFEval checker)")
            except ImportError:
                say(FAIL, f"{mod} MISSING — IFEval would silently score nothing",
                    "pip install -r requirements-run.txt")

    for mod in ("numpy", "yaml"):
        try:
            importlib.import_module(mod)
            say(OK, f"{mod} present")
        except ImportError:
            say(FAIL, f"{mod} MISSING — pip install -r requirements-analysis.txt")


def check_selftests():
    for name, path in (("collateral gate", "src/v9_gate.py"),
                       ("panel/config", "src/v11_panel.py"),
                       ("gate item-count regression", "src/v11_selftest.py")):
        p = subprocess.run([sys.executable, os.path.join(REPO, path)],
                           capture_output=True, text=True)
        if p.returncode == 0:
            say(OK, f"{name} selftest")
        else:
            say(FAIL, f"{name} selftest FAILED",
                (p.stdout + p.stderr).strip().splitlines()[-1] if
                (p.stdout + p.stderr).strip() else "")


def check_gpu(cfg):
    try:
        import torch
    except ImportError:
        say(WARN, "torch missing — skipping GPU check")
        return
    if not torch.cuda.is_available():
        say(FAIL, "no CUDA device visible")
        return
    n = torch.cuda.device_count()
    caps = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        caps.append(p.total_memory / 1024 ** 3)
        say(OK, f"gpu{i}: {p.name}, {caps[i]:.0f} GiB")

    if not cfg:
        return
    biggest, need = None, 0.0
    for cell in cfg["cells"]:
        g = PARAM_GIB.get(cell["hf"], 0.0)
        if cell.get("eightbit"):
            g /= 2
        if g > need:
            biggest, need = cell["hf"], g
    if not biggest:
        return
    # Training overhead is NOT proportional to model size here. The trainable
    # part is a rank-16 LoRA on the attention projections with gradient
    # checkpointing on, so optimizer state is tiny and activations are bounded
    # by batch and sequence, not by parameter count. A flat allowance is a much
    # better estimate than a percentage: at 70B a 1.5x rule demands ~196 GiB and
    # rules out hardware that would in fact work.
    #
    # colab_t2t4.load_model uses dispatch=False for trainable targets, so the
    # model must fit on ONE device: compare against the largest card, not the sum.
    TRAIN_OVERHEAD_GIB = 16
    total = need + TRAIN_OVERHEAD_GIB
    largest = max(caps) if caps else 0
    if largest >= total:
        say(OK, f"largest cell {biggest} ~{need:.0f} GiB weights + "
                f"{TRAIN_OVERHEAD_GIB} GiB training fits on a {largest:.0f} GiB card")
    else:
        say(FAIL, f"largest cell {biggest} needs ~{total:.0f} GiB on ONE "
                  f"device ({need:.0f} weights + {TRAIN_OVERHEAD_GIB} training); "
                  f"largest card is {largest:.0f} GiB",
            "trainable targets load with dispatch=False (colab_t2t4.load_model): "
            "accelerate's device_map hooks break training, so sharding across "
            "cards is not available. Use a bigger card or drop the cell.")


def check_cache(cfg):
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub = os.path.join(home, "hub")
    if not os.path.isdir(hub):
        say(WARN, f"no HF cache at {hub}",
            "every checkpoint will download; on a metered node that is the "
            "single biggest avoidable cost. Mount the cache instead.")
        return
    say(OK, f"HF cache at {hub}")
    if not cfg:
        return
    missing = []
    for cell in cfg["cells"]:
        d = "models--" + cell["hf"].replace("/", "--")
        if not os.path.isdir(os.path.join(hub, d)):
            missing.append(cell["hf"])
    if missing:
        say(WARN, f"{len(missing)} checkpoint(s) not cached",
            ", ".join(sorted(set(missing))) +
            "\n         gated repos (gemma, llama) also need HF_TOKEN.")
    else:
        say(OK, "every checkpoint in the config is cached")

    if not (os.environ.get("HF_TOKEN") or
            os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        gated = [c["hf"] for c in cfg["cells"]
                 if c["hf"].split("/")[0] in ("google", "meta-llama")]
        if gated and missing:
            say(FAIL, "HF_TOKEN unset but gated checkpoints are uncached",
                ", ".join(sorted(set(gated))))


def check_disk(cfg):
    out = os.environ.get("BS_OUT", os.path.join(REPO, "results"))
    os.makedirs(out, exist_ok=True)
    if not os.access(out, os.W_OK):
        say(FAIL, f"results dir not writable: {out}")
        return
    free = shutil.disk_usage(out).free / 1024 ** 3
    # A DEC unit writes one support dump (~tens of MB) plus jsonl rows.
    need = 5 + 0.2 * (len(cfg["cells"]) * len(cfg.get("seeds", [0])) if cfg else 0)
    say(OK if free > need else FAIL,
        f"{free:.0f} GiB free at {out} (need ~{need:.0f})")


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="panel config to check against")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--strict-versions", action="store_true",
                    help="treat a version mismatch as fatal (default when "
                         "--config is given, since a panel is about to run)")
    a = ap.parse_args()

    cfg = None
    if a.config:
        import v11_panel
        cfg = v11_panel.load(a.config)

    print(f"preflight: repo={REPO}"
          + (f"  config={a.config} ({len(cfg['cells'])} cells)" if cfg else ""))
    print("-" * 70)

    check_versions(strict=a.strict_versions or bool(a.config),
                   cfg_kind=(cfg or {}).get('kind'))
    check_selftests()
    if not a.no_gpu:
        check_gpu(cfg)
    if not a.no_cache:
        check_cache(cfg)
    check_disk(cfg)

    print("-" * 70)
    nf = sum(1 for s, _ in _results if s == FAIL)
    nw = sum(1 for s, _ in _results if s == WARN)
    print(f"PREFLIGHT {'FAIL' if nf else 'PASS'}: "
          f"{nf} blocking, {nw} advisory, {len(_results)} checks")
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
