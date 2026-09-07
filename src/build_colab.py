"""Assemble a single self-contained colab_t2t4.py from the EXACT tested sources.

Pulls function/constant source via inspect, concatenates in dependency order,
strips cross-module imports, and appends a Colab main that runs T2 (7-9B
removal, bf16) + T4 (27-72B profiles + designer, bf16<=32B / 8-bit>=70B).
Guarantees the shipped code is byte-identical to what was validated here.
"""
import inspect, textwrap
import common, edit, probes, elicit, elicit_gen, crows_axes, bbq_axes, collateral

OUT = "/home/jovyan/Blind_spot_spb/colab_t2t4.py"

HEADER = '''"""
Blind-spot study — T2 (H1 at 7-9B) + T4 (27-72B profiles + designer) in ONE file.
Self-contained; generated from the validated pipeline. Run in Colab (96GB GPU).
"""
import os, re, json, time, math, random, itertools, argparse, functools
import numpy as np
import torch
import torch.nn.functional as F
import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
datasets.disable_progress_bars()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
RESULTS = os.environ.get("BS_OUT", "results")
os.makedirs(os.path.join(RESULTS, "t4", "profiles"), exist_ok=True)
os.makedirs(os.path.join(RESULTS, "t4", "designer"), exist_ok=True)
'''

# constants to inline verbatim (name -> module attr)
CONSTS = [
    (probes, ["PROBE_TEMPLATES", "PROBE_OCC", "INJECT_OCC", "CANDIDATE_AXES",
              "PROBE_NEG", "PROBE_POS", "INJECT_NEG", "INJECT_POS",
              "PROBE_VAL_TEMPLATES"]),
    (elicit, ["POOL_OCC_TEMPLATES", "POOL_VAL_TEMPLATES", "PRON_M", "PRON_F"]),
    (elicit_gen, ["SYS", "SD_BIASED_PRE", "SD_DEBIASED_PRE"]),
    (edit, ["DEFAULT_TARGETS", "TARGET_FALLBACKS", "MLP_FALLBACKS"]),
    (crows_axes, ["CROWS_TYPES", "AXIS_PREFIX"]),
    (bbq_axes, ["BBQ_CATS"]),
    (collateral, ["HEADING_RE", "LETTERS", "MMLU_SYS"]),
]

# functions to inline verbatim (module, name). Order = dependency order.
FUNCS = [
    (common, "cont_loglik"), (common, "seq_loglik"), (common, "chat_prompt"),
    (common, "mcq_logprobs"), (common, "_letter_ids"), (common, "rotations"),
    (edit, "_batches"), (edit, "resolve_targets"), (edit, "train_task_vector"), (edit, "contrast"),
    (edit, "binarize"), (edit, "_resolve"), (edit, "apply_edit"), (edit, "merge_dw"),
    (collateral, "load_mmlu"), (collateral, "mmlu_predict"), (collateral, "eval_mmlu"),
    (collateral, "load_wikitext"), (collateral, "eval_perplexity"),
]


def const_src(mod, name):
    val = getattr(mod, name)
    return f"{name} = {val!r}\n"


def fn_src(mod, name):
    s = inspect.getsource(getattr(mod, name))
    # strip local 'from common import ...' / 'import ...' lines inside functions
    lines = []
    for ln in s.splitlines():
        st = ln.strip()
        if st.startswith("from common import") or st.startswith("from edit import") \
           or st.startswith("import common") or st.startswith("from probes import") \
           or st.startswith("from elicit import") or st.startswith("import crows_axes") \
           or st.startswith("import bbq_axes") or st.startswith("from analyze"):
            continue
        lines.append(ln)
    return "\n".join(lines) + "\n"


def main():
    parts = [HEADER, "\n# ===== constants =====\n"]
    for mod, names in CONSTS:
        for n in names:
            parts.append(const_src(mod, n))
    parts.append("\n# ===== core functions (verbatim) =====\n")
    for mod, n in FUNCS:
        parts.append(fn_src(mod, n) + "\n")
    # the hand-written axis/profile/elicit/eval glue + main goes in a sibling file
    with open("/home/jovyan/Blind_spot_spb/src/_colab_glue.py") as f:
        parts.append("\n# ===== axes, profiles, elicitation, T2/T4 drivers =====\n")
        parts.append(f.read())
    open(OUT, "w").write("".join(parts))
    print("wrote", OUT, "(%d lines)" % sum(p.count("\n") for p in parts))


if __name__ == "__main__":
    main()
