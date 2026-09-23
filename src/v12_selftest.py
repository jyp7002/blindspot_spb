"""v12 — end-to-end pipeline test with a fake model. No GPU, no downloads.

Drives the REAL code path  config -> v11_panel.expand -> run_unit._run_opsel ->
run_opsel.main -> v12_pstar fit/predict/validate  with real torch edit
construction (colab_t2t4.binarize, v8_edits.build_edit, apply_edit) and a
stand-in model whose "bias" and "collateral" are simple functions of the edit.
What it pins down is the protocol, not any number:

  1. the calibration panel completes, and v11_panel.done() agrees
  2. a held-out full run WITHOUT a prediction is refused (HeldOutViolation)
  3. `predict` works after phase A, and REFUSES once a held-out curve row exists
  4. phase B runs the predicted p* point and records the prediction's sha
  5. a retrain that does not reproduce the phase-A ΔW is refused
  6. resume: re-running a complete unit does no work
  7. calibration and evaluation items are disjoint; MMLU-1000 is a superset
  8. adaptive α* never exceeds the first calibration failure
  9. `validate` produces a record for every held-out cell

Run: python3 src/v12_selftest.py
"""
import json
import os
import shutil
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = tempfile.mkdtemp(prefix="v12st_")
os.environ["BS_OUT"] = TMP                       # before any project import
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


for _n in ("transformers", "peft", "datasets"):
    if _n not in sys.modules:
        _stub(_n, AutoModelForCausalLM=object, AutoTokenizer=object,
              BitsAndBytesConfig=object, LoraConfig=object,
              get_peft_model=lambda *a, **k: None,
              disable_progress_bars=lambda: None,
              load_dataset=lambda *a, **k: {})

import torch                                      # noqa: E402
import colab_t2t4 as C                            # noqa: E402

LAYERS, DIM = 4, 48
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
KEYS = [f"model.layers.{L}.self_attn.{p}" for L in range(LAYERS) for p in PROJ]
DRIFT = {"on": False}


# ------------------------------------------------------------- the fake ----
class _Lin(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(DIM, DIM, generator=g) * 0.02)


class FakeModel(torch.nn.Module):
    def __init__(self, hf):
        super().__init__()
        import zlib
        g = torch.Generator().manual_seed(zlib.crc32(hf.encode()))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for _L in range(LAYERS):
            blk = torch.nn.Module()
            blk.self_attn = torch.nn.Module()
            for p in PROJ:
                setattr(blk.self_attn, p, _Lin(g))
            self.model.layers.append(blk)
        self.W0 = {k: C._resolve(self, k).weight.detach().clone() for k in KEYS}
        # the "bias direction": heavy-tailed so magnitude selection matters
        # (Student-t with 2 dof from seeded normals: a real checkpoint is the
        # same on every load, and the phase-A/B fingerprint check relies on it)
        self.H = {}
        for k in KEYS:
            z = torch.randn(DIM, DIM, generator=g)
            chi2 = (torch.randn(2, DIM, DIM, generator=g) ** 2).sum(0)
            self.H[k] = z / torch.sqrt(chi2 / 2.0)
        self.hf = hf


def _delta(model):
    return {k: C._resolve(model, k).weight.detach() - model.W0[k] for k in KEYS}


def _state(model):
    d = _delta(model)
    align = sum(float((-d[k] * model.H[k]).sum()) for k in KEYS)
    size = float(torch.sqrt(sum((d[k] ** 2).sum() for k in KEYS)))
    return align, size


def fake_train(model, tok, texts, *, seed=0, **k):
    """Biased and debiased 'task vectors' whose contrast aligns with H."""
    g = torch.Generator().manual_seed(seed * 7 + (1 if texts[0] == "b" else 0)
                                      + (1000 if DRIFT["on"] else 0))
    sgn = 1.0 if texts[0] == "b" else 0.0
    dw = {kk: 1e-4 * sgn * model.H[kk] + 5e-5 * torch.randn(DIM, DIM, generator=g)
          for kk in KEYS}
    return dw, 1.0


def _items(model, n, base):
    align, size = _state(model)
    drop = int(min(n, (size * 9) ** 2 * n / 200))
    return (base - drop) / n


def fake_fast_eval(model, tok, axis, mmlu_rows, wt, bs=6):
    align, size = _state(model)
    skew = 0.5 - 0.005 * align + 0.3 * size ** 2
    return dict(skew=skew, mmlu_acc=_items(model, len(mmlu_rows), int(0.6 * len(mmlu_rows))),
                ppl=10.0 * (1 + size ** 2), n_items=len(mmlu_rows))


def fake_eval_mmlu(model, tok, rows, batch_size=6, n_rot=2):
    return dict(mmlu_acc=_items(model, len(rows), int(0.6 * len(rows))), n=len(rows))


def fake_ppl(model, tok, texts, batch_size=1, max_length=512):
    align, size = _state(model)
    return dict(ppl=10.0 * (1 + size ** 2) * (1.01 if texts[0].startswith("v") else 1))


def fake_mmlu(n=None, seed=0):
    return [dict(question=f"test-q{i}", options=list("abcd"), gold=0, subject="s")
            for i in range(n)]


def fake_wikitext(n_chunks=20, seed=0, split="test", **k):
    return [f"{split[0]}-chunk{i}" for i in range(n_chunks)]


def install():
    C.load_model = lambda hf, eightbit=False, dispatch=True: (FakeModel(hf), None)
    C._free = lambda *a: None
    C.train_task_vector = fake_train
    C.fast_eval = fake_fast_eval
    C.eval_mmlu = fake_eval_mmlu
    C.eval_perplexity = fake_ppl
    C.load_mmlu = fake_mmlu
    C.load_wikitext = fake_wikitext
    cdir = os.path.join(TMP, "corpora")
    os.makedirs(cdir, exist_ok=True)

    def corpus_fp(dn, ax, s):
        fp = os.path.join(cdir, f"{dn}|{ax}|s{s}.json")
        if not os.path.exists(fp):
            json.dump(dict(biased=["b"] * 4, debiased=["d"] * 4), open(fp, "w"))
        return fp
    C._t2x_corpus_fp = corpus_fp
    import run_opsel
    run_opsel.load_mmlu_calib = lambda n, seed=0: [
        dict(question=f"val-q{i}", options=list("abcd"), gold=0, subject="s")
        for i in range(n)]
    run_opsel.C = C


# --------------------------------------------------------------- driver ----
def _cfg(name, body):
    fp = os.path.join(TMP, name)
    open(fp, "w").write(body)
    return fp


CAL = """
panel: stcal
kind: opsel
out_panel: stcal
seeds: [0, 1]
sparsities: [0.0, 0.5, 0.9, 0.99, 0.999]
extra_variants: [fp, "C-a@0.01"]
adaptive_variants: [fp, s0.0, s0.99, "C-a@0.01"]
mmlu1k_variants: [s0.99]
mmlu1k_all_alphas: true
alpha_ladder: [1, 2, 4, 8, 16, 32, 64]
refine_steps: 2
cells:
  - {target: t1, hf: fake/t1, designer: t1, axis: occ_gender}
  - {target: t2, hf: fake/t2, designer: t2, axis: occ_gender}
  - {target: t3, hf: fake/t3, designer: t3, axis: occ_gender}
  - {target: t4, hf: fake/t4, designer: t4, axis: bbq_Age}
"""

BIG = """
panel: stbig
kind: opsel
out_panel: stbig
seeds: [0, 1]
sparsities: [0.0, 0.5, 0.9, 0.99, 0.999]
extra_variants: [fp, pstar]
adaptive_variants: [s0.99, pstar]
mmlu1k_variants: [pstar]
alpha_ladder: [1, 2, 4, 8, 16, 32, 64]
refine_steps: 2
pstar_file: PRED
cells:
  - {target: big1, hf: fake/big1, designer: big1, axis: occ_gender}
"""


def main():
    install()
    import v11_panel
    import run_unit
    import run_opsel
    import v12_pstar as P
    import v12_opsel as S
    v11_panel.RESULTS = TMP
    P.CAL_PANEL, P.HELD_PANEL = "stcal", "stbig"
    P.OUT_DIR = os.path.join(TMP, "v12")
    P.PRED_FP = os.path.join(P.OUT_DIR, "pstar_prediction.json")
    P.FIT_FP = os.path.join(P.OUT_DIR, "pstar_fit.json")
    P.VAL_FP = os.path.join(P.OUT_DIR, "pstar_validation.json")

    fails = []

    def check(name, cond):
        if not cond:
            fails.append(name)
            print(f"  FAIL {name}")

    def run(cfg, phase=None):
        c = v11_panel.load(cfg)
        if phase:
            c["phase"] = phase
        rcs = []
        for u in v11_panel.expand(c):
            if v11_panel.done(u):
                rcs.append("skip")
                continue
            run_unit._run_opsel(u)
            rcs.append(v11_panel.done(u))
        return rcs

    quiet = open(os.devnull, "w")
    real_stdout = sys.stdout
    try:
        sys.stdout = quiet
        # 1. calibration panel
        rcs = run(_cfg("cal.yaml", CAL))
        sys.stdout = real_stdout
        check("calibration units all complete", all(r is True for r in rcs))
        rem = P._rows("stcal", "removal.jsonl")
        check("calibration wrote fp, C-a, every sparsity",
              {"fp", "C-a@0.01", "s0.0", "s0.99", "s0.999"} <= {r["variant"] for r in rem})
        tr = P._rows("stcal", "alpha_trace.jsonl")
        check("calibration trace carries calib, eval and mmlu1k splits",
              {"calib", "eval", "mmlu1k"} <= {t["split"] for t in tr})
        # 8. α* never above the first calibration failure
        bad = 0
        for r in rem:
            ad = r.get("adaptive")
            if ad and ad.get("first_fail") and ad.get("alpha_star"):
                bad += ad["alpha_star"] >= ad["first_fail"]
        check("α* below first calibration failure", bad == 0)
        k1 = [r["mmlu1k_frozen"] for r in rem if r["variant"] == "s0.99"]
        check("MMLU-1000 argmax recorded", all(k and "argmax_1k_alpha" in k for k in k1))

        # 6. resume does nothing
        sys.stdout = quiet
        rcs = run(_cfg("cal.yaml", CAL))
        sys.stdout = real_stdout
        check("resume: complete panel re-run does no work", all(r == "skip" for r in rcs))

        # fit on calibration
        a = types.SimpleNamespace(rho=0.9, floor=0.0, margin=1.0, rule="mass",
                                  seeds=[0, 1], i_know=False)
        sys.stdout = quiet
        P.cmd_fit(a)
        sys.stdout = real_stdout
        check("fit written", os.path.exists(P.FIT_FP))

        big = _cfg("big.yaml", BIG.replace("PRED", P.PRED_FP))
        # 2. full phase without a prediction is refused
        try:
            sys.stdout = quiet
            run(big, "full")
            sys.stdout = real_stdout
            check("held-out full run without prediction refused", False)
        except S.HeldOutViolation:
            sys.stdout = real_stdout

        # phase A, then predict
        sys.stdout = quiet
        rcs = run(big, "geometry")
        P.cmd_predict(a)
        sys.stdout = real_stdout
        check("phase A complete", all(r in (True, "skip") for r in rcs))
        pred = json.load(open(P.PRED_FP))
        check("prediction has the held-out cell with both seed fingerprints",
              set(pred["cells"]["big1|occ_gender"]["geom_shas_by_seed"]) == {"0", "1"})

        # 5. a drifted retrain is refused (seed 0 only; restore after)
        DRIFT["on"] = True
        sys.stdout, real_stderr, sys.stderr = quiet, sys.stderr, quiet
        c = v11_panel.load(big)
        c["phase"] = "full"
        u0 = v11_panel.expand(c)[0]
        try:
            run_unit._run_opsel(u0)
            drift_refused = not v11_panel.done(u0)
        except S.HeldOutViolation:
            drift_refused = True
        sys.stdout, sys.stderr = real_stdout, real_stderr
        DRIFT["on"] = False
        check("retrain that does not reproduce phase A is refused", drift_refused)

        # 4. phase B
        sys.stdout = quiet
        rcs = run(big, "full")
        sys.stdout = real_stdout
        check("phase B complete", all(r is True for r in rcs))
        rows = [r for r in P._rows("stbig", "removal.jsonl") if r["variant"] == "pstar"]
        check("p* rows carry the prediction sha",
              rows and all(r["pstar_prediction_sha"] == S.file_sha(P.PRED_FP) for r in rows))
        check("p* row sparsity is 1 - p*",
              rows and all(abs(r["sparsity"] - (1 - r["p_star"])) < 1e-12 for r in rows))

        # 3. predict now refuses
        try:
            sys.stdout = quiet
            P.cmd_predict(a)
            sys.stdout = real_stdout
            check("predict refuses once held-out curve rows exist", False)
        except SystemExit:
            sys.stdout = real_stdout

        # 9. validate
        sys.stdout = quiet
        P.cmd_validate(a)
        sys.stdout = real_stdout
        val = json.load(open(P.VAL_FP))
        check("validation has the held-out cell",
              "retention_at_pstar" in val["cells"].get("big1|occ_gender", {}))

        # the analysis runs end to end on these panels
        import v12_analyze as A
        A.RESULTS, A.PANELS = TMP, ("stcal", "stbig", "none")
        sys.stdout = quiet
        rc = A.main([])
        sys.stdout = real_stdout
        summ = json.load(open(os.path.join(TMP, "v12", "opsel_summary.json")))
        check("analysis: every reading produced",
              rc == 0 and summ["joint_operating_point"]
              and summ["delta_selection"]["calibration"]["adaptive"]
              and summ["mmlu1k"]["calibration"]["units_with_full_1k_trace"] > 0)

        # frontier runner: edit arm, select + IFEval rows, done() agrees
        import run_frontier
        run_frontier.C = C
        FR = """
panel: stfr
kind: frontier
out_panel: stfr
seeds: [0, 1]
methods: [edit]
mmlu1k: true
ifeval_seeds: [0]
cells:
  - {target: t1, hf: fake/t1, designer: t1, axis: occ_gender}
"""
        c = v11_panel.load(_cfg("fr.yaml", FR))
        us = v11_panel.expand(c)
        sys.stdout, real_stderr, sys.stderr = quiet, sys.stderr, quiet
        for u in us:
            run_unit._run_frontier(u)
        sys.stdout, sys.stderr = real_stdout, real_stderr
        check("frontier units complete (select + ifeval rows)",
              all(v11_panel.done(u) for u in us))
        fr_rows = P._rows("stfr", "removal.jsonl")
        sel = [r for r in fr_rows if r.get("row") == "select"]
        check("frontier select rows carry the 1000-item re-selection",
              sel and all("removal_1k_gate" in r for r in sel))
        tr = P._rows("stfr", "alpha_trace.jsonl")
        check("frontier trace persists collateral per configuration",
              len(tr) == 8 and all("dmmlu_items" in t and "collateral_ok_1k" in t
                                   for t in tr))

        # 7. disjointness guard fires on overlap
        orig = run_opsel.load_mmlu_calib
        run_opsel.load_mmlu_calib = lambda n, seed=0: fake_mmlu(n)
        run_opsel.OUT = os.path.join(TMP, "overlap")
        run_opsel.PANEL = "overlap"
        run_opsel.PHASE = "full"
        run_opsel.CELLS = [("t9", "fake/t9", "t9", "occ_gender", None)]
        try:
            sys.stdout = quiet
            run_opsel.main()
            sys.stdout = real_stdout
            check("overlapping calibration/evaluation MMLU refused", False)
        except RuntimeError:
            sys.stdout = real_stdout
        run_opsel.load_mmlu_calib = orig
    finally:
        sys.stdout = real_stdout
        quiet.close()

    print(f"v12 pipeline selftest: {'PASS' if not fails else str(len(fails)) + ' FAILED'}"
          f"  (scratch {TMP})")
    if not fails:
        shutil.rmtree(TMP, ignore_errors=True)
    return not fails


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
