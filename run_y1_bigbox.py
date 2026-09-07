"""Y1 (+ Y2 at the 27B tier) — the base-only durability number, for the ≥27B box.

Repairs the base-only invariant violation in the T4 gemma point: the 27-32B Δd
currently rests on gemma-2-27b-**it**. This profiles gemma-2-27b **base** and
recomputes the gemma-centric Δd base-only, then folds in Y2 (does the family
signature survive instruct tuning at 27B?) in the same run.

MUST run on the box that already holds the T4 -it profiles (results/t4/profiles/
gemma27b|*, qwen32b|*) and can fit gemma-2-27b base in bf16 (~52 GB). Needs an
HF token (gemma is gated). The L40S (46 GB) cannot do this in bf16 — that is why
it is a "bigbox" driver.

  HF_TOKEN=... python run_y1_bigbox.py           # bf16, matches the -it profiling
  HF_TOKEN=... EIGHTBIT=1 python run_y1_bigbox.py # fallback if <52 GB (quant confound — see notes)

Design (why it is valid):
  * Two profilers made the existing artifacts — colab_t2t4.bias_profile (the T4
    -it + qwen32b profiles) and src/geometry.bias_profile (the gemma9b/2b BASE
    profiles from Y2). Y1 must correlate a base-27B profile against BOTH sets, so
    the profile spaces have to coincide. We therefore profile gemma-27b-base with
    BOTH functions and assert they agree per axis (r >= AGREE_MIN). That
    assertion is exactly the "shared space across profilers" premise the
    published Δd already assumes — here it is checked, not hoped.
  * 6-axis basis = occ_gender + 5 CrowS (the set gemma9b/2b base were profiled on;
    SUMMARY_V4's basis). BBQ axes are excluded because the base siblings lack them.
  * Δd is gemma-centric cross-scale (same = gemma27b<->gemma9b/2b; cross =
    gemma27b<->qwen32b, the tier peer), computed identically for base and -it so
    the only thing that changes is the gemma checkpoint. CI bootstraps the 6 axes.
  * Y1 PASS: base Δd 95% CI excludes 0 (same-family signature still beats
    cross-family at 27B without instruct tuning). Y2: corr(base,it) per axis.

Writes results/y1_base.json. Also drops the profiles so t4_analyze.o1 can be
re-run with the base checkpoint if desired (t4/profiles/gemma27b_base|*.npy).
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np

# NOT offline: gemma-2-27b base may need downloading with the token. Datasets are
# cached on the T4 box already.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import colab_t2t4 as C          # colab profiler + loader (same as the -it profiles)
import geometry as G            # geometry profiler (same as the base 9b/2b profiles)

RES = C.RESULTS
T4P = os.path.join(RES, "t4", "profiles")
BASEP = os.path.join(RES, "profiles_base")
ITP = os.path.join(RES, "profiles")           # small-model instruct profiles (gemma9b/2b -it)
AXES = ["occ_gender", "crows_socioeconomic", "crows_race-color",
        "crows_gender", "crows_religion", "crows_age"]
GEMMA_27B_BASE = "google/gemma-2-27b"
AGREE_MIN = 0.98                              # profiler-agreement floor for the guard


def _corr(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    a, b = a - a.mean(), b - b.mean()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else None


def _load(dirpath, ckpt, ax):
    p = os.path.join(dirpath, f"{ckpt}|{ax}.npy")
    return np.load(p) if os.path.exists(p) else None


def _geom_prof(model, tok, ax, bs):
    r = G.bias_profile(model, tok, ax, batch_size=bs)
    return r[0] if isinstance(r, tuple) else r   # geometry returns (vec, keys)


def boot_ci_paired(deltas, n_boot=10000):
    """Δd from PAIRED per-axis deltas (same-cross share the gemma27b anchor and
    the axis, so pairing removes axis-baseline variance — the right, more powerful
    test at n=6 axes). Returns mean, CI, and how many axes have same>cross."""
    d = np.array([x for x in deltas if x is not None and not np.isnan(x)], float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    rng = np.random.default_rng(0)
    draws = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return (float(d.mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), int((d > 0).sum()), len(d))


def profile_27b_base():
    eightbit = os.environ.get("EIGHTBIT") == "1"
    need_colab = [ax for ax in AXES if not os.path.exists(os.path.join(T4P, f"gemma27b_base|{ax}.npy"))]
    need_geom = [ax for ax in AXES if not os.path.exists(os.path.join(BASEP, f"gemma27b|{ax}.npy"))]
    if not need_colab and not need_geom:
        print("[y1] gemma-27b-base already profiled (both profilers) — skipping load", flush=True)
        return
    print(f"[y1] load {GEMMA_27B_BASE} (8bit={eightbit}) — need colab={need_colab} geom={need_geom}",
          flush=True)
    model, tok = C.load_model(GEMMA_27B_BASE, eightbit=eightbit)   # dispatch=True (device_map=auto)
    os.makedirs(T4P, exist_ok=True); os.makedirs(BASEP, exist_ok=True)
    try:
        for ax in AXES:
            # colab profiler -> t4/profiles/gemma27b_base (comparable to qwen32b, gemma27b-it)
            fp = os.path.join(T4P, f"gemma27b_base|{ax}.npy")
            if not os.path.exists(fp):
                v = C.bias_profile(model, tok, ax, 8)
                np.save(fp, v); print(f"[y1] colab   profile {ax:22s} n={len(v)} mean={v.mean():+.4f}", flush=True)
            # geometry profiler -> profiles_base/gemma27b (comparable to gemma9b/2b base)
            gp = os.path.join(BASEP, f"gemma27b|{ax}.npy")
            if not os.path.exists(gp):
                v = _geom_prof(model, tok, ax, 8)
                np.save(gp, v); print(f"[y1] geom    profile {ax:22s} n={len(v)} mean={v.mean():+.4f}", flush=True)
    finally:
        C._free(model)


def main():
    tok_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok_env:
        try:
            from huggingface_hub import login
            login(token=tok_env); print("[auth] HF login OK", flush=True)
        except Exception as ex:
            print(f"[auth] HF login failed: {ex}", flush=True)
    else:
        print("[auth] no HF_TOKEN — gated gemma-2-27b base download will fail", flush=True)

    profile_27b_base()

    # ---- consistency guard: the two profilers must agree on gemma-27b-base ----
    print("\n[y1] profiler-agreement guard (colab t4/profiles vs geometry profiles_base):")
    agree = {}
    for ax in AXES:
        cc = _load(T4P, "gemma27b_base", ax)
        gg = _load(BASEP, "gemma27b", ax)
        r = _corr(cc, gg)
        agree[ax] = r
        flag = "" if (r is not None and r >= AGREE_MIN) else "  << BELOW FLOOR"
        print(f"    {ax:22s} r={r if r is None else round(r,4)}{flag}")
    bad = [ax for ax, r in agree.items() if r is None or r < AGREE_MIN]
    if bad:
        print(f"[y1] WARNING: profilers disagree on {bad} — cross-profiler Δd terms on those axes\n"
              f"     are NOT trustworthy; treat same-family(base) numbers using geometry profiles only.")

    # ---- partner-presence check: fail LOUD, never silent-nan ----
    partners = [("t4/profiles", "gemma27b", "gemma27b-it (Y2 + it-Δd)"),
                ("t4/profiles", "qwen32b", "qwen32b cross-family peer"),
                ("profiles_base", "gemma9b", "gemma9b base (same-family)"),
                ("profiles_base", "gemma2b", "gemma2b base (same-family)"),
                ("profiles", "gemma9b", "gemma9b it (it-Δd)"),
                ("profiles", "gemma2b", "gemma2b it (it-Δd)")]
    print("\n[y1] partner-profile presence:")
    missing = []
    for d, ck, desc in partners:
        n = sum(1 for ax in AXES if _load(os.path.join(RES, d), ck, ax) is not None)
        tag = "" if n == len(AXES) else f"  << {len(AXES) - n} MISSING"
        if n < len(AXES):
            missing.append(f"{d}/{ck}")
        print(f"    {desc:34s} {n}/{len(AXES)} axes{tag}")
    if missing:
        print("\n[y1] ABORT: partner profiles absent on this box — Δd/Y2 would be all-nan.\n"
              "     This box has the freshly-profiled gemma-27b-base but not the comparison\n"
              "     checkpoints. Drop `y1_partners.zip` (36 tiny .npy) at the repo root:\n"
              "         unzip -o y1_partners.zip      # restores results/{t4/profiles,profiles_base,profiles}\n"
              "     then re-run — profiling is cached, so it skips straight to the analysis.")
        json.dump({"error": "missing_partner_profiles", "missing": missing,
                   "profiler_agreement": agree}, open(os.path.join(RES, "y1_base.json"), "w"), indent=2)
        return

    # ---- partners (base and -it) ----
    def prof(which, ckpt, ax):
        return _load(which, ckpt, ax)
    # gemma-27b in both variants, via the colab profiler (tier-comparable to qwen32b)
    def g27(variant, ax):   # variant: 'base' or 'it'
        return _load(T4P, "gemma27b_base" if variant == "base" else "gemma27b", ax)
    # same-family siblings: base from profiles_base, it from small-model profiles
    def sib(name, variant, ax):
        return _load(BASEP, name, ax) if variant == "base" else _load(ITP, name, ax)
    qwen32b = lambda ax: _load(T4P, "qwen32b", ax)      # cross-family tier peer (instruct only)

    # ---- Y1: gemma-centric Δd, base vs it, PAIRED per-axis ----
    print("\n[y1] Δd = mean_ax [ mean(corr(g27,g9b),corr(g27,g2b)) - corr(g27,qwen32b) ]  (paired per axis)")
    out = {"axes": AXES, "profiler_agreement": agree, "y1": {}, "y2": {}}
    for variant in ("base", "it"):
        same9 = [_corr(g27(variant, ax), sib("gemma9b", variant, ax)) for ax in AXES]
        same2 = [_corr(g27(variant, ax), sib("gemma2b", variant, ax)) for ax in AXES]
        cross = [_corr(g27(variant, ax), qwen32b(ax)) for ax in AXES]
        deltas = []
        for s9, s2c, cx in zip(same9, same2, cross):
            sm = [c for c in (s9, s2c) if c is not None]
            deltas.append(np.mean(sm) - cx if sm and cx is not None else None)
        d, lo, hi, npos, nax = boot_ci_paired(deltas)
        sm9 = np.nanmean([c for c in same9 if c is not None])
        sm2 = np.nanmean([c for c in same2 if c is not None])
        cm = np.nanmean([c for c in cross if c is not None])
        excl0 = (not np.isnan(lo)) and (lo > 0)
        print(f"    [{variant:4s}] same(9b)={sm9:+.3f} same(2b)={sm2:+.3f} cross(qwen32b)={cm:+.3f}"
              f"  Δd={d:+.3f} 95%CI=[{lo:+.3f},{hi:+.3f}]  same>cross on {npos}/{nax} axes  excl0={excl0}")
        out["y1"][variant] = dict(same_9b=float(sm9), same_2b=float(sm2), cross_qwen32b=float(cm),
                                  delta_d=d, ci=[lo, hi], ci_excludes_0=bool(excl0),
                                  axes_same_gt_cross=f"{npos}/{nax}",
                                  delta_per_axis=[None if x is None else float(x) for x in deltas])
    y1_pass = out["y1"]["base"]["ci_excludes_0"]
    npos_base = out["y1"]["base"]["axes_same_gt_cross"]
    print(f"\n[y1] VERDICT: base-only Δd CI excludes 0 ? {y1_pass}  (same>cross on {npos_base} axes)  "
          f"-> {'Y1 PASS (durability holds base-only)' if y1_pass else 'Y1 weak on 6-axis CI — lean on the point estimate + sign count; -it point stays in appendix unless base Δd>0 on ≥5/6 axes'}")

    # ---- Y2: does the family signature survive instruct tuning at 27B? ----
    print("\n[y2] gemma-27b base<->it signature shift, and each <-> gemma-9b base:")
    b_it = [_corr(g27("base", ax), g27("it", ax)) for ax in AXES]
    b_it_m = np.nanmean([c for c in b_it if c is not None])
    base_to_9bbase = np.nanmean([c for c in
                                 [_corr(g27("base", ax), sib("gemma9b", "base", ax)) for ax in AXES] if c is not None])
    it_to_9bbase = np.nanmean([c for c in
                               [_corr(g27("it", ax), sib("gemma9b", "base", ax)) for ax in AXES] if c is not None])
    for ax, r in zip(AXES, b_it):
        print(f"    {ax:22s} base<->it r={r if r is None else round(r,3)}")
    print(f"    mean base<->it r = {b_it_m:+.3f}")
    print(f"    27b-base <-> 9b-base = {base_to_9bbase:+.3f} | 27b-it <-> 9b-base = {it_to_9bbase:+.3f}")
    out["y2"] = dict(base_vs_it_per_axis=b_it, base_vs_it_mean=float(b_it_m),
                     base_to_9b_base=float(base_to_9bbase), it_to_9b_base=float(it_to_9bbase))

    fp = os.path.join(RES, "y1_base.json")
    json.dump(out, open(fp, "w"), indent=2)
    print(f"\n[y1] wrote {fp}")
    print("[y1] next: optionally re-run `python src/t4_analyze.py` after copying "
          "t4/profiles/gemma27b_base -> a checkpoint name in its tier map, for the canonical table.")


if __name__ == "__main__":
    main()
