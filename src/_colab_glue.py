
# ---- occupation-gender axis (templated; the removable inherited axis) ----
def occ_profile(model, tok, batch_size=12):
    occ = {**INJECT_OCC, **PROBE_OCC}
    prefixes, conts, meta = [], [], []
    for o, g in occ.items():
        for ti, (pre, w_he, w_she, tail) in enumerate(PROBE_TEMPLATES):
            ctx = pre.format(occ=o)
            for w in (w_he, w_she):
                prefixes.append(ctx); conts.append(" " + w + " " + tail)
            meta.append((o, g, ti))
    ll = cont_loglik(model, tok, prefixes, conts, batch_size).numpy()
    vals = []
    for i in range(len(meta)):
        d = float(ll[2 * i + 1] - ll[2 * i])
        p_f = 1.0 / (1.0 + np.exp(-d)); g = meta[i][1]
        vals.append((p_f - 0.5) if g == "f" else (0.5 - p_f))
    return np.array(vals)

def occ_skew(model, tok, batch_size=12):
    occ = PROBE_OCC
    prefixes, conts, meta = [], [], []
    for o, g in occ.items():
        for (pre, w_he, w_she, tail) in PROBE_TEMPLATES:
            ctx = pre.format(occ=o)
            for w in (w_he, w_she):
                prefixes.append(ctx); conts.append(" " + w + " " + tail)
            meta.append((o, g))
    ll = cont_loglik(model, tok, prefixes, conts, batch_size).numpy()
    per = {}
    for i, (o, g) in enumerate(meta):
        d = float(ll[2 * i + 1] - ll[2 * i]); p_f = 1.0 / (1.0 + np.exp(-d))
        per.setdefault(o, []).append(p_f)
    congr = []
    for o, g in occ.items():
        m = float(np.mean(per[o])); congr.append(m if g == "f" else 1.0 - m)
    return 2.0 * float(np.mean(congr)) - 1.0

# occ minimal pairs for elicitation/edit (templated pool)
def occ_sd_items():
    items = []
    for o, g in INJECT_OCC.items():
        for t in POOL_OCC_TEMPLATES:
            s_f = t.format(occ=o, **PRON_F); s_m = t.format(occ=o, **PRON_M)
            c, i = (s_f, s_m) if g == "f" else (s_m, s_f)
            items.append((c, i, o))
    return items

# ---- CrowS group-vs-B (natural; NOT removable control) ----
def _crows_axis(axis, seed=0, probe_frac=0.5):
    cat = axis[len(AXIS_PREFIX):]
    d = load_dataset("nyu-mll/crows_pairs", revision="refs/convert/parquet")["test"]
    pairs = []
    for r in d:
        if CROWS_TYPES.get(r["bias_type"]) != cat: continue
        more, less = r["sent_more"], r["sent_less"]
        cong, incong = (more, less) if r["stereo_antistereo"] == 0 else (less, more)
        if cong and incong and cong != incong:
            pairs.append((cong.strip(), incong.strip(), f"{cat}:{len(pairs)}"))
    rng = np.random.default_rng(seed); idx = rng.permutation(len(pairs))
    cut = int(len(pairs) * probe_frac)
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]

# ---- BBQ group-vs-UNKNOWN (natural; REMOVABLE — repeated toward-unknown) ----
def _norm(s): return set(re.split(r"[-_\s]+", str(s).strip().lower())) - {""}
def _is_tgt(g, sg): return any(_norm(g) & _norm(x) for x in sg)
def _bbq_axis(axis, seed=0, probe_frac=0.5):
    cat = axis[len("bbq_"):]
    d = load_dataset("oskarvanderwal/bbq", cat)["test"]
    pairs = []
    for r in d:
        if r["context_condition"] != "ambig": continue
        ans = [r["ans0"], r["ans1"], r["ans2"]]
        info = [r["answer_info"][f"ans{i}"] for i in range(3)]
        groups = [x[1] if len(x) > 1 else "unknown" for x in info]
        sg = r["additional_metadata"]["stereotyped_groups"]
        unk = [i for i, g in enumerate(groups) if str(g).strip().lower() == "unknown"]
        tgt = [i for i, g in enumerate(groups) if str(g).strip().lower() != "unknown" and _is_tgt(g, sg)]
        if len(unk) != 1 or len(tgt) != 1 or r["question_polarity"] != "neg": continue
        stem = f"{r['context']} {r['question']}"
        pairs.append((f"{stem} {ans[tgt[0]]}.", f"{stem} {ans[unk[0]]}.", f"{cat}:{r['example_id']}"))
    rng = np.random.default_rng(seed); idx = rng.permutation(len(pairs))
    cut = int(len(pairs) * probe_frac)
    return [pairs[i] for i in idx[:cut]], [pairs[i] for i in idx[cut:]]

@functools.lru_cache(maxsize=64)
def load_axis(axis):
    if axis.startswith("crows_"): return _crows_axis(axis)
    if axis.startswith("bbq_"):   return _bbq_axis(axis)
    raise ValueError(axis)

def pair_profile(model, tok, axis, batch_size=12):
    probe, _e = load_axis(axis)
    texts = []
    for c, i, _k in probe: texts += [c, i]
    ll = seq_loglik(model, tok, texts, batch_size, normalise=True).numpy()
    return np.array([float(ll[2 * n] - ll[2 * n + 1]) for n in range(len(probe))])

def bias_profile(model, tok, axis, batch_size=12):
    if axis == "occ_gender": return occ_profile(model, tok, batch_size)
    return pair_profile(model, tok, axis, batch_size)

def axis_skew(model, tok, axis, batch_size=12):
    if axis == "occ_gender": return occ_skew(model, tok, batch_size)
    return float(bias_profile(model, tok, axis, batch_size).mean())

# ---- elicitation (Schick self-debias likelihood) + corpora ----
def _sd_items(axis):
    if axis == "occ_gender": return occ_sd_items()
    _p, edit_half = load_axis(axis); return list(edit_half)

def elicit_selfdebias(model, tok, axis, seed=0, batch_size=16):
    items = _sd_items(axis); random.Random(seed).shuffle(items)
    tkey = "occ_gender" if axis == "occ_gender" else "_crows"
    chosen, congr = {}, {}
    for cond, tmpl in (("biased", SD_BIASED_PRE[tkey]), ("debiased", SD_DEBIASED_PRE[tkey])):
        prefixes, conts = [], []
        for c, i, key in items:
            pre = chat_prompt(tok, tmpl.format(key=key), SYS)
            prefixes += [pre, pre]; conts += [c, i]
        ll = cont_loglik(model, tok, prefixes, conts, batch_size).numpy().reshape(-1, 2)
        pick_c = ll[:, 0] > ll[:, 1]
        chosen[cond] = [items[n][0] if pick_c[n] else items[n][1] for n in range(len(items))]
        congr[cond] = float(pick_c.mean())
    diag = dict(contrast_gap=congr["biased"] - congr["debiased"], n_items=len(items))
    return dict(biased=chosen["biased"], debiased=chosen["debiased"], diag=diag)

def exogenous_corpora(axis, seed=0):
    if axis == "occ_gender":
        items = occ_sd_items()
    else:
        _p, items = load_axis(axis)
    b = [c for c, _i, _k in items]; d = [i for _c, i, _k in items]
    idx = list(range(len(b))); random.Random(seed).shuffle(idx)
    return [b[i] for i in idx], [d[i] for i in idx]

# ---- collateral eval (occ_skew primary; mmlu+ppl budget) ----
def fast_eval(model, tok, axis, mmlu_rows, wt_texts, batch_size=24):
    sk = axis_skew(model, tok, axis, batch_size)
    m = eval_mmlu(model, tok, mmlu_rows, batch_size=batch_size, n_rot=2)["mmlu_acc"]
    p = eval_perplexity(model, tok, wt_texts, batch_size=max(batch_size // 8, 1))["ppl"]
    return dict(skew=sk, mmlu_acc=m, ppl=p)

def bias_reduction(pre, post): return abs(pre["skew"]) - abs(post["skew"])
def collateral_ok(pre, post, dmmlu=0.02, pplr=1.10):
    return (pre["mmlu_acc"] - post["mmlu_acc"]) <= dmmlu and post["ppl"] / pre["ppl"] <= pplr

# ---- loader (bf16 <=32B ; 8-bit >=60B) ----
def load_model(model_id, eightbit=False, dispatch=True):
    """dispatch=True -> device_map='auto' (accelerate shards/hooks; use for
    inference-only large T4 models). dispatch=False -> plain single-device
    placement (REQUIRED for trainable T2 targets: device_map's accelerate hooks
    break LoRA backprop on 7-9B models even when the model fits on one GPU)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = dict(low_cpu_mem_usage=True)
    if eightbit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["dtype"] = DTYPE
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dispatch:
        kw["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        if not eightbit:  # 8-bit models are already placed by bitsandbytes
            model = model.to(dev)
    model.eval()
    globals()["DEVICE"] = dev
    return model, tok

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ============================ T4 ============================
T4_PROFILE_AXES = ["occ_gender", "crows_socioeconomic", "crows_race-color",
                   "crows_gender", "crows_religion", "crows_age",
                   "bbq_Gender_identity", "bbq_Race_ethnicity", "bbq_Age", "bbq_Religion"]
T4_DESIGNER_AXES = ["occ_gender", "bbq_Age", "bbq_Religion", "crows_socioeconomic"]

def _free(*objs):
    import gc
    for o in objs:
        try: o.to("meta")
        except Exception: pass
        del o
    gc.collect(); torch.cuda.empty_cache()


def run_t4(model_id, name, eightbit, batch_size):
    print(f"[t4] load {name} ({model_id}) 8bit={eightbit}", flush=True)
    model, tok = load_model(model_id, eightbit)
    try:
        _run_t4_body(model, tok, model_id, name, batch_size)
    finally:
        _free(model)  # free even if the body aborts, so the next model has VRAM


def _run_t4_body(model, tok, model_id, name, batch_size):
    for ax in T4_PROFILE_AXES:
        out = os.path.join(RESULTS, "t4", "profiles", f"{name}|{ax}.npy")
        if os.path.exists(out): continue
        try:
            v = bias_profile(model, tok, ax, batch_size)
            np.save(out, v); print(f"[t4] {name} profile {ax:22s} n={len(v)} skew={v.mean():+.4f}", flush=True)
        except Exception as e:
            print(f"[t4] FAIL profile {name} {ax}: {e}", flush=True)
    for ax in T4_DESIGNER_AXES:
        out = os.path.join(RESULTS, "t4", "designer", f"{name}|{ax}.json")
        if os.path.exists(out): continue
        try:
            r = elicit_selfdebias(model, tok, ax, 0, batch_size)
            json.dump(dict(axis=ax, model=model_id, biased=r["biased"],
                           debiased=r["debiased"], diag=r["diag"]), open(out, "w"))
            print(f"[t4] {name} designer {ax:22s} gap={r['diag']['contrast_gap']:+.4f}", flush=True)
        except Exception as e:
            print(f"[t4] FAIL designer {name} {ax}: {e}", flush=True)

# ============================ T2 (H1 removal at 7-9B) ============================
def run_t2(name, target_id, sibling_id, mmlu_rows, wt, axis="occ_gender",
           alphas=(2, 4, 8, 16), seeds=(0, 1, 2), batch_size=12, train_bs=8):
    rows = []
    for seed in seeds:
        # designers are target & sibling; edit trains on target. Elicit each
        # designer, edit on target. Free the model in finally so a mid-family
        # abort still releases VRAM for the next family. dispatch=False: the
        # target is TRAINED, so it must be single-device (no accelerate hooks).
        model, tok = load_model(target_id, dispatch=False)
        try:
            _run_t2_seed(rows, name, target_id, sibling_id, model, tok, axis,
                         alphas, seed, mmlu_rows, wt, batch_size, train_bs)
        finally:
            _free(model)
    return rows


def _run_t2_seed(rows, name, target_id, sibling_id, model, tok, axis, alphas,
                 seed, mmlu_rows, wt, batch_size, train_bs):
    if True:
        pre = fast_eval(model, tok, axis, mmlu_rows, wt, batch_size)
        for role, des_id in (("self", target_id), ("sibling", sibling_id)):
            # elicit on the designer
            if des_id == target_id:
                el = elicit_selfdebias(model, tok, axis, seed, batch_size)
            else:
                dmodel, dtok = load_model(des_id, dispatch=False)
                try:
                    el = elicit_selfdebias(dmodel, dtok, axis, seed, batch_size)
                finally:
                    _free(dmodel)
            dwb, _ = train_task_vector(model, tok, el["biased"], rank=16, steps=250,
                                       lr=1e-4, seed=seed, bs=train_bs, targets=ATTN,
                                       grad_checkpoint=True)
            dwd, _ = train_task_vector(model, tok, el["debiased"], rank=16, steps=250,
                                       lr=1e-4, seed=seed, bs=train_bs, targets=ATTN,
                                       grad_checkpoint=True)
            E, _ = binarize(contrast(dwb, dwd), "per_tensor", 0.0, seed)
            best = float("nan")
            for a in alphas:
                undo = apply_edit(model, E, alpha=a, sign=-1.0)
                try:
                    post = fast_eval(model, tok, axis, mmlu_rows, wt, batch_size)
                finally:
                    undo()
                if collateral_ok(pre, post):
                    r = bias_reduction(pre, post)
                    best = r if np.isnan(best) else max(best, r)
            rows.append(dict(family=name, role=role, seed=seed,
                             pre_skew=pre["skew"], self_removal=best))
            print(f"[t2] {name} {role} seed{seed} removal={best:+.4f}", flush=True)

# ============ T2-CROSS (self-penalty at 7-9B, full cross-family panel) ========
# The definitive H1-at-T2 test: per target, remove bias with SELF + SIBLING
# (same-family) and every OTHER family's 7-9B model (cross-family). The
# self-penalty = mean(cross removal) - mean(same removal); correlate with
# within-family sharing. Runs occ_gender (matches the small-scale ρ=-0.81) AND a
# naturalistic axis (dodges the occ ceiling). Pre-elicits each designer corpus
# ONCE, then trains per target. Resumable at (target,axis,seed,designer).

T2X = [  # (family, target 7-9B, sibling)
    ("qwen",    "Qwen/Qwen2.5-7B-Instruct",         "Qwen/Qwen2.5-3B-Instruct"),
    ("llama",   "meta-llama/Llama-3.1-8B-Instruct",  "meta-llama/Llama-3.2-3B-Instruct"),
    ("gemma",   "google/gemma-2-9b-it",              "google/gemma-2-2b-it"),
    ("olmo",    "allenai/OLMo-2-1124-7B-Instruct",   "allenai/OLMo-2-0425-1B-Instruct"),
    ("granite", "ibm-granite/granite-3.1-8b-instruct","ibm-granite/granite-3.1-2b-instruct"),
]


def _t2x_corpus_fp(dname, axis, seed):
    return os.path.join(RESULTS, "t2x", "corpora", f"{dname}|{axis}|s{seed}.json")


def t2x_elicit_pool(targets, axes, seeds, batch_size):
    """Elicit & cache each designer model's corpora ONCE. Self/cross designers
    are the 7-9B models; sibling designers are the smaller siblings."""
    os.makedirs(os.path.join(RESULTS, "t2x", "corpora"), exist_ok=True)
    # Any target uses every OTHER family as a cross designer, so elicit the full
    # pool (all 5 families' 7-9B for self/cross + their siblings), regardless of
    # which targets are requested. Sibling corpora only needed for requested tgts.
    pool = {}
    for fam, tgt, sib in T2X:
        pool[fam] = tgt            # 7-9B designer (self / cross) — always needed
        if fam in targets:
            pool[fam + "_sib"] = sib   # sibling designer — only if fam is a target
    for dname, mid in pool.items():
        need = [(ax, s) for ax in axes for s in seeds
                if not os.path.exists(_t2x_corpus_fp(dname, ax, s))]
        if not need:
            continue
        print(f"[t2x-elicit] load {dname} ({mid})", flush=True)
        m, t = load_model(mid, dispatch=False)
        try:
            for ax, s in need:
                el = elicit_selfdebias(m, t, ax, s, batch_size)
                json.dump(dict(designer=dname, axis=ax, seed=s, biased=el["biased"],
                               debiased=el["debiased"], diag=el["diag"]),
                          open(_t2x_corpus_fp(dname, ax, s), "w"))
                print(f"[t2x-elicit] {dname:12s} {ax:22s} s{s} "
                      f"gap={el['diag']['contrast_gap']:+.3f}", flush=True)
        finally:
            _free(m)


def t2x_run(targets, axes, seeds, batch_size, train_bs, alphas=(2, 4, 8, 16)):
    """Train edits on each target from every cached designer corpus; record
    in-budget removal. Resumable via results/t2x/removal.jsonl."""
    import traceback
    os.makedirs(os.path.join(RESULTS, "t2x"), exist_ok=True)
    out_fp = os.path.join(RESULTS, "t2x", "removal.jsonl")
    done = set()
    if os.path.exists(out_fp):
        for ln in open(out_fp):
            try:
                r = json.loads(ln); done.add((r["target"], r["axis"], r["seed"], r["designer"]))
            except Exception:
                pass
    mmlu = load_mmlu(n=200, seed=0); wt = load_wikitext(n_chunks=20, seed=0)
    for fam, tgt, sib in T2X:
        if fam not in targets:
            continue
        designers = [("self", fam), ("sibling", fam + "_sib")] + \
                    [("cross", of) for of, _, _ in T2X if of != fam]
        todo = [(role, dn, ax, s) for ax in axes for s in seeds for role, dn in designers
                if (fam, ax, s, dn) not in done and os.path.exists(_t2x_corpus_fp(dn, ax, s))]
        if not todo:
            print(f"[t2x] {fam}: nothing to do (done or corpora missing)", flush=True); continue
        bs = 6 if fam == "gemma" else batch_size
        model, tok = load_model(tgt, dispatch=False)
        try:
            pre_cache = {}
            for role, dn, ax, s in todo:
                try:
                    if ax not in pre_cache:
                        pre_cache[ax] = fast_eval(model, tok, ax, mmlu, wt, bs)
                    pre = pre_cache[ax]
                    c = json.load(open(_t2x_corpus_fp(dn, ax, s)))
                    dwb, _ = train_task_vector(model, tok, c["biased"], rank=16, steps=250,
                                               lr=1e-4, seed=s, bs=train_bs, targets=ATTN,
                                               grad_checkpoint=True)
                    dwd, _ = train_task_vector(model, tok, c["debiased"], rank=16, steps=250,
                                               lr=1e-4, seed=s, bs=train_bs, targets=ATTN,
                                               grad_checkpoint=True)
                    E, _ = binarize(contrast(dwb, dwd), "per_tensor", 0.0, s)
                    best = float("nan")
                    for a in alphas:
                        undo = apply_edit(model, E, alpha=a, sign=-1.0)
                        try:
                            post = fast_eval(model, tok, ax, mmlu, wt, bs)
                        finally:
                            undo()
                        if collateral_ok(pre, post):
                            r = bias_reduction(pre, post)
                            best = r if np.isnan(best) else max(best, r)
                    row = dict(target=fam, axis=ax, seed=s, role=role, designer=dn,
                               removal=best, pre_skew=pre["skew"])
                    with open(out_fp, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[t2x] tgt={fam:8s} {ax:20s} s{s} {role:7s}<-{dn:12s} "
                          f"removal={best:+.4f}", flush=True)
                except Exception:
                    print(f"[t2x] FAIL tgt={fam} {ax} s{s} {role}<-{dn}:", flush=True)
                    traceback.print_exc()
        finally:
            _free(model)
    print("[t2x] DONE", flush=True)


# ===== v5 generalized panel engine (V1 same-family-large, X 3B bridge) =========
# Shares the results/t2x/corpora cache so the v4 7B corpora are reused. A panel
# is: targets [(fam, target_hf_id)] + designers_by_fam {fam:[(role,dname,dhf)]}.

FAM_MODELS = {
    "qwen":    {"3b": "Qwen/Qwen2.5-3B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct",
                "large": "Qwen/Qwen2.5-14B-Instruct"},
    "llama":   {"3b": "meta-llama/Llama-3.2-3B-Instruct", "7b": "meta-llama/Llama-3.1-8B-Instruct"},
    "gemma":   {"3b": "google/gemma-2-2b-it", "7b": "google/gemma-2-9b-it"},
    "olmo":    {"3b": "allenai/OLMo-2-0425-1B-Instruct", "7b": "allenai/OLMo-2-1124-7B-Instruct",
                "large": "allenai/OLMo-2-1124-13B-Instruct"},
    "granite": {"3b": "ibm-granite/granite-3.1-2b-instruct", "7b": "ibm-granite/granite-3.1-8b-instruct"},
    "falcon":  {"3b": "tiiuae/Falcon3-3B-Instruct", "7b": "tiiuae/Falcon3-7B-Instruct",
                "large": "tiiuae/Falcon3-10B-Instruct"},
}
# The v4 t2x cached these 7B corpora under these dnames (fam / fam_sib); reuse them.
V4_7B_DNAME = {f: f for f in ["qwen", "llama", "gemma", "olmo", "granite"]}


def panel_run(panel, targets, designers_by_fam, axes, seeds, batch_size, train_bs,
              alphas=(2, 4, 8, 16)):
    """Generalized cross-designer removal panel. Writes results/<panel>/removal.jsonl,
    resumable at (target,axis,seed,designer). Corpora cached in results/t2x/corpora."""
    import traceback
    os.makedirs(os.path.join(RESULTS, panel), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "t2x", "corpora"), exist_ok=True)
    # 1) elicit every unique designer model once (cache in shared corpora dir)
    uniq = {}
    for fam, _tgt_hf in targets:
        for role, dname, dhf in designers_by_fam[fam]:
            uniq[dname] = dhf
    for dname, dhf in uniq.items():
        need = [(ax, s) for ax in axes for s in seeds
                if not os.path.exists(_t2x_corpus_fp(dname, ax, s))]
        if not need:
            continue
        print(f"[{panel}-elicit] load {dname} ({dhf})", flush=True)
        m, t = load_model(dhf, dispatch=False)
        try:
            for ax, s in need:
                el = elicit_selfdebias(m, t, ax, s, batch_size)
                json.dump(dict(designer=dname, axis=ax, seed=s, biased=el["biased"],
                               debiased=el["debiased"], diag=el["diag"]),
                          open(_t2x_corpus_fp(dname, ax, s), "w"))
                print(f"[{panel}-elicit] {dname:16s} {ax:22s} s{s} "
                      f"gap={el['diag']['contrast_gap']:+.3f}", flush=True)
        finally:
            _free(m)
    # 2) per target, train from each designer corpus (resumable)
    out_fp = os.path.join(RESULTS, panel, "removal.jsonl")
    done = set()
    if os.path.exists(out_fp):
        for ln in open(out_fp):
            try:
                r = json.loads(ln); done.add((r["target"], r["axis"], r["seed"], r["designer"]))
            except Exception:
                pass
    mmlu = load_mmlu(n=200, seed=0); wt = load_wikitext(n_chunks=20, seed=0)
    for fam, tgt_hf in targets:
        ds = designers_by_fam[fam]
        todo = [(role, dn, ax, s) for ax in axes for s in seeds for role, dn, _ in ds
                if (fam, ax, s, dn) not in done and os.path.exists(_t2x_corpus_fp(dn, ax, s))]
        if not todo:
            print(f"[{panel}] {fam}: nothing to do", flush=True); continue
        bs = 6 if fam == "gemma" else batch_size
        model, tok = load_model(tgt_hf, dispatch=False)
        try:
            pre_cache = {}
            for role, dn, ax, s in todo:
                try:
                    if ax not in pre_cache:
                        pre_cache[ax] = fast_eval(model, tok, ax, mmlu, wt, bs)
                    pre = pre_cache[ax]
                    c = json.load(open(_t2x_corpus_fp(dn, ax, s)))
                    dwb, _ = train_task_vector(model, tok, c["biased"], rank=16, steps=250,
                                               lr=1e-4, seed=s, bs=train_bs, targets=ATTN,
                                               grad_checkpoint=True)
                    dwd, _ = train_task_vector(model, tok, c["debiased"], rank=16, steps=250,
                                               lr=1e-4, seed=s, bs=train_bs, targets=ATTN,
                                               grad_checkpoint=True)
                    E, _ = binarize(contrast(dwb, dwd), "per_tensor", 0.0, s)
                    best = float("nan")
                    for a in alphas:
                        undo = apply_edit(model, E, alpha=a, sign=-1.0)
                        try:
                            post = fast_eval(model, tok, ax, mmlu, wt, bs)
                        finally:
                            undo()
                        if collateral_ok(pre, post):
                            r = bias_reduction(pre, post)
                            best = r if np.isnan(best) else max(best, r)
                    row = dict(panel=panel, target=fam, axis=ax, seed=s, role=role,
                               designer=dn, removal=best, pre_skew=pre["skew"])
                    with open(out_fp, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[{panel}] tgt={fam:8s} {ax:20s} s{s} {role:11s}<-{dn:16s} "
                          f"removal={best:+.4f}", flush=True)
                except Exception:
                    print(f"[{panel}] FAIL tgt={fam} {ax} s{s} {role}<-{dn}:", flush=True)
                    traceback.print_exc()
        finally:
            _free(model)
    print(f"[{panel}] DONE", flush=True)


def build_v1_panel():
    """V1: size-matched same-family-LARGE designers for qwen/olmo/falcon 7B targets.
    Designers per target: self(7b) + same_large + size-matched cross(other 7b larges? no:
    the v4 7B cross designers). Reuses v4 7B corpora for self/cross."""
    fams = ["qwen", "olmo", "falcon"]
    targets = [(f, FAM_MODELS[f]["7b"]) for f in fams]
    designers = {}
    all7 = ["qwen", "llama", "gemma", "olmo", "granite", "falcon"]
    for f in fams:
        d = [("self", f, FAM_MODELS[f]["7b"])]                       # self (7b)
        d.append(("same_large", f + "_large", FAM_MODELS[f]["large"]))  # size-matched same-family
        for of in all7:                                              # cross 7-9B (existing corpora where present)
            if of != f:
                d.append(("cross", of, FAM_MODELS[of]["7b"]))
        designers[f] = d
    return targets, designers


def build_x_panel():
    """X: 3B bridge tier. Targets qwen/llama/falcon 3B; designers self + size-matched
    cross (other 3B)."""
    fams = ["qwen", "llama", "falcon"]
    targets = [(f, FAM_MODELS[f]["3b"]) for f in fams]
    designers = {}
    for f in fams:
        d = [("self", f + "_3b", FAM_MODELS[f]["3b"])]
        for of in fams:
            if of != f:
                d.append(("cross", of + "_3b", FAM_MODELS[of]["3b"]))
        designers[f] = d
    return targets, designers


# ---- optional gradient checkpointing param shim (train_task_vector supports it) ----

def main():
    import argparse, traceback
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", nargs="*", default=["t4", "t2"])
    ap.add_argument("--t4-batch", type=int, default=8)
    ap.add_argument("--t2-batch", type=int, default=12)
    ap.add_argument("--t4-only", nargs="*", default=None,
                    help="subset of T4 names to run (e.g. gemma27b llama70b)")
    ap.add_argument("--t2-only", nargs="*", default=None,
                    help="subset of T2 family names to run (e.g. qwen olmo)")
    ap.add_argument("--t2x-axes", nargs="*",
                    default=["occ_gender", "crows_socioeconomic"],
                    help="axes for the cross-designer self-penalty run")
    ap.add_argument("--t2x-seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--t2x-targets", nargs="*",
                    default=["qwen", "llama", "gemma", "olmo", "granite"])
    ap.add_argument("--t2x-batch", type=int, default=12)
    ap.add_argument("--t2x-train-bs", type=int, default=8)
    a = ap.parse_args()

    # gated models (gemma, llama) need auth; log in from HF_TOKEN if present.
    tok_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok_env:
        try:
            from huggingface_hub import login
            login(token=tok_env); print("[auth] HF login OK", flush=True)
        except Exception as ex:
            print(f"[auth] HF login failed: {ex}", flush=True)
    else:
        print("[auth] no HF_TOKEN in env — gated models (gemma/llama) will fail", flush=True)

    if "t4" in a.do:
        # (hf_id, name, 8bit?, batch)
        T4 = [("google/gemma-2-27b-it", "gemma27b", False, 4),
              ("Qwen/Qwen2.5-32B-Instruct", "qwen32b", False, a.t4_batch),
              ("meta-llama/Llama-3.1-70B-Instruct", "llama70b", True, a.t4_batch),
              ("Qwen/Qwen2.5-72B-Instruct", "qwen72b", True, a.t4_batch)]
        if a.t4_only:
            T4 = [t for t in T4 if t[1] in a.t4_only]
        for mid, name, e8, bs in T4:
            try:
                run_t4(mid, name, e8, bs)
            except Exception:
                print(f"[t4] {name} ABORTED:", flush=True); traceback.print_exc()

    if "t2" in a.do:
        mmlu = load_mmlu(n=200, seed=0); wt = load_wikitext(n_chunks=20, seed=0)
        # (name, target 7-9B, sibling)
        T2 = [("qwen", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
              ("llama", "meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"),
              ("gemma", "google/gemma-2-9b-it", "google/gemma-2-2b-it"),
              ("olmo", "allenai/OLMo-2-1124-7B-Instruct", "allenai/OLMo-2-0425-1B-Instruct"),
              ("granite", "ibm-granite/granite-3.1-8b-instruct", "ibm-granite/granite-3.1-2b-instruct")]
        if a.t2_only:
            T2 = [t for t in T2 if t[0] in a.t2_only]
        # merge with any existing rows so per-family reruns accumulate.
        out_fp = os.path.join(RESULTS, "t2_selfremoval.json")
        allrows = json.load(open(out_fp)) if os.path.exists(out_fp) else []
        done = {(r["family"], r["role"], r["seed"]) for r in allrows}
        for name, tgt, sib in T2:
            bs = 6 if name == "gemma" else a.t2_batch
            try:
                for row in run_t2(name, tgt, sib, mmlu, wt, batch_size=bs):
                    if (row["family"], row["role"], row["seed"]) not in done:
                        allrows.append(row)
                json.dump(allrows, open(out_fp, "w"), indent=2)  # write after each family
            except Exception:
                print(f"[t2] {name} ABORTED:", flush=True); traceback.print_exc()
        json.dump(allrows, open(out_fp, "w"), indent=2)
        print(f"[t2] wrote t2_selfremoval.json ({len(allrows)} rows)", flush=True)

    if "t2cross" in a.do:
        print(f"[t2x] axes={a.t2x_axes} seeds={a.t2x_seeds} targets={a.t2x_targets}",
              flush=True)
        # phase 1: elicit every designer corpus once (cheap, inference)
        t2x_elicit_pool(a.t2x_targets, a.t2x_axes, a.t2x_seeds, a.t2x_batch)
        # phase 2: train per target from cached corpora (resumable)
        t2x_run(a.t2x_targets, a.t2x_axes, a.t2x_seeds, a.t2x_batch, a.t2x_train_bs)

    if "v1" in a.do:  # size-matched same-family-large designers (qwen/olmo/falcon 7B)
        tg, ds = build_v1_panel()
        print(f"[v1] targets={[f for f,_ in tg]} axes={a.t2x_axes} seeds={a.t2x_seeds}",
              flush=True)
        panel_run("v1", tg, ds, a.t2x_axes, a.t2x_seeds, a.t2x_batch, a.t2x_train_bs)

    if "x" in a.do:   # 3B bridge tier (qwen/llama/falcon 3B)
        tg, ds = build_x_panel()
        print(f"[x] targets={[f for f,_ in tg]} axes={a.t2x_axes} seeds={a.t2x_seeds}",
              flush=True)
        panel_run("x", tg, ds, a.t2x_axes, a.t2x_seeds, a.t2x_batch, a.t2x_train_bs)

    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
