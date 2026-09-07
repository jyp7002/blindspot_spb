"""Does designer<->target BIAS-PROFILE similarity predict removal?

This is the paper's mechanism stated at the level it actually claims: a designer
whose bias resembles the target's should be blind exactly where they overlap, and
so should debias it less well. G and O4 tested this at the FAMILY level (pairwise
co-encoding, aggregate multi-axis) and both failed. Neither had the design this
one does: pooling x, x2, mvb and t2x gives many designers per target, spanning
1-9B and six families, with an item-space bias profile cached for nearly all of
them (results/profiles/<model>|<axis>.npy).

Two forms, because they answer different questions:

  (1) INCLUDING self.  cos(profile) = 1 exactly when designer == target, so this
      asks whether the self arm sits on a continuum with other designers or is a
      discontinuity. cos and is_self are near-collinear here, so it is descriptive.

  (2) EXCLUDING self — the real test. If the blind spot is caused by shared bias
      direction, the effect must be CONTINUOUS: among non-self designers, greater
      profile similarity should still mean worse removal. If similarity predicts
      nothing once self is dropped, then "shared bias" is not a mechanism, and
      whatever the self arm suffers from is specific to being the target itself.

Spec: removal ~ profile_cos + log2(designer_size), target x axis CELL fixed
effects, cluster bootstrap over cells. nan removal -> 0.

Usage:  python src/v6_profile_similarity.py [--boot 5000] [--axis occ_gender]
"""
import os, json, argparse, collections
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(REPO, "results")
PROF = os.path.join(RES, "profiles")

# designer / target name -> profile model id, and parameter count (B)
PROFILE = {"qwen_3b": "qwen3b", "llama_3b": "llama3b", "falcon_3b": "falcon3b",
           "gemma_3b": "gemma2b", "phi_3b": "phi3.5", "granite_3b": "granite2b",
           "qwen_sib": "qwen3b", "llama_sib": "llama3b", "gemma_sib": "gemma2b",
           "olmo_sib": "olmo1b",
           "qwen": "qwen7b", "llama": "llama8b", "gemma": "gemma9b",
           "olmo": "olmo7b", "granite": "granite8b"}
SIZE = {"qwen_3b": 3.1, "llama_3b": 3.2, "falcon_3b": 3.2, "gemma_3b": 2.6,
        "phi_3b": 3.8, "granite_3b": 2.5, "qwen_sib": 3.1, "llama_sib": 3.2,
        "gemma_sib": 2.6, "olmo_sib": 1.0,
        "qwen": 7.0, "llama": 8.0, "gemma": 9.0, "olmo": 7.0, "granite": 8.0}
# a panel's target family -> the profile of the actual target checkpoint
TARGET_PROFILE = {
    ("x", "qwen"): "qwen3b", ("x", "llama"): "llama3b", ("x", "falcon"): "falcon3b",
    ("x2", "gemma"): "gemma2b", ("x2", "phi"): "phi3.5", ("x2", "granite"): "granite2b",
    ("mvb", "phi"): "phi3.5", ("mvb", "qwen"): "qwen3b", ("mvb", "gemma"): "gemma2b",
    ("t2x", "qwen"): "qwen7b", ("t2x", "llama"): "llama8b",
    ("t2x", "gemma"): "gemma9b", ("t2x", "olmo"): "olmo7b",
}
PANELS = {"x": "x/removal.jsonl", "x2": "x2/removal.jsonl",
          "t2x": "t2x/removal.jsonl", "mvb": "v6trace/mvb/removal.jsonl"}


def z(x):
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def prof(model, axis):
    fp = os.path.join(PROF, f"{model}|{axis}.npy")
    return np.load(fp) if os.path.exists(fp) else None


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def fit(rows, terms, rng, n_boot):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)

    def design(cells):
        Y, X = [], []
        for c in cells:
            qs = by[c]
            if len(qs) < 2:
                continue
            my = np.mean([q["y"] for q in qs])
            mu = [np.mean([q[t] for q in qs]) for t in terms]
            for q in qs:
                Y.append(q["y"] - my)
                X.append([q[t] - m for t, m in zip(terms, mu)])
        return np.asarray(Y), np.asarray(X)

    cells = list(by)
    Y, X = design(cells)
    if len(Y) == 0 or np.linalg.matrix_rank(X) < len(terms):
        return None
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    draws = []
    for _ in range(n_boot):
        pick = [cells[i] for i in rng.integers(0, len(cells), len(cells))]
        yy, xx = design(pick)
        if len(yy) == 0 or np.linalg.matrix_rank(xx) < len(terms):
            continue
        draws.append(np.linalg.lstsq(xx, yy, rcond=None)[0])
    D = np.asarray(draws)
    ci = [(float(np.percentile(D[:, i], 2.5)), float(np.percentile(D[:, i], 97.5)))
          for i in range(len(terms))] if len(D) else [(float("nan"),) * 2] * len(terms)
    return beta, ci, len(cells), len(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="occ_gender")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    rows, missing = [], collections.Counter()
    for panel, path in PANELS.items():
        fp = os.path.join(RES, path)
        if not os.path.exists(fp):
            continue
        for ln in open(fp):
            r = json.loads(ln)
            if r["axis"] != a.axis:
                continue
            tp = TARGET_PROFILE.get((panel, r["target"]))
            dp = PROFILE.get(r["designer"])
            if tp is None or dp is None:
                missing[f"name:{panel}/{r['designer']}"] += 1
                continue
            pt, pd = prof(tp, a.axis), prof(dp, a.axis)
            if pt is None or pd is None:
                missing[f"profile:{tp if pt is None else dp}"] += 1
                continue
            c = cos(pt, pd)
            if c is None:
                continue
            rows.append(dict(panel=panel, cell=(panel, r["target"]), y=z(r["removal"]),
                             pcos=c, size=float(np.log2(SIZE[r["designer"]])),
                             is_self=1.0 if r["role"] == "self" else 0.0,
                             designer=r["designer"], target=r["target"]))

    print(f"axis = {a.axis}")
    print(f"usable designer-cells: {len(rows)} across "
          f"{len({r['cell'] for r in rows})} cells, panels "
          f"{sorted({r['panel'] for r in rows})}")
    if missing:
        print(f"dropped (no profile / unmapped): {dict(missing.most_common(6))}")

    sf = [r["pcos"] for r in rows if r["is_self"]]
    ns = [r["pcos"] for r in rows if not r["is_self"]]
    if sf:
        print(f"\nprofile cos: self arms mean {np.mean(sf):+.4f} (should be ~1.0), "
              f"non-self mean {np.mean(ns):+.4f} range [{min(ns):+.3f}, {max(ns):+.3f}]")

    out = {}
    for label, sub, terms in [
            ("(1) ALL designers incl. self", rows, ["pcos", "size"]),
            ("(2) EXCLUDING self — the real test", [r for r in rows if not r["is_self"]],
             ["pcos", "size"])]:
        print("\n" + "=" * 72)
        print(f"{label}   n={len(sub)}")
        print("=" * 72)
        res = fit(sub, terms, rng, a.boot)
        if res is None:
            print("  rank-deficient / too few cells"); continue
        beta, ci, nc, nobs = res
        names = ["profile cos(designer, target)", "log2(designer size)"]
        for n, b, (lo, hi) in zip(names, beta, ci):
            star = "" if (np.isnan(lo) or lo < 0 < hi) else "  *excludes 0"
            print(f"  {n:32s} {b:+9.4f}  [{lo:+.4f}, {hi:+.4f}]{star}")
        out[label] = {n: dict(beta=float(b), ci=list(c)) for n, b, c in zip(names, beta, ci)}
        print(f"  ({nc} cells, {nobs} obs)")

    print("\n  Blind-spot prediction: profile cos coefficient NEGATIVE (more similar")
    print("  designer -> worse removal). If (2) is ~0, shared bias is not a continuous")
    print("  mechanism and the self arm's deficit is not explained by similarity.")

    if not a.no_write:
        fp = os.path.join(RES, "v6", f"profile_similarity_{a.axis}.json")
        json.dump(dict(axis=a.axis, n=len(rows), results=out,
                       spec="removal ~ profile_cos + log2(size), cell FE, cluster boot",
                       boot=a.boot, boot_seed=a.seed), open(fp, "w"), indent=2)
        print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
