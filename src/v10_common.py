"""v10 shared substrate — one definition of every convention the figures and the
number audit both depend on.

Everything here is copied semantics-for-semantics from the v9 scripts that
produced the citable numbers (`src/v9_regen.py`, `src/v9_rescore.py`,
`src/v9_gate.py`). The point is that Fig N and the manuscript sentence about
Fig N are computed by the SAME code, so they cannot drift.

Conventions, all inherited (do not "improve" them here — that silently moves
published numbers):

  * budget-fail (nan removal) -> 0.0                       (`z`)
  * bootstrap resamples the target x axis CELL, 10k, seed 0 (`boot_paired`)
  * a cell's value is the MEAN over its rows (seeds, designers)
  * CI status is "excludes 0" iff the interval does not straddle 0
  * patch sizes are MiB (bytes / 2**20) — see `mib`, and note the manuscript
    prints these as "MB"

`HERE` is the REPO ROOT, matching v9_regen.py / v9_rescore.py / v8_dec_analyze.py.
(src/figures.py uses a different, older convention; do not mix them.)
"""
import os, json, glob, collections, random
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
V9 = os.path.join(HERE, "results_v9")
OUT_V10 = os.path.join(RES, "v10")
FIGDIR = os.path.join(HERE, "figures")
AUDITDIR = os.path.join(HERE, "audit")

MIB = 2 ** 20
BOOT_N = 10000
BOOT_SEED = 0

# The five panels that predate the alpha-trace fix and can never be re-scored.
NOT_RESCORABLE = ("t2x", "v1", "x", "x2", "v6trace/ste")


def ensure_dirs():
    for d in (OUT_V10, FIGDIR, AUDITDIR):
        os.makedirs(d, exist_ok=True)


def z(x):
    """Budget-fail / missing -> 0.0. The project-wide nan->0 rule."""
    return 0.0 if x is None or (isinstance(x, float) and x != x) else float(x)


def jload(relpath, default=None):
    fp = relpath if os.path.isabs(relpath) else os.path.join(HERE, relpath)
    if not os.path.exists(fp):
        return default
    with open(fp) as fh:
        return json.load(fh)


def jdump(obj, relpath):
    fp = relpath if os.path.isabs(relpath) else os.path.join(HERE, relpath)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
    return fp


def rows(root, panel):
    """Rows of <root>/<panel>/removal.jsonl, or [] if the panel is absent."""
    fp = os.path.join(root, panel, "removal.jsonl")
    if not os.path.exists(fp):
        return []
    with open(fp) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def panels_under(root):
    """Every panel path under root that has a removal.jsonl."""
    out = []
    for fp in glob.glob(os.path.join(root, "**", "removal.jsonl"), recursive=True):
        out.append(os.path.relpath(os.path.dirname(fp), root))
    return sorted(out)


# ----------------------------------------------------------------- stats ----

def boot_paired(pairs, n=BOOT_N, seed=BOOT_SEED):
    """Percentile bootstrap of the paired difference. Verbatim from
    v9_regen.boot_paired — same RNG, same draw order, same indices."""
    d = [a - b for a, b in pairs]
    if not d:
        return None
    rng = random.Random(seed)
    bs = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in d])) for _ in range(n))
    return dict(point=float(np.mean(d)), lo=bs[int(.025 * n)], hi=bs[int(.975 * n) - 1],
                n=len(d))


def boot_mean(vals, n=BOOT_N, seed=BOOT_SEED):
    """Percentile bootstrap of a single mean, same resampling unit and RNG
    discipline as boot_paired. Used for per-condition CIs, which the v9
    artifacts do not carry (v9 bootstrapped only paired differences)."""
    d = [float(v) for v in vals]
    if not d:
        return None
    rng = random.Random(seed)
    bs = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in d])) for _ in range(n))
    return dict(point=float(np.mean(d)), lo=bs[int(.025 * n)], hi=bs[int(.975 * n) - 1],
                n=len(d))


def status(d):
    if d is None:
        return "n/a"
    return "excludes 0" if (d["lo"] > 0 or d["hi"] < 0) else "covers 0"


def fmt(d, p=4):
    return "n/a" if d is None else \
        f"{d['point']:+.{p}f} [{d['lo']:+.{p}f}, {d['hi']:+.{p}f}] (n={d['n']}, {status(d)})"


def mib(nbytes):
    """Patch size in MiB. The manuscript labels this 'MB'; the values in the
    §5.5 table only reproduce under 2**20, not 10**6."""
    return None if nbytes is None else float(nbytes) / MIB


# ------------------------------------------------------------------- DEC ----

DEC_CONDITIONS = ("C-ref", "C-a", "C-tensorshuf", "C-layershuf", "C-b",
                  "C-rand", "C-bottom")

# What each condition perturbs. Used by Fig 2's marker encoding. This mapping is
# a presentation choice made in v10 (it is not registered in any earlier
# artifact) and is recorded here so the figure and its caption cannot disagree.
DEC_PERTURBS = {
    "C-ref":         "reference",
    "C-a":           "support",   # random coordinates, TRUE signs
    "C-bottom":      "support",   # bottom-|delta| coordinates, TRUE signs
    "C-b":           "sign",      # selected coordinates, PERMUTED signs
    "C-rand":        "both",      # random coordinates AND random signs
    "C-layershuf":   "location",
    "C-tensorshuf":  "location",
}


def dec_cells(root):
    """(target, axis) -> {condition: mean removal}. Dedupe key and cell
    definition verbatim from v9_regen.dec."""
    seen = {}
    for r in rows(root, "v8dec"):
        seen[(r["target"], r["axis"], r["seed"], r["designer"], r["condition"])] = r
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in seen.values():
        cells[(r["target"], r["axis"])][r["condition"]].append(z(r["removal"]))
    return {k: {c: float(np.mean(v)) for c, v in d.items()} for k, d in cells.items()}


def write_csv(path, header, rowlist):
    """Per-figure data CSV. Plain stdlib so a clean tree needs no pandas."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rowlist:
            w.writerow(r)
    return path
