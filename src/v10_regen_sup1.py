"""v10 regeneration — Fig 4 LEFT: the layer x projection fold-enrichment matrix.

WHAT THIS FIXES
---------------
`experiments_v10.md` B4 names `sup1_analysis.json` as the source of the Fig 4
left heatmap. That file does NOT contain a layer x projection matrix. Its
`structure[<cell>]` entries persist only two 1-D MARGINALS:

    layer_enrichment : list over layers   (collapsed over projections)
    projection       : dict over q/k/v/o  (collapsed over layers)

plus `thirds` (early/middle/late) and `meta`. The 2-D object the figure needs
was never written. This module verifies that claim programmatically (see
`recon_check` in the artifact) and then REBUILDS the matrix from the underlying
`.npz` support dumps (`results/v8dec/supports/*.npz`, format documented in
`src/v8_support.py`), using the SAME null definition as `src/v8_sup1.py`.

THE NULL (identical to v8_sup1.describe)
----------------------------------------
Under the hypergeometric null a support of size k drawn uniformly over N
parameters puts k * N_cell / N coordinates in any cell, so

    fold_enrichment(cell) = (k_cell / N_cell) / (k_total / N_total)
                          = density(cell) / density(dump)

`density(dump)` is `meta["density"]`, exactly as v8_sup1 uses it. This module
reproduces v8_sup1's published 1-D marginals from the raw dumps bit-for-bit
before it emits anything 2-D (`reproduction.per_dump_marginals`).

CROSS-MODEL DEPTH
-----------------
The four non-fused targets have DIFFERENT depths (gemma 26, llama 28,
qwen7b 28, qwen-3B 36), so raw layer index is not poolable. Normalization
chosen and recorded here (`normalization`): a layer L of an n_layer model sits
at fractional depth (L + 0.5) / n_layers and falls in bin
min(K-1, floor(K * depth)) with K = DEPTH_BINS. The pooled cell value is the
MEAN over dumps of each dump's own fold-enrichment in that (bin, projection) --
the same "mean over dumps of a per-dump enrichment" statistic that produces the
numbers §5.4 and §5.6 of the draft already cite.

PHI
---
phi-3.5 edits a single FUSED `qkv_proj` and never touches `o_proj`, so it has
no q/k/v/o decomposition and cannot enter a layer x {q,k,v,o} matrix. It is
excluded from the heatmap. The exclusion is derived (a dump is excluded iff it
carries exactly one projection name), never hardcoded, and is recorded in
`inclusion`. NOTE that the draft's early->late gradient is pooled over ALL
dumps INCLUDING phi -- see `gradient` and `draft_check`.

GATE STATUS
-----------
SUP1 is computed from support dumps, not from gated removals, so the v9
re-score does not touch it (`RESULTS_METHOD_v9.md` §5, "Unaffected by the
gate"). There is no results_v9/ counterpart and none is needed; this is NOT a
silent as-run substitution. Recorded in `provenance.gate_status`.

R2-bug rule: every number below is computed from an artifact at run time,
including the draft values it is checked against (regex-lifted from
`binary_debiaser_draft_v3.md`, with line numbers).
"""
import os
import sys
import re
import json
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v10_common import HERE, RES, OUT_V10, ensure_dirs, jload, jdump  # noqa: E402
from v8_support import load_support, parse_module                     # noqa: E402
import v8_sup1                                                        # noqa: E402

# ---- design parameters (NOT results; recorded in the artifact) --------------
DEPTH_BINS = 8              # k for the pooled fractional-depth heatmap
THIRDS_BINS = 3             # the early/middle/late strip the draft cites
DUMP_ROOT = os.path.join(RES, "v8dec")
PUBLISHED_SUP1 = "results/v8/sup1_analysis.json"
DRAFT = "binary_debiaser_draft_v3.md"
OUT = os.path.join(OUT_V10, "sup1_matrix.json")


def cell_key(d):
    return f"{d['target']}|{d['axis']}|s{d['seed']}|{d['designer']}"


# --------------------------------------------------------------- recon ------

def recon_key_tree(pub):
    """Prove/disprove 'sup1_analysis.json holds only 1-D marginals'."""
    if not pub:
        return dict(present=False)
    struct = pub.get("structure", {})
    per_entry_keys = sorted({k for v in struct.values() for k in v})
    shapes = {}
    for k in per_entry_keys:
        kinds = set()
        for v in struct.values():
            if k not in v:
                continue
            x = v[k]
            if isinstance(x, list):
                kinds.add("list[%s]" % ("list" if x and isinstance(x[0], list) else "scalar"))
            elif isinstance(x, dict):
                inner = {type(y).__name__ for y in x.values()}
                kinds.add("dict{%s}" % ",".join(sorted(inner)))
            else:
                kinds.add(type(x).__name__)
        shapes[k] = sorted(kinds)
    # a 2-D matrix would show up as a list-of-lists or a dict-of-lists
    two_d = sorted(k for k, s in shapes.items()
                   if any("list[list]" in x or x == "dict{list}" for x in s))
    return dict(
        present=True,
        top_level_keys=sorted(pub),
        n_structure_entries=len(struct),
        structure_entry_keys=per_entry_keys,
        structure_entry_value_shapes=shapes,
        two_dimensional_keys=two_d,
        has_layer_by_projection_matrix=bool(two_d),
        verdict=("CONFIRMED: only 1-D marginals (layer_enrichment over layers, "
                 "projection over q/k/v/o) plus thirds/meta; no layer x projection "
                 "matrix is persisted anywhere in the file"
                 if not two_d else "REFUTED: a 2-D object is present"))


# ------------------------------------------------------- dump reduction -----

def reduce_dump(path):
    """(meta, {(layer, proj): [k, n_params]}) with the arrays dropped."""
    meta, mods = load_support(path)
    cnt = collections.defaultdict(int)
    tot = collections.defaultdict(int)
    n_layers = 0
    projs = set()
    unparsed = []
    for k, (shape, idx, _sgn) in mods.items():
        L, p = parse_module(k)
        if L is None:
            unparsed.append(k)
            continue
        cnt[(L, p)] += int(len(idx))
        tot[(L, p)] += int(np.prod(shape))
        projs.add(p)
        n_layers = max(n_layers, L + 1)
    del mods
    return dict(meta=meta, cnt=cnt, tot=tot, n_layers=n_layers,
                projections=sorted(projs), unparsed_modules=sorted(unparsed))


def enr(k, n, density):
    """Fold-enrichment over the hypergeometric null. v8_sup1's definition."""
    if not n or not density:
        return float("nan")
    return (k / n) / density


def bin_of(L, n_layers, k_bins):
    """Fractional-depth bin of layer L. Midpoint rule, stated in the artifact."""
    return min(k_bins - 1, int(k_bins * ((L + 0.5) / n_layers)))


def floor_thirds(L, n_layers):
    """v8_sup1.describe's own early/middle/late split (integer-floor thirds)."""
    b1, b2 = n_layers // 3, 2 * n_layers // 3
    return 0 if L < b1 else (1 if L < b2 else 2)


def binned_enrichment(red, assign, n_bins, projections):
    """Per-dump fold-enrichment on a (bin x projection) grid."""
    c = collections.defaultdict(int)
    t = collections.defaultdict(int)
    for (L, p), k in red["cnt"].items():
        c[(assign(L, red["n_layers"]), p)] += k
    for (L, p), n in red["tot"].items():
        t[(assign(L, red["n_layers"]), p)] += n
    d = red["meta"]["density"]
    return {(b, p): enr(c[(b, p)], t[(b, p)], d)
            for b in range(n_bins) for p in projections}, c, t


def binned_layer_marginal(red, assign, n_bins):
    """Per-dump fold-enrichment per depth bin, collapsed over projections."""
    c = collections.defaultdict(int)
    t = collections.defaultdict(int)
    for (L, p), k in red["cnt"].items():
        c[assign(L, red["n_layers"])] += k
    for (L, p), n in red["tot"].items():
        t[assign(L, red["n_layers"])] += n
    d = red["meta"]["density"]
    return [enr(c[b], t[b], d) for b in range(n_bins)]


# -------------------------------------------------------- draft lifting -----

DRAFT_PATTERNS = {
    "gradient": r"Support concentrates in late layers \(([0-9.]+)\s*[×x]\s*early\s*[→-]+>?\s*([0-9.]+)\s*[×x]\s*late\)",
    "density_not_importance": r"v_proj holds \*fewer\* surviving coordinates \(([0-9.]+)\s*[×x]\)\s*than q_proj \(([0-9.]+)\s*[×x]\)",
    "seed_overlap": r"Across seeds the support is stable \(([0-9.]+)\s*[×x]\s*global, signed agreement ([0-9.]+), n = (\d+)\)",
    "designer_overlap": r"across designers on the one measured cell, more so \(([0-9.]+)\s*[×x], n = (\d+)",
    "axis_bimodal": r"signed agreement ([0-9.]+)\) while every other axis pair sits at chance \(~([0-9.]+)\)",
}


def lift_draft(relpath):
    fp = os.path.join(HERE, relpath)
    if not os.path.exists(fp):
        return {}, []
    lines = open(fp).read().split("\n")
    out = {}
    for name, pat in DRAFT_PATTERNS.items():
        rx = re.compile(pat)
        for i, ln in enumerate(lines, start=1):
            m = rx.search(ln)
            if m:
                out[name] = dict(line=i, groups=list(m.groups()), text=ln.strip()[:400])
                break
    return out, lines


def close(a, b, tol):
    return a == a and b == b and abs(a - b) <= tol


# ---------------------------------------------------------------- main ------

def build():
    ensure_dirs()
    pub = jload(PUBLISHED_SUP1)
    recon = recon_key_tree(pub)

    dumps = v8_sup1.find_dumps([DUMP_ROOT])
    if not dumps:
        raise SystemExit(f"[v10 sup1] no support dumps under {DUMP_ROOT}")

    reds = {}
    for d in dumps:
        reds[cell_key(d)] = reduce_dump(d["path"])
    keys_all = sorted(reds)

    # ---- inclusion: fused-projection dumps cannot enter a layer x q/k/v/o grid
    fused = {k for k, r in reds.items() if len(r["projections"]) == 1}
    included = [k for k in keys_all if k not in fused]
    excluded = sorted(fused)
    projections = sorted({p for k in included for p in reds[k]["projections"]})
    ragged = sorted({k for k in included
                     if sorted(reds[k]["projections"]) != projections})

    # ---- reproduction of the published 1-D marginals, from the raw dumps
    dev_layer, dev_proj, dev_thirds, nlay_mismatch = 0.0, 0.0, 0.0, []
    for k in keys_all:
        r = reds[k]
        pv = (pub or {}).get("structure", {}).get(k)
        if not pv:
            continue
        d = r["meta"]["density"]
        nl = r["n_layers"]
        if nl != pv["n_layers"]:
            nlay_mismatch.append([k, nl, pv["n_layers"]])
        lay = [enr(sum(r["cnt"][(L, p)] for p in r["projections"]),
                   sum(r["tot"][(L, p)] for p in r["projections"]), d)
               for L in range(nl)]
        dev_layer = max(dev_layer, max(abs(a - b) for a, b in
                                       zip(lay, pv["layer_enrichment"])))
        for p in r["projections"]:
            mine = enr(sum(r["cnt"][(L, p)] for L in range(nl)),
                       sum(r["tot"][(L, p)] for L in range(nl)), d)
            dev_proj = max(dev_proj, abs(mine - pv["projection"][p]["enrichment"]))
        th = binned_layer_marginal(r, floor_thirds, THIRDS_BINS)
        for i, name in enumerate(("early", "middle", "late")):
            dev_thirds = max(dev_thirds, abs(th[i] - pv["thirds"][name]["enrichment"]))

    # ---- the 2-D matrix (pooled, fractional depth)
    per_cell = collections.defaultdict(list)
    pool_c = collections.defaultdict(int)
    pool_t = collections.defaultdict(int)
    for k in included:
        r = reds[k]
        grid, c, t = binned_enrichment(r, lambda L, n: bin_of(L, n, DEPTH_BINS),
                                       DEPTH_BINS, projections)
        for key, v in grid.items():
            per_cell[key].append(v)
        for key, v in c.items():
            pool_c[key] += v
        for key, v in t.items():
            pool_t[key] += v

    pooled_density = (sum(reds[k]["meta"]["n_support"] for k in included) /
                      sum(reds[k]["meta"]["n_params"] for k in included))
    matrix = [[float(np.mean(per_cell[(b, p)])) for p in projections]
              for b in range(DEPTH_BINS)]
    matrix_std = [[float(np.std(per_cell[(b, p)], ddof=1)) for p in projections]
                  for b in range(DEPTH_BINS)]
    matrix_n = [[len(per_cell[(b, p)]) for p in projections]
                for b in range(DEPTH_BINS)]
    matrix_pooled = [[enr(pool_c[(b, p)], pool_t[(b, p)], pooled_density)
                      for p in projections] for b in range(DEPTH_BINS)]

    # ---- marginals, computed the way the published ones were
    proj_marg = {}
    for p in projections:
        vals = [enr(sum(reds[k]["cnt"][(L, p)] for L in range(reds[k]["n_layers"])),
                    sum(reds[k]["tot"][(L, p)] for L in range(reds[k]["n_layers"])),
                    reds[k]["meta"]["density"])
                for k in included if p in reds[k]["projections"]]
        proj_marg[p] = dict(mean=float(np.mean(vals)), median=float(np.median(vals)),
                            n_dumps=len(vals))
    depth_marg = [float(np.mean([binned_layer_marginal(
        reds[k], lambda L, n: bin_of(L, n, DEPTH_BINS), DEPTH_BINS)[b]
        for k in included])) for b in range(DEPTH_BINS)]

    # ---- per-target native-resolution matrices (no depth normalization)
    by_target = collections.defaultdict(list)
    for k in included:
        by_target[k.split("|")[0]].append(k)
    native = {}
    for tgt, ks in sorted(by_target.items()):
        nl = reds[ks[0]]["n_layers"]
        vals = []
        for L in range(nl):
            row = []
            for p in projections:
                row.append(float(np.mean([
                    enr(reds[k]["cnt"][(L, p)], reds[k]["tot"][(L, p)],
                        reds[k]["meta"]["density"]) for k in ks])))
            vals.append(row)
        native[tgt] = dict(n_layers=nl, n_dumps=len(ks), projections=projections,
                           cells=sorted(ks), values=vals)

    # ---- early -> late gradient, all four ways
    def grad(assign, ks, label):
        rowsv = [binned_layer_marginal(reds[k], assign, THIRDS_BINS) for k in ks]
        m = [float(np.mean([r[i] for r in rowsv])) for i in range(THIRDS_BINS)]
        return dict(binning=label, population_n=len(ks),
                    early=m[0], middle=m[1], late=m[2],
                    late_over_early=(m[2] / m[0]) if m[0] else float("nan"))

    frac3 = lambda L, n: bin_of(L, n, THIRDS_BINS)
    gradient = dict(
        published_floor_thirds_all_dumps=grad(floor_thirds, keys_all,
                                              "v8_sup1 integer-floor thirds (L<n//3 | <2n//3 | rest)"),
        published_floor_thirds_nonfused=grad(floor_thirds, included,
                                             "v8_sup1 integer-floor thirds, fused-qkv dumps dropped"),
        fracdepth_thirds_all_dumps=grad(frac3, keys_all,
                                        "fractional depth (L+0.5)/n_layers, K=3"),
        fracdepth_thirds_nonfused=grad(frac3, included,
                                       "fractional depth (L+0.5)/n_layers, K=3, fused-qkv dumps dropped"),
        which_one_the_draft_cites="published_floor_thirds_all_dumps",
        which_one_the_heatmap_shows="fracdepth_thirds_nonfused (K=3 view of the K=%d panel)" % DEPTH_BINS,
    )

    # ---- overlap medians re-derived from the persisted pair rows (§5.6 text)
    overlap = {}
    if pub:
        for arm, rowsv in ((a, pub["overlap"][a]["pairs"]) for a in ("seed", "designer", "axis")):
            fg = [r["fold_global"] for r in rowsv if r["fold_global"] == r["fold_global"]]
            sf = [r["signed_frac_of_intersection"] for r in rowsv
                  if r["signed_frac_of_intersection"] == r["signed_frac_of_intersection"]]
            overlap[arm] = dict(n_pairs=len(rowsv),
                                fold_global_median=float(np.median(fg)),
                                signed_frac_median=float(np.median(sf)),
                                published_summary=pub["overlap"][arm]["summary"])
        ax = pub["overlap"]["axis"]["pairs"]
        axis_of = lambda s: s.split("|")[1]
        pairaxes = [(r, {axis_of(r["a"]), axis_of(r["b"])}) for r in ax]
        bbq_axes = sorted({a for _r, s in pairaxes for a in s if a.startswith("bbq_")})
        bbq = [r["signed_frac_of_intersection"] for r, s in pairaxes if s == set(bbq_axes)]
        oth = [r["signed_frac_of_intersection"] for r, s in pairaxes if s != set(bbq_axes)]
        overlap["axis_bimodality"] = dict(
            bbq_pair_axes=bbq_axes, n_bbq_pairs=len(bbq), n_other_pairs=len(oth),
            bbq_signed_mean=float(np.mean(bbq)), bbq_signed_median=float(np.median(bbq)),
            other_signed_mean=float(np.mean(oth)), other_signed_median=float(np.median(oth)))

    # ---- draft cross-check (draft values lifted by regex, never typed)
    draft, _lines = lift_draft(DRAFT)
    checks = []

    def chk(name, draft_key, gi, computed, tol, what, alt=None):
        d = draft.get(draft_key)
        if not d:
            checks.append(dict(id=name, status="NOT_FOUND_IN_DRAFT", computed=computed))
            return
        dv = float(d["groups"][gi])
        checks.append(dict(id=name, draft_line=d["line"], draft_value=dv,
                           computed=float(computed), what=what,
                           agrees=close(dv, float(computed), tol), tol=tol,
                           alternative=alt, draft_text=d["text"]))

    g = gradient["published_floor_thirds_all_dumps"]
    gn = gradient["published_floor_thirds_nonfused"]
    chk("sup1.gradient.early", "gradient", 0, g["early"], 5e-3,
        "mean over ALL dumps of per-dump early-third fold-enrichment (v8_sup1 thirds)",
        dict(nonfused_only=gn["early"]))
    chk("sup1.gradient.late", "gradient", 1, g["late"], 5e-3,
        "mean over ALL dumps of per-dump late-third fold-enrichment (v8_sup1 thirds)",
        dict(nonfused_only=gn["late"]))
    chk("sup1.proj.v_proj", "density_not_importance", 0, proj_marg["v_proj"]["mean"], 5e-3,
        "mean over non-fused dumps of per-dump v_proj fold-enrichment",
        dict(median=proj_marg["v_proj"]["median"]))
    chk("sup1.proj.q_proj", "density_not_importance", 1, proj_marg["q_proj"]["mean"], 5e-3,
        "mean over non-fused dumps of per-dump q_proj fold-enrichment",
        dict(median=proj_marg["q_proj"]["median"]))
    if overlap:
        chk("sup1.overlap.seed.fold", "seed_overlap", 0, overlap["seed"]["fold_global_median"],
            5e-3, "median fold_global over the 36 seed pairs")
        chk("sup1.overlap.seed.signed", "seed_overlap", 1, overlap["seed"]["signed_frac_median"],
            5e-4, "median signed_frac over the 36 seed pairs")
        chk("sup1.overlap.seed.n", "seed_overlap", 2, overlap["seed"]["n_pairs"],
            0.5, "n seed pairs")
        chk("sup1.overlap.designer.fold", "designer_overlap", 0,
            overlap["designer"]["fold_global_median"], 5e-3,
            "median fold_global over the 9 designer pairs")
        chk("sup1.overlap.designer.n", "designer_overlap", 1, overlap["designer"]["n_pairs"],
            0.5, "n designer pairs")
        b = overlap["axis_bimodality"]
        chk("sup1.overlap.axis.bbq_signed", "axis_bimodal", 0, b["bbq_signed_median"], 5e-4,
            "MEDIAN signed_frac over the bbq_Age x bbq_Race pairs (the statistic used "
            "for every other overlap number in the same sentence)",
            dict(mean=b["bbq_signed_mean"], n_pairs=b["n_bbq_pairs"]))
        chk("sup1.overlap.axis.chance", "axis_bimodal", 1, b["other_signed_median"], 5e-3,
            "median signed_frac over the non-bbq axis pairs",
            dict(mean=b["other_signed_mean"], n_pairs=b["n_other_pairs"]))

    payload = dict(
        artifact="v10_sup1_matrix",
        generated_by="src/v10_regen_sup1.py",
        purpose="Fig 4 LEFT — layer x projection fold-enrichment heatmap",
        provenance=dict(
            support_dumps_root=os.path.relpath(DUMP_ROOT, HERE),
            n_dumps_found=len(dumps),
            dump_files=sorted(os.path.relpath(d["path"], HERE) for d in dumps),
            published_marginals_file=PUBLISHED_SUP1,
            gate_status=("NOT GATED — SUP1 is computed from support dumps, not from "
                         "gated removals, so the v9 re-score does not touch it "
                         "(RESULTS_METHOD_v9.md §5). No results_v9/ counterpart exists "
                         "and none is required; this is not an as-run substitution."),
            null_definition=("fold = (k_cell/N_cell) / (k_dump/N_dump) — the "
                             "hypergeometric expectation, identical to "
                             "v8_sup1.describe's enrichment"),
        ),
        recon_check=recon,
        inclusion=dict(
            included_cells=included, n_included=len(included),
            excluded_cells=excluded, n_excluded=len(excluded),
            excluded_targets=sorted({k.split("|")[0] for k in excluded}),
            exclusion_rule=("a dump is excluded iff it carries exactly one projection "
                            "name, i.e. its attention is a single fused module; derived "
                            "from the dump's own module keys, not hardcoded"),
            excluded_projection_names=sorted({p for k in excluded
                                              for p in reds[k]["projections"]}),
            exclusion_reason=("phi-3.5 edits a fused qkv_proj and never touches o_proj, "
                              "so it has no q/k/v/o decomposition to place on the "
                              "heatmap's x axis"),
            ragged_included_cells=ragged,
            included_layer_counts={k.split("|")[0]: reds[k]["n_layers"] for k in included},
            composition=dict(
                by_target=dict(collections.Counter(k.split("|")[0] for k in included)),
                by_axis=dict(collections.Counter(k.split("|")[1] for k in included)),
                by_designer=dict(collections.Counter(k.split("|")[3] for k in included)),
                note=("the pooled heatmap averages over seeds, axes and designers as "
                      "well as targets; the per-target native matrices are kept "
                      "separately in `per_target_native`")),
        ),
        normalization=dict(
            depth_bins=DEPTH_BINS,
            depth_rule=("layer L of an n_layer model sits at fractional depth "
                        "(L + 0.5)/n_layers and falls in bin "
                        "min(K-1, floor(K * depth)); K = %d" % DEPTH_BINS),
            bin_edges_fractional=[[b / DEPTH_BINS, (b + 1) / DEPTH_BINS]
                                  for b in range(DEPTH_BINS)],
            cell_statistic=("mean over included dumps of that dump's own "
                            "fold-enrichment in the (bin, projection) cell — the same "
                            "statistic behind the marginals the draft already cites"),
            alternative_statistic=("`matrix_pooled_counts` instead sums k and N over "
                                   "dumps first, which weights by model size"),
            caption_line=("layer x projection fold-enrichment over a hypergeometric "
                          "null; depth pooled into %d equal fractional-depth bins "
                          "across %d supports from %d targets (gemma 26L, llama 28L, "
                          "qwen-7B 28L, qwen-3B 36L); phi excluded (fused qkv_proj, "
                          "no o_proj)" % (DEPTH_BINS, len(included), len(by_target))),
        ),
        matrix=dict(
            depth_bins=DEPTH_BINS, projections=projections,
            row_labels=["d%d/%d" % (b + 1, DEPTH_BINS) for b in range(DEPTH_BINS)],
            values=matrix, std=matrix_std, n_dumps=matrix_n,
            pooled_counts_variant=matrix_pooled,
            pooled_density=pooled_density,
            orientation="values[bin][projection]; bin 0 = shallowest"),
        marginals=dict(projection=proj_marg, depth_bin=depth_marg,
                       depth_bin_note=("collapsed over projections with counts, not by "
                                       "averaging the matrix row — the two differ "
                                       "because projections hold unequal parameter counts")),
        per_target_native=native,
        gradient=gradient,
        overlap_recomputed_from_pair_rows=overlap,
        reproduction=dict(
            source=PUBLISHED_SUP1,
            max_abs_dev_layer_enrichment=dev_layer,
            max_abs_dev_projection_enrichment=dev_proj,
            max_abs_dev_thirds_enrichment=dev_thirds,
            n_layers_mismatches=nlay_mismatch,
            verdict=("EXACT" if max(dev_layer, dev_proj, dev_thirds) == 0
                     else "DEVIATES")),
        draft_check=checks,
    )

    fp = jdump(payload, OUT)

    print(f"[v10 sup1] dumps={len(dumps)} included={len(included)} "
          f"excluded={len(excluded)} ({','.join(sorted({k.split('|')[0] for k in excluded})) or '-'})")
    print(f"[v10 sup1] recon: {recon.get('verdict')}")
    print(f"[v10 sup1] reproduction of published 1-D marginals: "
          f"{payload['reproduction']['verdict']} "
          f"(layer {dev_layer:g}, proj {dev_proj:g}, thirds {dev_thirds:g})")
    hdr = "bin ".ljust(8) + "".join(p.rjust(10) for p in projections)
    print("[v10 sup1] " + hdr)
    for b in range(DEPTH_BINS):
        print("[v10 sup1] " + ("d%d/%d" % (b + 1, DEPTH_BINS)).ljust(8) +
              "".join(f"{matrix[b][i]:10.3f}" for i in range(len(projections))))
    for name, gg in gradient.items():
        if isinstance(gg, dict):
            print(f"[v10 sup1] gradient {name:34s} n={gg['population_n']:2d} "
                  f"{gg['early']:.4f} -> {gg['middle']:.4f} -> {gg['late']:.4f}")
    bad = [c for c in checks if c.get("agrees") is False]
    for c in checks:
        if "draft_value" in c:
            print(f"[v10 sup1] draft L{c['draft_line']:>4} {c['id']:32s} "
                  f"draft={c['draft_value']:<9g} computed={c['computed']:<12.6g} "
                  f"{'OK' if c['agrees'] else 'CONFLICT'}")
    print(f"[v10 sup1] draft conflicts: {len(bad)}")
    print(f"[v10 sup1] wrote {fp}")
    return payload


if __name__ == "__main__":
    build()
