"""v10 — the patch-size artifact (Fig 1 callout + the manuscript's sec 5.5 table).

The v10 plan (B1) says the Fig 1 patch-size callout "is pulled from the size
artifact, not typed". There was no size artifact; this builds it, and in doing
so settles four things the printed table leaves implicit.

  1. THE UNIT. The manuscript prints "MB". The sec 5.5 table only reproduces
     under MiB (bytes / 2**20), which is this project's own convention
     (src/method_evidence.py:126 divides by 2**20 and prints "MB"). Both are
     emitted so the mislabelling is visible rather than inherited.

  2. TARGET IDENTITY. `target` is NOT unique across panels: 'llama' and 'qwen'
     name the 3B models in the small-tier panels and the 8B/7B models in the
     big-tier panels. Everything is keyed on (tier, target), and the mapping to
     the manuscript's row labels is ASSERTED against n_params in code, not
     claimed in a comment.

  3. WHAT "DENSE" MEANS. The dense column is `binary-per_tensor-s0.0` — the
     1-bit-per-weight patch at zero sparsity. It is NOT the full-precision
     edit. The `fp` variant carries no bytes anywhere in the tree, so full
     precision was never sized. A reader who takes "dense" to mean the
     unstripped edit is off by 16x.

  4. THE SUPPORT INDEX, WHICH IS NOT COUNTED. The stored relation is exactly

         bytes = retained_coordinates / 8  +  2 bytes per tensor

     i.e. one bit per retained coordinate and ZERO bits recording WHICH
     coordinates those are. That is fine for a dense patch, where position is
     implicit. It is not fine for a 1% patch: a merge target cannot apply a
     sparse sign field without its support. The headline "0.4-1.6 MB mergeable
     patch" therefore prices only the payload. This module computes the index
     cost under three encodings so the shippable size is on the record.

Run: python3 src/v10_regen_size.py
"""
import os, sys, math, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C

# Panels carrying `bytes`. v8spc stores the same sizes under bare sparsity
# labels; both are read so the source is explicit rather than incidental.
SIZE_PANELS = [("v6trace/ablate", "small"), ("v6trace/ablate_big", "big"),
               ("v8spc/small", "small"), ("v8spc/big", "big")]

DENSE_VARIANTS = ("binary-per_tensor-s0.0", "s0.0")
SPARSE_99_VARIANTS = ("binary-per_tensor-s0.99", "s0.99")

# The manuscript's sec 5.5 table as printed, for verify-mode. These are the
# CLAIMS being checked, never a data source.
DRAFT_TABLE = {
    "qwen-3B": ("40.5", "0.41"), "gemma-2.6B": ("43.9", "0.44"),
    "llama-3.2B": ("84.0", "0.84"), "phi-3.8B": ("108.0", "1.08"),
    "qwen-7B": ("98.0", "0.98"), "llama-8B": ("160.0", "1.60"),
}
ROW_LABEL = {
    ("small", "qwen"): "qwen-3B", ("small", "gemma"): "gemma-2.6B",
    ("small", "llama"): "llama-3.2B", ("small", "phi"): "phi-3.8B",
    ("big", "qwen"): "qwen-7B", ("big", "llama"): "llama-8B",
}
# phi edits the FUSED qkv_proj only; every other target edits q,k,v,o. Its
# patch is not commensurable with the others and must be flagged wherever the
# six are shown together.
FUSED_QKV_TARGETS = {("small", "phi")}

INDEX_BYTES = 4          # a plain 32-bit coordinate index


def collect():
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for panel, tier in SIZE_PANELS:
        for r in C.rows(C.V9, panel):
            if r.get("bytes") is None:
                continue
            k = (tier, r["target"], r.get("variant"))
            for f in ("bytes", "n_params", "n_flips", "effective_sparsity"):
                if r.get(f) is not None:
                    acc[k][f].append(float(r[f]))
            acc[k]["_panel"].append(panel)
    out = {}
    for k, d in acc.items():
        rec = {f: float(np.mean(v)) for f, v in d.items() if f != "_panel"}
        rec["n_rows"] = len(d.get("bytes", []))
        rec["panels"] = sorted(set(d["_panel"]))
        rec["spread_bytes"] = (float(np.max(d["bytes"])) - float(np.min(d["bytes"])))
        out[k] = rec
    return out


def pick(sizes, tier, target, variants):
    """Return every variant alias that resolves, so a panel that contributes
    nothing is visible as such instead of silently dropping out."""
    hits = [(v, sizes[(tier, target, v)]) for v in variants
            if (tier, target, v) in sizes]
    return (hits[0] if hits else (None, None)), [v for v, _ in hits]


def retained(rec):
    """Retained coordinates. n_flips is absent from these panels (verified:
    every row has n_flips == None), so it is derived from n_params and the
    measured effective sparsity rather than assumed."""
    npar, es = rec.get("n_params"), rec.get("effective_sparsity")
    if npar is None:
        return None
    return npar * (1.0 - (es if es is not None else 0.0))


def index_costs(n_retained, n_total):
    """What it costs to record WHICH coordinates survive, under three
    encodings. The stored `bytes` field pays none of this."""
    if not n_retained or not n_total:
        return None
    p = n_retained / n_total
    # Shannon bound on an unordered n_retained-subset of n_total, per element.
    bits_per = (-(p * math.log2(p) + (1 - p) * math.log2(1 - p)) / p) if 0 < p < 1 else 0.0
    return {
        "bitmap_bytes": n_total / 8.0,
        "explicit_index_bytes": n_retained * INDEX_BYTES,
        "entropy_bound_bytes": n_retained * bits_per / 8.0,
        "entropy_bits_per_retained_coord": bits_per,
        "note": ("bitmap = one bit per model coordinate; explicit = 32-bit "
                 "indices; entropy bound = the information-theoretic floor for "
                 "naming the support"),
    }


def main():
    C.ensure_dirs()
    sizes = collect()
    panels_seen = sorted({p for rec in sizes.values() for p in rec["panels"]})
    panels_declared = sorted({p for p, _ in SIZE_PANELS})
    dead_panels = [p for p in panels_declared if p not in panels_seen]

    rows_out, table, ident = [], {}, {}
    for (tier, target), label in sorted(ROW_LABEL.items(), key=lambda kv: kv[1]):
        (dv, dense), dense_aliases = pick(sizes, tier, target, DENSE_VARIANTS)
        (sv, sparse), sparse_aliases = pick(sizes, tier, target, SPARSE_99_VARIANTS)
        if dense is None:
            continue
        nret_d, nret_s = retained(dense), (retained(sparse) if sparse else None)
        npar = dense.get("n_params")

        # assert, do not assert-by-comment, that (tier,target) is one model
        ident[label] = {"tier": tier, "target": target, "edited_n_params": npar,
                        "distinct_from_same_name_other_tier": None}
        rec = {
            "tier": tier, "target": target, "draft_row_label": label,
            "edited_n_params": npar,
            "edited_n_params_note": ("this is the EDITED-tensor parameter count "
                                     "(attention projections only), not the model's "
                                     "parameter count, which the row label names"),
            "fused_qkv_only": (tier, target) in FUSED_QKV_TARGETS,
            "dense": {
                "variant": dv, "variant_aliases_found": dense_aliases,
                "bytes": dense["bytes"], "mib": C.mib(dense["bytes"]),
                "mb_decimal": dense["bytes"] / 1e6,
                "retained_coordinates": nret_d,
                "panels": dense["panels"], "n_rows": dense["n_rows"],
                "spread_bytes": dense["spread_bytes"],
                "what_this_is": ("the 1-bit-per-weight patch at zero sparsity, "
                                 "NOT the full-precision edit"),
            },
        }
        if sparse is not None:
            idx = index_costs(nret_s, npar)
            payload = sparse["bytes"]
            rec["s0.99"] = {
                "variant": sv, "variant_aliases_found": sparse_aliases,
                "bytes_payload_only": payload, "mib": C.mib(payload),
                "mb_decimal": payload / 1e6,
                "retained_coordinates": nret_s,
                "effective_sparsity": sparse.get("effective_sparsity"),
                "panels": sparse["panels"], "n_rows": sparse["n_rows"],
                "spread_bytes": sparse["spread_bytes"],
                "index_cost": idx,
                "shippable_mib": ({
                    "payload_plus_entropy_bound": C.mib(payload + idx["entropy_bound_bytes"]),
                    "payload_plus_explicit_index": C.mib(payload + idx["explicit_index_bytes"]),
                    "payload_plus_bitmap": C.mib(payload + idx["bitmap_bytes"]),
                } if idx else None),
            }
        # bytes <-> retained-coordinate fit, computed not assumed
        fits = []
        for k, r_, n_ in (("dense", dense, nret_d), ("s0.99", sparse, nret_s)):
            if r_ is None or not n_:
                continue
            resid = r_["bytes"] - n_ / 8.0
            fits.append({"variant": k, "bytes": r_["bytes"],
                         "retained_coordinates": n_,
                         "bits_per_retained_coordinate": 8.0 * r_["bytes"] / n_,
                         "residual_bytes_over_one_bit_each": resid})
        rec["bytes_fit"] = fits

        cd, cs = DRAFT_TABLE[label]
        chk = {}
        for unit, conv in (("mib", C.mib), ("mb_decimal", lambda b: b / 1e6)):
            dvv = conv(dense["bytes"])
            svv = conv(sparse["bytes"]) if sparse is not None else None
            chk[unit] = {
                "dense_computed": round(dvv, 3), "dense_claim": float(cd),
                "dense_matches": abs(dvv - float(cd)) < 0.05,
                "sparse_computed": (round(svv, 3) if svv is not None else None),
                "sparse_claim": float(cs),
                "sparse_matches": (abs(svv - float(cs)) < 0.005 if svv is not None else None),
            }
        rec["draft_table_check"] = chk
        rows_out.append(rec)
        table[label] = rec

    # target-name collision, established from the data
    byname = collections.defaultdict(list)
    for r in rows_out:
        byname[r["target"]].append((r["tier"], r["edited_n_params"], r["draft_row_label"]))
    collisions = {t: v for t, v in byname.items() if len(v) > 1}
    for t, v in collisions.items():
        distinct = len({p for _, p, _ in v}) == len(v)
        for r in rows_out:
            if r["target"] == t:
                ident[r["draft_row_label"]]["distinct_from_same_name_other_tier"] = distinct

    have99 = [r for r in rows_out if "s0.99" in r]
    mibs = [(r["s0.99"]["mib"], r["draft_row_label"]) for r in have99]
    ship = [(r["s0.99"]["shippable_mib"]["payload_plus_entropy_bound"],
             r["draft_row_label"]) for r in have99 if r["s0.99"].get("shippable_mib")]
    rng = {
        "payload_only": {
            "min_mib": min(mibs)[0], "min_target": min(mibs)[1],
            "max_mib": max(mibs)[0], "max_target": max(mibs)[1],
            "min_mb_decimal": min((r["s0.99"]["mb_decimal"], r["draft_row_label"])
                                  for r in have99)[0],
            "max_mb_decimal": max((r["s0.99"]["mb_decimal"], r["draft_row_label"])
                                  for r in have99)[0]},
        "with_entropy_bound_index": ({"min_mib": min(ship)[0], "min_target": min(ship)[1],
                                      "max_mib": max(ship)[0], "max_target": max(ship)[1]}
                                     if ship else None),
        "n_targets": len(have99),
    }

    report = {
        "artifact": "v10_regen_size", "module": "src/v10_regen_size.py",
        "gate": "v9 (results_v9/) only",
        "per_target": table,
        "target_identity": {
            "collisions": {t: v for t, v in collisions.items()},
            "resolved_by": "(tier, target) plus edited_n_params",
            "checked": ident,
        },
        "panels": {"declared": panels_declared, "contributing": panels_seen,
                   "contributing_nothing": dead_panels,
                   "note": ("v8spc stores the same sizes under bare sparsity labels; "
                            "the ablate panels resolve first, so v8spc is redundant "
                            "here rather than missing")},
        "patch_range_1pct": rng,
        "manuscript_callout": {
            "text": "0.4-1.6 MB mergeable patch",
            "reproduces_under_mib": (abs(rng["payload_only"]["min_mib"] - 0.4) < 0.05
                                     and abs(rng["payload_only"]["max_mib"] - 1.6) < 0.05),
            "reproduces_under_decimal_mb": (
                abs(rng["payload_only"]["min_mb_decimal"] - 0.4) < 0.05
                and abs(rng["payload_only"]["max_mb_decimal"] - 1.6) < 0.05),
            "unit_defect": ("the manuscript labels these MB; they are MiB "
                            "(bytes/2**20). Decimal MB gives numbers 4.9% larger."),
            "scope_defect": ("the range prices the PAYLOAD ONLY. bytes = retained/8 "
                             "+ 2 bytes per tensor, so nothing is spent recording "
                             "which coordinates are retained. A 1% patch cannot be "
                             "merged without its support."),
        },
        "conventions": {
            "dense_column_is": "binary-per_tensor-s0.0 (1 bit/weight, zero sparsity)",
            "full_precision_never_sized": ("the `fp` variant carries no bytes field "
                                           "anywhere in results_v9/"),
            "retained_coordinates": ("derived as n_params * (1 - effective_sparsity); "
                                     "n_flips is None on every row of these panels"),
            "mib": "bytes / 2**20", "mb_decimal": "bytes / 1e6",
        },
    }
    fp = C.jdump(report, "results/v10/patch_sizes.json")

    po = rng["payload_only"]
    print(f"[v10 size] {len(rows_out)} targets; {len(have99)} with a 1% size")
    print(f"  panels contributing: {panels_seen}")
    if dead_panels:
        print(f"  panels contributing NOTHING: {dead_panels}")
    print(f"  1% PAYLOAD range (MiB):  {po['min_mib']:.4f} ({po['min_target']}) "
          f"-> {po['max_mib']:.4f} ({po['max_target']})")
    print(f"  1% payload range (dec MB): {po['min_mb_decimal']:.4f} -> {po['max_mb_decimal']:.4f}")
    if rng["with_entropy_bound_index"]:
        w = rng["with_entropy_bound_index"]
        print(f"  1% SHIPPABLE (payload + entropy-bound index): "
              f"{w['min_mib']:.3f} -> {w['max_mib']:.3f} MiB")
    print(f"  callout reproduces: MiB={report['manuscript_callout']['reproduces_under_mib']} "
          f"decimalMB={report['manuscript_callout']['reproduces_under_decimal_mb']}")
    print("  per target (MiB):")
    for r in rows_out:
        c = r["draft_table_check"]["mib"]
        s = r.get("s0.99", {})
        sh = (s.get("shippable_mib") or {})
        print(f"    {r['draft_row_label']:12s} dense {c['dense_computed']:8.2f} "
              f"(claim {c['dense_claim']:6.1f}) {'OK' if c['dense_matches'] else 'MISMATCH'}"
              f" | 1% payload {str(c['sparse_computed']):>6} "
              f"{'OK' if c['sparse_matches'] else 'MISMATCH'}"
              f" | +index {sh.get('payload_plus_entropy_bound', float('nan')):7.2f}"
              f" / {sh.get('payload_plus_explicit_index', float('nan')):7.2f}"
              f"{'   [FUSED QKV ONLY]' if r['fused_qkv_only'] else ''}")
    for r in rows_out:
        for f in r["bytes_fit"]:
            if abs(f["bits_per_retained_coordinate"] - 1.0) > 0.01:
                print(f"    ! {r['draft_row_label']} {f['variant']}: "
                      f"{f['bits_per_retained_coordinate']:.4f} bits/coord")
    print(f"  target-name collisions: {sorted(collisions)}")
    print(f"wrote {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
