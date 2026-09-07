"""WS-A A1 — generate audit/manifest_v10.yaml.

The manifest maps every number in the manuscript to its source. It is
"hand-written once, machine-verified forever" — but hand-writing 500+ entries
invites exactly the transcription errors the audit exists to catch, so the
entries themselves are GENERATED from the artifacts:

  * a registry declares each estimand as (id, artifact, key, format, hint)
  * the value is resolved from the artifact at build time
  * occurrences in the draft are found by matching that value, constrained by
    a context hint and/or a line range so a coincidental digit match cannot
    bind the wrong sentence

What the generator CANNOT bind is reported, not hidden: entries that resolve
but appear nowhere, and draft numbers that no entry claims. That residue is
the work order for the manifest's manual tail.

Run: python3 src/v10_make_manifest.py
"""
import os, sys, re, json, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
from v10_audit import (scan_numbers, resolve_key, normalize_num, fmt_value,
                       apply_prose_edits)

DRAFT = "binary_debiaser_draft_v3.md"
OUT = "audit/manifest_v10.yaml"

V9DIFF = "results/v9/v9_diff_report.json"
RESC = "results_v9/rescore_report.json"
WS2 = "results/v9/ws2_determinism.json"
DEC = "results/v10/dec_analysis_v9.json"
SPC = "results/v10/spc_v9.json"
FRONT = "results/v10/frontier_v9.json"
LOPO = "results/v10/lopo_v9.json"
SUP1 = "results/v10/sup1_matrix.json"
SIZE = "results/v10/patch_sizes.json"
CLAIMS = "results/v10/claims_audit.json"
RECL = "results/v10/reclaimed.json"
MISS = "results/v10/missing_recomputed.json"


def E(eid, artifact, key, fmt, hint=None, lines=None, note=None, tol=None):
    e = {"id": eid, "artifact": artifact, "key": key, "format": fmt}
    if hint:
        e["hint"] = hint
    if lines:
        e["lines"] = list(lines)
    if note:
        e["note"] = note
    if tol is not None:
        e["tol"] = tol
    return e


def ci_parts(s):
    """'+0.2835 [+0.1692, +0.4072] (n=10, excludes 0)' -> point, lo, hi, n."""
    m = re.match(r"\s*([+-][\d.]+)\s*\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]\s*\(n=(\d+)", s or "")
    return m.groups() if m else (None, None, None, None)


def registry():
    reg = []

    # ---- gate constants (whitelist anchors, asserted against v9_gate) -------

    # ---- DEC per-condition means (Fig 2 / sec 5.3 table) --------------------
    dec = C.jload(DEC) or {}
    for cond in (dec.get("per_condition") or {}):
        reg.append(E(f"dec.mean.{cond}", DEC, f'per_condition."{cond}".mean',
                     "+.3f", hint=cond, note="v9 per-condition mean over cells"))

    # ---- DEC paired estimands (from the registered v9 diff) ----------------
    diff = (C.jload(V9DIFF) or {}).get("diff", {})
    for name, d in diff.items():
        slug = (name.lower().replace(" :: ", ".").replace(" ", "_")
                .replace("-", "_").replace("=", "").replace("__", "_"))
        pt, lo, hi, n = ci_parts(d.get("v9"))
        if pt is None:
            continue
        reg.append(E(f"{slug}.point", V9DIFF, f'diff."{name}".v9_point', "+.3f",
                     hint=name.split(" :: ")[-1]))
        for lab, val in (("lo", lo), ("hi", hi)):
            reg.append({"id": f"{slug}.{lab}", "artifact": V9DIFF,
                        "key": f'diff."{name}".v9', "format": "+.3f",
                        "extract": lab, "hint": name.split(" :: ")[-1]})

    # ---- re-score totals ----------------------------------------------------
    for key, eid, hint in (("totals.probes_or_rows_recovered", "rescore.recovered", "recovered"),
                           ("totals.cells_changed", "rescore.cells_changed", "cell values"),
                           ("totals.panels_rescored", "rescore.panels", "panels")):
        reg.append(E(eid, RESC, key, "d", hint=hint))

    # ---- LOPO (sec 5.4) -----------------------------------------------------
    lp = C.jload(LOPO) or {}
    for k, v in (lp.get("estimands") or lp.get("lopo") or {}).items():
        if isinstance(v, dict) and "point" in v:
            proj = re.sub(r".*?(\w+_proj).*", r"\1", k)
            reg.append(E(f"lopo.{proj}.point", LOPO,
                         f'{"estimands" if "estimands" in lp else "lopo"}."{k}".point',
                         "+.3f", hint=proj))

    # ---- patch sizes (sec 5.5 table + Fig 1 callout) -----------------------
    sz = C.jload(SIZE) or {}
    for label, rec in (sz.get("per_target") or {}).items():
        if "dense" in rec:
            reg.append(E(f"size.{label}.dense_mib", SIZE,
                         f'per_target."{label}".dense.mib', ".1f", hint=label))
        if "s0.99" in rec:
            reg.append(E(f"size.{label}.s99_mib_payload", SIZE,
                         f'per_target."{label}"."s0.99".mib', ".2f", hint=label))
    for eid, key, fmt in (
            ("size.range.min_mib_payload", "patch_range_1pct.payload_only.min_mib", ".1f"),
            ("size.range.max_mib_payload", "patch_range_1pct.payload_only.max_mib", ".1f")):
        reg.append(E(eid, SIZE, key, fmt, hint="MB"))

    # ---- boundary exposure (sec 6) -----------------------------------------
    reg += [
        E("boundary.within1_ppl_passing", CLAIMS,
          "boundary_exposure.ppl_passing_probes.within_1", "d", hint="probes"),
        E("boundary.total_probes", CLAIMS,
          "boundary_exposure.all_probes.total", "d", hint="7,960"),
        E("boundary.pct_of_ppl_passing", CLAIMS,
          "boundary_exposure.ppl_passing_probes.pct_of_ppl_passing", ".1f",
          hint="%", note="the manuscript prints 10.5% using a mismatched denominator"),
    ]

    # ---- gate effect space (sec 5.9 / sec 7) --------------------------------
    reg += [
        E("gate.disagreeing_pairs", CLAIMS, "ci_flips_registered.n_registered_estimands",
          "d", hint="estimands", note="registered estimand count"),
    ]

    # ---- frontier per-cell (sec 5.7 Table A) + pooled ----------------------
    fr = C.jload(FRONT) or {}
    for i, cell in enumerate(fr.get("cells") or []):
        cname = cell.get("cell", str(i))
        for meth, rec in (cell.get("methods") or {}).items():
            if not isinstance(rec, dict) or "v9" not in rec:
                continue
            drf = rec.get("draft") or {}
            ln, old = drf.get("line"), drf.get("value")
            ent = E(f"frontier.cell.{cname}.{meth}", FRONT,
                    f'cells.{i}.methods."{meth}".v9', "+.3f",
                    lines=[ln] if ln else None,
                    note=("zeroed: not a measurement" if rec.get("zeroed") else None))
            if old is not None:
                ent["expect_raw"] = f"{float(old):+.3f}"
            reg.append(ent)
    for name, rec in (fr.get("pooled_v9") or {}).items():
        if "[" in name:
            continue
        slug = name.replace(" - ", "_minus_").replace(" ", "_")
        for part, fmt in (("point", "+.3f"), ("lo", "+.3f"), ("hi", "+.3f")):
            reg.append(E(f"frontier.pooled.{slug}.{part}", FRONT,
                         f'pooled_v9."{name}".{part}', fmt, hint="edit"))

    # ---- LOPO (sec 5.4) ----------------------------------------------------
    lpo = C.jload(LOPO) or {}
    lo9 = (lpo.get("lopo") or {}).get("v9", {}).get("deltas", {})
    # the draft prints the sec 5.4 table on consecutive lines; the artifact
    # parsed them, so anchor each projection to its own line
    printed = (lpo.get("printed_values_parsed") or {}).get("draft_v3_5.4", {})
    for proj, rec in lo9.items():
        pl = printed.get(proj) if isinstance(printed.get(proj), dict) else None
        ln = (pl or {}).get("line")
        for part in ("point", "lo", "hi"):
            ent = E(f"lopo.{proj}.{part}", LOPO,
                    f'lopo.v9.deltas."{proj}".{part}', "+.3f", hint=proj,
                    lines=[ln] if ln else None)
            if pl is not None and pl.get(part) is not None:
                ent["expect_raw"] = f"{float(pl[part]):+.3f}"
            reg.append(ent)

    # ---- numbers whose ONLY source is an as-run artifact --------------------
    # These are NAMED ship-blockers: registering them against results/v6 would
    # launder an as-run value into a v9 claim, so they are declared blocked and
    # the audit fails them by id instead of reporting an anonymous stray number.
    reg += [
    ]

    # ---- SPC per-sparsity deltas (sec 5.5) ---------------------------------
    sp = C.jload(SPC) or {}
    for tier, trec in (sp.get("tiers") or {}).items():
        emitted_n = False
        for i, lvl in enumerate(trec.get("levels") or []):
            d = lvl.get("delta_vs_dense")
            if not d or lvl.get("is_dense_reference"):
                continue
            sl = str(lvl.get("sparsity")).replace(".", "_")
            for part in ("point", "lo", "hi"):
                reg.append(E(f"spc.{tier}.s{sl}.{part}", SPC,
                             f'tiers."{tier}".levels.{i}.delta_vs_dense.{part}',
                             "+.3f", hint="sparsity"))
            # One n per sparsity level would chase a single printed "n = 4",
            # and the bare hint "n =" now also matches the rewritten sec 5.5
            # binarization sentence. Emit the tier's cell count ONCE, hinted on
            # the tier, so it cannot drift onto a neighbouring count.
            if not emitted_n:
                # level 0 is the dense reference and is skipped above, so an
                # index test never fires here — use a flag.
                reg.append(E(f"spc.{tier}.n_cells", SPC,
                             f'tiers."{tier}".levels.{i}.n_cells', "d",
                             hint=f"{tier} tier"))
                emitted_n = True

    # ---- DEC coverage / unanimity counts (sec 5.3) -------------------------
    reg += [
        E("dec.unanimity.n", DEC, "unanimity.n_supporting", "d", hint="10/10"),
        E("dec.n_cells", DEC, "n_cells", "d", hint="cells"),
    ]

    # ---- gate effect space (sec 5.9 / sec 7) -------------------------------
    reg += [
        E("gate.disagreeing_pairs.count", CLAIMS,
          "gate_effect_space.disagreeing_pairs", "d", hint="pairs"),
        E("gate.grid_pairs_total", CLAIMS,
          "gate_effect_space.grid_pairs_total", ",d", hint="pairs"),
    ]

    # ---- WS2 determinism (sec 4 / sec 7) -----------------------------------
    ws = C.jload(WS2) or {}
    for k in ("replicates", "n_replicates", "identical_replicates"):
        if k in ws:
            reg.append(E(f"ws2.{k}", WS2, k, "d", hint="replicat"))

    # ---- reclaimed: C1 nulls, C2 sign structure, C3 retention (v9) ---------
    rc = C.jload(RECL) or {}
    for cellrow in (rc.get("c1_nulls") or {}).get("per_cell", []):
        base = (f"c1.{cellrow['target']}_{cellrow['axis']}_{cellrow['role']}"
                f".{cellrow['null']}")
        i = (rc["c1_nulls"]["per_cell"]).index(cellrow)
        reg.append(E(f"{base}.real", RECL, f"c1_nulls.per_cell.{i}.real", "+.3f",
                     hint=cellrow["target"]))
        reg.append(E(f"{base}.null", RECL, f"c1_nulls.per_cell.{i}.null_value",
                     "+.3f", hint=cellrow["target"]))
    for name in (rc.get("c1_nulls") or {}).get("pooled", {}):
        if "draft_table_cells" not in name:
            continue
        slug = name.split(" [")[0].replace(" ", "_")
        for part in ("point", "lo", "hi"):
            reg.append(E(f"c1.pooled.{slug}.{part}", RECL,
                         f'c1_nulls.pooled."{name}".cell_unit.{part}', "+.3f",
                         hint="pooled"))
    c2 = rc.get("c2_sign_structure") or {}
    if "_blocked" not in c2:
        for nm, key in (("real", "real_mean"), ("random", "random_mean")):
            reg.append(E(f"c2.{nm}", RECL, f"c2_sign_structure.{key}", "+.3f",
                         hint="sign"))
        for part in ("point", "lo", "hi"):
            reg.append(E(f"c2.delta.{part}", RECL,
                         f"c2_sign_structure.cell_unit.{part}", "+.3f", hint="sign"))
        reg.append(E("c2.n_sweeps", RECL, "c2_sign_structure.n_sweeps", "d",
                     hint="sweeps"))
    for tier in ("small", "big"):
        t = ((rc.get("c3_binarization") or {}).get("tiers") or {}).get(tier)
        if not t:
            continue
        reg.append(E(f"c3.{tier}.retention_pct", RECL,
                     f'c3_binarization.tiers."{tier}".retention_pct', ".1f",
                     hint="retention"))
        for part in ("point", "lo", "hi"):
            reg.append(E(f"c3.{tier}.delta.{part}", RECL,
                         f'c3_binarization.tiers."{tier}".cell_unit.{part}',
                         "+.3f", hint="binariz"))
        reg.append(E(f"c3.{tier}.n", RECL,
                     f'c3_binarization.tiers."{tier}".n_sweeps', "d", hint="n ="))

    # ---- recomputed: ENV 200-split sweep -----------------------------------
    ms = C.jload(MISS) or {}
    for tree in ("as_run", "v9"):
        sw = (((ms.get("trees") or {}).get(tree) or {}).get("env_split_sweep") or {})
        for gate, d in sw.items():
            for stat, fmt in (("mean_utility_gain", "+.4f"),
                              ("median_utility_gain", "+.4f"),
                              ("n_nondiscriminating", "d")):
                reg.append(E(f"env.{tree}.{gate}.{stat}", MISS,
                             f'trees."{tree}".env_split_sweep."{gate}".{stat}',
                             fmt, hint="split"))
        cg = (((ms.get("trees") or {}).get(tree) or {})
              .get("contrast_gap_correlation") or {})
        for lvl in ("cell_level", "row_level"):
            if cg.get(lvl):
                for part in ("point", "lo", "hi"):
                    reg.append(E(f"contrast_gap.{tree}.{lvl}.{part}", MISS,
                                 f'trees."{tree}".contrast_gap_correlation.'
                                 f'{lvl}.{part}', "+.3f", hint="contrast"))
    return reg


def _f(s):
    try:
        return float(normalize_num(s).rstrip("%"))
    except (TypeError, ValueError, AttributeError):
        return None


EXTRA = "audit/manifest_extra.yaml"


def load_extra():
    """Hand-authored registry entries, merged into the generated ones.

    The generated registry covers the estimands with a regular shape (per
    condition, per sparsity, per cell). Everything else -- one-off values,
    values that need a line + expected-old-value anchor to bind safely -- lives
    here. Each entry is the same dict shape the generator emits:

        id, artifact, key, format, [hint], [lines], [expect_raw], [note]
        or: id, blocked_as_run, why

    `lines` + `expect_raw` together are what let replace-mode write into a table
    row without risking a neighbouring column.
    """
    import yaml as _y
    fp = os.path.join(C.HERE, EXTRA)
    if not os.path.exists(fp):
        return []
    with open(fp) as fh:
        d = _y.safe_load(fh) or {}
    return d.get("entries", d if isinstance(d, list) else [])


nums_line = {}


def find_occurrences(entry, value_str, nums, used):
    """Bind an entry to draft occurrences NUMERICALLY, at the precision the
    DRAFT prints. The draft rounds to 3dp where the artifacts carry 4dp, so
    string equality would bind almost nothing; and a value that disagrees even
    at the draft's own precision must stay unbound so the audit reports it."""
    tv = _f(value_str)
    if tv is None:
        return []
    hint = (entry.get("hint") or "").lower()
    lines = entry.get("lines")
    out = []
    for n in nums:
        key = (n["line"], normalize_num(n["raw"]), n["col"])
        if key in used:
            continue
        if lines and not (lines[0] <= n["line"] <= lines[-1]):
            continue
        nv = _f(n["raw"])
        if nv is None:
            continue
        norm = normalize_num(n["raw"]).rstrip("%")
        dec = len(norm.split(".")[1]) if "." in norm else 0
        # A token with no decimals carries a +-0.5 tolerance, so a bare "0"
        # would match ANY entry resolving inside (-0.5, +0.5) -- that is how a
        # "+.3f" bound once landed on a stray "0". Require the token's
        # precision to be compatible with the entry's declared format.
        want = entry.get("format") or ""
        m_dec = re.search(r"\.(\d+)f", want)
        if m_dec and int(m_dec.group(1)) > 0 and dec == 0:
            continue
        if abs(nv - tv) <= 0.5 * (10 ** -dec) + 1e-12:
            out.append(n)
    # A hint ANNOTATES; it must not gate. Narrowing to hint-matching
    # occurrences drops the abstract's copy of a value (the abstract says
    # "recovers +0.353", not "C-ref"), which would then surface as an
    # unregistered number. Every numeric match is bound; the hint is recorded
    # so replace-mode can prefer hint-matching sites.
    if not out and lines and entry.get("expect_raw") is not None:
        # No numeric match, but the entry declares BOTH where it is printed and
        # what the stale printing says. A `*` value will never equal the v9
        # artifact, so value-matching can never bind it — but a line anchor
        # ALONE is unsafe: a Table A row carries four starred values, and
        # anchoring on the line would let this estimand claim a neighbour's
        # column. Requiring the expected old value pins the right one.
        ev = _f(entry["expect_raw"])
        for n in nums:
            if n["line"] not in lines:
                continue
            # The draft stars POINT estimates but not their CI bounds, even
            # where both are equally stale (sec 5.4's LOPO table, sec 5.2's
            # pooled nulls). An explicit line + expected-value anchor pins the
            # token tightly enough to bind an unstarred one safely; a bare line
            # anchor still would not.
            # Supplying BOTH a line and an expected value IS the intent to
            # anchor, so no separate opt-in flag is needed. A bare line anchor
            # still refuses to bind an unstarred token.
            pass
            if (n["line"], normalize_num(n["raw"]), n["col"]) in used:
                continue
            nv = _f(n["raw"])
            if nv is None or ev is None:
                continue
            norm = normalize_num(n["raw"]).rstrip("%")
            dec = len(norm.split(".")[1]) if "." in norm else 0
            if abs(nv - ev) > 0.5 * (10 ** -dec) + 1e-12:
                continue
            n["stale"] = True
            n["expect_matched"] = True
            out.append(n)
    ev = _f(entry.get("expect_raw")) if entry.get("expect_raw") is not None else None
    for n in out:
        n["hint_matched"] = bool(hint) and hint in n["context"].lower()
        # An entry that declares BOTH a line and an expected printed value has
        # identified its token exactly, however the binding was reached. Mark
        # it matched here too, not only on the anchor fallback: otherwise a
        # value that binds within tolerance but renders differently at the
        # draft's precision falls into verify-mode and reports a mismatch it
        # can never fix.
        if ev is not None and lines and n["line"] in lines:
            nv = _f(n["raw"])
            if nv is not None and abs(nv - ev) < 1e-9:
                n["expect_matched"] = True
                n["stale"] = True
    return out


def main():
    C.ensure_dirs()
    text = open(os.path.join(C.HERE, DRAFT)).read()
    # Bind against the SAME text the audit will scan: v3 + registered prose
    # edits. Scanning the un-edited source would leave every number introduced
    # by an edit permanently unregistered, and would anchor entries to line
    # content the audit no longer sees.
    text, prose = apply_prose_edits(text)
    for _i, _l in enumerate(text.split("\n"), 1):
        nums_line[_i] = _l
    bad = [p for p in prose if p.get("status") != "applied"]
    print(f"prose edits applied   : {len(prose) - len(bad)}/{len(prose)}"
          + (f"   ** {len(bad)} FAILED **" if bad else ""))
    nums = scan_numbers(text)
    reg = registry()
    extra = load_extra()
    # hand-authored entries go FIRST: they carry line/expect anchors, so letting
    # them bind before the generic value-matched entries stops a generic entry
    # from claiming a table cell that a specific one identifies exactly.
    reg = extra + reg
    print(f"registry: {len(reg) - len(extra)} generated + {len(extra)} hand-authored")

    art_cache = {}

    def load(p):
        if p not in art_cache:
            art_cache[p] = C.jload(p)
        return art_cache[p]

    entries, unresolved, unbound = [], [], []
    used = set()
    for e in reg:
        if e.get("blocked_as_run"):
            entries.append(e)
            continue
        obj = load(e["artifact"])
        if obj is None:
            unresolved.append((e["id"], f"artifact missing: {e['artifact']}"))
            continue
        raw = resolve_key(obj, e["key"])
        if raw is None:
            unresolved.append((e["id"], f"key not found: {e['key']}"))
            continue
        if e.get("extract"):                      # pull lo/hi out of a CI string
            pt, lo, hi, n = ci_parts(raw)
            raw = {"lo": lo, "hi": hi, "point": pt, "n": n}[e["extract"]]
            # Format it the same way the audit will, or the generator binds on a
            # 4-dp string while the audit writes a 3-dp value.
            try:
                shown = fmt_value(float(raw), e.get("format"))
            except (TypeError, ValueError):
                shown = raw
        else:
            shown = fmt_value(raw, e.get("format"))
        occ = find_occurrences(e, shown, nums, used)
        for o in occ:
            used.add((o["line"], normalize_num(o["raw"]), o["col"]))
        rec = {k: v for k, v in e.items() if k not in ("lines",)}
        rec.pop("expect_raw", None)
        rec["resolved"] = shown
        rec["occurrences"] = [{"line": o["line"], "col": o["col"], "raw": o["raw"],
                               "starred": o["starred"],
                               "hint_matched": o.get("hint_matched", False),
                               "stale": o.get("stale", False),
                               "expect_matched": o.get("expect_matched", False)}
                              for o in occ]
        entries.append(rec)
        if not occ:
            unbound.append((e["id"], shown))

    covered = {(l, r) for (l, r, _c) in used}
    residue = [n for n in nums if (n["line"], normalize_num(n["raw"])) not in covered]

    lines = ["# audit/manifest_v10.yaml — GENERATED by src/v10_make_manifest.py.",
             "# Every entry's value is resolved from an artifact at build time; the",
             "# occurrences below were bound by matching that value in the draft under a",
             "# context hint. Regenerate rather than hand-edit resolved values.",
             "#",
             f"# entries: {len(entries)}   bound occurrences: {len(covered)}",
             f"# draft numeric tokens: {len(nums)}   still unregistered: {len(residue)}",
             "entries:"]
    import yaml
    lines.append(yaml.safe_dump(entries, sort_keys=False, default_flow_style=False,
                                width=100).rstrip())
    fp = os.path.join(C.HERE, OUT)
    with open(fp, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"manifest entries      : {len(entries)}")
    print(f"unresolved keys       : {len(unresolved)}")
    for i, r in unresolved[:12]:
        print(f"   ? {i}: {r}")
    print(f"resolved but unbound  : {len(unbound)}  (value not found in the draft)")
    for i, v in unbound[:12]:
        print(f"   - {i} = {v}")
    print(f"draft numeric tokens  : {len(nums)}")
    print(f"bound occurrences     : {len(covered)}")
    print(f"STILL UNREGISTERED    : {len(residue)}  <- the manifest's work order")
    C.jdump({"unresolved": unresolved, "unbound": unbound,
             "residue": [{"line": n["line"], "raw": n["raw"], "starred": n["starred"],
                          "context": n["context"].strip()} for n in residue]},
            "results/v10/manifest_gaps.json")
    print(f"wrote {fp}")
    print("gaps: results/v10/manifest_gaps.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
