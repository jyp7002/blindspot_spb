"""v10 — the three cross-cutting claims the v9 tooling never made reproducible.

Each of these backs a sentence in the manuscript, and each was, until now,
hand-typed prose rather than a computed value. The v10 rule is that a number
which cannot be traced does not ship, so they are computed here.

  1. BOUNDARY EXPOSURE (draft sec 6). "833 probes - 10.5% of all 7,960 - sit
     within one item of the boundary." The 833 is real but it is a count over
     the PERPLEXITY-PASSING probes only; 7,960 is the count over ALL probes.
     Numerator and denominator come from different populations.

  2. CI-STATUS FLIP CENSUS (draft sec 5.9 / abstract). "The program-wide
     re-score ... changed 146 cell values and exactly ONE CI status." That is
     true of the 18 estimands registered in results/v9/v9_diff_report.json.
     It is NOT true of every estimand computable from a re-scorable panel: the
     C5/C6 ablation family flips as well, and was never re-scored into the diff.

  3. RESAMPLING-UNIT SENSITIVITY. src/method_evidence.py bootstraps the SWEEP
     (target, axis, seed, designer). The project convention, stated in v9_regen
     and used for every registered estimand, is the target x axis CELL. The two
     disagree on CI status in this family, so the unit is load-bearing and must
     be disclosed wherever a C3/C5/C6 number is quoted.

Run: python3 src/v10_regen_claims.py
"""
import os, sys, json, glob, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C

PPL_BUDGET = 1.10
PPL_TOL = 1e-9
BUDGET_ITEMS = 4
N_MMLU_ITEMS = 200

# Panels that carry an alpha_trace AND a v9 re-scored twin, so every estimand on
# them is re-scorable and therefore in scope for a flip census.
ABLATE_PANELS = ("v6trace/ablate", "v6trace/ablate_big")


# ------------------------------------------------------- 1. boundary count ---

def boundary_census():
    """Every stored dmmlu probe, binned by its integer item drop."""
    # EXCLUDE panels created after the manuscript's own census. The limitation
    # this backs describes the stored probes that existed when it was written;
    # v10's own post-hoc alpha-extension panel is not part of that population,
    # and letting it in would silently move a number the paper prints.
    POST_HOC = ("v10ext",)
    files = sorted(f for f in glob.glob(
        os.path.join(C.RES, "**", "alpha_trace.jsonl"), recursive=True)
        if os.path.basename(os.path.dirname(f)) not in POST_HOC)
    rows = []
    for f in files:
        panel = os.path.relpath(os.path.dirname(f), C.RES)
        with open(f) as fh:
            for l in fh:
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if r.get("dmmlu") is None:
                    continue
                r["_panel"] = panel
                rows.append(r)

    def band(rs):
        c = collections.Counter(int(round(r["dmmlu"] * N_MMLU_ITEMS)) for r in rs)
        lo, hi = BUDGET_ITEMS - 1, BUDGET_ITEMS + 1
        per = {str(k): c.get(k, 0) for k in range(lo, hi + 1)}
        return per, sum(per.values()), len(rs)

    ppl_ok = [r for r in rows
              if r.get("ppl_ratio") is not None
              and r["ppl_ratio"] <= PPL_BUDGET + PPL_TOL]

    all_per, all_n, all_tot = band(rows)
    ppl_per, ppl_n, ppl_tot = band(ppl_ok)

    return {
        "_what": "probes within +-1 MMLU item of the 4-item budget",
        "trace_files": len(files),
        "excluded_post_hoc_panels": list(POST_HOC),
        "_population_note": ("panels created by v10 itself are excluded; this "
                             "census describes the probe population the "
                             "manuscript's sec 6 limitation refers to"),
        "all_probes": {
            "per_item_drop": all_per, "within_1": all_n, "total": all_tot,
            "pct_of_total": round(100.0 * all_n / all_tot, 2) if all_tot else None,
        },
        "ppl_passing_probes": {
            "per_item_drop": ppl_per, "within_1": ppl_n, "total": ppl_tot,
            "pct_of_ppl_passing": round(100.0 * ppl_n / ppl_tot, 2) if ppl_tot else None,
            "pct_of_all_probes": round(100.0 * ppl_n / all_tot, 2) if all_tot else None,
        },
        "manuscript_claim": {
            "text": "833 probes - 10.5% of all 7,960 - sit within one item of the boundary",
            "numerator_833_matches": ppl_n == 833,
            "denominator_7960_matches": all_tot == 7960,
            "defect": ("numerator is over the ppl-passing subset, denominator is over "
                       "all probes; the two populations differ"),
            "correct_statements": [
                f"{ppl_n} of {ppl_tot} perplexity-passing probes "
                f"({100.0 * ppl_n / ppl_tot:.1f}%)" if ppl_tot else None,
                f"{all_n} of {all_tot} probes "
                f"({100.0 * all_n / all_tot:.1f}%)" if all_tot else None,
            ],
        },
        "per_panel": {p: band([r for r in rows if r["_panel"] == p])[1]
                      for p in sorted({r["_panel"] for r in rows})},
    }


# ------------------------------------------------ 2/3. flips + unit drift ---

def _best(rs):
    v = [C.z(r.get("removal")) for r in rs]
    return float(np.max(v)) if v else float("nan")


def _by_sweep(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["target"], r["axis"], r["seed"], r["designer"])].append(r)
    return {k: _best(v) for k, v in g.items()}


def _by_cell(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["target"], r["axis"])].append(C.z(r.get("removal")))
    return {k: float(np.mean(v)) for k, v in g.items()}


def _paired_np(a, b, seed=0, n=C.BOOT_N):
    """numpy-RNG bootstrap, matching src/method_evidence.py.paired exactly."""
    ks = sorted(set(a) & set(b))
    if not ks:
        return None
    d = np.array([a[k] - b[k] for k in ks], float)
    rng = np.random.default_rng(seed)
    draws = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n)]
    return dict(point=float(np.mean(d)),
                lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)), n=len(ks))


def ablation_flips():
    """CI-status census over the C5/C6 ablation family, as-run vs v9, under
    BOTH resampling units. None of these estimands appear in v9_diff_report."""
    out = {}
    for panel in ABLATE_PANELS:
        asrun, v9 = C.rows(C.RES, panel), C.rows(C.V9, panel)
        if not asrun or not v9:
            out[panel] = {"_absent": True}
            continue
        variants = sorted({r.get("variant") for r in v9 if r.get("variant")})
        base = "binary-per_tensor-s0.0"
        rec = {}
        for v in variants:
            if v == base:
                continue
            e = {}
            for unit, grp in (("sweep", _by_sweep), ("cell", _by_cell)):
                a = _paired_np(grp([r for r in asrun if r.get("variant") == v]),
                               grp([r for r in asrun if r.get("variant") == base]))
                b = _paired_np(grp([r for r in v9 if r.get("variant") == v]),
                               grp([r for r in v9 if r.get("variant") == base]))
                if a is None or b is None:
                    continue
                e[unit] = {
                    "as_run": C.fmt(a), "v9": C.fmt(b),
                    "as_run_status": C.status(a), "v9_status": C.status(b),
                    "ci_status_change": (None if C.status(a) == C.status(b)
                                         else f"{C.status(a)} -> {C.status(b)}"),
                }
            if e:
                # does the resampling unit alone change the verdict, under v9?
                if "sweep" in e and "cell" in e:
                    e["unit_changes_v9_status"] = (
                        e["sweep"]["v9_status"] != e["cell"]["v9_status"])
                rec[v] = e
        out[panel] = rec
    return out


def gate_effect_space():
    """The complete effect space of the float-gate defect, enumerated rather
    than quoted: every accuracy pair on the 1/200 grid, and the subset the old
    and new gates disagree on."""
    n = N_MMLU_ITEMS
    total = (n + 1) ** 2
    disagree, by_drop = 0, collections.Counter()
    for i in range(n + 1):
        for j in range(n + 1):
            old = (i / n - j / n) <= 0.02          # the float comparison
            new = (i - j) <= BUDGET_ITEMS          # the integer comparison
            if old != new:
                disagree += 1
                by_drop[i - j] += 1
    return {"grid_side": n + 1, "grid_pairs_total": total,
            "disagreeing_pairs": disagree,
            "disagreements_by_item_drop": {str(k): v for k, v in sorted(by_drop.items())},
            "all_disagreements_are_4_item_drops": set(by_drop) == {BUDGET_ITEMS}}


def registered_flips():
    """The flip count the manuscript actually cites, straight from the diff."""
    diff = C.jload("results/v9/v9_diff_report.json", {}).get("diff", {})
    changed = {k: v["ci_status_change"] for k, v in diff.items()
               if v.get("ci_status_change") not in (None, "none")}
    return {"n_registered_estimands": len(diff),
            "n_status_changes": len(changed), "changes": changed}


def main():
    C.ensure_dirs()
    reg = registered_flips()
    abl = ablation_flips()

    extra = []
    for panel, rec in abl.items():
        if rec.get("_absent"):
            continue
        for variant, e in rec.items():
            for unit in ("sweep", "cell"):
                if unit in e and e[unit]["ci_status_change"]:
                    extra.append({"panel": panel, "variant": variant, "unit": unit,
                                  "change": e[unit]["ci_status_change"]})

    report = {
        "artifact": "v10_regen_claims",
        "boundary_exposure": boundary_census(),
        "gate_effect_space": gate_effect_space(),
        "ci_flips_registered": reg,
        "ci_flips_ablation_family": abl,
        "ci_flips_outside_registered_set": extra,
        "manuscript_claim_exactly_one_flip": {
            "text": "changed 146 cell values and exactly one CI status",
            "true_of_registered_set": reg["n_status_changes"] == 1,
            "n_registered_estimands": reg["n_registered_estimands"],
            "n_flips_found_outside_registered_set": len(extra),
            "defect": ("the claim is unqualified in the draft but holds only over the "
                       "estimands registered in v9_diff_report.json; the C5/C6 ablation "
                       "family is re-scorable, was never re-scored into the diff, and "
                       "does contain status changes"),
        },
        "resampling_unit": {
            "project_convention": "target x axis CELL (v9_regen.boot_paired)",
            "method_evidence_uses": "SWEEP (target, axis, seed, designer)",
            "why_it_matters": ("per-seed pooling is anti-conservative; the project "
                               "banned it in v5. C3/C5/C6 as published use the sweep unit."),
            "variants_whose_v9_status_depends_on_the_unit": [
                {"panel": p, "variant": v}
                for p, rec in abl.items() if not rec.get("_absent")
                for v, e in rec.items() if e.get("unit_changes_v9_status")
            ],
        },
    }
    fp = C.jdump(report, "results/v10/claims_audit.json")
    b = report["boundary_exposure"]
    print(f"boundary: 833-match={b['manuscript_claim']['numerator_833_matches']} "
          f"7960-match={b['manuscript_claim']['denominator_7960_matches']}  "
          f"true pct = {b['ppl_passing_probes']['pct_of_ppl_passing']}% of ppl-passing, "
          f"{b['all_probes']['pct_of_total']}% of all")
    g = report["gate_effect_space"]
    print(f"gate effect space: {g['disagreeing_pairs']} disagreeing of "
          f"{g['grid_pairs_total']} pairs; all 4-item drops="
          f"{g['all_disagreements_are_4_item_drops']}")
    print(f"registered CI flips: {reg['n_status_changes']} of {reg['n_registered_estimands']} estimands")
    print(f"CI flips OUTSIDE the registered set: {len(extra)}")
    for e in extra:
        print(f"   {e['panel']:22s} {e['variant']:28s} [{e['unit']:5s}] {e['change']}")
    print(f"wrote {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
