#!/usr/bin/env python3
"""R1 E7b, registered arms -- table columns and the pronoun check. POST HOC, reports only.

    python3 src/r1_e7b_detail.py

The registered reading of E7b (E7-1 / E7-2) is made by src/r1_e7b.py and is not
touched here. This script was written after that reading was known. It adds what
the comparison table and the degeneracy caveat need for the registered DPO design
({lr 1e-5, 5e-5} x merge {1, 2}: four configurations per run, cells with at least
8 differing pairs):

  * the configuration each run selects (best in-budget removal, none -> 0), the
    medians of dMMLU and of the perplexity ratio over the selected configurations,
    and the share of all configurations in budget -- the columns of the table;
  * on the four occupation cells, mean P("she") on female- and male-stereotyped
    probe occupations before training and at the selected configuration, from the
    stored per-item log-likelihoods; a run is `collapsed` when both are above 0.95
    or both below 0.05 (the rule of src/r1_e7b.py);
  * the same for the 1% edit (C-ref, fixed grid) on the same four cells, from the
    E5 panel, so that the two can be put side by side.

Cell = seed mean; bootstrap over cells (v10_common). Writes
results/r1/E7b/e7b_registered_detail.json.
"""
import collections
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v10_common as V    # noqa: E402
import r1_e7b as B        # noqa: E402
import r1_e13 as E13      # noqa: E402

ARMS = ("lr1e-5", "lr5e-5")
OCC = [c for c in B.CELLS if c[1] == "occ_gender"]


def collapsed(f, m):
    return bool((f > 0.95 and m > 0.95) or (f < 0.05 and m < 0.05))


def main():
    rows = B.load()
    need = [(t, a, s, an) for t, a in B.CELLS for s in B.SEEDS for an in ARMS]
    if any(k not in rows for k in need):
        print("incomplete: the registered arms are not all on disk")
        return 1
    occ, n_tpl = B.probe_occ()
    runs, n_ok, n_pts = {}, 0, 0
    for t, a in B.CELLS:
        for s in B.SEEDS:
            pts = []
            for an in ARMS:
                r = rows[(t, a, s, an)]
                if not r.get("skipped"):
                    pts += [dict(p, arm=an) for p in r["points"]]
            n_ok += sum(bool(p["ok"]) for p in pts)
            n_pts += len(pts)
            b = B.best(pts) if pts else None
            rec = dict(trained=bool(pts), selected=None, removal=0.0,
                       pre_skew=rows[(t, a, s, ARMS[0])].get("pre_skew"))
            if b is not None:
                rec.update(selected=dict(arm=b["arm"], merge=float(b["merge"])),
                           removal=float(b["removal"]), dmmlu_items=int(b["dmmlu_items"]),
                           ppl_ratio=float(b["ppl_ratio"]), post_skew=float(b["post_skew"]))
                if a == "occ_gender":
                    fp = os.path.join(B.OUT, "probe_ll",
                                      f"{t}__{a}__s{s}__{B.DESIGNER[t]}__{b['arm']}.npz")
                    zf = np.load(fp)
                    i = [j for j, m in enumerate(zf["merge"]) if float(m) == float(b["merge"])]
                    if len(i) != 1:
                        raise SystemExit(f"merge {b['merge']} not stored exactly once in {fp}")
                    pre = B.pshe(zf["pre_ll"], occ, n_tpl)
                    post = B.pshe(zf["ll"][i[0]], occ, n_tpl)
                    if abs((post[0] - post[1]) - b["post_skew"]) > 1e-5:
                        raise SystemExit(f"gap of the stored likelihoods != recorded skew: {fp}")
                    rec.update(pre_f=pre[0], pre_m=pre[1], f=post[0], m=post[1],
                               collapsed=collapsed(*post))
            runs[(t, a, s)] = rec

    sel = [v for v in runs.values() if v["selected"] is not None]
    cm = B.cell_mean({k: v["removal"] for k, v in runs.items()})
    out = dict(
        script="src/r1_e7b_detail.py", status="post hoc; reports only",
        removal=V.boot_mean([cm[c] for c in B.CELLS]),
        cells={f"{t}|{a}": cm[(t, a)] for t, a in B.CELLS},
        dmmlu_items_median=float(np.median([v["dmmlu_items"] for v in sel])),
        ppl_ratio_median=float(np.median([v["ppl_ratio"] for v in sel])),
        configs_in_budget=f"{n_ok}/{n_pts}", budget_pass_rate=n_ok / n_pts,
        runs_trained=sum(v["trained"] for v in runs.values()),
        runs_with_nothing_in_budget=sum(v["trained"] and v["selected"] is None for v in runs.values()),
        selected_configs=dict(collections.Counter(
            f"{v['selected']['arm']} x merge {v['selected']['merge']:g}" for v in sel)))

    # ---- occupation cells: where the pronoun goes, DPO and the 1% edit side by side
    def side(get):
        cells = {}
        for t, a in OCC:
            vs = [get(t, s) for s in B.SEEDS]
            cells[t] = dict(pre_f=float(np.mean([v["pre_f"] for v in vs])),
                            pre_m=float(np.mean([v["pre_m"] for v in vs])),
                            f=float(np.mean([v["f"] for v in vs])),
                            m=float(np.mean([v["m"] for v in vs])),
                            removal=float(np.mean([v["removal"] for v in vs])),
                            collapsed_runs=int(sum(v["collapsed"] for v in vs)))
        for c in cells.values():
            c["shift"] = (c["f"] + c["m"]) / 2 - (c["pre_f"] + c["pre_m"]) / 2
        ts = [t for t, _ in OCC]
        return dict(cells=cells,
                    pre_f=float(np.mean([cells[t]["pre_f"] for t in ts])),
                    pre_m=float(np.mean([cells[t]["pre_m"] for t in ts])),
                    f=float(np.mean([cells[t]["f"] for t in ts])),
                    m=float(np.mean([cells[t]["m"] for t in ts])),
                    shift=V.boot_mean([cells[t]["shift"] for t in ts]),
                    removal=V.boot_mean([cells[t]["removal"] for t in ts]),
                    collapsed_runs=f"{sum(cells[t]['collapsed_runs'] for t in ts)}/{len(ts) * len(B.SEEDS)}")

    def dpo(t, s):
        v = runs[(t, "occ_gender", s)]
        if v["selected"] is None:
            raise SystemExit(f"occupation run without a selected configuration: {t} s{s}")
        return v

    arms, _ = E13.arm_rows()
    el = arms["elicited"]
    occ13, n13 = E13.probe_occ()

    def edit(t, s):
        r, pdir = el[(t, s)]
        u = E13.unit_stats(r, pdir, occ13, n13)
        g = u["grid"]
        return dict(pre_f=u["pre_f"], pre_m=u["pre_m"], f=g["f"], m=g["m"],
                    removal=g["removal"], collapsed=g["collapsed"])

    out["occupation"] = dict(dpo_registered=side(dpo), edit_1pct_fixed_grid=side(edit))
    out["runs"] = {f"{t}|{a}|s{s}": v for (t, a, s), v in runs.items()}
    with open(os.path.join(B.OUT, "e7b_registered_detail.json"), "w") as f:
        json.dump(out, f, indent=1)

    def fmt(d):
        return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"

    print(f"registered DPO design, 8 cells: removal {fmt(out['removal'])}")
    print(f"  median dMMLU {out['dmmlu_items_median']}, median ppl ratio {out['ppl_ratio_median']:.3f}, "
          f"configurations in budget {out['configs_in_budget']} ({100 * out['budget_pass_rate']:.1f}%), "
          f"runs trained {out['runs_trained']}/24, nothing in budget {out['runs_with_nothing_in_budget']}")
    print(f"  selected: {out['selected_configs']}")
    for name, label in (("dpo_registered", "DPO, registered design"),
                        ("edit_1pct_fixed_grid", "1% edit, fixed grid")):
        o = out["occupation"][name]
        print(f"\n  [{label}] four occupation cells: P(she) f/m {o['pre_f']:.2f} / {o['pre_m']:.2f} -> "
              f"{o['f']:.2f} / {o['m']:.2f}; base-rate shift {fmt(o['shift'])}; removal "
              f"{fmt(o['removal'])}; collapsed runs {o['collapsed_runs']}")
        for t, c in o["cells"].items():
            print(f"    {t:6s} {c['pre_f']:.2f} / {c['pre_m']:.2f} -> {c['f']:.2f} / {c['m']:.2f}   "
                  f"shift {c['shift']:+.3f}   removal {c['removal']:+.3f}   collapsed {c['collapsed_runs']}/3")
    print("\n  per occupation run (DPO): selected configuration, P(she) f/m after")
    for (t, a, s), v in runs.items():
        if a == "occ_gender":
            sc = v["selected"]
            print(f"    {t:6s} s{s}  {sc['arm']:7s} merge {sc['merge']:g}   {v['f']:.2f} / {v['m']:.2f}"
                  f"   removal {v['removal']:+.3f}   ppl x{v['ppl_ratio']:.3f}"
                  f"{'   COLLAPSED' if v['collapsed'] else ''}")
    print(f"\nwrote {os.path.relpath(os.path.join(B.OUT, 'e7b_registered_detail.json'), REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
