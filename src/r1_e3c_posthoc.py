"""POST HOC to src/r1_e3c.py: the seven variants on the ten main cells with a
gender-balanced occupation split (6 f + 6 m per half) instead of the registered
hash split (7 f + 5 m / 5 f + 7 m). Pair axes keep the registered split.
Writes results/r1/E3/e3c_balanced_posthoc.json (both this and the comparison)."""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "src")
import v10_common as V
import v12_astar as A
import r1_e5 as E5
import r1_e13 as E13
import r1_e3c as S3
import r1_e3c_front as F


def fmt(d):
    return f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


occ, n_tpl = E13.probe_occ()
h = lambda o: hashlib.sha256(o.encode("utf-8")).hexdigest()   # noqa: E731
half = [None] * len(occ)
for g in ("f", "m"):
    idx = sorted([i for i, (o, gg) in enumerate(occ) if gg == g], key=lambda i: h(occ[i][0]))
    for r, i in enumerate(idx):
        half[i] = r % 2
split = json.load(open(S3.SPLIT_FP))
split["occ_gender"]["half"] = half
orig_load = json.load


def patched(fp, *a, **k):
    if getattr(fp, "name", "") == S3.SPLIT_FP:
        return split
    return orig_load(fp, *a, **k)


json.load = patched
runs, missing = S3.load_runs()
runsF, missF, badF = F.load_all()
json.load = orig_load
assert not missing and not missF and not badF

out = dict(status="post hoc", occupation_split="within each gender, occupations ordered by sha256 and dealt alternately (6 f + 6 m per half)",
           ten_main_cells={}, comparison={})
print("ten main cells, seven variants, balanced occupation split")
for arm in S3.ARMS:
    _, full = S3.cells(runs, arm, "full")
    _, sp = S3.cells(runs, arm, "split")
    _, same = S3.cells(runs, arm, "same")
    ks = [k for k in runs if k[3] == arm]
    nd = sum(runs[k]["alpha_h"][0] != runs[k]["alpha_h"][1] for k in ks)
    rec = dict(full=V.boot_mean(full), split=V.boot_mean(sp), same=V.boot_mean(same),
               optimism=V.boot_paired(list(zip(same, sp))), halves_disagree=f"{nd}/{len(ks)}")
    out["ten_main_cells"][arm] = rec
    print(f"  {arm:13s} all {fmt(rec['full'])}  other half {fmt(rec['split'])}  same half {rec['same']['point']:+.3f}  "
          f"optimism {fmt(rec['optimism'])}  halves disagree {nd}/{len(ks)}")
for name, (x, y) in {"C-ref - C-a": ("C-ref", "C-a"), "C-a - C-rand": ("C-a", "C-rand"),
                     "C-ref - C-b": ("C-ref", "C-b"), "C-ref - C-layershuf": ("C-ref", "C-layershuf"),
                     "C-ref - C-bottom": ("C-ref", "C-bottom"), "C-ref - C-rand": ("C-ref", "C-rand")}.items():
    rec = {}
    for key in ("full", "split"):
        _, vx = S3.cells(runs, x, key)
        _, vy = S3.cells(runs, y, key)
        rec[key] = V.boot_paired(list(zip(vx, vy)))
    out["ten_main_cells"][name] = rec
    print(f"  {name:22s} all {fmt(rec['full'])}  other half {fmt(rec['split'])}")

print("eight comparison cells, balanced occupation split")
for m in F.METHODS:
    full, sp, same = (F.cell_vals(runsF, m, k) for k in ("full", "split", "same"))
    ks = [k for k in runsF if k[3] == m and runsF[k]["i_full"] is not None]
    nd = sum(runsF[k]["i_h"][0] != runsF[k]["i_h"][1] for k in ks)
    rec = dict(full=V.boot_mean(full), split=V.boot_mean(sp), same=V.boot_mean(same),
               optimism=V.boot_paired(list(zip(same, sp))), halves_disagree=f"{nd}/{len(ks)}")
    out["comparison"][m] = rec
    print(f"  {F.NAME[m]:16s} all {fmt(rec['full'])}  other half {fmt(rec['split'])}  same half {rec['same']['point']:+.3f}  "
          f"optimism {fmt(rec['optimism'])}  halves disagree {nd}/{len(ks)}")
for x in ("edit", "edit_1pct"):
    for y in ("steering", "sentdebias", "dpo"):
        rec = {k: V.boot_paired(list(zip(F.cell_vals(runsF, x, k), F.cell_vals(runsF, y, k)))) for k in ("full", "split")}
        out["comparison"][f"{F.NAME[x]} - {F.NAME[y]}"] = rec
        print(f"  {F.NAME[x]:10s} - {F.NAME[y]:15s} all {fmt(rec['full'])}  other half {fmt(rec['split'])}")
json.dump(out, open(os.path.join("results", "r1", "E3", "e3c_balanced_posthoc.json"), "w"), indent=1)
print("wrote results/r1/E3/e3c_balanced_posthoc.json")
