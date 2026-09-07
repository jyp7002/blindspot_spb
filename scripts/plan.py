#!/usr/bin/env python3
"""Plan — expand a panel config into an ordered, resumable work-unit list.

This is the seam that makes the launchers environment-neutral. It writes ONE
file, `work/<panel>.units.json`, and both `submit_local.sh` and
`submit_slurm.sh` index into it. A SLURM array task and a local worker
therefore agree on what unit 37 is without coordinating.

Resume is read from the panel's own artifacts, not from a ledger: a unit is
complete when its rows are on disk. Re-running plan.py after a crash produces a
shorter list; re-running it after nothing produces the same list.

  python3 scripts/plan.py configs/v11/dec_v11.yaml
  python3 scripts/plan.py configs/v11/dec_v11.yaml --all      # ignore resume
  python3 scripts/plan.py configs/v11/dec_v11.yaml --dry-run  # print, no write
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import v11_panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--all", action="store_true",
                    help="emit every unit, including ones already complete")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, metavar="KEY",
                    help="restrict to cells with this config key set truthy "
                         "(e.g. --only published, to replay just the cells "
                         "being checked for reproduction)")
    a = ap.parse_args()

    cfg = v11_panel.load(a.config)
    units = v11_panel.expand(cfg, only=a.only)
    todo = units if a.all else v11_panel.pending(units)

    suffix = f".{a.only}" if a.only else ""
    out = a.out or os.path.join(REPO, "work",
                                f"{cfg['out_panel']}{suffix}.units.json")

    print(f"config     : {a.config}")
    print(f"panel      : {cfg['out_panel']}  (kind={cfg['kind']}, "
          f"base_on={cfg.get('base_on', '-')})")
    print(f"cells      : {len(cfg['cells'])}")
    print(f"units      : {len(units)} total, {len(units) - len(todo)} done, "
          f"{len(todo)} to run")
    if cfg.get("excluded"):
        print(f"excluded   : {len(cfg['excluded'])} cell(s) deliberately not run")

    if not todo:
        print("\nnothing to do — the panel is complete on disk.")

    # Per-checkpoint grouping is what actually drives wall-clock: a worker that
    # gets all of one model's units loads that checkpoint once.
    by_hf = {}
    for u in todo:
        by_hf.setdefault(u["hf"], []).append(u)
    if by_hf:
        print("\n  units per checkpoint (group these onto one worker):")
        for hf, us in sorted(by_hf.items(), key=lambda kv: -len(kv[1])):
            print(f"    {len(us):3d}  {hf}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(config=os.path.relpath(a.config, REPO),
                       panel=cfg["out_panel"], kind=cfg["kind"],
                       n_total=len(units), units=todo), f, indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}  ({len(todo)} units)")
    print(f"  local : bash scripts/submit_local.sh {os.path.relpath(out, REPO)}")
    print(f"  slurm : sbatch --array=0-{max(len(todo) - 1, 0)} "
          f"scripts/submit_slurm.sh {os.path.relpath(out, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
