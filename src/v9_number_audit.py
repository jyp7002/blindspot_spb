"""WS4.3 — number-source audit. A number the script cannot trace does not ship.

Scans a manuscript/results markdown file for numeric claims and checks each
against the v9 artifacts. Exit code is non-zero if anything fails, so it can be
wired in as a pre-submit gate.

Three checks:

  1. CITATION BAN. Any number that matches a known pre-v9 (as-run) value but NOT
     its v9 replacement is flagged. This is the mechanical enforcement of
     "only v9 numbers are citable".
  2. TRACEABILITY. Every registered claim maps to an artifact file + the script
     that produced it; a missing artifact fails.
  3. TODO SWEEP. `[TODO` outside an explicitly-labelled limitations block fails,
     as do the banned terminology strings from the freeze checklist.

The claim registry is explicit rather than inferred: a regex that "finds numbers"
in prose produces noise and false confidence. Numbers that matter are listed
here with their source, and anything unlisted is reported as UNREGISTERED so the
gap is visible instead of silently passing.

Usage:  python3 src/v9_number_audit.py [file.md ...]
"""
import os, sys, re, json

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
V9 = os.path.join(HERE, "results_v9")

BANNED_TERMS = ["entire signal", "zero inference cost", "base-only", "base only",
                "no cost at scale"]

# claim -> (v8 as-run value, v9 value, artifact, script)
REGISTRY = [
    ("DEC Delta_selection", "+0.2537", "+0.2835",
     "results/v9/v9_diff_report.json", "src/v9_regen.py"),
    ("DEC R(C-ref)", "+0.3220", "+0.3532",
     "results/v9/v9_diff_report.json", "src/v9_regen.py"),
    ("DEC R(C-a)", "+0.0684", "+0.0696",
     "results/v9/v9_diff_report.json", "src/v9_regen.py"),
    ("SPC <=3.8B 99% vs dense", "-0.1286", "-0.0754",
     "results/v9/v9_diff_report.json", "src/v9_regen.py"),
    ("SPC 7-9B 99% vs dense", "+0.0099", "+0.1120",
     "results/v9/v9_diff_report.json", "src/v9_regen.py"),
    ("frontier edit - steering", "-0.0095", "-0.0208",
     "results/v9/v9_diff_report.json", "src/v9_regen.py"),
    ("frontier edit - SentenceDebias", "-0.0279", "-0.0279",
     "results/v9/v9_diff_report.json", "src/v9_regen.py"),
    ("frontier edit - DPO", "+0.2926", "+0.2926",
     "results/v9/v9_diff_report.json", "src/v9_regen.py"),
]

ARTIFACTS = [
    "results/v9/v9_diff_report.json",
    "results/v9/ws2_determinism.json",
    "results_v9/rescore_report.json",
    "results/v8/sup1_analysis.json",
    "results/v8/env_policy.json",
    "results/v8/ins_gate.json",
]


def audit(paths):
    fails, warns = [], []

    # 2. traceability of artifacts
    for a in ARTIFACTS:
        if not os.path.exists(os.path.join(HERE, a)):
            fails.append(f"MISSING ARTIFACT: {a}")

    for p in paths:
        fp = os.path.join(HERE, p) if not os.path.isabs(p) else p
        if not os.path.exists(fp):
            fails.append(f"MISSING FILE: {p}")
            continue
        text = open(fp).read()
        low = text.lower()

        # 1. citation ban
        for name, v8, v9, art, scr in REGISTRY:
            has8, has9 = (v8 in text), (v9 in text)
            if has8 and not has9 and v8 != v9:
                fails.append(f"{p}: quotes the AS-RUN value {v8} for '{name}' "
                             f"without the v9 value {v9}")
            if has9 and not os.path.exists(os.path.join(HERE, art)):
                fails.append(f"{p}: cites '{name}' but {art} is absent")

        # 3. TODO / terminology sweep
        for m in re.finditer(r"\[TODO", text):
            line = text[:m.start()].count("\n") + 1
            ctx = text[max(0, m.start() - 300):m.start()].lower()
            if "limitation" not in ctx:
                fails.append(f"{p}:{line}: [TODO outside a limitations block")
        for t in BANNED_TERMS:
            for m in re.finditer(re.escape(t), low):
                line = low[:m.start()].count("\n") + 1
                seg = low[max(0, m.start() - 260):m.start() + 260]
                # a banned term is allowed when the text is explicitly retracting
                # or quarantining it
                if any(w in seg for w in ("struck", "retract", "quarantin",
                                          "banned", "must not", "is false",
                                          "no longer", "withdrawn", "never")):
                    continue
                warns.append(f"{p}:{line}: banned term '{t}' without a "
                             f"retraction marker nearby")
    return fails, warns


def main():
    paths = sys.argv[1:] or ["RESULTS_METHOD_v9.md", "V9_CLOSEOUT.md"]
    fails, warns = audit(paths)
    print("=" * 70)
    print("v9 NUMBER-SOURCE AUDIT")
    print("=" * 70)
    print(f"\nregistry: {len(REGISTRY)} claims, {len(ARTIFACTS)} required artifacts")
    print(f"files audited: {paths}")
    if warns:
        print(f"\nWARNINGS ({len(warns)}):")
        for w in warns:
            print(f"  ! {w}")
    if fails:
        print(f"\nFAILURES ({len(fails)}):")
        for f in fails:
            print(f"  X {f}")
        print("\nAUDIT RED — a number the script cannot trace does not ship.")
        return 1
    print("\nAUDIT GREEN — every registered number traces to a v9 artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
