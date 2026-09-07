"""WS-B driver — rebuild every v10 figure from the v9 artifacts.

Each figure lives in its own module with a `build()` entry point, reads only
from results/v10/*.json, and writes figures/figN.{pdf,png} plus
figures/figN_data.csv. The CSVs are audited alongside the manuscript text
(WS-A B7), so a figure and the sentence about it cannot disagree.

Run: python3 src/v10_figures.py [1 2 3 ...]
"""
import os, sys, importlib, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v10_common as C
import v10_style as S

FIGURES = [1, 2, 3, 4, 5, 6]


def main():
    wanted = [int(a) for a in sys.argv[1:]] or FIGURES
    C.ensure_dirs()
    S.apply()
    ok, failed = [], []
    for n in wanted:
        name = f"v10_fig{n}"
        try:
            mod = importlib.import_module(name)
            if not hasattr(mod, "build"):
                raise AttributeError(f"{name} has no build()")
            mod.build()
            outs = [f"figures/fig{n}.pdf", f"figures/fig{n}.png",
                    f"figures/fig{n}_data.csv"]
            missing = [o for o in outs if not os.path.exists(os.path.join(C.HERE, o))]
            if missing:
                raise RuntimeError(f"did not produce {missing}")
            ok.append(n)
            print(f"  fig{n}: OK")
        except Exception:                                  # noqa: BLE001
            failed.append(n)
            print(f"  fig{n}: FAILED")
            traceback.print_exc()
    print(f"\nfigures built: {len(ok)}/{len(wanted)}  {ok}")
    if failed:
        print(f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
