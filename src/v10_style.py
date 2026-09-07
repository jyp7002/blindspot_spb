"""v10 figure style — ONE definition of column widths, colors and typography.

The v10 plan (WS-B B0) requires a single style file with ICLR column widths, a
colorblind-safe palette, and **fixed condition colors reused across figures**:
C-ref, C-a, the nulls, steering, SentenceDebias and DPO each own one color
everywhere they appear. A reader who learns a color in Fig 2 must not have to
relearn it in Fig 5.

Palette is Okabe-Ito, which is safe for deuteranopia, protanopia and
tritanopia, plus a neutral grey reserved for nulls and for "not a measurement".
Nothing here encodes meaning by hue alone: every figure that uses color to
separate conditions also separates them by position or marker.

ICLR is single-column with a 5.5in text width; every figure is sized as a
fraction of that so nothing is scaled at \\includegraphics time (scaling breaks
the point sizes below).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ sizes ---
TEXTWIDTH = 5.5          # ICLR \textwidth in inches
HALFWIDTH = 2.65         # side-by-side panel
FULL = (TEXTWIDTH, 3.1)
FULL_TALL = (TEXTWIDTH, 3.9)
WIDE = (TEXTWIDTH, 2.4)
HALF = (HALFWIDTH, 2.4)

# ----------------------------------------------------------------- colors ---
# Okabe-Ito
BLUE     = "#0072B2"
ORANGE   = "#E69F00"
GREEN    = "#009E73"
SKY      = "#56B4E9"
PURPLE   = "#CC79A7"
VERMIL   = "#D55E00"
YELLOW   = "#F0E442"
GREY     = "#999999"
DARKGREY = "#4D4D4D"
INK      = "#1A1A1A"

# Fixed condition colors — the same hue means the same thing in every figure.
COND = {
    "C-ref":        BLUE,
    "C-a":          ORANGE,
    "C-tensorshuf": PURPLE,
    "C-layershuf":  GREEN,
    "C-b":          SKY,
    "C-rand":       GREY,
    "C-bottom":     VERMIL,
}

METHOD = {
    "edit":           BLUE,      # same blue as C-ref: the edit IS the C-ref arm
    "steering":       ORANGE,
    "SentenceDebias": GREEN,
    "DPO":            PURPLE,
    "INLP":           VERMIL,
    "prompt":         GREY,
}

NULL_COLOR = GREY
BACKFIRE = VERMIL
OK_COLOR = BLUE

# Projections (Fig 4)
PROJ = {"q_proj": SKY, "k_proj": GREEN, "v_proj": ORANGE, "o_proj": PURPLE}

# Tiers (Fig 3)
TIER = {"small": BLUE, "big": ORANGE}


def apply():
    """Install the style. Call once at the top of the figure script."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType, not Type3 — ICLR requires it
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": DARKGREY,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": DARKGREY,
        "ytick.color": DARKGREY,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
        "errorbar.capsize": 0,
    })


def zeroline(ax, x=False, **kw):
    """The 0 reference every effect plot in this paper needs."""
    kw = dict(color=DARKGREY, lw=0.8, ls=(0, (3, 2)), zorder=1, **kw)
    (ax.axvline if x else ax.axhline)(0, **kw)


def save(fig, name, figdir=None):
    """Write PDF (camera) + PNG (draft). Returns both paths."""
    from v10_common import FIGDIR
    d = figdir or FIGDIR
    os.makedirs(d, exist_ok=True)
    pdf = os.path.join(d, f"{name}.pdf")
    png = os.path.join(d, f"{name}.png")
    # Deterministic output. Matplotlib stamps /CreationDate into the PDF and a
    # timestamp into the PNG, so an unchanged figure came out byte-different on
    # every build and dirtied six binaries in git each time `make freeze` ran.
    # The figure DATA is already stable -- fig*_data.csv, which is what the
    # audit contract is on, is byte-identical across builds -- so the images
    # should be too. Suppressing the date changes no pixel and no number.
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(png, metadata={"Software": None})
    plt.close(fig)
    return pdf, png
