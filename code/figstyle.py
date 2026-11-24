"""
Shared style for the TIFS figures — IEEEtran twocolumn, EPS vector output

Applies lessons A14-16 from the ESL review:
  * No figure titles (suptitle/ax.set_title) — the caption explains the figure
  * Vector (EPS). Create no elements that get rasterized (no transparency — in EPS it falls back to raster)
  * Tidy legends: fewer entries, distinguished by **markers and line styles**, not color (safe for black-and-white printing)
  * Put units in axis labels. Push the grid to the background

Usage:
    import figstyle as FS
    fig, ax = FS.figure(FS.ONE_COL, 2.3)
    ...
    FS.save(fig, "fig_nicv_pairs")
"""

import glob as _glob
import os as _os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEEtran text widths (inch)
ONE_COL = 3.5
TWO_COL = 7.16

# Safe for black-and-white printing: gray levels + distinct markers. Do not rely on color alone
INK = "#000000"
INK2 = "#555555"
INK3 = "#999999"
BAND = "#dcdcdc"          # shuffle floor band (opaque light gray instead of transparency)
GRID = "#cccccc"

MARKERS = ["o", "^", "s", "D", "v", "P"]
LINES = ["-", "--", "-.", ":"]


def setup(fontsize=8):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.25,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure(width=ONE_COL, height=2.3, **kw):
    setup()
    return plt.subplots(figsize=(width, height), **kw)


def grid(ax, axis="y"):
    """Background grid. Drawn beneath the data."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.4, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, outdir="figs"):
    import os
    os.makedirs(outdir, exist_ok=True)
    eps = os.path.join(outdir, f"{name}.eps")
    png = os.path.join(outdir, f"{name}.png")
    fig.savefig(eps, format="eps")                 # for the paper
    fig.savefig(png, format="png", dpi=300)        # for visual checking
    plt.close(fig)
    print(f"  saved: {eps}  ({os.path.getsize(eps)//1024} KB)  + {png}")
    return eps


# ======================================================================
# Locating data files — so the figure scripts run without command-line arguments
# ======================================================================
# If the location differs, fix **only this one line** (or set the TIFS_ROOT environment variable).
DATA_ROOT = _os.environ.get("TIFS_ROOT", "/home/bahoek/data/kyber/tifs")
SCAN_DEPTH = 4


def _qcols(path):
    """Number of ge_log2_q* columns. Used to tell fine (50) from coarse (9)."""
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for k in f.readline().split(",") if k.startswith("ge_log2_q"))
    except OSError:
        return 0


IS_FINE = lambda p: _qcols(p) >= 20          # eval_qfine/eval_runs.csv
IS_COARSE = lambda p: 0 < _qcols(p) < 20     # eval/eval_runs.csv (and eval_ctrl/)


def find(rel, check=None, depth=SCAN_DEPTH):
    """Look for `rel` in DATA_ROOT -> current folder -> up to depth levels below those two, in that order.

    The exact relative path comes first, so `eval/eval_runs.csv` and `eval_qfine/eval_runs.csv`
    do not get mixed up. `check` is used to discriminate only when scanning downward. Returns None if not found.
    """
    for c in (_os.path.join(DATA_ROOT, rel), rel):
        if _os.path.isfile(c) and (check is None or check(c)):
            return c
    name = _os.path.basename(rel)
    for base in (DATA_ROOT, "."):
        if not _os.path.isdir(base):
            continue
        for d in range(0, depth + 1):
            for c in sorted(_glob.glob(_os.path.join(base, *(["*"] * d), name))):
                if _os.path.isfile(c) and (check is None or check(c)):
                    return c
    return None


def need(rel, what, check=None):
    """Same as find, but if not found, stops **stating where it looked**."""
    p = find(rel, check)
    if not p:
        raise SystemExit(
            f"{what} not found ({rel}).\n"
            f"  Searched: {DATA_ROOT} and the current folder, {SCAN_DEPTH} levels below them\n"
            f"  Fix: edit DATA_ROOT in figstyle.py, or prefix the command with TIFS_ROOT=/path,\n"
            f"        or pass the path to that script directly as an argument")
    print(f"  {what:<24} {p}")
    return p
