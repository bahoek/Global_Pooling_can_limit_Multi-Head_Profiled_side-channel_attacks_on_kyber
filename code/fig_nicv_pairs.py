"""
fig_nicv_pairs — per-pair NICV peaks: 39 signal pairs (r0·r1), shuffle floor, 8 phantom pairs, L=39 boundary

What it shows
-------------
Within the window [0,40000), the 39 signal pairs have NICV 0.76~0.92 (27~191x the shuffle floor),
while pairs 39 and above (phantom) are **at or below their own shuffle floor** (ratio 0.52~1.02).
That is, the window contains no signal for pairs 39 and above -> the basis for L = 39.

Data
----
  eval/nicv/nicv_coef.csv   one row per coefficient (78 signal + 8 phantom). peak_nicv / shuf_peak_nicv / ratio_to_floor

Run
----
    python fig_nicv_pairs.py [path to nicv_coef.csv]
The default path is eval/nicv/nicv_coef.csv relative to OUT_DIR of eval_nscan.py.
Produces figs/fig_nicv_pairs.eps (for the paper) + .png (for checking).
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS

LEAK_PAIRS = 39


def default_path():
    return FS.find("eval/nicv/nicv_coef.csv") or "nicv_coef.csv"


def main(path=None):
    path = path or (sys.argv[1] if len(sys.argv) > 1 else default_path())
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist. Pass the path as an argument: python fig_nicv_pairs.py <path>")
    R = list(csv.DictReader(open(path, encoding="utf-8")))
    f = lambda rs, k: np.array([float(r[k]) for r in rs])
    g = lambda rs, k: np.array([int(r[k]) for r in rs])

    sig = [r for r in R if r["signal"] == "1"]
    pha = [r for r in R if r["signal"] == "0"]
    s_r0 = [r for r in sig if r["role"] == "r0"]
    s_r1 = [r for r in sig if r["role"] == "r1"]
    floor_all = f(R, "shuf_peak_nicv")
    print(f"signal {len(sig)} coefficients (r0 {len(s_r0)}, r1 {len(s_r1)}), phantom {len(pha)} coefficients")
    print(f"  signal peak {f(sig,'peak_nicv').min():.4f}~{f(sig,'peak_nicv').max():.4f}, "
          f"ratio {f(sig,'ratio_to_floor').min():.1f}~{f(sig,'ratio_to_floor').max():.1f}")
    print(f"  phantom peak {f(pha,'peak_nicv').min():.5f}~{f(pha,'peak_nicv').max():.5f}, "
          f"ratio {f(pha,'ratio_to_floor').min():.2f}~{f(pha,'ratio_to_floor').max():.2f}")
    print(f"  shuffle floor {floor_all.min():.5f}~{floor_all.max():.5f}")

    # Break the x axis: left is the 39 signal pairs (dense), right is the phantom sample (sparse).
    # Putting all of 0~127 on one axis crowds the 78 signals into 30% of the width, which prints as a black band
    FS.setup()
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FS.ONE_COL, 2.35), sharey=True,
                                   gridspec_kw=dict(width_ratios=[3.1, 1.0], wspace=0.06))

    for ax in (axL, axR):
        FS.grid(ax, "y")
        # shuffle floor band — opaque light gray, no transparency (so it is not rasterized in EPS)
        ax.axhspan(floor_all.min(), floor_all.max(), facecolor=FS.BAND, edgecolor="none", zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(3e-3, 1.6)
        ax.plot(g(s_r0, "pair"), f(s_r0, "peak_nicv"), FS.MARKERS[0], color=FS.INK,
                markerfacecolor=FS.INK, markeredgewidth=0, markersize=2.6,
                label=r"signal $r[0]$", zorder=4)
        ax.plot(g(s_r1, "pair"), f(s_r1, "peak_nicv"), FS.MARKERS[1], color=FS.INK,
                markerfacecolor="none", markeredgewidth=0.6, markersize=3.0,
                label=r"signal $r[1]$", zorder=4)
        ax.plot(g(pha, "pair"), f(pha, "peak_nicv"), FS.MARKERS[2], color=FS.INK,
                markerfacecolor=FS.INK3, markeredgewidth=0.6, markersize=3.4,
                label="phantom", zorder=4)

    # the L = 39 boundary is at the right end of the left panel
    axL.axvline(LEAK_PAIRS - 0.5, color=FS.INK2, linewidth=0.7, linestyle="--", zorder=2)
    axL.set_xlim(-1.5, 41)
    axR.set_xlim(44, 132)
    axL.set_xticks([0, 10, 20, 30, 39])
    axR.set_xticks([64, 96, 127])

    # axis-break marks: hide the facing spines and draw two slashes
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axL.tick_params(right=False, which="both")
    axR.tick_params(left=False, which="both")
    kw = dict(transform=fig.transFigure, color=FS.INK, linewidth=0.6, clip_on=False, zorder=10)
    import matplotlib.lines as mlines
    for xb in (0.0,):                                   # placeholder — drawn below after computing coordinates
        pass
    fig.canvas.draw()
    bl, br = axL.get_position(), axR.get_position()
    xm = (bl.x1 + br.x0) / 2.0
    d = 0.012
    for yy in (bl.y0, bl.y1):
        fig.add_artist(mlines.Line2D([xm - d, xm + d], [yy - d * 0.9, yy + d * 0.9], **kw))

    axL.set_ylabel("peak NICV")
    fig.supxlabel("coefficient pair index", fontsize=8, y=-0.035)

    # direct labels — point inside the plot without adding legend entries
    axL.annotate(r"$L=39$", xy=(LEAK_PAIRS - 0.5, 1.6), xytext=(29.0, 0.33),
                 color=FS.INK2, fontsize=7, va="center", ha="center")
    axR.annotate("shuffled-label\nfloor", xy=(127, floor_all.max()), xytext=(88, 1.2e-2),
                 color=FS.INK2, fontsize=6.5, va="center", ha="center", linespacing=1.1)

    axL.legend(loc="upper left", bbox_to_anchor=(0.01, 0.80), handletextpad=0.3)
    FS.save(fig, "fig_nicv_pairs")


if __name__ == "__main__":
    main()
