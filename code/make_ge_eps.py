r"""
16 GE-curve EPS files — panels to be assembled as subfigures in LaTeX

Produces one EPS per (backbone, N), with the same file names as the old manuscript.
    resnet_ge_N{9,19,29,39,49,69,99,128}.eps
    MobileNetV2_ge_N{9,19,29,39,49,69,99,128}.eps

In one panel
  * **one colored curve per seed** = $\log_2$GE vs Q for each seed. No averaging
    - solid = converged by the recovery criterion, dotted = not converged. The color indicates the seed number and is consistent across the 16 panels
  * two horizontal lines        = GE=1 (y=0), chance $\log_2(q^2/2)$ = 22.40
  * top left          = backbone, N and the number of converged seeds k/n

The mean line is not drawn by default (`SHOW_MEAN = False`). The Q at which GE drops differs per seed, and
averaging erases all of that variance — and that variance is exactly what should be visible when comparing across N.
To turn it on, `SHOW_MEAN = True`.

The panels are 1.8 in, so a legend does not fit. Instead **`ge_legend.eps` is produced separately** — a single horizontal strip
to be attached below the 4x4 grid with `\includegraphics[width=\textwidth]{ge_legend}`.

The convergence criterion is **`pairs_recovered == n_pairs_eval` (recovery criterion)**. The `converged` column is
accuracy-based and disagrees in 4 runs (MobileNet N34, N49, N69, N128, 1 each) — to stay consistent
with the other figures, only the recovery criterion is used here too.

The data are `ge_log2_q1..50` from the **fine** evaluation (`eval_qfine/eval_runs.csv`). On the coarse side the Q grid is
log-spaced (1,2,5,10,20,...), so the curves look kinked.

Run — **just run it without arguments**
---------------------------------------
    python make_ge_eps.py

The script finds the data itself. Search order: `DATA_ROOT` → the current folder → scanning up to 3 levels below both.
If the location differs, edit only the single `DATA_ROOT` line at the top or set the environment variable `TIFS_ROOT=/some/where`.
If you still want to give them directly, you can pass arguments as before:
    python make_ge_eps.py <fine eval_runs.csv> [output folder] [coarse eval_runs.csv]

Caution — `ResNet N128 s0`
    With coarse (Q<=500) it recovers 39/39 (Q*=100), but fine only goes up to Q<=50, so it is 34/39. Passing coarse
    as the third argument gives 8/10, the same as Table 2 in the main text, and that seed is **solid yet does not reach GE=1
    within Q<=20.** It is not wrong — it succeeded at Q=100; one line in the caption is enough.

LaTeX assembly example (4x4 in a two-column figure*)
----------------------------------------------------
    \begin{figure*}[t]\centering
    \foreach ... or directly:
    \begin{subfigure}{0.24\textwidth}\includegraphics[width=\linewidth]{resnet_ge_N9}
      \caption{}\label{fig:ge-rn9}\end{subfigure}\hfill
    ... (16 of them) ...
    \caption{...}\label{fig:ge}
    \end{figure*}
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================================
# Data location. If it differs, edit **only this one line** (or the TIFS_ROOT environment variable).
DATA_ROOT = os.environ.get("TIFS_ROOT", "/home/bahoek/data/kyber/tifs")
FINE_REL = "eval_qfine/eval_runs.csv"     # all of Q=1..50. For the curves
COARSE_REL = "eval/eval_runs.csv"         # Q<=500. For the convergence verdict (basis of Table 2 in the main text)
OUT_DIR = "ge_eps"                        # created under the current folder

N_LIST = [9, 19, 29, 39, 49, 69, 99, 128]
NAME = {"resnet": "resnet", "mobilenet": "MobileNetV2"}      # file-name prefix (same as the old manuscript)
TAG = {"resnet": "ResNet18", "mobilenet": "MobileNetV2"}     # label inside the panel

QMAX = 20            # every curve bottoms out at Q<=13. Drawing up to 50 would be mostly dead space
W, H = 1.80, 1.50    # inch. Fits 0.24\textwidth when placed 4x4 in a figure*
# Margins are **fixed**. With bbox="tight" the 16 sizes would vary with label length and
# the panels would not line up in LaTeX. Fixed margins give all 16 the same bounding box
MARGIN = dict(left=0.205, right=0.975, bottom=0.215, top=0.97)
FONT = 6
CHANCE = float(np.log2(3329 * 3329 / 2))                     # 22.40 (not log2(q^2))
YLIM = (-1.5, 26.5)   # stalled seeds lie on chance (22.40), so space is left at the top for the label
SHOW_MEAN = False    # if True, overlay the mean of converged seeds as a thick black line
LW_SEED = 0.65       # seed curve line width
SHOW_XLABEL = True   # subfigures are independent of each other, so each needs its own axis labels.
SHOW_YLABEL = True   # to attach them only to the bottom row / left column, turn them off per file and run twice
# ==========================================================


# seed -> color. Fixed so that the same seed has the same color in all 16 panels.
# ResNet N128 has extra seeds up to 10~14 because of R5, so 16 colors are used.
SEED_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
    "#17becf", "#bcbd22", "#e377c2", "#7f7f7f", "#393b79", "#b5651d",
    "#637939", "#8c6d31", "#843c39", "#7b4173",
]


def seed_color(seed):
    return SEED_COLORS[seed % len(SEED_COLORS)]


def n_qcols(path):
    """Number of ge_log2_q* columns. Used to tell whether a file is fine or coarse."""
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for k in f.readline().split(",") if k.startswith("ge_log2_q"))
    except OSError:
        return 0


def find_csv(rel, want_fine):
    """Find eval_runs.csv automatically. None if not found.

    Order: DATA_ROOT/rel -> ./rel -> scan up to 3 levels below DATA_ROOT and ./ and
    tell fine/coarse apart by the number of ge_log2_q columns (fine has 50, coarse has 9).
    """
    for base in (DATA_ROOT, "."):
        c = os.path.join(base, rel)
        if os.path.exists(c) and n_qcols(c) > 0:
            return c
    import glob
    best = None
    for base in (DATA_ROOT, "."):
        if not os.path.isdir(base):
            continue
        for d in range(1, 4):
            for c in glob.glob(os.path.join(base, *(["*"] * d), "eval_runs.csv")):
                k = n_qcols(c)
                if k == 0:
                    continue
                is_fine = k >= 20
                if is_fine == want_fine and (best is None or k > n_qcols(best)):
                    best = c
    return best


def conv_set(path, mode="sum"):
    """Read only the convergence verdict -> {(arch, N, seed)} (recovery criterion)."""
    out = set()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["eval_mode"] == mode and int(r["pairs_recovered"]) == int(r["n_pairs_eval"]):
            out.add((r["arch"], int(r["n_pairs"]), int(r["seed"])))
    return out


def load(path, mode="sum"):
    """(arch, N) -> (Q array, [(seed, y, converged), ...])"""
    qs, out = None, defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["eval_mode"] != mode:
            continue
        if qs is None:
            qs = sorted(int(k[len("ge_log2_q"):]) for k in r if k.startswith("ge_log2_q"))
        out[(r["arch"], int(r["n_pairs"]))].append((
            int(r["seed"]),
            np.array([float(r[f"ge_log2_q{q}"]) for q in qs]),
            int(r["pairs_recovered"]) == int(r["n_pairs_eval"])))
    if qs is None:
        raise SystemExit("No ge_log2_q* columns — check that this is the eval_runs.csv of the fine evaluation")
    return np.array(qs), out


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": FONT, "axes.labelsize": FONT,
        "xtick.labelsize": FONT - 0.5, "ytick.labelsize": FONT - 0.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.bbox": "standard",
        "ps.fonttype": 42, "pdf.fonttype": 42,
    })


def panel(qs, runs, arch, n, out_dir):
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(**MARGIN)
    ax.grid(True, axis="y", color="#cccccc", linewidth=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)

    conv = [y for _, y, c in runs if c]
    # One line per seed. Color = seed number (common to all panels), solid = converged / dotted = not converged.
    # Converged ones are drawn on top so they are not hidden.
    for seed, y, c in sorted(runs, key=lambda r: (r[2], -r[0])):
        ax.plot(qs, y, "-" if c else ":", color=seed_color(seed),
                linewidth=LW_SEED, zorder=3 + int(c),
                solid_capstyle="round", dash_capstyle="round")
    if SHOW_MEAN and conv:
        ax.plot(qs, np.mean(conv, 0), "-", color="black", linewidth=1.3, zorder=6)
    ax.axhline(CHANCE, color="0.60", linewidth=0.45, linestyle=":", zorder=2)
    ax.axhline(0.0, color="0.35", linewidth=0.45, linestyle="--", zorder=2)

    # Put it on one line. With two lines the lower line overlaps the stalled seeds lying on the chance line
    ax.text(0.04, 0.955, f"{TAG[arch]}  $N{{=}}{n}$,  {len(conv)}/{len(runs)}",
            transform=ax.transAxes, fontsize=FONT, ha="left", va="top")

    ax.set_xlim(0.6, QMAX + 0.4)
    ax.set_xticks([1, 10, 20])
    ax.set_ylim(*YLIM)
    ax.set_yticks([0, 10, 20])
    if SHOW_XLABEL:
        ax.set_xlabel("traces $Q$", labelpad=1.5)
    if SHOW_YLABEL:
        ax.set_ylabel("$\\log_2$ GE", labelpad=1.5)

    name = f"{NAME[arch]}_ge_N{n}"
    p = os.path.join(out_dir, name + ".eps")
    fig.savefig(p, format="eps")
    fig.savefig(os.path.join(out_dir, name + ".png"), format="png", dpi=300)
    plt.close(fig)
    # The mean is not drawn, but the numbers for the caption are left on the console
    m = np.mean(conv, 0) if conv else None
    q0 = [int(qs[np.argmax(y <= 0.0)]) if (y <= 0.0).any() else None
          for _, y, c in runs if c]
    rng = (f"   GE=1 reached at Q {min(q0)}~{max(q0)}" if q0 and all(v is not None for v in q0)
           else "")
    at = {}                                    # look up by Q number (does not break if the grid changes)
    for q in (5, 10):
        w = np.nonzero(qs == q)[0]
        at[q] = f"@{q}={m[w[0]]:5.2f}" if (m is not None and len(w)) else ""
    print(f"  {name+'.eps':<28} converged {len(conv)}/{len(runs)}"
          + (f"   mean GE {at[5]} {at[10]}{rng}" if m is not None else "   (0 converged)"))
    return p


def legend_strip(seeds, out_dir, width=7.16):
    """One horizontal legend to attach below the 4x4 grid. The panels are 1.8in, so it cannot go inside."""
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=seed_color(s_), lw=1.2, label=f"seed {s_}")
               for s_ in seeds]
    handles += [
        # the line-style entries refer to shape, not color, so they are black (distinct from the gray of seed 9)
        Line2D([], [], color="black", lw=1.2, ls="-", label="key recovered"),
        Line2D([], [], color="black", lw=1.2, ls=":", label="not recovered"),
    ]
    ncol = min(6, len(handles))        # 12 -> evenly as 6 columns x 2 rows
    nrow = int(np.ceil(len(handles) / ncol))
    fig = plt.figure(figsize=(width, 0.16 * nrow + 0.08))
    fig.legend(handles=handles, loc="center", ncol=ncol, frameon=False,
               fontsize=FONT, handlelength=1.6, columnspacing=1.1,
               labelspacing=0.3, handletextpad=0.5, borderpad=0.0)
    p_ = os.path.join(out_dir, "ge_legend.eps")
    fig.savefig(p_, format="eps", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(os.path.join(out_dir, "ge_legend.png"), format="png", dpi=300,
                bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  {'ge_legend.eps':<28} {len(seeds)} seeds + 2 line styles, {nrow} rows")
    return p_


def main():
    style()
    argv = sys.argv[1:]

    # If arguments are given, use them; otherwise the script finds the files
    fine = argv[0] if len(argv) > 0 else find_csv(FINE_REL, want_fine=True)
    out_dir = argv[1] if len(argv) > 1 else OUT_DIR
    coarse = argv[2] if len(argv) > 2 else find_csv(COARSE_REL, want_fine=False)

    if not fine:
        raise SystemExit(
            f"Could not find the fine evaluation file.\n"
            f"  searched: {os.path.join(DATA_ROOT, FINE_REL)} and the current folder, up to 3 levels below\n"
            f"  fix: edit DATA_ROOT at the top of this file,\n"
            f"        or pass TIFS_ROOT=/path python make_ge_eps.py,\n"
            f"        or give it directly as python make_ge_eps.py <fine eval_runs.csv>")

    print(f"curves {fine}")
    print(f"conv.  {coarse if coarse else '(not found — counting by the fine criterion itself)'}")
    print(f"output {os.path.abspath(out_dir)}\n")
    os.makedirs(out_dir, exist_ok=True)

    qs, D = load(fine)
    if len(qs) < 20 or not np.array_equal(qs[:5], np.arange(1, 6)):
        print(f"  ⚠️  The Q grid is {list(qs[:9])}... This looks like coarse —\n"
              f"      give fine (all of Q=1..50). With a log grid the curves look kinked.\n")

    # The convergence count k/n must be the same set as Table 2 in the main text. Table 2 is based on **coarse** (Q<=500) and
    # fine only goes up to Q<=50, so some runs disagree, such as ResNet N128 s0 (coarse 39/39, fine 34/39).
    if coarse:
        CS = conv_set(coarse)
        diff = []
        for (arch, n), runs in D.items():
            for i, (sd, y, c) in enumerate(runs):
                c2 = (arch, n, sd) in CS
                if c2 != c:
                    diff.append(f"{arch} N{n} s{sd}: fine {int(c)} -> coarse {int(c2)}")
                    runs[i] = (sd, y, c2)
        print("  Convergence verdict aligned to the coarse criterion"
              + (f" — {len(diff)} runs changed:\n      " + "\n      ".join(diff)
                 if diff else " (no differences)") + "\n")
    else:
        print("  ⚠️  Counting converged k/n by the fine criterion itself. Table 2 in the main text is coarse-based, so\n"
              "      ResNet N128 disagrees as 7/10 vs 8/10.\n")

    keep = qs <= QMAX
    qs = qs[keep]
    print(f"Q = {qs[0]}..{qs[-1]}  (using {int(keep.sum())}/{len(keep)} points of the original)")
    print(f"chance = log2(q^2/2) = {CHANCE:.2f}\n")

    n_made, seen = 0, set()
    for arch in ("resnet", "mobilenet"):
        for n in N_LIST:
            runs = D.get((arch, n))
            if not runs:
                print(f"  [missing] {NAME[arch]}_ge_N{n}")
                continue
            panel(qs, [(s_, y[keep], c) for s_, y, c in runs], arch, n, out_dir)
            seen.update(s_ for s_, _y, _c in runs)
            n_made += 1
    legend_strip(sorted(seen), out_dir)

    print(f"\nEPS {n_made} + 1 legend (+ PNGs for checking)")
    print("Panel size %.2f x %.2f in — fits 0.24\\textwidth when placed 4x4 in a figure*" % (W, H))
    print("To attach axis labels only to the bottom row / left column, turn off SHOW_XLABEL/SHOW_YLABEL and run twice")
    print("Attach the legend below the 4x4 grid with \\includegraphics[width=\\textwidth]{ge_legend}")


if __name__ == "__main__":
    main()
