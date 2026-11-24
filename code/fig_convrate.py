r"""
fig_convrate_vs_N — convergence rate vs N, two backbones

Definition of convergence
---------
**Unified to the single paper criterion**: `pairs_recovered == n_pairs_eval`
(= all min(N,39) pairs recovered at Q <= 500). Not an accuracy-threshold criterion.
There are 4 runs where the two differ (MobileNet N49·N69·N128 one each, N34 s3) — in all of them accuracy
crossed the threshold but not all pairs were recovered.

N=34 (control experiment S34) and seeds 10~14 of ResNet N=128 (R5) also went through the same evaluation, so
every point rests on the same definition. Only ResNet N=128 has 15 seeds, so the y axis is a **rate**,
not a count (with counts, points with different n would be placed on the same scale). k/n is written at each point.

Error bars are Wilson 95% intervals. The wide overlap is itself a result — both ends of the inverted U
are clear, but the peak cannot be pinned down in the middle range (N 29~39) (Fisher: N34 vs N29 p=0.14,
N34 vs N39 p=0.35, N29 vs N39 p=1.00).

Data
----
  eval/eval_runs.csv        160 main training runs
  eval_ctrl/eval_runs.csv   S34 (N=34) + R5 (N=128 seeds 10~14)

Run
----
    python fig_convrate.py <eval/eval_runs.csv> <eval_ctrl/eval_runs.csv>
"""

import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS

LEAK = 39
ARCHS = [("resnet", "ResNet18", FS.MARKERS[0], FS.INK),
         ("mobilenet", "MobileNetV2", FS.MARKERS[1], FS.INK)]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def tally(*paths):
    """(arch, N) -> [converged seed count, total seed count]. If the same (arch,N,seed) is in several files, counted once."""
    seen, out = set(), defaultdict(lambda: [0, 0])
    for p in paths:
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["eval_mode"] != "sum":
                continue
            key = (r["arch"], int(r["n_pairs"]), int(r["seed"]))
            if key in seen:
                continue
            seen.add(key)
            d = out[(r["arch"], int(r["n_pairs"]))]
            d[1] += 1
            d[0] += int(int(r["pairs_recovered"]) == int(r["n_pairs_eval"]))
    return out


def main():
    if len(sys.argv) >= 3:
        args = sys.argv[1:]
    else:
        args = [FS.need("eval/eval_runs.csv", "main evaluation", FS.IS_COARSE),
                FS.need("eval_ctrl/eval_runs.csv", "control evaluation (S34·R5)")]
        print()
    T = tally(*args)
    print(f"{'arch':<11}{'N':>5}{'conv':>9}{'rate':>8}{'Wilson 95%':>18}")
    for arch, lab, mk, col in ARCHS:
        for n in sorted({nn for a, nn in T if a == arch}):
            k, m = T[(arch, n)]
            lo, hi = wilson(k, m)
            print(f"{arch:<11}{n:>5}{str(k)+'/'+str(m):>9}{k/m:>8.2f}{f'[{lo:.2f}, {hi:.2f}]':>18}")

    fig, ax = FS.figure(FS.ONE_COL, 2.45)
    FS.grid(ax, "y")
    ax.axvline(LEAK, color=FS.INK2, linewidth=0.7, linestyle="--", zorder=2)

    for arch, lab, mk, col in ARCHS:
        ns = sorted({nn for a, nn in T if a == arch})
        ks = [T[(arch, n)][0] for n in ns]
        ms_ = [T[(arch, n)][1] for n in ns]
        ys = [k / m for k, m in zip(ks, ms_)]
        ci = [wilson(k, m) for k, m in zip(ks, ms_)]
        err = np.array([[y - lo for y, (lo, _) in zip(ys, ci)],
                        [hi - y for y, (_, hi) in zip(ys, ci)]])
        fc = col if arch == "resnet" else "none"
        ax.errorbar(ns, ys, yerr=err, fmt=FS.LINES[0], marker=mk, color=col,
                    markerfacecolor=fc, markeredgewidth=0.7, markersize=3.6,
                    capsize=1.6, elinewidth=0.5, label=lab, zorder=4)
        for n, k, m in zip(ns, ks, ms_):
            if m != 10:                                 # label only points with a different seed count (ResNet N128)
                # placed to the right of the point so it does not overlap the error-bar end
                ax.text(n * 1.07, k / m, f"{k}/{m}", fontsize=6, color=col,
                        ha="left", va="center")

    # N=34 is MobileNet only (control experiment S34). Ticks 34 and 39 overlap on the log axis,
    # so it is pointed to inside the plot instead of with a tick
    if ("mobilenet", 34) in T:
        k, m = T[("mobilenet", 34)]
        ax.annotate("$N=34$", xy=(34, k / m), xytext=(26.5, k / m - 0.165), fontsize=6.5,
                    color=FS.INK2, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=FS.INK3, linewidth=0.5))

    ax.set_xscale("log")
    xticks = [n for n in sorted({n for _, n in T}) if n != 34]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(n) for n in xticks])
    ax.minorticks_off()
    ax.set_xlim(7.5, 160)
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("targeted pairs $N$")
    ax.set_ylabel("convergence rate")
    ax.text(LEAK * 1.07, 0.03, "$L=39$", fontsize=7, color=FS.INK2, va="center")
    ax.legend(loc="lower left", handletextpad=0.4, borderpad=0.25, frameon=True,
              framealpha=1.0, facecolor="white", edgecolor="none")
    FS.save(fig, "fig_convrate_vs_N")


if __name__ == "__main__":
    main()
