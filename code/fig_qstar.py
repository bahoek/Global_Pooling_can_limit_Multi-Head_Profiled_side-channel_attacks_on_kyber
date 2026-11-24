"""
fig_qstar_vs_N  +  fig_qgrid_coarse_vs_fine

(1) fig_qstar_vs_N — capacity metric Q* vs N, two backbones
    Q*max (number of traces at which all pairs are recovered, solid) and Q*mean (pair mean, dashed).
    Mean±SD over converged seeds only (= all min(N,39) pairs recovered). Vertical line at L=39.
    ResNet N128 s0 has Q*>50 and is censored on the fine grid — shown separately with an upward arrow
    (Q*max=100 on the coarse grid).

(2) fig_qgrid_coarse_vs_fine — the same models re-evaluated with only the Q grid changed
    coarse: Q = 1,2,5,10,20,50,100,200,500 (log spacing, 10 repeats)
    fine  : Q = 1..50 every trace (20 repeats)
    A log-spaced grid **rounds Q* up to a grid point**, so it overestimates. At N128 by +11.9 traces;
    the N39->N128 increase is +10.0 (coarse) vs +1.0 (fine) — **a 10x overestimate**.

Convergence verdict
-------------------
`pairs_recovered == n_pairs_eval` (= converged_rec). It is computed this way even from CSVs produced by
versions of eval_nscan.py that have no dedicated column.

Data
----
  eval_qfine/eval_runs.csv   fine
  eval/eval_runs.csv         coarse

Run
---
    python fig_qstar.py <fine eval_runs.csv> <coarse eval_runs.csv>
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS

N_LIST = [9, 19, 29, 39, 49, 69, 99, 128]
LEAK = 39
ARCHS = [("resnet", "ResNet18", FS.MARKERS[0], FS.INK, FS.INK),
         ("mobilenet", "MobileNetV2", FS.MARKERS[1], FS.INK, "none")]


def per_seed(path):
    """(arch, N, seed) -> dict(conv, qmax, qmean). conv = all pairs recovered (= converged_rec)."""
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["eval_mode"] != "sum":
            continue
        g = lambda k: float(r[k]) if r[k] not in ("", "-1") else None
        out[(r["arch"], int(r["n_pairs"]), int(r["seed"]))] = dict(
            conv=int(r["pairs_recovered"]) == int(r["n_pairs_eval"]),
            qmax=g("q_star_max"), qmean=g("q_star_mean"))
    return out


def load(fine_s, coarse_s):
    """Collect only seeds converged by the paper criterion (coarse, Q<=500), and use the fine values for Q*.
    Seeds that converged on coarse but lie outside the fine grid (Q<=50) are counted separately as `beyond` —
    simply dropping them would underestimate Q* for that N (ResNet N128 s0: coarse Q*max=100)."""
    out = defaultdict(lambda: dict(qmax=[], qmean=[], n_conv=0, n_seed=0,
                                   beyond=0, beyond_coarse=[]))
    for (arch, n, seed), c in coarse_s.items():
        d = out[(arch, n)]
        d["n_seed"] += 1
        if not c["conv"]:
            continue
        d["n_conv"] += 1
        f_ = fine_s.get((arch, n, seed))
        if f_ and f_["conv"] and f_["qmax"] is not None:
            d["qmax"].append(f_["qmax"])
            if f_["qmean"] is not None:
                d["qmean"].append(f_["qmean"])
        else:                                          # outside the fine grid
            d["beyond"] += 1
            if c["qmax"] is not None:
                d["beyond_coarse"].append(c["qmax"])
    return out


def ms(v):
    return (np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0.0)) if v else (np.nan, 0.0)


def fig_qstar(fine, coarse):
    fig, ax = FS.figure(FS.ONE_COL, 2.45)
    FS.grid(ax, "y")
    ax.axvline(LEAK, color=FS.INK2, linewidth=0.7, linestyle="--", zorder=2)

    for arch, lab, mk, col, fc in ARCHS:                 # Q*max first (legend order)
        xs = [n for n in N_LIST if fine[(arch, n)]["qmax"]]
        mx = [ms(fine[(arch, n)]["qmax"]) for n in xs]
        ax.errorbar(xs, [m for m, _ in mx], yerr=[s for _, s in mx], fmt=FS.LINES[0],
                    marker=mk, color=col, markerfacecolor=fc, markeredgewidth=0.7,
                    markersize=3.4, capsize=1.6, elinewidth=0.6,
                    label=f"{lab}, $Q^\\star_{{\\max}}$", zorder=4)
    for arch, lab, mk, col, fc in ARCHS:
        xs = [n for n in N_LIST if fine[(arch, n)]["qmax"]]
        mn = [ms(fine[(arch, n)]["qmean"]) for n in xs]
        ax.plot(xs, [m for m, _ in mn], FS.LINES[1], marker=mk, color=FS.INK2,
                markerfacecolor=fc, markeredgecolor=FS.INK2, markeredgewidth=0.6,
                markersize=2.8, label=f"{lab}, $Q^\\star_{{\\mathrm{{mean}}}}$", zorder=3)

    # seeds that converged but whose Q* lies outside the fine grid (50) — ResNet N128 s0 (Q*max=100 on the coarse grid).
    # Looking only at the mean without these underestimates N128
    for arch, lab, mk, col, fc in ARCHS:
        for n in N_LIST:
            k = fine[(arch, n)]
            if not k["beyond"]:
                continue
            ax.plot([n], [22.4], marker=mk, color=col, markerfacecolor=fc,
                    markeredgewidth=0.7, markersize=3.4, clip_on=False, zorder=5)
            ax.annotate("", xy=(n, 24.2), xytext=(n, 22.9),
                        arrowprops=dict(arrowstyle="->", color=col, linewidth=0.7))
            ax.text(n * 0.93, 22.4, f"{k['beyond']} seed $>50$", fontsize=6,
                    color=col, ha="right", va="center")

    ax.set_xscale("log")
    ax.set_xticks(N_LIST)
    ax.set_xticklabels([str(n) for n in N_LIST])
    ax.minorticks_off()
    ax.set_xlim(7.5, 160)
    ax.set_ylim(0, 25)
    ax.set_xlabel("targeted pairs $N$")
    ax.set_ylabel("traces to $\\mathrm{GE}=1$")
    ax.text(LEAK * 1.06, 1.0, "$L=39$", fontsize=7, color=FS.INK2, va="center")
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: (0 if "max" in l[i] else 1, l[i]))
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", ncol=1,
              handletextpad=0.4, borderpad=0.25, frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="none")
    FS.save(fig, "fig_qstar_vs_N")


def fig_qgrid(fine, coarse):
    fig, ax = FS.figure(FS.ONE_COL, 2.2)
    FS.grid(ax, "y")
    ax.axvline(LEAK, color=FS.INK2, linewidth=0.7, linestyle="--", zorder=2)
    arch = "resnet"
    xs = [n for n in N_LIST if fine[(arch, n)]["qmax"] and coarse[(arch, n)]["qmax"]]
    cm = [ms(coarse[(arch, n)]["qmax"]) for n in xs]
    fm = [ms(fine[(arch, n)]["qmax"]) for n in xs]
    # the difference between the two grids as a band — the point is that it leans one way (up) at every N
    ax.fill_between(xs, [m for m, _ in fm], [m for m, _ in cm], facecolor=FS.BAND,
                    edgecolor="none", zorder=1)
    ax.errorbar(xs, [m for m, _ in cm], yerr=[s for _, s in cm], fmt=FS.LINES[1],
                marker=FS.MARKERS[2], color=FS.INK2, markerfacecolor="none",
                markeredgecolor=FS.INK2, markeredgewidth=0.7, markersize=3.4,
                capsize=1.6, elinewidth=0.6, label="log-spaced grid", zorder=3)
    ax.errorbar(xs, [m for m, _ in fm], yerr=[s for _, s in fm], fmt=FS.LINES[0],
                marker=FS.MARKERS[0], color=FS.INK, markerfacecolor=FS.INK,
                markeredgewidth=0, markersize=3.0, capsize=1.6, elinewidth=0.6,
                label="every trace", zorder=4)
    # The two curves use **the same seed set** (see same_seeds in main). In the old version only the coarse side
    # included ResNet N128 s0 (Q*=100), giving "+12" and "SD 28 vs 1", but that was not a grid effect;
    # it was one seed entering only one side's mean. With the same 7 seeds, coarse is 20.0, SD 0.
    ax.set_xscale("log")
    ax.set_xticks(N_LIST)
    ax.set_xticklabels([str(n) for n in N_LIST])
    ax.minorticks_off()
    ax.set_xlim(7.5, 160)
    ax.set_ylim(0, 26)
    ax.set_xlabel("targeted pairs $N$")
    ax.set_ylabel("$Q^\\star_{\\max}$ (traces)")
    ax.text(LEAK * 1.06, 1.2, "$L=39$", fontsize=7, color=FS.INK2, va="center")
    ax.legend(loc="upper left", handletextpad=0.4, borderpad=0.25,
              frameon=True, framealpha=1.0, facecolor="white", edgecolor="none")
    FS.save(fig, "fig_qgrid_coarse_vs_fine")


def main():
    if len(sys.argv) >= 3:
        a1, a2 = sys.argv[1], sys.argv[2]
    else:
        a1 = FS.need("eval_qfine/eval_runs.csv", "fine evaluation", FS.IS_FINE)
        a2 = FS.need("eval/eval_runs.csv", "coarse evaluation", FS.IS_COARSE)
        print()
    fs, cs = per_seed(a1), per_seed(a2)
    fine = load(fs, cs)
    coarse = load(cs, cs)
    # for the grid comparison: coarse restricted to seeds that have Q*max on fine. Compares the two grids on the same seeds
    same_seeds = {k: v for k, v in cs.items()
                  if fs.get(k) and fs[k]["conv"] and fs[k]["qmax"] is not None}
    coarse_same = load(same_seeds, same_seeds)
    print(f"{'arch':<10}{'N':>5}{'conv':>8}{'fine Q*max':>14}{'coarse Q*max':>15}{'diff':>7}")
    for arch, lab, *_ in ARCHS:
        for n in N_LIST:
            f_, c = fine[(arch, n)], coarse[(arch, n)]
            a, b = ms(f_["qmax"])[0], ms(c["qmax"])[0]
            if np.isnan(a) and np.isnan(b):
                continue
            print(f"{arch:<10}{n:>5}{str(f_['n_conv'])+'/'+str(f_['n_seed']):>8}"
                  f"{a:>14.1f}{b:>15.1f}{b-a:>+7.1f}"
                  + (f"   (outside fine grid: {f_['beyond']} seed(s), coarse Q*="
                     f"{','.join(f'{v:.0f}' for v in f_['beyond_coarse'])})"
                     if f_["beyond"] else ""))
    fig_qstar(fine, coarse)
    fig_qgrid(fine, coarse_same)


if __name__ == "__main__":
    main()
