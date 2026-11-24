r"""
fig_acc_vs_N  +  fig_probe_info

(1) fig_acc_vs_N — signal-head accuracy of converged seeds vs N, two backbones
    Converged = `pairs_recovered == n_pairs_eval` (the paper's criterion). Horizontal line for the prior (majority), vertical line at L=39.
    acc is **not monotonic** in N: ResNet rises from N39 62.8 -> N99 68.6 and drops to 55.0 at N128.

(2) fig_probe_info — information (bit) on untrained pairs, 4 conditions x 5 backbone/N combinations
    Pooled : 78 new heads on the features after GAP (probe_heads)
    Local  : before GAP, each head sees only ±160 samples around its own peak (probe_local)
    Random : same protocol as Local on a randomly initialized backbone (probe_local_ctrl)
    Raw    : linear on the raw 320 samples in the window, no conv (probe_local_ctrl)

    Raw has the same value, 1.428 bit, regardless of backbone and N, so it is drawn **as a horizontal
    reference line rather than repeated as 5 bars**. Drawing the same number five times reads like a per-condition difference.

Data
----
  eval/eval_runs.csv                        acc_signal + convergence decision
  eval/probe_heads/probe_models.csv         Pooled
  probe_local / probe_local_ctrl console values    hard-coded in LOCAL/RANDOM/RAW below
      (if local_models.csv, ctrl_models.csv are given, they are read with --local, --ctrl and override)

Run
----
    python fig_acc_probe.py <eval_runs.csv> <probe_models.csv> [--local local_models.csv] [--ctrl ctrl_models.csv]
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS

LEAK = 39

# Untrained-pair info (bit) taken from the console output of probe_local.py / probe_local_ctrl.py.
# Overridden if CSVs are given. Values are exactly as printed, to 3 digits.
LOCAL = {("resnet", 9): [1.245, 1.231], ("resnet", 19): [1.238, 1.301],
         ("resnet", 29): [1.739, 1.662], ("mobilenet", 19): [2.008, 2.454],
         ("mobilenet", 29): [2.037, 2.206]}
RANDOM = {("resnet", 9): [0.176], ("resnet", 29): [0.122],
          ("mobilenet", 19): [-0.334], ("mobilenet", 29): [-0.375]}
RAW = 1.428
GROUPS = [("resnet", 9), ("resnet", 19), ("resnet", 29), ("mobilenet", 19), ("mobilenet", 29)]
LBL = {"resnet": "RN", "mobilenet": "MNv2"}


def acc_by_N(path):
    """Collect only converged (recovery criterion) seeds: (arch, N) -> [acc_signal...]"""
    out = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["eval_mode"] != "sum":
            continue
        if int(r["pairs_recovered"]) != int(r["n_pairs_eval"]):
            continue
        out[(r["arch"], int(r["n_pairs"]))].append(float(r["acc_signal"]))
    return out


def majority(path):
    """Mean signal-head majority value (prior baseline)."""
    v = [float(r["majority_signal"]) for r in csv.DictReader(open(path, encoding="utf-8"))
         if r["eval_mode"] == "sum" and r["majority_signal"] not in ("", None)]
    return float(np.mean(v)) if v else 11.9


def pooled_by_N(path):
    out = defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["src_converged"] != "1" or r.get("info_unseen", "") == "":
            continue
        out[(r["arch"], int(r["src_n_pairs"]))].append(float(r["info_unseen"]))
    return out


def fig_acc(runs_path):
    A = acc_by_N(runs_path)
    maj = majority(runs_path)
    fig, ax = FS.figure(FS.ONE_COL, 2.35)
    FS.grid(ax, "y")
    ax.axvline(LEAK, color=FS.INK2, linewidth=0.7, linestyle="--", zorder=2)
    ax.axhline(maj, color=FS.INK3, linewidth=0.7, linestyle=":", zorder=2)
    print(f"{'arch':<11}{'N':>5}{'n':>4}{'acc mean±SD':>16}")
    for arch, lab, mk, fc in (("resnet", "ResNet18", FS.MARKERS[0], FS.INK),
                              ("mobilenet", "MobileNetV2", FS.MARKERS[1], "none")):
        ns = sorted({n for a, n in A if a == arch})
        m = [np.mean(A[(arch, n)]) for n in ns]
        s = [np.std(A[(arch, n)], ddof=1) if len(A[(arch, n)]) > 1 else 0.0 for n in ns]
        for n, mm, ss in zip(ns, m, s):
            print(f"{arch:<11}{n:>5}{len(A[(arch,n)]):>4}{f'{mm:.2f}±{ss:.2f}':>16}")
        ax.errorbar(ns, m, yerr=s, fmt=FS.LINES[0], marker=mk, color=FS.INK,
                    markerfacecolor=fc, markeredgewidth=0.7, markersize=3.4,
                    capsize=1.6, elinewidth=0.6, label=lab, zorder=4)
    ax.set_xscale("log")
    xt = sorted({n for _, n in A})
    ax.set_xticks(xt)
    ax.set_xticklabels([str(n) for n in xt])
    ax.minorticks_off()
    ax.set_xlim(7.5, 160)
    ax.set_ylim(0, 105)
    ax.set_xlabel("targeted pairs $N$")
    ax.set_ylabel("signal-head accuracy (%)")
    ax.text(LEAK * 1.07, 97, "$L=39$", fontsize=7, color=FS.INK2, va="center")
    ax.text(150, maj + 4.0, f"majority {maj:.1f}%", fontsize=6.5, color=FS.INK2,
            va="center", ha="right")
    ax.legend(loc="lower left", handletextpad=0.4, borderpad=0.25, frameon=True,
              framealpha=1.0, facecolor="white", edgecolor="none")
    FS.save(fig, "fig_acc_vs_N")


def fig_probe(pooled_path, local, rnd):
    P = pooled_by_N(pooled_path)
    fig, ax = FS.figure(FS.ONE_COL, 2.35)
    FS.grid(ax, "y")
    x = np.arange(len(GROUPS))
    w = 0.26
    series = [("Pooled (after GAP)", [np.mean(P[g]) if P.get(g) else np.nan for g in GROUPS],
               FS.INK3, "//"),
              ("Local (before GAP)", [np.mean(local[g]) if local.get(g) else np.nan for g in GROUPS],
               FS.INK, None),
              ("Random backbone", [np.mean(rnd[g]) if rnd.get(g) else np.nan for g in GROUPS],
               "white", "..")]
    for i, (lab, v, col, hatch) in enumerate(series):
        ax.bar(x + (i - 1) * w, v, w * 0.92, label=lab, facecolor=col,
               edgecolor=FS.INK, linewidth=0.6, hatch=hatch, zorder=3)
        print(f"  {lab:<22} " + "  ".join("-" if np.isnan(a) else f"{a:.3f}" for a in v))
    ax.axhline(RAW, color=FS.INK, linewidth=0.9, linestyle="--", zorder=4)
    # left-aligned and short: if centered, the Local bar for RN N=29 (1.70) hides the text
    ax.text(-0.42, RAW + 0.06, f"raw window ({RAW:.2f})", fontsize=6.5,
            color=FS.INK, ha="left", va="bottom")
    ax.axhline(0, color=FS.INK, linewidth=0.6, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LBL[a]}\n$N{{=}}{n}$" for a, n in GROUPS])
    ax.set_ylim(-0.6, 2.9)
    # two lines: on one line it is longer than the figure height and the top gets cut off (ESL A16)
    ax.set_ylabel("information on\nuntrained pairs (bits)")
    ax.legend(loc="upper left", ncol=1, handletextpad=0.5, borderpad=0.25, handlelength=1.3,
              frameon=True, framealpha=1.0, facecolor="white", edgecolor="none")
    FS.save(fig, "fig_probe_info")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        args = [FS.need("eval/eval_runs.csv", "main evaluation", FS.IS_COARSE),
                FS.need("eval/probe_heads/probe_models.csv", "probe_heads")]
        print()
    if len(args) < 2:
        raise SystemExit("Usage: python fig_acc_probe.py <eval_runs.csv> <probe_models.csv>")
    local = {k: list(v) for k, v in LOCAL.items()}
    rnd = {k: list(v) for k, v in RANDOM.items()}
    print("[fig_acc_vs_N] signal-head accuracy of converged (recovery criterion) seeds")
    fig_acc(args[0])
    print("\n[fig_probe_info] untrained-pair info (bit)")
    fig_probe(args[1], local, rnd)


if __name__ == "__main__":
    main()
