r"""
fig_ge_vs_Q — guessing entropy curves: $\log_2$ GE vs Q, per N, two backbones

This is the basic result figure of an SCA paper. It is the only figure showing that the attack actually succeeds,
and the first place a reviewer looks, so it is not a figure to drop for space.

(a) ResNet18  (b) MobileNetV2. Each panel overlays the N curves.
Only converged (recovery criterion: `pairs_recovered == n_pairs_eval`) seeds are averaged, using the pair-mean $\log_2$GE.
Horizontal lines are drawn at chance level $\log_2(q^2/2) = 22.40$ and at GE=1 (=0 bit).

N is an **ordered variable**, so it is laid out not by color but by a **brightness ladder** (light gray N=9 → black N=128),
and the N value is written directly at the end of each curve. Easier to read than an 8-line legend, and the order is visible in black-and-white printing too.

MobileNet has no curve for N=128 because 0/10 converged. N=34 is a control experiment evaluated only on the coarse grid,
so it has no fine curve (it is not included here).

Data
----
  eval_qfine/eval_runs.csv   ge_log2_q1 .. ge_log2_q50 (run x mode)

Run
----
    python fig_ge.py <eval_qfine/eval_runs.csv>
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS

N_LIST = [9, 19, 29, 39, 49, 69, 99, 128]
QMAX = 20          # every curve hits 0 by Q<=13. Plotting up to 50 is mostly dead space
CHANCE = float(np.log2(3329 * 3329 / 2))          # 22.40


def conv_set(path, mode="sum"):
    """Read only the convergence decision -> {(arch, N, seed)} (recovery criterion)."""
    out = set()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["eval_mode"] == mode and int(r["pairs_recovered"]) == int(r["n_pairs_eval"]):
            out.add((r["arch"], int(r["n_pairs"]), int(r["seed"])))
    return out


def load(path, mode="sum", conv=None):
    """(arch, N) -> (Q array, converged-seed mean log2GE array, number of seeds)

    If `conv` is given, convergence is decided by that set. Main-text Table 2 uses the **coarse** (Q<=500) criterion,
    and fine goes only up to Q<=50, so ResNet N128 s0 splits (coarse 39/39 Q*=100, fine 34/39).
    If not given, fine's own criterion is used, so only that figure gets n=7 and disagrees with the other figures and tables.
    """
    qs, out = None, defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["eval_mode"] != mode:
            continue
        ok = ((r["arch"], int(r["n_pairs"]), int(r["seed"])) in conv if conv is not None
              else int(r["pairs_recovered"]) == int(r["n_pairs_eval"]))
        if not ok:
            continue                                # converged seeds only
        if qs is None:
            qs = sorted(int(k[len("ge_log2_q"):]) for k in r if k.startswith("ge_log2_q"))
        out[(r["arch"], int(r["n_pairs"]))].append(
            [float(r[f"ge_log2_q{q}"]) for q in qs])
    return np.array(qs), {k: (np.mean(v, 0), len(v)) for k, v in out.items()}


def main():
    if len(sys.argv) > 1:
        path, coarse = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        path = FS.need("eval_qfine/eval_runs.csv", "fine evaluation", FS.IS_FINE)
        coarse = FS.find("eval/eval_runs.csv", FS.IS_COARSE)
        if coarse:
            print(f"  {'coarse eval (convergence)':<24} {coarse}")
        print()

    # Align the converged set to coarse. Same rule as make_ge_eps.py, so n does not differ between the two figures
    conv = conv_set(coarse) if coarse else None
    if conv is None:
        print("  ⚠️  coarse evaluation not found, counting by fine's own criterion — ResNet N128 comes out\n"
              "      as n=7, disagreeing with main-text Table 2 (8/10) and the 16 ge_eps panels.\n")
    qs, D = load(path, conv=conv)
    print(f"Q = {qs[0]}..{qs[-1]} ({len(qs)} points)")

    FS.setup()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(FS.ONE_COL, 3.9), sharex=True,
                             gridspec_kw=dict(hspace=0.13))

    for ax, (arch, lab, tag) in zip(axes, [("resnet", "ResNet18", "(a)"),
                                           ("mobilenet", "MobileNetV2", "(b)")]):
        FS.grid(ax, "y")
        ax.axhline(CHANCE, color=FS.INK3, linewidth=0.7, linestyle=":", zorder=2)
        ax.axhline(0.0, color=FS.INK2, linewidth=0.7, linestyle="--", zorder=2)
        ns = [n for n in N_LIST if (arch, n) in D]
        # N is an ordered variable -> brightness ladder (light gray = small N, black = large N)
        shades = np.linspace(0.72, 0.0, len(ns))
        for n, g in zip(ns, shades):
            y, k = D[(arch, n)]
            col = str(g)
            # No direct labels — the curves pass the same height at similar Q and all overlap.
            # N is an ordered variable and brightness already carries the order, so the legend is only in panel (a)
            ax.plot(qs, y, "-", color=col, linewidth=1.0, zorder=3 + ns.index(n),
                    label=f"$N{{=}}{n}$")
            print(f"  {arch:<10} N{n:<4} n={k:<2} GE@1={y[0]:5.2f} @5={y[4]:5.2f} "
                  f"@10={y[9]:5.2f} @20={y[19]:5.2f} @50={y[-1]:5.2f}")
        ax.set_ylim(-1.4, 24.2)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_xlim(0.6, QMAX + 0.4)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_ylabel("$\\log_2$ GE")
        ax.text(0.975, 0.52, f"{tag} {lab}", transform=ax.transAxes, fontsize=7.5,
                ha="right", va="center")

    for ax in axes:
        ax.text(0.6 + 0.3, CHANCE - 1.8, f"chance {CHANCE:.2f}", fontsize=6,
                color=FS.INK2, ha="left", va="center")
        ax.text(QMAX * 0.985, 1.9, "$\\mathrm{GE}=1$", fontsize=6,
                color=FS.INK2, ha="right", va="center")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, loc="upper right", bbox_to_anchor=(1.0, 1.04), ncol=4,
                   fontsize=5.6, handlelength=1.1, handletextpad=0.35,
                   columnspacing=0.8, labelspacing=0.2, borderpad=0.3,
                   frameon=True, framealpha=1.0, facecolor="white", edgecolor="none")
    axes[1].set_xlabel("traces $Q$")
    FS.save(fig, "fig_ge_vs_Q")


if __name__ == "__main__":
    main()
