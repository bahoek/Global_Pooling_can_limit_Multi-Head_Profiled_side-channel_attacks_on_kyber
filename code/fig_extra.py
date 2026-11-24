r"""
fig_gradnorm  +  fig_shift_probe  — the remaining two of the paper chat's "optional" list

(1) fig_gradnorm — figure supporting control experiment B (mean-loss dilution)
    (a) ResNet backbone gradient norm vs epoch, per N (mean over converged seeds)
    (b) phantom / signal head gradient norm ratio vs N
    Shows, as epoch trajectories, B's conclusion that as heads increase, the denominator of the
    mean-over-heads loss grows and the gradient reaching the backbone is diluted.

(2) fig_shift_probe — evidence that the mechanism of the shift probe is positional alignment
    r1 cannot be matched exactly with a single shift, so the misalignment (err_r1) differs per pair. Hence
    |err_r1| vs accuracy is itself an experiment. **Even a 10-sample misalignment lowers it.**
    As controls, 'own pair' (no shift) and 'misaligned shift' are placed alongside.

Data
----
  all_gradnorm.csv            source/arch/n_pairs/seed/epoch/gn_backbone/gn_heads_{signal,phantom}
  eval/shift/shift_heads.csv  kind(own/aligned/misaligned) x norm_mode x err_r1 x acc_r1

Run
----
    python fig_extra.py <all_gradnorm.csv> <shift_heads.csv> [<eval_runs.csv>]
If eval_runs.csv is given, only converged seeds are used (otherwise all seeds).
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
NORM = "before"          # the correct normalization for the shift probe (after is below the prior)


def converged(path):
    if not path or not os.path.exists(path):
        return None
    s = set()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["eval_mode"] == "sum" and int(r["pairs_recovered"]) == int(r["n_pairs_eval"]):
            s.add((r["arch"], int(r["n_pairs"]), int(r["seed"])))
    return s


def fig_gradnorm(path, conv):
    rows = defaultdict(lambda: defaultdict(list))        # (arch,N) -> epoch -> [gn_backbone]
    ratio = defaultdict(list)                            # (arch,N) -> [phantom/signal]
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["source"] != "main":
            continue
        key = (r["arch"], int(r["n_pairs"]), int(r["seed"]))
        if conv is not None and key not in conv:
            continue
        a, n = r["arch"], int(r["n_pairs"])
        try:
            bb = float(r["gn_backbone"])
            sg = float(r["gn_heads_signal"])
            ph = float(r["gn_heads_phantom"]) if r["gn_heads_phantom"] not in ("", None) else None
        except ValueError:
            continue
        rows[(a, n)][int(r["epoch"])].append(bb)
        if ph is not None and sg > 0 and n > LEAK:
            ratio[(a, n)].append(ph / sg)

    fig, (a1, a2) = FS.figure(FS.ONE_COL, 3.5, nrows=2,
                              gridspec_kw=dict(hspace=0.38))
    FS.grid(a1, "y")
    ns = [n for n in N_LIST if ("resnet", n) in rows]
    shades = np.linspace(0.72, 0.0, len(ns))
    print("[fig_gradnorm] ResNet backbone gradient norm")
    for n, g in zip(ns, shades):
        d = rows[("resnet", n)]
        eps = sorted(d)[:100]
        y = [np.mean(d[e]) for e in eps]
        a1.plot(eps, y, "-", color=str(g), linewidth=0.9, label=f"$N{{=}}{n}$")
        print(f"  N{n:<4} ep1 {y[0]:.4f}  ep10 {y[9] if len(y)>9 else float('nan'):.4f}  "
              f"ep50 {y[49] if len(y)>49 else float('nan'):.4f}")
    a1.set_yscale("log")
    a1.set_xlim(0, 100)
    a1.set_xlabel("epoch")
    a1.set_ylabel("backbone\ngradient norm")   # two lines: on one line the top gets cut off
    a1.text(0.975, 0.88, "(a) ResNet18", transform=a1.transAxes, fontsize=7.5,
            ha="right", va="center")
    a1.legend(loc="lower left", ncol=4, fontsize=5.6, handlelength=1.1,
              handletextpad=0.35, columnspacing=0.8, labelspacing=0.2, borderpad=0.3,
              frameon=True, framealpha=1.0, facecolor="white", edgecolor="none")

    FS.grid(a2, "y")
    print("[fig_gradnorm] phantom/signal head gradient norm ratio")
    for arch, lab, mk, fc in (("resnet", "ResNet18", FS.MARKERS[0], FS.INK),
                              ("mobilenet", "MobileNetV2", FS.MARKERS[1], "none")):
        xs = [n for n in N_LIST if ratio.get((arch, n))]
        if not xs:
            continue
        m = [np.mean(ratio[(arch, n)]) for n in xs]
        s = [np.std(ratio[(arch, n)], ddof=1) if len(ratio[(arch, n)]) > 1 else 0 for n in xs]
        a2.errorbar(xs, m, yerr=s, fmt=FS.LINES[0], marker=mk, color=FS.INK,
                    markerfacecolor=fc, markeredgewidth=0.7, markersize=3.4,
                    capsize=1.6, elinewidth=0.6, label=lab)
        for n, v in zip(xs, m):
            print(f"  {arch:<10} N{n:<4} {v:.3f}")
    a2.set_xscale("log")
    a2.set_xticks([n for n in N_LIST if n > LEAK])
    a2.set_xticklabels([str(n) for n in N_LIST if n > LEAK])
    a2.minorticks_off()
    a2.set_xlabel("targeted pairs $N$")
    a2.set_ylabel("phantom / signal")
    a2.text(0.975, 0.88, "(b)", transform=a2.transAxes, fontsize=7.5,
            ha="right", va="center")
    a2.legend(loc="lower right", handletextpad=0.4, borderpad=0.25, frameon=True,
              framealpha=1.0, facecolor="white", edgecolor="none")
    FS.save(fig, "fig_gradnorm")


def fig_shift(path):
    """r1 accuracy per |err_r1| bucket. Controls: own (no shift), misaligned (misaligned shift)."""
    buck = defaultdict(list)
    ctrl = defaultdict(list)
    prior = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["norm_mode"] != NORM:
            continue
        try:
            acc = float(r["acc_r1"])
            prior.append(float(r["majority_r1"]))
        except ValueError:
            continue
        if r["kind"] == "aligned":
            e = abs(float(r["err_r1"]))
            buck[min(int(e // 10) * 10, 40)].append(acc)
        else:
            ctrl[r["kind"]].append(acc)
    pr = float(np.mean(prior)) if prior else 11.9

    fig, ax = FS.figure(FS.ONE_COL, 2.3)
    FS.grid(ax, "y")
    ks = sorted(buck)
    m = [np.mean(buck[k]) for k in ks]
    se = [np.std(buck[k], ddof=1) / np.sqrt(len(buck[k])) for k in ks]
    x = np.arange(len(ks))
    print("[fig_shift_probe] r1 accuracy per |err_r1| bucket")
    for k, v in zip(ks, m):
        print(f"  {k}~{k+9} samples: {v:.2f}%  (n={len(buck[k]):,})")
    ax.bar(x, m, 0.62, yerr=se, facecolor=FS.INK3, edgecolor=FS.INK, linewidth=0.6,
           capsize=1.6, error_kw=dict(elinewidth=0.6), zorder=3)
    ax.axhline(pr, color=FS.INK2, linewidth=0.7, linestyle=":", zorder=4)
    ax.text(len(ks) - 0.4, pr + 0.35, f"majority {pr:.1f}%", fontsize=6,
            color=FS.INK2, ha="right", va="bottom")
    for i, (lab, key) in enumerate((("own pair\n(no shift)", "own"),
                                    ("misaligned\ncontrol", "misaligned"))):
        if key not in ctrl:
            continue
        v = float(np.mean(ctrl[key]))
        ax.bar([len(ks) + i], [v], 0.62, facecolor="white", edgecolor=FS.INK,
               linewidth=0.6, hatch="//", zorder=3)
        print(f"  {key}: {v:.2f}%")
    ax.set_xticks(list(x) + [len(ks), len(ks) + 1])
    ax.set_xticklabels([f"{k}–{k+9}" for k in ks] + ["own", "mis."])
    ax.set_xlabel("$|$alignment error$|$ (samples)")
    ax.set_ylabel("$r[1]$ accuracy (%)")
    FS.save(fig, "fig_shift_probe")


def main():
    if len(sys.argv) >= 3:
        a_gn, a_sh = sys.argv[1], sys.argv[2]
        a_ev = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        a_gn = FS.need("all_gradnorm.csv", "gradnorm")
        a_sh = FS.need("eval/shift/shift_heads.csv", "shift probe")
        a_ev = FS.need("eval/eval_runs.csv", "main evaluation", FS.IS_COARSE)
        print()
    conv = converged(a_ev)
    print(f"Converged-seed filter: {'used (' + str(len(conv)) + ')' if conv else 'none (all seeds)'}\n")
    fig_gradnorm(a_gn, conv)
    print()
    fig_shift(a_sh)


if __name__ == "__main__":
    main()
