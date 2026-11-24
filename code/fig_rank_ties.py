"""
fig_rank_ties — effect of the number of tied candidates and the rank definition on SR (methodology section)

What it shows
-------------
(a) How the number of candidates whose **score is bit-for-bit equal** to the correct candidate shrinks with Q.
    At Q=1, r[0] alone gives 1 million (9.1% of the candidate space of 11,082,241), r[0]+r[1] summed gives 92 thousand (0.83%).
    About one order of magnitude disappears per trace, reaching 0 at Q=5 (summed) / Q=9 (alone).

(b) Hence the rank definitions diverge **only at small Q**.
    The optimistic rank (all ties placed below the correct key) reports SR 0.44 (summed) / 0.63 (alone) at Q=1.
    With the tie-corrected rank it is **0.00**. For Q>=10 the two definitions agree exactly.

Therefore the main results of this paper (Q*, SR@5 summed) are unaffected by the rank definition — Q*max is
identical in 40/40 runs and SR@5 is identical in 20/20 runs. The effect lies in small Q and single-head attacks.

Data
----
  eval_qfine/rank_audit/rank_audit_curves.csv   (output of rank_audit.py)
      mode x N x seed x pair x Q -> n_tied_mean, sr_tie, sr_opt, ge_tie, ge_opt

Run
----
    python fig_rank_ties.py [path to rank_audit_curves.csv]
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as FS

N_SHOW = 39          # N at which all seeds converge. At N128 stalled seeds create a tail and must not be mixed in
QMAX_SHOW = 20
QSPACE = 3329 * 3329


def default_path():
    return (FS.find("eval_qfine/rank_audit/rank_audit_curves.csv")
            or "rank_audit_curves.csv")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else default_path()
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist. Pass the path as an argument.")

    tied, srt, sro = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if int(r["n_pairs"]) != N_SHOW:
            continue
        k = (r["mode"], int(r["q"]))
        tied[k].append(float(r["n_tied_mean"]))
        srt[k].append(float(r["sr_tie"]))
        sro[k].append(float(r["sr_opt"]))
    qs = sorted({q for _, q in tied} & set(range(1, QMAX_SHOW + 1)))
    m = lambda d, mode: np.array([np.mean(d[(mode, q)]) for q in qs])
    print(f"N={N_SHOW}, Q=1..{QMAX_SHOW}, mean over pairs and seeds")
    for mode in ("sum", "r0"):
        t = m(tied, mode)
        print(f"  {mode:<4} tied Q=1 {t[0]:,.0f} ({t[0]/QSPACE*100:.2f}%) -> Q=5 {t[4]:.1f} -> Q=10 {t[9]:.2f}")
        print(f"       SR@1 tie {m(srt,mode)[0]:.4f} vs opt {m(sro,mode)[0]:.4f}")

    FS.setup()
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(2, 1, figsize=(FS.ONE_COL, 3.5), sharex=True,
                               gridspec_kw=dict(hspace=0.12, height_ratios=[1, 1]))

    # ---- (a) number of tied candidates ----
    FS.grid(a, "y")
    a.plot(qs, np.maximum(m(tied, "r0"), 1e-2), FS.LINES[0], marker=FS.MARKERS[1],
           color=FS.INK, markerfacecolor="none", markeredgewidth=0.7, markersize=3.2,
           label=r"$r[0]$ only")
    a.plot(qs, np.maximum(m(tied, "sum"), 1e-2), FS.LINES[0], marker=FS.MARKERS[0],
           color=FS.INK, markerfacecolor=FS.INK, markeredgewidth=0, markersize=2.8,
           label=r"$r[0]+r[1]$")
    a.axhline(1.0, color=FS.INK3, linewidth=0.5, linestyle=":", zorder=1)
    a.set_yscale("log")
    a.set_ylim(5e-2, 5e6)
    a.set_yticks([1e0, 1e2, 1e4, 1e6])
    a.set_ylabel("tied candidates")
    a.legend(loc="upper right", handletextpad=0.4)
    a.text(0.03, 0.08, "(a)", transform=a.transAxes, fontsize=8)

    # ---- (b) SR, two rank definitions ----
    FS.grid(b, "y")
    for mode, mk, fc, lbl in (("sum", FS.MARKERS[0], FS.INK, r"$r[0]+r[1]$"),
                              ("r0", FS.MARKERS[1], "none", r"$r[0]$ only")):
        st_, so_ = m(srt, mode), m(sro, mode)
        # band over the range where the two definitions diverge — this is the 'manufactured success rate'
        b.fill_between(qs, st_, so_, where=(so_ > st_), facecolor=FS.BAND,
                       edgecolor="none", zorder=1)
        b.plot(qs, so_, FS.LINES[1], marker=mk, color=FS.INK2, markerfacecolor=fc,
               markeredgecolor=FS.INK2, markeredgewidth=0.7, markersize=3.0,
               label=lbl + ", optimistic", zorder=3)
        b.plot(qs, st_, FS.LINES[0], marker=mk, color=FS.INK, markerfacecolor=fc,
               markeredgewidth=0.7, markersize=3.0, label=lbl + ", tie-aware", zorder=4)
    b.set_ylim(-0.03, 1.05)
    b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_xlim(0.5, QMAX_SHOW + 0.5)
    b.set_xticks([1, 5, 10, 15, 20])
    b.set_xlabel("traces $Q$")
    b.set_ylabel("success rate")
    lg = b.legend(loc="lower right", ncol=1, handletextpad=0.4, frameon=True,
                  framealpha=1.0, facecolor="white", edgecolor="none", borderpad=0.25)
    lg.set_zorder(6)
    b.text(0.03, 0.88, "(b)", transform=b.transAxes, fontsize=8)
    # the optimistic curve's SR **drops** as Q grows — behavior impossible for a correct metric
    b.annotate("", xy=(1.0, 0.44), xytext=(1.0, 0.63),
               arrowprops=dict(arrowstyle="-", color=FS.INK2, linewidth=0.6))
    b.text(1.6, 0.53, "tie artifact", fontsize=6.5, color=FS.INK2, va="center")

    FS.save(fig, "fig_rank_ties")


if __name__ == "__main__":
    main()
