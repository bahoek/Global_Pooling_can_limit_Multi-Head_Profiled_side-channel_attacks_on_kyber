#!/usr/bin/env python3
"""
fig_controls_curves — per-epoch trajectories of control experiments D / E

(a) D: trajectory of ResNet N39 s0 retrained with EPOCHS=200 / PATIENCE=50.
    **Caution.** run_controls.py passes over=dict(EPOCHS=200, ...), and
    the scheduler in train_nscan.py is CosineAnnealingLR(T_max=EPOCHS), so
    D is not an experiment that "gave more epochs" but one that **"cooled the LR at half the rate"**.
    In fact D at epoch 100 is already 12.6%p ahead of the original run's final value (61.25%),
    and epochs 100 -> 169 add only 1.1%p. The figure does not hide this; it
    marks both epoch 100 and the original final value so the reader sees it immediately.

(b) E: trajectories of MobileNetV2 N9 retrained with PATIENCE=50 (seed 0/1/3).
    E did not touch EPOCHS, so the LR schedule is the same as the original — a clean control.
    The overlapping segment matches the original run to the second decimal place (s0 11.56 vs 11.56).
    The virtual stopping point obtained by applying the original rule (patience 15, min_delta 0.3) to the same trajectory
    is drawn with an open marker. That all of these points lie inside the plateau is the conclusion of E.

Usage — **just run it with no arguments**
    python fig_controls_curves.py

The script finds the logs **by their content, not by file name**. Since a line looks like
    [resnet N39(78h) s0 001/200] train 2.5595 | val 2.5021 | acc 11.22% | GE 5.566 | gn 0.312
it reads backbone·N·seed·epoch budget from the header to tell D and E apart:
    D = resnet,    N=39, s0,        budget 200
    E = mobilenet, N=9,  s0/s1/s3,  budget 100 and ran more than 50 epochs (the original runs end at 17~19)
It searches `DATA_ROOT` and the current folder, up to 4 levels below. If the location differs, change only the one
`DATA_ROOT` line at the top or set the environment variable `TIFS_ROOT=/some/where`. To give them directly, as before:
    python fig_controls_curves.py <D_log> <E_s0> <E_s1> <E_s3>
"""

import os
import re
import sys

import numpy as np

import figstyle as FS

# ---------------- finding logs ----------------
# If the location differs, change **only this one line** (or the TIFS_ROOT environment variable).
DATA_ROOT = os.environ.get("TIFS_ROOT", "/home/bahoek/data/kyber/tifs")
LOG_EXT = (".txt", ".log", ".out")
SCAN_DEPTH = 4

# Values of the original run (EPOCHS=100, PATIENCE=15). Same values as the orig_* columns of all_runs.csv.
D_ORIG_ACC = 61.2542        # resnet_r18_N39_s0 final acc
D_ORIG_EPOCHS = 100         # cut off at the 100-epoch wall (best_epoch == 100)
E_ORIG = {0: (19, 11.5618), 1: (18, 11.5167), 3: (17, 11.4906)}   # (epochs_run, final acc)

PATIENCE_ORIG, MIN_DELTA_ORIG = 15, 0.3

LINE_RE = re.compile(
    r"\[\s*(\S+)\s+N(\d+)\(\d+h\)\s+s(\d+)\s+(\d+)/(\d+)\]"
    r".*?acc\s+([\d.]+)%"
)


def peek(path):
    """Open one log and return (arch, N, seed, epoch budget, last epoch). None if it is not a training log."""
    first, last = None, None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = LINE_RE.search(line)
                if m:
                    if first is None:
                        first = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(5)))
                    last = int(m.group(4))
    except OSError:
        return None
    return first + (last,) if first else None


def scan_logs():
    """Scan under DATA_ROOT and the current folder and find 1 D log and 3 E logs **by content**."""
    import glob
    seen, found = set(), {}
    for base in (DATA_ROOT, "."):
        if not os.path.isdir(base):
            continue
        for d in range(0, SCAN_DEPTH + 1):
            for c in glob.glob(os.path.join(base, *(["*"] * d), "*")):
                if not c.lower().endswith(LOG_EXT) or not os.path.isfile(c):
                    continue
                rp = os.path.realpath(c)
                if rp in seen:
                    continue
                seen.add(rp)
                info = peek(c)
                if not info:
                    continue
                arch, n, seed, budget, lastep = info
                # D: ResNet N39 s0 run with a 200-epoch budget
                if arch == "resnet" and n == 39 and seed == 0 and budget == 200:
                    found.setdefault("D", c)
                # E: MobileNet N9 s0/s1/s3 that ran more than 50 epochs with a 100-epoch budget.
                #    The original runs end at epochs 17~19 because of patience 15, so this separates them
                elif (arch == "mobilenet" and n == 9 and seed in (0, 1, 3)
                      and budget == 100 and lastep > 50):
                    found.setdefault(f"E{seed}", c)
    return found


def parse(path):
    """log -> (epochs, accs). If an epoch number is duplicated, the last one is used."""
    if not os.path.exists(path):
        raise SystemExit(f"log not found: {path}\n  (see the docstring at the top of the file for usage)")
    acc = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = LINE_RE.search(line)
        if m:
            # groups: 1 arch, 2 N, 3 seed, 4 epoch, 5 epoch budget, 6 acc
            acc[int(m.group(4))] = float(m.group(6))
    if not acc:
        raise SystemExit(f"could not read a single epoch line: {path}")
    ep = np.array(sorted(acc))
    return ep, np.array([acc[e] for e in ep])


def virtual_stop(accs, patience=PATIENCE_ORIG, min_delta=MIN_DELTA_ORIG):
    """Same rule as run_controls.virtual_stop. 1-based stopping epoch, None if it never stops."""
    best, cnt = -1e9, 0
    for e, a in enumerate(accs, 1):
        if a > best + min_delta:
            best, cnt = a, 0
        else:
            cnt += 1
            if cnt >= patience:
                return e
    return None


def panel_d(ax, ep, acc):
    ax.plot(ep, acc, "-", color=FS.INK, lw=1.1, zorder=3)

    # where the original schedule ended
    ax.axvline(D_ORIG_EPOCHS, color=FS.INK3, lw=0.6, ls=":", zorder=1)
    # final accuracy of the original run
    ax.axhline(D_ORIG_ACC, color=FS.INK2, lw=0.8, ls="--", zorder=1)

    a100 = float(acc[ep == D_ORIG_EPOCHS][0])
    aend = float(acc[-1])

    # draw the gap at the same epoch as a vertical bar so it is visible
    ax.plot([D_ORIG_EPOCHS, D_ORIG_EPOCHS], [D_ORIG_ACC, a100],
            "-", color=FS.INK, lw=2.4, solid_capstyle="butt", zorder=2)
    ax.plot([D_ORIG_EPOCHS], [a100], "o", color=FS.INK, ms=3.4, zorder=4)
    ax.plot([D_ORIG_EPOCHS], [D_ORIG_ACC], "s", color="white",
            markeredgecolor=FS.INK2, markeredgewidth=0.8, ms=3.4, zorder=4)

    ax.text(D_ORIG_EPOCHS + 4, (a100 + D_ORIG_ACC) / 2, f"{a100 - D_ORIG_ACC:+.1f}",
            ha="left", va="center", fontsize=7, color=FS.INK)

    # the amount added after epoch 100 — shown with a bracket at the top
    yb, tick = 80.0, 1.6
    ax.plot([D_ORIG_EPOCHS, ep[-1]], [yb, yb], "-", color=FS.INK, lw=0.6, zorder=2)
    for x in (D_ORIG_EPOCHS, ep[-1]):
        ax.plot([x, x], [yb - tick, yb], "-", color=FS.INK, lw=0.6, zorder=2)
    ax.text((D_ORIG_EPOCHS + ep[-1]) / 2, yb + 0.8, f"{aend - a100:+.1f}",
            ha="center", va="bottom", fontsize=7, color=FS.INK)

    ax.text(D_ORIG_EPOCHS - 4, 14, "100-epoch\nschedule ends",
            ha="right", va="bottom", fontsize=6.5, color=FS.INK3)
    ax.text(ep[-1], D_ORIG_ACC - 2.0, "original run, final",
            ha="right", va="top", fontsize=6.5, color=FS.INK2)

    ax.set_xlim(0, ep[-1] + 2)
    ax.set_ylim(5, 87)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    FS.grid(ax)


def panel_e(ax, runs):
    stops = []
    for i, (seed, (ep, acc)) in enumerate(sorted(runs.items())):
        col = [FS.INK, FS.INK2, FS.INK3][i % 3]
        ax.plot(ep, acc, FS.LINES[i % len(FS.LINES)], color=col, lw=1.0,
                zorder=3, label=f"seed {seed}")

        vs = virtual_stop(acc)
        if vs is not None:
            stops.append(vs)
            ax.plot([vs], [acc[ep == vs][0]], FS.MARKERS[i % len(FS.MARKERS)],
                    color="white", markeredgecolor=col, markeredgewidth=0.9,
                    ms=4.2, zorder=5)

    # The stopping points of the three seeds nearly overlap at ep17~19. So they do not smear into a blob,
    # they are grouped and pulled out with a single leader line.
    if stops:
        ax.annotate(f"patience-15 stop\n(ep {min(stops)}–{max(stops)})",
                    xy=(float(np.mean(stops)), 12.5), xytext=(4, 62),
                    ha="left", va="center", fontsize=6.5, color=FS.INK2,
                    arrowprops=dict(arrowstyle="-", color=FS.INK2,
                                    lw=0.6, shrinkA=2, shrinkB=3))

    ax.set_xlim(0, 101)
    ax.set_ylim(5, 97)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.legend(loc="lower right", fontsize=7)
    FS.grid(ax)


def main():
    if len(sys.argv) >= 5:
        paths = dict(zip(("D", "E0", "E1", "E3"), sys.argv[1:5]))
    else:
        paths = scan_logs()
        missing = [k for k in ("D", "E0", "E1", "E3") if k not in paths]
        if missing:
            raise SystemExit(
                f"could not find logs: {', '.join(missing)}\n"
                f"  searched: {DATA_ROOT} and the current folder, {SCAN_DEPTH} levels below\n"
                f"  found:    {paths if paths else 'none'}\n"
                f"  fix: edit DATA_ROOT at the top of this file,\n"
                f"       or run TIFS_ROOT=/path python fig_controls_curves.py,\n"
                f"       or give them directly: python fig_controls_curves.py <D_log> <E_s0> <E_s1> <E_s3>")
        for k in ("D", "E0", "E1", "E3"):
            print(f"  {k:<3} {paths[k]}")
        print()

    d_ep, d_acc = parse(paths["D"])
    runs = {seed: parse(paths[f"E{seed}"]) for seed in (0, 1, 3)}

    # ---- leave the numbers as-is on the console. Copy the caption values from here ----
    a100 = float(d_acc[d_ep == D_ORIG_EPOCHS][0])
    print("[D] ResNet N39 s0, EPOCHS=200 / PATIENCE=50")
    print(f"    ep{D_ORIG_EPOCHS:<4d} acc {a100:6.2f}%   "
          f"(original run final {D_ORIG_ACC:.2f}%, diff {a100 - D_ORIG_ACC:+.2f}%p)")
    print(f"    ep{d_ep[-1]:<4d} acc {d_acc[-1]:6.2f}%   "
          f"(vs ep{D_ORIG_EPOCHS}: {d_acc[-1] - a100:+.2f}%p)")
    vd = virtual_stop(d_acc)
    print(f"    patience-15 virtual stop: ep{vd}  acc {d_acc[d_ep == vd][0]:.2f}%"
          if vd else "    patience-15 does not stop it")
    print("    -> most of the gain is already there within epoch 100."
          " The cause is not the epoch budget but the LR schedule stretched by T_max=EPOCHS.")
    print()
    print("[E] MobileNetV2 N9, PATIENCE=50 (EPOCHS unchanged at 100)")
    for seed, (ep, acc) in sorted(runs.items()):
        oe, oa = E_ORIG[seed]
        at_oe = float(acc[ep == oe][0])
        vs = virtual_stop(acc)
        print(f"    s{seed}: final ep{ep[-1]} acc {acc[-1]:6.2f}%  |  "
              f"ep{oe} acc {at_oe:6.2f}% vs original final {oa:6.2f}% "
              f"(diff {at_oe - oa:+.3f}%p)  |  virtual p15 stop ep{vs}")

    fig, axes = FS.figure(FS.TWO_COL, 2.45, ncols=2)
    panel_d(axes[0], d_ep, d_acc)
    panel_e(axes[1], runs)
    for ax, tag in zip(axes, "ab"):
        ax.text(0.5, -0.30, f"({tag})", transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
    fig.subplots_adjust(wspace=0.28, bottom=0.24)
    FS.save(fig, "fig_controls_curves")


if __name__ == "__main__":
    main()
