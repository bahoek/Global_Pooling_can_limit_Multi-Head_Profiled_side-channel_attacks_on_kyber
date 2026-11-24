#!/usr/bin/env python3
r"""
analyze_d29 — interpreting the D29 result (N29 with 200 epochs + patience 50)

    python run_controls.py D29      # about 8 hours
    python analyze_d29.py           # no arguments

Why there is a separate interpretation script
---------------------------------------------
The branching proposed by the paper chat **is inverted in the second branch**:

    "If it stays near 88.7% -> capacity.  If it rises to 93% -> budget, reconfirming the manuscript's conclusion"

The second branch is not a reconfirmation. What is being asked about is **the drop between N29 and N39**, and if the schedule
lifts both equally regardless of N, that drop **remains unexplained**. In numbers:

    s0 original gap          86.81 - 61.25 = 25.56 %p
    if N29 does not rise     86.81 - 74.97 = 11.84 %p  -> schedule explains 13.72 (54%)  [manuscript claim]
    if N29 reaches 93        93.00 - 74.97 = 18.03 %p  -> schedule explains  7.53 (29%)  [claim weakened]

So this script computes not "did it rise or not" but **how much the gap closed**,
and issues the verdict from that ratio. The reference values are also hard-coded — 88.7 is a number found in no seed
(N29 10 seeds: 86.81 86.47 86.43 86.68 87.01 87.47 87.01 86.82 87.19 86.64, max 87.47).

The same confound as in D is present in D29
-------------------------------------------
Line 390 of `train_nscan.py` is `CosineAnnealingLR(T_max=EPOCHS)`, so `EPOCHS=200` also changes the LR
schedule. **So D29 moves the same axis as D — that is the point of this comparison.**
Do not use the word "budget"; write "schedule".
"""

import csv
import os
import re
import sys

import numpy as np

# ---------------- reference values (extracted from all_runs.csv and hard-coded) ----------------
CTRL_DIR = os.environ.get("TIFS_CTRL",
                          "/home/bahoek/data/kyber/tifs/model_ctrl")

N29_SEEDS = [86.8120, 86.4718, 86.4346, 86.6821, 87.0125,
             87.4727, 87.0059, 86.8247, 87.1944, 86.6417]
N29_EPOCHS = [84, 87, 93, 90, 90, 78, 99, 92, 94, 92]      # 10/10 ended by patience
N39_SEEDS = [61.2542, 59.4724, 69.5167, 64.4097, 61.9583,
             66.0712, 61.7816, 66.1990, 61.7292, 56.3218]
N39_EPOCHS = [100] * 10                                     # 10/10 hit the budget wall

N29_S0 = N29_SEEDS[0]      # 86.81  <- the reference for D29. Not the 10-seed mean (86.86)
N39_S0 = N39_SEEDS[0]      # 61.25
D_FINAL = 74.97            # D (N39 s0, 200-epoch setting) final
D_AT_100 = 73.85           # D's value at epoch 100
N9_MEAN = 93.37            # for the monotonicity check
N19_MEAN = 90.74

ORIG_GAP = N29_S0 - N39_S0                 # 25.56
D_RECOVERED = D_FINAL - N39_S0             # 13.72

ACC_RE = re.compile(r"\|\s*acc\s+([0-9.]+)%")
EP_RE = re.compile(r"s\d+\s+(\d+)/(\d+)\]")


def read_log(path):
    """(epoch array, acc array). Reads the console log left by run_controls."""
    acc = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m, me = ACC_RE.search(line), EP_RE.search(line)
        if m and me and " GE " in line:
            acc[int(me.group(1))] = float(m.group(1))
    if not acc:
        raise SystemExit(f"could not read epoch lines: {path}")
    ep = np.array(sorted(acc))
    return ep, np.array([acc[e] for e in ep])


def virtual_stop(accs, patience=15, min_delta=0.3):
    """Stopping epoch (1-based) from applying the original rule to the same trajectory. Same as train_nscan lines 467~472."""
    best, cnt = -1e9, 0
    for e, a in enumerate(accs, 1):
        if a > best + min_delta:
            best, cnt = a, 0
        else:
            cnt += 1
            if cnt >= patience:
                return e
    return None


def find_run():
    """Find the D29 log and ctrl_summary row."""
    sub = os.path.join(CTRL_DIR, "D29")
    log = None
    if os.path.isdir(sub):
        for f in sorted(os.listdir(sub)):
            if f.startswith("log_") and f.endswith(".txt"):
                log = os.path.join(sub, f)
    row = None
    sp = os.path.join(CTRL_DIR, "ctrl_summary.csv")
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp, encoding="utf-8")):
            if r.get("experiment") == "D29":
                row = r
    return log, row


def main():
    log, row = (sys.argv[1], None) if len(sys.argv) > 1 else find_run()

    print("=" * 72)
    print("D29 — ResNet N29 s0 with the same setting as D (EPOCHS=200, PATIENCE=50)")
    print("=" * 72)

    # ---- what can be read even before running ----
    print("\n[0] What the existing data already says (before spending 8 hours)")
    print(f"    N39 epochs_run : {N39_EPOCHS}")
    print("                     -> all 10/10 are 100. Cut off at the budget wall")
    print(f"    N29 epochs_run : {N29_EPOCHS}")
    print("                     -> all 10/10 ended by patience. Not a single seed hit the wall")
    print("    So N29 ended after it had already plateaued. The expected landing point is near 86.8.")
    print(f"\n    reference: N29 s0 = {N29_S0:.2f}%  (10-seed mean {np.mean(N29_SEEDS):.2f}, "
          f"max {max(N29_SEEDS):.2f})")
    print("    ** 88.7% is a number found in no seed. Judging by that value would skew the conclusion **")

    if log is None or not os.path.exists(log):
        print(f"\nNot run yet. No log ({os.path.join(CTRL_DIR, 'D29')}).")
        print("    python run_controls.py D29")
        print("\n[note] Verdicts per outcome are precomputed:")
        table(None)
        return

    ep, acc = read_log(log)
    final = float(acc.max())
    at100 = float(acc[ep == 100][0]) if (ep == 100).any() else None
    vs = virtual_stop(acc)

    print(f"\n[1] trajectory   {log}")
    print(f"    epochs run    {int(ep[-1])}")
    print(f"    best acc      {final:.2f}%   (original s0 {N29_S0:.2f}%, "
          f"diff {final - N29_S0:+.2f}%p)")
    if at100 is not None:
        print(f"    epoch 100     {at100:.2f}%   (relative to where the original ended "
              f"{at100 - N29_S0:+.2f}%p)")
        print(f"    after epoch 100 {final - at100:+.2f}%p")
    print(f"    patience-15 virtual stop  ep{vs}" if vs else "    patience-15 does not stop it")
    if row:
        print(f"    ctrl_summary: best_epoch {row.get('best_epoch')}, "
              f"virtual_p15_stop {row.get('virtual_p15_stop')}, "
              f"{row.get('time_min')} min")

    table(final)


def table(final):
    print("\n[2] Gap computation — this is the basis of the verdict")
    print(f"    original s0 gap      {N29_S0:.2f} - {N39_S0:.2f} = {ORIG_GAP:.2f} %p")
    print(f"    amount D recovered   {D_FINAL:.2f} - {N39_S0:.2f} = {D_RECOVERED:.2f} %p")
    print()
    print(f"    {'N29 result':>10}  {'new gap':>9}  {'sched. share':>13}  verdict")
    print("    " + "-" * 66)
    rows = [86.8, 88.0, 89.0, 90.0, 91.5, 93.0]
    if final is not None:
        rows = sorted(set(rows + [round(final, 2)]))
    for v in rows:
        gap = v - D_FINAL
        closed = ORIG_GAP - gap
        frac = closed / ORIG_GAP * 100
        if frac >= 45:
            verdict = "schedule explains at least half of the drop. Manuscript claim stands"
        elif frac >= 30:
            verdict = "weakened. Lower the explained share in the text, with numbers"
        else:
            verdict = "global rise. Drop unexplained — rewrite the D paragraph"
        mark = " <-- measured" if final is not None and abs(v - final) < 0.005 else ""
        print(f"    {v:>10.2f}  {gap:>9.2f}  {closed:>6.2f} ({frac:>4.1f}%)  "
              f"{verdict}{mark}")

    print("\n[3] Monotonicity check")
    print(f"    N9 {N9_MEAN:.2f} > N19 {N19_MEAN:.2f} > N29 {np.mean(N29_SEEDS):.2f}  (100 epochs)")
    if final is not None:
        if final > N19_MEAN:
            print(f"    ** N29 reached {final:.2f}, exceeding N19 ({N19_MEAN:.2f}). "
                  f"N monotonicity breaks under the long schedule — that is a separate finding in itself **")
        else:
            mu, sd = float(np.mean(N29_SEEDS)), float(np.std(N29_SEEDS, ddof=1))
            band = mu + 3 * sd
            print(f"    10 seeds {mu:.2f} ± {sd:.2f} (SD), observed range "
                  f"{min(N29_SEEDS):.2f}~{max(N29_SEEDS):.2f}, +3SD = {band:.2f}")
            if final > band:
                print(f"    N29 is {final:.2f} — outside +3SD. The rise is real")
            elif final > max(N29_SEEDS):
                print(f"    N29 is {final:.2f} — higher than the 10-seed max ({max(N29_SEEDS):.2f}) but "
                      f"within +3SD ({band:.2f}).\n"
                      f"    One seed cannot establish a rise. The verdict is made by the gap in [2]")
            else:
                print(f"    N29 is {final:.2f} — within the 10-seed range. Read as no rise")

    print("\n[4] How to word it")
    print("    Do not write 'budget'. Line 390 of train_nscan.py is")
    print("    CosineAnnealingLR(T_max=EPOCHS), so EPOCHS=200 also changes the LR schedule.")
    print("    The axis that D and D29 move is 'schedule' in both cases.")
    print("=" * 72)


if __name__ == "__main__":
    main()
