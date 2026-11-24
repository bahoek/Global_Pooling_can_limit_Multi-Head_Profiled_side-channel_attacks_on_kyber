#!/usr/bin/env python3
r"""
d29_train — train ResNet N29 s0 with the same settings as D (EPOCHS=200, PATIENCE=50)

    python d29_train.py

**No existing file is modified.** It imports `run_controls.py`, inserts D29 into `EXPERIMENTS`,
and runs it as is. Checkpoint resume, epoch logs, `virtual_p15_stop` and comparison with the original run
all come along, already verified. Results go to `model_ctrl/D29/` and `model_ctrl/ctrl_summary.csv`.

What the experiment asks
--------------------
D showed that "N39 ends under-trained on the common schedule". This separates whether that effect comes **only to N39**
(= it explains the N-dependent drop) or lifts **everything regardless of N** (= it does not explain the drop).
The decision is made by `analyze_d29.py` on the gap — not on "did it go up".

Note: line 390 of `train_nscan.py` is `CosineAnnealingLR(T_max=EPOCHS)`, so `EPOCHS=200`
also changes the LR schedule. The axis that moves is not the 'budget' but the **'schedule'**.

About 6~9 hours (N29 takes 2.71 min per epoch, nearly fixed at 2.68~2.72 across all 10 runs).
If patience 50 fires at epoch 135 it is 6.1h, at 170 it is 7.7h, and going to 200 it is 9.0h.
If interrupted, running again resumes.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SPEC = dict(
    desc="N29 with the same settings as D (200 epochs + patience50) — is the schedule effect N39-specific or global",
    arch="resnet", ns=[29], seeds=[0],
    over=dict(EPOCHS=200, PATIENCE=50),
)


def main():
    try:
        import run_controls as RC
    except ImportError as e:
        raise SystemExit(f"Could not import run_controls.py: {e}\n"
                         f"  It must be in the same folder: {HERE}")

    for need in ("EXPERIMENTS", "main"):
        if not hasattr(RC, need):
            raise SystemExit(f"run_controls.py has no {need}. It seems to be a different version.")

    cur = RC.EXPERIMENTS.get("D29")
    if cur is None:
        RC.EXPERIMENTS["D29"] = SPEC
        print("Inserted D29 into run_controls.EXPERIMENTS (the file is not modified)")
    elif cur.get("over") != SPEC["over"] or cur.get("ns") != SPEC["ns"]:
        raise SystemExit(
            f"run_controls.py already has D29, but its settings differ.\n"
            f"  file side: ns={cur.get('ns')} seeds={cur.get('seeds')} over={cur.get('over')}\n"
            f"  this side: ns={SPEC['ns']} seeds={SPEC['seeds']} over={SPEC['over']}\n"
            f"  Decide which one is right and keep only one.")
    else:
        print("run_controls.py already has the same D29. Using it as is")

    print(f"  arch={SPEC['arch']}  N={SPEC['ns']}  seed={SPEC['seeds']}  over={SPEC['over']}")
    print("  Expected 6~9 hours (N29 at 2.71 min per epoch x the epoch at which patience 50 fires)\n")

    sys.argv = [sys.argv[0], "D29"]      # run_controls selects the experiment by argument
    RC.main()

    print("\nWhen finished, next:")
    print("  python analyze_d29.py        # gap / explained-share decision")
    print("  python d29_eval.py           # per-head accuracy (pair matching table)")


if __name__ == "__main__":
    main()
