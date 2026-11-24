#!/usr/bin/env python3
r"""
e6_mbv2_std — E6: retrain with the standard MobileNetV2 implementation (linear bottleneck)

Request document: `claude/TIFS additional experiment request before resubmission … (11-17)` section 8.

    python e6_mbv2_std.py --diff      # show only the changed lines and exit  <- do this first
    python e6_mbv2_std.py             # N 29·39 x seeds 0·1·2 = 6 runs
    python e6_mbv2_std.py --with-n9   # add N 9 seeds 0·1·2 = 9 runs

**No existing file is modified.** The experiment is inserted into `run_controls.py`,
only `train_nscan.build_model` is swapped out, and then it runs as is. Checkpoint resume, epoch logs,
`virtual_p15_stop` and the summary CSV all come along, already verified.

The change is one line
-----------------
The last line of `train_nscan.DWSep1d.forward`.

    variant (main text)   return F.relu6(self.bn2(self.pw(x)))     # ReLU6 after projection
    standard              return self.bn2(self.pw(x))              # BN only = linear bottleneck

The inverted residual of Sandler et al. (2018) places no activation after the projection (1x1).
That is because the point of that paper is that ReLU erases information in a narrow bottleneck.
Everything else (expansion 1x1 + BN + ReLU6, depthwise + BN + ReLU6, widths, stages, strides, final 1280-channel
1x1, GAP, heads) is identical to the main-text model.

Settings are exactly those of the main training — 64,000 / 16,000, 100 epochs, AdamW, cosine T_max=100,
patience 15, AMP, same lr, weight decay and batch. Seed numbers are also 0·1·2, the same as the main training, so
runs with the same seed can be placed side by side.

Results go to `model_ctrl/E6/` and `model_ctrl/ctrl_summary.csv` (`experiment = E6`).
3 runs per N, 5~8 hours per run → **6 runs take 24~48 hours**.
If interrupted, running again resumes.
"""

import os
import sys

import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

SPEC = dict(
    desc="E6: MobileNetV2 standard inverted residual (linear bottleneck, "
         "no activation after the 1x1 projection)",
    arch="mobilenet", ns=[29, 39], seeds=[0, 1, 2], over=dict(),
)


def make_std(T):
    """Standard inverted residual. Inherits train_nscan's class and changes only one line of forward."""

    class StdDWSep1d(T.DWSep1d):
        def forward(self, x):
            x = F.relu6(self.bn1(self.dw(x)))
            return self.bn2(self.pw(x))          # ← standard: no activation after projection

    class StdInvRes1d(nn.Module):
        """Same as T.InvRes1d but with the standard DWSep1d."""

        def __init__(self, cin, cout, stride, t):
            super().__init__()
            hid = int(cin * t)
            self.use_res = (stride == 1 and cin == cout)
            L = []
            if t != 1:
                L += [nn.Conv1d(cin, hid, 1, 1, 0, bias=False),
                      nn.BatchNorm1d(hid), nn.ReLU6()]
            L.append(StdDWSep1d(hid, cout, 3, stride, 1))
            self.conv = nn.Sequential(*L)

        def forward(self, x):
            return x + self.conv(x) if self.use_res else self.conv(x)

    class StdMobileNetV2(T.MobileNetV2):
        def __init__(self, num_targets):
            super().__init__(num_targets)
            # Rebuild only the body with the standard version. stem, proj and classifiers stay as they are
            L, cin = [], 32
            for t, c, n, s in self.CFG:
                for i in range(n):
                    L.append(StdInvRes1d(cin, c, s if i == 0 else 1, t))
                    cin = c
            self.features = nn.Sequential(*L)

    return StdMobileNetV2


def show_diff():
    print("=" * 74)
    print("E6 — standard MobileNetV2 implementation. The only change is the last line of DWSep1d.forward")
    print("=" * 74)
    print("""
  variant (main text, train_nscan.py):
      def forward(self, x):
          x = F.relu6(self.bn1(self.dw(x)))
          return F.relu6(self.bn2(self.pw(x)))     # ReLU6 after projection

  standard (E6, Sandler et al. 2018):
      def forward(self, x):
          x = F.relu6(self.bn1(self.dw(x)))
          return self.bn2(self.pw(x))              # BN only = linear bottleneck
""")
    print("  Everything else is identical — expansion 1x1 + BN + ReLU6, depthwise + BN + ReLU6,")
    print("  CFG widths, stages, strides, final 1280-channel 1x1, GAP, head structure, training settings.")
    print("  The parameter count is also the same (only an activation is removed).")
    print("=" * 74)


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    if "--diff" in flags:
        show_diff()
        return

    spec = dict(SPEC)
    if "--with-n9" in flags:
        spec["ns"] = [9, 29, 39]

    try:
        import run_controls as RC
        import train_nscan as T
    except ImportError as e:
        raise SystemExit(f"Could not import: {e}\n  They must be in the same folder: {HERE}")
    for need in ("EXPERIMENTS", "main"):
        if not hasattr(RC, need):
            raise SystemExit(f"run_controls.py has no {need}. It seems to be a different version.")

    show_diff()

    Std = make_std(T)
    base_build = T.build_model

    def build_model(arch, num_targets):
        if arch == "mobilenet":
            return Std(num_targets)
        return base_build(arch, num_targets)

    T.build_model = build_model                   # ← changed only within this process

    # Check that the parameter count equals the main-text model (only an activation is removed, so it must be equal)
    a = sum(p.numel() for p in base_build("mobilenet", 78).parameters())
    b = sum(p.numel() for p in Std(78).parameters())
    print(f"\nParameter count  variant {a/1e6:.3f}M  vs  standard {b/1e6:.3f}M  "
          f"{'match' if a == b else '**differ — needs checking**'}")
    if a != b:
        raise SystemExit("Parameter counts differ. The structure changed in an unintended way.")

    cur = RC.EXPERIMENTS.get("E6")
    if cur is None:
        RC.EXPERIMENTS["E6"] = spec
        print("Inserted E6 into run_controls.EXPERIMENTS (the file is not modified)")
    elif cur.get("ns") != spec["ns"] or cur.get("seeds") != spec["seeds"]:
        raise SystemExit(
            f"run_controls.py already has E6, but its settings differ.\n"
            f"  file side: ns={cur.get('ns')} seeds={cur.get('seeds')}\n"
            f"  this side: ns={spec['ns']} seeds={spec['seeds']}")
    else:
        print("run_controls.py already has the same E6. Using it as is")

    n = len(spec["ns"]) * len(spec["seeds"])
    print(f"  arch=mobilenet  N={spec['ns']}  seeds={spec['seeds']}  -> {n} run")
    print(f"  Expected {n*5}~{n*8} hours. If interrupted, running again resumes\n")

    sys.argv = [sys.argv[0], "E6"]
    RC.main()

    print("\nWhen finished, next:")
    print("  python eval_controls.py            # coarse key recovery of converged runs")
    print("  (fine separately, the same way as d29_qfine.py — just ask if needed)")
    print("  To place them side by side with the main-text model at the same N and seed, compare with the")
    print("  mobilenet N29·N39 s0·s1·s2 rows of all_runs.csv")


if __name__ == "__main__":
    main()
