#!/usr/bin/env python3
r"""
receptive_field — count the backbone's receptive field (RF) and position spacing (jump) **directly from the model**

    python receptive_field.py

Request document F-2-1: provide "the values the experiment chat computed from the actual kernel, stride and padding".
The paper chat's estimate was about 1,079 samples per position and about 1,300 samples for 4 positions.

Instead of counting by hand, it walks the modules built by `train_nscan.py`. If the model definition changes,
these values change with it, so they cannot drift apart.

Formula (same as written in the request)
    RF   <- RF + (k - 1) * jump
    jump <- jump * stride
A 1x1 conv does not enlarge the RF. BatchNorm, ReLU and Dropout have no effect either.

**Residual shortcuts are not counted.** `ResidualBlock1d.sc` is **parallel** to the main path, so
counting it in series multiplies the stride twice (counting that way gives RF 4719 / jump 640, which is wrong).
MobileNetV2's residual is `x + conv(x)` and has no module, so it does not apply.

Range the local readout actually sees
    one position           = RF
    n positions (adjacent) = RF + (n - 1) * jump
`probe_local.py` converts ±HALF(160) samples around the peak into feature-map coordinates and averages, so
ResNet sees 320/jump positions, and MobileNetV2 also sees 320/jump positions.
"""

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HALF = 160                                   # probe_local.HALF
SKIP = {"sc", "shortcut", "downsample"}      # residual shortcut — parallel to the main path, so not counted


def walk(mod, rf=1, jump=1, trace=None, path=""):
    """Walk the module tree in execution order, accumulating RF and jump."""
    if trace is None:
        trace = []
    for name, m in mod.named_children():
        if name in SKIP:                       # parallel branch. See the explanation below
            continue
        p = f"{path}.{name}" if path else name
        if isinstance(m, nn.Conv1d):
            k = m.kernel_size[0]
            s = m.stride[0]
            g = m.groups
            if k > 1:
                rf = rf + (k - 1) * jump
            jump = jump * s
            trace.append((p, f"Conv1d k{k} s{s}" + (" (dw)" if g > 1 else ""), rf, jump))
        elif isinstance(m, (nn.MaxPool1d, nn.AvgPool1d)):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            rf = rf + (k - 1) * jump
            jump = jump * s
            trace.append((p, f"{type(m).__name__} k{k} s{s}", rf, jump))
        else:
            rf, jump, trace = walk(m, rf, jump, trace, p)
    return rf, jump, trace


def report(arch, model, verbose=False):
    # the heads (classifiers) come after the feature map, so they do not enter the RF
    body = nn.ModuleList([m for n, m in model.named_children() if n != "classifiers"])
    rf, jump, tr = walk(body)
    print(f"\n{'=' * 72}\n{arch}\n{'=' * 72}")
    if verbose:
        print(f"  {'layer':<46}{'RF':>8}{'jump':>8}")
        for p, d, r, j in tr:
            print(f"  {p + '  ' + d:<46}{r:>8}{j:>8}")
    n_pos = max(1, (2 * HALF) // jump)
    print(f"  {len(tr)} conv/pool layers")
    print(f"  **receptive field RF = {rf} samples**, position spacing jump = {jump} samples")
    print(f"  local readout uses peak ±{HALF} samples -> {n_pos} feature-map positions")
    print(f"  range those {n_pos} positions actually see = RF + ({n_pos}-1)*jump "
          f"= **{rf + (n_pos - 1) * jump} samples**")
    print(f"  (reading only one position = {rf} samples — the S readout of E5)")
    span = rf + (n_pos - 1) * jump
    print(f"  **E1's W-RF window = center ±{span // 2} ({span // 2 * 2} samples)** "
          f"— same as the range the local head sees")
    return rf, jump, n_pos


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    try:
        import eval_nscan as E
        T = E.import_train()
    except Exception as e:
        raise SystemExit(f"Could not import train_nscan: {e}\n"
                         f"  Run it from the same folder: {HERE}")

    out = {}
    for arch, n in (("resnet", 39), ("mobilenet", 39)):
        model = E.build(T, arch, n, 2 * n, "resnet18" if arch == "resnet" else None)
        out[arch] = report(arch, model, verbose)

    print(f"\n{'=' * 72}\nLines to send for F-2-1")
    print("=" * 72)
    for arch, (rf, jump, n_pos) in out.items():
        name = "ResNet-18-1D" if arch == "resnet" else "MobileNetV2-1D"
        print(f"  {name:<16} RF {rf:>5} samples, stride {jump:>4} samples/position, "
              f"{n_pos} positions in ±{HALF} -> {rf + (n_pos-1)*jump} samples")
    print("\n  Pass -v to print the per-layer accumulation in full")


if __name__ == "__main__":
    main()
