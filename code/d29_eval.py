#!/usr/bin/env python3
r"""
d29_eval — per-head evaluation of the D·D29 checkpoints on the **test set** (for the pair matching table)

    python d29_eval.py

**It does not modify any existing file.** It imports `eval_controls.py`, swaps in only `JOBS` and the output folder,
and runs it. Results go to **`eval_ctrl_d/`**.

Why a separate folder — otherwise the figures break
---------------------------------------------------
D is `resnet N39 s0`, D29 is `resnet N29 s0`. **Their `(arch, N, seed)` are identical to names
already present in the main training.** `tally()` in `fig_convrate.py` merges the main evaluation and
`eval_ctrl/`, counting the same `(arch, N, seed)` only once. If D were put into `eval_ctrl/`,
the convergence-rate figure **could slot the retraining result into the main-training spot**.
Right now the main evaluation, which comes first, survives thanks to the order `tally(main_eval, ctrl)`, but having
the result depend on argument order is dangerous. **D·D29 are not convergence-rate data points to begin with** — they are
retrainings of runs already counted. So they are separated into `eval_ctrl_d/`. No figure reads this folder.

Why the test set
----------------
The 100-epoch rows of the pair matching table are test values. The 74.97 in `train_summary` is a **validation** value, so
it cannot go in the same table (the difference is small — N29 s0 validation 86.81 / test 86.77).

What it reads
-------------
Per-head accuracy from `eval_ctrl_d/eval_heads.csv`. Two heads make one pair (even r0, odd r1).
pairs 1~9 = heads 0~17, pairs 1~29 = heads 0~57, pairs 30~39 = heads 58~77.
With `--table` this script groups them itself and prints a table.
"""

import csv
import inspect
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

JOBS_D = [("D",   "resnet", 39, [0]),
          ("D29", "resnet", 29, [0])]

# pair blocks (1-based pair numbers, inclusive) -> head indices are 2*(p-1) .. 2*p-1
BLOCKS = [("pair 1-9", 1, 9), ("pair 1-29", 1, 29), ("pair 30-39", 30, 39)]

# comparison reference (extracted from eval_heads.csv and hard-coded. ResNet s0, test 2000 traces)
REF = {
    ("N29", "100 epochs"): {"pair 1-9": 88.67, "pair 1-29": 86.77, "pair 30-39": None, "all": 86.77},
    ("N39", "100 epochs"): {"pair 1-9": 69.09, "pair 1-29": 61.10, "pair 30-39": 61.26, "all": 61.14},
}


def run_eval():
    try:
        import eval_controls as EC
    except ImportError as e:
        raise SystemExit(f"could not import eval_controls.py: {e}\n  it must be in the same folder: {HERE}")

    for need in ("JOBS", "stage", "configure", "copy_indices", "E"):
        if not hasattr(EC, need):
            raise SystemExit(f"eval_controls.py has no {need}. It seems to be a different version.")

    out = os.path.join(os.path.dirname(EC.E.MODEL_DIR.rstrip("/\\")), "eval_ctrl_d")
    EC.JOBS = JOBS_D
    EC.CTRL_OUT = out                      # the original version reads this inside configure

    print("=" * 78)
    print(f"D·D29 per-head evaluation  ->  {out}")
    print("  (separate from eval_ctrl/. Detailed reason at the top of this file)")
    print("=" * 78)

    staged = EC.stage()
    if not staged:
        raise SystemExit(
            "There is not a single checkpoint.\n"
            "  D   : model_ctrl/D/resnet_N39_s0.pth   (python run_controls.py D)\n"
            "  D29 : model_ctrl/D29/resnet_N29_s0.pth (python d29_train.py)")

    # eval_controls' converged_rec warning is **meaningless here** — that warning asks to align the convergence-rate points
    # (S34·R5) with the same definition as main training, whereas D·D29 are not convergence-rate points and
    # only per-pair accuracy is used. People sometimes abort by mistyping y, so it is answered automatically.
    import builtins
    real_input = builtins.input

    def auto_yes(prompt=""):
        print(prompt + "y   [auto answer by d29_eval — irrelevant, D·D29 are not convergence-rate points]")
        return "y"

    builtins.input = auto_yes
    try:
        # some versions of configure take the output folder as an argument, others read CTRL_OUT
        if "out_dir" in inspect.signature(EC.configure).parameters:
            EC.configure(out)
        else:
            EC.configure()
    finally:
        builtins.input = real_input
    if EC.E.OUT_DIR != out:
        EC.E.OUT_DIR = out                 # pin the output here whatever the version
    os.makedirs(out, exist_ok=True)

    EC.copy_indices()
    EC.E.main()
    return out


def block_acc(rows, lo, hi):
    """Mean head accuracy over pairs lo..hi (1-based, inclusive)."""
    h0, h1 = 2 * (lo - 1), 2 * hi - 1
    v = [a for h, a in rows if h0 <= h <= h1]
    return float(np.mean(v)) if v else None


def table(out):
    p = os.path.join(out, "eval_heads.csv")
    if not os.path.exists(p):
        print(f"\n{p} does not exist. Run the evaluation first.")
        return

    got = defaultdict(list)
    cols = None
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if cols is None:
            cols = r.keys()
            hcol = next((c for c in cols if c in ("head", "head_idx", "head_index")), None)
            acol = next((c for c in cols if "acc" in c), None)
            if not hcol or not acol:
                print(f"\ncannot recognize the columns of eval_heads.csv: {list(cols)}")
                return
        if r.get("eval_mode", "sum") != "sum":
            continue
        got[(r["arch"], int(r["n_pairs"]), int(r["seed"]))].append(
            (int(r[hcol]), float(r[acol])))

    print("\n" + "=" * 78)
    print("Pair matching table (ResNet seed 0, test 2000 traces, %)")
    print("=" * 78)
    hdr = f"{'':<22}" + "".join(f"{b[0]:>11}" for b in BLOCKS) + f"{'all heads':>12}"
    print(hdr)
    print("-" * len(hdr))

    def ref_row(lab, ref):
        line = f"{lab:<22}"
        for name, _, _ in BLOCKS:
            v = ref.get(name)
            line += f"{(f'{v:.2f}' if v is not None else '—'):>11}"
        print(line + f"{ref['all']:>12.2f}")

    def new_row(exp, arch, n):
        rows = got.get((arch, n, 0))
        line = f"{f'N{n}, long sched ({exp})':<22}"
        if not rows:
            print(line + "".join(f"{'?':>11}" for _ in BLOCKS) + f"{'?':>12}")
            return
        for name, lo, hi in BLOCKS:
            v = None if n < hi else block_acc(rows, lo, min(hi, n))
            line += f"{(f'{v:.2f}' if v is not None else '—'):>11}"
        print(line + f"{float(np.mean([a for _, a in rows])):>12.2f}")

    # print in the same order as the paper chat table
    ref_row("N29, 100 epochs", REF[("N29", "100 epochs")])
    new_row("D29", "resnet", 29)
    ref_row("N39, 100 epochs", REF[("N39", "100 epochs")])
    new_row("D", "resnet", 39)

    # compute the explained share in the pair 1~29 column — it only makes sense here
    d = got.get(("resnet", 39, 0))
    if d:
        d129 = block_acc(d, 1, 29)
        n29_129 = REF[("N29", "100 epochs")]["pair 1-29"]
        n39_129 = REF[("N39", "100 epochs")]["pair 1-29"]
        gap0 = n29_129 - n39_129
        res = n29_129 - d129
        print(f"\nIn the pair 1-29 column (N29 100-epoch reference {n29_129:.2f})")
        print(f"  original gap    {n29_129:.2f} - {n39_129:.2f} = {gap0:.2f} %p")
        print(f"  residual after D  {n29_129:.2f} - {d129:.2f} = {res:.2f} %p")
        print(f"  share explained by schedule  {gap0 - res:.2f} / {gap0:.2f} = "
              f"**{(gap0 - res) / gap0 * 100:.1f}%**   (compare with 54% on all heads)")
        # ---- comparison with both on the long schedule. The only number without a protocol mismatch ----
        d29 = got.get(("resnet", 29, 0))
        if d29:
            x129 = block_acc(d29, 1, 29)
            g_long = x129 - d129
            print(f"\n  ** long schedule vs long schedule (protocol matched) **")
            print(f"     100 epochs, both     {n29_129:.2f} - {n39_129:.2f} = {gap0:.2f} %p")
            print(f"     long schedule, both  {x129:.2f} - {d129:.2f} = {g_long:.2f} %p")
            print(f"     -> amount closed by the schedule {gap0 - g_long:.2f} = {(gap0 - g_long)/gap0*100:.1f}%")
            print(f"     -> remaining N effect   {g_long:.2f} %p "
                  f"({g_long/gap0*100:.1f}%) — not explained by the schedule")
            print(f"     ※ 54%/55.9% were N29@100 epochs vs N39@long schedule, so the protocols")
            print(f"       were mismatched. The numbers above are matched.")

        d3039 = block_acc(d, 30, 39)
        if d3039 is not None:
            diff = d3039 - d129
            print(f"\n  block uniformity of D: pair 1-29 {d129:.2f}  vs  pair 30-39 {d3039:.2f}  "
                  f"({diff:+.2f} %p)")
            if abs(diff) < 1.5:
                print("  -> uniform. The schedule lifted all pairs equally. 54% stands")
            elif diff > 0:
                print("  -> the starved new pairs recovered more. Existing pairs rose less -> explained share below 54%")
            else:
                print("  -> existing pairs recovered more -> explained share above 54%")
    print("=" * 78)


def main():
    if "--table" in sys.argv:
        out = os.path.join(os.path.dirname(
            __import__("eval_controls").E.MODEL_DIR.rstrip("/\\")), "eval_ctrl_d")
        table(out)
        return
    table(run_eval())


if __name__ == "__main__":
    main()
