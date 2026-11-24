#!/usr/bin/env python3
r"""
d29_qfine — re-evaluate the D·D29 checkpoints on the **fine grid** (Q=1..50 every trace, 20 repeats)

    python d29_qfine.py           # evaluate (a few minutes)
    python d29_qfine.py --table   # only redo the table, no evaluation

All $Q^\star$ in the main text are on the fine grid, so this is needed to write "17.1 -> ?" as a number.
coarse (`d29_eval.py`) uses Q=[1,2,5,10,20,50,100,200,500], so values like 17.1 do not come out.

**It does not modify any existing file.** It switches on fine the same way as `eval_qfine.py`
(puts "fine" into `sys.argv` and imports `eval_nscan`), borrows `eval_controls.stage()` for staging,
and only redirects the output to **`eval_qfine_d/`**.

Why a separate folder — otherwise the figures break
---------------------------------------------------
D is `resnet N39 s0`, D29 is `resnet N29 s0`. **Their `(arch, N, seed)` are identical to names
already present in the main training.** `per_seed()` in `fig_qstar.py` is a dict keyed by `(arch, N, seed)`,
so if D's row goes into `eval_qfine/eval_runs.csv` it **silently overwrites the $Q^\star$ of main-training
N39 s0.** That is why it is separated into `eval_qfine_d/`. No figure reads this folder.

Evaluation indices
------------------
$Q^\star$ is comparable only on **the same 2000 traces** as the main fine evaluation. `eval_qfine/eval_indices.npz`
is copied and used as-is. If it is missing, it is regenerated with the same seed, and a warning is printed in that case.
"""

import csv
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# fine must be switched on **before** importing eval_nscan (same way as eval_qfine.py)
if "eval_nscan" in sys.modules:
    raise SystemExit("eval_nscan is already imported. Run this file on its own.")
_user_args = sys.argv[1:]
sys.argv = [sys.argv[0], "fine"]

import eval_nscan as E                                              # noqa: E402

JOBS_D = [("D", "resnet", 39, [0]), ("D29", "resnet", 29, [0])]

# Main-training fine Q*max (ResNet, converged seeds). Hard-coded for comparison.
MAIN_QMAX = {9: 6.3, 19: 7.3, 29: 8.6, 39: 17.1, 49: 16.8, 69: 16.3, 99: 15.0, 128: 18.1}


def out_dir():
    return E.OUT_DIR.rstrip("/\\") + "_d" if E.OUT_DIR.endswith("_qfine") \
        else os.path.join(os.path.dirname(E.MODEL_DIR.rstrip("/\\")), "eval_qfine_d")


def run():
    if not getattr(E, "FINE", False) or not E.OUT_DIR.endswith("_qfine"):
        raise SystemExit("fine setting is not on — check the FINE block of eval_nscan.py")

    main_qfine, out = E.OUT_DIR, out_dir()
    print("=" * 78)
    print(f"D·D29 fine re-evaluation  Q={E.Q_LIST[0]}..{E.Q_LIST[-1]} every trace, {E.N_REPEAT} repeats")
    print(f"  output {out}   (separate from eval_qfine/ — reason at the top of this file)")
    print("=" * 78)

    import eval_controls as EC
    EC.JOBS = JOBS_D
    staged = EC.stage()
    if not staged:
        raise SystemExit(
            "No checkpoints.\n"
            "  D   : model_ctrl/D/resnet_N39_s0.pth\n"
            "  D29 : model_ctrl/D29/resnet_N29_s0.pth")

    # Take the evaluation indices **from the main fine evaluation**. Q* is comparable only on the same 2000 traces.
    os.makedirs(out, exist_ok=True)
    src = os.path.join(main_qfine, "eval_indices.npz")
    dst = os.path.join(out, "eval_indices.npz")
    if os.path.exists(dst):
        print("  eval indices: reusing existing file")
    elif os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  eval indices: copied from the main fine evaluation ({src})")
    else:
        print(f"  [caution] {src} is missing, so new ones are created with the same seed.\n"
              f"         The same seed gives the same 2000 traces, but to put Q* side by side with the main-text values\n"
              f"         check eval_qfine/eval_indices.npz.")

    E.MODEL_DIR = EC.STAGE
    E.OUT_DIR = out
    E.ARCHS = ["resnet"]
    E.N_LIST = sorted({n for _, _, n, _ in JOBS_D})
    E.SEEDS = [0]
    if hasattr(E, "CONV_FROM"):
        E.CONV_FROM = None
    print(f"  MODEL_DIR={E.MODEL_DIR}")
    print(f"  ARCHS={E.ARCHS}  N_LIST={E.N_LIST}  SEEDS={E.SEEDS}\n")

    # The converged_rec warning is meaningless here — for D·D29 only Q* is used, not convergence-rate points
    import builtins
    real_input = builtins.input
    builtins.input = lambda p="": (print(p + "y   [d29_qfine auto answer — only Q* is used]"), "y")[1]
    try:
        E.main()
    finally:
        builtins.input = real_input
    return out


def check_indices(root):
    """Check that the evaluation indices used by D·D29 are **the same 2000 traces** as the main evaluation.

    In the last d29_eval run, eval_ctrl_d/eval_indices.npz could not be copied and was regenerated
    with the same seed. With the same seed they should be identical, but if they differ, the D·D29 numbers cannot be
    put side by side with the 100-epoch rows. So instead of checking by eye, they are compared here.
    """
    pairs = [("coarse", os.path.join(root, "eval"), os.path.join(root, "eval_ctrl_d")),
             ("fine", os.path.join(root, "eval_qfine"), os.path.join(root, "eval_qfine_d"))]
    print("\nEval index comparison (must be the same 2000 traces to sit side by side with the 100-epoch rows)")
    for lab, a, b in pairs:
        pa = os.path.join(a, "eval_indices.npz")
        pb = os.path.join(b, "eval_indices.npz")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"  {lab:<7} skipped (one side is missing)")
            continue
        try:
            za, zb = np.load(pa), np.load(pb)
            k = "idx" if "idx" in za.files else za.files[0]
            same = np.array_equal(np.asarray(za[k]), np.asarray(zb[k]))
        except Exception as e:                       # the key name may differ between versions
            print(f"  {lab:<7} comparison failed: {e}")
            continue
        print(f"  {lab:<7} {'match ✓' if same else '**mismatch ✗ — do not put the numbers side by side**'}")
        if not same:
            print(f"          delete {pb}, copy {pa}, then run again")


def read(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["eval_mode"] != "sum":
            continue
        g = lambda k: float(r[k]) if r.get(k) not in ("", "-1", None) else None
        out[(r["arch"], int(r["n_pairs"]), int(r["seed"]))] = dict(
            qmax=g("q_star_max"), qmean=g("q_star_mean"),
            rec=int(r["pairs_recovered"]), tot=int(r["n_pairs_eval"]),
            acc=g("acc_signal") or g("acc_sig"))
    return out


def table(out):
    p = os.path.join(out, "eval_runs.csv")
    if not os.path.exists(p):
        print(f"\n{p} does not exist. Run the evaluation first.")
        return
    d = read(p)

    print("\n" + "=" * 78)
    print("fine-grid Q* (ResNet seed 0, Q=1..50 every trace, 20 repeats)")
    print("=" * 78)
    print(f"{'':<24}{'Q*max':>9}{'Q*mean':>9}{'recovered':>10}")
    print("-" * 52)

    rows = []
    for n, lab, key in ((29, "N29, 100 epochs", None), (29, "N29, long sched. (D29)", ("resnet", 29, 0)),
                        (39, "N39, 100 epochs", None), (39, "N39, long sched. (D)", ("resnet", 39, 0))):
        if key is None:
            print(f"{lab:<24}{MAIN_QMAX[n]:>9.1f}{'—':>9}{'—':>10}")
            rows.append((n, "main", MAIN_QMAX[n]))
            continue
        r = d.get(key)
        if not r:
            print(f"{lab:<24}{'?':>9}{'?':>9}{'?':>10}")
            continue
        rec = "{}/{}".format(r["rec"], r["tot"])
        print(f"{lab:<24}{r['qmax']:>9.1f}{r['qmean']:>9.1f}{rec:>10}")
        rows.append((n, "long", r["qmax"]))

    got = {(n, k): v for n, k, v in rows}
    m29, m39 = got.get((29, "main")), got.get((39, "main"))
    l29, l39 = got.get((29, "long")), got.get((39, "long"))

    print("\nSentences for the main text")
    if l39 is not None:
        arrow = "decreases" if l39 < m39 else "increases"
        print(f"  N39 Q*max : {m39:.1f} -> {l39:.1f}  ({l39 - m39:+.1f}, {arrow})")
    if l29 is not None:
        print(f"  N29 Q*max : {m29:.1f} -> {l29:.1f}  ({l29 - m29:+.1f})")
    if l29 is not None and l39 is not None:
        g0, g1 = m39 - m29, l39 - l29
        print(f"\n  trace gap (N39 - N29)")
        print(f"    100 epochs, both     {m39:.1f} - {m29:.1f} = {g0:+.1f}")
        print(f"    long schedule, both  {l39:.1f} - {l29:.1f} = {g1:+.1f}")
        if g1 <= 0:
            print(f"    -> **the gap closed (reversed).** The 'twice' sentence in VI-C and the selection rule in VII-A")
            print(f"       need the condition 'on one seed with a long schedule, the trace gap closes'")
        elif g1 < g0 * 0.5:
            print(f"    -> the gap shrank by {(1 - g1/g0)*100:.0f}%. 'Twice' holds only under the common protocol")
        else:
            print(f"    -> the gap shrank by only {(1 - g1/g0)*100:.0f}%. The 'twice' sentence largely survives")
        print(f"\n  ※ This is a single seed. The condition 'on one seed' must not be dropped.")
    print("=" * 78)


def main():
    root = os.path.dirname(E.MODEL_DIR.rstrip("/\\"))
    if "--table" in _user_args:
        table(out_dir())
        check_indices(root)
        return
    out = run()
    table(out)
    check_indices(root)


if __name__ == "__main__":
    main()
