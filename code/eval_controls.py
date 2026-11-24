r"""
Key recovery evaluation of control experiment checkpoints — unify the convergence verdict under the single paper criterion

Why
---
Among the points of `fig_convrate_vs_N`, **N=34 (S34) and the 5 extra N=128 seeds (R5) had no key recovery run.**
All they have is the training-side escape flag (`escaped`), whereas the convergence verdict in the paper body is
**`converged_rec`** in `eval_runs.csv` (= all min(N,39) pairs recovered at Q<=Q_max).
Two definitions cannot be mixed in one figure, so these two also go through the same evaluation.

What it does
------------
1. Gathers the checkpoints of `model_ctrl/S34` and `model_ctrl/R5` into one folder (`_eval_stage`) via symbolic links
   (norm.npz is the same file for both experiments — run_controls.py copied it from the original)
2. Runs `eval_nscan.py` on that folder. Settings·grid·Q grid·number of repeats are identical to the main evaluation
3. The evaluation indices are **copied** from the main evaluation's `eval_indices.npz` — they are comparable only on the same 2000 traces
   (if missing they are regenerated with the same seed, but copying is certain)

Output goes separately to `eval_ctrl/`. The main evaluation CSVs are not touched.

Evaluation targets
------------------
  MobileNetV2  N=34   seeds 0~9    (S34)   -> 34 signal pairs
  ResNet18     N=128  seeds 10~14  (R5)    -> 39 signal pairs + phantom

Run
---
    python eval_controls.py             # evaluate (resumes if interrupted)
    python eval_controls.py summary     # only re-aggregate
    python eval_controls.py stage       # only create the links and check (no evaluation)

About 1 hour. Place it in the same folder as eval_nscan.py.
"""

import glob
import os
import re
import shutil
import sys

SUMMARY_ONLY = "summary" in sys.argv[1:]
STAGE_ONLY = "stage" in sys.argv[1:]
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a not in ("summary", "stage")]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_nscan as E                                   # noqa: E402

# ==========================================================
CTRL_DIR = r"/home/bahoek/data/kyber/tifs/model_ctrl"
STAGE = os.path.join(CTRL_DIR, "_eval_stage")
CTRL_OUT = os.path.join(os.path.dirname(E.MODEL_DIR.rstrip("/\\")), "eval_ctrl")

# (experiment folder, arch, N, seeds)
JOBS = [("S34", "mobilenet", 34, list(range(10))),
        ("R5",  "resnet",   128, [10, 11, 12, 13, 14])]
# ==========================================================


def stage():
    """Gather the checkpoints of the two experiments into one folder. Copy if linking fails."""
    os.makedirs(STAGE, exist_ok=True)
    found, missing, norm_src = [], [], None
    for exp, arch, n, seeds in JOBS:
        d = os.path.join(CTRL_DIR, exp)
        if not os.path.isdir(d):
            print(f"  [missing] {d}")
            missing += [(exp, arch, n, s) for s in seeds]
            continue
        if norm_src is None and os.path.exists(os.path.join(d, "norm.npz")):
            norm_src = os.path.join(d, "norm.npz")
        for s in seeds:
            name = f"{arch}_N{n}_s{s}.pth"
            src = os.path.join(d, name)
            if not os.path.exists(src):                  # the name may differ, so search once more
                cand = [p for p in glob.glob(os.path.join(d, "*.pth"))
                        if re.search(rf"(^|[_/]){arch}_N{n}_s{s}\.pth$", p)
                        or os.path.basename(p).endswith(f"{arch}_N{n}_s{s}.pth")]
                src = cand[0] if cand else None
            if not src:
                missing.append((exp, arch, n, s))
                continue
            dst = os.path.join(STAGE, name)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)
            found.append((exp, arch, n, s))
    if norm_src and not os.path.exists(os.path.join(STAGE, "norm.npz")):
        shutil.copy2(norm_src, os.path.join(STAGE, "norm.npz"))
    print(f"staging {STAGE}")
    print(f"  {len(found)} checkpoints ready" + (f", {len(missing)} missing" if missing else ""))
    for exp, arch, n, s in missing:
        print(f"    [missing] {exp}: {arch}_N{n}_s{s}.pth")
    if not os.path.exists(os.path.join(STAGE, "norm.npz")):
        raise SystemExit(f"{STAGE}/norm.npz is missing — check model_ctrl/<experiment>/norm.npz")
    return found


def copy_indices():
    """Take the main evaluation's indices as-is. Numbers are comparable only on the same 2000 traces."""
    src = os.path.join(E.OUT_DIR, "eval_indices.npz")
    dst = os.path.join(CTRL_OUT, "eval_indices.npz")
    os.makedirs(CTRL_OUT, exist_ok=True)
    if os.path.exists(dst):
        print("  eval indices: reusing existing file")
    elif os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  eval indices: copied from the main evaluation ({src})")
    else:
        print(f"  [caution] {src} is missing, so new ones are created with the same seed (numbers should match, but checking is recommended)")


def configure():
    """Point eval_nscan's settings at the control evaluation. Q grid·repeats·phantom are kept identical to the main evaluation."""
    E.MODEL_DIR = STAGE
    E.OUT_DIR = CTRL_OUT
    E.ARCHS = sorted({a for _, a, _, _ in JOBS})
    E.N_LIST = sorted({n for _, _, n, _ in JOBS})
    E.SEEDS = sorted({s for _, _, _, ss in JOBS for s in ss})
    if hasattr(E, "CONV_FROM"):
        E.CONV_FROM = None                               # judged by its own results
    print(f"config: MODEL_DIR={E.MODEL_DIR}")
    print(f"      OUT_DIR={E.OUT_DIR}")
    print(f"      ARCHS={E.ARCHS}  N_LIST={E.N_LIST}  SEEDS={E.SEEDS}")
    print(f"      Q={E.Q_LIST}  N_REPEAT={E.N_REPEAT}"
          + (f"  CONV_MODE={E.CONV_MODE}" if hasattr(E, "CONV_MODE") else ""))
    print(f"      grid needed for {len(E.needed_pairs())} pairs")

    # Check whether this version of eval_nscan has the paper-criterion convergence verdict (converged_rec).
    # Without it, the result is only the accuracy-based `converged`, and the definition mismatches the other points in the figure
    if "converged_rec" not in getattr(E, "RUN_FIELDS", []):
        print()
        print("  " + "!" * 84)
        print("  [warning] This eval_nscan.py has no converged_rec (convergence by number of recovered pairs).")
        print("         It is an old version. Running it as-is yields only the accuracy-based `converged`, so")
        print("         the definition mismatches the main-training points — the figure cannot use a single definition.")
        print("         Switch to a version that includes CONV_MODE='recovery' and run again.")
        print("  " + "!" * 84)
        print()
        if input("  Continue anyway? (y/N) ").strip().lower() != "y":
            raise SystemExit("Aborted.")
    print("  Note: checkpoints absent from the ARCHS x N_LIST x SEEDS combinations are skipped as '[missing]'")
    print("        (mobilenet N128, resnet N34, etc. — this is normal)")


def main():
    stage()
    configure()
    if STAGE_ONLY:
        print("\nOnly stage was run. Run again without arguments to evaluate.")
        return
    copy_indices()
    E.main()
    note()


def note():
    print("\n" + "=" * 92)
    print("Where this result is used")
    print("=" * 92)
    print(f"  converged_rec in {CTRL_OUT}/eval_runs.csv is the paper-criterion convergence verdict.")
    print("  The N=34 point of fig_convrate_vs_N (MobileNetV2, 10 seeds) and the 5 extra N=128 seeds (ResNet18)")
    print("  can be plotted **with the same definition** as the 9 main-training points.")
    print("  ResNet N=128 is reported as 10 main-training seeds + 5 seeds here = 15 seeds combined.")
    print()
    print("  Values may differ from the training-side escape flag (escaped in ctrl_summary.csv).")
    print("  That is not an error — escape means 'did accuracy exceed prior + 5%p', convergence means")
    print("  'were all pairs recovered'. Use only the latter in the main text, and note runs where the two disagree in the limitations section.")


if __name__ == "__main__":
    if SUMMARY_ONLY:
        configure()
        E.write_summary()
        E.print_table()
    else:
        main()
