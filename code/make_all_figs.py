#!/usr/bin/env python3
r"""
All TIFS figures in one go — **runs without arguments**

    python make_all_figs.py

Each script finds its own paths. If the location differs, edit just the one `DATA_ROOT` line in `figstyle.py`,
or prefix an environment variable:

    TIFS_ROOT=/some/where/tifs python make_all_figs.py

Output
---------
    figs/    11 paper figures (EPS + PNG for checking)
    ge_eps/  16 GE panels + 1 legend (EPS + PNG)

TVLA (`kyber_tvla_profile.eps`) reuses the existing figure as decided, so it is not made here.

If one fails, it **does not stop**: it keeps running the rest and at the end reports together what failed.
Usually one input file for that figure is missing, and the message states where it searched.

Selecting
------
    python make_all_figs.py ge qstar      # only those whose name contains ge / qstar
    python make_all_figs.py --list        # just show the list
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (display name, script, figures that script produces)
JOBS = [
    ("nicv",      "fig_nicv_pairs.py",      ["fig_nicv_pairs"]),
    ("convrate",  "fig_convrate.py",        ["fig_convrate_vs_N"]),
    ("acc+probe", "fig_acc_probe.py",       ["fig_acc_vs_N", "fig_probe_info"]),
    ("qstar",     "fig_qstar.py",           ["fig_qstar_vs_N", "fig_qgrid_coarse_vs_fine"]),
    ("ge",        "fig_ge.py",              ["fig_ge_vs_Q"]),
    ("ge 16-panel", "make_ge_eps.py",         ["ge_eps/ (17)"]),
    ("controls",  "fig_controls_curves.py", ["fig_controls_curves"]),
    ("ties",      "fig_rank_ties.py",       ["fig_rank_ties"]),
    ("extra",     "fig_extra.py",           ["fig_gradnorm", "fig_shift_probe"]),
]


def run(script):
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       cwd=HERE, capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or ""), time.time() - t0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        for name, script, outs in JOBS:
            print(f"  {name:<10} {script:<24} -> {', '.join(outs)}")
        return

    jobs = [j for j in JOBS
            if not args or any(a.lower() in (j[0] + " " + j[1]).lower() for a in args)]
    if not jobs:
        raise SystemExit(f"Nothing matches the selected names. List: python {os.path.basename(__file__)} --list")

    # First check that the scripts themselves are in the same folder. If missing, it is not an input-file problem
    # but incompletely received code, and the Python error from run() below cannot tell you that
    missing = [(n, sc) for n, sc, _ in jobs if not os.path.isfile(os.path.join(HERE, sc))]
    if missing:
        print(f"Missing figure scripts in this folder: {len(missing)}:\n   {HERE}\n")
        for n, sc in missing:
            print(f"   missing  {sc}")
        have = [f for f in sorted(os.listdir(HERE))
                if f.startswith(("fig_", "make_", "figstyle"))]
        print(f"\n   present: {', '.join(have) if have else 'none'}")
        print("\n   -> All figure code must be placed in this folder. figstyle.py is also required"
              "\n      (figstyle.py handles path finding, so if only it is missing, everything fails).")
        raise SystemExit(2)

    if not os.path.isfile(os.path.join(HERE, "figstyle.py")):
        raise SystemExit(f"figstyle.py is missing: {HERE}\n"
                         "  All the other scripts import it. Please include it.")

    print(f"Making {len(jobs)} figure groups. Working folder {HERE}\n")
    ok, bad = [], []
    for name, script, outs in jobs:
        print(f"── {name}  ({script})")
        code, log, dt = run(script)
        body = "\n".join("   " + l for l in log.rstrip().splitlines())
        if code == 0:
            print(body)
            print(f"   ✓ {dt:.1f}s\n")
            ok += outs
        else:
            print(body)
            print(f"   ✗ failed (code {code})\n")
            bad.append((name, script, log.rstrip().splitlines()[-1] if log.strip() else ""))

    print("=" * 66)
    print(f"Succeeded {len(ok)}: {', '.join(ok)}" if ok else "No figure succeeded")
    if bad:
        print(f"\nFailed {len(bad)} — usually one input file is missing:")
        for name, script, last in bad:
            print(f"  {name:<10} {script:<24} {last}")
        print("\n  First check that DATA_ROOT in figstyle.py is correct "
              "(or prefix the command with TIFS_ROOT=/path)")
    print(f"\nEPS files are in the {os.path.join(HERE, 'figs')} and {os.path.join(HERE, 'ge_eps')} folders")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
