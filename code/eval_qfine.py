"""
High-resolution Q* re-evaluation — a version you just run without arguments

Exactly the same as `python eval_nscan.py fine`. Forgetting the argument makes the main evaluation run again
(= skipping completed parts and only aggregating), so using this file prevents that mistake.

    python eval_qfine.py            # evaluate
    python eval_qfine.py rebuild    # regenerate the CSV from eval_qfine/res_*.npz
    python eval_qfine.py summary    # redo only the aggregation

Just put it in the same folder as eval_nscan.py. Settings are managed in one place only, the `if FINE:` block of eval_nscan.py
(Q = 1..50 every trace, 20 repeats, phantom key recovery off, output OUT_DIR + "_qfine").
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "fine"] + sys.argv[1:]      # must be inserted before import for FINE to be on

import eval_nscan as E

if __name__ == "__main__":
    assert E.FINE and E.OUT_DIR.endswith("_qfine"), \
        "fine setting is not on — check the FINE block of eval_nscan.py"
    print(f"[eval_qfine] Q={E.Q_LIST[0]}..{E.Q_LIST[-1]} every trace, {E.N_REPEAT} repeats, "
          f"phantom off, output {E.OUT_DIR}")
    E.main()
