"""
TIFS — merge all training records into a single CSV (160 main training runs + 31 control experiment runs)

Why
---
Run records are scattered across three places.
  1. Main training          MODEL_DIR/train_summary.csv            (160 runs)
  2. Control verdicts       CTRL_DIR/ctrl_summary.csv              (escape epoch, virtual patience15 stop, etc.)
  3. Control training       CTRL_DIR/<experiment>/train_summary.csv (lr·heads·params etc., same schema as main training)
Column names also differ (main training `val_acc_allheads`·`total_time_min` vs control `best_acc`·`time_min`).
This code aligns the three into one table and outputs a single **all_runs.csv**. Per-epoch gradnorm is also merged into **all_gradnorm.csv**.

Run
---
    python merge_runs.py

If an input file is missing, only that part is skipped and the rest is built. The original CSVs are not touched.

Fixed in v2 (always regenerate with this version before exporting for release)
-----------------------------------------------------------------------------
  * Fixed the error where `phantom_label` for experiment B was recorded as "real" → corrected to "shuffled".
    run_controls.py does not touch the CONFIG value and only shuffles the label array, so train_summary
    still records "real". To someone looking only at the table, B was indistinguishable from a main N128 rerun.
  * Added the `desc` column — what the experiment changed stays in the table. For experiments whose overrides are {}
    (B·S34·R5), without this the setup looks exactly the same as main training.
  * Fill `min_delta` with a constant (the training code does not record it, but it is part of the early stopping rule).
  * Also fill `escaped` for main training rows, and record the rule used for the verdict in `escape_rule`.
    Main training has no per-epoch acc record, so the verdict can only use a final-acc threshold. The paper's convergence verdict is
    not this column but `converged_rec` in eval_runs.csv (based on the number of recovered pairs).

Output (OUT_DIR)
----------------
  all_runs.csv       one row per run. source = main / D / E / E2 / B / S34 / S2 / R5
  all_gradnorm.csv   one row per run x epoch (source column added)
  after merging, prints a summary table + warnings for runs near the threshold
"""

import csv
import glob
import json
import os

# ==========================================================
MODEL_DIR = r"/home/bahoek/data/kyber/tifs/model"        # main training
CTRL_DIR  = r"/home/bahoek/data/kyber/tifs/model_ctrl"   # control experiments
OUT_DIR   = r"/home/bahoek/data/kyber/tifs"              # where all_runs.csv goes
MERGE_GRADNORM = True        # whether to also merge per-epoch gradnorm (False if the file is large)

MIN_DELTA   = 0.3            # early stopping constant the training code does not write to CSV (%p)
ESCAPE_ACC  = 16.6           # main-training escape threshold = prior 11.6%p + 5%p
BORDER      = (13.0, 22.0)   # main-training runs between these are decided by the threshold, so warn on screen
PHANTOM_SHUFFLED = {"B"}     # experiments that shuffled the phantom labels (a fact train_summary does not know)
MAIN_DESC   = "N-scan main training (common protocol: 100 epochs, patience 15, LR 1e-3)"
# ==========================================================

FIELDS = [
    # identity
    "source", "desc", "run_id", "arch", "resnet_variant", "n_pairs", "heads", "seed", "overrides",
    # setup
    "lr", "batch", "max_epochs", "patience", "min_delta", "grad_clip", "loss_reduction",
    "label_type", "phantom_label", "crop_lo", "crop_hi", "leak_pairs", "git_commit",
    # results (names aligned between main training and controls)
    "acc_allheads", "ge_allheads", "epochs_run", "best_epoch",
    # escaped/stalled verdict
    "escaped", "escape_rule", "escape_epoch", "virtual_p15_stop", "acc_at_virtual_stop",
    "stuck_under_p15", "orig_acc_allheads", "orig_epochs_run", "gn_same_until_epoch",
    # cost
    "params_m", "time_min", "best_model_time_min", "vram_max_mb", "vram_avg_mb",
]
GN_FIELDS = ["source", "run_id", "arch", "n_pairs", "seed", "epoch",
             "gn_total", "gn_backbone", "gn_heads_signal", "gn_heads_phantom", "clipped"]

# Column name alignment: main training name -> merged table name
MAIN_MAP = {"val_acc_allheads": "acc_allheads", "total_time_min": "time_min"}
# control ctrl_summary.csv name -> merged table name
CTRL_MAP = {"best_acc": "acc_allheads", "orig_val_acc": "orig_acc_allheads",
            "orig_epochs": "orig_epochs_run", "experiment": "source"}


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick(row, mapping):
    """Rename with mapping and keep only those in FIELDS."""
    out = {}
    for k, v in row.items():
        k2 = mapping.get(k, k)
        if k2 in FIELDS:
            out[k2] = v
    return out


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    rows = []

    # ---------- 1. main training ----------
    main_rows = read(os.path.join(MODEL_DIR, "train_summary.csv"))
    for r in main_rows:
        d = pick(r, MAIN_MAP)
        d["source"] = "main"
        d["desc"] = MAIN_DESC
        # main training has no per-epoch acc record → verdict only by final-acc threshold. Record the rule in a column.
        a = fnum(d.get("acc_allheads"))
        if a is not None:
            d["escaped"] = "1" if a > ESCAPE_ACC else "0"
            d["escape_rule"] = f"acc_allheads > {ESCAPE_ACC}"
        rows.append(d)
    print(f"main training {len(main_rows):>4} runs  ({MODEL_DIR}/train_summary.csv)")

    # ---------- 2. control experiments: merge verdicts + training records ----------
    ctrl = read(os.path.join(CTRL_DIR, "ctrl_summary.csv"))
    # index each experiment's train_summary.csv by run_id (to pull setup columns such as lr·heads·params)
    detail = {}
    for p in sorted(glob.glob(os.path.join(CTRL_DIR, "*", "train_summary.csv"))):
        exp = os.path.basename(os.path.dirname(p))
        for r in read(p):
            rid = r.get("run_id", "")
            detail[rid] = r
            detail.setdefault(f"{exp}_{rid}", r)       # also covers run_ids prefixed with the experiment name
    n_det = len(set(id(v) for v in detail.values()))
    for r in ctrl:
        d = pick(r, CTRL_MAP)
        rid = r.get("run_id", "")
        det = detail.get(rid)
        if det is None:                                # strip the prefix and look again
            exp = r.get("experiment", "")
            det = detail.get(rid[len(exp) + 1:] if rid.startswith(exp + "_") else rid)
        if det:
            for k, v in pick(det, MAIN_MAP).items():
                if d.get(k, "") == "":                 # verdict values take priority; only fill empty cells
                    d[k] = v
        # actually apply overrides to the setup columns — otherwise the original values (100 epochs·patience 15) stay in
        # the table, producing contradictions such as "ran for 200 epochs but max_epochs is 100"
        try:
            ov = json.loads(r.get("overrides") or "{}")
        except Exception:
            ov = {}
        for k, col in (("EPOCHS", "max_epochs"), ("PATIENCE", "patience"), ("LR", "lr"),
                       ("MIN_DELTA", "min_delta"), ("BATCH", "batch")):
            if k in ov:
                d[col] = ov[k]
        # changes not expressible as overrides — label shuffling touches the array, not CONFIG, so
        # train_summary keeps "real". Correct it so B is distinguishable from a main N128 rerun in the table
        if d.get("source") in PHANTOM_SHUFFLED:
            d["phantom_label"] = "shuffled"
        d["escape_rule"] = "run_controls.py verdict (per-epoch acc trajectory)"
        rows.append(d)
    print(f"controls      {len(ctrl):>4} runs  ({CTRL_DIR}/ctrl_summary.csv"
          + (f", {n_det} training records joined)" if n_det else ", no per-experiment train_summary.csv)"))

    # ---------- fill common constants ----------
    for d in rows:
        if d.get("min_delta", "") == "":
            d["min_delta"] = MIN_DELTA

    # ---------- save ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "all_runs.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for d in rows:
            w.writerow({k: d.get(k, "") for k in FIELDS})
    print(f"\nsaved: {out}  ({len(rows)} rows, {len(FIELDS)} columns)")

    # ---------- 3. gradnorm ----------
    if MERGE_GRADNORM:
        gp = os.path.join(OUT_DIR, "all_gradnorm.csv")
        n = 0
        with open(gp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=GN_FIELDS)
            w.writeheader()
            srcs = [("main", os.path.join(MODEL_DIR, "train_gradnorm.csv"))]
            srcs += [(os.path.basename(os.path.dirname(p)), p)
                     for p in sorted(glob.glob(os.path.join(CTRL_DIR, "*", "train_gradnorm.csv")))]
            for src, p in srcs:
                k = 0
                for r in read(p):
                    r["source"] = src
                    w.writerow({c: r.get(c, "") for c in GN_FIELDS})
                    k += 1
                if k:
                    print(f"  gradnorm {src:<6} {k:>7} rows")
                n += k
        print(f"saved: {gp}  ({n} rows)")

    summary(rows)


def summary(rows):
    print("\n" + "=" * 100)
    print("Merged table summary — by source (acc = validation-set mean over all heads, escaped runs only)")
    print("=" * 100)
    print(f"{'source':<8}{'arch':<11}{'N':>5}{'run':>5}{'escaped':>8}{'acc(esc)':>12}"
          f"{'epochs':>7}{'esc_ep':>8}{'time(min)':>10}")
    print("-" * 100)

    def mean(vs):
        vs = [v for v in vs if v is not None]
        return f"{sum(vs)/len(vs):.2f}" if vs else "-"

    key = lambda r: (r.get("source", ""), r.get("arch", ""), int(r.get("n_pairs") or 0))
    for k in sorted({key(r) for r in rows}, key=lambda x: (x[0] != "main", x[0], x[1], x[2])):
        rs = [r for r in rows if key(r) == k]
        e = [r for r in rs if r.get("escaped") == "1"]
        print(f"{k[0]:<8}{k[1]:<11}{k[2]:>5}{len(rs):>5}{len(e):>5}/{len(rs):<2}"
              f"{mean([fnum(r.get('acc_allheads')) for r in e]):>12}"
              f"{mean([fnum(r.get('epochs_run')) for r in rs]):>7}"
              f"{mean([fnum(r.get('escape_epoch')) for r in e]):>8}"
              f"{mean([fnum(r.get('time_min')) for r in rs]):>10}")
    print("=" * 100)

    tot = sum(fnum(r.get("time_min")) or 0 for r in rows)
    print(f"total training time {tot/60:.1f} hours ({tot/60/24:.2f} days)")
    print("  setup columns (lr·max_epochs·patience) are the actual values with overrides applied. Changed items also remain in the overrides column")
    print("  B has phantom_label=shuffled — a change not expressible as overrides, so it is marked separately here")
    print("  the paper's convergence verdict is not escaped in this table but converged_rec (number of recovered pairs) in eval_runs.csv")

    # main-training runs whose verdict is decided by the threshold — inspect by eye before citing numbers
    lo, hi = BORDER
    border = [r for r in rows
              if r.get("source") == "main" and (fnum(r.get("acc_allheads")) or 0) > lo
              and (fnum(r.get("acc_allheads")) or 0) < hi]
    if border:
        print(f"\n[check needed] main-training runs with acc in {lo}~{hi}% — the {ESCAPE_ACC}% threshold decides the verdict")
        for r in sorted(border, key=lambda r: fnum(r.get("acc_allheads")) or 0):
            print(f"  {r['arch']:<10} N{r['n_pairs']:<4} s{r['seed']:<2} "
                  f"acc={fnum(r['acc_allheads']):6.2f}  epochs={r['epochs_run']:>3}  "
                  f"→ {'escaped' if r.get('escaped') == '1' else 'stalled'}")
        print("  a run with epochs close to 100 and low acc may have 'run out of budget while escaping'")


if __name__ == "__main__":
    main()
