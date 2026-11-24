"""
TIFS control experiments (priority 1 and 2) — E, E′, D
Place in the same folder as train_nscan.py and run. The training code calls train_nscan.run_single as is;
the only thing changed is one or two hyperparameters in EXPERIMENTS below (the rest of the protocol is identical).

    python run_controls.py            # all of them, in the order E -> E2 -> D
    python run_controls.py E          # E only
    python run_controls.py E2 D       # pick a subset

Experiments
-----------
  D   200 epochs + patience 50  ResNet N39 s0          "is the 29->39 cliff capacity or budget"  10h
  E   patience 50          MobileNet N9 s0 s1 s3  "is the stall structural or patience truncation"  one day
  E2  LR 3e-4              MobileNet N9 s0 s1     "is the stall structural or LR"            8h
  B   phantom label shuffle     ResNet N128 x5         "is the N128 drop accumulation of residual gradient or dilution of the mean loss"  20h
  S34 refinement N=34          MobileNet x10          "is the convergence-rate peak at 29, 34 or 39 (=L)"  42h
  S2  refinement N=24,44        MobileNet x10          slope on both sides of the peak (decide after seeing S34)     78h
  R5  add seeds 10~14       ResNet N128            "is the 2/10 stall luck"               one day

  Why D also gets patience 50: in the main experiment the best epoch for N39 was the last epoch, so patience
  never triggered. If late improvement slows down over 200 epochs, patience 15 would cut it and D would separate nothing.
  The shuffle in B is the same method as PHANTOM_LABEL="shuffled" in train_nscan (heads 2*39 and above, rng 12345).

What is used for the verdict (all of it comes from a single run)
----------------------------------------------------------------
  * Read the per-epoch val acc from the log,
      - escape epoch: the first epoch at which acc exceeds ESCAPE_ACC (prior 11.6% + 5%p = 16.6%)
      - virtual patience-15 stop epoch: at which epoch the same trajectory would have been cut under the original rule (MIN_DELTA, patience 15)
    -> In E, "did a run that would have been cut under patience 15 escape afterwards" is decided directly within the same run.
       Even if the trajectory differs from the original run because of hardware or non-determinism, the verdict is not affected.
  * Compare per-epoch gn_total with the original train_gradnorm.csv -> up to which epoch the trajectory matches the original run
    (if deterministic on the same machine (Spark), it should match up to the original stop epoch. On the 5080 it may differ from the start — the verdict above is still valid)

Safeguards
----------
  * Saves go only to CTRL_DIR/<experiment>/. torch.save / np.save / np.savez are wrapped so that
    any attempt to write into the original MODEL_DIR stops with an exception (prevents overwriting the original checkpoints). The label cache (CACHE_DIR) is allowed.
  * The z-score statistics are read as is from the original training's norm.npz (not recomputed, which also saves memory).

Output (CTRL_DIR)
-----------------
  ctrl_summary.csv              1 row per run (verdict summary)
  <experiment>/{arch}_N{n}_s{seed}.pth, heads_*.npz   checkpoints (key recovery possible with eval_nscan)
  <experiment>/train_gradnorm.csv      per-epoch gn
  <experiment>/log_<run_id>.txt        raw console log
"""

import csv
import glob
import io
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import torch

# ==========================================================
HERE      = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = HERE
CTRL_DIR  = r"/home/bahoek/data/kyber/tifs/model_ctrl"
ORIG_MODEL_DIR = None        # if None, train_nscan.SAVE_DIR (the original training results folder, used read-only)

LEAK_PAIRS = 39              # number of signal pairs. Heads 0..2*LEAK_PAIRS-1 are signal, those above are phantom

# One experiment per hypothesis (project doc "Additional experiments — reorganized as hypothesis-testing experiments", 10/30)
EXPERIMENTS = {
    "D":   dict(desc="200 epochs + patience 50 — is the 29->39 cliff capacity or budget",
                arch="resnet",    ns=[39],  seeds=[0],
                over=dict(EPOCHS=200, PATIENCE=50)),
    "E":   dict(desc="patience 50 — is the MobileNet stall structural or truncation",
                arch="mobilenet", ns=[9],   seeds=[0, 1, 3],
                over=dict(PATIENCE=50)),
    "E2":  dict(desc="LR 3e-4 — is the MobileNet stall structural or LR",
                arch="mobilenet", ns=[9],   seeds=[0, 1],
                over=dict(LR=3e-4)),
    "B":   dict(desc="N128 phantom label shuffle — accumulation of residual gradient or dilution of the mean loss",
                arch="resnet",    ns=[128], seeds=list(range(10)),  # keep 10 seeds (reason below)
                # Reducing to 5 saves 20h, but N128 originally stalled in 2/10, so if even 1 of 5 stalls
                # we could no longer read "does the shuffle bend less". A whole hypothesis would be lost, so keep 10
                over={}, shuffle_phantom=True),
    "S34": dict(desc="refinement N=34 — is the convergence-rate peak at 29 / 34 / 39 (the key point)",
                arch="mobilenet", ns=[34],     seeds=list(range(10)),
                over={}),
    "S2":  dict(desc="refinement N=24,44 — slope on both sides of the peak (decide whether to run after seeing S34; excluded from the default run)",
                arch="mobilenet", ns=[24, 44], seeds=list(range(10)),
                over={}),
    "R5":  dict(desc="ResNet N128 5 extra seeds — is the 2/10 stall luck",
                arch="resnet",    ns=[128], seeds=[10, 11, 12, 13, 14],
                over={}),
}
# Running without arguments runs only ORDER. S2 is left out because it is decided after seeing the S34 results (`python run_controls.py S2`)
ORDER = ["D", "E", "E2", "B", "S34", "R5"]
SHUFFLE_SEED = 12345         # same value as PHANTOM_LABEL="shuffled" in train_nscan

ESCAPE_ACC = 11.6 + 5.0      # escape criterion (16.6% proposed by the paper chat). Fixed before interpreting results
# ==========================================================

SUMMARY_FIELDS = ["experiment", "desc", "arch", "n_pairs", "seed", "overrides",
                  "epochs_run", "best_acc", "best_epoch", "escape_epoch",
                  "virtual_p15_stop", "acc_at_virtual_stop", "stuck_under_p15", "escaped",
                  "orig_val_acc", "orig_epochs", "gn_same_until_epoch", "time_min", "run_id"]


# ---------------- console tee (to read per-epoch acc from the log) ----------------
class Tee(io.TextIOBase):
    def __init__(self, stream):
        self.stream, self.buf = stream, None
    def write(self, s):
        self.stream.write(s)
        if self.buf is not None:
            self.buf.append(s)
        return len(s)
    def flush(self):
        self.stream.flush()

TEE = Tee(sys.stdout)
sys.stdout = TEE

ACC_RE = re.compile(r"\|\s*acc\s+([0-9.]+)%")


def parse_acc(text):
    return [float(m.group(1)) for line in text.splitlines()
            if (m := ACC_RE.search(line)) and " GE " in line]


def virtual_stop(accs, patience, min_delta):
    """Stop epoch (1-based) when the original rule (counter resets only on a rise of more than MIN_DELTA) is applied to the same trajectory. None if never cut."""
    best, cnt = -1e9, 0
    for e, a in enumerate(accs, 1):
        if a > best + min_delta:
            best, cnt = a, 0
        else:
            cnt += 1
            if cnt >= patience:
                return e
    return None


# ---------------- forbid writes to the original folder ----------------
def guard_saves(orig_dir):
    orig = os.path.realpath(orig_dir)
    ctrl = os.path.realpath(CTRL_DIR)

    def check(path):
        if isinstance(path, (str, os.PathLike)):
            p = os.path.realpath(os.fspath(path))
            if p.startswith(orig + os.sep) and not p.startswith(ctrl + os.sep):
                raise RuntimeError(f"Attempt to write into the original training folder — aborting: {p}")

    def wrap(fn, pos, kw):
        """pos: position of the path argument (1 for torch.save(obj, f), 0 for np.save(file, arr))"""
        def inner(*a, **k):
            check(a[pos] if len(a) > pos else k.get(kw))
            return fn(*a, **k)
        return inner
    torch.save = wrap(torch.save, 1, "f")
    np.save = wrap(np.save, 0, "file")
    np.savez = wrap(np.savez, 0, "file")
    np.savez_compressed = wrap(np.savez_compressed, 0, "file")


def repair(ctrl_dir):
    """Clean up incomplete runs so that "resume" runs exactly what is left.

    A run finishes in the order (1) checkpoint saved -> (2) train_summary row -> (3) ctrl_summary row.
    So **a finished run = the .pth opens + it has a row in ctrl_summary**. If either is missing,
    it was interrupted, so the .pth is deleted (otherwise the script would think it finished and skip it).
    A .pth half-written by a sudden power loss (file exists but does not open) is also caught here."""
    sp = os.path.join(ctrl_dir, "ctrl_summary.csv")
    rows, fields = [], SUMMARY_FIELDS
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames or SUMMARY_FIELDS
            rows = list(rd)
    done = {r.get("run_id", "") for r in rows}
    bad = []
    for ck in sorted(glob.glob(os.path.join(ctrl_dir, "*", "*.pth"))):
        exp = os.path.basename(os.path.dirname(ck))
        rid = f"{exp}_{os.path.basename(ck)[:-4]}"
        why = None
        if rid not in done:
            why = "no row in ctrl_summary (interrupted before the run finished)"
        else:
            try:
                torch.load(ck, map_location="cpu", weights_only=False)
            except Exception as e:
                why = f"checkpoint does not open ({type(e).__name__})"
        if why:
            os.remove(ck)
            bad.append((rid, why))
    if bad:
        keep = [r for r in rows if r.get("run_id", "") not in {b[0] for b in bad}]
        if len(keep) != len(rows):
            with open(sp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in keep:
                    w.writerow({k: r.get(k, "") for k in fields})
        print("[repair] cleaned up unfinished runs — they will run again this time")
        for rid, why in bad:
            print(f"    {rid}: {why}")
    return len(rows) - len(bad)


def prune_gn(gn_path, sub, nm, fields):
    """Remove from train_gradnorm.csv the rows of run_ids that have no checkpoint.

    Per-epoch gn is appended during training, so an interrupted run leaves leftover rows for "epochs 1~40".
    If that seed is rerun, rows accumulate again from epoch 1, giving two sets of rows with the same run_id.
    The checkpoint (.pth) is saved only when a run finishes, so a run_id without one = an unfinished run."""
    if not os.path.exists(gn_path):
        return
    rows = list(csv.DictReader(open(gn_path, encoding="utf-8")))
    keep, dropped = [], set()
    for r in rows:
        rid = r.get("run_id", "")
        ck = rid[len(nm) + 1:] if rid.startswith(nm + "_") else rid
        if os.path.exists(os.path.join(sub, ck + ".pth")):
            keep.append(r)
        else:
            dropped.add(rid)
    if not dropped:
        return
    with open(gn_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  [cleanup] {len(dropped)} interrupted runs: removed {len(rows)-len(keep)} gradnorm rows "
          f"({', '.join(sorted(dropped))})")


# ---------------- data (same as train_nscan.main, z-score from the original norm.npz) ----------------
def load_data(T, orig_dir):
    crop_lo, crop_hi = T.CROP_LO, T.CROP_HI

    def load(name):
        d = np.load(os.path.join(T.SPLIT_DIR, f"kyber_{name}_crop.npz"))
        return (d["traces"][:, crop_lo:crop_hi].astype(np.float32, copy=False),
                np.asarray(d["ciphertexts"], np.uint8),
                np.asarray(d["labels"], np.int64))

    Xtr, ct_tr, lab_tr = load("train")
    Xva, ct_va, _ = load("valid")
    sk = lab_tr[0]
    h_max = T.heads_of(max(T.N_LIST))            # match the original cache file name
    Ytr = T.make_labels(ct_tr, sk, "train", h_max)
    Yva = T.make_labels(ct_va, sk, "valid", h_max)

    nz = np.load(os.path.join(orig_dir, "norm.npz"))
    assert [int(v) for v in nz["crop"]] == [crop_lo, crop_hi], "norm.npz crop differs from the train_nscan setting"
    mu, sd = nz["mu"].astype(np.float32), nz["sd"].astype(np.float32)
    for X in (Xtr, Xva):                         # in-place normalization (no copy)
        X -= mu
        X /= sd
    print(f"train {Xtr.shape}  valid {Xva.shape}  (z-score: original norm.npz)")
    return Xtr, Ytr, Xva, Yva


def orig_info(orig_dir, arch, n, seed):
    acc, ep, gn = None, None, {}
    p = os.path.join(orig_dir, "train_summary.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["arch"] == arch and int(r["n_pairs"]) == n and int(r["seed"]) == seed:
                acc, ep = float(r["val_acc_allheads"]), int(r["epochs_run"])
    p = os.path.join(orig_dir, "train_gradnorm.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["arch"] == arch and int(r["n_pairs"]) == n and int(r["seed"]) == seed:
                gn[int(r["epoch"])] = float(r["gn_total"])
    return acc, ep, gn


def gn_same_until(gn_orig, gn_new, rtol=1e-3):
    last = 0
    for e in sorted(gn_new):
        if e not in gn_orig:
            break
        a, b = gn_orig[e], gn_new[e]
        if abs(a - b) > rtol * max(abs(a), 1e-12):
            break
        last = e
    return last


# ---------------- run ----------------
def main():
    names = sys.argv[1:] or ORDER
    for nm in names:
        if nm not in EXPERIMENTS:
            raise SystemExit(f"Unknown experiment: {nm} (available: {list(EXPERIMENTS)})")

    sys.path.insert(0, TRAIN_DIR)
    import train_nscan as T
    for need in ("run_single", "make_labels", "heads_of", "GN_FIELDS"):
        if not hasattr(T, need):
            raise SystemExit(f"train_nscan.py has no {need} — please report this")
    orig_dir = ORIG_MODEL_DIR or T.SAVE_DIR
    os.makedirs(CTRL_DIR, exist_ok=True)
    if os.path.realpath(CTRL_DIR) == os.path.realpath(orig_dir):
        raise SystemExit("CTRL_DIR is the same as the original training folder — use a different folder")
    guard_saves(orig_dir)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev} | experiments {names} | original results {orig_dir} (read-only) | saving to {CTRL_DIR}")
    base = {k: getattr(T, k) for k in ("SAVE_DIR", "LR", "EPOCHS", "PATIENCE", "MIN_DELTA",
                                       "SUMMARY_CSV", "GN_CSV") if hasattr(T, k)}
    n_done = repair(CTRL_DIR)
    todo = {}
    for nm in names:
        ex = EXPERIMENTS[nm]
        sub = os.path.join(CTRL_DIR, nm)
        left = [(n, s_) for n in ex["ns"] for s_ in ex["seeds"]
                if not os.path.exists(os.path.join(sub, f"{ex['arch']}_N{n}_s{s_}.pth"))]
        todo[nm] = left
    tot = sum(len(v) for v in todo.values())
    print(f"\n{n_done} runs finished | {tot} runs remaining")
    for nm in names:
        ex = EXPERIMENTS[nm]
        k = len(ex["ns"]) * len(ex["seeds"])
        print(f"  {nm:<4} {k-len(todo[nm]):>2}/{k:<2} done"
              + (f"  remaining: {', '.join(f'N{n} s{s_}' for n, s_ in todo[nm])}" if todo[nm] else "  (skipped)"))
    if tot == 0:
        print("\nNothing to run. Everything is finished.")
        print_summary(os.path.join(CTRL_DIR, "ctrl_summary.csv"))
        return

    Xtr, Ytr_all, Xva, Yva_all = load_data(T, orig_dir)
    sum_path = os.path.join(CTRL_DIR, "ctrl_summary.csv")

    for nm in names:
        ex = EXPERIMENTS[nm]
        sub = os.path.join(CTRL_DIR, nm)
        os.makedirs(sub, exist_ok=True)
        for k, v in base.items():                       # each experiment starts from the original values
            setattr(T, k, v)
        T.SAVE_DIR = sub
        if hasattr(T, "SUMMARY_CSV"):
            T.SUMMARY_CSV = os.path.join(sub, "train_summary.csv")
        if hasattr(T, "GN_CSV"):
            T.GN_CSV = os.path.join(sub, "train_gradnorm.csv")
        for k, v in ex["over"].items():
            if not hasattr(T, k):
                raise SystemExit(f"train_nscan.py has no {k} attribute")
            setattr(T, k, v)
        print(f"\n===== {nm}: {ex['desc']}\n      {ex['arch']} N{ex['ns']} seeds {ex['seeds']} | "
              f"LR {T.LR} EPOCHS {T.EPOCHS} PATIENCE {T.PATIENCE}"
              f"{' | phantom label shuffle' if ex.get('shuffle_phantom') else ''} =====")
        if not os.path.exists(os.path.join(sub, "norm.npz")):   # so eval_nscan can read it directly
            shutil.copy(os.path.join(orig_dir, "norm.npz"), os.path.join(sub, "norm.npz"))

        gn_path = os.path.join(sub, "train_gradnorm.csv")
        prune_gn(gn_path, sub, nm, T.GN_FIELDS)     # remove leftover rows of interrupted runs
        gn_new_file = not os.path.exists(gn_path)
        gn_f = open(gn_path, "a", newline="", encoding="utf-8")
        gn_w = csv.DictWriter(gn_f, fieldnames=T.GN_FIELDS)
        if gn_new_file:
            gn_w.writeheader()

        arch = ex["arch"]
        jobs = todo[nm]                 # only what actually remains after repair
        for n, seed in jobs:
            h = T.heads_of(n)
            ck = os.path.join(sub, f"{arch}_N{n}_s{seed}.pth")
            if os.path.exists(ck):
                print(f"  [skipped] {os.path.basename(ck)} already exists")
                continue
            run_id = f"{nm}_{arch}_N{n}_s{seed}"
            print(f"\n[{run_id}]  ({h} heads)")
            Ytr, Yva = Ytr_all[:, :h], Yva_all[:, :h]
            if ex.get("shuffle_phantom"):          # shuffle only the phantom head labels along rows
                sh = 2 * LEAK_PAIRS
                if h <= sh:
                    raise SystemExit(f"{nm}: N{n} has no phantom heads, nothing to shuffle")
                Ytr, Yva = Ytr.copy(), Yva.copy()
                rng = np.random.default_rng(SHUFFLE_SEED)
                for Y in (Ytr, Yva):
                    for j in range(sh, h):
                        Y[:, j] = Y[rng.permutation(len(Y)), j]
            TEE.buf = []
            t0 = time.time()
            out = T.run_single(arch, n, seed, Xtr, Ytr, Xva, Yva, dev, run_id, gn_w)
            gn_f.flush()
            log = "".join(TEE.buf)
            TEE.buf = None
            t_min = (time.time() - t0) / 60
            with open(os.path.join(sub, f"log_{run_id}.txt"), "w", encoding="utf-8") as f:
                f.write(log)
            if isinstance(out, dict):
                p = os.path.join(sub, "train_summary.csv")
                new = not os.path.exists(p)
                with open(p, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(out.keys()))
                    if new:
                        w.writeheader()
                    w.writerow(out)
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            accs = parse_acc(log)
            o_acc, o_ep, o_gn = orig_info(orig_dir, arch, n, seed)
            gn_new = {}
            for r in csv.DictReader(open(gn_path, encoding="utf-8")):
                if r["run_id"] == run_id:
                    gn_new[int(r["epoch"])] = float(r["gn_total"])
            if accs:
                b = int(np.argmax(accs))
                esc = next((e for e, a in enumerate(accs, 1) if a > ESCAPE_ACC), None)
                vs = virtual_stop(accs, base.get("PATIENCE", 15), base.get("MIN_DELTA", 0.3))
                acc_vs = max(accs[:vs]) if vs else max(accs)
            else:
                print("  [warning] could not read per-epoch acc from the log — please report this if the output format differs")
                b, esc, vs, acc_vs = 0, None, None, float("nan")
            row = dict(experiment=nm, desc=ex["desc"], arch=arch, n_pairs=n, seed=seed,
                       overrides=json.dumps(ex["over"]), epochs_run=len(accs),
                       best_acc=round(max(accs), 3) if accs else "", best_epoch=b + 1 if accs else "",
                       escape_epoch=esc if esc else "", virtual_p15_stop=vs if vs else "",
                       acc_at_virtual_stop=round(acc_vs, 3),
                       stuck_under_p15=int(acc_vs <= ESCAPE_ACC),
                       escaped=int(bool(accs) and max(accs) > ESCAPE_ACC),
                       orig_val_acc=o_acc if o_acc is not None else "",
                       orig_epochs=o_ep if o_ep is not None else "",
                       gn_same_until_epoch=gn_same_until(o_gn, gn_new) if o_gn else "",
                       time_min=round(t_min, 1), run_id=run_id)
            new = not os.path.exists(sum_path)
            with open(sum_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"  -> best {row['best_acc']}% (epoch {row['best_epoch']}/{row['epochs_run']}) | "
                  f"escape epoch {row['escape_epoch'] or 'none'} | patience15 virtual stop {row['virtual_p15_stop'] or 'not cut'} "
                  f"(acc then {row['acc_at_virtual_stop']}%) | original {o_acc if o_acc is not None else '-'}% / {o_ep if o_ep is not None else '-'} epochs | "
                  f"same trajectory as original ~{row['gn_same_until_epoch']} epochs | {t_min:.0f} min")
        gn_f.close()

    for k, v in base.items():
        setattr(T, k, v)
    print_summary(sum_path)


def print_summary(p):
    if not os.path.exists(p):
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    print("\n" + "=" * 100)
    print(f"Control experiment summary (escape criterion val acc > {ESCAPE_ACC:.1f}%)")
    print("=" * 100)
    print(f"{'exp':<5}{'run':<24}{'best%':>8}{'ep':>9}{'esc_ep':>8}{'p15stop':>9}"
          f"{'acc@p15':>10}{'orig%':>8}{'orig_ep':>8}{'same_traj':>9}")
    for r in rows:
        print(f"{r['experiment']:<5}{r['run_id'][len(r['experiment'])+1:]:<24}{r['best_acc']:>8}"
              f"{r['best_epoch']+'/'+r['epochs_run']:>9}{r['escape_epoch'] or '-':>8}"
              f"{r['virtual_p15_stop'] or '-':>9}{r['acc_at_virtual_stop']:>10}"
              f"{r['orig_val_acc']:>8}{r['orig_epochs']:>8}{r['gn_same_until_epoch'] or '-':>9}")
    print("=" * 100)
    print("How to read")
    print("  D : compare best with the original N39 10 seeds (56.3~69.5%, mean 62.9).")
    print("      If it recovers a lot, 29->39 is not a capacity cliff but a shortage of the 100-epoch budget; if unchanged, the capacity cliff is confirmed")
    print("  E : acc@p15 <= 16.6 (a run that would have been cut while trapped under patience 15) but best > 16.6")
    print("        -> 'waiting gets it out' = patience truncation. Look at the escape epoch per N")
    print("      acc@p15 <= 16.6 and best also <= 16.6 -> cannot get out even with 50 more epochs = structural lock-in")
    print("      acc@p15 > 16.6 -> this time it escaped within patience 15 (non-determinism/hardware) — no information for the E question")
    print("  E2: if best > 16.6, it escapes at LR 3e-4 = due to LR. Only by looking at it separately from E can patience/LR be separated")
    print("  B : compare escape rate and best with the original ResNet N128 seeds 0~4 (5/5 escaped, signal acc 49.4~58.4%).")
    print("      Pairing the same seeds gives a direction even with 5 (the original stalls are s8 and s9, so none among 0~4)")
    print("      If the shuffle bends less, it is accumulation of residual gradient from phantom labels; if the same, dilution of the mean-loss denominator")
    print("  S34: place the N34 escape rate between N29 9/10 and N39 8/10 to judge the peak (peak=L means 'the convergence rate reveals L')")
    print("  S2 : slope on both sides of the peak with N24/N44. Decide whether to run after seeing the S34 results")
    print("  R5: merge seeds 10~14 with the existing 8/10 for an escape rate over 15 seeds")
    print("\n  Checkpoints are saved in CTRL_DIR/<experiment>/ in the same format as the originals + norm.npz, so")
    print("  pointing MODEL_DIR in eval_nscan.py at that folder and matching N_LIST/SEEDS runs key recovery directly")


if __name__ == "__main__":
    main()
