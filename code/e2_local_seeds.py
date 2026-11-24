#!/usr/bin/env python3
r"""
e2_local_seeds — E2: local readout with 10 seeds

Request document: `claude/TIFS additional experiments before resubmission … (11-17)` section 4.

    python e2_local_seeds.py --plan              # checks, targets, time. No computation   <- first
    python e2_local_seeds.py --stage1            # probe (produces logp)
    python e2_local_seeds.py --stage2            # key recovery (fine, canonical form)
    python e2_local_seeds.py --aggregate         # 4-4 output (collect without computation)
    python e2_local_seeds.py --stage1 --stage2 --aggregate
    …adding --rec includes the recommended set too (required 30 -> 65 runs)
    --max-runs N  only N runs in this invocation (for running in pieces)

Existing API as measured (confirmed with --plan)
------------------------------------------------
    probe_local       PreGAP, read_peaks(), pool_windows(fmap, win),
                      make_heads(H, C, ncls, dev), train(heads,Ftr,Ytr,Fva,Yva,dev,seed),
                      evaluate(heads, Fte, Y, logpr, dev)
                      EPOCHS 60  PATIENCE 12  N_TRAIN 16000  N_VALID 4000
                      DROPOUT 0.4  HALF 160  BATCH 256  LR 1e-3  WD 3e-3
    probe_local_ext   MODELS = [(tag, group, arch, n, seed, ckpt), …]  length 11
                      run_id(tag, arch, n, seed, ckpt), ckpt_path(...), main()
                      OUT  = <eval_nscan.OUT_DIR>/probe_local_ext
                      (the existing 11 ran without fine, so they actually live in eval/probe_local_ext)

So **tags are kept short.** `run_id` builds the name as `<tag>_<arch>_N<n>_s<seed>`, so
putting arch, N and seed into the tag would give `E2_resnet_N39_s0_resnet_N39_s0`.
Required uses tag "E2", recommended "E2r" — run_id becomes unique in the form `E2_resnet_N39_s0`.
The names do not collide with the existing `A0`~`B2b` logp files, so the s0/s1 reproduction
comparison required by 4-2 can be done side by side.

**No existing file is modified.** `probe_local_ext.MODELS` and `OUT` are swapped in inside this
process and that module's `main()` is called as-is (the way E6 did it for run_controls).
Protocol, head structure and training settings stay bit-identical to 11-12 and 11-13.

Targets (4-2)
-------------
  required 1  ResNet N39 s0~s9   39 pairs   probe + key recovery   (including seeds that did not converge)
  required 2  ResNet N49 s0~s9   39 pairs   probe + key recovery
  required 3  ResNet N29 s0~s9   the 29 learned pairs   probe + key recovery
  recommended 1  ResNet N9·N19 s0~s9   learned pairs   probe only
  recommended 2  MobileNetV2 N39 all converged seeds   39 pairs   probe + key recovery
  recommended 3  ResNet N69·N99 s0·s1, N128 s2~s4   39 pairs   probe + key recovery

Key recovery is exactly as in request 1-2: **canonical-form candidates** (same as the main evaluation, fine Q=1..50, 20 repeats).
The existing local Q*max values in section 12 are on the canonical-form basis, so the reproduction check only holds this way. If firmware-representation
(oracle) values are needed, run `e3_firmware_repr.py --local` after E2 finishes.

Outputs
-------
  <probe_local_ext.OUT>_e2/        logp_E2_*.npz, local_models.csv, local_pairs.csv
  <eval_qfine>_e2local/
    eval_runs.csv     two rows each, original head / local, including converged_orig — 4-4(2)
    pair_qstar.csv    per pair — 4-4(2)
    summary_4_4.csv   summary table mean±std [min,max] — 4-4(3)
    worst_heads.csv   lowest accuracy per head with its run and pair — 4-4(4)
    pair_compare.csv  local is faster/same/slower — 4-4(5)
    verify.txt        reproduction checks (section 12 local Q*max, 4-4 accuracy and information)
    res_<run_id>.npz  rank arrays — on rerun, finished runs are skipped
"""

import csv
import glob
import inspect
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_USER = sys.argv[1:]
# stage1 (probe) runs **without fine**. Same conditions as when the existing 11 logp files were made.
#   probe_local.NICV / probe_local_ext's evaluation indices / original-head statistics paths are all
#   derived from eval_nscan.OUT_DIR, and turning on fine moves them to eval_qfine/….
#   (measured 11-21: the 2000-trace idx in eval/ and eval_qfine/ are the same, but nicv/ exists only in eval/)
# stage2 (key recovery) needs the fine grid (Q=1..50, 20 repeats), so it runs with fine.
# If both are requested at once, stage1 is launched separately as a child process (see main()).
_WORK = {a for a in _USER if a.startswith("--")} - {"--rec", "--max-runs", "--plan"}
_STAGE1_ALONE = (_WORK == {"--stage1"})
sys.argv = [sys.argv[0]] if _STAGE1_ALONE else [sys.argv[0], "fine"]

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import eval_nscan as E                                              # noqa: E402

MAIN = E.OUT_DIR                                     # stage1 alone: …/eval, otherwise: …/eval_qfine
FINE_DIR = MAIN if MAIN.endswith("_qfine") else MAIN.rstrip("/\\") + "_qfine"
OUT_K = FINE_DIR + "_e2local"                         # …/eval_qfine_e2local in either mode
LEAK = E.LEAK_PAIRS

# (set, tag, group, arch, N list, seed list or None=converged seeds, key recovery)
REQUIRED = [("required 1", "E2", "A", "resnet", [39], list(range(10)), True),
            ("required 2", "E2", "A", "resnet", [49], list(range(10)), True),
            ("required 3", "E2", "A", "resnet", [29], list(range(10)), True)]
RECOMMENDED = [("recommended 1", "E2r", "B", "resnet", [9, 19], list(range(10)), False),
               ("recommended 2", "E2r", "B", "mobilenet", [39], None, True),
               ("recommended 3a", "E2r", "B", "resnet", [69, 99], [0, 1], True),
               ("recommended 3b", "E2r", "B", "resnet", [128], [2, 3, 4], True)]

# request section 12 — local head Q*max (canonical form). None = >50
REF_Q = {("resnet", 29, 0): 8, ("resnet", 39, 0): 7, ("resnet", 39, 1): 6,
         ("resnet", 49, 0): 6, ("resnet", 49, 1): 7,
         ("resnet", 128, 0): 16, ("resnet", 128, 1): 9,
         ("mobilenet", 39, 1): None, ("mobilenet", 39, 2): 8}
# Accuracy and information reproduction is not hard-coded; it is compared directly with the existing local_models.csv
# (repro_against_old). In the 11-17 request table, N39 s0 "97.0 / 3.02" and N49 s0 "97.1 / 3.33" are
# the s0·s1 average placed in the s0 cell — the actual s0 values are 96.89 / 2.84 and 97.01 / 3.32 (11-13 document
# section 0, local_models.csv, localrec eval_heads.csv all three agree). Confirmed 11-21.

RUN_FIELDS = ["run_id", "heads_src", "arch", "n_pairs", "seed", "group",
              "converged_orig", "n_pairs_eval", "q_star_max", "q_star_mean",
              "q_star_median", "n_pairs_censored", "slowest_pair_idx", "slowest_pair",
              "sr_q5", "ge_log2_qmax", "acc_mean", "acc_min", "info_mean", "info_min",
              "n_repeat", "eval_time_sec"]
PQ_FIELDS = ["run_id", "heads_src", "arch", "n_pairs", "seed", "pair_idx", "pair",
             "q_star", "censored", "ge_log2_qmax"]
SUM_FIELDS = ["heads_src", "arch", "n_pairs", "n_seeds", "n_converged_orig",
              "q_star_max_mean", "q_star_max_std", "q_star_max_min", "q_star_max_max",
              "q_star_mean_mean", "q_star_mean_std",
              "sr_q5_mean", "sr_q5_std", "acc_mean", "acc_std",
              "info_mean", "info_std", "n_censored_runs"]
WH_FIELDS = ["run_id", "arch", "n_pairs", "seed", "head", "pair", "role",
             "acc", "info", "rank_in_run"]
PC_FIELDS = ["arch", "n_pairs", "seed", "pair_idx", "pair",
             "q_star_orig", "q_star_local", "verdict"]


# ================= targets =================
def converged_seeds(arch, n):
    p = os.path.join(FINE_DIR, "eval_runs.csv")
    if not os.path.exists(p):
        return None
    out = set()
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("eval_mode") == "sum" and r.get("arch") == arch
                    and r.get("n_pairs") == str(n) and r.get("converged") == "1"):
                out.add(int(r["seed"]))
    return sorted(out)


def conv_flag(arch, n, seed):
    p = os.path.join(FINE_DIR, "eval_runs.csv")
    if not os.path.exists(p):
        return ""
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("eval_mode") == "sum" and r.get("arch") == arch
                    and r.get("n_pairs") == str(n) and r.get("seed") == str(seed)):
                return r.get("converged", "")
    return ""


def targets(with_rec, quiet=False):
    out = []
    for grp, tag, g, arch, ns, seeds, rec in REQUIRED + (RECOMMENDED if with_rec else []):
        for n in ns:
            ss = seeds
            if ss is None:
                ss = converged_seeds(arch, n)
                if ss is None:
                    if not quiet:
                        print(f"  [{grp}] {arch} N{n}: no eval_runs.csv, so the converged seeds"
                              f" cannot be determined — skipping")
                    continue
                if not quiet:
                    print(f"  [{grp}] {arch} N{n}: converged seeds {ss}")
            for s in ss:
                ck = os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{s}.pth")
                out.append(dict(group=grp, tag=tag, g=g, arch=arch, n_pairs=n,
                                seed=s, ckpt=ck, pairs=min(n, LEAK), recover=rec,
                                exists=os.path.exists(ck)))
    return out


def px_mod():
    try:
        import probe_local_ext as px
    except Exception as e:
        raise SystemExit(f"could not import probe_local_ext: {type(e).__name__}: {e}\n"
                         f"  it must be in the same folder: {HERE}")
    for need in ("MODELS", "OUT", "run_id", "main"):
        if not hasattr(px, need):
            raise SystemExit(f"probe_local_ext has no {need} — looks like a different version.\n"
                             f"  please send the output of python e2_local_seeds.py --plan.")
    return px


ROOT = os.path.dirname(MAIN.rstrip("/\\"))           # …/tifs
BASE = MAIN[:-len("_qfine")] if MAIN.endswith("_qfine") else MAIN    # …/eval


def resolve_paths(quiet=False):
    r"""Fix the paths that the `fine` profile throws off by **finding them where they exist**.

    `probe_local.NICV` and `probe_local_ext.OUT` are derived from `eval_nscan.OUT_DIR`.
    This script puts "fine" into `sys.argv` because key recovery needs the fine grid, so
    OUT_DIR becomes `…/eval_qfine`, and then the two paths point to `eval_qfine/nicv/…` and
    `eval_qfine/probe_local_ext`. But the existing probe ran without fine, so the
    actual files are in `…/eval/nicv/…` and `…/eval/probe_local_ext`.
    (The same thing was fixed in probe_local_recover for E3. It is not repeated here.)

    Returns: (nicv path or None, existing probe_local_ext output folder, E2 output folder)
    """
    import probe_local as pl
    px = px_mod()

    cands = [os.path.join(MAIN, "nicv", "nicv_pairs.csv"),
             os.path.join(BASE, "nicv", "nicv_pairs.csv")]
    cands += sorted(glob.glob(os.path.join(ROOT, "*", "nicv", "nicv_pairs.csv")))
    nicv = next((c for c in cands if os.path.exists(c)), None)

    old = [d for d in ([os.path.join(MAIN, "probe_local_ext"),
                        os.path.join(BASE, "probe_local_ext")]
                       + sorted(glob.glob(os.path.join(ROOT, "*", "probe_local_ext"))))
           if os.path.isdir(d)]
    old_best = next((d for d in old if glob.glob(os.path.join(d, "logp_*.npz"))),
                    (old[0] if old else os.path.join(BASE, "probe_local_ext")))
    e2 = old_best.rstrip("/\\") + "_e2"

    if not quiet:
        print("\nPath resolution (fine-profile correction)")
        print(f"  probe_local.NICV default   {pl.NICV}")
        print(f"  -> value actually used     {nicv or '**not found**'}")
        print(f"  probe_local_ext.OUT default {px.OUT}")
        print(f"  -> folder with existing logp {old_best} "
              f"({len(glob.glob(os.path.join(old_best, 'logp_*.npz')))} files)")
        print(f"  -> E2 output               {e2}")
    if nicv:
        pl.NICV = nicv                     # read_peaks() reads it at call time
    return nicv, old_best, e2


def out_probe(px):
    return resolve_paths(quiet=True)[2]


def rid_of(px, t):
    return px.run_id(t["tag"], t["arch"], t["n_pairs"], t["seed"], None)


# ================= --plan =================
def plan(with_rec):
    print("=" * 80)
    print("E2 plan — no computation, no writes")
    print("=" * 80)
    import probe_local as pl
    px = px_mod()
    print(f"\nprobe_local      {pl.__file__}")
    for k in ("EPOCHS", "PATIENCE", "N_TRAIN", "N_VALID", "DROPOUT", "HALF",
              "BATCH", "LR", "WD", "NICV"):
        print(f"  {k:<9} = {getattr(pl, k, '**missing**')!r}")
    for fn in ("read_peaks", "pool_windows", "make_heads", "train", "evaluate"):
        f = getattr(pl, fn, None)
        print(f"  {fn}{inspect.signature(f) if f else ' **missing**'}")
    print(f"\nprobe_local_ext  {px.__file__}")
    print(f"  MODELS length {len(px.MODELS)}  entry form {px.MODELS[0]!r}")
    print(f"  OUT = {px.OUT}")
    print(f"  run_id{inspect.signature(px.run_id)}")

    nicv, old_dir, OP = resolve_paths()
    print(f"\nE2 output")
    print(f"  probe   {OP}")
    print(f"  key recovery {OUT_K}")
    if not nicv:
        print("\n  **nicv_pairs.csv not found anywhere.**")
        print("    python nicv_pairs.py must be run first")
        print("    (the peak spacing alternates between 984/1048, so measured values are needed)")

    print(f"\n{'=' * 80}\nTargets (request 4-2)\n{'=' * 80}")
    tg = targets(with_rec)
    print(f"  {'group':<8}{'run_id':<24}{'pair':>4}{'key recovery':>7}"
          f"{'ckpt':>7}{'logp':>7}{'conv':>6}")
    print("  " + "-" * 64)
    todo = 0
    for t in tg:
        rid = rid_of(px, t)
        has = os.path.exists(os.path.join(OP, f"logp_{rid}.npz"))
        todo += int(t["exists"] and not has)
        print(f"  {t['group']:<8}{rid:<24}{t['pairs']:>4}"
              f"{'O' if t['recover'] else '-':>7}"
              f"{'O' if t['exists'] else '**X**':>7}{'O' if has else '-':>7}"
              f"{conv_flag(t['arch'], t['n_pairs'], t['seed']) or '?':>6}")
    print("  " + "-" * 64)
    nrec = sum(1 for t in tg if t["recover"] and t["exists"])
    print(f"  total {len(tg)} runs | no ckpt {sum(1 for t in tg if not t['exists'])} runs"
          f" | to probe {todo} runs | key recovery targets {nrec} runs")
    print(f"\n  the 'conv' (converged) column is converged in eval_qfine/eval_runs.csv — converged_orig of 4-3")
    print(f"  time is printed as an estimate for the whole set after running 1 run (request estimate 3~4 hours)")
    print("=" * 80)
    print("\n  python e2_local_seeds.py --stage1" + (" --rec" if with_rec else ""))


# ================= stage1: probe =================
def stage1(with_rec, maxruns):
    px = px_mod()
    nicv, _old, OP = resolve_paths()
    if not nicv:
        raise SystemExit(
            "nicv_pairs.csv not found anywhere.\n"
            "  run python nicv_pairs.py first (measured peak values are needed).\n"
            f"  searched: {MAIN}/nicv/, {BASE}/nicv/, {ROOT}/*/nicv/")
    os.makedirs(OP, exist_ok=True)
    proto = px.MODELS[0]
    if not (isinstance(proto, (list, tuple)) and len(proto) == 6):
        raise SystemExit(f"probe_local_ext.MODELS entry form differs: {proto!r}\n"
                         f"  (expected 6 fields: tag, group, arch, n, seed, ckpt).")
    tg = [t for t in targets(with_rec) if t["exists"]]
    tg = [t for t in tg
          if not os.path.exists(os.path.join(OP, f"logp_{rid_of(px, t)}.npz"))]
    if maxruns:
        tg = tg[:maxruns]
    if not tg:
        print("stage1: nothing to run (all logp files exist)")
        return
    rows = [type(proto)((t["tag"], t["g"], t["arch"], t["n_pairs"], t["seed"], None))
            if isinstance(proto, tuple) else
            [t["tag"], t["g"], t["arch"], t["n_pairs"], t["seed"], None] for t in tg]
    print(f"replacing probe_local_ext.MODELS with the E2 set of {len(rows)} "
          f"(the existing {len(px.MODELS)} are not touched)")
    print(f"  e.g.: {rows[0]!r}  ->  run_id {rid_of(px, tg[0])}")
    print(f"  output OUT -> {OP}")
    px.MODELS = type(px.MODELS)(rows) if isinstance(px.MODELS, tuple) else rows
    px.OUT = OP
    import probe_local as pl
    pl.OUT = OP                      # anything written via probe_local also goes to the same folder
    print(f"  probe_local.NICV = {pl.NICV}")
    sys.argv = [sys.argv[0]]
    t0 = time.time()
    px.main()
    print(f"\nstage1 done — {(time.time()-t0)/60:.1f} min, logp "
          f"{len(glob.glob(os.path.join(OP, 'logp_*.npz')))} files")
    repro_against_old(_old, OP)


def repro_against_old(old_dir, new_dir):
    """Request 4-2 — rerun the already-run s0·s1 and check they match the existing values.

    Compare directly against the existing `local_models.csv` (A0~B2b) for the same (arch, N, seed, ckpt=main).
    The numbers in the request table are not used as reference — the N39·N49 s0 cells there are s0·s1 averages.
    """
    def load(p):
        out = {}
        if not os.path.exists(p):
            return out
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("ckpt", "main") != "main":
                    continue
                out[(r["arch"], int(r["src_n_pairs"]), int(r["seed"]))] = r
        return out
    old = load(os.path.join(old_dir, "local_models.csv"))
    new = load(os.path.join(new_dir, "local_models.csv"))
    keys = sorted(set(old) & set(new))
    if not keys:
        return
    lines = ["", "probe reproduction (request 4-2) — direct comparison with the existing local_models.csv"]
    for k in keys:
        o, n = old[k], new[k]
        da = float(n["acc_seen"]) - float(o["acc_seen"])
        di = float(n["info_seen"]) - float(o["info_seen"])
        good = abs(da) < 0.5 and abs(di) < 0.05
        lines.append(f"  {k[0]} N{k[1]} s{k[2]}:  acc {o['acc_seen']} -> {n['acc_seen']}"
                     f" ({da:+.3f})   info {o['info_seen']} -> {n['info_seen']} ({di:+.3f})"
                     f"   early stopping ep {o['probe_epochs']} -> {n['probe_epochs']}"
                     f"   {'✓' if good else '**differs**'}")
    for s in lines:
        print(s)
    os.makedirs(OUT_K, exist_ok=True)
    with open(os.path.join(OUT_K, "verify.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines[1:]) + "\n")


# ================= stage2: key recovery =================
def load_pairs_csv(path, mode="sum"):
    if not os.path.exists(path):
        return {}
    acc = {}
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        qc = [c for c in (rd.fieldnames or []) if c.startswith("rank_q")]
        if [int(c[6:]) for c in qc] != list(E.Q_LIST):
            raise SystemExit(f"{path}: Q grid differs from the current one (check that it is a fine result)")
        for r in rd:
            if r.get("eval_mode") != mode:
                continue
            acc.setdefault((r["run_id"], int(r["pair_idx"])), []).append(
                [float(r[c]) if r[c] != "" else np.nan for c in qc])
    return {k: np.array(v, np.float64) for k, v in acc.items()}


def run_stats(by_pair, pairs):
    qs, ge_l, sr = {}, [], []
    i5 = E.Q_LIST.index(5) if 5 in E.Q_LIST else None
    for p in pairs:
        rk = by_pair[p]
        g = rk.mean(0)
        qs[p] = E.q_star_of(g)
        ge_l.append(float(np.log2(g[-1])))
        if i5 is not None:
            sr.append(float((rk[:, i5] == 1).mean()))
    ok = [q for q in qs.values() if q is not None]
    cens = [p for p, q in qs.items() if q is None]
    slow = (max(cens, key=lambda p: by_pair[p].mean(0)[-1]) if cens else
            (max(qs, key=lambda p: (qs[p], by_pair[p].mean(0)[-1])) if qs else None))
    return dict(qs=qs, q_max=(max(ok) if ok and not cens else None),
                q_mean=(float(np.mean(ok)) if ok else None),
                q_med=(float(np.median(ok)) if ok else None),
                n_cens=len(cens), slow=slow,
                sr5=(float(np.mean(sr)) if sr else None),
                ge=(float(np.mean(ge_l)) if ge_l else None))


def _r(x, nd):
    return "" if x is None or not np.isfinite(x) else round(float(x), nd)


def write_rows(path, fields, rows, key):
    """Overwrite instead of appending. If the column layout changed, move the old file to .bak and write anew."""
    keep = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if list(rd.fieldnames or []) != list(fields):
                os.replace(path, path + ".bak")
                print(f"    [note] {os.path.basename(path)}: column layout differs, moved to .bak")
            else:
                mine = {tuple(str(r.get(k, "")) for k in key) for r in rows}
                keep = [r for r in rd
                        if tuple(str(r.get(k, "")) for k in key) not in mine]
    if os.path.exists(path):
        os.remove(path)
    E.append_rows(path, fields, keep + rows)
    print(f"    saved {os.path.basename(path)}  {len(keep)+len(rows)} rows"
          + (f" (kept {len(keep)} existing rows)" if keep else ""))


def stage2(with_rec, maxruns):
    px = px_mod()
    OP = resolve_paths(quiet=True)[2]
    os.makedirs(OUT_K, exist_ok=True)
    verify = []

    tg = [t for t in targets(with_rec, quiet=True) if t["recover"] and t["exists"]]
    tg = [t for t in tg
          if os.path.exists(os.path.join(OP, f"logp_{rid_of(px, t)}.npz"))]
    pend = [t for t in tg
            if not os.path.exists(os.path.join(OUT_K, f"res_{rid_of(px, t)}.npz"))]
    if maxruns:
        pend = pend[:maxruns]
    if not pend:
        print("stage2: nothing to run")
        return
    print(f"stage2: key recovery targets with logp: {len(tg)} runs, remaining {len(pend)} runs")

    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = E.CHUNK or (64 if dev.type == "cuda" and
                        torch.cuda.get_device_properties(0).total_memory / 2**30 >= 40
                        else 16)
    src = os.path.join(FINE_DIR, "eval_indices.npz")
    dst = os.path.join(OUT_K, "eval_indices.npz")
    if not os.path.exists(dst):
        if not os.path.exists(src):
            raise SystemExit(f"evaluation indices not found: {src}")
        import shutil
        shutil.copy2(src, dst)
    E.needed_pairs = lambda: list(range(LEAK))
    E.OUT_DIR = OUT_K
    X, Y, prior, tabs, tflat, mism, orders = E.prepare(T, dev)
    a, b = np.load(src), np.load(dst)
    same = (np.array_equal(a["idx"], b["idx"])
            and np.array_equal(a["orders"], b["orders"]))
    verify.append(f"index check {'match ✓' if same else '**mismatch**'}")
    print("  " + verify[-1])
    hw = torch.from_numpy(np.asarray(T.HW16, np.int64)).to(dev)
    ln2 = np.log(2.0)

    for t in pend:
        rid = rid_of(px, t)
        z = np.load(os.path.join(OP, f"logp_{rid}.npz"))
        k = "logp" if "logp" in z.files else z.files[0]
        lp = np.asarray(z[k], np.float32)
        if lp.shape[0] != len(X):
            print(f"[{rid}] skipped — logp trace count {lp.shape[0]} != {len(X)}")
            continue
        # whether the logp rows are in the same order as the current 2000 evaluation traces — if not, they misalign with the labels
        if "te_idx" in z.files and not np.array_equal(np.asarray(z["te_idx"]), a["idx"]):
            print(f"[{rid}] **skipped — te_idx in logp differs from the evaluation indices**")
            verify.append(f"{rid}: te_idx mismatch — skipped")
            continue
        H = min(lp.shape[1], 2 * t["pairs"])
        pl_ = list(range(t["pairs"]))
        logpr = np.log(prior[:H])
        acc = np.array([(lp[:, h].argmax(1) == Y[:, h]).mean() * 100 for h in range(H)])
        ll = np.take_along_axis(lp[:, :H], Y[:, :H, None], 2)[..., 0]
        info = (ll - logpr[np.arange(H)[None, :], Y[:, :H]]).mean(0) / ln2
        print(f"[{rid}] pairs {len(pl_)}  heads {H}  "
              f"acc {acc.mean():.2f}% (min {acc.min():.2f}%)  info {info.mean():.3f}bit")
        t0 = time.time()
        rk = {}
        for i, p in enumerate(pl_):
            td = [torch.from_numpy(x).to(dev) for x in tabs[p]]
            l0 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p])).to(dev)
            l1 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p + 1])).to(dev)
            r, _ = E.recover_pair(td, l0, l1, hw, tflat[p], orders, chunk, dev)
            rk[p] = r["sum"]
            del td, l0, l1
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if i == 0:
                el = time.time() - t0
                print(f"    1 pair {el:.1f}s -> this run about {el*len(pl_)/60:.1f} min, "
                      f"remaining {len(pend)} runs about {el*len(pl_)*len(pend)/3600:.1f} h")
        np.savez(os.path.join(OUT_K, f"res_{rid}.npz"),
                 acc=acc, info=info, heads=H, eval_sec=time.time() - t0,
                 arch=t["arch"], n_pairs=t["n_pairs"], seed=t["seed"],
                 **{f"p{p}": v for p, v in rk.items()})
        st = run_stats(rk, pl_)
        key = (t["arch"], t["n_pairs"], t["seed"])
        if key in REF_Q:
            ref, got = REF_Q[key], st["q_max"]
            good = (got == ref) if ref is not None else (st["n_cens"] > 0)
            verify.append(f"local reproduction {rid}: Q*max {got if got is not None else '>50'}"
                          f" (previous {ref if ref is not None else '>50'})"
                          f" {'✓' if good else '**differs**'}")
            print("    " + verify[-1])
        print(f"    Q*max {st['q_max'] if st['q_max'] is not None else '>50'}  "
              f"Q*mean {_r(st['q_mean'],2)}  SR@5 {_r(st['sr5'],3)}  "
              f"censor {st['n_cens']}  ({(time.time()-t0)/60:.1f} min)")
        del lp

    E.OUT_DIR = MAIN
    with open(os.path.join(OUT_K, "verify.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(verify) + "\n")
    print("\nVerification output (request 9-3)")
    for v in verify:
        print("  " + v)


# ================= aggregate: 4-4 =================
def aggregate(with_rec):
    px = px_mod()
    resolve_paths(quiet=True)
    if not os.path.isdir(OUT_K):
        print(f"{OUT_K} not found — run stage2 first")
        return
    orig = load_pairs_csv(os.path.join(FINE_DIR, "eval_pairs.csv"))
    tg = {rid_of(px, t): t for t in targets(with_rec, quiet=True)}
    rows_run, rows_pq, rows_wh, rows_pc = [], [], [], []
    for f in sorted(glob.glob(os.path.join(OUT_K, "res_*.npz"))):
        rid = os.path.basename(f)[4:-4]
        z = np.load(f, allow_pickle=False)
        arch, n, seed = str(z["arch"]), int(z["n_pairs"]), int(z["seed"])
        npair = min(n, LEAK)
        loc = {int(k[1:]): z[k] for k in z.files if k.startswith("p")}
        if not loc:
            continue
        acc, info = np.asarray(z["acc"]), np.asarray(z["info"])
        t = tg.get(rid, {})
        cf = conv_flag(arch, n, seed)
        orid = E.run_id_of(arch, n, seed)
        oh = {p: orig[(orid, p)] for p in range(npair) if (orid, p) in orig}

        for src_nm, by in (("local", loc), ("orig", oh)):
            if len(by) < npair:
                if src_nm == "orig":
                    print(f"  {rid}: original-head canonical-form data is short by "
                          f"{npair-len(by)} pairs — orig row omitted")
                continue
            st = run_stats(by, list(range(npair)))
            rows_run.append(dict(
                run_id=rid, heads_src=src_nm, arch=arch, n_pairs=n, seed=seed,
                group=t.get("group", ""), converged_orig=cf, n_pairs_eval=npair,
                q_star_max=("" if st["q_max"] is None else st["q_max"]),
                q_star_mean=_r(st["q_mean"], 2), q_star_median=_r(st["q_med"], 1),
                n_pairs_censored=st["n_cens"],
                slowest_pair_idx=("" if st["slow"] is None else st["slow"]),
                slowest_pair=("" if st["slow"] is None else st["slow"] + 1),
                sr_q5=_r(st["sr5"], 4), ge_log2_qmax=_r(st["ge"], 3),
                acc_mean=_r(acc.mean() if src_nm == "local" else None, 3),
                acc_min=_r(acc.min() if src_nm == "local" else None, 3),
                info_mean=_r(info.mean() if src_nm == "local" else None, 4),
                info_min=_r(info.min() if src_nm == "local" else None, 4),
                n_repeat=by[0].shape[0], eval_time_sec=_r(float(z["eval_sec"]), 1)))
            for p in range(npair):
                q = st["qs"][p]
                rows_pq.append(dict(run_id=rid, heads_src=src_nm, arch=arch,
                                    n_pairs=n, seed=seed, pair_idx=p, pair=p + 1,
                                    q_star=("" if q is None else q),
                                    censored=int(q is None),
                                    ge_log2_qmax=_r(float(np.log2(
                                        by[p].mean(0)[-1])), 3)))
        # 4-4(4) lowest accuracy per head
        order = np.argsort(acc)
        for r_i, h in enumerate(order[:3]):
            rows_wh.append(dict(run_id=rid, arch=arch, n_pairs=n, seed=seed,
                                head=int(h), pair=int(h) // 2 + 1,
                                role="r0" if h % 2 == 0 else "r1",
                                acc=_r(acc[h], 3), info=_r(info[h], 4),
                                rank_in_run=r_i + 1))
        # 4-4(5) per pair faster/same/slower
        if len(oh) >= npair:
            so = run_stats(oh, list(range(npair)))
            sl = run_stats(loc, list(range(npair)))
            for p in range(npair):
                a_, b_ = so["qs"][p], sl["qs"][p]
                if a_ is None and b_ is None:
                    v = "both unrecovered"
                elif b_ is None:
                    v = "slower"
                elif a_ is None:
                    v = "faster"
                else:
                    v = "faster" if b_ < a_ else ("slower" if b_ > a_ else "same")
                rows_pc.append(dict(arch=arch, n_pairs=n, seed=seed, pair_idx=p,
                                    pair=p + 1,
                                    q_star_orig=("" if a_ is None else a_),
                                    q_star_local=("" if b_ is None else b_),
                                    verdict=v))
    if not rows_run:
        print("res_*.npz not found — run stage2 first")
        return

    print("")
    for nm, fl, rw, ky in (("eval_runs.csv", RUN_FIELDS, rows_run,
                            ("run_id", "heads_src")),
                           ("pair_qstar.csv", PQ_FIELDS, rows_pq,
                            ("run_id", "heads_src", "pair_idx")),
                           ("worst_heads.csv", WH_FIELDS, rows_wh,
                            ("run_id", "head")),
                           ("pair_compare.csv", PC_FIELDS, rows_pc,
                            ("arch", "n_pairs", "seed", "pair_idx"))):
        write_rows(os.path.join(OUT_K, nm), fl, rw, ky)
    table()


def table():
    p = os.path.join(OUT_K, "eval_runs.csv")
    if not os.path.exists(p):
        print(f"{p} not found")
        return
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    g = lambda r, k: (float(r[k]) if r.get(k, "") not in ("", None) else None)
    print("\n" + "=" * 106)
    print("E2 summary — request 4-4(3).  fine Q=1..50, 20 repeats, 2000 test traces, canonical-form candidates")
    print("=" * 106)
    print(f"{'arch':<11}{'N':>4}{'seeds':>6}{'conv':>6}  "
          f"{'orig Q*max':>18}{'Q*mean':>8}{'SR@5':>7}  "
          f"{'local Q*max':>18}{'Q*mean':>8}{'SR@5':>7}{'acc%':>7}{'info':>7}")
    print("-" * 106)
    sumrows = []
    keys = sorted({(r["arch"], int(r["n_pairs"])) for r in rows},
                  key=lambda k: (k[0] != "resnet", k[1]))
    for arch, n in keys:
        out = {}
        for src in ("orig", "local"):
            rs = [r for r in rows if r["arch"] == arch and int(r["n_pairs"]) == n
                  and r["heads_src"] == src]
            if not rs:
                continue
            qm = [g(r, "q_star_max") for r in rs if g(r, "q_star_max") is not None]
            qa = [g(r, "q_star_mean") for r in rs if g(r, "q_star_mean") is not None]
            s5 = [g(r, "sr_q5") for r in rs if g(r, "sr_q5") is not None]
            ac = [g(r, "acc_mean") for r in rs if g(r, "acc_mean") is not None]
            inf = [g(r, "info_mean") for r in rs if g(r, "info_mean") is not None]
            out[src] = (rs, qm, qa, s5, ac, inf)
        if "local" not in out:
            continue
        fm = lambda v: (f"{np.mean(v):.1f}±{np.std(v):.1f} [{min(v):.0f},{max(v):.0f}]"
                        if len(v) > 1 else (f"{v[0]:.0f}" if v else "—"))
        fa = lambda v, nd=2: (f"{np.mean(v):.{nd}f}" if v else "—")
        lr, lqm, lqa, ls5, lac, linf = out["local"]
        nconv = sum(1 for r in lr if r["converged_orig"] == "1")
        if "orig" in out:
            orr, oqm, oqa, os5, _, _ = out["orig"]
        else:
            oqm = oqa = os5 = []
        print(f"{arch:<11}{n:>4}{len(lr):>6}{nconv:>6}  "
              f"{fm(oqm):>18}{fa(oqa):>8}{fa(os5,3):>7}  "
              f"{fm(lqm):>18}{fa(lqa):>8}{fa(ls5,3):>7}"
              f"{fa(lac,1):>7}{fa(linf,2):>7}")
        for src, (rs, qm, qa, s5, ac, inf) in out.items():
            sumrows.append(dict(
                heads_src=src, arch=arch, n_pairs=n, n_seeds=len(rs),
                n_converged_orig=sum(1 for r in rs if r["converged_orig"] == "1"),
                q_star_max_mean=_r(np.mean(qm) if qm else None, 2),
                q_star_max_std=_r(np.std(qm) if len(qm) > 1 else None, 2),
                q_star_max_min=(min(qm) if qm else ""),
                q_star_max_max=(max(qm) if qm else ""),
                q_star_mean_mean=_r(np.mean(qa) if qa else None, 2),
                q_star_mean_std=_r(np.std(qa) if len(qa) > 1 else None, 2),
                sr_q5_mean=_r(np.mean(s5) if s5 else None, 4),
                sr_q5_std=_r(np.std(s5) if len(s5) > 1 else None, 4),
                acc_mean=_r(np.mean(ac) if ac else None, 3),
                acc_std=_r(np.std(ac) if len(ac) > 1 else None, 3),
                info_mean=_r(np.mean(inf) if inf else None, 4),
                info_std=_r(np.std(inf) if len(inf) > 1 else None, 4),
                n_censored_runs=sum(1 for r in rs
                                    if int(r["n_pairs_censored"] or 0) > 0)))
    print("-" * 106)
    print("  Q*max is the mean±std [min,max] of the per-run maxima. Runs with censoring are excluded")
    print("  conv = number of runs whose original head converged (converged_orig of 4-3)")
    sp = os.path.join(OUT_K, "summary_4_4.csv")
    if os.path.exists(sp):
        os.remove(sp)
    if sumrows:
        E.append_rows(sp, SUM_FIELDS, sumrows)
        print(f"\nsaved: {sp}")

    pc = os.path.join(OUT_K, "pair_compare.csv")
    if os.path.exists(pc):
        with open(pc, encoding="utf-8") as f:
            pr = list(csv.DictReader(f))
        if pr:
            print(f"\n4-4(5) per pair — local head compared with the original head")
            tot = {}
            for r in pr:
                tot[r["verdict"]] = tot.get(r["verdict"], 0) + 1
            print("  total: " + " / ".join(f"{k} {v}" for k, v in
                                         sorted(tot.items(), key=lambda x: -x[1]))
                  + f"   (pair-runs: {len(pr)})")
            for arch, n in sorted({(r["arch"], int(r["n_pairs"])) for r in pr}):
                s = [r for r in pr if r["arch"] == arch and int(r["n_pairs"]) == n]
                d = {}
                for r in s:
                    d[r["verdict"]] = d.get(r["verdict"], 0) + 1
                print(f"  {arch} N{n}: " + " / ".join(f"{k} {v}" for k, v in
                                                      sorted(d.items(), key=lambda x: -x[1])))

    wh = os.path.join(OUT_K, "worst_heads.csv")
    if os.path.exists(wh):
        with open(wh, encoding="utf-8") as f:
            wr = [r for r in csv.DictReader(f) if r["rank_in_run"] == "1"]
        if wr:
            wr.sort(key=lambda r: float(r["acc"]))
            print(f"\n4-4(4) lowest accuracy per head — bottom 10")
            print(f"  {'run_id':<24}{'head':>5}{'pair':>5}{'role':>6}{'acc%':>8}{'info':>8}")
            for r in wr[:10]:
                print(f"  {r['run_id']:<24}{r['head']:>5}{r['pair']:>5}"
                      f"{r['role']:>6}{float(r['acc']):>8.2f}{float(r['info']):>8.3f}")
            n80 = sum(1 for r in wr if float(r["acc"]) < 80)
            print(f"  lowest head below 80%: {n80}/{len(wr)} runs")
    print("=" * 106)
    print(f"raw data: {OUT_K}")


def main():
    flags = {a for a in _USER if a.startswith("--")}
    rec = "--rec" in flags
    mx = 0
    if "--max-runs" in _USER:
        i = _USER.index("--max-runs")
        mx = int(_USER[i + 1]) if i + 1 < len(_USER) else 0
    work = flags - {"--rec", "--max-runs"}
    if "--plan" in flags or not work:
        plan(rec)
        return
    if "--stage1" in flags:
        if _STAGE1_ALONE:
            stage1(rec, mx)
        else:
            # This process is running with fine. stage1 must run without fine, so
            # launch the same script as a child with only --stage1
            import subprocess
            cmd = [sys.executable, os.path.abspath(__file__), "--stage1"]
            if rec:
                cmd.append("--rec")
            if mx:
                cmd += ["--max-runs", str(mx)]
            print(f"launching stage1 separately without fine: {' '.join(cmd[1:])}\n")
            env = dict(os.environ, PYTHONUNBUFFERED="1")   # so the child's output shows up in the log immediately
            rc = subprocess.call(cmd, cwd=HERE, env=env)
            if rc != 0:
                raise SystemExit(f"stage1 failed (exit code {rc}) — not running stage2")
    if "--stage2" in flags:
        stage2(rec, mx)
    if "--aggregate" in flags:
        aggregate(rec)


if __name__ == "__main__":
    main()
