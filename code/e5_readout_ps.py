#!/usr/bin/env python3
r"""
e5_readout_ps — E5: global average (P) vs. a single position (S) with a head of the same size

Request document: `claude/TIFS additional experiment request before resubmission … (11-17)` section 7.

    python e5_readout_ps.py --plan                    # check targets, windows, boundaries. No computation  <- run first
    python e5_readout_ps.py --stage1                  # P/S probe (produces logp)
    python e5_readout_ps.py --stage2                  # P/S key recovery (fine, canonical form)
    python e5_readout_ps.py --aggregate               # 7-4 output (includes existing orig/local4 values)
    python e5_readout_ps.py --stage1 --stage2 --aggregate
    … adding --rec also runs the recommended group (MobileNetV2 N39 s1/s2, ResNet N128 s0/s1)

**The reported results are the `--bn` version (author decision, 11-21).** P/S head = BatchNorm1d(C) + Dropout 0.4 +
Linear(C->17), bias = log prior — the same structure as the original head (train_nscan._head). With the head of
request 7-2 (Dropout + Linear), the channel variance of the GAP features is small and P does not train (N39 s0 27.9% /
0.07 bit). Reproduction commands:
    python e5_readout_ps.py --stage1 --stage2 --aggregate --bn
    python e5_readout_ps.py --stage1 --stage2 --aggregate --bn --rec

Question (7-1)
--------------
With the same frozen backbone, a head of the same size and the same training conditions, if **only the read range**
is switched between the global average and a single position, how much do accuracy and information change?

Two readouts — same head structure and parameter count (7-2)
------------------------------------------------------------
  P (pooled refit)   global average of the last feature map (GAP)     Dropout 0.4 + Linear(C -> 17)
  S (one position)   the one feature-map position containing its peak  Dropout 0.4 + Linear(C -> 17)
  C = 512 (ResNet), 1280 (MobileNetV2)

Uses the `probe_local` machinery as is and **changes only the window**. Since `pool_windows(fmap, win)` produces
`fmap[:, :, lo:hi].mean(2)` for each head,
  P : win = [(0, T')] * H                     -> every head reads the same GAP vector
  S : win = [(k, k+1)],  k = floor(c / r)     -> the single position containing peak c
  (the local 4 positions are [int((c-160)/r), int((c+160)/r)) = [k-2, k+2), so S is one cell inside it)
r = input length / T' = ResNet 80, MobileNetV2 160. k equals the cell whose position center (k + 1/2)·r is
closest to c. If the fractional part of c/r is below 0.1 or above 0.9, it is marked "boundary" in the log (7-2).

Training conditions — bit-identical to probe_local_ext
------------------------------------------------------
  frozen backbone, train 16,000 (first 16,000 of the main training train set) / valid 4,000 (first 4,000)
  60 epochs, AdamW lr 1e-3 wd 3e-3, cosine, patience 12, MIN_DELTA 0.3, BATCH 256
  head training seed = model seed (P and S use the same seed). Test = main evaluation 2,000 traces (eval_indices.npz)
  **Head initial values are also fixed with the same seed** (manual_seed right before make_heads). In the existing
  probe the initialization was outside the seed, so the third decimal place fluctuated between reruns — so that this
  fluctuation does not mix into the P/S difference.
  78 heads (39 signal pairs). For N < 39 only the learned pairs (0..N-1) are aggregated.
The P and S features are extracted together in **one forward pass** (same batch, same order).

Key recovery follows request 1-2: canonical-form candidates, fine (Q=1..50, 20 repeats) — the same path as E2.
stage1 (probe) runs without fine, stage2 (key recovery) runs with fine (same reason as E2: the probe-side
NICV path and evaluation indices are derived from eval_nscan.OUT_DIR). When run together, stage1 is
launched as a child process.

Targets (7-3)
-------------
  required 1  ResNet N39 s0/s1, N49 s0/s1          P/S probe + key recovery
  required 2  ResNet N9/N19/N29 s0/s1 (learned pairs)  P/S probe. N29 also key recovery
  required 3  D, D29                               P/S probe + key recovery
  recommended MobileNetV2 N39 s1/s2, ResNet N128 s0/s1   P/S probe + key recovery

Output
------
  <eval>/probe_e5/        logp_E5_<arch>_N<n>_s<seed>[_D|_D29]_{P,S}.npz,
                          readout_probe.csv (1 row per model x readout), readout_pairs.csv (1 row per head),
                          windows.csv (S position and boundary flag per head)
  <eval_qfine>_e5/
    readout_models.csv    run_id, readout ∈ {orig, P, S, local4}, acc_mean, acc_min,
                          info_mean, info_min, probe_epochs, q_star_max, q_star_mean, sr_q5
                          — orig/local4 copy existing values (7-4(1))
    eval_runs.csv, pair_qstar.csv   P/S key recovery (7-4(3))
    summary_7_4.txt       7-4(4) summary table + numbers used for the 7-5 decision
    verify.txt            index check, how much P differs from the existing pooled refit (probe_heads)
    res_<run_id>.npz      rank arrays — on rerun, finished runs are skipped

Does not modify any existing file. probe_local is only imported.
"""

import csv
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_USER = sys.argv[1:]
_OPTS = {"--rec", "--plan", "--bn", "--only"}
_WORK = {a for a in _USER if a.startswith("--")} - _OPTS
_STAGE1_ALONE = (_WORK == {"--stage1"})
_PLAN = "--plan" in _USER and not _WORK
# --bn : head = BatchNorm + Dropout + Linear, bias = log prior (same structure as the original head
#        train_nscan._head and the pooled refit of 11-09 probe_heads). Output folder is separated as _bn.
#        With the head of request 7-2 (Dropout + Linear), the channel variance of the GAP features is too small
#        and P does not train (11-21 measurement: N39 s0 P 27.9% / 0.07 bit, not converged in 60 epochs).
# --only : comma-separated run_ids (e.g. --only E5_resnet_N39_s0)
_BN = "--bn" in _USER
_ONLY = None
if "--only" in _USER and _USER.index("--only") + 1 < len(_USER):
    _ONLY = _USER[_USER.index("--only") + 1].split(",")
# stage1/plan without fine (so the probe paths resolve as before), everything else with fine
sys.argv = [sys.argv[0]] if (_STAGE1_ALONE or _PLAN) else [sys.argv[0], "fine"]

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import eval_nscan as E                                              # noqa: E402
import probe_local as PL                                            # noqa: E402

MAIN = E.OUT_DIR
FINE_DIR = MAIN if MAIN.endswith("_qfine") else MAIN.rstrip("/\\") + "_qfine"
BASE = FINE_DIR[:-len("_qfine")]                     # …/eval
ROOT = os.path.dirname(BASE)                         # …/tifs
CTRL = os.path.join(ROOT, "model_ctrl")
_SUF = "_bn" if _BN else ""
OUT_P = os.path.join(BASE, "probe_e5" + _SUF)
OUT_K = FINE_DIR + "_e5" + _SUF
LEAK = E.LEAK_PAIRS
H = 2 * LEAK
NICV_PATH = os.path.join(BASE, "nicv", "nicv_pairs.csv")

# (group, arch, N, seed, ctrl or None, key recovery)
REQUIRED = ([("required 1", "resnet", n, s, None, True) for n in (39, 49) for s in (0, 1)]
            + [("required 2", "resnet", n, s, None, n == 29) for n in (9, 19, 29) for s in (0, 1)]
            + [("required 3", "resnet", 39, 0, "D", True), ("required 3", "resnet", 29, 0, "D29", True)])
RECOMMENDED = ([("recommended", "mobilenet", 39, s, None, True) for s in (1, 2)]
               + [("recommended", "resnet", 128, s, None, True) for s in (0, 1)])

PROBE_FIELDS = ["run_id", "readout", "arch", "n_pairs", "seed", "ckpt", "feat_ch",
                "feat_len", "stride", "n_seen", "probe_epochs", "best_val_acc",
                "acc_mean", "acc_min", "info_mean", "info_min", "n_boundary", "time_min"]
PAIRH_FIELDS = ["run_id", "readout", "arch", "n_pairs", "seed", "ckpt", "head", "pair",
                "role", "seen", "acc", "majority_acc", "info_bits"]
WIN_FIELDS = ["run_id", "arch", "n_pairs", "seed", "ckpt", "head", "pair", "role",
              "peak_sample", "c_rel", "pos_S", "frac", "boundary", "local4_lo", "local4_hi"]
RM_FIELDS = ["run_id", "readout", "arch", "n_pairs", "seed", "ckpt", "acc_mean", "acc_min",
             "info_mean", "info_min", "probe_epochs", "q_star_max", "q_star_mean", "sr_q5",
             "n_pairs_censored", "slowest_pair", "source"]
RUN_FIELDS = ["run_id", "readout", "arch", "n_pairs", "seed", "ckpt", "n_pairs_eval",
              "q_star_max", "q_star_mean", "q_star_median", "n_pairs_censored",
              "slowest_pair_idx", "slowest_pair", "sr_q5", "ge_log2_qmax",
              "n_repeat", "eval_time_sec"]
PQ_FIELDS = ["run_id", "readout", "arch", "n_pairs", "seed", "ckpt", "pair_idx", "pair",
             "q_star", "censored", "ge_log2_qmax"]


def run_id(arch, n, seed, ctrl):
    return f"E5_{arch}_N{n}_s{seed}" + (f"_{ctrl}" if ctrl else "")


def ckpt_file(arch, n, seed, ctrl):
    if ctrl:
        return os.path.join(CTRL, ctrl, f"{arch}_N{n}_s{seed}.pth")
    return os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")


def targets(with_rec):
    out = []
    for grp, arch, n, s, ctrl, rec in REQUIRED + (RECOMMENDED if with_rec else []):
        f = ckpt_file(arch, n, s, ctrl)
        out.append(dict(group=grp, arch=arch, n=n, seed=s, ctrl=ctrl, recover=rec,
                        rid=run_id(arch, n, s, ctrl), ckpt=f, exists=os.path.exists(f),
                        seen=min(n, LEAK)))
    if _ONLY:
        out = [t for t in out if t["rid"] in _ONLY]
    return out


def make_heads(C, ncls, dev, Ytr):
    """Default: probe_local.make_heads (Dropout + Linear) — exactly as in request 7-2.
    --bn : BatchNorm + Dropout + Linear, bias = log prior — same as train_nscan._head and probe_heads."""
    if not _BN:
        return PL.make_heads(H, C, ncls, dev)
    import torch.nn as nn
    heads = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(C), nn.Dropout(PL.DROPOUT),
                                         nn.Linear(C, ncls)) for _ in range(H)]).to(dev)
    with torch.no_grad():
        for i, h in enumerate(heads):
            cnt = np.bincount(Ytr[:, i], minlength=ncls).astype(np.float64)
            h[-1].bias.copy_(torch.from_numpy(np.log((cnt + 1e-6) / cnt.sum())).float())
    return heads


def read_peaks():
    if not os.path.exists(NICV_PATH):
        raise SystemExit(f"{NICV_PATH} does not exist — run nicv_pairs.py first")
    PL.NICV = NICV_PATH
    return PL.read_peaks()


def windows(peaks, lo, r, Tf):
    """Per-head windows. Returns (winP, winS, winL, rows) — winL is the local 4 positions (for comparison, not trained)."""
    winP, winS, winL, rows = [], [], [], []
    for p in range(LEAK):
        for role in (0, 1):
            c = peaks[(p, role)] - lo
            x = c / r
            k = int(np.floor(x))
            k = max(0, min(Tf - 1, k))
            frac = x - np.floor(x)
            a = int(max(0, (c - PL.HALF) / r))
            b = int(min(Tf, max(a + 1, (c + PL.HALF) / r)))
            winP.append((0, Tf))
            winS.append((k, k + 1))
            winL.append((a, b))
            rows.append(dict(head=2 * p + role, pair=p + 1, role=f"r{role}",
                             peak_sample=peaks[(p, role)], c_rel=c, pos_S=k,
                             frac=round(float(frac), 3),
                             boundary=int(frac < 0.1 or frac > 0.9),
                             local4_lo=a, local4_hi=b))
    return winP, winS, winL, rows


def append_csv(path, fields, rows):
    E.append_rows(path, fields, rows)


def done_probe():
    p = os.path.join(OUT_P, "readout_probe.csv")
    if not os.path.exists(p):
        return set()
    with open(p, encoding="utf-8") as f:
        return {(r["run_id"], r["readout"]) for r in csv.DictReader(f)}


# ================= --plan =================
def plan(with_rec):
    print("=" * 84)
    print("E5 plan — no computation, no writes")
    print("=" * 84)
    print(f"  head         {'BatchNorm + Dropout + Linear (--bn)' if _BN else 'Dropout + Linear (request 7-2)'}")
    print(f"  probe_local  EPOCHS {PL.EPOCHS}  PATIENCE {PL.PATIENCE}  N_TRAIN {PL.N_TRAIN}"
          f"  N_VALID {PL.N_VALID}  DROPOUT {PL.DROPOUT}  HALF {PL.HALF}  LR {PL.LR}"
          f"  WD {PL.WD}  BATCH {PL.BATCH}")
    print(f"  NICV         {NICV_PATH}  {'present' if os.path.exists(NICV_PATH) else '**missing**'}")
    print(f"  eval indices {os.path.join(MAIN, 'eval_indices.npz')}")
    print(f"  probe output {OUT_P}")
    print(f"  key recovery output {OUT_K}")
    peaks = read_peaks()
    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    lo, hi = [int(v) for v in nz["crop"]]
    print(f"  window [{lo}, {hi})  length {hi-lo}")
    for arch, Tf in (("resnet", (hi - lo) // 80), ("mobilenet", (hi - lo) // 160)):
        r = (hi - lo) / Tf
        _, wS, wL, rows = windows(peaks, lo, r, Tf)
        nb = sum(x["boundary"] for x in rows)
        inside = sum(1 for (k, _), (a, b) in zip(wS, wL) if a <= k < b)
        print(f"\n  {arch}: T' = {Tf} (expected)  r = {r:.0f}  local window width "
              f"{int(np.mean([b - a for a, b in wL]))}")
        print(f"    heads whose S position lies inside the local 4-position window {inside}/{H}")
        print(f"    boundary heads (fractional part <0.1 or >0.9) {nb}/{H}"
              + (f": {[(x['pair'], x['role']) for x in rows if x['boundary']][:12]}" if nb else ""))
        print("    (the actual T' is re-measured from the model in stage1)")

    print(f"\n{'=' * 84}\nTargets (request 7-3)\n{'=' * 84}")
    dp = done_probe()
    print(f"  {'group':<7}{'run_id':<28}{'learned pairs':>6}{'key recovery':>7}{'ckpt':>6}{'P':>4}{'S':>4}")
    print("  " + "-" * 62)
    for t in targets(with_rec):
        print(f"  {t['group']:<7}{t['rid']:<28}{t['seen']:>6}"
              f"{'O' if t['recover'] else '-':>7}{'O' if t['exists'] else '**X**':>6}"
              f"{'O' if (t['rid'], 'P') in dp else '-':>4}"
              f"{'O' if (t['rid'], 'S') in dp else '-':>4}")
    tg = targets(with_rec)
    print("  " + "-" * 62)
    print(f"  {len(tg)} models x 2 readouts = {2*len(tg)} probes, "
          f"key recovery {2*sum(t['recover'] for t in tg)} runs")
    print(f"  time: probe 1.5~4 min per model (P and S), key recovery 3~4.5 min per run")
    print("=" * 84)


# ================= stage1: P/S probe =================
def stage1(with_rec):
    os.makedirs(OUT_P, exist_ok=True)
    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncls = int(getattr(T, "NUM_CLASSES", 17))
    peaks = read_peaks()
    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    lo, hi = [int(v) for v in nz["crop"]]
    mu, sd = nz["mu"], nz["sd"]

    # same rule as load() in probe_local_ext.main() (first 16,000 / first 4,000 / full test)
    def load(name, n_sub=None):
        d = np.load(os.path.join(E.SPLIT_DIR, f"kyber_{name}_crop.npz"))
        ct = np.asarray(d["ciphertexts"], np.uint8)
        sk = np.asarray(d["labels"][0], np.int64)
        uh = T.v_ntt(T.v_unpack_10bit(ct))
        Y = np.stack([T.HW16[np.asarray(T.v_stored(sk, uh, j)) & 0xFFFF]
                      for j in range(H)], 1).astype(np.int64)
        X = d["traces"][:, lo:hi]
        if n_sub is not None and n_sub < len(X):
            X, Y = X[:n_sub], Y[:n_sub]
        return ((X.astype(np.float32) - mu) / sd).astype(np.float32), Y

    todo = [t for t in targets(with_rec) if t["exists"]]
    dp = done_probe()
    todo = [t for t in todo if not ((t["rid"], "P") in dp and (t["rid"], "S") in dp)]
    if not todo:
        print("stage1: nothing to run")
        return
    print(f"stage1: {len(todo)} models")

    Xtr, Ytr = load("train", PL.N_TRAIN)
    Xva, Yva = load("valid", PL.N_VALID)
    Xte_all, Yte_all = load("test")
    prior = np.stack([(np.bincount(Yte_all[:, j], minlength=ncls) + 1.0) /
                      (len(Yte_all) + ncls) for j in range(H)])
    logpr = np.log(prior)
    maj = prior.max(1) * 100
    ip = os.path.join(MAIN, "eval_indices.npz")
    te_idx = np.load(ip)["idx"]
    Xte, Yte = Xte_all[te_idx], Yte_all[te_idx]
    del Xte_all
    print(f"device {dev} | train {Xtr.shape} valid {Xva.shape} test {Xte.shape}  ({ip})")

    for t in todo:
        t1 = time.time()
        cp = torch.load(t["ckpt"], map_location="cpu", weights_only=False)
        model = E.build(T, t["arch"], t["n"], int(cp.get("heads", 2 * t["n"])),
                        cp.get("resnet_variant")).to(dev)
        E.load_state(model, cp["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        pg = PL.PreGAP(model, torch.from_numpy(Xtr[:2]).unsqueeze(1), dev)
        C, Tf = pg.shape[1], pg.shape[2]
        r = Xtr.shape[1] / Tf
        winP, winS, winL, wrows = windows(peaks, lo, r, Tf)
        nb = sum(x["boundary"] for x in wrows)
        print(f"\n[{t['rid']}] C={C} T'={Tf} (downsampling {r:.1f}x) | S boundary heads {nb}/{H} | "
              f"{t['ckpt']}")
        append_csv(os.path.join(OUT_P, "windows.csv"), WIN_FIELDS,
                   [dict(run_id=t["rid"], arch=t["arch"], n_pairs=t["n"], seed=t["seed"],
                         ckpt=t["ctrl"] or "main", **w) for w in wrows])

        def feats(X):
            fp = np.empty((len(X), H, C), np.float16)
            fs = np.empty((len(X), H, C), np.float16)
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    xb = torch.from_numpy(X[i:i + 128]).unsqueeze(1).to(dev)
                    model(xb)
                    fp[i:i + 128] = PL.pool_windows(pg.buf, winP).half().cpu().numpy()
                    fs[i:i + 128] = PL.pool_windows(pg.buf, winS).half().cpu().numpy()
            return torch.from_numpy(fp), torch.from_numpy(fs)

        FtrP, FtrS = feats(Xtr)
        FvaP, FvaS = feats(Xva)
        FteP, FteS = feats(Xte)
        pg.close()
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        t_feat = time.time() - t1
        nseen = t["seen"]
        seen = np.zeros(H, bool)
        seen[:2 * nseen] = True

        for ro, Ftr, Fva, Fte in (("P", FtrP, FvaP, FteP), ("S", FtrS, FvaS, FteS)):
            if (t["rid"], ro) in dp:
                continue
            t2 = time.time()
            # Same initial head values for P and S — probe_local.train seeds only the training order
            # and dropout; the initialization in make_heads is outside the seed. Fixing it with the same seed
            # makes the difference between the two readouts come only from the read range (same measure as
            # the probe_local_recover raw fix)
            torch.manual_seed(t["seed"])
            if dev.type == "cuda":
                torch.cuda.manual_seed_all(t["seed"])
            heads = make_heads(C, ncls, dev, Ytr)
            best, eps = PL.train(heads, Ftr, torch.from_numpy(Ytr), Fva,
                                 torch.from_numpy(Yva), dev, t["seed"])
            acc, info = PL.evaluate(heads, Fte, Yte, logpr, dev)
            heads.eval()
            lp = []
            with torch.no_grad():
                for i in range(0, len(Fte), 1024):
                    o = PL.run_heads(heads, Fte[i:i + 1024].to(dev).float())
                    lp.append(torch.stack([torch.log_softmax(x, 1) for x in o], 1)
                              .cpu().numpy().astype(np.float32))
            np.savez_compressed(os.path.join(OUT_P, f"logp_{t['rid']}_{ro}.npz"),
                                logp=np.concatenate(lp), te_idx=te_idx, arch=t["arch"],
                                n_pairs=t["n"], seed=t["seed"],
                                ckpt=(t["ctrl"] or "main"), readout=ro, heads=H)
            tm = (time.time() - t2 + t_feat / 2) / 60
            print(f"    {ro}: {nseen} learned pairs acc {acc[seen].mean():.3f}% "
                  f"(min {acc[seen].min():.2f}%) info {info[seen].mean():.3f}bit "
                  f"| early stopping ep {eps} (val {best:.2f}%) | {tm:.2f} min")
            append_csv(os.path.join(OUT_P, "readout_probe.csv"), PROBE_FIELDS, [dict(
                run_id=t["rid"], readout=ro, arch=t["arch"], n_pairs=t["n"], seed=t["seed"],
                ckpt=t["ctrl"] or "main", feat_ch=C, feat_len=Tf, stride=round(r, 1),
                n_seen=nseen, probe_epochs=eps, best_val_acc=round(best, 3),
                acc_mean=round(float(acc[seen].mean()), 3),
                acc_min=round(float(acc[seen].min()), 3),
                info_mean=round(float(info[seen].mean()), 4),
                info_min=round(float(info[seen].min()), 4),
                n_boundary=nb, time_min=round(tm, 2))])
            append_csv(os.path.join(OUT_P, "readout_pairs.csv"), PAIRH_FIELDS, [dict(
                run_id=t["rid"], readout=ro, arch=t["arch"], n_pairs=t["n"],
                seed=t["seed"], ckpt=t["ctrl"] or "main", head=h, pair=h // 2 + 1,
                role=f"r{h % 2}", seen=int(seen[h]), acc=round(float(acc[h]), 3),
                majority_acc=round(float(maj[h]), 3), info_bits=round(float(info[h]), 5))
                for h in range(H)])
            del heads
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del FtrP, FtrS, FvaP, FvaS, FteP, FteS
    print(f"\nstage1 done — {OUT_P}")


# ================= stage2: key recovery =================
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
                q_med=(float(np.median(ok)) if ok else None), n_cens=len(cens), slow=slow,
                sr5=(float(np.mean(sr)) if sr else None),
                ge=(float(np.mean(ge_l)) if ge_l else None))


def _r(x, nd):
    return "" if x is None or not np.isfinite(x) else round(float(x), nd)


def write_rows(path, fields, rows, key):
    """Overwrites. If the column layout changed, moves the old file to .bak and writes a new one."""
    keep = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if list(rd.fieldnames or []) != list(fields):
                os.replace(path, path + ".bak")
            else:
                mine = {tuple(str(r.get(k, "")) for k in key) for r in rows}
                keep = [r for r in rd if tuple(str(r.get(k, "")) for k in key) not in mine]
    if os.path.exists(path):
        os.remove(path)
    E.append_rows(path, fields, keep + rows)
    print(f"    saved {os.path.basename(path)}  {len(keep)+len(rows)} rows")


def stage2(with_rec):
    os.makedirs(OUT_K, exist_ok=True)
    verify = []
    tg = [t for t in targets(with_rec) if t["recover"] and t["exists"]]
    jobs = []
    for t in tg:
        for ro in ("P", "S"):
            f = os.path.join(OUT_P, f"logp_{t['rid']}_{ro}.npz")
            if os.path.exists(f) and not os.path.exists(
                    os.path.join(OUT_K, f"res_{t['rid']}_{ro}.npz")):
                jobs.append((t, ro, f))
    if not jobs:
        print("stage2: nothing to run")
        return
    print(f"stage2: key recovery {len(jobs)} runs")
    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = E.CHUNK or (64 if dev.type == "cuda" and
                        torch.cuda.get_device_properties(0).total_memory / 2**30 >= 40
                        else 16)
    src = os.path.join(FINE_DIR, "eval_indices.npz")
    dst = os.path.join(OUT_K, "eval_indices.npz")
    if not os.path.exists(dst):
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

    for j, (t, ro, f) in enumerate(jobs):
        z = np.load(f)
        lp = np.asarray(z["logp"], np.float32)
        if not np.array_equal(np.asarray(z["te_idx"]), a["idx"]):
            print(f"[{t['rid']}_{ro}] **skipped — te_idx mismatch**")
            verify.append(f"{t['rid']}_{ro}: te_idx mismatch")
            continue
        pl_ = list(range(t["seen"]))
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
                left = sum(len(range(x[0]["seen"])) for x in jobs[j:])
                print(f"[{t['rid']}_{ro}] {len(pl_)} pairs — {el:.1f} s per pair, "
                      f"{len(jobs)-j} runs left, about {el*left/3600:.1f} h")
        np.savez(os.path.join(OUT_K, f"res_{t['rid']}_{ro}.npz"),
                 arch=t["arch"], n_pairs=t["n"], seed=t["seed"], ckpt=t["ctrl"] or "main",
                 readout=ro, eval_sec=time.time() - t0, **{f"p{p}": v for p, v in rk.items()})
        st = run_stats(rk, pl_)
        print(f"    Q*max {st['q_max'] if st['q_max'] is not None else '>50'}  "
              f"Q*mean {_r(st['q_mean'], 2)}  SR@5 {_r(st['sr5'], 3)}  "
              f"censor {st['n_cens']}  ({(time.time()-t0)/60:.1f} min)")
    E.OUT_DIR = MAIN
    with open(os.path.join(OUT_K, "verify.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(verify) + "\n")


# ================= aggregate: 7-4 =================
def _csv(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def orig_stats(t):
    """Original head: acc/info from eval_heads.csv, Q* from fine eval_runs.csv."""
    erid = E.run_id_of(t["arch"], t["n"], t["seed"])
    if t["ctrl"]:
        hp = [os.path.join(BASE + "_qfine_d", "eval_heads.csv"),
              os.path.join(BASE + "_ctrl_d", "eval_heads.csv")]
        rp = os.path.join(BASE + "_qfine_d", "eval_runs.csv")
    else:
        hp = [os.path.join(FINE_DIR, "eval_heads.csv"), os.path.join(BASE, "eval_heads.csv")]
        rp = os.path.join(FINE_DIR, "eval_runs.csv")
    acc = info = None
    for p in hp:
        rs = [r for r in _csv(p) if r["run_id"] == erid
              and int(r["head"]) < 2 * t["seen"]]
        if rs:
            acc = np.array([float(r["acc"]) for r in rs])
            info = np.array([float(r["info_bits"]) for r in rs])
            break
    q = next((r for r in _csv(rp) if r["run_id"] == erid and r.get("eval_mode") == "sum"),
             None)
    return dict(acc_mean=_r(acc.mean() if acc is not None else None, 3),
                acc_min=_r(acc.min() if acc is not None else None, 3),
                info_mean=_r(info.mean() if info is not None else None, 4),
                info_min=_r(info.min() if info is not None else None, 4),
                probe_epochs="",
                q_star_max=(q["q_star_max"] if q else ""),
                q_star_mean=(q["q_star_mean"] if q else ""),
                sr_q5=(q.get("sr_q5", "") if q else ""),
                n_pairs_censored=(q["n_pairs_censored"] if q else ""),
                slowest_pair=((int(q["slowest_pair_idx"]) + 1)
                              if q and q.get("slowest_pair_idx", "") != "" else ""),
                source="eval_heads/eval_runs (fine)")


def local4_stats(t):
    """Local 4 positions: looks in probe_local_ext (A/B) -> probe_local (N9/19/29) in that order,
    and Q* in localrec (11-13) -> E2 in that order."""
    ck = t["ctrl"] or "main"
    out = dict(acc_mean="", acc_min="", info_mean="", info_min="", probe_epochs="",
               q_star_max="", q_star_mean="", sr_q5="", n_pairs_censored="",
               slowest_pair="", source="")
    ext = os.path.join(BASE, "probe_local_ext")
    m = next((r for r in _csv(os.path.join(ext, "local_models.csv"))
              if r["arch"] == t["arch"] and int(r["src_n_pairs"]) == t["n"]
              and int(r["seed"]) == t["seed"] and r.get("ckpt", "main") == ck), None)
    tag = None
    if m:
        tag = m["run_id"].split("_", 1)[0]
        prs = [r for r in _csv(os.path.join(ext, "local_pairs.csv"))
               if r["run_id"] == m["run_id"] and r.get("seen", "1") in ("1", "True")]
        out.update(acc_mean=m["acc_seen"], info_mean=m["info_seen"],
                   probe_epochs=m["probe_epochs"],
                   acc_min=(_r(min(float(r["acc"]) for r in prs), 3) if prs else ""),
                   info_min=(_r(min(float(r["info_bits"]) for r in prs), 4) if prs else ""),
                   source=f"probe_local_ext {m['run_id']}")
    elif ck == "main":
        pl_dir = os.path.join(BASE, "probe_local")
        m = next((r for r in _csv(os.path.join(pl_dir, "local_models.csv"))
                  if r["arch"] == t["arch"] and int(r["src_n_pairs"]) == t["n"]
                  and int(r["seed"]) == t["seed"]), None)
        if m:
            prs = [r for r in _csv(os.path.join(pl_dir, "local_pairs.csv"))
                   if r["arch"] == t["arch"] and int(r["src_n_pairs"]) == t["n"]
                   and int(r["seed"]) == t["seed"] and r.get("seen", "1") in ("1", "True")]
            accs = [float(r[k]) for r in prs for k in ("acc_r0", "acc_r1") if r.get(k)]
            infs = [float(r[k]) for r in prs for k in ("info_r0", "info_r1") if r.get(k)]
            out.update(acc_mean=m["acc_seen"], info_mean=m["info_seen"],
                       probe_epochs=m["probe_epochs"],
                       acc_min=(_r(min(accs), 3) if accs else ""),
                       info_min=(_r(min(infs), 4) if infs else ""),
                       source="probe_local (11-09)")
    # Q*: localrec -> E2
    q = None
    if tag:
        q = next((r for r in _csv(os.path.join(BASE + "_qfine_localrec", "eval_runs.csv"))
                  if r["run_id"] == f"{tag}_local" and r.get("eval_mode") == "sum"), None)
        if q:
            out.update(q_star_max=q["q_star_max"], q_star_mean=q["q_star_mean"],
                       sr_q5=q.get("sr_q5", ""), n_pairs_censored=q["n_pairs_censored"],
                       slowest_pair=((int(q["slowest_pair_idx"]) + 1)
                                     if q.get("slowest_pair_idx", "") != "" else ""))
            out["source"] += f" + localrec {tag}_local"
    if q is None and ck == "main":
        q = next((r for r in _csv(os.path.join(FINE_DIR + "_e2local", "eval_runs.csv"))
                  if r["heads_src"] == "local" and r["arch"] == t["arch"]
                  and int(r["n_pairs"]) == t["n"] and int(r["seed"]) == t["seed"]), None)
        if q:
            out.update(q_star_max=q["q_star_max"], q_star_mean=q["q_star_mean"],
                       sr_q5=q.get("sr_q5", ""), n_pairs_censored=q["n_pairs_censored"],
                       slowest_pair=q.get("slowest_pair", ""))
            out["source"] += " + E2"
    return out


def aggregate(with_rec):
    os.makedirs(OUT_K, exist_ok=True)
    probe = {(r["run_id"], r["readout"]): r
             for r in _csv(os.path.join(OUT_P, "readout_probe.csv"))}
    rm, runs, pq = [], [], []
    for t in targets(with_rec):
        if (t["rid"], "P") not in probe:
            continue
        base = dict(run_id=t["rid"], arch=t["arch"], n_pairs=t["n"], seed=t["seed"],
                    ckpt=t["ctrl"] or "main")
        rm.append(dict(base, readout="orig", **orig_stats(t)))
        for ro in ("P", "S"):
            pr = probe.get((t["rid"], ro))
            if not pr:
                continue
            row = dict(base, readout=ro, acc_mean=pr["acc_mean"], acc_min=pr["acc_min"],
                       info_mean=pr["info_mean"], info_min=pr["info_min"],
                       probe_epochs=pr["probe_epochs"], q_star_max="", q_star_mean="",
                       sr_q5="", n_pairs_censored="", slowest_pair="",
                       source="E5 probe")
            f = os.path.join(OUT_K, f"res_{t['rid']}_{ro}.npz")
            if os.path.exists(f):
                z = np.load(f, allow_pickle=False)
                rk = {int(k[1:]): z[k] for k in z.files if k.startswith("p")}
                pl_ = sorted(rk)
                st = run_stats(rk, pl_)
                row.update(q_star_max=("" if st["q_max"] is None else st["q_max"]),
                           q_star_mean=_r(st["q_mean"], 2), sr_q5=_r(st["sr5"], 4),
                           n_pairs_censored=st["n_cens"],
                           slowest_pair=("" if st["slow"] is None else st["slow"] + 1),
                           source="E5 probe + E5 key recovery")
                runs.append(dict(base, readout=ro, n_pairs_eval=len(pl_),
                                 q_star_max=row["q_star_max"], q_star_mean=row["q_star_mean"],
                                 q_star_median=_r(st["q_med"], 1),
                                 n_pairs_censored=st["n_cens"],
                                 slowest_pair_idx=("" if st["slow"] is None else st["slow"]),
                                 slowest_pair=row["slowest_pair"], sr_q5=row["sr_q5"],
                                 ge_log2_qmax=_r(st["ge"], 3), n_repeat=rk[pl_[0]].shape[0],
                                 eval_time_sec=_r(float(z["eval_sec"]), 1)))
                for p in pl_:
                    q = st["qs"][p]
                    pq.append(dict(base, readout=ro, pair_idx=p, pair=p + 1,
                                   q_star=("" if q is None else q), censored=int(q is None),
                                   ge_log2_qmax=_r(float(np.log2(rk[p].mean(0)[-1])), 3)))
            rm.append(row)
        rm.append(dict(base, readout="local4", **local4_stats(t)))
    if not rm:
        print("readout_probe.csv does not exist — run stage1 first")
        return
    print("")
    write_rows(os.path.join(OUT_K, "readout_models.csv"), RM_FIELDS, rm,
               ("run_id", "readout"))
    if runs:
        write_rows(os.path.join(OUT_K, "eval_runs.csv"), RUN_FIELDS, runs,
                   ("run_id", "readout"))
        write_rows(os.path.join(OUT_K, "pair_qstar.csv"), PQ_FIELDS, pq,
                   ("run_id", "readout", "pair_idx"))
    summary(rm)


def summary(rm):
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    by = {}
    for r in rm:
        by.setdefault(r["run_id"], {})[r["readout"]] = r
    f3 = lambda r: ("—" if not r else
                    f"{_fmt(r['acc_mean'], 1)} / {_fmt(r['info_mean'], 2)} / "
                    f"{r['q_star_max'] if r['q_star_max'] != '' else ('–' if r['q_star_mean'] == '' else '>50')}")
    say("=" * 112)
    say("E5 summary — request 7-4(4).  learned pairs, 2000 test traces — accuracy % / information bit / Q*max (fine)")
    say("=" * 112)
    say(f"{'run':<26}{'original head':>20}{'P (pooled refit)':>22}{'S (one position)':>20}"
        f"{'local 4 positions (existing)':>22}")
    say("-" * 112)
    for rid in by:
        d = by[rid]
        say(f"{rid:<26}{f3(d.get('orig')):>20}{f3(d.get('P')):>22}{f3(d.get('S')):>20}"
            f"{f3(d.get('local4')):>22}")
    say("-" * 112)
    say("  Q*max '–' = not a key recovery target (N9/N19), '>50' = has censored pairs")

    say("")
    say("Numbers used for the 7-5 decision — the four N39/N49 models (criteria fixed before seeing the results)")
    say(f"  {'run':<24}{'P-orig acc':>12}{'S acc':>9}{'S-P acc':>10}"
        f"{'local4-S':>10}{'P-orig info':>13}{'S-P info':>10}")
    for rid in by:
        d = by[rid]
        if not (d.get("P") and d.get("orig")) or d["P"]["n_pairs"] not in (39, 49, "39", "49"):
            continue
        if d["P"]["ckpt"] != "main":
            continue
        g = lambda ro, k: (float(d[ro][k]) if d.get(ro) and d[ro][k] not in ("", None)
                           else np.nan)
        say(f"  {rid:<24}{g('P','acc_mean')-g('orig','acc_mean'):>+12.2f}"
            f"{g('S','acc_mean'):>9.2f}{g('S','acc_mean')-g('P','acc_mean'):>+10.2f}"
            f"{g('local4','acc_mean')-g('S','acc_mean'):>+10.2f}"
            f"{g('P','info_mean')-g('orig','info_mean'):>+13.3f}"
            f"{g('S','info_mean')-g('P','info_mean'):>+10.3f}")
    say("  criteria: |P-orig acc| ≤ 5%p and S ≥ 90% -> cell 1 / P-orig ≥ 15%p -> cell 2 /"
        " S≈P and only local4 higher -> cell 3")
    say("")
    say("N ≤ 29 — whether the information of P equals S and local (7-5 cell 4)")
    for rid in by:
        d = by[rid]
        if not d.get("P") or int(d["P"]["n_pairs"]) > 29 or d["P"]["ckpt"] != "main":
            continue
        g = lambda ro: (float(d[ro]["info_mean"]) if d.get(ro) and d[ro]["info_mean"] != ""
                        else np.nan)
        say(f"  {rid:<24} orig {g('orig'):.3f}  P {g('P'):.3f}  S {g('S'):.3f}  "
            f"local4 {g('local4'):.3f}  bit")
    say("=" * 112)

    # difference from the existing pooled refit (probe_heads) — 7-3 "if different, state the difference in conditions"
    ph = _csv(os.path.join(BASE, "probe_heads", "probe_models.csv"))
    if ph:
        say("")
        say("Reference — existing pooled refit (probe_heads, 11-09) vs. this P")
        for rid in by:
            d = by[rid]
            if not d.get("P") or d["P"]["ckpt"] != "main":
                continue
            old = next((r for r in ph if r.get("arch") == d["P"]["arch"]
                        and str(r.get("src_n_pairs", r.get("n_pairs"))) == str(d["P"]["n_pairs"])
                        and str(r.get("seed")) == str(d["P"]["seed"])), None)
            if old:
                say(f"  {rid:<24} probe_heads acc {old.get('acc_seen','')} / info "
                    f"{old.get('info_seen','')}   this P {d['P']['acc_mean']} / "
                    f"{d['P']['info_mean']}")
    with open(os.path.join(OUT_K, "summary_7_4.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved: {os.path.join(OUT_K, 'summary_7_4.txt')}")
    print(f"raw data: {OUT_P}\n        {OUT_K}")


def _fmt(v, nd):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def main():
    flags = {a for a in _USER if a.startswith("--")}
    rec = "--rec" in flags
    if _PLAN or not (flags - _OPTS):
        plan(rec)
        return
    if "--stage1" in flags:
        if _STAGE1_ALONE:
            stage1(rec)
        else:
            import subprocess
            cmd = ([sys.executable, os.path.abspath(__file__), "--stage1"]
                   + (["--rec"] if rec else []) + (["--bn"] if _BN else [])
                   + (["--only", ",".join(_ONLY)] if _ONLY else []))
            print(f"launching stage1 separately without fine\n")
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            rc = subprocess.call(cmd, cwd=HERE, env=env)
            if rc != 0:
                raise SystemExit(f"stage1 failed (exit code {rc}) — not running stage2")
    if "--stage2" in flags:
        stage2(rec)
    if "--aggregate" in flags:
        aggregate(rec)


if __name__ == "__main__":
    main()
