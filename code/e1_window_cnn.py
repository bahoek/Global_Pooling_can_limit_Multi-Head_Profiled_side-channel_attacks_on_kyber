#!/usr/bin/env python3
r"""
e1_window_cnn — E1: a CNN trained directly on a window for each pair

Request document: `claude/TIFS additional experiment request before resubmission … (11-17)` section 3.

    python e1_window_cnn.py --plan                     # check windows, model shape, targets. No computation  <- run first
    python e1_window_cnn.py --train                    # W-RF, seed 0, train 39 pairs (+ per-pair test logp)
    python e1_window_cnn.py --train --max-pairs 1      # time one pair
    python e1_window_cnn.py --recover                  # gather the 39-pair logp and run key recovery (fine, canonical form)
    python e1_window_cnn.py --table                    # 3-3(4) comparison table + 3-4 decision numbers
    python e1_window_cnn.py --train --recover --table  # all at once
    --half H    window half-width (default 659 = W-RF). --seed S (default 0)

Question (3-1)
--------------
Given the pair locations, in how many traces does a network trained directly on just those windows recover the 39 pairs?

Design (3-2) — exactly as in the request
----------------------------------------
  One model per pair, 2 heads (r0, r1). 39 pairs x 1 model.
  Input: a window cut from the raw trace. Center = midpoint of that pair's r0/r1 NICV peaks (measured, nicv_pairs.csv)
  W-RF: center ±659 (1,318 samples) — same as the local head range counted from the model by receptive_field.py
        (RF 1,079 + 3 x 80 = 1,319 samples, 4 positions)
  Normalization: the per-position μ/σ from the main training norm.npz, cut to the window span as is (not refit per window)
  Architecture: train_nscan's ResNet-18-1D as is, only the input length differs (2 heads).
        input 1,318 -> last feature map 17 cells (4 cells for 320)
  Training: **calls train_nscan.run_single as is** — 64,000 / 16,000, 100 epochs, AdamW,
        cosine T_max 100, patience 15, AMP, grad clip 5, HW16, head BN+Dropout+Linear,
        bias = log prior, checkpoint with best validation accuracy. Seed 0.
  Test: log-softmax on the main evaluation 2,000 traces (eval_indices.npz) saved per pair -> the 39 pairs gathered into
        logp (2000 x 78 x 17, head order 2p=r0, 2p+1=r1 = same as the local readout npz)
        -> the same key recovery path as E2 (canonical-form candidates, fine Q=1..50, 20 repeats)

W320 is not run (11-21 measurement)
-----------------------------------
The r0->r1 peak spacing is 320~348 samples (mean 337), so both peaks do not fit in a midpoint ±160 window
(in 31 of 39 pairs one peak is outside the window; in all 39 pairs a peak is within 20 samples of the window edge). This falls under
request 3-2 "if some pairs do not fit, inform the paper chat before running", so we wait for the decision.
It can be run with --half 160, but --plan warns.

**Does not touch the main training folder.** run_single writes the checkpoint to
`SAVE_DIR/{arch}_N{n}_s{seed}.pth`, but E1 uses N=1 for each pair, so all 39 would get the same name.
Therefore `train_nscan.SAVE_DIR` is changed to `tifs/model_e1/W<len>/p<NN>/` for each pair before calling.
Does not modify any existing file.

Output
------
  tifs/model_e1/W<len>/
    p01 .. p39/resnet_N1_s<seed>.pth, heads_resnet_N1_s<seed>.npz   checkpoints (to be released)
    window_models.csv   run_id, width, pair_idx, pair, seed, acc_r0, acc_r1, info_r0, info_r1,
                        best_epoch, epochs_run, time_min, … (3-3(1))
    train_gradnorm.csv  per epoch (format written by run_single)
    epoch_log.csv       validation accuracy per pair x epoch (3-3(5) per-pair epoch curves)
    logp_pair_<NN>_s<seed>.npz   per-pair test log-softmax (2000 x 2 x 17)
    logp_E1_W<len>_s<seed>.npz   39 pairs combined (2000 x 78 x 17)
  tifs/eval_qfine_e1/
    eval_runs.csv, pair_qstar.csv, res_*.npz, verify.txt, table_3_3.txt
"""

import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_USER = sys.argv[1:]
sys.argv = [sys.argv[0], "fine"]

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
import eval_nscan as E                                              # noqa: E402

T = E.import_train()
FINE_DIR = E.OUT_DIR                                  # …/eval_qfine
BASE = FINE_DIR[:-len("_qfine")]                      # …/eval
ROOT = os.path.dirname(BASE)                          # …/tifs
NICV = os.path.join(BASE, "nicv", "nicv_pairs.csv")
LEAK = E.LEAK_PAIRS


def _arg(name, default, cast=int):
    if name in _USER and _USER.index(name) + 1 < len(_USER):
        return cast(_USER[_USER.index(name) + 1])
    return default


HALF = _arg("--half", 659)
SEED = _arg("--seed", 0)
MAXP = _arg("--max-pairs", 0)
WLEN = 2 * HALF
OUT_M = os.path.join(ROOT, "model_e1", f"W{WLEN}")
OUT_K = FINE_DIR + "_e1"
TAG = f"E1_W{WLEN}_s{SEED}"

WM_FIELDS = ["run_id", "width", "half", "pair_idx", "pair", "seed", "center", "lo", "hi",
             "shift",
             "peak_r0", "peak_r1", "acc_r0", "acc_r1", "info_r0", "info_r1",
             "val_acc", "best_epoch", "epochs_run", "time_min", "params_m"]
EP_FIELDS = ["run_id", "pair_idx", "pair", "epoch", "val_acc"]
RUN_FIELDS = ["run_id", "width", "seed", "n_pairs_eval", "q_star_max", "q_star_mean",
              "q_star_median", "n_pairs_censored", "slowest_pair_idx", "slowest_pair"] + \
             [f"sr_q{q}" for q in range(1, 51)] + \
             [f"ge_log2_q{q}" for q in range(1, 51)] + ["n_repeat", "eval_time_sec"]
PQ_FIELDS = ["run_id", "width", "seed", "pair_idx", "pair", "q_star", "censored",
             "ge_log2_qmax"]


# ================= windows =================
def read_peaks():
    out = {}
    with open(NICV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = int(r["pair"])
            if p < LEAK:
                out[p] = (int(r["peak_sample_r0"]), int(r["peak_sample_r1"]))
    return out


def windows(half):
    pk = read_peaks()
    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    lo0, hi0 = [int(v) for v in nz["crop"]]
    rows = []
    for p in range(LEAK):
        a, b = pk[p]
        c = int(round((a + b) / 2))
        lo, hi = c - half, c + half
        # A window that runs past the trace end keeps its length and is pushed inward (boundary case not in the request.
        # 11-21: in W1318 only pair 39 is affected — pushed by 291 samples to [38682, 40000), both peaks inside)
        shift = 0
        if lo < lo0:
            shift = lo0 - lo
        elif hi > hi0:
            shift = hi0 - hi
        lo, hi = lo + shift, hi + shift
        rows.append(dict(p=p, r0=a, r1=b, c=c, lo=lo, hi=hi, shift=shift,
                         in_bounds=(lo >= lo0 and hi <= hi0),
                         r0_in=(lo <= a < hi), r1_in=(lo <= b < hi),
                         margin=min(a - lo, hi - 1 - b)))
    return rows, (lo0, hi0)


def feat_len(L):
    m = T.build_model("resnet", 2).eval()
    with torch.no_grad():
        pg = []
        h = m.body.register_forward_hook(lambda mod, i, o: pg.append(tuple(o.shape)))
        m(torch.zeros(2, 1, L))
        h.remove()
    return pg[-1], sum(x.numel() for x in m.parameters())


# ================= --plan =================
def plan():
    print("=" * 84)
    print("E1 plan — no computation, no writes")
    print("=" * 84)
    print(f"  train_nscan  EPOCHS {T.EPOCHS}  PATIENCE {T.PATIENCE}  MIN_DELTA {T.MIN_DELTA}"
          f"  LR {T.LR}  WD {T.WD}  BATCH {T.BATCH}  GRAD_CLIP {T.GRAD_CLIP}"
          f"  DROPOUT {T.DROPOUT}  RESNET_VARIANT {T.RESNET_VARIANT}")
    print(f"  NICV         {NICV}")
    print(f"  model output  {OUT_M}")
    print(f"  key recovery output  {OUT_K}")
    for half in sorted({HALF, 160, 659}):
        rows, (lo0, hi0) = windows(half)
        bad = [r["p"] + 1 for r in rows if not (r["r0_in"] and r["r1_in"])]
        edge = [r["p"] + 1 for r in rows if r["margin"] < 20]
        oob = [r["p"] + 1 for r in rows if not r["in_bounds"]]
        sh = [(r["p"] + 1, r["shift"]) for r in rows if r["shift"]]
        (B, C, Tf), npar = feat_len(2 * half)
        mark = "  <- this run" if half == HALF else ""
        print(f"\n  W{2*half} (midpoint ±{half}){mark}")
        print(f"    feature map {C}ch x {Tf} cells, parameters {npar/1e6:.2f}M")
        print(f"    pairs whose two peaks are not both inside the window {len(bad)}/39 {bad[:10]}"
              f"{' …' if len(bad) > 10 else ''}")
        print(f"    pairs with a peak within 20 samples of the window edge {len(edge)}/39")
        print(f"    pairs whose window runs outside the trace {len(oob)}/39 (trace [{lo0},{hi0}))")
        if sh:
            print(f"    windows pushed inward because of the trace end (pair, samples): {sh}")
        if bad:
            print("    **request 3-2: if some pairs do not fit, inform the paper chat before running**")
    rows, _ = windows(HALF)
    print(f"\n  W{WLEN} per-pair windows (first 5)")
    for r in rows[:5]:
        print(f"    pair {r['p']+1:>2}: r0 {r['r0']} r1 {r['r1']} (spacing {r['r1']-r['r0']})  "
              f"center {r['c']}  window [{r['lo']}, {r['hi']})  margin {r['margin']}")
    done = sum(os.path.exists(os.path.join(OUT_M, f"logp_pair_{p+1:02d}_s{SEED}.npz"))
               for p in range(LEAK))
    print(f"\n  seed {SEED}: {done} of 39 pairs done")
    print("  time: estimate by timing one pair with --train --max-pairs 1 (request estimate: all of E1 8~12 GPU hours)")
    print("=" * 84)


# ================= --train =================
def train():
    rows, (lo0, hi0) = windows(HALF)
    bad = [r["p"] + 1 for r in rows if not (r["r0_in"] and r["r1_in"])]
    if bad:
        print(f"**warning: in W{WLEN}, {len(bad)} pairs do not have both peaks inside {bad[:10]}**")
        print("  per request 3-2, not running without a paper chat decision. Stopping.")
        raise SystemExit(1)
    oob = [r["p"] + 1 for r in rows if not r["in_bounds"]]
    if oob:
        raise SystemExit(f"pairs whose window runs outside the trace: {oob}")
    os.makedirs(OUT_M, exist_ok=True)
    todo = [r for r in rows
            if not os.path.exists(os.path.join(OUT_M, f"logp_pair_{r['p']+1:02d}_s{SEED}.npz"))]
    if MAXP:
        todo = todo[:MAXP]
    if not todo:
        print("train: all 39 pairs are done")
        return
    print(f"train: W{WLEN} seed {SEED}, {len(todo)} pairs")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    mu = np.asarray(nz["mu"], np.float32).reshape(-1)
    sd = np.asarray(nz["sd"], np.float32).reshape(-1)

    def load(name):
        d = np.load(os.path.join(E.SPLIT_DIR, f"kyber_{name}_crop.npz"))
        return (d["traces"], np.asarray(d["ciphertexts"], np.uint8),
                np.asarray(d["labels"], np.int64))

    t0 = time.time()
    Xtr_raw, ct_tr, lab_tr = load("train")
    Xva_raw, ct_va, _ = load("valid")
    Xte_raw, ct_te, _ = load("test")
    sk = lab_tr[0]
    h_max = T.heads_of(max(T.N_LIST))
    Ytr = T.make_labels(ct_tr, sk, "train", h_max)
    Yva = T.make_labels(ct_va, sk, "valid", h_max)
    Yte_all = T.make_labels(ct_te, sk, "test", h_max)
    ncls = T.NUM_CLASSES
    prior = np.stack([(np.bincount(Yte_all[:, j], minlength=ncls) + 1.0) /
                      (len(Yte_all) + ncls) for j in range(2 * LEAK)])
    te_idx = np.load(os.path.join(FINE_DIR, "eval_indices.npz"))["idx"]
    Yte = Yte_all[te_idx]
    print(f"data {time.time()-t0:.0f} s | train {Xtr_raw.shape} valid {Xva_raw.shape} "
          f"test {Xte_raw.shape} -> eval {len(te_idx)} traces | device {dev}")

    def crop(X, lo, hi, idx=None):
        x = X[:, lo - lo0:hi - lo0] if idx is None else X[idx, lo - lo0:hi - lo0]
        return ((x.astype(np.float32) - mu[lo - lo0:hi - lo0]) /
                sd[lo - lo0:hi - lo0]).astype(np.float32)

    gn_path = os.path.join(OUT_M, "train_gradnorm.csv")
    gn_new = not os.path.exists(gn_path)
    gn_f = open(gn_path, "a", newline="", encoding="utf-8")
    gn_w = csv.DictWriter(gn_f, fieldnames=T.GN_FIELDS)
    if gn_new:
        gn_w.writeheader()

    base_eval = T.evaluate
    for k, r in enumerate(todo):
        p = r["p"]
        rid = f"E1_W{WLEN}_p{p+1:02d}_s{SEED}"
        sub = os.path.join(OUT_M, f"p{p+1:02d}")
        os.makedirs(sub, exist_ok=True)
        T.SAVE_DIR = sub                                  # so as not to touch the main training folder
        T.CROP_LO, T.CROP_HI = r["lo"], r["hi"]           # record the window in the checkpoint
        hist = []

        def ev(model, loader, dev_, nt):                  # record validation accuracy per epoch
            out = base_eval(model, loader, dev_, nt)
            hist.append(float(np.mean(out[1])))
            return out
        T.evaluate = ev

        Xtr = crop(Xtr_raw, r["lo"], r["hi"])
        Xva = crop(Xva_raw, r["lo"], r["hi"])
        print(f"\n[{k+1}/{len(todo)}] {rid}  window [{r['lo']}, {r['hi']})  "
              f"r0 {r['r0']} r1 {r['r1']}  margin {r['margin']}")
        t1 = time.time()
        row = T.run_single("resnet", 1, SEED, Xtr, Ytr[:, 2 * p:2 * p + 2],
                           Xva, Yva[:, 2 * p:2 * p + 2], dev, rid, gn_w)
        gn_f.flush()
        T.evaluate = base_eval
        tm = (time.time() - t1) / 60
        best_ep = int(np.argmax(hist)) + 1 if hist else ""

        # test 2000 traces log-softmax
        ck = torch.load(os.path.join(sub, f"resnet_N1_s{SEED}.pth"), map_location="cpu",
                        weights_only=False)
        model = T.build_model("resnet", 2).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        Xte = crop(Xte_raw, r["lo"], r["hi"], te_idx)
        lp = []
        with torch.no_grad():
            for i in range(0, len(Xte), 256):
                xb = torch.from_numpy(Xte[i:i + 256]).unsqueeze(1).to(dev)
                lp.append(torch.stack([torch.log_softmax(o.float(), 1) for o in model(xb)],
                                      1).cpu().numpy())
        lp = np.concatenate(lp)                                # (2000, 2, 17)
        y = Yte[:, 2 * p:2 * p + 2]
        acc = [(lp[:, j].argmax(1) == y[:, j]).mean() * 100 for j in range(2)]
        ll = np.take_along_axis(lp, y[:, :, None], 2)[..., 0]
        lpr = np.log(prior[2 * p:2 * p + 2])
        info = [(ll[:, j] - lpr[j][y[:, j]]).mean() / np.log(2) for j in range(2)]
        np.savez_compressed(os.path.join(OUT_M, f"logp_pair_{p+1:02d}_s{SEED}.npz"),
                            logp=lp.astype(np.float32), te_idx=te_idx, pair_idx=p,
                            heads=np.array([2 * p, 2 * p + 1]), window=np.array([r["lo"], r["hi"]]))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        E.append_rows(os.path.join(OUT_M, "window_models.csv"), WM_FIELDS, [dict(
            run_id=rid, width=WLEN, half=HALF, pair_idx=p, pair=p + 1, seed=SEED,
            center=r["c"], lo=r["lo"], hi=r["hi"], shift=r["shift"],
            peak_r0=r["r0"], peak_r1=r["r1"],
            acc_r0=round(acc[0], 3), acc_r1=round(acc[1], 3),
            info_r0=round(float(info[0]), 4), info_r1=round(float(info[1]), 4),
            val_acc=row["val_acc_allheads"], best_epoch=best_ep, epochs_run=row["epochs_run"],
            time_min=round(tm, 2), params_m=row["params_m"])])
        E.append_rows(os.path.join(OUT_M, "epoch_log.csv"), EP_FIELDS,
                      [dict(run_id=rid, pair_idx=p, pair=p + 1, epoch=e + 1,
                            val_acc=round(v, 3)) for e, v in enumerate(hist)])
        left = len(todo) - k - 1
        print(f"  -> test acc r0 {acc[0]:.2f}% r1 {acc[1]:.2f}%  info {info[0]:.3f} / "
              f"{info[1]:.3f} bit | best ep {best_ep}/{row['epochs_run']} | {tm:.1f} min"
              f" | {left} pairs left, about {tm*left/60:.1f} h")
    gn_f.close()


# ================= --recover =================
def assemble():
    parts = []
    for p in range(LEAK):
        f = os.path.join(OUT_M, f"logp_pair_{p+1:02d}_s{SEED}.npz")
        if not os.path.exists(f):
            return None, f"pair {p+1} logp missing — --train must finish first"
        parts.append(np.load(f))
    te = parts[0]["te_idx"]
    for z in parts:
        if not np.array_equal(z["te_idx"], te):
            return None, "per-pair te_idx differ from each other"
    lp = np.concatenate([z["logp"] for z in parts], 1)       # (2000, 78, 17), order 2p, 2p+1
    out = os.path.join(OUT_M, f"logp_{TAG}.npz")
    np.savez_compressed(out, logp=lp, te_idx=te, arch="resnet", n_pairs=LEAK, seed=SEED,
                        heads=2 * LEAK, width=WLEN)
    return out, None


def run_stats(by_pair, pairs):
    qs, ge_l, sr = {}, [], []
    i5 = E.Q_LIST.index(5)
    for p in pairs:
        rk = by_pair[p]
        g = rk.mean(0)
        qs[p] = E.q_star_of(g)
        ge_l.append(float(np.log2(g[-1])))
        sr.append(float((rk[:, i5] == 1).mean()))
    ok = [q for q in qs.values() if q is not None]
    cens = [p for p, q in qs.items() if q is None]
    slow = (max(cens, key=lambda p: by_pair[p].mean(0)[-1]) if cens else
            max(qs, key=lambda p: (qs[p], by_pair[p].mean(0)[-1])))
    return dict(qs=qs, q_max=(max(ok) if ok and not cens else None),
                q_mean=(float(np.mean(ok)) if ok else None),
                q_med=(float(np.median(ok)) if ok else None), n_cens=len(cens), slow=slow,
                sr5=float(np.mean(sr)), ge=float(np.mean(ge_l)))


def _r(x, nd):
    return "" if x is None or not np.isfinite(x) else round(float(x), nd)


def recover():
    f, err = assemble()
    if err:
        raise SystemExit(f"recover: {err}")
    print(f"combined logp: {f}")
    os.makedirs(OUT_K, exist_ok=True)
    rf = os.path.join(OUT_K, f"res_{TAG}.npz")
    done = [r for r in (_csv(os.path.join(OUT_K, "eval_runs.csv"))) if r["run_id"] == TAG]
    if done:
        print(f"already done: {rf} (eval_runs.csv has a {TAG} row)")
        return
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
    a = np.load(src)
    z = np.load(f)
    lp = np.asarray(z["logp"], np.float32)
    v = [f"index check {'match ✓' if np.array_equal(np.load(dst)['idx'], a['idx']) else '**mismatch**'}",
         f"logp te_idx {'match ✓' if np.array_equal(z['te_idx'], a['idx']) else '**mismatch**'}"]
    print("  " + "  |  ".join(v))
    hw = torch.from_numpy(np.asarray(T.HW16, np.int64)).to(dev)
    t0 = time.time()
    rk = {}
    if os.path.exists(rf):                       # ranks already computed — only rewrite the CSV
        zr = np.load(rf)
        rk = {int(k[1:]): zr[k] for k in zr.files if k.startswith("p")}
        print(f"  reusing rank arrays: {rf} ({len(rk)} pairs)")
    else:
        for i, p in enumerate(range(LEAK)):
            td = [torch.from_numpy(x).to(dev) for x in tabs[p]]
            l0 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p])).to(dev)
            l1 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p + 1])).to(dev)
            r, _ = E.recover_pair(td, l0, l1, hw, tflat[p], orders, chunk, dev)
            rk[p] = r["sum"]
            if i == 0:
                print(f"  1 pair {time.time()-t0:.1f} s -> about {(time.time()-t0)*LEAK/60:.1f} min")
        np.savez(rf, eval_sec=time.time() - t0, **{f"p{p}": v_ for p, v_ in rk.items()})
    st = run_stats(rk, list(range(LEAK)))
    mean_rk = {p: rk[p] for p in rk}
    srq = {q: float(np.mean([(mean_rk[p][:, j] == 1).mean() for p in rk]))
           for j, q in enumerate(E.Q_LIST)}
    geq = {q: float(np.mean([np.log2(mean_rk[p][:, j].mean()) for p in rk]))
           for j, q in enumerate(E.Q_LIST)}
    E.append_rows(os.path.join(OUT_K, "eval_runs.csv"), RUN_FIELDS, [dict(
        run_id=TAG, width=WLEN, seed=SEED, n_pairs_eval=LEAK,
        q_star_max=("" if st["q_max"] is None else st["q_max"]),
        q_star_mean=_r(st["q_mean"], 2), q_star_median=_r(st["q_med"], 1),
        n_pairs_censored=st["n_cens"], slowest_pair_idx=st["slow"],
        slowest_pair=st["slow"] + 1,
        **{f"sr_q{q}": _r(srq[q], 4) for q in E.Q_LIST},
        **{f"ge_log2_q{q}": _r(geq[q], 3) for q in E.Q_LIST},
        n_repeat=rk[0].shape[0], eval_time_sec=_r(time.time() - t0, 1))])
    E.append_rows(os.path.join(OUT_K, "pair_qstar.csv"), PQ_FIELDS, [dict(
        run_id=TAG, width=WLEN, seed=SEED, pair_idx=p, pair=p + 1,
        q_star=("" if st["qs"][p] is None else st["qs"][p]), censored=int(st["qs"][p] is None),
        ge_log2_qmax=_r(float(np.log2(rk[p].mean(0)[-1])), 3)) for p in range(LEAK)])
    with open(os.path.join(OUT_K, "verify.txt"), "a", encoding="utf-8") as fh:
        fh.write(f"{TAG}: " + "  |  ".join(v) + "\n")
    E.OUT_DIR = FINE_DIR
    print(f"  Q*max {st['q_max'] if st['q_max'] is not None else '>50'}  Q*mean "
          f"{_r(st['q_mean'], 2)}  SR@5 {_r(st['sr5'], 3)}  censor {st['n_cens']}  "
          f"slowest pair {st['slow']+1}")


# ================= --table =================
def _csv(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    e2 = _csv(os.path.join(FINE_DIR + "_e2local", "eval_runs.csv"))
    lr = _csv(os.path.join(BASE + "_qfine_localrec", "eval_runs.csv"))
    lh = _csv(os.path.join(BASE + "_qfine_localrec", "eval_heads.csv"))
    e1 = _csv(os.path.join(OUT_K, "eval_runs.csv"))
    wm = _csv(os.path.join(OUT_M, "window_models.csv"))

    def e2row(n, s, src):
        return next((r for r in e2 if r["n_pairs"] == str(n) and r["seed"] == str(s)
                     and r["heads_src"] == src), None)

    def heads_mean(rid):
        rs = [r for r in lh if r["run_id"] == rid and r.get("signal", "1") == "1"]
        if not rs:
            return "", ""
        return (f"{np.mean([float(r['acc']) for r in rs]):.1f}",
                f"{np.mean([float(r['info_bits']) for r in rs]):.2f}")

    say("=" * 92)
    say("E1 — request 3-3(4).  39 pairs, fine, seed 0, 2000 test traces, canonical-form candidates")
    say("=" * 92)
    say(f"{'':<34}{'Q*max':>7}{'Q*mean':>8}{'SR@5':>7}{'acc%':>8}{'info(bit)':>11}")
    say("-" * 92)
    o = e2row(39, 0, "orig")
    if o:
        say(f"{'original head N39 s0 (pooled)':<34}{o['q_star_max'] or '>50':>7}{o['q_star_mean']:>8}"
            f"{float(o['sr_q5']):>7.2f}{'61.1':>8}{'2.23':>11}")
    for n in (39, 49):
        l = e2row(n, 0, "local")
        if l:
            say(f"{f'local readout N{n} s0':<34}{l['q_star_max'] or '>50':>7}{l['q_star_mean']:>8}"
                f"{float(l['sr_q5']):>7.2f}{float(l['acc_mean']):>8.1f}"
                f"{float(l['info_mean']):>11.2f}")
    raw = next((r for r in lr if r["run_id"] == "A1_raw" and r.get("eval_mode") == "sum"), None)
    if raw:
        a, i = heads_mean("A1_raw")
        say(f"{'raw linear 320 (A1)':<34}{raw['q_star_max'] or '>50':>7}{raw['q_star_mean']:>8}"
            f"{float(raw['sr_q5']):>7.2f}{a:>8}{i:>11}")
    for r in e1:
        if not wm:
            continue
        ws = [w for w in wm if w["width"] == r["width"] and w["seed"] == r["seed"]]
        acc = np.mean([float(w[k]) for w in ws for k in ("acc_r0", "acc_r1")]) if ws else np.nan
        inf = np.mean([float(w[k]) for w in ws for k in ("info_r0", "info_r1")]) if ws else np.nan
        label = "window CNN W" + r["width"] + " s" + r["seed"]
        say(f"{label:<34}{r['q_star_max'] or '>50':>7}"
            f"{r['q_star_mean']:>8}{float(r['sr_q5']):>7.2f}{acc:>8.1f}{inf:>11.2f}")
    say("-" * 92)
    say("  window CNN trained on 64,000 traces, local heads on 16,000 — in favor of the window CNN (3-5)")

    # 3-4 decision — reference = E2 10-seed local (N39/N49)
    say("")
    say("3-4 decision numbers — reference = E2 local readout 10 seeds (N39/N49, 39 pairs, fine)")
    for n in (39, 49):
        ls = [r for r in e2 if r["n_pairs"] == str(n) and r["heads_src"] == "local"]
        qm = [float(r["q_star_max"]) for r in ls if r["q_star_max"]]
        qa = [float(r["q_star_mean"]) for r in ls if r["q_star_mean"]]
        if qm:
            say(f"  local N{n}: Q*max {np.mean(qm):.1f} [{min(qm):.0f},{max(qm):.0f}]  "
                f"Q*mean {np.mean(qa):.2f}")
    for r in e1:
        if r["width"] != str(2 * 659):
            continue
        ref = [float(x["q_star_mean"]) for x in e2 if x["heads_src"] == "local"
               and x["n_pairs"] in ("39", "49") and x["q_star_mean"]]
        qm = int(r["q_star_max"]) if r["q_star_max"] else None
        dq = float(r["q_star_mean"]) - float(np.mean(ref)) if ref else np.nan
        say(f"  window CNN W-RF: Q*max {qm if qm is not None else '>50'}  Q*mean {r['q_star_mean']}"
            f"  (Q*mean vs. local {dq:+.2f})")
        if qm is not None and qm <= 5 and dq <= -0.5:
            v = "cell 1 — window-level training is better. Local readout is a diagnostic"
        elif qm is not None and 6 <= qm <= 8 and abs(dq) < 0.5:
            v = "cell 2 — the jointly trained backbone has the same quality as window-level, the loss is only in the readout"
        elif qm is None or qm >= 9:
            v = "cell 3 — benefit of the multi-head shared representation"
        else:
            v = "between the reference cells — paper chat to judge"
        say(f"  -> {v}")
    say("=" * 92)
    os.makedirs(OUT_K, exist_ok=True)
    with open(os.path.join(OUT_K, "table_3_3.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved: {os.path.join(OUT_K, 'table_3_3.txt')}")


def main():
    flags = {a for a in _USER if a.startswith("--")} - {"--half", "--seed", "--max-pairs"}
    if not flags or "--plan" in flags:
        plan()
        return
    if "--train" in flags:
        train()
    if "--recover" in flags:
        recover()
    if "--table" in flags:
        table()


if __name__ == "__main__":
    main()
