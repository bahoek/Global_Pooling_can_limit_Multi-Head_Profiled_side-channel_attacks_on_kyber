"""
TIFS — per-pair NICV (leakage strength) measurement

Why
---
1. **Check pair 2**: in key recovery, pair 2 is often the slowest pair at small N, yet canon_mismatch is 0.
   This decides whether the leakage itself is weak for that pair, or whether the problem is on the training/evaluation side.
2. **Reconfirm L=39**: the supporting numbers for the main-text IV-C claim that the window [0,40000) contains only 39 pairs (78 coefficients).
   Also checks whether pairs 39 and above (phantom) sit at the shuffle floor.
3. **Peak position constants**: pin down PEAK0 / COEF_SPACING used by shift_probe.py with measured values.
   A least-squares fit checks whether the currently recorded values (872, 507.2) are correct.

What is measured
----------------
For the label HW16 of coefficient j (= r0/r1 of pair p, j = 2p or 2p+1), at every sample position
    NICV(t) = Var_y( E[X_t | Y_j = y] ) / Var(X_t)
The formal definition with a class-weighted mean. A shuffled-label control (same computation, labels shuffled by row) is reported as the floor.
Without information, NICV hovers around (C-1)/n — the peak verdict is relative to the shuffle floor.

Phantom coefficients (pairs 39 and above) are computed outside the window, so they should equal the floor. Only PHANTOM_COEFS of them are sampled.

Run (same folder as eval_nscan.py, train_nscan.py)
    python nicv_pairs.py

Output <eval_nscan.OUT_DIR>/nicv/
    nicv_coef.csv    one row per coefficient — peak position·value, shuffle floor, ratio to floor
    nicv_pairs.csv   one row per pair       — summary of both coefficients + rank
    nicv_curves.npz  all curves (78+samples x 40000) — for figures
"""

import csv
import os
import time

import numpy as np
import torch

import eval_nscan as E

OUT = os.path.join(E.OUT_DIR, "nicv")

SPLIT         = "test"     # test (20k) / train (64k). train is statistically better but costs more memory·time
N_TRACES      = 0          # 0 = all. Number of traces to reduce to
PHANTOM_COEFS = 8          # number of coefficients from pairs 39+ measured alongside to check the floor (evenly spaced). 0 = off
SHUFFLE_SEED  = 20261030

COEF_FIELDS = ["coef", "pair", "role", "signal", "peak_sample", "peak_nicv",
               "shuf_peak_nicv", "ratio_to_floor", "mean_nicv", "n_traces"]
PAIR_FIELDS = ["pair", "signal", "peak_nicv_r0", "peak_nicv_r1", "peak_nicv_min",
               "peak_sample_r0", "peak_sample_r1", "rank_by_min", "ratio_to_floor_min"]


def nicv_curves(Xg, Y, ncls, dev):
    """Xg: (n, L) GPU float32, Y: (n, H) int64 -> (H, L) float32 NICV"""
    n, L = Xg.shape
    H = Y.shape[1]
    grand = Xg.mean(0)
    var = Xg.var(0, unbiased=False).clamp_min(1e-20)
    out = torch.empty((H, L), dtype=torch.float32, device=dev)
    for j in range(H):
        y = torch.as_tensor(Y[:, j], device=dev)
        cnt = torch.bincount(y, minlength=ncls).clamp_min(1).float()
        sums = torch.zeros((ncls, L), dtype=torch.float32, device=dev).index_add_(0, y, Xg)
        mu = sums / cnt[:, None]
        w = (cnt / n)[:, None]
        out[j] = ((mu - grand) ** 2 * w).sum(0) / var
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    for nm in ("nicv_coef.csv", "nicv_pairs.csv"):       # fresh each time, not appended
        f = os.path.join(OUT, nm)
        if os.path.exists(f):
            os.remove(f)
    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncls = int(getattr(T, "NUM_CLASSES", 17))

    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    lo, hi = [int(v) for v in nz["crop"]]
    d = np.load(os.path.join(E.SPLIT_DIR, f"kyber_{SPLIT}_crop.npz"))
    X = d["traces"][:, lo:hi]
    ct = np.asarray(d["ciphertexts"], np.uint8)
    sk = np.asarray(d["labels"][0], np.int64)
    if N_TRACES and N_TRACES < len(X):
        X, ct = X[:N_TRACES], ct[:N_TRACES]
    X = X.astype(np.float32)
    n, L = X.shape
    print(f"{SPLIT} {n} traces x {L} samples, window [{lo},{hi})  device {dev}")

    sig = list(range(2 * E.LEAK_PAIRS))
    pha = []
    if PHANTOM_COEFS > 0:
        cand = list(range(2 * E.LEAK_PAIRS, 256))
        pha = [cand[int(round(i * (len(cand) - 1) / (PHANTOM_COEFS - 1)))]
               for i in range(PHANTOM_COEFS)]
    coefs = sig + pha

    t0 = time.time()
    uh = T.v_ntt(T.v_unpack_10bit(ct))
    Y = np.stack([T.HW16[np.asarray(T.v_stored(sk, uh, j)) & 0xFFFF] for j in coefs], 1)
    print(f"labels {Y.shape} ({time.time()-t0:.0f} s)")

    Xg = torch.from_numpy(X).to(dev)
    t0 = time.time()
    nic = nicv_curves(Xg, Y, ncls, dev).cpu().numpy()
    print(f"NICV ({time.time()-t0:.0f} s)")
    rng = np.random.default_rng(SHUFFLE_SEED)
    Ys = np.stack([Y[rng.permutation(n), k] for k in range(Y.shape[1])], 1)
    t0 = time.time()
    nsh = nicv_curves(Xg, Ys, ncls, dev).cpu().numpy()
    print(f"shuffle control ({time.time()-t0:.0f} s).  theoretical floor (C-1)/n = {(ncls-1)/n:.2e}")
    del Xg
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    pk = nic.argmax(1)
    pv = nic.max(1)
    sv = nsh.max(1)
    rows = [dict(coef=c, pair=c // 2, role="r0" if c % 2 == 0 else "r1",
                 signal=int(c < 2 * E.LEAK_PAIRS), peak_sample=int(pk[i]),
                 peak_nicv=round(float(pv[i]), 6), shuf_peak_nicv=round(float(sv[i]), 6),
                 ratio_to_floor=round(float(pv[i] / max(sv[i], 1e-12)), 2),
                 mean_nicv=round(float(nic[i].mean()), 8), n_traces=n)
            for i, c in enumerate(coefs)]
    E.append_rows(os.path.join(OUT, "nicv_coef.csv"), COEF_FIELDS, rows)

    # --- peak position fit (signal coefficients only) ---
    ns = len(sig)
    j = np.arange(ns, dtype=np.float64)
    A = np.vstack([np.ones(ns), j]).T
    (p0, sp), *_ = np.linalg.lstsq(A, pk[:ns].astype(np.float64), rcond=None)
    pred = p0 + sp * j
    r2 = 1 - ((pk[:ns] - pred) ** 2).sum() / ((pk[:ns] - pk[:ns].mean()) ** 2).sum()
    resid = np.abs(pk[:ns] - pred)
    print(f"\npeak position fit ({ns} coefficients):  peak ≈ {p0:.1f} + {sp:.2f}·j   R²={r2:.5f}  "
          f"max residual {resid.max():.0f} samples")
    print(f"  compared with current shift_probe.py values PEAK0=872.0, COEF_SPACING=507.2 → "
          f"{'match' if abs(p0-872) < 60 and abs(sp-507.2) < 3 else 'different — change the shift_probe constants to the values above'}")

    # --- pair table ---
    pm = np.minimum(pv[0:ns:2], pv[1:ns:2])
    order = np.argsort(-pm)
    rank = np.empty(len(pm), int)
    rank[order] = np.arange(1, len(pm) + 1)
    prow = [dict(pair=p, signal=1, peak_nicv_r0=round(float(pv[2 * p]), 6),
                 peak_nicv_r1=round(float(pv[2 * p + 1]), 6),
                 peak_nicv_min=round(float(pm[p]), 6),
                 peak_sample_r0=int(pk[2 * p]), peak_sample_r1=int(pk[2 * p + 1]),
                 rank_by_min=int(rank[p]),
                 ratio_to_floor_min=round(float(pm[p] / max(sv[:ns].max(), 1e-12)), 2))
            for p in range(ns // 2)]
    E.append_rows(os.path.join(OUT, "nicv_pairs.csv"), PAIR_FIELDS, prow)
    np.savez(os.path.join(OUT, "nicv_curves.npz"), coefs=np.array(coefs),
             nicv=nic.astype(np.float32), nicv_shuffled=nsh.astype(np.float32),
             peak_sample=pk, split=SPLIT, n_traces=n, crop=np.array([lo, hi]))

    print(f"\nsignal coefficient peak NICV  min {pv[:ns].min():.4f} / median {np.median(pv[:ns]):.4f} / "
          f"max {pv[:ns].max():.4f}")
    print(f"shuffle floor peak      max {sv[:ns].max():.6f}  (signal min / floor = "
          f"{pv[:ns].min()/max(sv[:ns].max(),1e-12):.0f}x)")
    if pha:
        print(f"phantom: {len(pha)} coefficients, peak NICV  max {pv[ns:].max():.6f} "
              f"(floor {sv[ns:].max():.6f}) → "
              f"{'floor level, L=39 confirmed' if pv[ns:].max() < 3*sv[ns:].max() else '[warning] above floor — recheck L'}")

    print("\nWeak pairs (by the smaller peak NICV of the two coefficients)")
    print(f"{'rank':>4}{'pair':>5}{'peak r0':>11}{'peak r1':>11}{'min':>11}{'pos r0':>9}")
    for p in order[-6:][::-1]:
        print(f"{rank[p]:>4}{p:>5}{pv[2*p]:>11.4f}{pv[2*p+1]:>11.4f}{pm[p]:>11.4f}{pk[2*p]:>9}")
    print("Strong pairs")
    for p in order[:3]:
        print(f"{rank[p]:>4}{p:>5}{pv[2*p]:>11.4f}{pv[2*p+1]:>11.4f}{pm[p]:>11.4f}{pk[2*p]:>9}")
    print(f"\npair 2: rank {rank[2]}/{len(pm)}, min peak NICV {pm[2]:.4f} "
          f"(median {np.median(pm):.4f}) → "
          f"{'indeed a weakly leaking pair' if rank[2] > 0.75*len(pm) else 'leakage is ordinary — the slowness has another cause'}")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
