"""
probe_local control experiment — does the local readout read 'learned features', or is the window simply leaking?

Why this is needed
------------------
The `probe_local.py` result overturned the conclusion: unlearned pairs reach 66~82% (baseline 11.9%), and nearly
all pairs exceed 50% (ResNet N19/N29, MobileNet 20/20, 10/10). Taken at face value, this means an N=9 backbone with a
local head attached reads all 39 pairs.

But three things cast doubt on the measurement itself.
  1. Accuracy on learned pairs **exceeds the original head** (ResNet N9: local 95.92% vs original 93.12%).
     If the learned representation were doing the work, it could not beat its own head on the pairs it learned
  2. The window covers only 4 positions in the ResNet feature map / 2 positions in MobileNet. What the head receives is
     the local conv response around the peak = in effect a **linear template attack** on that window. The HW of an
     unprotected implementation is readable even by a linear model
  3. The local/GAP ratio is larger the less the backbone knows (N9 19.5x -> N29 11.8x). If learned features
     were transferring, the trend should be the opposite

What is measured
----------------
Same local window, same head capacity, same training setup — **only the input changes**.

  trained  pre-GAP features of the trained backbone (same as probe_local.py)
  random   pre-GAP features of a **randomly initialized** backbone — no checkpoint is loaded
  raw      no conv; a linear head directly on the 2*HALF **raw (z-scored) samples inside the window**
           = pure linear template attack baseline

Verdict
-------
  random ~= trained   -> the local measurement is unrelated to learned features. It cannot be read as "the backbone
                         holds unlearned pairs"; the probe_local result measured window leakage
  random << trained   -> learned features really transfer. The min(N,39) framing must be revisited
  raw ~= trained      -> the conv stack does nothing. A stronger form: the local window alone suffices

raw does not depend on the backbone, so it is run only once and re-aggregated by changing only the per-N seen/unseen split.

Run
---
    python probe_local_ctrl.py

Place in the same folder as eval_nscan.py. 4 models x 2 modes + 1 raw run ≈ 20 min. Resumes if interrupted.

Output <OUT_DIR>/probe_local/
  ctrl_pairs.csv     mode x model x pair x role
  ctrl_models.csv    one row per mode x model
  per-mode comparison table and verdict on screen
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_nscan as E
import probe_local as PL          # reuse window computation, heads, training and evaluation as-is

OUT = PL.OUT
MODELS = [("resnet", 9, 0), ("resnet", 29, 0), ("mobilenet", 19, 0), ("mobilenet", 29, 2)]
MODES = ["trained", "random"]     # raw does not depend on the backbone, so it runs separately once
RAW_N_SPLITS = [9, 19, 29]        # which N to use for splitting raw into seen/unseen

PAIR_FIELDS = ["mode"] + PL.PAIR_FIELDS
MODEL_FIELDS = ["mode", "arch", "src_n_pairs", "seed", "in_dim", "probe_epochs",
                "acc_seen", "acc_unseen", "majority_unseen", "info_seen", "info_unseen",
                "pairs_unseen_over_50", "n_unseen", "time_min"]


def summarize(mode, arch, n, seed, acc, info, maj, in_dim, eps, t1):
    H = len(acc)
    seen = np.zeros(H, bool)
    seen[:2 * min(n, E.LEAK_PAIRS)] = True
    pair_acc = (acc[0::2] + acc[1::2]) / 2
    unseen_p = ~seen[0::2]
    g = lambda v, m: round(float(v[m].mean()), 3) if m.any() else ""
    row = dict(mode=mode, arch=arch, src_n_pairs=n, seed=seed, in_dim=in_dim, probe_epochs=eps,
               acc_seen=g(acc, seen), acc_unseen=g(acc, ~seen), majority_unseen=g(maj, ~seen),
               info_seen=g(info, seen), info_unseen=g(info, ~seen),
               pairs_unseen_over_50=int((pair_acc[unseen_p] >= 50).sum()),
               n_unseen=int(unseen_p.sum()), time_min=round((time.time() - t1) / 60, 2))
    prows = [dict(mode=mode, arch=arch, src_n_pairs=n, seed=seed, pair=p, seen=int(seen[2 * p]),
                  acc_r0=round(float(acc[2 * p]), 3), acc_r1=round(float(acc[2 * p + 1]), 3),
                  majority_r0=round(float(maj[2 * p]), 3),
                  majority_r1=round(float(maj[2 * p + 1]), 3),
                  info_r0=round(float(info[2 * p]), 5), info_r1=round(float(info[2 * p + 1]), 5))
             for p in range(E.LEAK_PAIRS)]
    return row, prows


def main():
    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncls = int(getattr(T, "NUM_CLASSES", 17))
    peaks = PL.read_peaks()
    os.makedirs(OUT, exist_ok=True)
    H = 2 * E.LEAK_PAIRS

    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    lo, hi = [int(v) for v in nz["crop"]]
    mu, sd = nz["mu"], nz["sd"]

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

    Xtr, Ytr = load("train", PL.N_TRAIN)
    Xva, Yva = load("valid", PL.N_VALID)
    Xte_all, Yte_all = load("test")
    prior = np.stack([(np.bincount(Yte_all[:, j], minlength=ncls) + 1.0) /
                      (len(Yte_all) + ncls) for j in range(H)])
    logpr, maj = np.log(prior), prior.max(1) * 100
    ip = os.path.join(E.OUT_DIR, "eval_indices.npz")
    te_idx = np.load(ip)["idx"] if os.path.exists(ip) else np.sort(
        np.random.default_rng(E.EVAL_SEED).choice(len(Xte_all), E.EVAL_N, replace=False))
    Xte, Yte = Xte_all[te_idx], Yte_all[te_idx]
    Ytr_t, Yva_t = torch.from_numpy(Ytr), torch.from_numpy(Yva)
    print(f"device {dev} | train {Xtr.shape} test {Xte.shape} | window peak ±{PL.HALF} samples")

    pp, mp = os.path.join(OUT, "ctrl_pairs.csv"), os.path.join(OUT, "ctrl_models.csv")
    done = set()
    if os.path.exists(mp):
        done = {(r["mode"], r["arch"], int(r["src_n_pairs"]), int(r["seed"]))
                for r in csv.DictReader(open(mp, encoding="utf-8"))}
        if done:
            print(f"  skipping {len(done)} completed")
    t0 = time.time()

    # ---------- raw: raw samples inside the window, no conv ----------
    if ("raw", "-", 0, 0) not in done:
        t1 = time.time()
        wins = [(max(0, peaks[(p, r)] - lo - PL.HALF), peaks[(p, r)] - lo + PL.HALF)
                for p in range(E.LEAK_PAIRS) for r in (0, 1)]
        W = min(b - a for a, b in wins)
        print(f"\n[raw] no conv — linear directly on {W} raw samples per head")

        def raw_feats(X):
            return torch.from_numpy(
                np.stack([X[:, a:a + W] for a, b in wins], 1).astype(np.float16))

        Ftr, Fva, Fte = raw_feats(Xtr), raw_feats(Xva), raw_feats(Xte)
        heads = PL.make_heads(H, W, ncls, dev)
        _, eps = PL.train(heads, Ftr, Ytr_t, Fva, Yva_t, dev, 0)
        acc, info = PL.evaluate(heads, Fte, Yte, logpr, dev)
        np.savez(os.path.join(OUT, "ctrl_raw.npz"), acc=acc, info=info)
        for n in RAW_N_SPLITS:                      # re-aggregate the same result under per-N splits
            row, prows = summarize("raw", "-", n, 0, acc, info, maj, W, eps, t1)
            row["seed"] = 0
            E.append_rows(pp, PAIR_FIELDS, prows)
            E.append_rows(mp, MODEL_FIELDS, [row])
            print(f"  split by N{n}: learned pairs {row['acc_seen']}% / unlearned pairs {row['acc_unseen']}% "
                  f"(info {row['info_unseen']}bit) | over 50% {row['pairs_unseen_over_50']}/{row['n_unseen']}")
        del heads, Ftr, Fva, Fte
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---------- trained / random ----------
    for arch, n, seed in MODELS:
        f = os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")
        if not os.path.exists(f):
            print(f"  [missing] {os.path.basename(f)}")
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        for mode in MODES:
            if (mode, arch, n, seed) in done:
                continue
            t1 = time.time()
            model = E.build(T, arch, n, int(ck.get("heads", 2 * n)),
                            ck.get("resnet_variant")).to(dev)
            if mode == "trained":
                E.load_state(model, ck["model"])
            else:
                torch.manual_seed(12345 + seed)     # use the random initialization as-is
            model.eval()
            for p_ in model.parameters():
                p_.requires_grad_(False)

            pg = PL.PreGAP(model, torch.from_numpy(Xtr[:2]).unsqueeze(1), dev)
            C, Tf = pg.shape[1], pg.shape[2]
            r = Xtr.shape[1] / Tf
            win = []
            for p in range(E.LEAK_PAIRS):
                for role in (0, 1):
                    c = peaks[(p, role)] - lo
                    a = int(max(0, (c - PL.HALF) / r))
                    b = int(min(Tf, max(a + 1, (c + PL.HALF) / r)))
                    win.append((a, b))
            print(f"\n[{mode}] {arch} N{n} s{seed} | C={C} T'={Tf} | window {int(np.mean([b-a for a,b in win]))} positions")

            def feats(X):
                out = np.empty((len(X), H, C), np.float16)
                with torch.no_grad():
                    for i in range(0, len(X), 128):
                        xb = torch.from_numpy(X[i:i + 128]).unsqueeze(1).to(dev)
                        model(xb)
                        out[i:i + 128] = PL.pool_windows(pg.buf, win).half().cpu().numpy()
                return torch.from_numpy(out)

            Ftr, Fva, Fte = feats(Xtr), feats(Xva), feats(Xte)
            pg.close()
            heads = PL.make_heads(H, C, ncls, dev)
            _, eps = PL.train(heads, Ftr, Ytr_t, Fva, Yva_t, dev, seed)
            acc, info = PL.evaluate(heads, Fte, Yte, logpr, dev)
            row, prows = summarize(mode, arch, n, seed, acc, info, maj, C, eps, t1)
            E.append_rows(pp, PAIR_FIELDS, prows)
            E.append_rows(mp, MODEL_FIELDS, [row])
            print(f"  learned pairs {row['acc_seen']}% / unlearned pairs {row['acc_unseen']}% "
                  f"(baseline {row['majority_unseen']}%, info {row['info_unseen']}bit) | "
                  f"over 50% {row['pairs_unseen_over_50']}/{row['n_unseen']} | "
                  f"{row['time_min']} min (cumulative {(time.time()-t0)/60:.0f} min)")
            del model, heads, Ftr, Fva, Fte
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    verdict(mp)


def verdict(mp):
    R = [r for r in csv.DictReader(open(mp, encoding="utf-8"))] if os.path.exists(mp) else []
    if not R:
        return
    print("\n" + "=" * 104)
    print("Comparison by mode — only the input changed (window, head capacity, training setup identical)")
    print("=" * 104)
    print(f"{'mode':<9}{'arch':<10}{'N':>4}{'s':>3}{'in_dim':>8}{'learned':>9}{'unlearned':>10}"
          f"{'base':>8}{'info(un)':>10}{'over50%':>10}")
    print("-" * 104)
    key = lambda r: (r["arch"], int(r["src_n_pairs"]), int(r["seed"]))
    for r in sorted(R, key=lambda r: (r["arch"], int(r["src_n_pairs"]), int(r["seed"]),
                                      {"trained": 0, "random": 1, "raw": 2}[r["mode"]])):
        print(f"{r['mode']:<9}{r['arch']:<10}{r['src_n_pairs']:>4}{r['seed']:>3}{r['in_dim']:>8}"
              f"{r['acc_seen']:>9}{r['acc_unseen']:>10}{r['majority_unseen']:>8}"
              f"{r['info_unseen']:>10}{r['pairs_unseen_over_50']:>6}/{r['n_unseen']}")
    print("=" * 104)

    ratios = []
    for r in R:
        if r["mode"] != "trained":
            continue
        rnd = next((q for q in R if q["mode"] == "random" and key(q) == key(r)), None)
        if rnd and float(r["info_unseen"]) > 0:
            ratios.append((key(r), float(rnd["info_unseen"]) / float(r["info_unseen"])))
    if not ratios:
        print("No random control yet.")
        return
    m = float(np.mean([v for _, v in ratios]))
    print("random / trained  (info ratio on unlearned pairs)")
    for k, v in ratios:
        print(f"  {k[0]} N{k[1]} s{k[2]}: {v:.2f}")
    print(f"  mean {m:.2f}")
    print()
    if m > 0.7:
        print("→ **The local measurement is nearly unrelated to learned features.** A random backbone reads about as well.")
        print("  probe_local's 'unlearned pairs 66~82%' is not evidence that the backbone holds those pairs; it means")
        print("  the peak window itself leaks HW and the linear head reads it.")
        print("  → keep the min(N,39) framing. Use probe_local only as a separate observation that 'given the window,")
        print("     any backbone reads it', or only as the structural point that GAP discards position information.")
    elif m < 0.3:
        print("→ **Learned features really transfer.** They cannot be read with a random backbone.")
        print("  This means an N=9 backbone + local head reads 39 pairs, and min(N,39) must be revisited.")
        print("  This would change the results section, so confirm with more seeds.")
    else:
        print("→ In between. Learned features contribute, but leakage of the window itself is also large. Report the two shares separately.")
    raw = [r for r in R if r["mode"] == "raw"]
    if raw:
        print(f"\nraw (no conv, {raw[0]['in_dim']} raw samples) unlearned pairs: "
              + ", ".join(f"N{r['src_n_pairs']} {r['acc_unseen']}%" for r in raw))
        print("  If raw is close to trained, that is the stronger form: the conv stack contributes nothing.")


if __name__ == "__main__":
    main()
