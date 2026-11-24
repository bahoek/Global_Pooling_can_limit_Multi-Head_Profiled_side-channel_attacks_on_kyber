#!/usr/bin/env python3
r"""
probe_local_ext — local readout extension (N >= 39 + control checkpoints)

Request document: `claude/TIFS local readout extension experiment request (paper chat→experiment chat, 11-12).md` stage 1.

    python probe_local_ext.py            # group A (A1~A4)
    python probe_local_ext.py A B        # groups A + B
    python probe_local_ext.py --table    # only reprint the table, no measurement
    python probe_local_ext.py --list     # list of targets

**It does not modify any existing file.** It imports `probe_local.py` and uses `PreGAP` / `read_peaks` /
`pool_windows` / `make_heads` / `train` / `evaluate` as they are. Changing the protocol would make it incomparable with the existing table
(95.9/95.7/95.5 for N9/19/29), so the head structure, epochs, optimizer, window width and checkpoint
selection criterion are all read from the constants of `probe_local` as they are.

Differences from the existing code — only what the request asked for
--------------------------------------------------------------------
1. The targets include **control checkpoints** (D, D29). They live in `model_ctrl/<experiment>/`, so
   they are not found by the `MODEL_DIR/{arch}_N{n}_s{seed}.pth` rule of `probe_local` → the path is taken directly
2. **N >= 39 has `n_unseen = 0`**. The aggregation in `probe_local` is set up to output a blank when `~seen` is the empty set,
   but `ratio_local_over_gap` could hit a division by zero, so all of it was blocked
3. A per-pair CSV is written separately in the **request format**: `run_id, pair, role, acc, majority_acc, info_bits`
   (`local_pairs.csv` of `probe_local` has one row per pair, which is inconvenient for cutting blocks)
4. For stage 2, **the test 2000-trace log-softmax (2000 x 17) is saved to npz for every head** (turn off with `--no-logp`)
5. Block tables for pairs 1-9 / 1-29 / 30-39 are printed side by side with the original head values

Output `<OUT_DIR>/probe_local_ext/`
  local_models.csv    1 row per model
  local_pairs.csv     model x pair x role (request format)
  logp_<run_id>.npz   test 2000 x H x 17 log-softmax (stage-2 input)

Caution (request 3-4)
  * Do not mix seeds — every value in the table is from a single seed
  * Match the pair set — comparisons are on the pairs 1~29 column
  * Do not mix validation/test — all on the 2,000 test traces (`eval_indices.npz`)
  * Do not look at accuracy alone — information (bit) is printed alongside
"""

import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import probe_local as PL                                            # noqa: E402
import eval_nscan as E                                              # noqa: E402

OUT = os.path.join(E.OUT_DIR, "probe_local_ext")
CTRL = os.path.join(os.path.dirname(E.MODEL_DIR.rstrip("/\\")), "model_ctrl")

# (tag, group, arch, N, seed, checkpoint path — None means the main training rule)
MODELS = [
    ("A1", "A", "resnet", 39, 0, None),
    ("A2", "A", "resnet", 39, 1, None),
    ("A3", "A", "resnet", 39, 0, os.path.join(CTRL, "D", "resnet_N39_s0.pth")),
    ("A4", "A", "resnet", 29, 0, os.path.join(CTRL, "D29", "resnet_N29_s0.pth")),
    ("A0", "A", "resnet", 29, 0, None),          # for re-checking (whether 95.48 / 3.11 is reproduced)
    ("B1a", "B", "resnet", 49, 0, None),
    ("B1b", "B", "resnet", 49, 1, None),
    ("B1c", "B", "resnet", 128, 0, None),
    ("B1d", "B", "resnet", 128, 1, None),
    ("B2a", "B", "mobilenet", 39, 1, None),
    ("B2b", "B", "mobilenet", 39, 2, None),
]

PAIR_FIELDS = ["run_id", "arch", "src_n_pairs", "seed", "ckpt", "pair", "role",
               "acc", "majority_acc", "info_bits", "seen"]
MODEL_FIELDS = ["run_id", "arch", "src_n_pairs", "seed", "ckpt", "feat_ch", "feat_len",
                "pool_width", "probe_epochs", "n_seen", "n_unseen",
                "acc_seen", "info_seen", "acc_unseen", "info_unseen",
                "majority_unseen", "time_min"]

BLOCKS = [("pairs 1-9", 1, 9), ("pairs 1-29", 1, 29), ("pairs 30-39", 30, 39)]


def run_id(tag, arch, n, seed, ckpt):
    base = f"{tag}_{arch}_N{n}_s{seed}"
    return base if ckpt is None else base + "_" + os.path.basename(os.path.dirname(ckpt))


def ckpt_path(arch, n, seed, ckpt):
    return ckpt if ckpt else os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")


def done_set(mp):
    if not os.path.exists(mp):
        return set()
    d = {r["run_id"] for r in csv.DictReader(open(mp, encoding="utf-8"))}
    if d:
        print(f"  skipping {len(d)} finished runs: {sorted(d)}")
    return d


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    if "--list" in flags:
        for tag, grp, arch, n, seed, ck in MODELS:
            p = ckpt_path(arch, n, seed, ck)
            print(f"  [{grp}] {tag:<4} {arch} N{n} s{seed}  "
                  f"{'exists' if os.path.exists(p) else 'missing'}  {p}")
        return
    if "--table" in flags:
        table()
        return

    groups = set(a.upper() for a in args) or {"A"}
    todo = [m for m in MODELS if m[1] in groups]
    if not todo:
        raise SystemExit(f"No groups selected. Choose from A / B (given: {args})")

    save_logp = "--no-logp" not in flags
    T = PL.E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncls = int(getattr(T, "NUM_CLASSES", 17))
    peaks = PL.read_peaks()
    os.makedirs(OUT, exist_ok=True)
    H = 2 * E.LEAK_PAIRS                       # always only the 39 signal pairs. N49 and N128 also read only pairs 0~38

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
    logpr = np.log(prior)
    maj = prior.max(1) * 100

    ip = os.path.join(E.OUT_DIR, "eval_indices.npz")
    if os.path.exists(ip):
        te_idx = np.load(ip)["idx"]
        print(f"  test indices: same as the main evaluation ({ip})")
    else:
        te_idx = np.sort(np.random.default_rng(E.EVAL_SEED).choice(
            len(Xte_all), E.EVAL_N, replace=False))
        print(f"  [caution] {ip} does not exist, so it is created anew with the same seed")
    Xte, Yte = Xte_all[te_idx], Yte_all[te_idx]

    print(f"device {dev} | train {Xtr.shape} valid {Xva.shape} test {Xte.shape}")
    print(f"heads: {H} (39 signal pairs), each head reads only its own peak ±{PL.HALF} samples")
    print(f"protocol: same as probe_local (EPOCHS={PL.EPOCHS} LR={PL.LR} "
          f"BATCH={PL.BATCH} PATIENCE={PL.PATIENCE})\n")

    pp, mp = os.path.join(OUT, "local_pairs.csv"), os.path.join(OUT, "local_models.csv")
    done = done_set(mp)

    for tag, grp, arch, n, seed, ck in todo:
        rid = run_id(tag, arch, n, seed, ck)
        if rid in done:
            continue
        f = ckpt_path(arch, n, seed, ck)
        if not os.path.exists(f):
            print(f"  [missing] {rid}: {f}")
            continue
        t1 = time.time()
        cp = torch.load(f, map_location="cpu", weights_only=False)
        model = E.build(T, arch, n, int(cp.get("heads", 2 * n)),
                        cp.get("resnet_variant")).to(dev)
        E.load_state(model, cp["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

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
        w = int(np.mean([b - a for a, b in win]))
        print(f"\n[{tag}] {rid} | C={C} T'={Tf} (downsampling {r:.1f}x) | window width {w} | {f}")

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
        _, eps = PL.train(heads, Ftr, torch.from_numpy(Ytr), Fva,
                          torch.from_numpy(Yva), dev, seed)
        acc, info = PL.evaluate(heads, Fte, Yte, logpr, dev)

        if save_logp:
            lp = []
            heads.eval()
            with torch.no_grad():
                for i in range(0, len(Fte), 1024):
                    o = PL.run_heads(heads, Fte[i:i + 1024].to(dev).float())
                    lp.append(torch.stack([torch.log_softmax(t, 1) for t in o], 1)
                              .cpu().numpy().astype(np.float32))
            np.savez_compressed(os.path.join(OUT, f"logp_{rid}.npz"),
                                logp=np.concatenate(lp), te_idx=te_idx,
                                arch=arch, n_pairs=n, seed=seed, heads=H)
            print(f"    log-softmax saved (stage-2 input): logp_{rid}.npz")

        # pairs this model learned = pairs 0 .. min(n, 39)-1. If N>=39 all are seen -> n_unseen = 0
        n_seen_p = min(n, E.LEAK_PAIRS)
        seen = np.zeros(H, bool)
        seen[:2 * n_seen_p] = True
        g = lambda v, m: round(float(v[m].mean()), 3) if m.any() else ""

        with open(pp, "a", newline="", encoding="utf-8") as fp:
            wr = csv.DictWriter(fp, fieldnames=PAIR_FIELDS)
            if fp.tell() == 0:
                wr.writeheader()
            for p in range(E.LEAK_PAIRS):
                for role in (0, 1):
                    h = 2 * p + role
                    wr.writerow(dict(run_id=rid, arch=arch, src_n_pairs=n, seed=seed,
                                     ckpt=("main" if ck is None else os.path.basename(
                                         os.path.dirname(ck))),
                                     pair=p + 1, role=f"r{role}",
                                     acc=round(float(acc[h]), 3),
                                     majority_acc=round(float(maj[h]), 3),
                                     info_bits=round(float(info[h]), 4),
                                     seen=int(seen[h])))

        row = dict(run_id=rid, arch=arch, src_n_pairs=n, seed=seed,
                   ckpt=("main" if ck is None else os.path.basename(os.path.dirname(ck))),
                   feat_ch=C, feat_len=Tf, pool_width=w, probe_epochs=eps,
                   n_seen=n_seen_p, n_unseen=E.LEAK_PAIRS - n_seen_p,
                   acc_seen=g(acc, seen), info_seen=g(info, seen),
                   acc_unseen=g(acc, ~seen), info_unseen=g(info, ~seen),
                   majority_unseen=g(maj, ~seen),
                   time_min=round((time.time() - t1) / 60, 2))
        with open(mp, "a", newline="", encoding="utf-8") as fm:
            wr = csv.DictWriter(fm, fieldnames=MODEL_FIELDS)
            if fm.tell() == 0:
                wr.writeheader()
            wr.writerow(row)

        print(f"    learned pairs {n_seen_p}: acc {row['acc_seen']}% / info {row['info_seen']}bit"
              + (f" | unlearned pairs {E.LEAK_PAIRS - n_seen_p}: "
                 f"acc {row['acc_unseen']}% / info {row['info_unseen']}bit"
                 if E.LEAK_PAIRS - n_seen_p else "  (n_unseen = 0)")
              + f" | {row['time_min']} min")
        del model, heads, Ftr, Fva, Fte
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    table()


# ---------------- table ----------------
def block(rows, lo, hi, key):
    v = [float(r[key]) for r in rows if lo <= int(r["pair"]) <= hi and r[key] != ""]
    return float(np.mean(v)) if v else None


def orig_heads():
    """Per-pair values of the original heads. Reads both the main evaluation and the control evaluation."""
    root = os.path.dirname(E.MODEL_DIR.rstrip("/\\"))
    out = {}
    for src, path in (("main", os.path.join(E.OUT_DIR, "eval_heads.csv")),
                      ("ctrl", os.path.join(root, "eval_ctrl_d", "eval_heads.csv"))):
        if not os.path.exists(path):
            continue
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("eval_mode", "sum") != "sum":
                continue
            hcol = next((c for c in r if c in ("head", "head_idx", "head_index")), None)
            acol = next((c for c in r if "acc" in c), None)
            icol = next((c for c in r if "info" in c or "kl" in c), None)
            if hcol is None or acol is None:
                continue
            k = (src, r["arch"], int(r["n_pairs"]), int(r["seed"]))
            out.setdefault(k, []).append(dict(
                pair=int(r[hcol]) // 2 + 1, acc=r[acol],
                info_bits=(r[icol] if icol else "")))
    return out


def table():
    mp, pp = os.path.join(OUT, "local_models.csv"), os.path.join(OUT, "local_pairs.csv")
    if not os.path.exists(pp):
        print(f"\n{pp} does not exist. Run the measurement first.")
        return
    rows = list(csv.DictReader(open(pp, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["run_id"], []).append(r)
    OH = orig_heads()

    print("\n" + "=" * 92)
    print("ResNet, 2000 test traces, accuracy % / information bit")
    print("=" * 92)
    hdr = f"{'':<30}" + "".join(f"{b[0]:>18}" for b in BLOCKS) + f"{'all (39 pairs)':>18}"
    print(hdr)
    print("-" * len(hdr))

    def line(lab, rs, akey, ikey):
        s = f"{lab:<30}"
        for _, lo, hi in BLOCKS:
            a, i = block(rs, lo, hi, akey), block(rs, lo, hi, ikey)
            s += f"{(f'{a:.2f} / {i:.2f}' if a is not None and i is not None else (f'{a:.2f} / —' if a is not None else '—')):>18}"
        a, i = block(rs, 1, 39, akey), block(rs, 1, 39, ikey)
        s += f"{(f'{a:.2f} / {i:.2f}' if a is not None and i is not None else (f'{a:.2f} / —' if a is not None else '—')):>18}"
        print(s)

    for rid in sorted(by):
        r0 = by[rid][0]
        arch, n, seed = r0["arch"], int(r0["src_n_pairs"]), int(r0["seed"])
        src = "main" if r0["ckpt"] == "main" else "ctrl"
        oh = OH.get((src, arch, n, seed))
        tagn = f"{rid}"
        if oh:
            line(f"{tagn}  original heads", oh, "acc", "info_bits")
        else:
            print(f"{tagn + '  original heads':<30}{'(no eval_heads.csv)':>18}")
        line(f"{tagn}  local", by[rid], "acc", "info_bits")
        print()

    print("Verdict criteria (request 3-5) — reference column pairs 1~29, reference model D, N29 local 95.5% / 3.11bit")
    d = next((k for k in by if k.startswith("A3")), None)
    if d:
        a, i = block(by[d], 1, 29, "acc"), block(by[d], 1, 29, "info_bits")
        print(f"  D local (pairs 1~29): {a:.2f}% / {i:.3f}bit")
        if a >= 92 and i >= 2.9:
            print("  -> **pooling bottleneck.** The backbone produces N29-level features even at N39. Proceed to stage 2")
        elif a <= 80 or i <= 2.4:
            print("  -> **drop inside the backbone.** Pooling hypothesis rejected. No stage 2")
        else:
            print("  -> **mixed.** Report the fraction of the original gap that local readout closes. Discuss with the paper chat")
    a1 = next((k for k in by if k.startswith("A1")), None)
    if d and a1:
        x = block(by[a1], 1, 29, "acc")
        y = block(by[d], 1, 29, "acc")
        print(f"\n  A1(100 epochs N39) {x:.2f}%  vs  A3(D) {y:.2f}%   diff {y - x:+.2f}%p")
        print("  -> if the difference is clear, A1 is an 'under-trained backbone' (last line of request 3-5)")
    print(f"\nraw data: {mp}\n        {pp}")
    print("=" * 92)


if __name__ == "__main__":
    main()
