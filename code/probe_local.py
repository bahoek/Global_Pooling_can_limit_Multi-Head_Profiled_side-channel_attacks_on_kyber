"""
Pre-GAP probe — are the unlearned pairs absent from the backbone, or did pooling discard them?

Why
---
`probe_heads.py` reads the vector that goes into `classifiers`, i.e. features **after GAP** (confirmed by feat_dim
512/1280). That unlearned pairs only reach 14~27% there can be read in two ways.

  (a) the conv stack never had a detector for that pair        -> N decides what the backbone learns
  (b) the detector exists but global average pooling discarded it  -> N decides the number of channels that pass through pooling

Grounds for suspecting (b): in the shift probe, MobileNet reads 33 of 39 pairs at 50% or more when the trace
is shifted. If the information were absent altogether, shifting should not make it readable either. Also, the phenomenon that only pairs 30, 34, 37 are read is
the same in both backbones (r=+0.876) and unrelated to NICV leakage strength, which fits the
contamination hypothesis that "untrained segments get mixed into the average over the whole window".

What it does
------------
Freeze the backbone and extract the **pre-GAP feature map** (B, C, T'). For each head (pair p, role r),
pool **only around that pair's peak position**. That is, ask directly "can it still not read it when shown its own position?".

Caching the full feature map would be 64000 traces x 1250 positions x 512 channels = 164GB, which is impossible, so only the result
after per-head local pooling is saved (78 heads x C channels). This is more accurate for the verdict anyway.

  GAP (probe_heads)  unlearned pairs 14~27%
  local (this code)      if >> GAP, (b) pooling bottleneck.  if ~= GAP, (a) absent from the backbone

The peak positions use the measured NICV values (peak_sample_r0 / peak_sample_r1 in `nicv_pairs.csv`).
The pair spacing alternates between 984/1048, so a uniform-spacing assumption is not used (same reason as the shift probe).

Run
---
    python probe_local.py

Place it in the same folder as eval_nscan.py. The model list and train count are reduced so that it finishes within 20 min
(MODELS, N_TRAIN). If interrupted, the same command resumes.

Output <OUT_DIR>/probe_local/
  local_pairs.csv    model x pair x role: acc / info / majority, seen or not
  local_models.csv   1 row per model + columns side by side with the GAP result (probe_models.csv)
  verdict table on screen
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

# ==========================================================
OUT = os.path.join(E.OUT_DIR, "probe_local")
NICV = os.path.join(E.OUT_DIR, "nicv", "nicv_pairs.csv")

# Minimal set needed for the verdict. Centered on N19 and N29, where pairs 30, 34, 37 are read
MODELS = [("resnet", 9, 0), ("resnet", 9, 1), ("resnet", 19, 0), ("resnet", 19, 1),
          ("resnet", 29, 0), ("resnet", 29, 1),
          ("mobilenet", 19, 0), ("mobilenet", 19, 1),      # N19 converged seeds: 0,1,6,9
          ("mobilenet", 29, 2), ("mobilenet", 29, 3)]      # N29 converged seeds: 1~9 (0 stalled)

N_TRAIN = 16000          # only the heads are trained, so the full set (64000) is not needed. Saves memory and time
N_VALID = 4000
HALF = 160               # pool only peak ±HALF samples. Within a pair the r0->r1 offset is 320~348, so
                         # with ±160 the r0 window and r1 window do not overlap
EPOCHS = 60
LR = 1e-3
BATCH = 256
WD = 3e-3
DROPOUT = 0.4
PATIENCE = 12
MIN_DELTA = 0.3
# ==========================================================

PAIR_FIELDS = ["arch", "src_n_pairs", "seed", "pair", "seen", "acc_r0", "acc_r1",
               "majority_r0", "majority_r1", "info_r0", "info_r1"]
MODEL_FIELDS = ["arch", "src_n_pairs", "seed", "feat_ch", "feat_len", "pool_width",
                "probe_epochs", "acc_seen", "acc_unseen", "majority_unseen",
                "info_seen", "info_unseen", "pairs_unseen_over_50", "n_unseen",
                "gap_acc_unseen", "gap_info_unseen", "ratio_local_over_gap", "time_min"]
LN2 = float(np.log(2.0))


# ---------------- extract the pre-GAP feature map ----------------
class PreGAP:
    """So that the structure need not be known, hook the last module that outputs a 3-D (B,C,T) tensor in the forward pass.

    The code does not assume the backbone structure of train_nscan. It passes one batch, looks at the output
    dimensions of every module, and takes the **last-called** module that outputs (B,C,T) as the one right before GAP.
    """
    def __init__(self, model, x1, dev):
        self.hits = []
        hs = [m.register_forward_hook(self._probe(i)) for i, m in enumerate(model.modules())]
        with torch.no_grad():
            model(x1.to(dev))
        for h in hs:
            h.remove()
        if not self.hits:
            raise SystemExit("No module outputs a 3-D (B,C,T) tensor — please report the structure")
        self.idx, self.shape = self.hits[-1]
        self.mod = list(model.modules())[self.idx]
        self.buf = None
        self.h = self.mod.register_forward_hook(self._keep)

    def _probe(self, i):
        def f(m, inp, out):
            if torch.is_tensor(out) and out.dim() == 3:
                self.hits.append((i, tuple(out.shape)))
        return f

    def _keep(self, m, inp, out):
        self.buf = out

    def close(self):
        self.h.remove()


def read_peaks():
    """Measured NICV peak positions (pair x role). Abort if missing — the uniform-spacing assumption is not used."""
    if not os.path.exists(NICV):
        raise SystemExit(f"{NICV} does not exist. Run nicv_pairs.py first "
                         f"(the peak spacing alternates between 984/1048, so measured values are needed)")
    pk = {}
    for r in csv.DictReader(open(NICV, encoding="utf-8")):
        p = int(r["pair"])
        pk[(p, 0)] = int(r["peak_sample_r0"])
        pk[(p, 1)] = int(r["peak_sample_r1"])
    return pk


def pool_windows(fmap, win):
    """Average the feature map (B,C,T') over each head's window. win = [(lo,hi), ...] (feature-map coordinates)
    returns (B, H, C)"""
    return torch.stack([fmap[:, :, lo:hi].mean(2) for lo, hi in win], 1)


# ---------------- heads ----------------
def make_heads(H, C, ncls, dev):
    """Each head reads only the C-dim vector of its own window (same capacity as the GAP version)"""
    return nn.ModuleList([nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(C, ncls))
                          for _ in range(H)]).to(dev)


def run_heads(heads, F):
    """F (B,H,C) -> list[H] of (B,ncls)"""
    return [h(F[:, i]) for i, h in enumerate(heads)]


def train(heads, Ftr, Ytr, Fva, Yva, dev, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(heads.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    H = Ftr.shape[1]
    best, best_sd, bad, eps = -1.0, None, 0, 0
    for ep in range(1, EPOCHS + 1):
        heads.train()
        perm = torch.randperm(len(Ftr))
        for i in range(0, len(Ftr), BATCH):
            b = perm[i:i + BATCH]
            xb = Ftr[b].to(dev, non_blocking=True).float()
            yb = Ytr[b].to(dev, non_blocking=True)
            outs = run_heads(heads, xb)
            loss = sum(ce(o, yb[:, k]) for k, o in enumerate(outs)) / H
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
        heads.eval()
        hit = 0
        with torch.no_grad():
            for i in range(0, len(Fva), 1024):
                xb = Fva[i:i + 1024].to(dev).float()
                yb = Yva[i:i + 1024].to(dev)
                outs = run_heads(heads, xb)
                hit += sum((o.argmax(1) == yb[:, k]).sum().item() for k, o in enumerate(outs))
        acc = hit / (len(Fva) * H) * 100
        eps = ep
        if acc > best + MIN_DELTA:
            best, bad = acc, 0
            best_sd = {k: v.detach().clone() for k, v in heads.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_sd:
        heads.load_state_dict(best_sd)
    return best, eps


def evaluate(heads, Fte, Y, logpr, dev):
    """acc(%) / info(bit) per head. Same definition as eval_nscan and probe_heads"""
    heads.eval()
    H = Fte.shape[1]
    lp = []
    with torch.no_grad():
        for i in range(0, len(Fte), 1024):
            xb = Fte[i:i + 1024].to(dev).float()
            outs = run_heads(heads, xb)
            lp.append(torch.stack([torch.log_softmax(o, 1) for o in outs], 1).cpu().numpy())
    lp = np.concatenate(lp)                                   # (n, H, ncls)
    acc = np.array([(lp[:, h, :].argmax(1) == Y[:, h]).mean() * 100 for h in range(H)])
    ll = lp[np.arange(len(Y))[:, None], np.arange(H)[None, :], Y[:, :H]]
    info = (ll - logpr[np.arange(H)[None, :], Y[:, :H]]).mean(0) / LN2
    return acc, info


def repair(pp, mp):
    """Clean up interrupted runs (same principle as probe_heads.py)"""
    if not os.path.exists(mp):
        return set()
    rows = list(csv.DictReader(open(mp, encoding="utf-8")))
    done = {(r["arch"], int(r["src_n_pairs"]), int(r["seed"])) for r in rows}
    if done:
        print(f"  skipping {len(done)} finished runs")
    return done


def gap_reference():
    """probe_heads GAP results as (arch,N,seed) -> (acc_unseen, info_unseen)"""
    p = os.path.join(E.OUT_DIR, "probe_heads", "probe_models.csv")
    if not os.path.exists(p):
        print(f"  [note] {p} does not exist, so the GAP comparison columns will be blank")
        return {}
    g = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["src_converged"] != "1" or r["acc_unseen"] == "":
            continue
        g[(r["arch"], int(r["src_n_pairs"]), int(r["seed"]))] = \
            (float(r["acc_unseen"]), float(r["info_unseen"]))
    return g


def main():
    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncls = int(getattr(T, "NUM_CLASSES", 17))
    peaks = read_peaks()
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
        X = ((X.astype(np.float32) - mu) / sd).astype(np.float32)
        return X, Y

    Xtr, Ytr = load("train", N_TRAIN)
    Xva, Yva = load("valid", N_VALID)
    Xte_all, Yte_all = load("test")
    prior = np.stack([(np.bincount(Yte_all[:, j], minlength=ncls) + 1.0) /
                      (len(Yte_all) + ncls) for j in range(H)])
    logpr = np.log(prior)
    maj = prior.max(1) * 100
    ip = os.path.join(E.OUT_DIR, "eval_indices.npz")
    te_idx = np.load(ip)["idx"] if os.path.exists(ip) else np.sort(
        np.random.default_rng(E.EVAL_SEED).choice(len(Xte_all), E.EVAL_N, replace=False))
    Xte, Yte = Xte_all[te_idx], Yte_all[te_idx]
    print(f"device {dev} | train {Xtr.shape} valid {Xva.shape} test {Xte.shape}")
    print(f"heads: {H}, each head reads only its own peak ±{HALF} samples (measured NICV positions)")

    pp, mp = os.path.join(OUT, "local_pairs.csv"), os.path.join(OUT, "local_models.csv")
    done = repair(pp, mp)
    GAP = gap_reference()
    t0 = time.time()

    for arch, n, seed in MODELS:
        if (arch, n, seed) in done:
            continue
        f = os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")
        if not os.path.exists(f):
            print(f"  [missing] {os.path.basename(f)}")
            continue
        t1 = time.time()
        ck = torch.load(f, map_location="cpu", weights_only=False)
        model = E.build(T, arch, n, int(ck.get("heads", 2 * n)), ck.get("resnet_variant")).to(dev)
        E.load_state(model, ck["model"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # check feature-map size + convert sample -> feature coordinates
        x1 = torch.from_numpy(Xtr[:2]).unsqueeze(1)
        pg = PreGAP(model, x1, dev)
        C, Tf = pg.shape[1], pg.shape[2]
        Ts = Xtr.shape[1]
        r = Ts / Tf
        win = []
        for p in range(E.LEAK_PAIRS):
            for role in (0, 1):
                c = peaks[(p, role)] - lo                       # to window coordinates
                a = int(max(0, (c - HALF) / r))
                b = int(min(Tf, max(a + 1, (c + HALF) / r)))
                win.append((a, b))
        w = int(np.mean([b - a for a, b in win]))
        print(f"\n{arch} N{n} s{seed} | feature map C={C} T'={Tf} (downsampling {r:.1f}x) | window width {w} positions")

        def feats(X):
            out = np.empty((len(X), H, C), np.float16)
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    xb = torch.from_numpy(X[i:i + 128]).unsqueeze(1).to(dev)
                    model(xb)
                    out[i:i + 128] = pool_windows(pg.buf, win).half().cpu().numpy()
            return torch.from_numpy(out)

        Ftr, Fva, Fte = feats(Xtr), feats(Xva), feats(Xte)
        pg.close()

        heads = make_heads(H, C, ncls, dev)
        vacc, eps = train(heads, Ftr, torch.from_numpy(Ytr), Fva, torch.from_numpy(Yva), dev, seed)
        acc, info = evaluate(heads, Fte, Yte, logpr, dev)

        seen = np.zeros(H, bool)
        seen[:2 * min(n, E.LEAK_PAIRS)] = True
        pair_acc = (acc[0::2] + acc[1::2]) / 2
        unseen_p = ~seen[0::2]
        g = lambda v, m: round(float(v[m].mean()), 3) if m.any() else ""
        ga, gi = GAP.get((arch, n, seed), ("", ""))
        ratio = round(float(info[~seen].mean()) / gi, 2) if gi not in ("", 0) else ""

        row = dict(arch=arch, src_n_pairs=n, seed=seed, feat_ch=C, feat_len=Tf, pool_width=w,
                   probe_epochs=eps, acc_seen=g(acc, seen), acc_unseen=g(acc, ~seen),
                   majority_unseen=g(maj, ~seen), info_seen=g(info, seen),
                   info_unseen=g(info, ~seen),
                   pairs_unseen_over_50=int((pair_acc[unseen_p] >= 50).sum()),
                   n_unseen=int(unseen_p.sum()), gap_acc_unseen=ga, gap_info_unseen=gi,
                   ratio_local_over_gap=ratio, time_min=round((time.time() - t1) / 60, 2))
        prows = [dict(arch=arch, src_n_pairs=n, seed=seed, pair=p, seen=int(seen[2 * p]),
                      acc_r0=round(float(acc[2 * p]), 3), acc_r1=round(float(acc[2 * p + 1]), 3),
                      majority_r0=round(float(maj[2 * p]), 3),
                      majority_r1=round(float(maj[2 * p + 1]), 3),
                      info_r0=round(float(info[2 * p]), 5),
                      info_r1=round(float(info[2 * p + 1]), 5))
                 for p in range(E.LEAK_PAIRS)]
        E.append_rows(pp, PAIR_FIELDS, prows)
        E.append_rows(mp, MODEL_FIELDS, [row])

        print(f"  local: learned pairs {row['acc_seen']}% / unlearned pairs {row['acc_unseen']}% "
              f"(baseline {row['majority_unseen']}%, info {row['info_unseen']}bit) | "
              f"unlearned pairs over 50% {row['pairs_unseen_over_50']}/{row['n_unseen']}")
        print(f"  GAP : unlearned pairs {ga}% (info {gi}bit)   →  local/GAP info = {ratio}")
        print(f"  {row['time_min']} min (cumulative {(time.time()-t0)/60:.0f} min)")
        del model, heads, Ftr, Fva, Fte
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    verdict(mp)


def verdict(mp):
    if not os.path.exists(mp):
        return
    R = list(csv.DictReader(open(mp, encoding="utf-8")))
    print("\n" + "=" * 104)
    print("Verdict — are unlearned pairs readable when shown only their own peak?")
    print("=" * 104)
    print(f"{'arch':<10}{'N':>4}{'s':>3}{'local acc':>10}{'GAP acc':>9}{'base':>8}"
          f"{'local info':>11}{'GAP info':>10}{'local/GAP':>10}{'pairs>50%':>11}")
    print("-" * 104)
    rs = []
    for r in R:
        if r["info_unseen"] == "":
            continue
        rs.append(r)
        print(f"{r['arch']:<10}{r['src_n_pairs']:>4}{r['seed']:>3}{float(r['acc_unseen']):>10.2f}"
              f"{(float(r['gap_acc_unseen']) if r['gap_acc_unseen'] else float('nan')):>9.2f}"
              f"{float(r['majority_unseen']):>8.2f}{float(r['info_unseen']):>11.3f}"
              f"{(float(r['gap_info_unseen']) if r['gap_info_unseen'] else float('nan')):>10.3f}"
              f"{r['ratio_local_over_gap'] or '-':>10}{r['pairs_unseen_over_50']:>7}/{r['n_unseen']}")
    print("=" * 104)
    rt = [float(r["ratio_local_over_gap"]) for r in rs if r["ratio_local_over_gap"]]
    if not rt:
        print("No GAP comparison data, so the ratio cannot be computed (check probe_models.csv)")
        return
    m = float(np.mean(rt))
    print(f"mean local/GAP info ratio = {m:.2f}  (n={len(rt)})")
    if m > 2.0:
        print("→ (b) **pooling bottleneck**. The conv stack has detectors for the unlearned pairs and GAP discarded them.")
        print("  N decides not what the backbone learns but the number of channels that pass through pooling.")
        print("  This ties together with the shift probe covering 33/39 on MobileNet.")
    elif m < 1.3:
        print("→ (a) **absent from the backbone**. Not readable even when shown its own position = the conv stack has no detector.")
        print("  N decides what the backbone learns in the first place. Pairs 30, 34, 37 are described separately as the exception.")
    else:
        print("→ In between. Pooling discarded some, but undoing that does not reach the level needed for recovery.")
    print("  Either way, the conclusion that 'recovering pairs outside N by adding more heads' is impossible is the same.")
    print("  (if the pairs>50% column is close to 0, an attack is still not possible at all)")


if __name__ == "__main__":
    main()
