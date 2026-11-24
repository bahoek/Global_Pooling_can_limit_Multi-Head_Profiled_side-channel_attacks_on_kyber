"""
TIFS re-experiment — N-scan training (capacity function measurement)

What was kept from the two original scripts (scalability_results_resnet / MOB)
-----------------------------------------------------------------------------
  * CSV saved immediately after each run (results survive if the process dies)
  * VRAM MAX / AVG logging (reset_peak_memory_stats + per-batch sampling)
  * loop over the N x seed grid, data/labels shared per N
  * early stopping, best checkpoint saving, time at which best performance was reached is recorded

What was fixed
--------------
1. Labels: upper(fqmul) / stored lo8  ->  HW16(stored) 17-class
   upper(fqmul) was at the noise floor with NICV 0.0031, and in the ESL re-experiment
   training was confirmed to converge with HW16(stored).

2. Normalization: per-trace z-score -> per-sample-position z-score (profiling statistics)
   In a narrow window, per-trace normalization erases the absolute differences between traces.
   The train statistics are applied to valid as is, so training and evaluation scales match.

3. Loss: sum over heads -> mean over heads
   sum + clip_grad_norm(1.0) clips the gradient more as N grows,
   cutting the effective learning rate. In an experiment where N is the independent variable, this would measure
   clipping, not capacity. Use the mean and relax the clip to 5.0.

4. GE: int(mean(rank)) 0-based -> float, 1-based (+1)
   So that a perfect prediction is 1, not 0. Truncation also removed.

5. L fixed by fixing the window (crop)
       N < L : under-allocation
       N = L : match
       N > L : over-allocation. The extra heads become phantom heads with no leakage inside the window
               and only consume trunk capacity and gradient.
   This regime is needed to derive the N* selection rule.

6. Scheduler T_max=30 (mismatched with EPOCHS=100, LR periodically revives)
   -> T_max=EPOCHS

    python train_nscan.py
"""

import os
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==========================================================
SPLIT_DIR = r"/home/bahoek/data/kyber/tifs/split"
SAVE_DIR  = r"/home/bahoek/data/kyber/tifs/model"
CACHE_DIR = r"/home/bahoek/data/kyber/tifs/cache"
SUMMARY_CSV = os.path.join(SAVE_DIR, "train_summary.csv")    # 1 row per run
GN_CSV      = os.path.join(SAVE_DIR, "train_gradnorm.csv")   # run x epoch

ARCHS   = ["resnet", "mobilenet"]
# N is in units of 'pairs'. Number of heads = 2 x N (both coefficients of each pair are learned).
#   N=39 pairs = 78 heads = same as the ESL setting = number of pairs actually leaking in the window (L).
#   N=128 pairs = 256 heads = all of poly0.
N_LIST  = [9, 19, 29, 39, 49, 69, 99, 128]
SEEDS   = list(range(10))

def heads_of(n_pairs):
    return 2 * n_pairs

# ResNet variant choice
#   "resnet18" : same as the ESL re-experiment (16->512ch, 8 blocks)
#   "tifs"     : same as the original TIFS submission (16->64ch, 6 blocks, small trunk)
RESNET_VARIANT = "resnet18"

# window = all 40k (all 78 coefficients = 39 pairs fit). N<39 under, N=39 match, N>39 over.
CROP_LO, CROP_HI = 0, 40000
LEAK_PAIRS = 39                  # number of pairs with leakage in this window = L (for recording/display)

LABEL_MODE  = "hw16"
NUM_CLASSES = 17

# whether to scramble the labels of phantom (out-of-window coefficient) heads. Control group of additional experiment B.
#   "real"     : normal labels (default)
#   "shuffled" : labels of heads outside the signal coefficient range (0~77) are shuffled row-wise -> unlearnable signal
PHANTOM_LABEL = "real"

# number of signal coefficients = leaking pairs x 2 (pairs 0~38 -> coefficients 0~77 -> heads 0~77).
# head index < SIGNAL_HEADS is a signal head, anything above is a phantom head.
SIGNAL_HEADS = 2 * LEAK_PAIRS

BATCH     = 128
LR        = 1e-3                 # 3e-4 if the MBConv family diverges
EPOCHS    = 100
PATIENCE  = 15
MIN_DELTA = 0.3
WD        = 3e-3
DROPOUT   = 0.4
GRAD_CLIP = 5.0                  # recorded in the summary CSV. The original TIFS used 1.0.


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


GIT_COMMIT = git_commit()

KYBER_Q, QINV = 3329, 62209
HW16 = np.array([bin(v).count("1") for v in range(65536)], dtype=np.int64)
ZETAS = np.array([
 -1044,-758,-359,-1517,1493,1422,287,202,-171,622,1577,182,962,-1202,-1474,1468,
 573,-1325,264,383,-829,1458,-1602,-130,-681,1017,732,608,-1542,411,-205,-1571,
 1223,652,-552,1015,-1293,1491,-282,-1544,516,-8,-320,-666,-1618,-1162,126,1469,
 -853,-90,-271,830,107,-1421,-247,-951,-398,961,-1508,-725,448,-1065,677,-1275,
 -1103,430,555,843,-1251,871,1550,105,422,587,177,-235,-291,-460,1574,1653,
 -246,778,1159,-147,-777,1483,-602,1119,-1590,644,-872,349,418,329,-156,-75,
 817,1097,603,610,1322,-1285,-1465,384,-1215,-136,1218,-1335,-874,220,-1187,-1659,
 -1185,-1530,-1278,794,-1510,-854,-870,478,-108,-308,996,991,958,-1460,1522,1628],
 dtype=np.int64)


# ---------------- Kyber arithmetic (batch-vectorized, same as the ESL version) ----------------
def i16(v):
    return ((np.asarray(v, dtype=np.int64) & 0xFFFF) ^ 0x8000) - 0x8000

def v_fqmul(a, b):
    ai = i16(a).astype(np.int64) * i16(b).astype(np.int64)
    return i16((ai - i16(ai * QINV) * KYBER_Q) >> 16)

def v_barrett(a):
    a = i16(a)
    return i16(a - i16((20159 * a + (1 << 25)) >> 26) * KYBER_Q)

def v_unpack_10bit(cts):
    c = cts[:, :320].astype(np.int64)
    n = c.shape[0]
    b0 = c[:, 0::5]; b1 = c[:, 1::5]; b2 = c[:, 2::5]; b3 = c[:, 3::5]; b4 = c[:, 4::5]
    d0 = b0 | ((b1 & 0x03) << 8)
    d1 = (b1 >> 2) | ((b2 & 0x0f) << 6)
    d2 = (b2 >> 4) | ((b3 & 0x3f) << 4)
    d3 = (b3 >> 6) | (b4 << 2)
    r = np.empty((n, 256), dtype=np.int64)
    r[:, 0::4] = i16(((d0 * KYBER_Q) + 512) >> 10)
    r[:, 1::4] = i16(((d1 * KYBER_Q) + 512) >> 10)
    r[:, 2::4] = i16(((d2 * KYBER_Q) + 512) >> 10)
    r[:, 3::4] = i16(((d3 * KYBER_Q) + 512) >> 10)
    return r

def v_ntt(r):
    r = r.copy(); k = 1
    for length in [128, 64, 32, 16, 8, 4, 2]:
        for start in range(0, 256, 2 * length):
            zeta = int(ZETAS[k]); k += 1
            lo = r[:, start:start + length]
            hi = r[:, start + length:start + 2 * length]
            t = v_fqmul(zeta, hi)
            r[:, start + length:start + 2 * length] = i16(lo - t)
            r[:, start:start + length] = i16(lo + t)
    return v_barrett(r)

def v_stored(sk, b, idx):
    """The value basemul writes to memory. Batch version of the original stored_value()."""
    i4 = idx // 4
    z = int(ZETAS[64 + i4])
    base = 4 * i4 if (idx % 4) < 2 else 4 * i4 + 2
    if (idx % 4) >= 2:
        z = -z
    a0, a1 = int(sk[base]), int(sk[base + 1])
    b0, b1 = b[:, base], b[:, base + 1]
    if (idx % 2) == 0:
        return i16(v_fqmul(v_fqmul(a1, b1), z) + v_fqmul(a0, b0))
    return i16(v_fqmul(a0, b1) + v_fqmul(a1, b0))


def make_labels(cts, sk, tag, n_max):
    """Builds labels once for up to n_max heads (= 2 x max pairs) and caches them. Each N slices from the front."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"lab_{LABEL_MODE}_{n_max}_{tag}_{len(cts)}.npy")
    if os.path.exists(cache):
        print(f"  loading label cache: {os.path.basename(cache)}")
        Y = np.load(cache)
    else:
        t0 = time.time()
        poly0 = v_ntt(v_unpack_10bit(cts))
        Y = np.zeros((len(cts), n_max), dtype=np.int64)
        for j in range(n_max):
            Y[:, j] = HW16[v_stored(sk, poly0, j) & 0xFFFF]
        np.save(cache, Y)
        print(f"  labels built {tag} ({time.time()-t0:.1f} s)")

    # label distribution check (the original's coverage check, on a 17-class basis)
    lo, hi = int(Y.min()), int(Y.max())
    cls0 = len(np.unique(Y[:, 0]))
    print(f"  [{tag}] HW range {lo}~{hi}, coefficient 0 has {cls0} classes "
          f"{'OK' if cls0 >= 13 else '[warning] low coverage'}")
    return Y


# ---------------- model ----------------
def _head(fin, num_targets):
    return nn.ModuleList([
        nn.Sequential(nn.BatchNorm1d(fin), nn.Dropout(DROPOUT),
                      nn.Linear(fin, NUM_CLASSES))
        for _ in range(num_targets)])

def _init_bias(classifiers, Y):
    for i, clf in enumerate(classifiers):
        cnt = np.bincount(Y[:, i], minlength=NUM_CLASSES).astype(np.float64)
        pri = (cnt + 1e-6) / cnt.sum()
        clf[-1].bias.copy_(torch.from_numpy(np.log(pri)).float())


class ResidualBlock1d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm1d(cout))
    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.sc(x))


class ResNet(nn.Module):
    """Chooses the trunk size via RESNET_VARIANT.
       resnet18 : stem 64ch, (64,128,256,512) x2  — same as the ESL re-experiment
       tifs     : stem 16ch, (16,32,64) x2        — same as the original TIFS submission
    """
    def __init__(self, num_targets):
        super().__init__()
        if RESNET_VARIANT == "tifs":
            widths, blocks, stem_ch, pool = (16, 32, 64), 2, 16, 4
        else:
            widths, blocks, stem_ch, pool = (64, 128, 256, 512), 2, 64, 5
        self.in_ch = stem_ch
        self.stem = nn.Sequential(
            nn.Conv1d(1, stem_ch, 11, 2, 5, bias=False), nn.BatchNorm1d(stem_ch),
            nn.ReLU(), nn.MaxPool1d(pool, pool))
        layers = []
        for i, w in enumerate(widths):
            layers.append(self._mk(w, blocks, 1 if i == 0 else 2))
        self.body = nn.Sequential(*layers)
        self.classifiers = _head(widths[-1], num_targets)

    def _mk(self, cout, n, stride):
        L = []
        for s in [stride] + [1] * (n - 1):
            L.append(ResidualBlock1d(self.in_ch, cout, s)); self.in_ch = cout
        return nn.Sequential(*L)

    def forward(self, x):
        x = self.body(self.stem(x)).mean(dim=2)
        return [c(x) for c in self.classifiers]

    @torch.no_grad()
    def init_bias(self, Y): _init_bias(self.classifiers, Y)


class DWSep1d(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, s, p, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm1d(cin); self.bn2 = nn.BatchNorm1d(cout)
    def forward(self, x):
        x = F.relu6(self.bn1(self.dw(x)))
        return F.relu6(self.bn2(self.pw(x)))


class InvRes1d(nn.Module):
    def __init__(self, cin, cout, stride, t):
        super().__init__()
        hid = int(cin * t); self.use_res = (stride == 1 and cin == cout)
        L = []
        if t != 1:
            L += [nn.Conv1d(cin, hid, 1, 1, 0, bias=False),
                  nn.BatchNorm1d(hid), nn.ReLU6()]
        L.append(DWSep1d(hid, cout, 3, stride, 1))
        self.conv = nn.Sequential(*L)
    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileNetV2(nn.Module):
    CFG = [[1,16,1,1],[6,24,2,2],[6,32,3,2],[6,64,4,2],
           [6,96,3,1],[6,160,3,2],[6,320,1,1]]
    def __init__(self, num_targets):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 11, 2, 5, bias=False), nn.BatchNorm1d(32),
            nn.ReLU6(), nn.MaxPool1d(5, 5))
        L, cin = [], 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                L.append(InvRes1d(cin, c, s if i == 0 else 1, t)); cin = c
        self.features = nn.Sequential(*L)
        self.proj = nn.Sequential(nn.Conv1d(cin, 1280, 1, 1, 0, bias=False),
                                  nn.BatchNorm1d(1280), nn.ReLU6())
        self.classifiers = _head(1280, num_targets)
    def forward(self, x):
        x = self.proj(self.features(self.stem(x))).mean(dim=2)
        return [c(x) for c in self.classifiers]
    @torch.no_grad()
    def init_bias(self, Y): _init_bias(self.classifiers, Y)


def build_model(arch, num_targets):
    return {"resnet": ResNet, "mobilenet": MobileNetV2}[arch](num_targets)


# ---------------- training ----------------
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, dev, nt):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss = 0.0
    correct = np.zeros(nt, np.int64)
    rank = np.zeros(nt, np.float64)
    seen = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        outs = model(x); seen += x.size(0)
        for i in range(nt):
            o = outs[i]
            loss += ce(o, y[:, i]).item()
            correct[i] += (o.argmax(1) == y[:, i]).sum().item()
            ts = o.gather(1, y[:, i:i + 1])
            rank[i] += ((o > ts).sum(1) + 1).float().sum().item()   # 1-based
    return loss / (seen * nt), correct / seen * 100, rank / seen


def _pnorm(params):
    sq = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            sq += float(g.norm().item()) ** 2
    return sq ** 0.5


def grad_group_norms(model):
    """Based on the current .grad (after unscaling, before clipping).
    Returns the backbone norm, the mean per-head norm of signal heads, and the mean per-head norm of phantom heads."""
    backbone = _pnorm(p for nm, p in model.named_parameters()
                      if not nm.startswith("classifiers."))
    sig, pha = [], []
    for i, clf in enumerate(model.classifiers):
        nrm = _pnorm(clf.parameters())
        (sig if i < SIGNAL_HEADS else pha).append(nrm)
    gn_sig = float(np.mean(sig)) if sig else 0.0
    gn_pha = float(np.mean(pha)) if pha else None   # None -> empty CSV cell
    return backbone, gn_sig, gn_pha


def run_single(arch, n, seed, Xtr, Ytr, Xva, Yva, dev, run_id, gn_writer):
    # n = number of pairs, h = number of heads = 2n. Ytr/Yva are already passed with h columns.
    h = heads_of(n)
    set_seed(seed)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr).unsqueeze(1),
                                  torch.from_numpy(Ytr)),
                    batch_size=BATCH, shuffle=True, drop_last=True)
    va = DataLoader(TensorDataset(torch.from_numpy(Xva).unsqueeze(1),
                                  torch.from_numpy(Yva)),
                    batch_size=256, shuffle=False)

    model = build_model(arch, h).to(dev)
    model.init_bias(Ytr)
    n_param = sum(p.numel() for p in model.parameters())

    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    # VRAM tracking (reset the peak so only what this run used is measured)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    vram_samples = []

    best, best_p, stale = 0.0, 0.0, 0
    best_acc, best_ge, best_t = None, None, 0.0
    ep_run = 0
    ckpt = os.path.join(SAVE_DIR, f"{arch}_N{n}_s{seed}.pth")
    t0 = time.time()

    try:
        for ep in range(1, EPOCHS + 1):
            ep_run = ep
            model.train()
            tot, nb, gns = 0.0, 0, []
            g_tot, g_bb, g_sig, g_pha, n_clip = [], [], [], [], 0
            for x, y in tr:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outs = model(x)
                        loss = sum(crit(outs[i], y[:, i]) for i in range(h)) / h
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    bb, gs, gp = grad_group_norms(model)           # before clipping
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(opt); scaler.update()
                else:
                    outs = model(x)
                    loss = sum(crit(outs[i], y[:, i]) for i in range(h)) / h
                    loss.backward()
                    bb, gs, gp = grad_group_norms(model)           # before clipping
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    opt.step()
                tot += loss.item(); nb += 1
                if np.isfinite(float(gn)):
                    gns.append(float(gn))
                    g_tot.append(float(gn)); g_bb.append(bb); g_sig.append(gs)
                    if gp is not None:
                        g_pha.append(gp)
                    if float(gn) > GRAD_CLIP:
                        n_clip += 1
                if dev.type == "cuda":
                    vram_samples.append(torch.cuda.memory_allocated(dev) / (1024 ** 2))

            # per-epoch gradient norm record (batch mean)
            gn_writer.writerow({
                "run_id": run_id, "arch": arch, "n_pairs": n, "seed": seed,
                "epoch": ep,
                "gn_total": round(float(np.mean(g_tot)), 5) if g_tot else "",
                "gn_backbone": round(float(np.mean(g_bb)), 5) if g_bb else "",
                "gn_heads_signal": round(float(np.mean(g_sig)), 5) if g_sig else "",
                "gn_heads_phantom": round(float(np.mean(g_pha)), 5) if g_pha else "",
                "clipped": round(n_clip / nb, 3) if nb else "",
            })

            vl, vacc, vge = evaluate(model, va, dev, h)
            sch.step()
            print(f"    [{arch} N{n}({h}h) s{seed} {ep:03d}/{EPOCHS}] "
                  f"train {tot/nb:.4f} | val {vl:.4f} | acc {vacc.mean():.2f}% "
                  f"| GE {vge.mean():.3f} | gn {np.median(gns):.3f}")

            cur = vacc.mean()
            if cur > best:
                best = cur
                best_acc, best_ge = vacc.copy(), vge.copy()
                best_t = time.time() - t0
                torch.save({"model": model.state_dict(), "arch": arch,
                            "resnet_variant": RESNET_VARIANT, "N": n, "heads": h,
                            "seed": seed, "crop": (CROP_LO, CROP_HI),
                            "leak_pairs": LEAK_PAIRS, "label_mode": LABEL_MODE,
                            "acc": vacc, "ge": vge}, ckpt)
            if cur > best_p + MIN_DELTA:
                best_p, stale = cur, 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
    except KeyboardInterrupt:
        print(f"\n  [{arch} N{n} s{seed}] stopped by user — ending only this run and moving to the next")

    elapsed = time.time() - t0
    if dev.type == "cuda":
        vram_max = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
        vram_avg = float(np.mean(vram_samples)) if vram_samples else 0.0
    else:
        vram_max = vram_avg = 0.0

    row = {
        # identification
        "run_id": run_id,
        "arch": arch,
        "resnet_variant": RESNET_VARIANT if arch == "resnet" else "",
        "n_pairs": n,
        "heads": h,
        "seed": seed,
        # training settings (for answering reviewers)
        "lr": LR,
        "batch": BATCH,
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "grad_clip": GRAD_CLIP,
        "loss_reduction": "mean",
        "label_type": LABEL_MODE,
        "phantom_label": PHANTOM_LABEL,
        "crop_lo": CROP_LO, "crop_hi": CROP_HI, "leak_pairs": LEAK_PAIRS,
        "git_commit": GIT_COMMIT,
        # training results (mean over all heads including phantom — for monitoring)
        "val_acc_allheads": round(best, 4),
        "ge_allheads": round(float(best_ge.mean()), 4) if best_ge is not None else "",
        "epochs_run": ep_run,
        "params_m": round(n_param / 1e6, 3),
        "total_time_min": round(elapsed / 60, 2),
        "best_model_time_min": round(best_t / 60, 2),
        "vram_max_mb": round(vram_max, 1),
        "vram_avg_mb": round(vram_avg, 1),
    }
    print(f"  [{arch} N{n}({h}h) s{seed}] acc {best:.2f}% | GE {row['ge_allheads']} | "
          f"{row['params_m']}M | VRAM max {vram_max:.0f}MB avg {vram_avg:.0f}MB | "
          f"{row['total_time_min']} min")

    # per-head details go separately into npz (only means in the CSV). N is the number of pairs.
    if best_acc is not None:
        np.savez(os.path.join(SAVE_DIR, f"heads_{arch}_N{n}_s{seed}.npz"),
                 acc=best_acc, ge=best_ge, n_pairs=n, heads=h)

    del model, opt, scaler
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return row


# file 1 — run summary (train_summary.csv, 1 row per run)
SUMMARY_FIELDS = [
    "run_id", "arch", "resnet_variant", "n_pairs", "heads", "seed",
    "lr", "batch", "max_epochs", "patience", "grad_clip", "loss_reduction",
    "label_type", "phantom_label", "crop_lo", "crop_hi", "leak_pairs",
    "git_commit",
    "val_acc_allheads", "ge_allheads", "epochs_run", "params_m",
    "total_time_min", "best_model_time_min", "vram_max_mb", "vram_avg_mb",
]

# file 2 — per-epoch gradient norm (train_gradnorm.csv, run x epoch)
GN_FIELDS = ["run_id", "arch", "n_pairs", "seed", "epoch",
             "gn_total", "gn_backbone", "gn_heads_signal",
             "gn_heads_phantom", "clipped"]


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary(rows):
    print("\n" + "=" * 86)
    print("N-scan summary (mean ± std over seeds)   * = N == L")
    print("=" * 86)
    print(f"{'arch':<11s}{'N(pair)':>8s}{'heads':>7s}{'acc %':>16s}{'GE':>16s}"
          f"{'VRAM max MB':>14s}{'min/run':>10s}")
    print("-" * 86)
    for arch in ARCHS:
        for n in N_LIST:
            sub = [r for r in rows if r["arch"] == arch and r["n_pairs"] == n]
            if not sub:
                continue
            a = np.array([r["val_acc_allheads"] for r in sub], float)
            g = np.array([r["ge_allheads"] for r in sub if r["ge_allheads"] != ""], float)
            v = np.array([r["vram_max_mb"] for r in sub], float)
            t = np.array([r["total_time_min"] for r in sub], float)
            gs = f"{g.mean():.3f}±{g.std():.3f}" if len(g) else "N/A"
            mark = " *" if n == LEAK_PAIRS else ""
            print(f"{arch:<11s}{n:>8d}{heads_of(n):>7d}{a.mean():>9.2f}±{a.std():<6.2f}"
                  f"{gs:>16s}{v.mean():>14.0f}{t.mean():>10.1f}{mark}")
    print("=" * 86)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_max = heads_of(max(N_LIST))          # labels are built per head up to the maximum
    total = len(ARCHS) * len(N_LIST) * len(SEEDS)
    print(f"device {dev} | window [{CROP_LO}, {CROP_HI}) | L={LEAK_PAIRS} pairs | "
          f"labels {LABEL_MODE}({NUM_CLASSES})")
    print(f"grid: {ARCHS} x N(pairs)={N_LIST} (heads {[heads_of(n) for n in N_LIST]}) "
          f"x {len(SEEDS)} seeds = {total} runs total\n")

    def load(name):
        d = np.load(os.path.join(SPLIT_DIR, f"kyber_{name}_crop.npz"))
        return (d["traces"][:, CROP_LO:CROP_HI].astype(np.float32),
                np.asarray(d["ciphertexts"], np.uint8),
                np.asarray(d["labels"], np.int64))

    Xtr, ct_tr, lab_tr = load("train")
    Xva, ct_va, _ = load("valid")
    sk = lab_tr[0]
    print(f"train {Xtr.shape}  valid {Xva.shape}")

    Ytr_all = make_labels(ct_tr, sk, "train", h_max)
    Yva_all = make_labels(ct_va, sk, "valid", h_max)

    # per-position z-score (profiling statistics) — saved for the evaluation scripts to use
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    np.savez(os.path.join(SAVE_DIR, "norm.npz"), mu=mu, sd=sd,
             crop=np.array([CROP_LO, CROP_HI]))

    # phantom control group: labels outside the signal heads (0~SIGNAL_HEADS-1) shuffled row-wise
    if PHANTOM_LABEL == "shuffled" and h_max > SIGNAL_HEADS:
        rng = np.random.default_rng(12345)
        for Y in (Ytr_all, Yva_all):
            for j in range(SIGNAL_HEADS, h_max):
                Y[:, j] = Y[rng.permutation(len(Y)), j]
        print(f"phantom_label=shuffled: labels of heads {SIGNAL_HEADS}~{h_max-1} shuffled\n")

    base = np.array([np.bincount(Yva_all[:, i], minlength=NUM_CLASSES).max()
                     / len(Yva_all) * 100 for i in range(h_max)])
    print(f"no-training baseline {base.mean():.2f}%  (git {GIT_COMMIT or 'n/a'})\n")

    # the per-epoch gradient norm CSV is appended in streaming fashion (run x epoch, so it is large)
    gn_exists = os.path.exists(GN_CSV)
    gn_f = open(GN_CSV, "a", newline="", encoding="utf-8")
    gn_writer = csv.DictWriter(gn_f, fieldnames=GN_FIELDS)
    if not gn_exists:
        gn_writer.writeheader()

    rows, done = [], 0
    for arch in ARCHS:
        for n in N_LIST:
            for seed in SEEDS:
                done += 1
                h = heads_of(n)
                run_id = f"{arch}_{RESNET_VARIANT if arch=='resnet' else 'na'}" \
                         f"_N{n}_s{seed}"
                print(f"\n[{done}/{total}] {run_id}  ({h} heads)")
                rows.append(run_single(arch, n, seed, Xtr, Ytr_all[:, :h],
                                       Xva, Yva_all[:, :h], dev, run_id, gn_writer))
                gn_f.flush()
                save_csv(rows, SUMMARY_CSV)    # saved immediately after every run

    gn_f.close()
    print_summary(rows)
    print(f"\nsummary CSV: {SUMMARY_CSV}\nper-epoch gradient CSV: {GN_CSV}")


if __name__ == "__main__":
    main()
