"""
TIFS — keep the backbone as is and attach only 39-pair heads (linear probe)

Question
--------
"Why not just ask a model trained with N=9 to predict all 39 pairs?"
-> As it stands it cannot, since there are only 18 output heads. But heads can be newly attached **without touching the backbone**.
   A head is a single linear layer feature vector -> 17 classes, so once the backbone features are extracted and cached, training takes a few seconds.

What this measures
------------------
"Does the N=9 backbone actually hold the information of all 39 pairs, and was only the output channel missing?"
  it holds it -> the probe heads also predict the unlearned pairs (9~38). This opens a strategy of training with a small N and only attaching heads
  it does not     -> the unlearned pairs stay at the prior. Means the backbone learns "only the requested pairs" = N decides the backbone itself
Either way it is a result. Unlike the shift probe (shift_probe.py), no peak-position constants are needed.

Fairness
--------
The original heads were trained **together** with the backbone, while the probe heads are trained **on top of a frozen backbone**. Comparing them directly is unfair, so
the probe retrains all 78 heads from scratch. Then the learned pairs (0~N-1) and the unlearned pairs (N~38) are under the same conditions.
  learned-pair probe acc  = the maximum extractable from this backbone (baseline under the same protocol)
  unlearned-pair probe acc = the answer to the question
For reference, the same probe is also run on the N=39 backbone to see the ceiling (REF_N, 10 seeds. To leave it out, REF_N=[]).

Threat model
------------
Training the probe heads requires the 39-pair labels of the profiling data. In the profiling phase the attacker
knows the key and ciphertexts of their own chip, so all 39-pair labels can be built -> an attack possible within the known-ciphertext model.

Run (same folder as eval_nscan.py, train_nscan.py)
    python probe_heads.py

Output <eval_nscan.OUT_DIR>/probe_heads/
    probe_pairs.csv    1 row per model x pair (acc, majority, info, whether it is a learned pair)
    probe_models.csv   1 row per model (summary)
    N{n}/{arch}_N39_s{seed}.pth + norm.npz
        -> frozen backbone + 78 probe heads. Pointing MODEL_DIR in eval_nscan.py at this folder with N_LIST=[39]
           runs 39-pair key recovery with the probe model as is
"""

import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

import eval_nscan as E

OUT = os.path.join(E.OUT_DIR, "probe_heads")

N_PROBE = [9, 19, 29]          # targets: models that have no 39-pair heads
SEEDS   = E.SEEDS
REF_N   = [39]                 # ceiling reference. It is the baseline of every comparison and N39 has a large seed spread (56~70%), so all 10 seeds
REF_SEEDS = SEEDS

FEAT_BATCH   = 128
PROBE_EPOCHS = 60
PROBE_LR     = 1e-3
PROBE_WD     = 3e-3
PROBE_BATCH  = 1024
PATIENCE     = 10
MIN_DELTA    = 0.1             # %p
CONV_MARGIN  = 1.0             # if the original heads do not exceed even this much, the model stalled -> separated in the summary

PAIR_FIELDS = ["arch", "src_n_pairs", "seed", "pair", "seen", "acc_r0", "acc_r1",
               "majority_r0", "majority_r1", "info_r0", "info_r1"]
MODEL_FIELDS = ["arch", "src_n_pairs", "seed", "src_converged", "feat_dim", "probe_epochs",
                "acc_seen", "majority_seen", "info_seen",
                "acc_unseen", "majority_unseen", "info_unseen",
                "acc_orig_heads", "pairs_unseen_over_50", "n_unseen", "time_min"]


# ---------------- restart cleanup ----------------
def repair(pp, mp):
    """Remove runs that died midway and return the set of finished runs.

    The completion condition is **both the CSV row and the checkpoint exist**. If only one exists, the run was
    interrupted while saving, so the CSV row is removed to make it run again (otherwise a rerun produces two rows).
    Same principle as repair in run_controls.py. The checkpoint is saved before the CSV, so
    the case actually caught is almost always 'no CSV, only .pth', and that gets overwritten.
    """
    if not os.path.exists(mp):
        return set()
    rows = list(csv.DictReader(open(mp, encoding="utf-8")))
    done, drop = set(), set()
    for r in rows:
        k = (r["arch"], int(r["src_n_pairs"]), int(r["seed"]))
        ck = os.path.join(OUT, f"N{k[1]}", f"{k[0]}_N{E.LEAK_PAIRS}_s{k[2]}.pth")
        ok = os.path.exists(ck)
        if ok and os.path.getsize(ck) > 0:
            done.add(k)
        else:
            drop.add(k)
            print(f"  [rerun] {k[0]} N{k[1]} s{k[2]} — checkpoint missing or empty")
    if drop:
        keep = lambda r: (r["arch"], int(r["src_n_pairs"]), int(r["seed"])) not in drop
        for path, fields in ((mp, MODEL_FIELDS), (pp, PAIR_FIELDS)):
            if not os.path.exists(path):
                continue
            kept = [r for r in csv.DictReader(open(path, encoding="utf-8")) if keep(r)]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in kept:
                    w.writerow({c: r.get(c, "") for c in fields})
            print(f"  {os.path.basename(path)} cleaned → {len(kept)} rows")
    if done:
        print(f"  skipping {len(done)} finished runs")
    return done


# ---------------- extract backbone features ----------------
class Feat(nn.Module):
    """Replace the model's head list with a single Identity so that forward returns the feature vector.
    (uses the fact that train_nscan's forward has the form `return [c(x) for c in self.classifiers]`)"""
    def __init__(self, model):
        super().__init__()
        self.m = model
        self.keep = model.classifiers
        model.classifiers = nn.ModuleList([nn.Identity()])
    def forward(self, x):
        o = self.m(x)
        if not (isinstance(o, (list, tuple)) and len(o) == 1 and o[0].dim() == 2):
            raise SystemExit("Could not extract backbone features — forward is not of the form [head(x) ...]. Please report this")
        return o[0]
    def restore(self):
        self.m.classifiers = self.keep


def extract(model, X, dev):
    f = Feat(model).to(dev).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), FEAT_BATCH):
            xb = torch.from_numpy(X[s:s + FEAT_BATCH]).unsqueeze(1).to(dev)
            out.append(f(xb).float().cpu())
    f.restore()
    return torch.cat(out)


# ---------------- probe heads ----------------
def make_heads(T, fin, n_heads, Ytr, ncls, dev):
    drop = float(getattr(T, "DROPOUT", 0.4))
    heads = nn.ModuleList([nn.Sequential(nn.BatchNorm1d(fin), nn.Dropout(drop),
                                         nn.Linear(fin, ncls)) for _ in range(n_heads)]).to(dev)
    with torch.no_grad():                                   # same bias as the original training = log prior
        for i, h in enumerate(heads):
            cnt = np.bincount(Ytr[:, i], minlength=ncls).astype(np.float64)
            h[-1].bias.copy_(torch.from_numpy(np.log((cnt + 1e-6) / cnt.sum())).float())
    return heads


def train_heads(heads, Ftr, Ytr, Fva, Yva, dev, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(heads.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PROBE_EPOCHS)
    crit = nn.CrossEntropyLoss()
    nh = len(heads)
    Ftr, Fva = Ftr.to(dev), Fva.to(dev)
    Ytr_t = torch.from_numpy(Ytr).to(dev)
    Yva_t = torch.from_numpy(Yva).to(dev)
    best, best_state, mark, cnt, ep = -1e9, None, -1e9, 0, 0
    for ep in range(1, PROBE_EPOCHS + 1):
        heads.train()
        perm = torch.randperm(len(Ftr), device=dev)
        for s in range(0, len(perm), PROBE_BATCH):
            b = perm[s:s + PROBE_BATCH]
            if len(b) < 2:
                continue
            f, y = Ftr[b], Ytr_t[b]
            loss = sum(crit(h(f), y[:, i]) for i, h in enumerate(heads)) / nh
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sch.step()
        heads.eval()
        with torch.no_grad():
            acc = float(np.mean([
                (torch.cat([h(Fva[s:s + 4096]).argmax(1) for s in range(0, len(Fva), 4096)])
                 == Yva_t[:, i]).float().mean().item() * 100
                for i, h in enumerate(heads)]))
        if acc > best:                                     # save on any improvement at all
            best, best_state = acc, {k: v.detach().clone() for k, v in heads.state_dict().items()}
        if acc > mark + MIN_DELTA:                         # patience resets only on a rise of more than MIN_DELTA
            mark, cnt = acc, 0
        else:
            cnt += 1
            if cnt >= PATIENCE:
                break
    heads.load_state_dict(best_state)
    return best, ep


def eval_heads(heads, Fte, Y, logpr, dev):
    heads.eval()
    F = Fte.to(dev)
    acc, info = [], []
    with torch.no_grad():
        for i, h in enumerate(heads):
            lp = torch.cat([torch.log_softmax(h(F[s:s + 4096]).float(), 1)
                            for s in range(0, len(F), 4096)]).cpu().numpy()
            y = Y[:, i]
            acc.append(float((lp.argmax(1) == y).mean() * 100))
            info.append(float((lp[np.arange(len(y)), y] - logpr[i, y]).mean() / np.log(2.0)))
    return np.array(acc), np.array(info)


def main():
    os.makedirs(OUT, exist_ok=True)
    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ncls = int(getattr(T, "NUM_CLASSES", 17))
    H = 2 * E.LEAK_PAIRS                                    # 78
    print(f"device {dev} | target N {N_PROBE} (+reference {REF_N}) | {H} probe heads")

    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    lo, hi = [int(v) for v in nz["crop"]]
    mu, sd = nz["mu"].astype(np.float32), nz["sd"].astype(np.float32)

    def load(name):
        d = np.load(os.path.join(E.SPLIT_DIR, f"kyber_{name}_crop.npz"))
        X = d["traces"][:, lo:hi].astype(np.float32)
        X -= mu
        X /= sd
        return X, np.asarray(d["ciphertexts"], np.uint8), np.asarray(d["labels"], np.int64)

    Xtr, ct_tr, lab = load("train")
    Xva, ct_va, _ = load("valid")
    Xte, ct_te, _ = load("test")
    sk = lab[0]
    lab_of = lambda ct: np.stack(
        [T.HW16[np.asarray(T.v_stored(sk, T.v_ntt(T.v_unpack_10bit(ct)), j)) & 0xFFFF]
         for j in range(H)], 1)
    Ytr, Yva, Yte_all = lab_of(ct_tr), lab_of(ct_va), lab_of(ct_te)
    prior = np.stack([(np.bincount(Yte_all[:, j], minlength=ncls) + 1.0) /
                      (len(Yte_all) + ncls) for j in range(H)])
    logpr, maj = np.log(prior), prior.max(1) * 100

    ip = os.path.join(E.OUT_DIR, "eval_indices.npz")
    te_idx = np.load(ip)["idx"] if os.path.exists(ip) else np.sort(
        np.random.default_rng(E.EVAL_SEED).choice(len(Xte), E.EVAL_N, replace=False))
    Xte, Yte = Xte[te_idx], Yte_all[te_idx]
    print(f"train {Xtr.shape} valid {Xva.shape} test {Xte.shape}")

    pp, mp = os.path.join(OUT, "probe_pairs.csv"), os.path.join(OUT, "probe_models.csv")
    done = repair(pp, mp)

    jobs = [(a, n, s) for a in E.ARCHS for n in N_PROBE for s in SEEDS]
    jobs += [(a, n, s) for a in E.ARCHS for n in REF_N for s in REF_SEEDS]
    t0 = time.time()
    for arch, n, seed in jobs:
        if (arch, n, seed) in done:
            continue
        f = os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")
        if not os.path.exists(f):
            print(f"  [missing] {os.path.basename(f)}")
            continue
        t1 = time.time()
        ck = torch.load(f, map_location="cpu", weights_only=False)
        src_heads = int(ck.get("heads", 2 * n))
        model = E.build(T, arch, n, src_heads, ck.get("resnet_variant")).to(dev)
        E.load_state(model, ck["model"])
        model.eval()

        # performance of the original heads on their own pairs (to identify stalled models)
        with torch.no_grad():
            k = min(512, len(Xte))
            xb = torch.from_numpy(Xte[:k]).unsqueeze(1).to(dev)
            outs = model(xb)
            a_orig = float(np.mean([
                (o.argmax(1).cpu().numpy() == Yte[:k, i]).mean() * 100
                for i, o in enumerate(outs) if i < H]))
        src_conv = a_orig - float(maj[:min(src_heads, H)].mean()) > CONV_MARGIN

        Ftr = extract(model, Xtr, dev)
        Fva = extract(model, Xva, dev)
        Fte = extract(model, Xte, dev)
        fin = Ftr.shape[1]
        for p in model.parameters():                        # freeze the backbone (train only the probe heads)
            p.requires_grad_(False)

        heads = make_heads(T, fin, H, Ytr, ncls, dev)
        vacc, eps = train_heads(heads, Ftr, Ytr, Fva, Yva, dev, seed)
        acc, info = eval_heads(heads, Fte, Yte, logpr, dev)

        seen = np.zeros(H, bool)
        seen[:2 * min(n, E.LEAK_PAIRS)] = True
        g = lambda v, m: round(float(v[m].mean()), 3) if m.any() else ""
        pair_acc = (acc[0::2] + acc[1::2]) / 2
        unseen_p = ~seen[0::2]
        row = dict(arch=arch, src_n_pairs=n, seed=seed, src_converged=int(src_conv),
                   feat_dim=fin, probe_epochs=eps,
                   acc_seen=g(acc, seen), majority_seen=g(maj, seen), info_seen=g(info, seen),
                   acc_unseen=g(acc, ~seen), majority_unseen=g(maj, ~seen),
                   info_unseen=g(info, ~seen), acc_orig_heads=round(a_orig, 3),
                   pairs_unseen_over_50=int((pair_acc[unseen_p] >= 50).sum()),
                   n_unseen=int(unseen_p.sum()), time_min=round((time.time() - t1) / 60, 2))
        prows = [dict(arch=arch, src_n_pairs=n, seed=seed, pair=p, seen=int(seen[2 * p]),
                      acc_r0=round(float(acc[2 * p]), 3), acc_r1=round(float(acc[2 * p + 1]), 3),
                      majority_r0=round(float(maj[2 * p]), 3),
                      majority_r1=round(float(maj[2 * p + 1]), 3),
                      info_r0=round(float(info[2 * p]), 5), info_r1=round(float(info[2 * p + 1]), 5))
                 for p in range(E.LEAK_PAIRS)]
        # Save the checkpoint before the CSV rows — if the order were reversed and the run died in between,
        # the restart would skip it as "already done", leaving a state with only the .pth missing, and since running
        # key recovery with this .pth is the purpose of this experiment, an empty folder would be fatal
        sub = os.path.join(OUT, f"N{n}")                    # this folder, for running key recovery with eval_nscan
        os.makedirs(sub, exist_ok=True)
        if not os.path.exists(os.path.join(sub, "norm.npz")):
            np.savez(os.path.join(sub, "norm.npz"), mu=mu, sd=sd, crop=np.array([lo, hi]))
        full = E.build(T, arch, E.LEAK_PAIRS, H, ck.get("resnet_variant"))
        bb = {k: v for k, v in model.state_dict().items() if not k.startswith("classifiers.")}
        full.load_state_dict({**bb, **{f"classifiers.{k}": v.cpu()
                                       for k, v in heads.state_dict().items()}})
        torch.save(dict(model=full.state_dict(), arch=arch, resnet_variant=ck.get("resnet_variant"),
                        N=E.LEAK_PAIRS, heads=H, seed=seed, crop=(lo, hi),
                        leak_pairs=E.LEAK_PAIRS, label_mode="hw16", acc=acc,
                        probe_from_N=n, probe_val_acc=vacc),
                   os.path.join(sub, f"{arch}_N{E.LEAK_PAIRS}_s{seed}.pth"))

        E.append_rows(pp, PAIR_FIELDS, prows)
        E.append_rows(mp, MODEL_FIELDS, [row])

        print(f"{arch:<10}N{n:<3}s{seed} conv {int(src_conv)} | original heads {a_orig:.2f}% | "
            f"probe learned pairs {row['acc_seen']}% / unlearned pairs {row['acc_unseen']}% "
            f"(baseline {row['majority_unseen']}%) | unlearned pairs over 50% "
            f"{row['pairs_unseen_over_50']}/{row['n_unseen']} | {row['time_min']} min "
            f"(cumulative {(time.time()-t0)/60:.0f} min)")
        del model, heads, Ftr, Fva, Fte
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    summary(mp)


def summary(mp):
    rows = list(csv.DictReader(open(mp, encoding="utf-8")))
    print("\n" + "=" * 100)
    print("Frozen backbone + 78 new heads (probe). 'unlearned pairs' = pairs that backbone did not see during training")
    print("=" * 100)
    print(f"{'arch':<10}{'bb_N':>6}{'esc':>6}{'orig_heads':>10}{'probe_seen':>13}"
          f"{'probe_unseen':>15}{'base(mode)':>12}{'pairs>50%':>11}")
    print("-" * 100)
    for arch in E.ARCHS:
        for n in N_PROBE + REF_N:
            rs = [r for r in rows if r["arch"] == arch and int(r["src_n_pairs"]) == n]
            if not rs:
                continue
            cv = [r for r in rs if r["src_converged"] == "1"]
            use = cv or rs
            f = lambda k: np.array([float(r[k]) for r in use if r[k] != ""])
            m = lambda k: f"{f(k).mean():.2f}" if f(k).size else "-"
            cov = f("pairs_unseen_over_50")
            nu = f("n_unseen")
            print(f"{arch:<10}{n:>6}{len(cv):>3}/{len(rs):<2}{m('acc_orig_heads'):>10}"
                  f"{m('acc_seen'):>13}{m('acc_unseen'):>15}{m('majority_unseen'):>12}"
                  f"{(f'{cov.mean():.1f}/{nu.mean():.0f}' if cov.size and nu.mean() else '-'):>11}")
    print("=" * 100)
    print("How to read")
    print("  unlearned pairs ≈ learned pairs   -> the backbone held all 39 pairs. Just attach heads (N is only the number of channels)")
    print("  unlearned pairs ≈ baseline      -> the backbone learns only the requested pairs. N decides the backbone itself")
    print("  The N=39 row is the ceiling. Same protocol (frozen backbone + new heads), so directly comparable")
    print("  To also see key recovery, set MODEL_DIR in eval_nscan to probe_heads/N<n> and N_LIST=[39]")


if __name__ == "__main__":
    main()
