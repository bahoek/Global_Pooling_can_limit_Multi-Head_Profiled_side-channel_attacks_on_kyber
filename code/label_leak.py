"""
Label confounding test — is probe_heads' 'unlearned pair' performance trace transfer or label correlation?

Why this is needed
------------------
probe_heads.py trains 78 new heads on the features of a frozen backbone. Those features contain **the values of the
learned pairs** (that is what the backbone does). So if the label of pair 30 is statistically correlated with the
labels of pairs 0~28, the pair-30 head does not read pair 30 from the trace; it scores by **reading pair 5 and
inferring pair 30**. Then the 'unlearned pair' figure is confounding, not transfer.

The observed evidence points this way:
  * two backbones with completely different architectures read **the same pairs** (30·34·37) (per-pair info correlation r = +0.876)
  * the NICV leakage of those pairs is not among the top (pair 37 ranks 34th of 39)
  * they do not match the pairs well covered by the push probe either (r = +0.41 / +0.05)
If it were a property of the features, there would be no reason for two architecturally different backbones to agree this closely.

What is measured
----------------
Only the label arrays are used. Traces are not read at all (npz is read lazily per array, so
the `traces` key is never touched).

  (1) per-pair mutual information I(Y_j ; Y_k) — how much the label of unlearned pair j is explained by
      the label of learned pair k. The path "the head reads k and infers j" is inherently pairwise
  (2) label-only predictor — takes the **true labels** of the learned pairs as input and predicts pair j.
      Since it is given the true labels, it is an **upper bound** for this path. The probe head reads imperfectly through
      the features, so the actual confounding is smaller than this

Verdict
-------
  upper bound ≈ 0            -> probe's 'unlearned pair' figure is trace transfer. Use the result as-is
  upper bound ≈ probe value  -> confounding. The 'unlearned pair' figure cannot be interpreted as transfer
  in between                 -> the two paths are mixed. The difference is the transfer share

info_bits has the same definition as in eval_nscan·probe_heads: (mean log p(true) − mean log prior(true)) / ln2.
The prior is likewise taken over the whole test set. Evaluation rows use the 2000 traces of eval_indices.npz as-is,
so the result is directly comparable with the probe results.

Run
---
    python label_leak.py

Place it in the same folder as eval_nscan.py (paths·seed·EVAL_N are taken from there).
If sklearn is available it does up to (2), otherwise only (1).

Output
------
  <OUT_DIR>/probe_heads/label_leak.csv   info/acc per pair x N x path
  comparison table side by side with the measured probe values on screen
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_nscan as E

OUT = os.path.join(E.OUT_DIR, "probe_heads")
N_SRC = [9, 19, 29]            # same backbone N as the probe
PAIRS = list(range(29, 39))    # common unlearned set (pairs unlearned under all three N)
TOP_K = 8                      # number of top input pairs fed to the predictor (by mutual information)
LN2 = np.log(2.0)

FIELDS = ["src_n_pairs", "pair", "role", "mi_best_bits", "mi_best_from_pair", "mi_sum_top8_bits",
          "pred_info_bits", "pred_acc", "majority_acc", "n_inputs"]


def labels_of(T, path):
    """Read only ciphertexts·labels from the npz and build the label matrix. traces are not read."""
    d = np.load(path)
    ct = np.asarray(d["ciphertexts"], np.uint8)
    sk = np.asarray(d["labels"][0], np.int64)
    uh = T.v_ntt(T.v_unpack_10bit(ct))
    h = 2 * E.LEAK_PAIRS
    Y = np.stack([T.HW16[np.asarray(T.v_stored(sk, uh, j)) & 0xFFFF] for j in range(h)], 1)
    return Y.astype(np.int64)


def mutual_info(a, b, ncls):
    """I(a;b) in bits, directly from the joint histogram."""
    j = np.bincount(a * ncls + b, minlength=ncls * ncls).reshape(ncls, ncls).astype(np.float64)
    j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return float((j[nz] * np.log(j[nz] / (pa @ pb)[nz])).sum() / LN2)


def main():
    T = E.import_train()
    ncls = int(getattr(T, "NUM_CLASSES", 17))

    Ytr = labels_of(T, os.path.join(E.SPLIT_DIR, "kyber_train_crop.npz"))
    Yte_all = labels_of(T, os.path.join(E.SPLIT_DIR, "kyber_test_crop.npz"))
    print(f"labels only — train {Ytr.shape}  test {Yte_all.shape}  (traces not used)")

    # match the prior and evaluation rows exactly to eval_nscan / probe_heads
    prior = np.stack([(np.bincount(Yte_all[:, j], minlength=ncls) + 1.0) /
                      (len(Yte_all) + ncls) for j in range(Yte_all.shape[1])])
    logpr = np.log(prior)
    ip = os.path.join(E.OUT_DIR, "eval_indices.npz")
    if os.path.exists(ip):
        idx = np.load(ip)["idx"]
        print(f"eval rows: {len(idx)} traces from eval_indices.npz (same as probe)")
    else:
        idx = np.sort(np.random.default_rng(E.EVAL_SEED).choice(
            len(Yte_all), E.EVAL_N, replace=False))
        print(f"eval rows: no eval_indices.npz, so {len(idx)} traces regenerated with the same seed")
    Yte = Yte_all[idx]

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        have_sk = True
    except Exception as e:
        have_sk = False
        print(f"[note] no sklearn ({e}) — producing mutual information only. "
              f"To also get the predictor: pip install --break-system-packages scikit-learn")

    rows = []
    for n in N_SRC:
        src = list(range(2 * n))                                  # head indices of the learned pairs
        print(f"\n{'='*92}\nbackbone N={n} — predict unlearned pairs from the labels of learned pairs 0~{n-1}\n{'='*92}")
        print(f"{'pair':>4}{'role':>5}{'MImax':>9}{'from':>7}{'MItop8sum':>11}"
              f"{'pred info':>12}{'pred acc':>11}{'major acc':>10}")
        for p in PAIRS:
            if p < n:
                continue
            for role, h in (("r0", 2 * p), ("r1", 2 * p + 1)):
                mis = np.array([mutual_info(Ytr[:, h], Ytr[:, k], ncls) for k in src])
                order = np.argsort(-mis)
                top = [src[k] for k in order[:TOP_K]]
                maj = float(prior[h].max() * 100)
                info = acc = None
                if have_sk:
                    clf = HistGradientBoostingClassifier(
                        max_iter=120, learning_rate=0.1, max_depth=None,
                        early_stopping=True, validation_fraction=0.1,
                        categorical_features=[True] * len(top), random_state=0)
                    clf.fit(Ytr[:, top], Ytr[:, h])
                    pr = np.clip(clf.predict_proba(Yte[:, top]), 1e-12, 1.0)
                    cls = clf.classes_
                    # probability of the true label (1e-12 for classes absent from training)
                    pos = {int(c): i for i, c in enumerate(cls)}
                    y = Yte[:, h]
                    pt = np.array([pr[i, pos[v]] if v in pos else 1e-12
                                   for i, v in enumerate(y)])
                    info = float((np.log(pt) - logpr[h][y]).mean() / LN2)
                    acc = float((cls[pr.argmax(1)] == y).mean() * 100)
                rows.append(dict(src_n_pairs=n, pair=p, role=role,
                                 mi_best_bits=round(float(mis.max()), 4),
                                 mi_best_from_pair=int(top[0] // 2),
                                 mi_sum_top8_bits=round(float(mis[order[:TOP_K]].sum()), 4),
                                 pred_info_bits="" if info is None else round(info, 4),
                                 pred_acc="" if acc is None else round(acc, 3),
                                 majority_acc=round(maj, 3), n_inputs=len(top)))
                print(f"{p:>4}{role:>5}{mis.max():>9.4f}{top[0]//2:>7}{mis[order[:TOP_K]].sum():>11.4f}"
                      f"{'-' if info is None else f'{info:>12.4f}'}"
                      f"{'-' if acc is None else f'{acc:>11.2f}'}{maj:>10.2f}")

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "label_leak.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\nsaved: {out}  ({len(rows)} rows)")
    verdict(rows, have_sk)


def verdict(rows, have_sk):
    """Verdict, side by side with the measured probe values."""
    pm = os.path.join(OUT, "probe_models.csv")
    pp = os.path.join(OUT, "probe_pairs.csv")
    probe = {}
    if os.path.exists(pp) and os.path.exists(pm):
        conv = {(r["arch"], int(r["src_n_pairs"]), int(r["seed"]))
                for r in csv.DictReader(open(pm, encoding="utf-8"))
                if r["src_converged"] == "1"}
        acc = {}
        for r in csv.DictReader(open(pp, encoding="utf-8")):
            k = (r["arch"], int(r["src_n_pairs"]), int(r["seed"]))
            if k not in conv or r["seen"] != "0":
                continue
            for role in ("r0", "r1"):
                acc.setdefault((r["arch"], int(r["src_n_pairs"]), int(r["pair"]), role), []) \
                   .append(float(r[f"info_{role}"]))
        probe = {k: float(np.mean(v)) for k, v in acc.items()}

    print("\n" + "=" * 92)
    print("Verdict — upper bound of the label path vs measured probe (info, bit)")
    print("=" * 92)
    if not have_sk:
        print("No sklearn, so predictor values are empty. Mutual information alone allows a first judgement:")
        print("  if MImax is near 0 everywhere, there is no label path and the probe figures are trace transfer.")
        return
    print(f"{'pair':>4}{'role':>5}{'N':>4}{'label UB':>10}{'probe RN':>10}{'probe MNv2':>12}"
          f"{'  verdict':<12}")
    hi = []
    for r in sorted(rows, key=lambda r: (r["pair"], r["role"], r["src_n_pairs"])):
        lab = r["pred_info_bits"]
        if lab == "":
            continue
        rn = probe.get(("resnet", r["src_n_pairs"], r["pair"], r["role"]))
        mn = probe.get(("mobilenet", r["src_n_pairs"], r["pair"], r["role"]))
        ref = max(v for v in (rn, mn, 0.0) if v is not None)
        if lab < 0.02:
            v = "no label path"
        elif ref > 0 and lab > 0.5 * ref:
            v = "confounding suspected"
            hi.append((r["pair"], r["role"], r["src_n_pairs"], lab, ref))
        else:
            v = "partly mixed"
        print(f"{r['pair']:>4}{r['role']:>5}{r['src_n_pairs']:>4}{lab:>10.4f}"
              f"{'-' if rn is None else f'{rn:>10.3f}'}"
              f"{'-' if mn is None else f'{mn:>12.3f}'}  {v:<12}")
    print("=" * 92)
    labs = [r["pred_info_bits"] for r in rows if r["pred_info_bits"] != ""]
    print(f"label upper bound overall: max {max(labs):.4f} bit, mean {np.mean(labs):.4f} bit")
    if max(labs) < 0.02:
        print("→ There is effectively no label path. probe's 'unlearned pair' figures may be interpreted as trace transfer.")
        print("  (even given all true labels of the learned pairs, the unlearned pairs cannot be predicted = labels are mutually independent)")
    elif hi:
        print(f"→ in {len(hi)} (pair,role,N) cases the label upper bound exceeds half the probe value. This is confounding.")
        print("  The 'unlearned pair' figures cannot be used as transfer. Exclude those pairs or report after subtracting the confounding share.")
    else:
        print("→ A label path exists but is smaller than the probe value. The difference is the trace transfer share.")


if __name__ == "__main__":
    main()
