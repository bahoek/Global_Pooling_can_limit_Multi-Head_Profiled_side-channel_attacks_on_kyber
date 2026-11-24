"""
TIFS — trace shift probe: do the heads of N<39 models also read "other pairs"?

Question
--------
The operation of pair k is the same basemul as pair 0, and the leaked value has the same form (stored-value HW). The only difference is the position in the waveform.
So if the trace is shifted to put pair k in the place of pair 0, the pair-0 head should read pair k.
If so, even an N=9 model (18 heads) could cover all 39 pairs.

If right: the pattern transfers as is between pairs -> train with a small N and cover by shifting. A paper result
If wrong: the heads memorized something other than the position too -> grounds that the current evaluation (min(N,L) pairs) is correct
Targets: only N < 39 models (N>=39 already has all 39-pair heads)

The peak positions use the NICV measurements as they are (nicv_pairs.py must have been run first)
--------------------------------------------------------------------------------------------------
Measured (10/30, 20k test traces): the pair spacing is not uniform and **984 / 1048 alternate** (mean 1016).
The within-pair r0 -> r1 offset also varies over 320~348. So shifting by a uniform spacing such as "k x 1016"
misaligns half of the heads by 32 samples for odd k. Here such a model is not used; instead:

  * distance needed for head pair j to read pair j+k = peak[2(j+k)] - peak[2j]   (r0-based, measured values)
  * if one k has several such values (odd k: two kinds), **shift separately per distance** and score only the heads matching each distance
    -> r0 is aligned exactly for every head
  * r1 cannot be matched with the same distance (the within-pair offset differs per pair), so the misalignment is recorded in err_r1.
    If r0 is right but only r1 drops, that "alignment precision matters" is itself a result

Controls
--------
  own(k=0)    own pair, no shift — baseline
  misaligned  exact distance + half the within-pair offset (about 169 samples) — lands between the peaks.
              if it is right here too, there is something unrelated to position alignment
  two normalization modes  after: shift then per-position z-score / before: z-score then shift
              (cannot know in advance whether destination-position or original-position statistics should be used, so both)

Run (same folder as eval_nscan.py, train_nscan.py. nicv_pairs.py first)
    python shift_probe.py

Output: <eval_nscan.OUT_DIR>/shift_probe/shift_heads.csv, shift_summary.csv
"""

import csv
import os
import time

import numpy as np
import torch

import eval_nscan as E

OUT = os.path.join(E.OUT_DIR, "shift_probe")
NICV_NPZ = os.path.join(E.OUT_DIR, "nicv", "nicv_curves.npz")

N_PROBE     = [9, 19, 29]        # only N<39
SEEDS       = E.SEEDS
PROBE_N     = 500                # number of traces used for inference (enough for the accuracy estimate, ±2%p)
SHIFTS      = list(range(1, 39)) # number of pairs to shift, k
MISALIGN_AT = [1, 5, 10, 20]     # for these k, also the misaligned control
NORM_MODES  = ["after", "before"]
CONV_MARGIN = 1.0                # own-pair acc - majority must exceed this for the model to count as 'escaped' and be included in the summary

# fallback values used only when nicv_curves.npz is missing (uniform-spacing assumption — ±32-sample error for odd k)
FALLBACK_PEAK0, FALLBACK_SPACING = 756.4, 507.97

HEAD_FIELDS = ["arch", "n_pairs", "seed", "norm_mode", "kind", "shift_pairs", "shift_samples",
               "head_pair", "target_pair", "err_r0", "err_r1",
               "acc_r0", "acc_r1", "majority_r0", "majority_r1", "info_r0", "info_r1"]
SUM_FIELDS = ["arch", "n_pairs", "seed", "norm_mode", "converged",
              "acc_own", "majority_own", "acc_shift1", "acc_shift_mean", "acc_shift_far",
              "info_own", "info_shift_mean", "acc_misaligned", "pairs_covered_50pct",
              "acc_r0_shift_mean", "acc_r1_shift_mean"]


def load_peaks():
    """Measured peak sample positions of coefficients 0..77."""
    H = 2 * E.LEAK_PAIRS
    if os.path.exists(NICV_NPZ):
        d = np.load(NICV_NPZ)
        c, p = np.asarray(d["coefs"]), np.asarray(d["peak_sample"])
        pk = np.full(H, -1, np.int64)
        for cc, pp in zip(c, p):
            if cc < H:
                pk[int(cc)] = int(pp)
        if (pk < 0).any():
            raise SystemExit("nicv_curves.npz does not contain all 78 signal coefficients — run nicv_pairs.py first")
        print(f"peak positions: using measured values ({NICV_NPZ})")
        d0 = np.diff(pk[0::2])
        print(f"  pair spacing {sorted(set(d0.tolist()))} (mean {d0.mean():.1f}), "
              f"within-pair r0->r1 offset {int((pk[1::2]-pk[0::2]).min())}~{int((pk[1::2]-pk[0::2]).max())}")
        return pk
    print(f"[warning] {NICV_NPZ} missing -> uniform-spacing assumption "
          f"(peak = {FALLBACK_PEAK0} + {FALLBACK_SPACING}·j). It is better to run nicv_pairs.py first")
    return np.round(FALLBACK_PEAK0 + FALLBACK_SPACING * np.arange(H)).astype(np.int64)


def build_plan(pk, n_max):
    """[(kind, k, shift, [head_pair j ...])] — groups the heads shifted by the same distance."""
    L = E.LEAK_PAIRS
    half = int(round(float(np.mean(pk[1::2] - pk[0::2])) / 2))
    plan = [("own", 0, 0, list(range(n_max)))]
    for k in SHIFTS:
        groups = {}
        for j in range(min(n_max, L - k)):
            groups.setdefault(int(pk[2 * (j + k)] - pk[2 * j]), []).append(j)
        for s, js in sorted(groups.items()):
            plan.append(("aligned", k, s, js))
        if k in MISALIGN_AT:
            s, js = max(groups.items(), key=lambda kv: len(kv[1]))
            plan.append(("misaligned", k, s + half, js))
    return plan, half


def shift_left(X, s, fill_row):
    if s <= 0:
        return X
    L = X.shape[1]
    if s >= L:
        raise SystemExit(f"shift distance {s} is at least the trace length {L} — check the position measurements")
    out = np.empty_like(X)
    out[:, :L - s] = X[:, s:]
    out[:, L - s:] = fill_row[L - s:]
    return out


def infer(model, X, heads, ncls, dev):
    lp = np.empty((len(X), heads, ncls), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), E.INFER_BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(X[s:s + E.INFER_BATCH])).unsqueeze(1).to(dev)
            outs = model(xb)
            lp[s:s + len(xb)] = torch.stack(
                [torch.log_softmax(o.float(), 1) for o in outs], 1).cpu().numpy()
    return lp


def head_stats(lp, Y, logpr, h_src, h_tgt):
    y = Y[:, h_tgt]
    acc = float((lp[:, h_src, :].argmax(1) == y).mean() * 100)
    info = float((lp[np.arange(len(y)), h_src, y] - logpr[h_tgt, y]).mean() / np.log(2.0))
    return acc, info


def main():
    os.makedirs(OUT, exist_ok=True)
    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pk = load_peaks()
    plan, half = build_plan(pk, max(N_PROBE))
    print(f"device {dev} | N {N_PROBE} | traces {PROBE_N} | shifts {len(plan)}/model "
          f"(misaligned control offset +{half})")

    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    lo, hi = [int(v) for v in nz["crop"]]
    mu, sd = nz["mu"].astype(np.float32).ravel(), nz["sd"].astype(np.float32).ravel()
    zero = np.zeros_like(mu)

    d = np.load(os.path.join(E.SPLIT_DIR, "kyber_test_crop.npz"))
    ip = os.path.join(E.OUT_DIR, "eval_indices.npz")
    idx = np.load(ip)["idx"] if os.path.exists(ip) else np.sort(
        np.random.default_rng(E.EVAL_SEED).choice(len(d["ciphertexts"]), E.EVAL_N, replace=False))
    idx = idx[:PROBE_N]
    Xraw = d["traces"][idx, lo:hi].astype(np.float32)
    Xz = ((Xraw - mu) / sd).astype(np.float32)
    ct = np.asarray(d["ciphertexts"], np.uint8)
    sk = np.asarray(d["labels"][0], np.int64)
    print(f"test {len(idx)} traces, window [{lo},{hi})")

    H = 2 * E.LEAK_PAIRS
    uh = T.v_ntt(T.v_unpack_10bit(ct))
    Y_all = np.stack([T.HW16[np.asarray(T.v_stored(sk, uh, j)) & 0xFFFF] for j in range(H)], 1)
    ncls = int(getattr(T, "NUM_CLASSES", 17))
    prior = np.stack([(np.bincount(Y_all[:, j], minlength=ncls) + 1.0) / (len(Y_all) + ncls)
                      for j in range(H)])
    Y, logpr, maj = Y_all[idx], np.log(prior), prior.max(1) * 100

    hp, sp = os.path.join(OUT, "shift_heads.csv"), os.path.join(OUT, "shift_summary.csv")
    for path, fields in ((hp, HEAD_FIELDS), (sp, SUM_FIELDS)):   # start fresh if the column layout changed
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                head = f.readline().strip().split(",")
            if head != fields:
                os.rename(path, path + ".bak")
                print(f"existing {os.path.basename(path)} has a different column layout, moved to .bak")
    done = set()
    if os.path.exists(sp):
        done = {(r["arch"], int(r["n_pairs"]), int(r["seed"]), r["norm_mode"])
                for r in csv.DictReader(open(sp, encoding="utf-8"))}

    t0 = time.time()
    for arch in E.ARCHS:
        for n in N_PROBE:
            for seed in SEEDS:
                f = os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")
                if not os.path.exists(f):
                    print(f"  [missing] {os.path.basename(f)}")
                    continue
                if all((arch, n, seed, m) in done for m in NORM_MODES):
                    continue
                ck = torch.load(f, map_location="cpu", weights_only=False)
                heads = int(ck.get("heads", 2 * n))
                model = E.build(T, arch, n, heads, ck.get("resnet_variant")).to(dev)
                E.load_state(model, ck["model"])
                model.eval()

                for mode in NORM_MODES:
                    if (arch, n, seed, mode) in done:
                        continue
                    rows, tab = [], {}
                    for kind, k, s, js in plan:
                        js = [j for j in js if j < n]
                        if not js:
                            continue
                        Xs = (((shift_left(Xraw, s, mu) - mu) / sd).astype(np.float32)
                              if mode == "after" else shift_left(Xz, s, zero))
                        lp = infer(model, Xs, heads, ncls, dev)
                        for j in js:
                            tp = j + k
                            a0, i0 = head_stats(lp, Y, logpr, 2 * j, 2 * tp)
                            a1, i1 = head_stats(lp, Y, logpr, 2 * j + 1, 2 * tp + 1)
                            e0 = int(pk[2 * tp] - pk[2 * j]) - s
                            e1 = int(pk[2 * tp + 1] - pk[2 * j + 1]) - s
                            rows.append(dict(arch=arch, n_pairs=n, seed=seed, norm_mode=mode,
                                             kind=kind, shift_pairs=k, shift_samples=s,
                                             head_pair=j, target_pair=tp, err_r0=e0, err_r1=e1,
                                             acc_r0=round(a0, 3), acc_r1=round(a1, 3),
                                             majority_r0=round(float(maj[2 * tp]), 3),
                                             majority_r1=round(float(maj[2 * tp + 1]), 3),
                                             info_r0=round(i0, 5), info_r1=round(i1, 5)))
                            tab.setdefault((kind, k), []).append(
                                ((a0 + a1) / 2, (i0 + i1) / 2,
                                 (maj[2 * tp] + maj[2 * tp + 1]) / 2, tp, a0, a1))
                    E.append_rows(hp, HEAD_FIELDS, rows)

                    g = lambda kind, ks: [v for (kk, k), vs in tab.items()
                                          if kk == kind and k in ks for v in vs]
                    own, al = g("own", [0]), lambda ks: g("aligned", ks)
                    m1 = lambda v, i=0: round(float(np.mean([x[i] for x in v])), 3) if v else ""
                    cov = {x[3] for x in al(SHIFTS) if x[0] >= 50}
                    conv = bool(own) and (np.mean([x[0] for x in own]) -
                                          np.mean([x[2] for x in own])) > CONV_MARGIN
                    srow = dict(arch=arch, n_pairs=n, seed=seed, norm_mode=mode,
                                converged=int(conv), acc_own=m1(own), majority_own=m1(own, 2),
                                acc_shift1=m1(al([1])), acc_shift_mean=m1(al(SHIFTS)),
                                acc_shift_far=m1(al([k for k in SHIFTS if k >= 10])),
                                info_own=m1(own, 1), info_shift_mean=m1(al(SHIFTS), 1),
                                acc_misaligned=m1(g("misaligned", MISALIGN_AT)),
                                pairs_covered_50pct=len(cov),
                                acc_r0_shift_mean=m1(al(SHIFTS), 4),
                                acc_r1_shift_mean=m1(al(SHIFTS), 5))
                    E.append_rows(sp, SUM_FIELDS, [srow])
                    print(f"{arch:<10}N{n:<3}s{seed} {mode:<7} conv {int(conv)} | "
                          f"own pair {srow['acc_own']}% (baseline {srow['majority_own']}%) | "
                          f"all shifts {srow['acc_shift_mean']}% (r0 {srow['acc_r0_shift_mean']} / "
                          f"r1 {srow['acc_r1_shift_mean']}) | 10+ pairs {srow['acc_shift_far']}% | "
                          f"misaligned {srow['acc_misaligned']}% | ({(time.time()-t0)/60:.1f} min)")
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    summary(sp, hp)


def summary(sp, hp):
    rows = [r for r in csv.DictReader(open(sp, encoding="utf-8")) if r["converged"] == "1"]
    if not rows:
        print("\nNo escaped models, summary skipped")
        return
    print("\n" + "=" * 118)
    print("Trace shift probe (escaped models only). own = unshifted baseline. r0 is aligned exactly, r1 is misaligned by the within-pair offset")
    print("=" * 118)
    print(f"{'arch':<10}{'N':>4}{'norm':>8}{'n':>4}{'own':>9}{'base':>8}{'shift1':>9}"
          f"{'shiftAll':>9}{'r0':>8}{'r1':>8}{'10+':>8}{'misal':>8}{'pairs>50%':>11}")
    print("-" * 118)
    for arch in E.ARCHS:
        for n in N_PROBE:
            for mode in NORM_MODES:
                rs = [r for r in rows if r["arch"] == arch and int(r["n_pairs"]) == n
                      and r["norm_mode"] == mode]
                if not rs:
                    continue
                f = lambda k: np.array([float(r[k]) for r in rs if r[k] != ""])
                m = lambda k: f"{f(k).mean():.2f}" if f(k).size else "-"
                print(f"{arch:<10}{n:>4}{mode:>8}{len(rs):>4}{m('acc_own'):>9}{m('majority_own'):>8}"
                      f"{m('acc_shift1'):>9}{m('acc_shift_mean'):>9}{m('acc_r0_shift_mean'):>8}"
                      f"{m('acc_r1_shift_mean'):>8}{m('acc_shift_far'):>8}"
                      f"{m('acc_misaligned'):>8}{m('pairs_covered_50pct'):>11}")
    print("=" * 118)

    # relation between alignment error and accuracy (for r1, err_r1 differs per pair so it can be seen directly)
    buck = {}
    for r in csv.DictReader(open(hp, encoding="utf-8")):
        if r["kind"] != "aligned":
            continue
        b = min(abs(int(r["err_r1"])) // 10 * 10, 40)
        buck.setdefault(b, []).append(float(r["acc_r1"]))
    if buck:
        print("accuracy by r1 alignment error (error 0 = perfectly aligned)")
        for b in sorted(buck):
            v = np.array(buck[b])
            print(f"  |err_r1| {b:>2}~{b+9 if b < 40 else '':<3}  n={len(v):<6} acc {v.mean():.2f}%")
    print("\nHow to read")
    print("  shift ≈ own  -> the pattern transfers as is between pairs. 39-pair coverage possible even with a small N")
    print("  shift ≈ baseline    -> the heads memorized something other than the position too. The current min(N,L) evaluation is correct")
    print("  if only r0 works and r1 does not -> alignment precision matters at the level of tens of samples (check with the per-error table)")
    print("  if misaligned is similar to shift -> it was right for a reason unrelated to peak alignment. Suspect the position measurement first")


if __name__ == "__main__":
    main()
