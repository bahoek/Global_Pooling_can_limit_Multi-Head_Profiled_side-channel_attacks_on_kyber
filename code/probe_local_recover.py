#!/usr/bin/env python3
r"""
probe_local_recover — local readout stage 2: key recovery

Request document: `claude/TIFS local readout stage 2 — key recovery request details (paper chat→experiment chat, 11-13).md`

    python probe_local_recover.py verify     # section 2 verification only (run this first)
    python probe_local_recover.py            # verification + group A
    python probe_local_recover.py A B        # A + B
    python probe_local_recover.py --table    # table only, no computation

**Does not modify any existing file.** Calls the ranking code of `eval_nscan.py` as is —
`prepare()` / `recover_pair()` / `tie_rank()` / `make_rows()`. The only thing that changes is **the head's
log-softmax array**. Candidates 3329², canonical-form ground truth, r0+r1 summation, tie-aware ranking,
handling of mismatched pairs, evaluation indices and accumulation order — all run through the same code as the main evaluation.

Three sources
-------------
  orig   log-softmax obtained by running the checkpoint as is (= the same as the main evaluation)
  local  `logp_<run_id>.npz` saved by `probe_local_ext.py` (±160 window before GAP)
  raw    **a linear model on the raw 320 samples** of the same ±160 window. Baseline of request section 4.
         The only exception to "no new training" (only logp is saved)

Output `<OUT_DIR>_localrec/`   ← **separate folder** from the main fine results
  eval_runs.csv / eval_pairs.csv / eval_heads.csv   (same column layout as the main evaluation)
  pair_qstar.csv        run_id, pair, q_star, censored, slowest flag
  verify.txt            section 2 verification output
  res_*.npz             raw data

Why a separate folder
---------------------
`run_id` has the form `resnet_N39_s0` and collides with the main evaluation. In one folder,
`per_seed()` in `fig_qstar.py` is a dict keyed by `(arch, N, seed)`, so it would
**silently overwrite the main training Q\***. No figure reads this folder.

Request section 7 cautions
--------------------------
  * do not mix seeds — compare orig vs local vs raw of the same run
  * do not mix learned and unlearned pairs (A0/A4 aggregate pairs 1~29 / 30~39 separately)
  * report the slowest pair as well
  * do not touch the calibration of the local heads (no temperature scaling — that would be tuning on the test set)
  * record the probe training set train 16,000 / valid 4,000 in tables and logs
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if "eval_nscan" in sys.modules:
    raise SystemExit("eval_nscan is already imported. Run this file on its own.")
_user = sys.argv[1:]
sys.argv = [sys.argv[0], "fine"]                     # import after turning on the fine grid

import eval_nscan as E                                              # noqa: E402

OUT = E.OUT_DIR.rstrip("/\\") + "_localrec"
ROOT = os.path.dirname(E.MODEL_DIR.rstrip("/\\"))
# probe_local_ext writes under eval/, **not fine** (since it does not pass "fine" on import).
# Here we are in fine mode so E.OUT_DIR is ..._qfine — the suffix must be stripped to find it.
_BASE_OUT = E.OUT_DIR[:-len("_qfine")] if E.OUT_DIR.endswith("_qfine") else E.OUT_DIR
EXT = os.path.join(_BASE_OUT, "probe_local_ext")     # probe_local_ext output (logp_*.npz)
CTRL = os.path.join(ROOT, "model_ctrl")

PROBE_TRAIN, PROBE_VALID = 16000, 4000               # request section 7: record in tables and logs
HALF = 160                                           # same window as probe_local

# (tag, group, arch, N, seed, checkpoint path or None, pairs to evaluate)
#   pairs="seen"   -> pairs 0 .. min(N,39)-1
#   pairs="unseen" -> pairs min(N,39) .. 38   (pairs 30~39 of A0/A4. Not mixed with learned pairs)
MODELS = [
    ("A0", "A", "resnet", 29, 0, None, "seen"),
    ("A1", "A", "resnet", 39, 0, None, "seen"),
    ("A2", "A", "resnet", 39, 1, None, "seen"),
    ("A3", "A", "resnet", 39, 0, os.path.join(CTRL, "D", "resnet_N39_s0.pth"), "seen"),
    ("A4", "A", "resnet", 29, 0, os.path.join(CTRL, "D29", "resnet_N29_s0.pth"), "seen"),
    ("B1a", "A", "resnet", 49, 0, None, "seen"),     # request section 4: B1a/B1b have priority A
    ("B1b", "A", "resnet", 49, 1, None, "seen"),
    ("B1c", "B", "resnet", 128, 0, None, "seen"),
    ("B1d", "B", "resnet", 128, 1, None, "seen"),
    ("B2a", "B", "mobilenet", 39, 1, None, "seen"),
    ("B2b", "B", "mobilenet", 39, 2, None, "seen"),
    ("A0u", "B", "resnet", 29, 0, None, "unseen"),
    ("A4u", "B", "resnet", 29, 0, os.path.join(CTRL, "D29", "resnet_N29_s0.pth"), "unseen"),
]

# request 2(b)/section 4: fine values of the original heads for the same seed (for reproduction check)
REF = {("resnet", 39, 0, "main"): dict(qmax=18, qmean=8.4, sr5=0.500, sr10=0.973),
       ("resnet", 29, 0, "main"): dict(qmax=10, qmean=5.8, sr5=0.921, sr10=1.000),
       ("resnet", 39, 0, "D"): dict(qmax=10, qmean=6.6, sr5=0.780),
       ("resnet", 29, 0, "D29"): dict(qmax=9, qmean=5.9, sr5=0.935)}

SOURCES = ("orig", "local", "raw")
PQ_FIELDS = ["run_id", "source", "arch", "n_pairs", "seed", "ckpt", "pair_set",
             "pair", "q_star", "censored", "slowest"]


def tag_of(tag, src):
    return f"{tag}_{src}"


def ckpt_path(arch, n, seed, ck):
    return ck if ck else os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")


def ck_name(ck):
    return "main" if ck is None else os.path.basename(os.path.dirname(ck))


def ext_rid(tag, arch, n, seed, ck):
    """Matches the run_id rule written by probe_local_ext."""
    base = f"{tag}_{arch}_N{n}_s{seed}"
    return base if ck is None else base + "_" + ck_name(ck)


# ---------------- three log-softmax sources ----------------
def logp_orig(T, arch, n, seed, ck, X, dev):
    f = ckpt_path(arch, n, seed, ck)
    if not os.path.exists(f):
        return None, f"checkpoint missing: {f}"
    cp = torch.load(f, map_location="cpu", weights_only=False)
    heads = int(cp.get("heads", 2 * n))
    model = E.build(T, arch, n, heads, cp.get("resnet_variant")).to(dev)
    E.load_state(model, cp["model"])
    model.eval()
    lp = np.empty((len(X), heads, 17), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), E.INFER_BATCH):
            xb = torch.from_numpy(X[s:s + E.INFER_BATCH]).unsqueeze(1).to(dev)
            o = model(xb)
            lp[s:s + len(xb)] = torch.stack(
                [torch.log_softmax(t.float(), 1) for t in o], 1).cpu().numpy()
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return lp, None


def logp_local(tag, arch, n, seed, ck, te_idx=None):
    p = os.path.join(EXT, f"logp_{ext_rid(tag[:-1] if tag.endswith("u") else tag, arch, n, seed, ck)}.npz")
    if not os.path.exists(p):
        return None, f"local logp missing: {p}  (run probe_local_ext.py first)"
    z = np.load(p)
    # The local logp was made from the 2000 traces of eval/eval_indices.npz. X here uses the fine-side
    # indices. If the two differ, ranks are computed with misaligned rows and **silently wrong values come out**.
    if te_idx is not None and "te_idx" in z.files:
        if not np.array_equal(np.asarray(z["te_idx"]), np.asarray(te_idx)):
            return None, ("the test indices of the local logp differ from the indices of this evaluation "
                          f"({p}). Align the idx of eval/eval_indices.npz and "
                          "eval_qfine/eval_indices.npz, then "
                          "rerun probe_local_ext.")
    return z["logp"].astype(np.float32), None


def logp_raw(T, X_raw_tr, Y_tr, X_raw_va, Y_va, X_raw_te, peaks, lo, dev, cache,
             prior=None, Y_te=None):
    """A linear model on the raw 320 samples of the same ±160 window. Baseline of request section 4.

    **Calls probe_local's `make_heads` / `train` as is** — fix of 11-15.
    The first version used a bare `Linear(320,17)`, but `probe_local.make_heads` is
    `Sequential(Dropout(0.4), Linear(C,17))`. Because of that omission the 78-head average came out
    at 39.6% / 1.34bit, lower than 43.9% / 1.43bit of 11-09 `probe_local_ctrl`.
    Now the structure, optimizer, scheduler, patience and evaluation are all the same as probe_local.
    """
    if cache.get("lp") is not None:
        return cache["lp"], None
    import probe_local as PL
    H = 2 * E.LEAK_PAIRS
    dim = 2 * PL.HALF

    def cut(X):
        out = np.empty((len(X), H, dim), np.float32)
        for p in range(E.LEAK_PAIRS):
            for role in (0, 1):
                c = peaks[(p, role)] - lo
                a_ = max(0, c - PL.HALF)
                b_ = min(X.shape[1], a_ + dim)
                a_ = max(0, b_ - dim)
                out[:, 2 * p + role, :] = X[:, a_:b_]
        return torch.from_numpy(out)

    print(f"    raw baseline: window {dim} samples x {H} heads, probe_local.make_heads "
          f"(Dropout {PL.DROPOUT} + Linear)")
    print(f"      train {len(X_raw_tr)} / valid {len(X_raw_va)} | "
          f"EPOCHS={PL.EPOCHS} LR={PL.LR} BATCH={PL.BATCH} WD={PL.WD} "
          f"PATIENCE={PL.PATIENCE} MIN_DELTA={PL.MIN_DELTA}")
    Ftr, Fva, Fte = cut(X_raw_tr), cut(X_raw_va), cut(X_raw_te)
    # Fix the head initialization. PL.train calls manual_seed, but that is **after** make_heads, so
    # it does not apply to the initialization. The old version missed this, so the group A run and the group B run
    # trained different raw models (same 39 pairs, but SR@5 split as 0.029 vs 0.032).
    torch.manual_seed(0)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(0)
    heads = PL.make_heads(H, dim, 17, dev)
    best, eps = PL.train(heads, Ftr, torch.from_numpy(Y_tr[:, :H]),
                         Fva, torch.from_numpy(Y_va[:, :H]), dev, 0)
    print(f"      validation acc {best:.2f}% ({eps} epochs)")

    heads.eval()
    lp = []
    with torch.no_grad():
        for i in range(0, len(Fte), 1024):
            o = PL.run_heads(heads, Fte[i:i + 1024].to(dev).float())
            lp.append(torch.stack([torch.log_softmax(t, 1) for t in o], 1).cpu().numpy())
    cache["lp"] = np.concatenate(lp).astype(np.float32)

    if prior is not None and Y_te is not None:                 # compare directly with the 11-09 values
        acc, info = PL.evaluate(heads, Fte, Y_te, np.log(prior[:H]), dev)
        print(f"      [raw 78-head average] acc {acc.mean():.2f}% / info {info.mean():.4f}bit")
        print(f"      11-09 probe_local_ctrl raw: 43.9% / 1.43bit (both learned and unlearned near 44%)")
        cache["acc78"], cache["info78"] = float(acc.mean()), float(info.mean())
    del heads, Ftr, Fva, Fte
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return cache["lp"], None


def check_ctrl_raw():
    """Prints the 78-head average of the 11-09 ctrl_raw.npz and the average per N split (for comparison)."""
    p = os.path.join(_BASE_OUT, "probe_local", "ctrl_raw.npz")
    if not os.path.exists(p):
        print(f"  [comparison skipped] {p} does not exist")
        return
    z = np.load(p)
    a, i = z["acc"], z["info"]
    print(f"\nraw of 11-09 probe_local_ctrl (ctrl_raw.npz, {len(a)} heads)")
    print(f"  all {len(a)} heads : acc {a.mean():.2f}% / info {i.mean():.4f}bit")
    for n in (9, 19, 29):
        m = np.zeros(len(a), bool)
        m[2 * n:] = True
        print(f"  pairs {n}~38 only   : acc {a[m].mean():.2f}% / info {i[m].mean():.4f}bit")
        m2 = ~m
        print(f"  pairs 0~{n-1} only   : acc {a[m2].mean():.2f}% / info {i[m2].mean():.4f}bit")


def purge_raw():
    """Deletes only the raw rows. Does not touch the orig/local rows or their res_*.npz."""
    n = 0
    for f in sorted(os.listdir(OUT)):
        if f.startswith("res_") and f.endswith("_raw.npz"):
            os.remove(os.path.join(OUT, f))
            n += 1
    for name in ("eval_runs.csv", "eval_pairs.csv", "eval_heads.csv", "pair_qstar.csv"):
        fp = os.path.join(OUT, name)
        if not os.path.exists(fp):
            continue
        rows = list(csv.DictReader(open(fp, encoding="utf-8")))
        keep = [r for r in rows if not r.get("run_id", "").endswith("_raw")]
        if len(keep) == len(rows):
            continue
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in keep:
                w.writerow(r)
        print(f"  {name}: removed {len(rows)-len(keep)} raw rows (kept the other {len(keep)} rows)")
    print(f"  deleted {n} res_*_raw.npz files")


# ---------------- recovery ----------------
def recover(lp, plist, Y, prior, uh_tabs, true_flats, orders, chunk, dev, hw_idx, label):
    """Uses the ranking code of eval_nscan as is. Only lp changes."""
    t0 = time.time()
    heads = lp.shape[1]
    ln2 = np.log(2.0)
    logpr = np.log(prior[:heads])
    acc = np.array([(lp[:, h, :].argmax(1) == Y[:, h]).mean() * 100 for h in range(heads)])
    ll = np.take_along_axis(lp, Y[:, :heads, None], 2)[..., 0]
    info = (ll - logpr[np.arange(heads)[None, :], Y[:, :heads]]).mean(0) / ln2
    kl = (np.exp(lp) * (lp - logpr[None, :, :])).sum(2).mean(0) / ln2

    ranks = {m: [] for m in E.MODES}
    extra = {m: [] for m in E.MODES}
    for k, p in enumerate(plist):
        tabs_d = [torch.from_numpy(t).to(dev) for t in uh_tabs[p]]
        lp0 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p, :])).to(dev)
        lp1 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p + 1, :])).to(dev)
        rk, ex = E.recover_pair(tabs_d, lp0, lp1, hw_idx, true_flats[p], orders, chunk, dev)
        for m in E.MODES:
            ranks[m].append(rk[m])
            extra[m].append(ex[m])
        if k == 0:
            el = time.time() - t0
            print(f"      1 pair {el:.1f} s -> {label} about {el*len(plist)/60:.1f} min")
    for m in E.MODES:
        ranks[m] = np.array(ranks[m])
        extra[m] = np.array(extra[m])
    return dict(pairs_sig=list(plist), pairs_pha=[],
                ranks={m: {"sig": ranks[m], "pha": np.zeros((0, 1, len(E.Q_LIST)))}
                       for m in E.MODES},
                extra={m: {"sig": extra[m], "pha": np.zeros((0, 1, 4))} for m in E.MODES},
                head_acc=acc, head_info=info, head_kl=kl), time.time() - t0


def pair_qstars(pack, plist):
    """Per-pair Q* (sum mode, based on rank averaged over repeats). censored = not recovered within 50 traces"""
    rk = pack["ranks"]["sum"]["sig"]                  # (P, R, Q)
    out = []
    for i, p in enumerate(plist):
        ge = rk[i].mean(0)
        q = E.q_star_of(ge)
        out.append((p, q))
    return out


# ---------------- section 2 verification ----------------
def verify_heads(vf):
    """(a) Does the argmax accuracy of logp_*.npz match the acc in local_pairs.csv for all 78 heads?"""
    pp = os.path.join(EXT, "local_pairs.csv")
    if not os.path.exists(pp):
        return say(vf, f"[verify a] skipped — {pp} does not exist")
    rows = {}
    for r in csv.DictReader(open(pp, encoding="utf-8")):
        h = 2 * (int(r["pair"]) - 1) + (0 if r["role"] == "r0" else 1)
        rows.setdefault(r["run_id"], {})[h] = float(r["acc"])
    ok_all = True
    for rid, want in sorted(rows.items()):
        p = os.path.join(EXT, f"logp_{rid}.npz")
        if not os.path.exists(p):
            say(vf, f"[verify a] {rid}: logp missing — skipped")
            continue
        z = np.load(p)
        lp, te_idx = z["logp"], z["te_idx"]
        Y = _test_labels()[te_idx]
        H = lp.shape[1]
        got = np.array([(lp[:, h].argmax(1) == Y[:, h]).mean() * 100 for h in range(H)])
        d = np.array([abs(got[h] - want[h]) for h in sorted(want)])
        ok = bool((d < 0.01).all())
        ok_all &= ok
        say(vf, f"[verify a] {rid}: {H} heads max diff {d.max():.4f}%p  "
                f"{'match ✓' if ok else '**mismatch ✗**'}")
        if not ok:
            bad = np.argsort(-d)[:5]
            say(vf, f"          mismatched heads: {[(int(b), round(float(d[b]),3)) for b in bad]}")
    return ok_all


_LBL = {}


def _test_labels():
    if "Y" not in _LBL:
        T = E.import_train()
        d = np.load(os.path.join(E.SPLIT_DIR, "kyber_test_crop.npz"))
        ct = np.asarray(d["ciphertexts"], np.uint8)
        sk = np.asarray(d["labels"][0], np.int64)
        uh = T.v_ntt(T.v_unpack_10bit(ct))
        _LBL["Y"] = np.stack([T.HW16[np.asarray(T.v_stored(sk, uh, j)) & 0xFFFF]
                              for j in range(2 * E.LEAK_PAIRS)], 1).astype(np.int64)
    return _LBL["Y"]


def verify_path(vf, ctx):
    """(b) Feeding the original heads' log-softmax into this path, are the existing fine values reproduced?"""
    T, X, Y, prior, uh_tabs, true_flats, orders, chunk, dev, hw_idx = ctx
    ok_all = True
    for arch, n, seed in (("resnet", 39, 0), ("resnet", 29, 0)):
        ref = REF[(arch, n, seed, "main")]
        lp, err = logp_orig(T, arch, n, seed, None, X, dev)
        if lp is None:
            say(vf, f"[verify b] {arch} N{n} s{seed}: {err}")
            ok_all = False
            continue
        plist = list(range(min(n, E.LEAK_PAIRS)))
        pack, _ = recover(lp, plist, Y, prior, uh_tabs, true_flats, orders, chunk, dev,
                          hw_idx, f"verify {arch} N{n} s{seed}")
        rk = pack["ranks"]["sum"]["sig"]
        qs = [E.q_star_of(rk[i].mean(0)) for i in range(len(plist))]
        qmax = max(q if q is not None else E.QMAX + 1 for q in qs)
        qmean = float(np.mean([q if q is not None else E.QMAX + 1 for q in qs]))
        i5, i10 = E.Q_LIST.index(5), E.Q_LIST.index(10)
        sr5 = float((rk[:, :, i5] == 1).mean())
        sr10 = float((rk[:, :, i10] == 1).mean())
        d = [abs(qmax - ref["qmax"]), abs(qmean - ref["qmean"]),
             abs(sr5 - ref["sr5"]), abs(sr10 - ref.get("sr10", sr10))]
        ok = d[0] < 0.5 and d[1] < 0.15 and d[2] < 0.01 and d[3] < 0.01
        ok_all &= ok
        say(vf, f"[verify b] {arch} N{n} s{seed}")
        say(vf, f"          existing fine : Qmax {ref['qmax']} / Qmean {ref['qmean']} / "
                f"SR@5 {ref['sr5']:.3f} / SR@10 {ref.get('sr10', float('nan')):.3f}")
        say(vf, f"          this path     : Qmax {qmax} / Qmean {qmean:.1f} / "
                f"SR@5 {sr5:.3f} / SR@10 {sr10:.3f}")
        say(vf, f"          {'reproduced ✓' if ok else '**reproduction failed ✗ — do not run the local results**'}")
    return ok_all


def say(vf, s):
    print(s)
    vf.write(s + "\n")
    vf.flush()
    return True


# ---------------- main body ----------------
def _verify(ctx, vpath):
    with open(vpath, "w", encoding="utf-8") as vf:
        say(vf, "\n" + "=" * 78)
        say(vf, "Section 2 verification — this must pass before the local results are run")
        say(vf, "=" * 78)
        ok_a = verify_heads(vf)
        ok_b = verify_path(vf, ctx)
        say(vf, "")
        say(vf, f"Verification result: head order {'OK' if ok_a else 'FAIL'} / "
                f"computation path {'OK' if ok_b else 'FAIL'}")
    return ok_b


def main():
    os.makedirs(OUT, exist_ok=True)
    args = [a for a in _user if not a.startswith("-")]
    flags = [a for a in _user if a.startswith("-")]
    if "--table" in flags:
        table()
        return

    verify_only = bool(args) and args[0].lower() == "verify"
    raw_only = bool(args) and args[0].lower() == "raw"      # redo only the raw rows (11-15)
    groups = set(a.upper() for a in args if a.upper() in ("A", "B")) or {"A"}
    if raw_only:
        groups = {"A", "B"}                                  # raw rows for every run already done

    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = E.CHUNK
    if chunk is None:
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if dev.type == "cuda" else 0
        chunk = 64 if mem >= 40 else 16

    # evaluation indices and orders are reused by copying the main fine evaluation's file (request section 3)
    src = os.path.join(E.OUT_DIR, "eval_indices.npz")
    dst = os.path.join(OUT, "eval_indices.npz")
    if not os.path.exists(dst) and os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        print(f"evaluation indices/orders: copied from the main fine evaluation ({src})")
    main_out = E.OUT_DIR
    E.OUT_DIR = OUT                                   # so that prepare reads dst
    print(f"device {dev}  CHUNK={chunk}  EVAL_N={E.EVAL_N}  N_REPEAT={E.N_REPEAT}")
    print(f"  Q={E.Q_LIST[0]}..{E.Q_LIST[-1]} every trace, {E.N_REPEAT} repeats")
    print(f"  probe training set: train {PROBE_TRAIN} / valid {PROBE_VALID}  (to be stated in the paper)")
    print(f"  output {OUT}  (separate from the main fine results)")
    X, Y, prior, uh_tabs, true_flats, mism, orders = E.prepare(T, dev)
    hw_idx = torch.from_numpy(np.asarray(T.HW16, np.int64)).to(dev)
    cur_idx = np.load(dst)["idx"] if os.path.exists(dst) else None

    # index check (last line of request section 3)
    if os.path.exists(src) and os.path.exists(dst):
        a, b = np.load(src), np.load(dst)
        same = (np.array_equal(a["idx"], b["idx"]) and
                np.array_equal(np.asarray(a["orders"]), np.asarray(b["orders"])))
        print(f"index check {'match' if same else '**mismatch — stopping recommended**'}")
    else:
        print("index check skipped (main fine file missing)")

    ctx = (T, X, Y, prior, uh_tabs, true_flats, orders, chunk, dev, hw_idx)
    vpath = os.path.join(OUT, "verify.txt")
    if ((not verify_only) and "--reverify" not in flags and os.path.exists(vpath)
            and "head order OK / computation path OK" in open(vpath, encoding="utf-8").read()):
        print(f"\nSection 2 verification: already passed ({vpath}) — skipping")
        print("  to rerun, use --reverify")
        ok_b = True
    else:
        ok_b = _verify(ctx, vpath)
    if verify_only:
        return
    if not ok_b:
        raise SystemExit("\nVerification (b) failed — as the request says, the local results are not run. Find the cause first.")

    srcs = ("raw",) if raw_only else SOURCES
    if raw_only:
        print("\n" + "=" * 78)
        print("Recomputing only the raw rows (11-15: fix for missing Dropout 0.4)")
        print("  the orig/local rows and their res_*.npz are not touched")
        print("=" * 78)
        check_ctrl_raw()
        # Decide the targets **first**. If we only find out there are 0 after deleting, there is nothing to restore.
        PRE, SUF = "res_", "_orig.npz"
        have = {f[len(PRE):-len(SUF)] for f in os.listdir(OUT)
                if f.startswith(PRE) and f.endswith(SUF)}
        todo = [m for m in MODELS if m[0] in have]
        print(f"\n{len(todo)} targets (runs that already have orig): "
              f"{', '.join(m[0] for m in todo) or 'none'}")
        if not todo:
            raise SystemExit(
                f"\nNo res_*{SUF} found — nothing was deleted.\n"
                f"  res_ files in {OUT}: "
                f"{sorted(f for f in os.listdir(OUT) if f.startswith(PRE))[:6]}\n"
                f"  run orig/local first: python {os.path.basename(__file__)} A B")
        print()
        purge_raw()
        print()
    else:
        todo = [m for m in MODELS if m[1] in groups]
        print(f"\n{len(todo)} targets x {len(srcs)} sources\n")
    peaks = _peaks()
    lo = int(np.load(os.path.join(E.MODEL_DIR, "norm.npz"))["crop"][0])
    rawc, Xr = {}, None

    for tag, grp, arch, n, seed, ck, pset in todo:
        nseen = min(n, E.LEAK_PAIRS)
        plist = list(range(nseen)) if pset == "seen" else list(range(nseen, E.LEAK_PAIRS))
        if not plist:
            print(f"[{tag}] no pairs to evaluate (pair_set={pset}, N={n}) — skipped")
            continue
        for srcname in srcs:
            rid = f"{tag}_{srcname}"
            if os.path.exists(os.path.join(OUT, f"res_{rid}.npz")):
                continue
            print(f"[{tag}/{srcname}] {arch} N{n} s{seed} ck={ck_name(ck)} "
                  f"pairs {plist[0]+1}~{plist[-1]+1} ({len(plist)})")
            if srcname == "orig":
                lp, err = logp_orig(T, arch, n, seed, ck, X, dev)
            elif srcname == "local":
                lp, err = logp_local(tag, arch, n, seed, ck, te_idx=cur_idx)
            else:
                if Xr is None:
                    Xr = _raw_sets(T, lo)
                lp, err = logp_raw(T, Xr[0], Xr[1], Xr[2], Xr[3], Xr[4],
                                   peaks, lo, dev, rawc, prior=prior, Y_te=Y)
            if lp is None:
                print(f"    [skipped] {err}")
                continue
            pack, sec = recover(lp, plist, Y, prior, uh_tabs, true_flats, orders,
                                chunk, dev, hw_idx, f"{tag}/{srcname}")
            np.savez(os.path.join(OUT, f"res_{rid}.npz"),
                     q_list=np.array(E.Q_LIST), pairs_sig=np.array(plist, np.int64),
                     pairs_pha=np.array([], np.int64), heads=lp.shape[1], eval_sec=sec,
                     **{f"ranks_{m}_sig": pack["ranks"][m]["sig"] for m in E.MODES},
                     **{f"extra_{m}_sig": pack["extra"][m]["sig"] for m in E.MODES},
                     head_acc=pack["head_acc"], head_info=pack["head_info"],
                     head_kl=pack["head_kl"])
            hr, pr, rr = E.make_rows(arch, n, seed, lp.shape[1], pack, prior, mism, sec)
            for row in hr + pr + rr:
                row["run_id"] = rid
            E.append_rows(os.path.join(OUT, "eval_heads.csv"), E.HEAD_FIELDS, hr)
            E.append_rows(os.path.join(OUT, "eval_pairs.csv"), E.PAIR_FIELDS, pr)
            E.append_rows(os.path.join(OUT, "eval_runs.csv"), E.RUN_FIELDS, rr)

            qs = pair_qstars(pack, plist)
            cens = [p for p, q in qs if q is None]
            fin = [(p, q) for p, q in qs if q is not None]
            slow = max(fin, key=lambda t: t[1])[0] if fin else None
            E.append_rows(os.path.join(OUT, "pair_qstar.csv"), PQ_FIELDS,
                          [dict(run_id=rid, source=srcname, arch=arch, n_pairs=n, seed=seed,
                                ckpt=ck_name(ck), pair_set=pset, pair=p + 1,
                                q_star=("" if q is None else q), censored=int(q is None),
                                slowest=int(p == slow)) for p, q in qs])
            qm = max((q for _, q in fin), default=None)
            print(f"    Q*max {qm}  censor {len(cens)}  "
                  f"slowest pair {slow + 1 if slow is not None else '—'}"
                  f"  ({sec/60:.1f} min)")
            if cens:
                print(f"    [notice] {len(cens)} pairs not recovered within 50 traces -> "
                      f"coarse run needed (request section 3). pairs {[p+1 for p in cens]}")
            del lp
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    E.OUT_DIR = main_out
    table()


def _peaks():
    """Measured NICV peaks. probe_local's NICV constant is frozen to E.OUT_DIR at import time,
    but here E.OUT_DIR has already been changed to localrec, so it would look in the wrong folder. Set it directly."""
    import probe_local as PL
    PL.NICV = os.path.join(_BASE_OUT, "nicv", "nicv_pairs.csv")
    return PL.read_peaks()


def _raw_sets(T, lo):
    """Raw (normalized) train/valid/test for the raw baseline. Same split sizes as probe_local."""
    nz = np.load(os.path.join(E.MODEL_DIR, "norm.npz"))
    hi = int(nz["crop"][1])
    mu, sd = nz["mu"], nz["sd"]
    H = 2 * E.LEAK_PAIRS

    def load(name, nsub=None):
        d = np.load(os.path.join(E.SPLIT_DIR, f"kyber_{name}_crop.npz"))
        ct = np.asarray(d["ciphertexts"], np.uint8)
        sk = np.asarray(d["labels"][0], np.int64)
        uh = T.v_ntt(T.v_unpack_10bit(ct))
        Yl = np.stack([T.HW16[np.asarray(T.v_stored(sk, uh, j)) & 0xFFFF]
                       for j in range(H)], 1).astype(np.int64)
        Xl = d["traces"][:, lo:hi]
        if nsub is not None and nsub < len(Xl):
            Xl, Yl = Xl[:nsub], Yl[:nsub]
        return ((Xl.astype(np.float32) - mu) / sd).astype(np.float32), Yl

    Xtr, Ytr = load("train", PROBE_TRAIN)
    Xva, Yva = load("valid", PROBE_VALID)
    Xte_all, _ = load("test")
    idx = np.load(os.path.join(OUT, "eval_indices.npz"))["idx"]
    return Xtr, Ytr, Xva, Yva, Xte_all[idx]


# ---------------- table ----------------
def table():
    p = os.path.join(OUT, "pair_qstar.csv")
    if not os.path.exists(p):
        print(f"\n{p} does not exist. Run the computation first.")
        return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    runs = {}
    for r in rows:
        runs.setdefault((r["run_id"].rsplit("_", 1)[0], r["source"]), []).append(r)

    rr = {}
    rp = os.path.join(OUT, "eval_runs.csv")
    if os.path.exists(rp):
        for r in csv.DictReader(open(rp, encoding="utf-8")):
            if r.get("eval_mode") == "sum":
                rr[r["run_id"]] = r

    print("\n" + "=" * 100)
    print("Local readout key recovery (fine Q=1..50, 20 repeats, 2000 test traces)")
    print(f"probe training set: train {PROBE_TRAIN} / valid {PROBE_VALID}")
    print("=" * 100)
    hdr = (f"{'run':<8}{'source':<8}{'Q*max':>7}{'Q*mean':>8}{'SR@5':>8}{'SR@10':>8}"
           f"{'censor':>8}{'slowest':>8}")
    print(hdr)
    print("-" * len(hdr))
    last = None
    for (tag, src) in sorted(runs, key=lambda t: (t[0], SOURCES.index(t[1]))):
        rs = runs[(tag, src)]
        fin = [int(r["q_star"]) for r in rs if r["q_star"] != ""]
        cens = sum(int(r["censored"]) for r in rs)
        slow = next((r["pair"] for r in rs if r["slowest"] == "1"), "—")
        run = rr.get(f"{tag}_{src}", {})
        g = lambda k: (float(run[k]) if run.get(k) not in (None, "", "-1") else None)
        sr5, sr10 = g("sr_q5"), g("sr_q10")
        if last is not None and last != tag:
            print()
        last = tag
        print(f"{tag:<8}{src:<8}"
              f"{(max(fin) if fin else '—'):>7}"
              f"{(f'{np.mean(fin):.1f}' if fin else '—'):>8}"
              f"{(f'{sr5:.3f}' if sr5 is not None else '—'):>8}"
              f"{(f'{sr10:.3f}' if sr10 is not None else '—'):>8}"
              f"{cens:>8}{slow:>8}")

    print("\nDecision (request section 6) — reference models A1/A2/A3 (N39) + B1a/B1b (N49)")
    keys = ["A1", "A2", "A3", "B1a", "B1b"]
    ok, miss = [], []
    for k in keys:
        o = [int(r["q_star"]) for r in runs.get((k, "orig"), []) if r["q_star"] != ""]
        l = [int(r["q_star"]) for r in runs.get((k, "local"), []) if r["q_star"] != ""]
        if not o or not l:
            miss.append(k)
            continue
        ok.append((k, max(o), max(l)))
    for k, a, b in ok:
        print(f"  {k:<5} original Q*max {a:>3}  ->  local {b:>3}   {'improved' if b < a else 'worse/same'}")
    if miss:
        print(f"  (not yet available: {miss})")
    if ok and not miss:
        allbetter = all(b < a for _, a, b in ok)
        allle10 = all(b <= 10 for _, _, b in ok)
        if allbetter and allle10:
            print("  -> **the attack improves** (Q*max decreases for all 5 + local Q*max <= 10)")
        elif any(b > a for _, a, b in ok):
            print("  -> **gap between information and Q*** (some model has a larger Q*max)")
        else:
            print("  -> **the representation improves but the attack cost stays the same** (only some improve)")
    # raw evaluates one model with only the pair set changed, so it is summarized separately (11-15 request item 2)
    rawrows = {t: rs for (t, src), rs in runs.items() if src == "raw"}
    if rawrows:
        print("\nraw baseline — per pair set (same model, only the evaluated pairs differ)")
        print(f"  {'run':<7}{'pairs':>5}{'Q*max':>7}{'Q*mean':>8}{'SR@5':>8}{'censor':>8}")
        seen_np = {}
        for t in sorted(rawrows):
            rs = rawrows[t]
            fin = [int(r["q_star"]) for r in rs if r["q_star"] != ""]
            cens = sum(int(r["censored"]) for r in rs)
            run = rr.get(f"{t}_raw", {})
            sr5 = (float(run["sr_q5"]) if run.get("sr_q5") not in (None, "", "-1") else None)
            npair = len(rs)
            print(f"  {t:<7}{npair:>5}{(max(fin) if fin else '—'):>7}"
                  f"{(f'{np.mean(fin):.1f}' if fin else '—'):>8}"
                  f"{(f'{sr5:.3f}' if sr5 is not None else '—'):>8}{cens:>8}")
            seen_np.setdefault(npair, t)
        if 39 in seen_np and 29 in seen_np:
            print(f"  -> 39 pairs = {seen_np[39]}_raw, 29 pairs = {seen_np[29]}_raw")

    # State which runs are actually in this table. Previously it always printed "aggregated separately",
    # which made the absence of the unrun group B in the table look like an aggregation problem.
    have = sorted({t for t, _ in runs})
    want = [m[0] for m in MODELS]
    missing = [t for t in want if t not in have]
    print(f"\nruns in this table ({len(have)}): {', '.join(have)}")
    if missing:
        grp = {m[0]: m[1] for m in MODELS}
        byg = {}
        for t in missing:
            byg.setdefault(grp[t], []).append(t)
        print(f"missing runs ({len(missing)}): {', '.join(missing)}")
        for g in sorted(byg):
            print(f"   group {g}: {', '.join(byg[g])}"
                  f"   -> python {os.path.basename(__file__)} {g}")
    seen_u = [t for t in have if t.endswith("u")]
    print("\nNote: seeds were not mixed. "
          + (f"Unlearned pairs ({', '.join(seen_u)}) were aggregated separately from learned pairs."
             if seen_u else "Unlearned pairs (A0u/A4u) have not been run yet — not in this table."))
    print(f"raw data: {OUT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
