"""
TIFS N scan — 39-pair key recovery evaluation (all 160 models in one go)

Place in the same folder as train_nscan.py and run:
    python eval_nscan.py              # evaluate (finished models are skipped)
    python eval_nscan.py rebuild      # rebuild the 3 CSVs from res_*.npz (no evaluation)
    python eval_nscan.py summary      # re-aggregate only eval_summary.csv from eval_runs.csv
                                      #   -> if you change the escape criterion (CONV_MARGIN_PP), rebuild; if you only change the aggregation, summary

Model definitions, Kyber arithmetic and label functions are imported as-is from train_nscan.py (bit-identical code to training).

What it does
------------
For each model (arch, N, seed):
  1. Outputs of the 2N heads (log-softmax) on EVAL_N (2000) traces drawn once and fixed from the test set
  2. Per-head statistics (all heads, including phantom) — exhaustive evidence for additional experiment A, at zero cost
       acc / majority / info_bits (needs labels) / kl_bits (no labels needed = observable by the attacker too)
  3. Key recovery: exhaustively enumerate the pair (a0,a1) over 3329^2 = 11.08 million candidates
       signal pairs  p = 0 .. min(N,39)-1          (pairs actually present in the trace)
       phantom pairs: PHANTOM_MAX of 39 .. N-1    (evenly spaced sample. Same number of repeats as signal pairs = same protocol)
         - The exhaustive evidence (acc/info/kl) for every phantom "head" goes into eval_heads.csv at zero cost.
           Only a sample of them is run through key recovery — running all pairs 39~N-1 would add about 36 hours of evaluation time (PHANTOM_MAX=0)
       main result  "sum": sum of r[0] + r[1] head log-likelihoods
       auxiliary    "r0" : r[0] head only (computed in the same accumulation, no extra cost)
  4. Accumulation order: draw N_REPEAT random orders of 500 traces from the 2000 and fix them. Shared by all models and all pairs
     (so that order luck does not mix into model comparisons). Record the rank of the correct pair at Q = 1,2,5,...,500

Rank definition (ESL lesson B6: pinned down as 1-based)
-------------------------------------------------------
Candidates that produce the same HW sequence have bit-identical scores and tie (hundreds of thousands when Q is small).
  rank = (number of candidates with a higher score) + 1 + (number of tied candidates)/2        <- expected rank with ties broken at random
Using the optimistic rank (number ahead + 1) gives rank 1 even at Q=1 and inflates GE and SR.
  GE(Q) = mean rank over repeats,  SR(Q) = fraction of sole first place (rank==1 with no ties),  Q* = first Q where GE(Q)=1

Correct pair and candidate representation
-----------------------------------------
Candidates are in canonical form 0..q-1, the correct answer is sk mod q. For pairs where the firmware sk is >= q (non-canonical), the HW computed
from the canonical form disagrees with the training label on some traces -> the correct score is slightly reduced, a conservative direction. The per-pair rate is recorded as canon_mismatch.
(The decomposition itself is checked for bit-exact agreement with the training labels using the raw sk values, and aborts if wrong)

Restart
-------
After each model finishes, save res_{arch}_N{n}_s{seed}.npz + append to CSV. Rerunning skips finished models.

Outputs (OUT_DIR)
-----------------
  eval_pairs.csv    [file 1] run x pair x mode x repeat — raw ranks per depth. GE, SR and Q* can all be recomputed from here
  eval_runs.csv     [file 2] run x mode — SR/GE curves, Q* statistics, censored, phantom, escape verdict + threshold
  eval_summary.csv  [file 3] arch x N x mode — source for the paper tables (mean±std over converged seeds only + convergence rate)
  eval_heads.csv              run x head — acc/majority/info/kl, signal/phantom split
  res_*.npz                   raw data (full rank arrays, head statistics) — for regenerating CSVs
  eval_indices.npz            test indices used for evaluation, accumulation orders
"""

import csv
import glob
import os
import re
import sys
import time

import numpy as np
import torch

# ==========================================================
HERE      = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = HERE                                     # location of train_nscan.py
SPLIT_DIR = r"/home/bahoek/data/kyber/tifs/split"
MODEL_DIR = r"/home/bahoek/data/kyber/tifs/model"
OUT_DIR   = r"/home/bahoek/data/kyber/tifs/eval"

# ---- Profile: if the first argument is "fine", high-resolution Q* re-evaluation ----------------------
#   The main evaluation uses log-spaced Q (1,2,5,10,20,50,...), so it cannot distinguish Q*=10 from 20.
#   "Q* doubles from 29->39" may really be "<=10 -> 11~20", so the resolution is insufficient.
#   fine: Q = 1..50 every trace, 20 repeats, phantom off, output to OUT_DIR+"_qfine".
#   Resolution comes from the Q spacing; repeats only reduce the jitter of the GE mean — more precise than the main evaluation's 10 is enough.
#   The 2000 traces and accumulation orders are the same as the main evaluation (same rule and same seed as eval_indices.npz).
#   Run:  python eval_nscan.py fine        /  fine rebuild  /  fine summary
FINE = len(sys.argv) > 1 and sys.argv[1] == "fine"

ARCHS  = ["resnet", "mobilenet"]
N_LIST = [9, 19, 29, 39, 49, 69, 99, 128]
SEEDS  = list(range(10))

LEAK_PAIRS     = 39
EVAL_N         = 2000      # number of traces drawn once and fixed from test
Q_LIST         = [1, 2, 5, 10, 20, 50, 100, 200, 500]
N_REPEAT       = 10        # number of accumulation-order repeats for signal pairs. Evaluation time is proportional to this
PHANTOM_MAX    = 10        # number of phantom pairs (evenly spaced) to run key recovery on for N>39 models. 0 = none, None = all pairs 39~N-1
PHANTOM_REPEAT = N_REPEAT  # number of repeats for phantom pairs. Same value as signal pairs = exactly the same protocol (cost +16%)
                           #   to reduce, set to 2. 2 repeats are enough for the chance check itself, and the cost is +3%
CONV_MARGIN_PP = 5.0       # escape criterion: signal-head test acc > (mean signal-head majority + this value)
EVAL_SEED      = 0
CHUNK          = None      # None: automatic from GPU memory (>=40GB: 64, otherwise 16)
USE_COMPILE    = True      # fuse add+mask+gather+sum with torch.compile. Falls back to eager automatically if the first call fails
INFER_BATCH    = 128
SUMMARY_Q      = [50]      # low Q to include with std in file 3 (a point not at the ceiling). The maximum Q is always included

if FINE:
    Q_LIST, N_REPEAT, PHANTOM_MAX = list(range(1, 51)), 20, 0   # 20 repeats (50 is excessive, 11h vs 25h)
    SUMMARY_Q = [5, 10, 20]
    OUT_DIR = OUT_DIR.rstrip("/\\") + "_qfine"
# ==========================================================

Q = 3329
QMAX = max(Q_LIST)
SUMMARY_Q = [q for q in SUMMARY_Q if q in Q_LIST]      # drop values not in Q_LIST
MODES = ["sum", "r0"]

PAIR_FIELDS = (["run_id", "arch", "n_pairs", "seed", "eval_mode", "pair_idx", "signal", "repeat"]
               + [f"rank_q{q}" for q in Q_LIST]
               + ["q_star", f"loglik_true_q{QMAX}", f"loglik_top1_q{QMAX}",
                  "top1_a0", "top1_a1", "canon_mismatch"])
RUN_FIELDS = (["run_id", "arch", "n_pairs", "heads", "seed", "eval_mode",
               "converged", "conv_threshold", "acc_signal", "majority_signal",
               "acc_phantom", "majority_phantom", "kl_signal_min", "kl_phantom_max",
               "n_pairs_eval"]
              + [f"sr_q{q}" for q in Q_LIST] + [f"ge_log2_q{q}" for q in Q_LIST]
              + ["q_star_mean", "q_star_median", "q_star_max", "n_pairs_censored",
                 "slowest_pair_idx", "pairs_recovered",
                 f"phantom_sr_q{QMAX}", f"phantom_ge_log2_q{QMAX}", "n_phantom_pairs",
                 "n_repeat", "q_max", "eval_n", "eval_time_sec"])
SUM_FIELDS = (["arch", "n_pairs", "heads", "eval_mode", "n_seeds", "n_converged", "conv_rate"]
              + [f"sr_q{q}_mean" for q in Q_LIST]
              + [f"sr_q{q}_std" for q in sorted(set(SUMMARY_Q + [QMAX]))]
              + ["acc_signal_mean", "acc_signal_std"]
              + [f"ge_log2_q{q}_mean" for q in sorted(set(SUMMARY_Q))]
              + [f"ge_log2_q{QMAX}_mean", f"ge_log2_q{QMAX}_std",
                 "q_star_max_mean", "q_star_max_std", "q_star_mean_mean", "q_star_mean_std",
                 "n_censored_mean", "pairs_recovered_mean",
                 "phantom_sr_mean", "phantom_ge_log2_mean", "expected_capacity"])
HEAD_FIELDS = ["run_id", "arch", "n_pairs", "seed", "head", "pair", "role", "signal",
               "acc", "majority_acc", "info_bits", "kl_bits"]


def i16(v):
    return ((np.asarray(v, dtype=np.int64) & 0xFFFF) ^ 0x8000) - 0x8000


def run_id_of(arch, n, seed):
    return f"{arch}_N{n}_s{seed}"


# ---------------- import from train_nscan ----------------
def import_train():
    sys.path.insert(0, TRAIN_DIR)
    import train_nscan as T
    for name in ("v_fqmul", "v_ntt", "v_unpack_10bit", "v_stored", "ZETAS", "HW16", "build_model"):
        if not hasattr(T, name):
            raise SystemExit(f"train_nscan.py has no {name} — if the name is different, please report it")
    return T


def build(T, arch, n, heads, variant):
    if variant and hasattr(T, "RESNET_VARIANT"):
        T.RESNET_VARIANT = variant
    last = None
    for arg in (heads, n):                 # supports both build_model(arch, number of heads) / (arch, number of pairs)
        try:
            m = T.build_model(arch, arg)
        except TypeError as e:
            raise SystemExit(f"build_model call signature differs: {e} — please share its definition")
        last = len(m.classifiers)
        if last == heads:
            return m
    raise SystemExit(f"{arch} N{n}: built model has {last} heads != checkpoint {heads} heads")


def load_state(model, sd):
    fixed = {}
    for k, v in sd.items():
        for pre in ("_orig_mod.", "module."):
            if k.startswith(pre):
                k = k[len(pre):]
        fixed[k] = v
    try:
        model.load_state_dict(fixed)
    except RuntimeError:
        model.load_state_dict({k.replace(".shortcut.", ".sc."): v for k, v in fixed.items()})


# ---------------- pair info / candidate grid ----------------
def phantom_pairs(n):
    """Phantom pairs to run key recovery on for N>39 models (PHANTOM_MAX evenly spaced from pairs 39..N-1).
    The side that pre-builds the pair grids and the side that evaluates must not disagree, so this is decided in this one place only."""
    if n <= LEAK_PAIRS or PHANTOM_MAX == 0:
        return []
    cand = list(range(LEAK_PAIRS, n))
    if PHANTOM_MAX is None:
        return cand
    k = min(PHANTOM_MAX, len(cand))
    if k == 1:
        return [cand[0]]
    return sorted({cand[int(round(i * (len(cand) - 1) / (k - 1)))] for i in range(k)})


def needed_pairs():
    """All pairs that actually need a grid in this run (signal + phantom samples for every N)."""
    s = set()
    for n in N_LIST:
        s |= set(range(min(n, LEAK_PAIRS)))
        s |= set(phantom_pairs(n))
    return sorted(s)


def pair_zeta(T, p):
    """Pair p (coefficients 2p, 2p+1). base = 2p, zeta follows the same rule as train_nscan.v_stored."""
    idx = 2 * p
    i4 = idx // 4
    z = int(T.ZETAS[64 + i4])
    if (idx % 4) >= 2:
        z = -z
    return 2 * p, z


def pair_tables(T, uh, p):
    """stored = A[a0] + B[a1] decomposition.  r0(even): A0=fq(a0,b0), B0=fq(fq(a1,b1),z)
                                             r1(odd):  A1=fq(a0,b1), B1=fq(a1,b0)"""
    base, z = pair_zeta(T, p)
    b0 = uh[:, base][:, None]
    b1 = uh[:, base + 1][:, None]
    c = np.arange(Q, dtype=np.int64)[None, :]
    A0 = T.v_fqmul(c, b0)
    B0 = T.v_fqmul(T.v_fqmul(c, b1), z)
    A1 = T.v_fqmul(c, b1)
    B1 = T.v_fqmul(c, b0)
    return [(np.asarray(t, np.int64) & 0xFFFF).astype(np.int32) for t in (A0, B0, A1, B1)]


def check_pair(T, uh, sk, p, tabs):
    """(1) decomposition with raw sk values == training label (v_stored), bit-exact (abort if wrong)
       (2) fraction of traces where the value computed from the canonical-form answer differs from the raw one"""
    base, z = pair_zeta(T, p)
    a0r, a1r = int(sk[base]), int(sk[base + 1])
    b0, b1 = uh[:, base], uh[:, base + 1]
    s0 = (T.v_fqmul(a0r, b0) + T.v_fqmul(T.v_fqmul(a1r, b1), z)) & 0xFFFF
    s1 = (T.v_fqmul(a0r, b1) + T.v_fqmul(a1r, b0)) & 0xFFFF
    l0 = np.asarray(T.v_stored(sk, uh, 2 * p)) & 0xFFFF
    l1 = np.asarray(T.v_stored(sk, uh, 2 * p + 1)) & 0xFFFF
    if not (np.array_equal(s0, l0) and np.array_equal(s1, l1)):
        raise SystemExit(f"pair {p}: decomposition differs from training labels — evaluation aborted (please report)")
    a0c, a1c = int(i16(a0r)) % Q, int(i16(a1r)) % Q
    A0, B0, A1, B1 = tabs
    c0 = (A0[:, a0c].astype(np.int64) + B0[:, a1c]) & 0xFFFF
    c1 = (A1[:, a0c].astype(np.int64) + B1[:, a1c]) & 0xFFFF
    return a0c * Q + a1c, float(np.mean((c0 != l0) | (c1 != l1)))


# ---------------- accumulated score ----------------
def _score_eager(A_c, B_c, lut):
    """A_c,B_c:(T,Q) int32, lut:(T,65536) float32 -> (Q*Q,) : sum_t lut_t[(A[a0]+B[a1]) & 0xFFFF]"""
    T_ = A_c.size(0)
    idx = (A_c[:, :, None].long() + B_c[:, None, :].long()) & 0xFFFF
    return lut.gather(1, idx.view(T_, -1)).sum(0)

_score_fn = _score_eager
if USE_COMPILE:
    try:
        _score_fn = torch.compile(_score_eager, dynamic=True)
    except Exception as e:
        print(f"torch.compile unavailable, proceeding with eager: {e}")


def _chunk_score(A_c, B_c, lut):
    """compile is actually built on the first call. If it fails there (no Python.h, etc.), switch to eager automatically.
    The numerical results are the same for both; only the speed differs."""
    global _score_fn
    try:
        return _score_fn(A_c, B_c, lut)
    except Exception as e:
        if _score_fn is _score_eager:
            raise
        print(f"    [note] torch.compile failed -> switching to eager (same numbers, slower): {type(e).__name__}")
        _score_fn = _score_eager
        return _score_eager(A_c, B_c, lut)


def tie_rank(score, true_flat):
    """Expected rank with tie handling (1-based)."""
    st = score[true_flat]
    gt = int((score > st).sum().item())
    eq = int((score == st).sum().item()) - 1
    return gt + 1 + eq / 2.0


def recover_pair(tabs_d, lp0, lp1, hw_idx, true_flat, orders, chunk, dev):
    # If the Q spacing is narrower than chunk (fine: spacing 1), chunk is automatically used only up to that spacing —
    # because the accumulation must be cut at every checkpoint. The numbers do not depend on chunk.
    """Returns: ranks[mode] (R, len(Q_LIST)) and extra[mode] (R, 4) = [loglik_true, loglik_top1, a0, a1] @ QMAX"""
    A0, B0, A1, B1 = tabs_d
    R = len(orders)
    ranks = {m: np.zeros((R, len(Q_LIST)), np.float64) for m in MODES}
    extra = {m: np.zeros((R, 4), np.float64) for m in MODES}
    for rep, order in enumerate(orders):
        s = {m: torch.zeros(Q * Q, device=dev, dtype=torch.float32) for m in MODES}
        ptr = 0
        for j, qn in enumerate(Q_LIST):
            ids_np = order[ptr:qn]
            for c in range(0, len(ids_np), chunk):
                ids = torch.as_tensor(ids_np[c:c + chunk], device=dev)
                g0 = _chunk_score(A0[ids], B0[ids], lp0[ids][:, hw_idx].contiguous())
                g1 = _chunk_score(A1[ids], B1[ids], lp1[ids][:, hw_idx].contiguous())
                s["r0"] += g0
                s["sum"] += g0
                s["sum"] += g1
            ptr = qn
            for m in MODES:
                ranks[m][rep, j] = tie_rank(s[m], true_flat)
        for m in MODES:                                  # margin and top-1 candidate at the last depth
            am = int(s[m].argmax().item())
            extra[m][rep] = [float(s[m][true_flat].item()), float(s[m][am].item()),
                             am // Q, am % Q]
    return ranks, extra


# ---------------- curve statistics ----------------
def curve(ranks):
    """ranks (R,Q) -> ge, sr, q_star (None if none)"""
    ge = ranks.mean(0)
    sr = (ranks == 1).mean(0)
    hit = np.where(ge <= 1.0)[0]
    return ge, sr, (Q_LIST[hit[0]] if len(hit) else None)


def q_star_of(rank_row):
    hit = np.where(np.asarray(rank_row) <= 1.0)[0]
    return Q_LIST[hit[0]] if len(hit) else None


# ---------------- CSV ----------------
def append_rows(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return round(float(x), nd)


def as_int(x):
    """When rebuilding from an old-version npz, missing values (NaN) become blank."""
    return "" if x is None or not np.isfinite(x) else int(x)


# ---------------- build rows (shared by evaluation and rebuild) ----------------
def make_rows(arch, n, seed, heads, pack, prior, mism, eval_sec):
    """pack: dict(pairs_sig, pairs_pha, ranks{mode}{sig,pha}, extra{mode}{sig,pha},
                  head_acc, head_info, head_kl)"""
    rid = run_id_of(arch, n, seed)
    maj = prior.max(1) * 100
    acc, info, kl = pack["head_acc"], pack["head_info"], pack["head_kl"]
    ns = min(heads, 2 * LEAK_PAIRS)
    has_pha = heads > ns
    thr = float(maj[:ns].mean()) + CONV_MARGIN_PP
    conv = int(float(acc[:ns].mean()) > thr)

    head_rows = [dict(run_id=rid, arch=arch, n_pairs=n, seed=seed, head=h, pair=h // 2,
                      role="r0" if h % 2 == 0 else "r1", signal=int(h < 2 * LEAK_PAIRS),
                      acc=fmt(acc[h], 3), majority_acc=fmt(maj[h], 3),
                      info_bits=fmt(info[h], 5), kl_bits=fmt(kl[h], 5))
                 for h in range(heads)]

    sig, pha = pack["pairs_sig"], pack["pairs_pha"]
    pair_rows, run_rows = [], []
    for m in MODES:
        for kind, plist in (("sig", sig), ("pha", pha)):
            rk, ex = pack["ranks"][m][kind], pack["extra"][m][kind]
            for pi, p in enumerate(plist):
                for rep in range(rk.shape[1]):
                    row = dict(run_id=rid, arch=arch, n_pairs=n, seed=seed, eval_mode=m,
                               pair_idx=int(p), signal=int(kind == "sig"), repeat=rep,
                               q_star=q_star_of(rk[pi, rep]) or -1,
                               canon_mismatch=fmt(mism.get(int(p), np.nan), 5))
                    row.update({f"rank_q{q}": fmt(v, 2) for q, v in zip(Q_LIST, rk[pi, rep])})
                    row[f"loglik_true_q{QMAX}"] = fmt(ex[pi, rep, 0], 3)
                    row[f"loglik_top1_q{QMAX}"] = fmt(ex[pi, rep, 1], 3)
                    row["top1_a0"] = as_int(ex[pi, rep, 2])
                    row["top1_a1"] = as_int(ex[pi, rep, 3])
                    pair_rows.append(row)

        rs = pack["ranks"][m]["sig"]                                  # (P, R, Q)
        ge = np.array([curve(rs[i])[0] for i in range(len(sig))])     # (P, Q)
        sr = np.array([curve(rs[i])[1] for i in range(len(sig))])
        qs = [curve(rs[i])[2] for i in range(len(sig))]
        ok = [q for q in qs if q is not None]
        cens = [sig[i] for i, q in enumerate(qs) if q is None]
        if cens:                                                      # slowest pair = worst GE among censored
            worst = int(cens[int(np.argmax([ge[sig.index(p), -1] for p in cens]))])
        else:
            worst = int(sig[int(np.argmax([Q_LIST.index(q) for q in qs]))]) if qs else ""
        rp = pack["ranks"][m]["pha"]
        p_sr = p_ge = None
        if len(pha):
            p_sr = float(np.mean([(rp[i] == 1).mean() for i in range(len(pha))]))
            p_ge = float(np.mean([np.log2(rp[i][:, -1].mean()) for i in range(len(pha))]))
        row = dict(run_id=rid, arch=arch, n_pairs=n, heads=heads, seed=seed, eval_mode=m,
                   converged=conv, conv_threshold=fmt(thr, 3),
                   acc_signal=fmt(acc[:ns].mean(), 3), majority_signal=fmt(maj[:ns].mean(), 3),
                   acc_phantom=fmt(acc[ns:].mean() if has_pha else None, 3),
                   majority_phantom=fmt(maj[ns:].mean() if has_pha else None, 3),
                   kl_signal_min=fmt(kl[:ns].min(), 5),
                   kl_phantom_max=fmt(kl[ns:].max() if has_pha else None, 5),
                   n_pairs_eval=len(sig),
                   q_star_mean=fmt(np.mean(ok) if ok else None, 1),
                   q_star_median=fmt(np.median(ok) if ok else None, 1),
                   q_star_max=(max(ok) if ok and not cens else ""),
                   n_pairs_censored=len(cens), slowest_pair_idx=worst,
                   pairs_recovered=int((ge[:, -1] <= 1).sum()),
                   n_repeat=rs.shape[1], q_max=QMAX, eval_n=EVAL_N,
                   eval_time_sec=fmt(eval_sec, 1), n_phantom_pairs=len(pha))
        row.update({f"sr_q{q}": fmt(v, 4) for q, v in zip(Q_LIST, sr.mean(0))})
        row.update({f"ge_log2_q{q}": fmt(v, 3) for q, v in zip(Q_LIST, np.log2(ge).mean(0))})
        row[f"phantom_sr_q{QMAX}"] = fmt(p_sr, 4)
        row[f"phantom_ge_log2_q{QMAX}"] = fmt(p_ge, 3)
        run_rows.append(row)
    return head_rows, pair_rows, run_rows


def write_summary():
    """File 3 — aggregate eval_runs.csv by (arch, N, mode). Can be rerun at any time without evaluation."""
    rp = os.path.join(OUT_DIR, "eval_runs.csv")
    if not os.path.exists(rp):
        print("eval_runs.csv not found"); return
    rows = list(csv.DictReader(open(rp, encoding="utf-8")))
    out = []
    for arch in ARCHS:
        for n in N_LIST:
            for m in MODES:
                rs = [r for r in rows if r["arch"] == arch and int(r["n_pairs"]) == n
                      and r["eval_mode"] == m]
                if not rs:
                    continue
                cv = [r for r in rs if r["converged"] == "1"]
                g = lambda rr, k: np.array([float(r[k]) for r in rr if r[k] not in ("", "-1")])
                ms = lambda k, nd=4: fmt(g(cv, k).mean() if g(cv, k).size else None, nd)
                sd = lambda k, nd=4: fmt(g(cv, k).std() if g(cv, k).size else None, nd)
                rate = len(cv) / len(rs)
                row = dict(arch=arch, n_pairs=n, heads=rs[0]["heads"], eval_mode=m,
                           n_seeds=len(rs), n_converged=len(cv), conv_rate=fmt(rate, 3))
                row.update({f"sr_q{q}_mean": ms(f"sr_q{q}") for q in Q_LIST})
                row.update({f"sr_q{q}_std": sd(f"sr_q{q}")
                            for q in sorted(set(SUMMARY_Q + [QMAX]))})
                row["acc_signal_mean"] = ms("acc_signal", 2)
                row["acc_signal_std"] = sd("acc_signal", 2)
                for q in sorted(set(SUMMARY_Q)):
                    row[f"ge_log2_q{q}_mean"] = ms(f"ge_log2_q{q}", 3)
                row[f"ge_log2_q{QMAX}_mean"] = ms(f"ge_log2_q{QMAX}", 3)
                row[f"ge_log2_q{QMAX}_std"] = sd(f"ge_log2_q{QMAX}", 3)
                row["q_star_max_mean"] = ms("q_star_max", 1)
                row["q_star_max_std"] = sd("q_star_max", 1)
                row["q_star_mean_mean"] = ms("q_star_mean", 1)
                row["q_star_mean_std"] = sd("q_star_mean", 1)
                row["n_censored_mean"] = ms("n_pairs_censored", 2)
                row["pairs_recovered_mean"] = ms("pairs_recovered", 2)
                row["phantom_sr_mean"] = ms(f"phantom_sr_q{QMAX}")
                row["phantom_ge_log2_mean"] = ms(f"phantom_ge_log2_q{QMAX}", 3)
                srm = g(cv, f"sr_q{QMAX}")
                row["expected_capacity"] = fmt(rate * srm.mean() if srm.size else None, 4)
                out.append(row)
    p = os.path.join(OUT_DIR, "eval_summary.csv")
    if os.path.exists(p):
        os.remove(p)
    append_rows(p, SUM_FIELDS, out)
    print(f"saved: {p} ({len(out)} rows)")


def print_table():
    p = os.path.join(OUT_DIR, "eval_summary.csv")
    if not os.path.exists(p):
        return
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["eval_mode"] == "sum"]
    print("\n" + "=" * 112)
    print(f"39-pair key recovery (main result = r0+r1 sum, mean±std over converged seeds only). Escape criterion: signal-head acc > majority + {CONV_MARGIN_PP}%p")
    print("=" * 112)
    lowq = sorted(set(SUMMARY_Q))[0] if SUMMARY_Q else QMAX
    print(f"{'arch':<10}{'N':>4}{'conv':>7}{'acc_sig':>9}{f'SR@{QMAX}':>14}{f'SR@{lowq}':>14}"
          f"{f'log2GE@{lowq}':>12}{'Q*mean':>9}{'Q*max':>8}{'censor':>8}{'pha SR':>8}{'pha GE':>8}")
    print("-" * 112)
    for r in rows:
        f = lambda k: r[k] if r.get(k, "") != "" else "-"
        pm = lambda a, b: f"{f(a)}±{f(b)}"
        print(f"{r['arch']:<10}{r['n_pairs']:>4}{r['n_converged']+'/'+r['n_seeds']:>7}"
              f"{f('acc_signal_mean'):>9}"
              f"{pm(f'sr_q{QMAX}_mean', f'sr_q{QMAX}_std'):>14}"
              f"{pm(f'sr_q{lowq}_mean', f'sr_q{lowq}_std'):>14}"
              f"{f(f'ge_log2_q{lowq}_mean'):>12}{f('q_star_mean_mean'):>9}"
              f"{f('q_star_max_mean'):>8}"
              f"{f('n_censored_mean'):>8}{f('phantom_sr_mean'):>8}{f('phantom_ge_log2_mean'):>8}")
    print("=" * 112)
    print(f"  log2GE@{QMAX} is all 0 (GE=1) for converged seeds, so the table shows a low Q ({lowq}) — both are in eval_summary.csv")
    print(f"  pha SR/GE = phantom-pair key recovery — additional experiment A. With zero information SR≈0, expected rank = candidates/2 "
          f"-> log2GE≈{np.log2((Q*Q+1)/2):.2f} (not log2(candidates)={2*np.log2(Q):.2f})")
    print("  Exhaustive per-head evidence is in eval_heads.csv (kl_bits: computed without labels = observable by the attacker too)")


# ---------------- one model ----------------
def eval_model(T, arch, n, seed, X, Y, prior, uh_tabs, true_flats, mism, orders, chunk, dev, hw_idx):
    f = os.path.join(MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")
    if not os.path.exists(f):
        print(f"  [missing] {os.path.basename(f)}")
        return None
    t0 = time.time()
    ck = torch.load(f, map_location="cpu", weights_only=False)
    heads = int(ck.get("heads", 2 * n))
    model = build(T, arch, n, heads, ck.get("resnet_variant")).to(dev)
    load_state(model, ck["model"])
    model.eval()

    ncls = prior.shape[1]
    lp = np.empty((len(X), heads, ncls), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), INFER_BATCH):
            xb = torch.from_numpy(X[s:s + INFER_BATCH]).unsqueeze(1).to(dev)
            outs = model(xb)
            lp[s:s + len(xb)] = torch.stack(
                [torch.log_softmax(o.float(), 1) for o in outs], 1).cpu().numpy()
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    ln2 = np.log(2.0)
    logpr = np.log(prior[:heads])
    acc = np.array([(lp[:, h, :].argmax(1) == Y[:, h]).mean() * 100 for h in range(heads)])
    ll = np.take_along_axis(lp, Y[:, :heads, None], 2)[..., 0]
    info = (ll - logpr[np.arange(heads)[None, :], Y[:, :heads]]).mean(0) / ln2
    kl = (np.exp(lp) * (lp - logpr[None, :, :])).sum(2).mean(0) / ln2

    sig = list(range(min(n, LEAK_PAIRS)))
    pha = phantom_pairs(n)
    ranks = {m: {} for m in MODES}
    extra = {m: {} for m in MODES}
    for kind, plist, reps in (("sig", sig, orders), ("pha", pha, orders[:PHANTOM_REPEAT])):
        rr = {m: [] for m in MODES}
        ee = {m: [] for m in MODES}
        for k, p in enumerate(plist):
            tabs_d = [torch.from_numpy(t).to(dev) for t in uh_tabs[p]]
            lp0 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p, :])).to(dev)
            lp1 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p + 1, :])).to(dev)
            rk, ex = recover_pair(tabs_d, lp0, lp1, hw_idx, true_flats[p], reps, chunk, dev)
            for m in MODES:
                rr[m].append(rk[m])
                ee[m].append(ex[m])
            if kind == "sig" and k == 0:
                el = time.time() - t0
                tot = len(sig) + len(pha) * PHANTOM_REPEAT / max(1, len(orders))
                print(f"    1 pair {el:.1f}s -> this model about {el*tot/60:.1f} min "
                      f"(signal {len(sig)} pairs + phantom {len(pha)} pairs x{PHANTOM_REPEAT} repeats)")
        for m in MODES:
            ranks[m][kind] = np.array(rr[m]) if plist else np.zeros((0, 1, len(Q_LIST)))
            extra[m][kind] = np.array(ee[m]) if plist else np.zeros((0, 1, 4))
    eval_sec = time.time() - t0

    pack = dict(pairs_sig=sig, pairs_pha=pha, ranks=ranks, extra=extra,
                head_acc=acc, head_info=info, head_kl=kl)
    np.savez(os.path.join(OUT_DIR, f"res_{arch}_N{n}_s{seed}.npz"),
             q_list=np.array(Q_LIST), pairs_sig=np.array(sig, np.int64),
             pairs_pha=np.array(pha, np.int64), heads=heads, eval_sec=eval_sec,
             **{f"ranks_{m}_{k}": ranks[m][k] for m in MODES for k in ("sig", "pha")},
             **{f"extra_{m}_{k}": extra[m][k] for m in MODES for k in ("sig", "pha")},
             head_acc=acc, head_info=info, head_kl=kl,
             val_acc_ckpt=np.asarray(ck.get("acc", [])))
    hr, pr, rr2 = make_rows(arch, n, seed, heads, pack, prior, mism, eval_sec)
    append_rows(os.path.join(OUT_DIR, "eval_heads.csv"), HEAD_FIELDS, hr)
    append_rows(os.path.join(OUT_DIR, "eval_pairs.csv"), PAIR_FIELDS, pr)
    append_rows(os.path.join(OUT_DIR, "eval_runs.csv"), RUN_FIELDS, rr2)
    return rr2[0], eval_sec / 60


# ---------------- common preparation ----------------
def prepare(T, dev, need_grid=True):
    nz = np.load(os.path.join(MODEL_DIR, "norm.npz"))
    lo, hi = [int(v) for v in nz["crop"]]
    d = np.load(os.path.join(SPLIT_DIR, "kyber_test_crop.npz"))
    n_test = len(d["ciphertexts"])
    ip = os.path.join(OUT_DIR, "eval_indices.npz")
    if os.path.exists(ip):
        e = np.load(ip)
        idx, orders = e["idx"], list(e["orders"])
        assert len(idx) == EVAL_N and len(orders) == N_REPEAT and e["orders"].shape[1] == QMAX, \
            "eval_indices.npz settings differ from the current ones — delete it and start over, or match the settings"
        print("evaluation indices and orders: reusing existing file")
    else:
        rng = np.random.default_rng(EVAL_SEED)
        idx = np.sort(rng.choice(n_test, EVAL_N, replace=False))
        orders = [rng.permutation(EVAL_N)[:QMAX] for _ in range(N_REPEAT)]
        np.savez(ip, idx=idx, orders=np.array(orders), seed=EVAL_SEED)
    X = d["traces"][idx, lo:hi].astype(np.float32)
    X = ((X - nz["mu"]) / nz["sd"]).astype(np.float32)
    ct_all = np.asarray(d["ciphertexts"], np.uint8)
    sk = np.asarray(d["labels"][0], np.int64)
    print(f"test {n_test} traces, using {EVAL_N} of them, window [{lo},{hi})  X {X.shape}")

    h_max = 2 * max(N_LIST)
    uh_all = T.v_ntt(T.v_unpack_10bit(ct_all))
    Y_all = np.stack([T.HW16[np.asarray(T.v_stored(sk, uh_all, j)) & 0xFFFF]
                      for j in range(h_max)], 1)
    ncls = int(getattr(T, "NUM_CLASSES", 17))
    prior = np.stack([(np.bincount(Y_all[:, j], minlength=ncls) + 1.0) /
                      (len(Y_all) + ncls) for j in range(h_max)])     # based on the full test set
    Y, uh = Y_all[idx], uh_all[idx]

    plist = needed_pairs()
    uh_tabs, true_flats, mism = {}, {}, {}
    t0 = time.time()
    for p in plist:
        tabs = pair_tables(T, uh, p)
        true_flats[p], mism[p] = check_pair(T, uh, sk, p, tabs)
        if need_grid:
            uh_tabs[p] = tabs
    bad = {p: m for p, m in mism.items() if m > 0}
    gb = len(plist) * 4 * EVAL_N * Q * 4 / 2**30
    n_sig_tab = sum(1 for p in plist if p < LEAK_PAIRS)
    print(f"pair grids {len(plist)} (signal {n_sig_tab} + phantom {len(plist)-n_sig_tab}, "
          f"{time.time()-t0:.0f}s{f', {gb:.1f}GB' if need_grid else ''}), decomposition = training labels match OK")
    print("canonical-form answer vs raw sk label mismatch: "
          + (", ".join(f"pair{p} {m*100:.2f}%" for p, m in bad.items()) if bad else "none"))
    return X, Y, prior, uh_tabs, true_flats, mism, orders


# ---------------- rebuild ----------------
def rebuild():
    """Rebuild the 3 CSVs + eval_heads.csv from res_*.npz. No re-evaluation.
    Also reads npz files saved by the old version (ranks_sum/ranks_r0, phantom not separated) (new columns left blank)."""
    T = import_train()
    _, _, prior, _, _, mism, _ = prepare(T, torch.device("cpu"), need_grid=False)
    for nm in ("eval_heads.csv", "eval_pairs.csv", "eval_runs.csv"):
        p = os.path.join(OUT_DIR, nm)
        if os.path.exists(p):
            os.rename(p, p + ".bak")
            print(f"existing {nm} -> {nm}.bak")
    files = sorted(glob.glob(os.path.join(OUT_DIR, "res_*_N*_s*.npz")))
    done = 0
    for f in files:
        m = re.search(r"res_(.+)_N(\d+)_s(\d+)\.npz$", os.path.basename(f))
        arch, n, seed = m.group(1), int(m.group(2)), int(m.group(3))
        d = np.load(f, allow_pickle=False)
        keys = set(d.files)
        if "ranks_sum_sig" in keys:
            sig = [int(v) for v in d["pairs_sig"]]
            pha = [int(v) for v in d["pairs_pha"]]
            ranks = {mo: {k: d[f"ranks_{mo}_{k}"] for k in ("sig", "pha")} for mo in MODES}
            extra = {mo: {k: d[f"extra_{mo}_{k}"] for k in ("sig", "pha")} for mo in MODES}
        else:                                            # old version
            allp = [int(v) for v in d["pairs"]]
            sig = [p for p in allp if p < LEAK_PAIRS]
            pha = [p for p in allp if p >= LEAK_PAIRS]
            old = {"sum": d["ranks_sum"], "r0": d["ranks_r0"]}
            ranks, extra = {}, {}
            for mo in MODES:
                a = old[mo]
                ranks[mo] = {"sig": a[:len(sig)], "pha": a[len(sig):]}
                extra[mo] = {k: np.full((len(ranks[mo][k]), ranks[mo][k].shape[1], 4), np.nan)
                             for k in ("sig", "pha")}
        heads = int(d["heads"]) if "heads" in keys else 2 * n
        pack = dict(pairs_sig=sig, pairs_pha=pha, ranks=ranks, extra=extra,
                    head_acc=d["head_acc"], head_info=d["head_info"], head_kl=d["head_kl"])
        sec = float(d["eval_sec"]) if "eval_sec" in keys else float("nan")
        hr, pr, rr = make_rows(arch, n, seed, heads, pack, prior, mism, sec)
        append_rows(os.path.join(OUT_DIR, "eval_heads.csv"), HEAD_FIELDS, hr)
        append_rows(os.path.join(OUT_DIR, "eval_pairs.csv"), PAIR_FIELDS, pr)
        append_rows(os.path.join(OUT_DIR, "eval_runs.csv"), RUN_FIELDS, rr)
        done += 1
    print(f"{done} models: CSVs regenerated")
    write_summary()
    print_table()


# ---------------- main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    args = sys.argv[1:]
    if args and args[0] == "fine":
        args = args[1:]
    cmd = args[0] if args else ""
    if cmd == "summary":
        write_summary(); print_table(); return
    if cmd == "rebuild":
        rebuild(); return

    T = import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = CHUNK
    if chunk is None:
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if dev.type == "cuda" else 0
        chunk = 64 if mem >= 40 else 16
    pha_s = "off" if PHANTOM_MAX == 0 else f"{PHANTOM_MAX or 'all'} pairs x{PHANTOM_REPEAT} repeats"
    print(f"device {dev}  CHUNK={chunk}  EVAL_N={EVAL_N}  N_REPEAT={N_REPEAT}  phantom {pha_s}"
          f"{'  [fine: high-resolution Q*]' if FINE else ''}\n  Q={Q_LIST if len(Q_LIST)<=12 else f'{Q_LIST[0]}..{Q_LIST[-1]} every trace'}")
    X, Y, prior, uh_tabs, true_flats, mism, orders = prepare(T, dev)
    hw_idx = torch.from_numpy(np.asarray(T.HW16, np.int64)).to(dev)

    todo = [(a, n, s) for a in ARCHS for n in N_LIST for s in SEEDS
            if not os.path.exists(os.path.join(OUT_DIR, f"res_{a}_N{n}_s{s}.npz"))]
    print(f"\n{len(todo)} models remaining (finished ones skipped)\n")
    t_all, done = time.time(), 0
    for arch, n, seed in todo:
        print(f"[{done+1}/{len(todo)}] {arch} N{n} s{seed}")
        out = eval_model(T, arch, n, seed, X, Y, prior, uh_tabs, true_flats, mism,
                         orders, chunk, dev, hw_idx)
        done += 1
        if out is None:
            continue
        r, t_min = out
        print(f"    conv {r['converged']} acc_sig {r['acc_signal']}%  SR@{QMAX} {r[f'sr_q{QMAX}']}  "
              f"recovered {r['pairs_recovered']}/{r['n_pairs_eval']} pairs  "
              f"log2GE {r[f'ge_log2_q{QMAX}']}  Q*max {r['q_star_max']}  "
              f"phaSR {r[f'phantom_sr_q{QMAX}']}  ({t_min:.1f} min, cumulative {(time.time()-t_all)/3600:.1f} h)")
    write_summary()
    print_table()


if __name__ == "__main__":
    main()
