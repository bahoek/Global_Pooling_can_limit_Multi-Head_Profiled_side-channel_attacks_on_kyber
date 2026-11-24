#!/usr/bin/env python3
r"""
e3_firmware_repr — E3: re-evaluate with candidates enumerated in the firmware representation

Request document: `claude/TIFS additional experiments before resubmission … (11-17)` section 5.
paper chat instructions (11-19): keep the oracle label / do not build the 2-representation version / use existing values for offset-0 pairs /
                     output exactly as in 5-4, the "pairs 1~29 only" row with the 10-seed average is required.

    python e3_firmware_repr.py --diag      # diagnosis only, a few seconds   <- step 1
    python e3_firmware_repr.py --probe     # 8 pairs x (N39 s0, N29 s0)      <- step 2
    python e3_firmware_repr.py --body      # required 1·2 — orig 30 runs + D·D29
    python e3_firmware_repr.py --local     # required 3 — local heads (those that have logp)
    python e3_firmware_repr.py --body --local
    python e3_firmware_repr.py --table     # table only, no computation

What is the problem
-------------------
`eval_nscan.pair_tables` lays out the candidate key coefficients in **canonical form c = 0..q-1** and takes the correct answer
as `i16(sk) % q`. Some firmware sk coefficients are outside canonical form, and `fqmul(a,b)` and
`fqmul(a+q,b)` are equal only mod q, so **the stored 16-bit pattern can differ.** As a result
the correct candidate's predicted label disagrees with the actual label on some traces (`canon_mismatch`:
2~3% for 6 pairs, 9.2% for pair 32) and the correct score is reduced.

Representation used — oracle (`--diag` finds it directly for each coefficient)
------------------------------------------------------------------------------
For each coefficient j compute  o_j = ( i16(sk[j]) - i16(sk[j]) % q ) / q  and lay out the candidates as `c + o_j*q`.
The correct candidate becomes bit-identical to the firmware value, the mismatch becomes 0%, and the number of candidates
**stays 3329** (candidate space q², chance level 22.40, same cost).

**This is an upper bound, not an attack.** o_j can only be obtained by knowing the correct key. A real attacker would have to
lay out both {v, v+q} for each coefficient and take the maximum per residue class, which means 4x the candidates.
paper chat decision: the 4x version is not built; the paper states `oracle` explicitly.

The key to reducing the cost (paper chat instruction 2)
-------------------------------------------------------
For pairs where both coefficient offsets are 0, the oracle table is **identical** to the canonical-form table, so Q* is the same too.
So oracle computes anew **only the pairs with offset != 0 + 1 control pair**, and for the remaining pairs
the ranks from the existing canonical-form results (`eval_qfine/eval_pairs.csv` etc.) are **taken as-is and merged**.
39 pairs x 30 runs becomes 8 pairs x 30 runs.

The canonical side is not recomputed either; it is read from the existing CSVs. Whether the values read are correct is checked against
the two reproduction reference rows of request 1-3 (N39 s0 = 18 / 8.4 / 0.500, N29 s0 = 10 / 5.8 / 0.921) and
the common-pair reference values (N39 s0 13, s1 14, N49 s0 11, s1 14), and the result is logged.

No existing file is modified
----------------------------
Only `pair_tables` and `needed_pairs` of `eval_nscan` are swapped inside this process;
`prepare`, `recover_pair` and `q_star_of` are used as-is. Local head tags are read and parsed
directly from `probe_local_ext` (no guessing).

Outputs  …/eval_qfine_e3/
  diag.txt           per-coefficient offsets + per-pair mismatch + note on the cause (item F homework)
  verify.txt         reproduction checks and index comparison (request 1-1, 1-3, 9-3)
  eval_runs.csv      run x candidate_repr x pair set  — 5-4(1)
  eval_pairs.csv     raw ranks per pair x repeat (only pairs computed with oracle) — 1-11
  pair_qstar.csv     run x candidate_repr x pair — 5-4(2)
  summary_5_4.csv    the 5-4(3) summary table as-is (10-seed average)
  pair32.csv         5-4(4) pair 32 alone — original head and local head, per run
  res_<run_id>.npz   oracle rank arrays — on rerun, finished runs are skipped
"""

import csv
import glob
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if "eval_nscan" in sys.modules:
    raise SystemExit("eval_nscan is already imported. Run this script on its own.")
_USER = sys.argv[1:]
sys.argv = [sys.argv[0], "fine"]                    # fine profile = Q 1..50, 20 repeats

import eval_nscan as E                                              # noqa: E402

Q = E.Q
NQ = len(E.Q_LIST)
MAIN = E.OUT_DIR                                    # …/eval_qfine
BASE = MAIN[:-len("_qfine")] if MAIN.endswith("_qfine") else MAIN    # …/eval
OUT = MAIN + "_e3"
CTRL = os.path.join(os.path.dirname(E.MODEL_DIR.rstrip("/\\")), "model_ctrl")
SRC_D = BASE + "_qfine_d"                           # D·D29 canonical form
SRC_LR = BASE + "_qfine_localrec"                   # local heads canonical form
EXT = os.path.join(BASE, "probe_local_ext")         # logp_<tag>.npz

CLEAN_PAIR = 0                                      # control pair for checking path reproduction
SUBSETS = (("1-39", 39), ("1-29", 29))              # pair sets of 5-4(3)
PAIR32 = 31                                         # "pair 32" in the paper
PROBE_DEFAULT = ["N39_s0", "N29_s0"]                # the two runs that have 1-3 reproduction reference values

# request 1-3 and section 12 — reference values to check that the canonical-form read is correct
REF_ALL = {"resnet_N39_s0": (18, 8.4, 0.500), "resnet_N29_s0": (10, 5.8, 0.921),
           "resnet_N39_s1": (18, 9.3, 0.43), "resnet_N49_s0": (12, 7.1, 0.69),
           "resnet_N49_s1": (19, 9.0, 0.42)}
REF_29 = {"resnet_N39_s0": 13, "resnet_N39_s1": 14,
          "resnet_N49_s0": 11, "resnet_N49_s1": 14}
# request section 12 — local head Q*max (canonical form). None = >50 (censored)
REF_LOCAL = {("resnet", 29, 0, ""): 8, ("resnet", 39, 0, ""): 7,
             ("resnet", 39, 1, ""): 6, ("resnet", 49, 0, ""): 6,
             ("resnet", 49, 1, ""): 7, ("resnet", 39, 0, "D"): 8,
             ("resnet", 29, 0, "D29"): 12, ("resnet", 128, 0, ""): 16,
             ("resnet", 128, 1, ""): 9, ("mobilenet", 39, 1, ""): None,
             ("mobilenet", 39, 2, ""): 8}

_OFF = {}                                           # coefficient j -> offset o_j

RUN_FIELDS = ["run_id", "tag", "heads_src", "candidate_repr", "pair_set",
              "arch", "n_pairs", "seed", "ckpt", "n_pairs_eval",
              "q_star_max", "q_star_mean", "q_star_median", "n_pairs_censored",
              "slowest_pair_idx", "slowest_pair", "sr_q5", "ge_log2_qmax",
              "n_oracle_pairs", "n_reused_pairs", "n_repeat", "eval_time_sec"]
PQ_FIELDS = ["run_id", "tag", "heads_src", "candidate_repr", "arch", "n_pairs", "seed",
             "pair_idx", "pair", "q_star", "censored", "ge_log2_qmax",
             "computed", "canon_mismatch", "off_a0", "off_a1"]
EP_FIELDS = (["run_id", "tag", "heads_src", "candidate_repr", "pair_idx", "pair",
              "repeat", "q_star"] + [f"rank_q{q}" for q in E.Q_LIST])
SUM_FIELDS = ["heads_src", "arch", "n_pairs", "pair_set", "n_seeds",
              "canon_q_star_max_mean", "canon_q_star_max_std",
              "canon_q_star_mean_mean", "canon_q_star_mean_std", "canon_slowest_top",
              "oracle_q_star_max_mean", "oracle_q_star_max_std",
              "oracle_q_star_mean_mean", "oracle_q_star_mean_std", "oracle_slowest_top",
              "d_q_star_max", "d_q_star_mean", "n_censored_canon", "n_censored_oracle"]


# ================= candidate representation =================
def offsets_of(sk, n_coef):
    v = np.array([int(E.i16(sk[j])) for j in range(n_coef)], np.int64)
    return {j: int((v[j] - v[j] % Q) // Q) for j in range(n_coef)}


def cand_array(j, repr_name):
    c = np.arange(Q, dtype=np.int64)
    return c if repr_name == "canonical" else c + _OFF.get(j, 0) * Q


def make_pair_tables(repr_name):
    """Table A is indexed by a0 and table B by a1, so the offsets are attached the same way."""
    def pair_tables(T, uh, p):
        base, z = E.pair_zeta(T, p)
        b0 = uh[:, base][:, None]
        b1 = uh[:, base + 1][:, None]
        c0 = cand_array(base, repr_name)[None, :]
        c1 = cand_array(base + 1, repr_name)[None, :]
        A0 = T.v_fqmul(c0, b0)
        B0 = T.v_fqmul(T.v_fqmul(c1, b1), z)
        A1 = T.v_fqmul(c0, b1)
        B1 = T.v_fqmul(c1, b0)
        return [(np.asarray(t, np.int64) & 0xFFFF).astype(np.int32)
                for t in (A0, B0, A1, B1)]
    return pair_tables


def true_mismatch(T, uh, sk, p, repr_name):
    """Compute only the single column of the correct candidate (cheap, no full table). Same formula as check_pair."""
    base, z = E.pair_zeta(T, p)
    b0, b1 = uh[:, base], uh[:, base + 1]
    v0 = int(cand_array(base, repr_name)[int(E.i16(sk[base])) % Q])
    v1 = int(cand_array(base + 1, repr_name)[int(E.i16(sk[base + 1])) % Q])
    c0 = np.asarray(T.v_fqmul(v0, b0) + T.v_fqmul(T.v_fqmul(v1, b1), z)) & 0xFFFF
    c1 = np.asarray(T.v_fqmul(v0, b1) + T.v_fqmul(v1, b0)) & 0xFFFF
    l0 = np.asarray(T.v_stored(sk, uh, 2 * p)) & 0xFFFF
    l1 = np.asarray(T.v_stored(sk, uh, 2 * p + 1)) & 0xFFFF
    return float(np.mean((c0 != l0) | (c1 != l1)))


def oracle_pairs(drows):
    """Instruction (2): pairs whose offsets are all 0 have the same table and hence the same Q* -> not computed."""
    return sorted({r[0] for r in drows if (r[5], r[6]) != (0, 0)} | {CLEAN_PAIR})


# ================= read existing canonical-form results =================
def load_pairs_csv(path, want=None, mode="sum"):
    """eval_pairs.csv -> {(run_id, pair_idx): ranks (R, NQ)}.  {} if missing."""
    if not os.path.exists(path):
        return {}
    acc = {}
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        qc = [c for c in (rd.fieldnames or []) if c.startswith("rank_q")]
        qs = [int(c[len("rank_q"):]) for c in qc]
        if qs != list(E.Q_LIST):
            raise SystemExit(f"{path}: Q grid differs from the current one\n"
                             f"  file {qs[:6]}…{qs[-3:]} ({len(qs)} values)\n"
                             f"  now  {list(E.Q_LIST)[:6]}…{list(E.Q_LIST)[-3:]} "
                             f"({NQ} values)  — check that it is a fine result")
        for r in rd:
            if r.get("eval_mode") != mode:
                continue
            rid = r["run_id"]
            if want is not None and rid not in want:
                continue
            k = (rid, int(r["pair_idx"]))
            acc.setdefault(k, []).append([float(r[c]) if r[c] != "" else np.nan
                                          for c in qc])
    return {k: np.array(v, np.float64) for k, v in acc.items()}


def run_stats(ranks_by_pair, pairs):
    """Run statistics from pair -> ranks (R,NQ). Same definition as make_rows."""
    qs, ge_l, srs = {}, [], []
    for p in pairs:
        rk = ranks_by_pair[p]
        ge = rk.mean(0)
        qs[p] = E.q_star_of(ge)
        ge_l.append(float(np.log2(ge[-1])))
        srs.append(float((rk[:, E.Q_LIST.index(5)] == 1).mean())
                   if 5 in E.Q_LIST else np.nan)
    ok = [q for q in qs.values() if q is not None]
    cens = [p for p, q in qs.items() if q is None]
    if cens:
        slow = max(cens, key=lambda p: ranks_by_pair[p].mean(0)[-1])
    else:
        slow = max(qs, key=lambda p: (qs[p], ranks_by_pair[p].mean(0)[-1])) if qs else None
    return dict(qs=qs, q_max=(max(ok) if ok and not cens else None),
                q_mean=(float(np.mean(ok)) if ok else None),
                q_med=(float(np.median(ok)) if ok else None),
                n_cens=len(cens), slow=slow,
                sr5=float(np.mean(srs)) if srs else None,
                ge=float(np.mean(ge_l)) if ge_l else None)


# ================= diagnosis =================
def load_ct_sk(T, verify=None):
    os.makedirs(OUT, exist_ok=True)
    ip = os.path.join(OUT, "eval_indices.npz")
    src = os.path.join(MAIN, "eval_indices.npz")
    if not os.path.exists(ip):
        if not os.path.exists(src):
            raise SystemExit(f"evaluation indices not found: {src}\n"
                             f"  the main fine evaluation (eval_nscan.py fine) must be run first.")
        shutil.copy2(src, ip)
    if verify is not None and os.path.exists(src):
        a, b = np.load(src), np.load(ip)
        same = (np.array_equal(a["idx"], b["idx"])
                and np.array_equal(a["orders"], b["orders"]))
        verify.append(f"index check {'match ✓' if same else '**mismatch**'}"
                      f"  (2000 traces, {len(b['orders'])} accumulation orders, compared with the main fine evaluation)")
    d = np.load(os.path.join(E.SPLIT_DIR, "kyber_test_crop.npz"))
    idx = np.load(ip)["idx"]
    ct = np.asarray(d["ciphertexts"], np.uint8)[idx]
    sk = np.asarray(d["labels"][0], np.int64)
    return T.v_ntt(T.v_unpack_10bit(ct)), sk


def diag(T, verify):
    uh, sk = load_ct_sk(T, verify)
    n_coef = 2 * E.LEAK_PAIRS
    _OFF.update(offsets_of(sk, n_coef))
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    vals = np.array([int(E.i16(sk[j])) for j in range(n_coef)], np.int64)
    offs = np.array([_OFF[j] for j in range(n_coef)], np.int64)

    say("=" * 80)
    say("E3 diagnosis — firmware sk representation and candidate enumeration")
    say("=" * 80)
    say(f"  signal coefficients {n_coef} (pairs 1~{E.LEAK_PAIRS})   actual range {vals.min()} ~ {vals.max()}")
    say(f"  q = {Q} | inside canonical form 0~{Q-1}: {int(((vals>=0)&(vals<Q)).sum())}/{n_coef}"
        f" | {int((vals<0).sum())} negative | {int((vals>=Q).sum())} at or above q")
    say("")
    say("  per-coefficient offset o = (sk - sk%q)/q  —  0 means canonical form")
    for o in sorted(set(offs.tolist())):
        js = np.nonzero(offs == o)[0]
        say(f"    o = {o:+d} : {len(js):>3} coefs"
            + (f"   coefficients {[int(j) for j in js[:12]]}{' …' if len(js)>12 else ''}"
               if o != 0 else ""))

    say("")
    say("Per pair — mismatch rate of the correct candidate's predicted label vs the actual label")
    say(f"  {'pair':>4}{'sk[2p]':>9}{'o':>4}{'sk[2p+1]':>10}{'o':>4}"
        f"{'canonical':>10}{'oracle':>9}   cause")
    say("  " + "-" * 76)
    rows = []
    for p in range(E.LEAK_PAIRS):
        base, _ = E.pair_zeta(T, p)
        a0, a1 = int(vals[base]), int(vals[base + 1])
        o0, o1 = _OFF[base], _OFF[base + 1]
        m_can = true_mismatch(T, uh, sk, p, "canonical")
        m_or = true_mismatch(T, uh, sk, p, "oracle")
        why = ""
        if m_can > 0:
            nz = [f"sk[{j}]={v} (o={o:+d})" for j, v, o in
                  ((base, a0, o0), (base + 1, a1, o1)) if o != 0]
            why = ("outside canonical form: " + ", ".join(nz)) if nz else "offset 0 but mismatch — needs checking"
        rows.append((p, a0, a1, m_can, m_or, o0, o1, why))
        if m_can > 0 or m_or > 0 or p == CLEAN_PAIR:
            say(f"  {p+1:>4}{a0:>9}{o0:>+4d}{a1:>10}{o1:>+4d}"
                f"{m_can*100:>9.2f}%{m_or*100:>8.2f}%   {why}")
    n_can = sum(1 for r in rows if r[3] > 0)
    n_or = sum(1 for r in rows if r[4] > 0)
    say("")
    say(f"  mismatching pairs with canonical form {n_can} / with oracle {n_or} pairs")
    ok = (n_or == 0)
    msg = (f"E3 mismatch-rate check: 0% on all 39 pairs with the oracle representation "
           f"{'✓' if ok else '**failed**'}  (canonical form mismatches on {n_can} pairs)")
    verify.append(msg)
    say(f"  -> {'**0% on all 39 pairs. Request 5-2 check passed**' if ok else '**not passed — not evaluating**'}")

    say("")
    say("paper chat additional check (3)")
    nz_coef = [j for j in range(n_coef) if _OFF[j] != 0]
    nz_pair = sorted({j // 2 for j in nz_coef})
    can_pair = [r[0] for r in rows if r[3] > 0]
    say(f"  (a) coefficients with offset != 0: {len(nz_coef)} -> pairs {[p+1 for p in nz_pair]}")
    say(f"      canonical-form mismatch pairs   -> pairs {[p+1 for p in can_pair]}")
    only_in = [p for p in nz_pair if p not in can_pair]
    only_mm = [p for p in can_pair if p not in nz_pair]
    if not only_in and not only_mm:
        say("      -> **the two sets are exactly the same.** All mismatches are explained by offsets")
    else:
        if only_in:
            say(f"      -> pairs with an offset but 0% mismatch {[p+1 for p in only_in]}")
            say("         (cases where that offset did not change the pattern under this b distribution. Harmless)")
        if only_mm:
            say(f"      -> **pairs with a mismatch but offset 0 {[p+1 for p in only_mm]}** "
                f"— not explained by offsets")
    say("")
    if PAIR32 < E.LEAK_PAIRS:
        b32, _ = E.pair_zeta(T, PAIR32)
        o0, o1 = _OFF[b32], _OFF[b32 + 1]
        v0, v1 = int(vals[b32]), int(vals[b32 + 1])
        say(f"  (b) pair 32 (pair_idx {PAIR32}) — which of the two coefficients is shifted")
        say(f"      a0 = sk[{b32}] = {v0:>6}  o = {o0:+d}        "
            f"a1 = sk[{b32+1}] = {v1:>6}  o = {o1:+d}")
        which = ("both" if o0 and o1 else "a0 (even coefficient)" if o0
                 else "a1 (odd coefficient)" if o1 else None)
        concl = ((which + ": outside canonical form") if which
                 else "**both have offset 0 — the 9.2% is not explained by offsets**")
        say(f"      -> {concl}")
        say(f"      canonical-form mismatch {rows[PAIR32][3]*100:.2f}%  ->  "
            f"oracle {rows[PAIR32][4]*100:.2f}%")

    orp = oracle_pairs(rows)
    say("")
    say("paper chat instruction (2) — pairs to compute anew with oracle")
    say(f"  newly computed {len(orp)} pairs: {[p+1 for p in orp]}")
    say(f"    = pairs with offset != 0: {len(nz_pair)} + control pair for checking path reproduction "
        f"{CLEAN_PAIR+1} (1 pair)")
    say(f"  the remaining {E.LEAK_PAIRS-len(orp)} pairs have tables identical to canonical form -> existing values are taken and merged")

    say("")
    say("Note on the cause of the mismatch (paper chat item F)")
    say("  Some sk coefficients held by the firmware are outside canonical form 0..q-1 (offset table above).")
    say("  The evaluation folds those coefficients mod q and searches the 0..q-1 candidates, but fqmul is a Montgomery")
    say("  reduction, so fqmul(a,b) and fqmul(a+q,b) are equal only mod q and the stored 16-bit")
    say("  patterns can differ. The mismatch rate depends on the distribution of b (= ciphertext NTT coefficients)")
    say("  and so differs per pair — hence 2~3% per pair, and 9.2% for pair 32.")
    say("  Laying out the candidates with that coefficient's actual offset makes the correct candidate bit-identical")
    say("  to the firmware value, giving 0%. However, the offset can only be obtained by knowing the correct key, so this is not an attack but")
    say("  an **upper bound** (paper chat decision: the 2-representation version at 4x cost is not built; labelled as oracle).")
    say("=" * 80)

    open(os.path.join(OUT, "diag.txt"), "w", encoding="utf-8").write(
        "\n".join(lines) + "\n")
    print(f"\nsaved: {os.path.join(OUT, 'diag.txt')}")
    return ok, rows, orp


# ================= targets =================
def orig_targets():
    """required 1 + 2 — original heads."""
    t = [(f"N{n}_s{s}", E.run_id_of("resnet", n, s), "orig", "resnet", n, s,
          None, MAIN, "main")
         for n in (29, 39, 49) for s in range(10)]
    t += [("D", E.run_id_of("resnet", 39, 0), "orig", "resnet", 39, 0,
           os.path.join(CTRL, "D", "resnet_N39_s0.pth"), SRC_D, "D"),
          ("D29", E.run_id_of("resnet", 29, 0), "orig", "resnet", 29, 0,
           os.path.join(CTRL, "D29", "resnet_N29_s0.pth"), SRC_D, "D29")]
    return t


TAG_RE = __import__("re").compile(
    r"(?P<arch>resnet|mobilenet)_N(?P<n>\d+)_s(?P<s>\d+)(?P<ctrl>_D29|_D)?\b")


def local_targets():
    """required 3 — local heads.

    Tags carry arch, N and seed in the form `A1_resnet_N39_s0`, `A3_resnet_N39_s0_D`, so
    they are read as-is (probe_local_ext is not imported).

    The canonical-form comparison values must come from `eval_qfine_localrec/eval_pairs.csv`, but the
    run_id notation of that file is unknown. So **instead of guessing, candidates are tried in turn and
    the name that matched is printed.** If none is found, only that run is skipped and the list of
    run_ids in that file is shown.
    """
    fs = sorted(glob.glob(os.path.join(EXT, "logp_*.npz")))
    if not fs:
        print(f"  local head: no logp_*.npz in {EXT} — skipping")
        print("  (once E2 finishes, giving --local again to the same command continues from there)")
        return []
    tags = [os.path.basename(f)[len("logp_"):-len(".npz")] for f in fs]
    print(f"  local head: logp {len(fs)} files.  keys of {os.path.basename(fs[0])}: "
          f"{sorted(np.load(fs[0]).files)}")
    have = load_pairs_csv(os.path.join(SRC_LR, "eval_pairs.csv"))
    rids = sorted({k[0] for k in have})
    if not rids:
        print(f"  local head: no fine data in {SRC_LR}/eval_pairs.csv")
        print("    (probe_local_recover must first be run with fine to produce the canonical-form comparison values)")
        return []
    out, unresolved = [], []
    for tg in tags:
        # use arch, n_pairs and seed from inside the npz. If absent, extract them from the tag name
        z = np.load(os.path.join(EXT, f"logp_{tg}.npz"))
        k = set(z.files)
        arch = n = s = None
        if {"arch", "n_pairs", "seed"} <= k:
            arch, n, s = str(z["arch"]), int(z["n_pairs"]), int(z["seed"])
        else:
            m = TAG_RE.search(tg)
            if not m:
                print(f"  local head: {tg} — npz has no arch·n_pairs·seed and"
                      f" they could not be read from the tag either (keys {sorted(k)}) — skipped")
                continue
            arch, n, s = m.group("arch"), int(m.group("n")), int(m.group("s"))
            print(f"  local head: {tg} — npz lacks the keys, read from the tag name")
        m = TAG_RE.search(tg)
        ctrl = str(z["ckpt"]) if "ckpt" in k else ((m.group("ctrl") or "").lstrip("_")
                                                  if m else "")
        if ctrl in ("main", "None", "none"):
            ctrl = ""
        navail = min(n, E.LEAK_PAIRS)
        short = tg.split("_", 1)[0]                      # A1, B1a, …
        core = f"{arch}_N{n}_s{s}"
        cands = ([str(z["run_id"])] if "run_id" in k else []) + \
                [tg, tg + "_local", short, short + "_local",
                 core, core + "_local", ctrl or None,
                 (ctrl + "_local") if ctrl else None]
        pick = None
        for c in cands:
            if c and all((c, p) in have for p in range(navail)):
                pick = c
                break
        if pick is None:
            unresolved.append(tg)
            continue
        print(f"  local head: {tg:<26} -> {arch} N{n} s{s}"
              f"{' (' + ctrl + ')' if ctrl else '':<7}  canonical-form run_id '{pick}'")
        out.append((tg, pick, "local", arch, n, s,
                    os.path.join(EXT, f"logp_{tg}.npz"), SRC_LR, ctrl or "main"))
    if unresolved:
        print(f"  local head: tags with no canonical-form comparison values found: {len(unresolved)} "
              f"-> {unresolved}")
        print(f"    {SRC_LR}/eval_pairs.csv run_ids ({len(rids)}):")
        for i in range(0, min(len(rids), 24), 4):
            print("      " + "  ".join(f"{x:<22}" for x in rids[i:i + 4]))
        print("    send this list and the names will be pinned down")
    return out


# ================= oracle computation =================
def lp_of(T, kind, arch, n, seed, src, X, dev):
    """Original heads from the checkpoint, local heads from the logp npz."""
    if kind == "local":
        z = np.load(src)
        k = "logp" if "logp" in z.files else z.files[0]
        lp = np.asarray(z[k], np.float32)
        if lp.shape[0] != len(X):
            return None, None, f"logp trace count {lp.shape[0]} != {len(X)}"
        return lp, lp.shape[1], None
    f = src if src else os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")
    if not os.path.exists(f):
        return None, None, f"checkpoint not found: {f}"
    cp = torch.load(f, map_location="cpu", weights_only=False)
    h = int(cp.get("heads", 2 * n))
    model = E.build(T, arch, n, h, cp.get("resnet_variant")).to(dev)
    E.load_state(model, cp["model"])
    model.eval()
    lp = np.empty((len(X), h, 17), np.float32)
    with torch.no_grad():
        for s0 in range(0, len(X), E.INFER_BATCH):
            xb = torch.from_numpy(X[s0:s0 + E.INFER_BATCH]).unsqueeze(1).to(dev)
            lp[s0:s0 + len(xb)] = torch.stack(
                [torch.log_softmax(o.float(), 1) for o in model(xb)], 1).cpu().numpy()
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return lp, h, None


def oracle_ranks(lp, tabs, tflat, plist, orders, hw_idx, chunk, dev, label):
    out, t0 = {}, time.time()
    for k, p in enumerate(plist):
        tabs_d = [torch.from_numpy(t).to(dev) for t in tabs[p]]
        lp0 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p, :])).to(dev)
        lp1 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p + 1, :])).to(dev)
        rk, _ = E.recover_pair(tabs_d, lp0, lp1, hw_idx, tflat[p],
                              orders, chunk, dev)
        out[p] = rk["sum"]
        del tabs_d, lp0, lp1
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if k == 0:
            el = time.time() - t0
            print(f"    1 pair {el:.1f}s -> this run about {el*len(plist)/60:.1f} min")
    return out, time.time() - t0


# ================= main body =================
def main():
    flags = {a for a in _USER if a.startswith("-")}
    want = [a for a in _USER if not a.startswith("-")]
    os.makedirs(OUT, exist_ok=True)
    verify = []

    T = E.import_train()
    ok, drows, ORP = diag(T, verify)
    if "--diag" in flags:
        open(os.path.join(OUT, "verify.txt"), "w", encoding="utf-8").write(
            "\n".join(verify) + "\n")
        print("\n" + "\n".join(verify))
        return
    if not ok:
        raise SystemExit("\nDiagnosis not passed — not evaluating, per request 5-2.")
    can_m = {r[0]: r[3] for r in drows}
    offs = {r[0]: (r[5], r[6]) for r in drows}

    probe = "--probe" in flags
    targets = []
    if probe:
        tg = want or PROBE_DEFAULT
        targets = [t for t in orig_targets() if t[0] in tg]
        if not targets:
            raise SystemExit(f"no --probe targets. tag examples: {PROBE_DEFAULT}")
    else:
        if "--body" in flags or not (flags - {"--table"}):
            targets += orig_targets()
        if "--local" in flags:
            targets += local_targets()
        if want:
            targets = [t for t in targets if t[0] in want]
    if not targets:
        raise SystemExit("no targets. Give one of --probe / --body / --local.")

    # ---- read existing canonical form + reproduction check ----
    srcs = sorted({t[7] for t in targets})
    canon = {}
    for s in srcs:
        d = load_pairs_csv(os.path.join(s, "eval_pairs.csv"))
        for (rid, p), v in d.items():
            canon[(s, rid, p)] = v
        print(f"canonical-form read {s}: {len({k[0] for k in d})} runs, "
              f"{len({k[1] for k in d})} pairs")
    verify.append(_reproduce(canon))

    # ---- grids ----
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunk = E.CHUNK or (64 if dev.type == "cuda" and
                        torch.cuda.get_device_properties(0).total_memory / 2**30 >= 40
                        else 16)
    plist = ORP if not probe else ORP
    print(f"\ndevice {dev}  CHUNK={chunk}  Q=1..{max(E.Q_LIST)}  repeated {E.N_REPEAT} times")
    print(f"pairs to compute with oracle: {len(plist)}, {[p+1 for p in plist]}")
    print(f"targets: {len(targets)}\n")
    E.needed_pairs = lambda: list(plist)
    E.OUT_DIR = OUT
    E.pair_tables = make_pair_tables("oracle")
    X, Y, prior, tabs, tflat, mism, orders = E.prepare(T, dev)
    bad = {p + 1: round(m * 100, 2) for p, m in mism.items() if m > 0}
    if bad:
        raise SystemExit(f"mismatch remains even with the oracle representation: {bad} — aborting")
    hw_idx = torch.from_numpy(np.asarray(T.HW16, np.int64)).to(dev)

    # ---- per run ----
    rows_run, rows_pq, rows_p32, rows_ep = [], [], [], []
    for tag, rid, kind, arch, n, seed, ck, src, cklab in targets:
        npz = os.path.join(OUT, f"res_{kind}_{tag}.npz")
        # local-head logp is 78 heads wide too, but only the learned pairs are valid (N29 has 29 pairs)
        navail = min(n, E.LEAK_PAIRS)
        pl = [p for p in plist if p < navail]
        allp = list(range(navail))
        have = {p for p in allp if (src, rid, p) in canon}
        miss = [p for p in allp if p not in have]
        if miss:
            print(f"[{kind}/{tag}] no canonical-form data — pairs {len(miss)} "
                  f"({[q+1 for q in miss[:6]]}…) in {src}  -> skipped")
            continue
        if os.path.exists(npz):
            orc = {int(k[1:]): v for k, v in np.load(npz).items() if k.startswith("p")}
            sec = 0.0
            print(f"[{kind}/{tag}] reusing existing oracle results")
        else:
            lp, heads, err = lp_of(T, kind, arch, n, seed, ck, X, dev)
            if lp is None:
                print(f"[{kind}/{tag}] skipped — {err}")
                continue
            print(f"[{kind}/{tag}] oracle pairs {len(pl)} ({heads} heads)")
            orc, sec = oracle_ranks(lp, tabs, tflat, pl, orders, hw_idx,
                                    chunk, dev, tag)
            np.savez(npz, **{f"p{p}": v for p, v in orc.items()})
            del lp
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        can_r = {p: canon[(src, rid, p)] for p in allp}
        orc_r = {p: orc.get(p, can_r[p]) for p in allp}   # offset-0 pairs are used as-is
        for repr_name, rr in (("canonical", can_r), ("oracle", orc_r)):
            for sname, lim in SUBSETS:
                # A run that does not even have lim pairs must not get that set name.
                # N29 has only 29 pairs, so asking for "1-39" would pick up 29 pairs and add one more row
                # with the same value under the wrong name -> N29 outputs only "1-29"
                if lim > navail:
                    continue
                ps = [p for p in allp if p < lim]
                if not ps:
                    continue
                st = run_stats(rr, ps)
                rows_run.append(dict(
                    run_id=f"{tag}_{repr_name}", tag=tag, heads_src=kind,
                    candidate_repr=repr_name, pair_set=sname, arch=arch,
                    n_pairs=n, seed=seed, ckpt=cklab, n_pairs_eval=len(ps),
                    q_star_max=("" if st["q_max"] is None else st["q_max"]),
                    q_star_mean=_r(st["q_mean"], 2), q_star_median=_r(st["q_med"], 1),
                    n_pairs_censored=st["n_cens"],
                    slowest_pair_idx=("" if st["slow"] is None else st["slow"]),
                    slowest_pair=("" if st["slow"] is None else st["slow"] + 1),
                    sr_q5=_r(st["sr5"], 4), ge_log2_qmax=_r(st["ge"], 3),
                    n_oracle_pairs=len([p for p in ps if p in orc]),
                    n_reused_pairs=len([p for p in ps if p not in orc]),
                    n_repeat=rr[ps[0]].shape[0], eval_time_sec=_r(sec, 1)))
            st = run_stats(rr, allp)
            if kind == "local" and repr_name == "canonical":
                key = (arch, n, seed, "" if cklab == "main" else cklab)
                if key in REF_LOCAL:
                    ref, got = REF_LOCAL[key], st["q_max"]
                    good = (got == ref) if ref is not None else (st["n_cens"] > 0)
                    verify.append(
                        f"local reproduction {tag}: canonical-form Q*max "
                        f"{got if got is not None else '>50'} "
                        f"(previous {ref if ref is not None else '>50'})"
                        f" {'✓' if good else '**differs**'}")
            for p in allp:
                q = st["qs"][p]
                rows_pq.append(dict(
                    run_id=f"{tag}_{repr_name}", tag=tag, heads_src=kind,
                    candidate_repr=repr_name, arch=arch, n_pairs=n, seed=seed,
                    pair_idx=p, pair=p + 1, q_star=("" if q is None else q),
                    censored=int(q is None),
                    ge_log2_qmax=_r(float(np.log2(rr[p].mean(0)[-1])), 3),
                    computed=int(p in orc), canon_mismatch=_r(can_m.get(p, 0.0), 5),
                    off_a0=offs.get(p, (0, 0))[0], off_a1=offs.get(p, (0, 0))[1]))
            if PAIR32 in allp:
                q = st["qs"][PAIR32]
                rows_p32.append(dict(
                    run_id=f"{tag}_{repr_name}", tag=tag, heads_src=kind,
                    candidate_repr=repr_name, arch=arch, n_pairs=n, seed=seed,
                    pair_idx=PAIR32, pair=PAIR32 + 1,
                    q_star=("" if q is None else q), censored=int(q is None),
                    ge_log2_qmax=_r(float(np.log2(rr[PAIR32].mean(0)[-1])), 3),
                    computed=int(PAIR32 in orc),
                    canon_mismatch=_r(can_m.get(PAIR32, 0.0), 5),
                    off_a0=offs.get(PAIR32, (0, 0))[0],
                    off_a1=offs.get(PAIR32, (0, 0))[1]))
        # 1-11: raw per-repeat ranks of the pairs computed with oracle
        rows_ep += [dict(run_id=f"{tag}_oracle", tag=tag, heads_src=kind,
                         candidate_repr="oracle", pair_idx=p, pair=p + 1,
                         repeat=rep, q_star=(E.q_star_of(orc[p][rep]) or -1),
                         **{f"rank_q{q}": round(float(v), 2)
                            for q, v in zip(E.Q_LIST, orc[p][rep])})
                    for p in sorted(orc) for rep in range(orc[p].shape[0])]
        s = next((r for r in rows_run if r["tag"] == tag
                  and r["candidate_repr"] == "oracle" and r["pair_set"] == "1-39"),
                 None) or rows_run[-1]
        c = next((r for r in rows_run if r["tag"] == tag
                  and r["candidate_repr"] == "canonical"
                  and r["pair_set"] == s["pair_set"]), None)
        print(f"    {s['pair_set']}  canonical-form Q*max {c['q_star_max'] or '>50'} / "
              f"mean {c['q_star_mean']}  ->  oracle {s['q_star_max'] or '>50'} / "
              f"mean {s['q_star_mean']}   slowest pair "
              f"{c['slowest_pair']} -> {s['slowest_pair']}")

    print("")
    for nm, fields, rows, key in (
            # the pair set is not put into key — if it were, wrong-set rows written by an earlier version
            # (like "1-39" for N29) would survive as "not produced this time"
            ("eval_runs.csv", RUN_FIELDS, rows_run,
             ("tag", "heads_src", "candidate_repr")),
            ("pair_qstar.csv", PQ_FIELDS, rows_pq,
             ("tag", "heads_src", "candidate_repr", "pair_idx")),
            ("pair32.csv", PQ_FIELDS, rows_p32,
             ("tag", "heads_src", "candidate_repr")),
            ("eval_pairs.csv", EP_FIELDS, rows_ep,
             ("tag", "heads_src", "pair_idx", "repeat"))):
        if rows:
            write_rows(os.path.join(OUT, nm), fields, rows, key)
    E.OUT_DIR = MAIN
    open(os.path.join(OUT, "verify.txt"), "a", encoding="utf-8").write(
        "\n".join(verify) + "\n")
    print("\nVerification output (request 9-3)")
    for v in verify:
        print("  " + v)
    table()


def _r(x, nd):
    return "" if x is None or not np.isfinite(x) else round(float(x), nd)


def write_rows(path, fields, rows, key):
    """**Overwrites** instead of appending.

    `eval_nscan.append_rows` does not rewrite the header if the file exists. So rerunning
    with a version whose column layout changed silently puts rows in the new column order under the old header
    (that was the KeyError 'heads_src'). Rerunning the same run also duplicates rows.
    The expensive part is cached in res_*.npz, so rewriting the CSV every time is correct.

    - if the header differs from the current columns, move the existing file to .bak and write anew
    - if it is the same, drop only the existing rows whose key was recomputed this time and keep the rest
    """
    keep = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            old = list(rd.fieldnames or [])
            if old != list(fields):
                bak = path + ".bak"
                os.replace(path, bak)
                print(f"    [note] column layout of {os.path.basename(path)} differs; "
                      f"moving it to {os.path.basename(bak)} and writing anew")
                print(f"           before {len(old)} columns -> now {len(fields)} columns")
            else:
                mine = {tuple(r.get(k, "") for k in key) for r in rows}
                keep = [r for r in rd
                        if tuple(str(r.get(k, "")) for k in key) not in
                        {tuple(str(x) for x in t) for t in mine}]
    if os.path.exists(path):
        os.remove(path)
    E.append_rows(path, fields, keep + rows)
    n = f" (kept {len(keep)} existing rows)" if keep else ""
    print(f"    saved {os.path.basename(path)}  {len(keep)+len(rows)} rows{n}")


def _reproduce(canon):
    """Request 1-3 — whether the canonical-form read matches the previously published values."""
    print("\nReproduction check (request 1-3) — re-aggregating the existing fine results")
    print(f"  {'run':<18}{'pair set':>7}{'Q*max':>14}{'Q*mean':>14}{'SR@5':>14}")
    hit = miss = 0
    for rid, (qm, qa, sr) in REF_ALL.items():
        pr = {p: v for (s, r, p), v in canon.items() if r == rid and s == MAIN}
        if not pr:
            continue
        st = run_stats(pr, sorted(pr))
        good = (st["q_max"] == qm and abs((st["q_mean"] or 0) - qa) < 0.15
                and abs((st["sr5"] or 0) - sr) < 0.006)
        hit += good
        miss += (not good)
        c1 = "{} (={})".format(st["q_max"] if st["q_max"] is not None else ">50", qm)
        c2 = "{:.1f} (={})".format(st["q_mean"] or float("nan"), qa)
        c3 = "{:.3f} (={:.3f})".format(st["sr5"] or float("nan"), sr)
        mark = "✓" if good else "**differs**"
        print(f"  {rid:<18}{'1-39':>7}{c1:>14}{c2:>14}{c3:>14}  {mark}")
    for rid, q29 in REF_29.items():
        pr = {p: v for (s, r, p), v in canon.items()
              if r == rid and s == MAIN and p < 29}
        if not pr:
            continue
        st = run_stats(pr, sorted(pr))
        good = (st["q_max"] == q29)
        hit += good
        miss += (not good)
        c1 = "{} (={})".format(st["q_max"] if st["q_max"] is not None else ">50", q29)
        mark = "✓" if good else "**differs**"
        print(f"  {rid:<18}{'1-29':>7}{c1:>14}{'':>14}{'':>14}  {mark}")
    s = (f"reproduction check (1-3, section 12): of {hit+miss} reference values, {hit} match"
         + ("" if miss == 0 else f", **{miss} differ**"))
    print("  " + s)
    return s


# ================= table =================
def table():
    p = os.path.join(OUT, "eval_runs.csv")
    if not os.path.exists(p):
        print(f"\n{p} not found.")
        return
    with open(p, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = list(rd.fieldnames or [])
        rows = list(rd)
    lack = [c for c in ("heads_src", "candidate_repr", "pair_set", "q_star_max")
            if c not in cols]
    if lack:
        print(f"\n{p} is missing columns: {lack}")
        print("  This file was written by an earlier version. Delete it and rerun to write the new columns:")
        print(f"    rm {p}")
        print("  (if res_*.npz files remain, the oracle computation is not redone)")
        return
    if not rows:
        print(f"\n{p} has 0 rows.")
        return
    g = lambda r, k: (float(r[k]) if r[k] not in ("", None) else None)

    print("\n" + "=" * 104)
    print("E3 summary — request 5-4(3).  oracle = **upper bound** using the correct key's coefficient offsets, not an attack")
    print("=" * 104)
    print(f"{'head':<7}{'N':>4}{'pairs':>8}{'seeds':>6}"
          f"{'canon Q*max':>13}{'Q*mean':>8}{'slowest':>13}"
          f"{'oracle Q*max':>14}{'Q*mean':>8}{'slowest':>13}{'dQ*max':>8}")
    print("-" * 104)
    sumrows = []
    keys = sorted({(r["heads_src"], r["arch"], int(r["n_pairs"]), r["pair_set"],
                    r["ckpt"]) for r in rows},
                  key=lambda k: (k[0] != "orig", k[4] != "main", k[1], k[2], k[3]))
    for hs, arch, n, ps, ck in keys:
        sel = [r for r in rows if r["heads_src"] == hs and r["arch"] == arch
               and int(r["n_pairs"]) == n and r["pair_set"] == ps and r["ckpt"] == ck]
        out = {}
        for rep in ("canonical", "oracle"):
            rs = [r for r in sel if r["candidate_repr"] == rep]
            if not rs:
                continue
            qm = [g(r, "q_star_max") for r in rs if g(r, "q_star_max") is not None]
            qa = [g(r, "q_star_mean") for r in rs if g(r, "q_star_mean") is not None]
            sl = [r["slowest_pair"] for r in rs if r["slowest_pair"] != ""]
            top = (max(set(sl), key=sl.count) if sl else "—")
            out[rep] = (len(rs), qm, qa, f"{top} ({sl.count(top)}/{len(sl)})" if sl else "—",
                        sum(int(r["n_pairs_censored"]) for r in rs))
        if "canonical" not in out:
            continue
        nc, qmc, qac, slc, cc = out["canonical"]
        no, qmo, qao, slo, co = out.get("oracle", (0, [], [], "—", 0))
        f = lambda v: (f"{np.mean(v):.1f}±{np.std(v):.1f}" if len(v) > 1
                       else (f"{v[0]:.0f}" if v else "—"))
        fa = lambda v: (f"{np.mean(v):.2f}" if v else "—")
        d = (f"{np.mean(qmo)-np.mean(qmc):+.1f}" if qmc and qmo else "—")
        lbl = ck if ck != "main" else f"{hs}"
        print(f"{lbl:<7}{n:>4}{ps:>8}{nc:>6}"
              f"{f(qmc):>13}{fa(qac):>8}{slc:>13}"
              f"{f(qmo):>14}{fa(qao):>8}{slo:>13}{d:>8}")
        sumrows.append(dict(
            heads_src=hs, arch=arch, n_pairs=n, pair_set=ps, n_seeds=nc,
            canon_q_star_max_mean=_r(np.mean(qmc) if qmc else None, 2),
            canon_q_star_max_std=_r(np.std(qmc) if len(qmc) > 1 else None, 2),
            canon_q_star_mean_mean=_r(np.mean(qac) if qac else None, 2),
            canon_q_star_mean_std=_r(np.std(qac) if len(qac) > 1 else None, 2),
            canon_slowest_top=slc,
            oracle_q_star_max_mean=_r(np.mean(qmo) if qmo else None, 2),
            oracle_q_star_max_std=_r(np.std(qmo) if len(qmo) > 1 else None, 2),
            oracle_q_star_mean_mean=_r(np.mean(qao) if qao else None, 2),
            oracle_q_star_mean_std=_r(np.std(qao) if len(qao) > 1 else None, 2),
            oracle_slowest_top=slo,
            d_q_star_max=_r((np.mean(qmo) - np.mean(qmc)) if qmc and qmo else None, 2),
            d_q_star_mean=_r((np.mean(qao) - np.mean(qac)) if qac and qao else None, 2),
            n_censored_canon=cc, n_censored_oracle=co))
    print("-" * 104)
    print("  Q*max is the mean±std of the per-seed maxima. Seeds with censoring are excluded from Q*max")
    print("  'pairs 1-29 only' rows = comparison on the same pair set as N29 (request 5-4)")

    sp = os.path.join(OUT, "summary_5_4.csv")
    if os.path.exists(sp):
        os.remove(sp)
    if sumrows:
        E.append_rows(sp, SUM_FIELDS, sumrows)
        print(f"\nsaved: {sp}")

    pp = os.path.join(OUT, "pair32.csv")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            rd = csv.DictReader(f)
            pc = list(rd.fieldnames or [])
            pr = list(rd)
        if not pr or any(c not in pc for c in ("tag", "candidate_repr", "q_star")):
            print(f"\n{pp} could not be read (if it is a file from an earlier version, delete it and rerun)")
            print("=" * 104)
            print(f"raw data: {OUT}")
            return
        print("\n" + "=" * 70)
        print("Pair 32 alone — request 5-4(4)")
        print("=" * 70)
        print(f"  {'head':<7}{'run':<10}{'canon Q*':>11}{'oracle Q*':>11}{'diff':>7}")
        for tg in sorted({r["tag"] for r in pr}):
            a = next((r for r in pr if r["tag"] == tg
                      and r["candidate_repr"] == "canonical"), None)
            b = next((r for r in pr if r["tag"] == tg
                      and r["candidate_repr"] == "oracle"), None)
            if not (a and b):
                continue
            d = (f"{int(b['q_star'])-int(a['q_star']):+d}"
                 if a["q_star"] and b["q_star"] else "—")
            print(f"  {a['heads_src']:<7}{tg:<10}{(a['q_star'] or '>50'):>11}"
                  f"{(b['q_star'] or '>50'):>11}{d:>7}")
        print(f"  mismatch rate {float(pr[0]['canon_mismatch'])*100:.2f}%  "
              f"offset a0 {pr[0]['off_a0']} / a1 {pr[0]['off_a1']}")
    print("=" * 104)
    print(f"raw data: {OUT}")


if __name__ == "__main__":
    if "--table" in _USER:
        table()
    else:
        main()
