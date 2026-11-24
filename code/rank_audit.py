r"""
Tie audit — optimistic rank vs tie-aware rank, and the number of tied candidates as a function of Q

Why this is needed
------------------
`tie_rank()` in `eval_nscan.py` computes three values internally and **returns only the tie-aware one**.

    gt = (score > true).sum()          # optimistic rank = gt + 1
    eq = (score == true).sum() - 1     # number of tied candidates
    return gt + 1 + eq / 2.0           # only this is saved

So the optimistic rank and the tie count cannot be recovered from `eval_pairs.csv` (two values cannot be
separated from one). A re-evaluation is needed.

The cost is small. The paper chat request covers only **ResNet N39·N128**, so it is **20 models**, not 160
(fine evaluation took 9.4 hours for 160 → about 1.2 hours).

What it produces
----------------
All three values are recorded at every Q depth. Because they come from the same score array, the two rank
definitions are compared under **exactly the same conditions** (no nondeterminism between reruns creeps in).

  rank_audit_curves.csv   model x pair x mode x Q — repeat-averaged GE·SR·tie count, under both definitions
  rank_audit_runs.csv     model x mode — Q*_max / Q*_mean / SR@5 under both definitions
  audit_{arch}_N{n}_s{seed}.npz   raw data (full gt·eq for every repeat)

The Q\* definition is the same as in eval_nscan: the Q at which the per-pair GE (repeat-averaged rank) first becomes ≤ 1.
The optimistic rank sends all ties below, so it reaches GE=1 sooner — that difference is what is measured.

Run
---
    python rank_audit.py                # evaluate (resumes if interrupted)
    python rank_audit.py summary        # only re-aggregate from the npz files

Place in the same folder as eval_nscan.py. It runs with the fine profile, so it reuses `eval_qfine/eval_indices.npz`
(2000 traces·20 repeats·Q 1..50) as-is and is directly comparable with the existing fine results.
"""

import csv
import glob
import os
import re
import sys

import numpy as np
import torch

SUMMARY_ONLY = "summary" in sys.argv[1:]                 # capture before overwriting argv
sys.argv = [sys.argv[0], "fine"] + [a for a in sys.argv[1:] if a != "summary"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_nscan as E                                   # noqa: E402  (placed after because it must be switched to fine)

# ==========================================================
AUDIT_ARCHS = ["resnet"]
AUDIT_N = [39, 128]
TIE_Q = [1, 5, 10, 20]        # Q values at which the tied-candidate count is extracted separately (figure material)
SR_Q = 5                      # SR@5
OUT = os.path.join(E.OUT_DIR, "rank_audit")
# ==========================================================

CURVE_FIELDS = ["arch", "n_pairs", "seed", "pair", "mode", "q",
                "ge_tie", "ge_opt", "sr_tie", "sr_opt", "n_tied_mean", "n_tied_median"]
RUN_FIELDS = ["arch", "n_pairs", "seed", "mode", "n_pairs_eval",
              "q_star_max_tie", "q_star_max_opt", "q_star_mean_tie", "q_star_mean_opt",
              "n_censored_tie", "n_censored_opt",
              f"sr_q{SR_Q}_tie", f"sr_q{SR_Q}_opt"] + \
             [f"n_tied_q{q}" for q in TIE_Q]


def audit_pair(tabs_d, lp0, lp1, hw_idx, true_flat, orders, chunk, dev):
    """Same score accumulation as eval_nscan.recover_pair, but keeps gt·eq as-is for every Q.

    Returns: gt[mode] (R, nQ), eq[mode] (R, nQ)
      optimistic rank = gt + 1,  tie-aware rank = gt + 1 + eq/2,  number of tied candidates = eq
    """
    A0, B0, A1, B1 = tabs_d
    R, nQ = len(orders), len(E.Q_LIST)
    gt = {m: np.zeros((R, nQ), np.int64) for m in E.MODES}
    eq = {m: np.zeros((R, nQ), np.int64) for m in E.MODES}
    for rep, order in enumerate(orders):
        s = {m: torch.zeros(E.Q * E.Q, device=dev, dtype=torch.float32) for m in E.MODES}
        ptr = 0
        for j, qn in enumerate(E.Q_LIST):
            ids_np = order[ptr:qn]
            for c in range(0, len(ids_np), chunk):
                ids = torch.as_tensor(ids_np[c:c + chunk], device=dev)
                g0 = E._chunk_score(A0[ids], B0[ids], lp0[ids][:, hw_idx].contiguous())
                g1 = E._chunk_score(A1[ids], B1[ids], lp1[ids][:, hw_idx].contiguous())
                s["r0"] += g0
                s["sum"] += g0
                s["sum"] += g1
            ptr = qn
            for m in E.MODES:
                st = s[m][true_flat]
                gt[m][rep, j] = int((s[m] > st).sum().item())
                eq[m][rep, j] = int((s[m] == st).sum().item()) - 1
    return gt, eq


def q_star(ge_row):
    """The Q at which the GE curve first becomes <=1. Same rule as eval_nscan.curve."""
    hit = np.where(np.asarray(ge_row) <= 1.0)[0]
    return E.Q_LIST[hit[0]] if len(hit) else None


def main():
    T = E.import_train()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    # The grid needs only signal pairs (fine has PHANTOM_MAX=0). Narrow N_LIST to reduce memory
    E.ARCHS, E.N_LIST = AUDIT_ARCHS, AUDIT_N
    chunk = E.CHUNK
    if chunk is None:
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if dev.type == "cuda" else 0
        chunk = 64 if mem >= 40 else 16
    print(f"device {dev}  CHUNK={chunk}  EVAL_N={E.EVAL_N}  N_REPEAT={E.N_REPEAT}")
    print(f"audit targets {AUDIT_ARCHS} x N{AUDIT_N} x {len(list(E.SEEDS))} seeds "
          f"= {len(AUDIT_ARCHS)*len(AUDIT_N)*len(list(E.SEEDS))} models,  Q = 1..{max(E.Q_LIST)} every trace")

    X, Y, prior, uh_tabs, true_flats, mism, orders = E.prepare(T, dev)
    hw_idx = torch.from_numpy(np.asarray(T.HW16, np.int64)).to(dev)

    todo = [(a, n, s) for a in AUDIT_ARCHS for n in AUDIT_N for s in E.SEEDS
            if not os.path.exists(os.path.join(OUT, f"audit_{a}_N{n}_s{s}.npz"))]
    print(f"\n{len(todo)} models remaining (completed ones skipped)\n")

    for k, (arch, n, seed) in enumerate(todo, 1):
        f = os.path.join(E.MODEL_DIR, f"{arch}_N{n}_s{seed}.pth")
        if not os.path.exists(f):
            print(f"  [missing] {os.path.basename(f)}")
            continue
        print(f"[{k}/{len(todo)}] {arch} N{n} s{seed}")
        ck = torch.load(f, map_location="cpu", weights_only=False)
        heads = int(ck.get("heads", 2 * n))
        model = E.build(T, arch, n, heads, ck.get("resnet_variant")).to(dev)
        E.load_state(model, ck["model"])
        model.eval()
        ncls = prior.shape[1]
        lp = np.empty((len(X), heads, ncls), np.float32)
        with torch.no_grad():
            for s0 in range(0, len(X), E.INFER_BATCH):
                xb = torch.from_numpy(X[s0:s0 + E.INFER_BATCH]).unsqueeze(1).to(dev)
                outs = model(xb)
                lp[s0:s0 + len(xb)] = torch.stack(
                    [torch.log_softmax(o.float(), 1) for o in outs], 1).cpu().numpy()
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        sig = list(range(min(n, E.LEAK_PAIRS)))
        GT = {m: [] for m in E.MODES}
        EQ = {m: [] for m in E.MODES}
        for p in sig:
            tabs_d = [torch.from_numpy(t).to(dev) for t in uh_tabs[p]]
            lp0 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p, :])).to(dev)
            lp1 = torch.from_numpy(np.ascontiguousarray(lp[:, 2 * p + 1, :])).to(dev)
            g, e = audit_pair(tabs_d, lp0, lp1, hw_idx, true_flats[p], orders, chunk, dev)
            for m in E.MODES:
                GT[m].append(g[m])
                EQ[m].append(e[m])
        np.savez(os.path.join(OUT, f"audit_{arch}_N{n}_s{seed}.npz"),
                 pairs=np.array(sig), q_list=np.array(E.Q_LIST),
                 **{f"gt_{m}": np.array(GT[m]) for m in E.MODES},
                 **{f"eq_{m}": np.array(EQ[m]) for m in E.MODES})
        tied1 = float(np.mean(EQ["sum"][0][:, 0]))
        print(f"    saved {len(sig)} pairs. Mean tied candidates for pair 0 at Q=1: {tied1:,.0f} candidates")

    summary()


def summary():
    files = sorted(glob.glob(os.path.join(OUT, "audit_*_N*_s*.npz")))
    if not files:
        print("No npz files.")
        return
    for p in (os.path.join(OUT, "rank_audit_curves.csv"), os.path.join(OUT, "rank_audit_runs.csv")):
        if os.path.exists(p):
            os.remove(p)
    for f in files:
        m = re.search(r"audit_(.+)_N(\d+)_s(\d+)\.npz$", os.path.basename(f))
        arch, n, seed = m.group(1), int(m.group(2)), int(m.group(3))
        d = np.load(f)
        sig, Q_LIST = list(d["pairs"]), list(d["q_list"])
        crows, rrows = [], []
        for mode in E.MODES:
            gt, eq = d[f"gt_{mode}"], d[f"eq_{mode}"]          # (P, R, nQ)
            r_opt = gt + 1.0
            r_tie = gt + 1.0 + eq / 2.0
            ge_opt, ge_tie = r_opt.mean(1), r_tie.mean(1)      # (P, nQ)
            sr_opt = (r_opt == 1).mean(1)
            sr_tie = (r_tie == 1).mean(1)
            for pi, p in enumerate(sig):
                for qi, q in enumerate(Q_LIST):
                    crows.append(dict(arch=arch, n_pairs=n, seed=seed, pair=int(p), mode=mode,
                                      q=int(q),
                                      ge_tie=round(float(ge_tie[pi, qi]), 4),
                                      ge_opt=round(float(ge_opt[pi, qi]), 4),
                                      sr_tie=round(float(sr_tie[pi, qi]), 4),
                                      sr_opt=round(float(sr_opt[pi, qi]), 4),
                                      n_tied_mean=round(float(eq[pi, :, qi].mean()), 1),
                                      n_tied_median=int(np.median(eq[pi, :, qi]))))
            qs_t = [q_star(ge_tie[pi]) for pi in range(len(sig))]
            qs_o = [q_star(ge_opt[pi]) for pi in range(len(sig))]
            ok_t = [q for q in qs_t if q is not None]
            ok_o = [q for q in qs_o if q is not None]
            si = Q_LIST.index(SR_Q) if SR_Q in Q_LIST else -1
            row = dict(arch=arch, n_pairs=n, seed=seed, mode=mode, n_pairs_eval=len(sig),
                       q_star_max_tie=(max(ok_t) if ok_t and len(ok_t) == len(sig) else ""),
                       q_star_max_opt=(max(ok_o) if ok_o and len(ok_o) == len(sig) else ""),
                       q_star_mean_tie=(round(float(np.mean(ok_t)), 2) if ok_t else ""),
                       q_star_mean_opt=(round(float(np.mean(ok_o)), 2) if ok_o else ""),
                       n_censored_tie=len(sig) - len(ok_t), n_censored_opt=len(sig) - len(ok_o))
            row[f"sr_q{SR_Q}_tie"] = round(float(sr_tie[:, si].mean()), 4) if si >= 0 else ""
            row[f"sr_q{SR_Q}_opt"] = round(float(sr_opt[:, si].mean()), 4) if si >= 0 else ""
            for q in TIE_Q:
                qi = Q_LIST.index(q) if q in Q_LIST else -1
                row[f"n_tied_q{q}"] = round(float(eq[:, :, qi].mean()), 1) if qi >= 0 else ""
            rrows.append(row)
        E.append_rows(os.path.join(OUT, "rank_audit_curves.csv"), CURVE_FIELDS, crows)
        E.append_rows(os.path.join(OUT, "rank_audit_runs.csv"), RUN_FIELDS, rrows)
    print(f"\nsaved: {OUT}/rank_audit_curves.csv, rank_audit_runs.csv")
    table()


def table():
    p = os.path.join(OUT, "rank_audit_runs.csv")
    R = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["mode"] == "sum"]
    if not R:
        return
    print("\n" + "=" * 100)
    print(f"optimistic rank vs tie-aware rank  (mode=sum, seed mean, SR@{SR_Q})")
    print("=" * 100)
    print(f"{'arch':<9}{'N':>5}{'seeds':>5}{'Q*max tie':>11}{'Q*max opt':>11}"
          f"{'Q*mean tie':>12}{'Q*mean opt':>12}{'SR tie':>9}{'SR opt':>9}")
    print("-" * 100)
    f = lambda v: float(v) if v not in ("", None) else None
    mean = lambda vs: f"{np.mean([v for v in vs if v is not None]):.2f}" \
        if any(v is not None for v in vs) else "-"
    for n in sorted({int(r["n_pairs"]) for r in R}):
        rs = [r for r in R if int(r["n_pairs"]) == n]
        print(f"{rs[0]['arch']:<9}{n:>5}{len(rs):>5}"
              f"{mean([f(r['q_star_max_tie']) for r in rs]):>11}"
              f"{mean([f(r['q_star_max_opt']) for r in rs]):>11}"
              f"{mean([f(r['q_star_mean_tie']) for r in rs]):>12}"
              f"{mean([f(r['q_star_mean_opt']) for r in rs]):>12}"
              f"{mean([f(r[f'sr_q{SR_Q}_tie']) for r in rs]):>9}"
              f"{mean([f(r[f'sr_q{SR_Q}_opt']) for r in rs]):>9}")
        cz = sum(int(r["n_censored_tie"]) for r in rs), sum(int(r["n_censored_opt"]) for r in rs)
        print(f"{'':<9}{'':>5}{'':>5}  (pairs not reaching GE=1: tie {cz[0]}, opt {cz[1]} / "
              f"{sum(int(r['n_pairs_eval']) for r in rs)} pairs)")
    print("=" * 100)
    print("Mean number of tied candidates vs Q (all pairs·all repeats·all seeds)")
    print(f"  {'N':>5}" + "".join(f"{'Q='+str(q):>14}" for q in TIE_Q))
    for n in sorted({int(r["n_pairs"]) for r in R}):
        rs = [r for r in R if int(r["n_pairs"]) == n]
        print(f"  {n:>5}" + "".join(
            f"{np.mean([f(r[f'n_tied_q{q}']) for r in rs]):>14,.0f}" for q in TIE_Q))
    print(f"\n  Relative to a candidate space of {E.Q*E.Q:,}. Ties arise because candidates giving the same HW receive")
    print("  bit-identical scores. The optimistic rank sends all of them below the true candidate and inflates GE/SR.")


if __name__ == "__main__":
    if SUMMARY_ONLY:
        summary()
    else:
        main()
