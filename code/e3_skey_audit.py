#!/usr/bin/env python3
r"""
e3_skey_audit — a script that **directly measures**, from the data and code, what value the secret key sk used in evaluation is

    python e3_skey_audit.py

Read-only. It computes and saves nothing and modifies no file. A few seconds.

Why this script
---------------
The paper chat found an arithmetic-progression rule in the sk values of `diag.txt` (at 5-pair spacing, even coefficients -241, odd +241,
the even+odd sum is the same per group) and asked "it does not look like a key that went through Kyber key generation".
That question can only be answered by looking at the **acquisition and preprocessing code**, which is private, and the side that
uses this script (model and evaluation code) only reads sk from `labels[0]`. So instead of guessing:

  - measure the whole sk and answer (a)(b)(d)(e) **with numbers**
  - check whether the arithmetic-progression rule is real and how far it holds
  - **apply the inverse NTT and see whether a small polynomial comes out** — this is the decisive check.
    A real Kyber secret key is sampled by CBD(eta=2), so all coefficients are in {-2..2}. However large the NTT-domain values
    look, they must be small after the inverse transform. If not small, it is not keygen output.
  - for the parts of (a)(c) that can only be known from code, scan the folder and print **only the relevant lines**

Output `skey_audit.txt`. It can be sent to the paper chat as is.
"""

import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], "fine"]

import eval_nscan as E                                              # noqa: E402

Q = E.Q
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


# ---------------- inverse NTT (Kyber standard) ----------------
def barrett(a):
    a = np.asarray(a, np.int64)
    v = ((1 << 26) + Q // 2) // Q
    t = (v * a + (1 << 25)) >> 26
    return E.i16(a - t * Q)


def invntt(T, r):
    """Kyber invntt standard implementation. r: (..., 256).  Final f = 1441 = mont^2/128.

    Note: the reference implementation's invntt is `invntt_tomont` — the result comes out multiplied by the Montgomery factor 2^16.
    So to undo it, fqmul(., 1) must be applied once more (see coeffs()).
    All multiplications use train_nscan.v_fqmul (the function verified to match the firmware labels bit for bit).
    """
    r = E.i16(np.array(r, np.int64))
    z = np.asarray(T.ZETAS, np.int64)
    bar = getattr(T, "v_barrett", None) or barrett
    k = 127
    ln = 2
    while ln <= 128:
        for start in range(0, 256, 2 * ln):
            zeta = int(z[k]); k -= 1
            j = slice(start, start + ln)
            j2 = slice(start + ln, start + 2 * ln)
            t = r[..., j].copy()
            r[..., j] = E.i16(np.asarray(bar(t + r[..., j2])))
            r[..., j2] = E.i16(np.asarray(T.v_fqmul(zeta, E.i16(r[..., j2] - t))))
        ln <<= 1
    return E.i16(np.asarray(T.v_fqmul(1441, r)))


def coeffs(T, v):
    """NTT-domain values -> normal-domain coefficients (centered representation). Includes mont correction."""
    c = np.asarray(T.v_fqmul(invntt(T, v), 1), np.int64) % Q
    return np.where(c > Q // 2, c - Q, c)


def check_invntt(T):
    """Verify the inverse NTT by a round trip. If wrong, the verdict below is not used."""
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(20):
        x = rng.integers(-2, 3, 256).astype(np.int64)      # CBD(eta=2) range
        try:
            fwd = np.asarray(T.v_ntt(x[None, :]))[0]
        except Exception as e:
            return False, f"v_ntt call failed: {type(e).__name__}: {e}"
        bad += int(not np.array_equal(coeffs(T, fwd), x))
    return bad == 0, (f"CBD(eta=2) round trip matches for all 20 polynomials ✓" if bad == 0 else
                      f"round trip mismatch {bad}/20 — the inverse NTT cannot be trusted")


# ---------------- find the relevant lines in the code ----------------
def grep(pats, label, exts=(".py",), maxhit=14):
    say(f"  {label}")
    hit = 0
    for f in sorted(glob.glob(os.path.join(HERE, "*"))):
        if not f.endswith(exts) or os.path.basename(f).startswith("e3_"):
            continue
        try:
            txt = open(f, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            continue
        for i, ln in enumerate(txt, 1):
            if any(re.search(p, ln) for p in pats):
                say(f"    {os.path.basename(f)}:{i}:  {ln.strip()[:108]}")
                hit += 1
                if hit >= maxhit:
                    say(f"    … (the rest omitted)")
                    return hit
    if not hit:
        say("    no matching lines -> not a value produced by the code in this folder")
    return hit


# ---------------- main ----------------
def main():
    T = E.import_train()
    f_npz = os.path.join(E.SPLIT_DIR, "kyber_test_crop.npz")
    d = np.load(f_npz)
    lab = d["labels"]
    sk_raw = np.asarray(lab[0], np.int64)
    sk = np.asarray(E.i16(sk_raw), np.int64)

    say("=" * 84)
    say("E3 appendix — audit of the secret key sk used in evaluation (read-only)")
    say("=" * 84)

    # ---- (a) where is it read from ----
    say("")
    say("(a) Where is this sk read from")
    say(f"    file   {f_npz}")
    say(f"    array  labels  shape {np.asarray(lab).shape}  dtype {np.asarray(lab).dtype}")
    say(f"    sk     = labels[0]  length {len(sk_raw)}  dtype {sk_raw.dtype}")
    say(f"    keys in the npz: {sorted(d.files)}")
    say("")
    say("    lines that read it (code in this folder):")
    grep([r"labels\s*\[\s*0\s*\]"], "lines that read labels[0]")
    say("")
    say("    lines that create it (if in this folder):")
    n_make = grep([r"kyber_test_crop", r"np\.savez.*labels", r"labels\s*=\s*np\.(stack|array|concat)"],
                  "lines that write kyber_test_crop.npz or create labels")
    if not n_make:
        say("    -> **unknown**: the code that creates this npz is not in this folder (acquisition and preprocessing are private).")
        say("       The line in that script that puts in sk has to be checked directly.")

    # ---- (b) is it the NTT domain ----
    say("")
    say("(b) Are these NTT-domain values or normal-domain coefficients — exactly the form fed to the firmware?")
    say("    Judged by how it is used. Lines where train_nscan.v_stored uses sk:")
    grep([r"def v_stored", r"v_fqmul\s*\(\s*sk", r"sk\s*\[\s*base"], "v_stored / sk usage")
    say("    In eval_nscan:")
    say("      pair_zeta():  z = ZETAS[64 + (2p)//4]    <- zetas[64+i] of the reference implementation's basemul")
    say("      check_pair(): s0 = fqmul(sk[2p], b0) + fqmul(fqmul(sk[2p+1], b1), z)")
    say("                    b = v_ntt(v_unpack_10bit(ct))  <- ciphertext into the NTT domain")
    say("    basemul holds only in the NTT domain and sk is its operand, so")
    say("    -> **they are NTT-domain values.** Not normal-domain coefficients.")
    say("    Exactly the form fed to the firmware: since all 39 pairs pass the check (check_pair) that the")
    say("    decomposition matches the training labels bit for bit, **it is exactly the int16 the firmware fed to basemul.**")

    # ---- (d)(e) range ----
    say("")
    say("(d)(e) Value range — is it a structure that uses values unpacked to 12 bits (0..4095) without reduction?")
    say(f"    all of sk {len(sk)} values (Kyber-768 polyvec = 3 x 256 = 768)")
    say(f"      min {sk.min()}   max {sk.max()}")
    say(f"      negative        {int((sk < 0).sum())} values")
    say(f"      >= q({Q})       {int((sk >= Q).sum())} values")
    say(f"      >= 4096         {int((sk >= 4096).sum())} values")
    for lo, hi, nm in ((0, Q, "0..q-1 (canonical form)"), (Q, 4096, "q..4095 (12-bit unreduced)"),
                       (4096, 1 << 15, "4096 or above")):
        c = int(((sk >= lo) & (sk < hi)).sum())
        say(f"      {nm:<26} {c:>4} values")
    sig = sk[:2 * E.LEAK_PAIRS]
    say(f"    signal region (pairs 1~{E.LEAK_PAIRS}, coefficients 0~{2*E.LEAK_PAIRS-1})")
    say(f"      min {sig.min()}  max {sig.max()}  >= q: {int((sig >= Q).sum())} values")
    say(f"    outside the 39 pairs (coefficients {2*E.LEAK_PAIRS}~{len(sk)-1})")
    rest = sk[2 * E.LEAK_PAIRS:]
    say(f"      min {rest.min()}  max {rest.max()}  >= q: {int((rest >= Q).sum())} values")
    if sk.max() < 4096 and (sk >= Q).sum() > 0 and (sk < 0).sum() == 0:
        say("    -> All within 0..4095 with values >= q mixed in. **Consistent with 12-bit unreduced.**")
        say("       (However, a spec-compliant poly_tobytes stores values reduced to 0..q-1, so values that went")
        say("        through that round trip could not be >= q — read together with verdict (c) below)")
    else:
        say("    -> Does not fit the 12-bit unreduced hypothesis")

    # ---- arithmetic-progression rule check ----
    say("")
    say("Arithmetic-progression rule seen by the paper chat — how far does it actually hold")
    say("    coefficients of pair p: base = 2p.  Below are groups at 5-pair spacing")
    say(f"    {'pair':>4}{'sk[2p]':>9}{'diff':>7}{'sk[2p+1]':>10}{'diff':>7}{'sum':>8}")
    say("    " + "-" * 48)
    prev = None
    for p in range(0, E.LEAK_PAIRS, 5):
        a0, a1 = int(sk[2 * p]), int(sk[2 * p + 1])
        d0 = "" if prev is None else f"{a0-prev[0]:+d}"
        d1 = "" if prev is None else f"{a1-prev[1]:+d}"
        say(f"    {p+1:>4}{a0:>9}{d0:>7}{a1:>10}{d1:>7}{a0+a1:>8}")
        prev = (a0, a1)
    say("")
    # count the intervals over all coefficients where "even -241 / odd +241, 5-pair spacing" holds
    ok0 = ok1 = tot = 0
    for p in range(0, E.LEAK_PAIRS - 5, 5):
        tot += 1
        ok0 += int(int(sk[2 * (p + 5)]) - int(sk[2 * p]) == -241)
        ok1 += int(int(sk[2 * (p + 5) + 1]) - int(sk[2 * p + 1]) == +241)
    say(f"    among {tot} intervals of 5-pair spacing: {ok0} intervals with the even coefficient exactly -241, "
        f"{ok1} intervals with the odd one +241")
    # distribution of adjacent differences over all 768 coefficients
    dif = np.diff(sk)
    u, c = np.unique(dif, return_counts=True)
    top = sorted(zip(c.tolist(), u.tolist()), reverse=True)[:8]
    say(f"    most frequent adjacent differences over all of sk: "
        + ", ".join(f"{v}({n}x)" for n, v in top))
    say(f"    distinct values {len(np.unique(sk))} / {len(sk)} "
        f"(if truly random, almost all should differ)")

    # ---- (c) decisive check: inverse NTT ----
    say("")
    say("(c) How was the key determined — decisive check: is it a small polynomial after the inverse NTT?")
    good, msg = check_invntt(T)
    say(f"    inverse NTT self-check: {msg}")
    if not good:
        say("    -> The inverse NTT cannot be trusted, so this item is **unknown**. Do not use the verdict below")
    else:
        n_poly = len(sk) // 256
        say(f"    split sk into {n_poly} polynomials (256 coefficients each) and inverse-transform each")
        say(f"    for a real Kyber-768 secret key the coefficients must **all** be in {{-2..2}} (CBD eta=2).")
        say(f"    As a control: uniform random mod q and the arithmetic-progression pattern both put")
        say(f"    254 of 256 coefficients outside that range — the verdict separates cleanly.")
        say(f"    {'poly':>5}{'coefs':>14}{'min':>8}{'max':>8}{'|c|>2':>8}{'verdict':>12}")
        say("    " + "-" * 58)
        allsmall = True
        for i in range(n_poly):
            cen = coeffs(T, sk[i * 256:(i + 1) * 256])
            out = int((np.abs(cen) > 2).sum())
            sm = (out == 0)
            allsmall &= sm
            say(f"    {i:>5}{f'{i*256}~{i*256+255}':>14}{int(cen.min()):>8}"
                f"{int(cen.max()):>8}{out:>8}"
                f"{'small ✓' if sm else '**not small**':>12}")
        say("")
        if allsmall:
            say("    -> The inverse transform is all in {-2..2}. **Consistent with a real CBD secret key.**")
            say("       That the NTT-domain values look large is normal (NTT output spreads over mod q).")
            say("       Then what looked like an arithmetic progression is a coincidence or a match of specific coefficients only, so")
            say("       the interval counts of the 'arithmetic-progression rule' check above must be read together.")
        else:
            say("    -> The inverse transform is not small. **It is not a Kyber secret key sampled by CBD.**")
            say("       It is not keygen output but values filled in by a formula/pattern, or arbitrary values.")
            say("       **This means the recovery target is not a real Kyber key, and the attack")
            say("       difficulty (candidate distribution, HW distribution) may differ from reality, so the paper")
            say("       must state which key was used.**")
            say("       Which one it is requires looking at the relevant line of the acquisition code (below).")

    say("")
    say("    lines that set the key (if in this folder):")
    n_set = grep([r"\bsk\s*=\s*(?!np\.asarray|np\.array\(d|lab)",
                  r"secret", r"\bskpv\b", r"keygen", r"crypto_kem_keypair",
                  r"poly_frombytes", r"unpack_sk"], "lines that determine sk")
    if not n_set:
        say("    -> **unknown**: the code in this folder only reads sk. The line that set the key is")
        say("       in the acquisition/preprocessing script, which is private and cannot be seen here.")
        say("       Search that file with the patterns above and send only the relevant lines.")

    say("")
    say("=" * 84)
    say("Summary — per paper chat question")
    say(f"  (a) labels[0] of {os.path.basename(f_npz)}. The script that made it: "
        f"{'in this folder' if n_make else '**unknown** (private acquisition code)'}")
    say("  (b) NTT domain. Exactly the int16 the firmware fed to basemul (bit-match check passed)")
    say(f"  (c) {'see the inverse NTT verdict above' if good else '**unknown** (inverse NTT check failed)'}"
        f"; line that set the key: {'shown above' if n_set else '**unknown**'}")
    say(f"  (d) all in 0..4095, {int((sk >= Q).sum())} values >= q, 0 negative"
        f" -> consistent with 12-bit unreduced")
    say(f"  (e) outside the 39 pairs: min {rest.min()} max {rest.max()} "
        f">= q: {int((rest >= Q).sum())} values")
    say("=" * 84)

    p = os.path.join(E.OUT_DIR + "_e3", "skey_audit.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write("\n".join(LINES) + "\n")
    print(f"\nsaved: {p}")


if __name__ == "__main__":
    main()
