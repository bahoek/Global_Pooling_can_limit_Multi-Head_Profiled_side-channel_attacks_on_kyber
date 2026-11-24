E5 — version with the request 7-2 head (Dropout 0.4 + Linear, no BatchNorm). Run for only 5 models, then stopped.

This is not the main result. The main result is the BatchNorm version:
    ../probe_e5_bn/          (probe, logp)
    ../../eval_qfine_e5_bn/  (key recovery, readout_models.csv, summary_7_4.txt)

Why this folder is kept: evidence that P (global average) does not train with a head without BatchNorm.
  N39 s0 P 27.86% / 0.07 bit (not converged at 60 epochs)  vs  BN version P 60.77% / 2.22 bit
S is 97% with or without BN (no BN 97.00 / 2.95, BN 97.03 / 3.19).

Decision: 2026-11-21, author — BatchNorm heads for both P and S (same structure as the original head train_nscan._head).
