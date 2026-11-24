# Global Pooling Can Limit Multi-Head Profiled Side-Channel Attacks on Kyber

Code, per-run result tables and training logs for the paper

> S.-W. Bae, "Global Pooling Can Limit Multi-Head Profiled Side-Channel Attacks on Kyber: Over-Provisioning, Under-Provisioning, and Position-Local Readout," submitted to *IEEE Transactions on Information Forensics and Security* (manuscript T-IFS-28706-2026).

**Traces and model weights** are archived separately on Zenodo: **DOI: [10.5281/zenodo.24476776](https://doi.org/10.5281/zenodo.24476776)**.
This repository and the Zenodo record do not overlap: text files (code, CSV, logs) are here, binary files (traces, normalization statistics, evaluation indices, `.pth` weights, `.npz` outputs) are on Zenodo.

The version that matches the submitted manuscript is the tag **`v1.0-submission`**.

---

## What the paper does, in one paragraph

A multi-head CNN reads an uncropped power trace of the pqm4 Kyber-768 implementation on an STM32F415 (ARM Cortex-M4) and predicts the Hamming weights of the stored BaseMul outputs of N coefficient pairs, N ∈ {9, 19, 29, 39, 49, 69, 99, 128}, while the capture window contains L = 39 leaking pairs. ResNet-18-1D and MobileNetV2-1D are trained ten times each under one protocol. The paper measures convergence and key recovery against N, shows with probes on frozen backbones that for ResNet at N = 39 and 49 the cost arises in the globally pooled readout, and reports two measurement effects (logarithmic depth grids and tie handling in rank computation).

## Conventions (read first)

- **Pair numbering.** CSV columns `pair_idx` and `slowest_pair_idx` are **zero-based**. The paper and the column `pair` are **one-based**. *Pair 32 in the paper is `pair_idx = 31`.*
- **Data sets.** 100,000 traces split 64,000 / 16,000 / 20,000 (train / validation / test). All attack and probe metrics use a fixed set of **2,000 test traces** (`eval_indices_*.npz` on Zenodo). Training-time accuracies (`val_acc_*`, `acc_allheads` in `all_runs.csv`) are on the 16,000 validation traces.
- **Probe training.** Probe heads on frozen backbones are trained on the first 16,000 traces of the training split and validated on the first 4,000 of the validation split.
- **Information (bits).** Mean of log p(y|x) − log prior(y) over the 2,000 evaluation traces, divided by ln 2; the prior is the label frequency of the 20,000 test traces with add-one smoothing.
- **Key-recovery grids.** *Coarse*: Q ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500}, 10 orderings. *Fine*: every Q from 1 to 50, 20 orderings. Q\* is the smallest Q at which the guessing entropy averaged over orderings reaches 1. Ranks are tie-aware: rank = (#higher) + 1 + (#tied)/2.
- **`ge_allheads`** in the training tables is a training-time monitor over all heads, not the guessing entropy reported in the paper.
- **Key.** The fixed key is a test pattern (bytes 00, 01, 02, … loaded as the packed key), not an output of key generation; seven of the 78 coefficients in the window lie in [q, 2^12). `derived/e3_stored_repr/` evaluates their effect.
- **Run sources** in `all_runs.csv` (`source` column): `main` (N-scan, 160 runs), `D`, `D29` (slower schedule), `E`, `E2` (stall controls), `E6` (MobileNetV2 with linear bottleneck), `B` (phantom labels permuted), `S34`, `R5` (additional runs).
- Paths in CSV and log files are replaced by `<DATA>`, `<CODE>`, `<HOME>`. The scripts contain the absolute paths of the original machine as constants near the top; edit them to your layout.

## Layout

```
code/      37 Python scripts (training, evaluation, probes, controls, figures)
derived/   result tables (CSV) and verification notes (TXT)
logs/      console logs of the control runs
```

| Folder in `derived/` | Content | Grid / orderings |
|---|---|---|
| `all_runs.csv`, `all_gradnorm.csv`, `train_summary.csv`, `train_gradnorm.csv` | one row per training run; per-epoch gradient norms | — |
| `eval_coarse/` | key recovery, main scan (160 runs) | coarse / 10 |
| `eval_fine/` | key recovery, main scan | fine / 20 |
| `eval_ctrl/`, `eval_ctrl_d/` | key recovery, S34·R5 and D·D29 | coarse / 10 |
| `eval_fine_d/` | key recovery, D·D29 | fine / 20 |
| `controls/` | `ctrl_summary.csv`, per-experiment training tables, `train_curves.csv` (per-epoch validation accuracy parsed from the logs) | — |
| `nicv/` | NICV per sample and per pair, peak positions | — |
| `shift_probe/`, `probe_pooled/`, `probe_local/`, `probe_local_ext/` | position probes on frozen backbones | — |
| `local_recover/` | key recovery with original, local and raw-window heads (11 models) | fine / 20 |
| `e1_window_cnn/` | one network per pair trained on its window | fine / 20 |
| `e2_local_seeds/` | local readout over ten seeds, N = 29, 39, 49 | fine / 20 |
| `e3_stored_repr/` | key recovery with candidates in the stored representation | fine / 20 |
| `e5_readouts/` | re-fitted pooled head vs. one-position head (`probe_without_batchnorm/` is a stopped variant kept as a record) | fine / 20 |
| `rank_audit/` | tie statistics and both rank conventions | fine / 20 |

## Paper → files

| Paper | Files |
|---|---|
| Fig. 1 (TVLA) | same figure as the companion ESL paper; no data here |
| Fig. 2, L = 39 | `derived/nicv/` |
| Table I | `code/train_nscan.py`, `derived/all_runs.csv` |
| Table II, Fig. 3 | `derived/eval_coarse/eval_runs.csv` (`converged`, `pairs_recovered`, `acc_signal`), `derived/all_runs.csv`; 13/15 at N = 128 adds `derived/eval_ctrl/` |
| Table III, Fig. 4 | `derived/eval_fine/eval_runs.csv` |
| Figs. 5–8 (GE curves) | `derived/eval_coarse/eval_runs.csv` (`ge_log2_q*`) |
| Phantom heads, KL | `derived/eval_coarse/eval_heads.csv` |
| Stored-representation result (Sec. VI-B) | `derived/e3_stored_repr/` |
| Table IV, Fig. 9 | `derived/controls/`, `logs/controls/`, `derived/eval_ctrl*/`, `derived/eval_fine_d/` |
| Table V, Fig. 10 | `derived/shift_probe/`, `derived/probe_pooled/`, `derived/probe_local/` |
| Table VI | `derived/eval_coarse/eval_heads.csv` (original), `derived/e5_readouts/` (re-fitted, one position), `derived/probe_local/`, `derived/probe_local_ext/` (local) |
| Table VII | `derived/e2_local_seeds/` (ten-seed rows), `derived/local_recover/`, `derived/e5_readouts/recover/`, `derived/e1_window_cnn/recover/` |
| Fig. 11 | `derived/eval_coarse/` and `derived/eval_fine/` |
| Fig. 12 | `derived/rank_audit/` |
| Selection rule (Sec. VII-A) | computed from Tables II–III: C(N) = p_conv(N) / Q\*max(N) |

## Reproducing

Order of the main pipeline (each script documents its inputs and outputs at the top):

1. `train_nscan.py` — trains the 160 main runs (needs the traces from Zenodo)
2. `eval_nscan.py` (coarse) and `eval_qfine.py` (fine) — key recovery
3. `run_controls.py`, `d29_train.py`, `e6_mbv2_std.py` — controls; `eval_controls.py`, `d29_eval.py`, `d29_qfine.py`
4. `nicv_pairs.py`, `shift_probe.py`, `probe_heads.py`, `label_leak.py`, `probe_local.py`, `probe_local_ctrl.py`, `probe_local_ext.py`, `probe_local_recover.py`
5. `e1_window_cnn.py`, `e2_local_seeds.py`, `e3_firmware_repr.py`, `e5_readout_ps.py`, `rank_audit.py`
6. `merge_runs.py` — merges run tables; `make_all_figs.py` — all figures

GPU training is not bit-reproducible; the released weights (Zenodo) reproduce the reported numbers exactly through the evaluation scripts. Most scripts write a `verify.txt` that checks that a new code path reproduces earlier values.

## Not released

Acquisition scripts and the preprocessing script that merges and splits the captured chunks are tied to institutional infrastructure and are not released. The released traces are the output of that preprocessing.

## License

Code: MIT (see `LICENSE`). Traces, model weights and result tables: CC BY 4.0.

## Citation

Please cite the paper above. A BibTeX entry will be added on publication.
