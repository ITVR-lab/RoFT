# RoFT Evaluation Protocol

This document is the public specification of the few-shot benchmark used in the RoFT paper.
Code entry points and reproduction commands are in the repository root [`README.md`](../README.md).

## 1. Assets

| Asset | Source | How to obtain |
|---|---|---|
| 18 HAR datasets + official few-shot indices | UniMTS release | [Hugging Face `xiyuanz/UniMTS`](https://huggingface.co/xiyuanz/UniMTS) or [Google Drive archive](https://drive.google.com/file/d/1ybD5Fx6c4ykJiDGLPQlLn0m77z9EkjLb/view?usp=sharing) |
| Pre-trained UniMTS weights | UniMTS release | `UniMTS.pth` from the same Hugging Face repo → `./checkpoint/UniMTS.pth` |
| RoFT training / evaluation code | this repo | `tools/fewshot_bench.py`, `scripts/run_roft.sh` |
| Table / figure builders | this repo | `tools/make_bench_tables.py`, `tools/visualization/` |
| LODO recipe selection | this repo | `tools/loo_select.py`, `protocol/loo_selection_grid.md` |
| Joint / placement mapping notes | this repo | `docs/JOINT_MAPPING.md`, `tools/audit_joint_mapping.py` |

Expected data layout after extraction:

```text
UniMTS_data/
  <Dataset>/ X_train.npy X_test.npy y_train.npy y_test.npy …
  few_shot_data_2/ <Dataset>_k=<k>.pth
```

We use the UniMTS-prepared windows (acceleration channels, resampled / padded to \(T=200\) at 20 Hz, mapped onto \(V=22\) joints). Sequences shorter than 200 steps are cyclically padded by the UniMTS loader; the tensor length is therefore fixed, but not every window is an independent 10-second observation.

## 2. Splits and support sets

- Train / test splits: **official UniMTS releases** for each of the 18 datasets.
- Few-shot support: indices in `few_shot_data_2/{dataset}_k={k}.pth` for \(k\in\{1,2,3,5,10\}\).
- All methods in a given \((k,\text{dataset})\) cell see **exactly the same support windows**.
- Evaluation uses the **complete test split**.
- Random seeds \(\{0,1,2\}\) change linear-head initialization, mini-batch order, and rotation sampling. **They do not resample the support set.** Reported means are therefore stability over training randomness under fixed supports.

Subject / session isolation follows the UniMTS public splits; we do not redefine them. Overlap audits relative to those released files can be run with `tools/audit_support_overlap.py` once `--data_path` is set.

## 3. Method tags used in logs

| Tag / flags | Meaning |
|---|---|
| `ft_small_ens` (`--aug small --aug_angle 30 --ens 1`) | **RoFT** (paper main result) |
| `ft_small` (`--aug small --ens 0`) | bounded rotation, no prior ensemble |
| `unimts_ft` / `ft_noaug` (`--aug none --ens 0`) | official UniMTS full fine-tuning |
| `ft_noaug_ens` (`--aug none --ens 1`) | FT + prior ensemble only |
| `ft_so3_ens` (`--aug so3 --ens 1`) | unconstrained \(\mathrm{SO}(3)\) ablation |
| `ft_a15_ens` / `ft_a45_ens` / `ft_a60_ens` | angle sensitivity (\(\theta_{\max}\in\{15^\circ,45^\circ,60^\circ\}\)) |
| `probe` | linear probe on frozen UniMTS features |
| `frozen` | zero-shot / 1-NN / ProtoNet / Tip-Adapter(+F) |

Optimization (all full fine-tuning runs unless noted): Adam, learning rate \(10^{-4}\), batch size 64, 200 epochs, keep the checkpoint with lowest **support-set training loss**. No target validation or test labels are used for hyper-parameter or architecture selection.

## 4. Leave-one-dataset-out (LODO) meta-selection

Selection metric: **mean accuracy** on the other 17 datasets (seed 0), for \(k\in\{1,5\}\).

Candidate grid (see `protocol/loo_selection_grid.md`):

- UniMTS FT ± ensemble
- FT + uniform \(\mathrm{SO}(3)\) ± ensemble
- FT + small-angle \(30^\circ\) ± ensemble (**RoFT** when ensembled)
- FT with 30 epochs ± ensemble
- TipGuided family variants listed in the grid file

Command:

```bash
python tools/loo_select.py \
  --family all=unimts_ft,ft_so3,ft_so3_ens,ft_small,ft_small_ens,ft_noaug_ens \
  --k 1 5 --seed 0 --metric acc
```

In the released development grid, `ft_small_ens` (RoFT) was selected for **18/18** held-out datasets at both \(k=1\) and \(k=5\). Angle tables that report macro-F1 are diagnostic; LODO used accuracy.

## 5. Statistics

Wilcoxon signed-rank tests pair seed-averaged scores of RoFT and each baseline on each dataset (18 pairs per \(k\)). Holm correction is applied over the pooled comparison family. Reproduction helpers:

```bash
python tools/make_bench_tables.py   # rebuilds Acc / F1 / R@2 tables from JSON logs
python tools/stats_wilcoxon.py      # prints per-k and Holm-corrected p-values
```

## 6. What this release is intended to make auditable

1. Exact support indices and method flags for every main-table cell.
2. LODO candidate list, selection metric, and selection counts.
3. Joint-mapping rules inherited from UniMTS (including UTD-MHAD placement and MotionSense duplication).
4. Scripts to rebuild tables/figures from logged JSON without re-training.

Pre-trained UniMTS weights and the 18-dataset archive remain hosted by the UniMTS authors; this repository pins the adaptation protocol and RoFT-specific code on top of those assets.
