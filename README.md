# RoFT: Rotation-Perturbed Few-Shot Fine-Tuning for Wearable HAR

Official code for **RoFT** — bounded per-joint rotation perturbation during few-shot fine-tuning of a pre-trained motion foundation model ([UniMTS](https://github.com/xiyuanzh/UniMTS)), plus equal-weight probability ensembling with the frozen zero-shot prior at inference.

This repository builds on UniMTS (NeurIPS 2024). We keep the backbone / data loaders and add the RoFT training recipe and the 18-dataset few-shot benchmark used in our paper.

## Reproducibility (protocol, data, checkpoints)

| Item | Location |
|---|---|
| Full evaluation protocol (splits, seeds, LODO, metrics) | [`docs/PROTOCOL.md`](docs/PROTOCOL.md) |
| Joint / placement mapping notes (UTD-MHAD, MotionSense) | [`docs/JOINT_MAPPING.md`](docs/JOINT_MAPPING.md) |
| LODO candidate grid + selection counts | [`protocol/loo_selection_grid.md`](protocol/loo_selection_grid.md) |
| 18 datasets + few-shot indices | [Hugging Face `xiyuanz/UniMTS`](https://huggingface.co/xiyuanz/UniMTS) · [Google Drive](https://drive.google.com/file/d/1ybD5Fx6c4ykJiDGLPQlLn0m77z9EkjLb/view?usp=sharing) |
| Pre-trained weights `UniMTS.pth` | same Hugging Face repo → `./checkpoint/UniMTS.pth` |
| Main RoFT sweep | `scripts/run_roft.sh` / `scripts/run_roft.ps1` |
| Audit support indices | `python tools/audit_support_overlap.py --data_path ...` |
| Audit joint mapping | `python tools/audit_joint_mapping.py --data_path ...` |
| LODO selection | `python tools/loo_select.py ...` |
| Wilcoxon + Holm | `python tools/stats_wilcoxon.py --bench_dir logs/bench` |

## Installation

```bash
conda create -n roft python=3.9
conda activate roft
pip install -r requirements.txt
```

PyTorch with CUDA should match your GPU driver. CLIP is installed from the OpenAI GitHub pin listed in `requirements.txt`.

## Data & checkpoint

1. Download UniMTS evaluation data (18 datasets + official few-shot indices) from [Hugging Face](https://huggingface.co/xiyuanz/UniMTS) or [Google Drive](https://drive.google.com/file/d/1ybD5Fx6c4ykJiDGLPQlLn0m77z9EkjLb/view?usp=sharing).
2. Download `UniMTS.pth` from the same Hugging Face repo into `./checkpoint/UniMTS.pth`.
3. Point `--data_path` at the extracted folder that contains dataset subdirs and `few_shot_data_2/`.

Expected layout (abbreviated):

```text
UniMTS_data/
  UCIHAR/ X_train.npy X_test.npy y_*.npy …
  … (18 datasets)
  few_shot_data_2/ UCIHAR_k=1.pth …
```

## Quick start — run RoFT

RoFT corresponds to `--method unimts_ft --aug small --ens 1` (tag `ft_small_ens` in our logs).

```bash
# one-shot, seed 0
python tools/fewshot_bench.py \
  --method unimts_ft --tag ft_small_ens \
  --aug small --aug_angle 30 --ens 1 \
  --k 1 --seed 0 --epochs 200 --batch_size 64 \
  --data_path /path/to/UniMTS_data \
  --checkpoint ./checkpoint/UniMTS.pth \
  --out_dir ./logs/bench
```

Sweep $k\in\{1,2,3,5,10\}$ and seeds $\{0,1,2\}$:

```bash
bash scripts/run_roft.sh
# Windows: .\scripts\run_roft.ps1
```

### Main baselines in the same entry point

| Command flags | Meaning |
|---|---|
| `--method unimts_ft --aug none --ens 0` | Official UniMTS full fine-tuning |
| `--method probe` | Linear probe |
| `--method frozen` | Zero-shot / 1-NN / prototype / Tip-Adapter(+F) |
| `--method unimts_ft --aug so3 --ens 1` | Full $\mathrm{SO}(3)$ ablation (+ ensemble) |
| `--method unimts_ft --aug small --ens 0` | Small-angle FT without prior ensemble |

Optional external backbones (`units`, `limu`, `onehar`, `biobank`, `gpt4ts`, `imagebind`) need extra checkpoints; see `tools/baselines/`.

## Reproduce paper tables & figures

After benchmarks finish, JSON summaries live under `--out_dir`. Then:

```bash
python tools/make_bench_tables.py
python tools/visualization/plot_fewshot_curves.py
python tools/visualization/plot_eff_acc.py
python tools/visualization/plot_roft_transform.py
```

## Citation

If you use this code, please cite UniMTS and our paper (BibTeX will be updated upon publication):

```bibtex
@inproceedings{zhang2024unimts,
  title     = {{UniMTS}: Unified Pre-training for Motion Time Series},
  author    = {Zhang, Xiyuan and others},
  booktitle = {NeurIPS},
  year      = {2024}
}

@article{roft2026,
  title   = {{RoFT}: Rotation-Perturbed Few-Shot Fine-Tuning for Wearable Human Activity Recognition},
  author  = {Anonymous},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  note    = {under review}
}
```

## Acknowledgements

Graph encoder, synthetic pre-training pipeline, and the 18-dataset protocol are from [UniMTS](https://github.com/xiyuanzh/UniMTS). We thank the authors for releasing models and data.
