#!/usr/bin/env bash
# Run RoFT on the UniMTS 18-dataset few-shot protocol.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_PATH=${DATA_PATH:-./UniMTS_data}
CKPT=${CKPT:-./checkpoint/UniMTS.pth}
OUT=${OUT:-./logs/bench}
KS=${KS:-"1 2 3 5 10"}
SEEDS=${SEEDS:-"0 1 2"}

mkdir -p "$OUT" checkpoint

for k in $KS; do
  for s in $SEEDS; do
    python tools/fewshot_bench.py \
      --method unimts_ft \
      --tag ft_small_ens \
      --aug small --aug_angle 30 --ens 1 \
      --k "$k" --seed "$s" \
      --epochs 200 --batch_size 64 --lr 1e-4 \
      --data_path "$DATA_PATH" \
      --checkpoint "$CKPT" \
      --out_dir "$OUT"
  done
done
