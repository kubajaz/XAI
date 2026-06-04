#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/checkpoints

for bs in 256 512; do
  for dim in 64 128; do
    python train.py \
      --batch-size "$bs" \
      --embed-dim "$dim" \
      --lr 1e-3 \
      --weight-decay 1e-5 \
      --num-neighbors 15 10 \
      --wandb-run-name "bs${bs}_dim${dim}" \
      --checkpoint "outputs/checkpoints/bs${bs}_dim${dim}.pth"
  done
done
