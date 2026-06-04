#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p outputs/checkpoints

# Edytuj wartości tutaj (jedno źródło prawdy dla lokalnego i SLURM gridu)
DIM_VALUES=(64 256 512)
BS_VALUES=(128 256 )
LR_VALUES=(1e-2 1e-3 1e-4)
WD_VALUES=(1e-6 1e-5)
NEIGHBOR_PAIRS=("15 10" "20 15" "10 5")

N_GRID=$(( ${#BS_VALUES[@]} * ${#DIM_VALUES[@]} * ${#LR_VALUES[@]} \
         * ${#WD_VALUES[@]} * ${#NEIGHBOR_PAIRS[@]} ))
RUN=0

GRID_PREFIX="${GRID_PREFIX:-}"

for bs in "${BS_VALUES[@]}"; do
  for dim in "${DIM_VALUES[@]}"; do
    for lr in "${LR_VALUES[@]}"; do
      for wd in "${WD_VALUES[@]}"; do
        for nh in "${NEIGHBOR_PAIRS[@]}"; do
          read -r nh1 nh2 <<< "$nh"
          RUN=$((RUN + 1))
          tag="bs${bs}_dim${dim}_lr${lr}_wd${wd}_nh${nh1}-${nh2}"
          if [[ -n "$GRID_PREFIX" ]]; then
            ckpt="outputs/checkpoints/grid_${GRID_PREFIX}_${RUN}_${tag}.pth"
            wandb_name="grid-${GRID_PREFIX}_${RUN}_${tag}"
          else
            ckpt="outputs/checkpoints/${tag}.pth"
            wandb_name="$tag"
          fi
          echo "=== [${RUN}/${N_GRID}] ${tag} ==="
          python train.py \
            --batch-size "$bs" \
            --embed-dim "$dim" \
            --lr "$lr" \
            --weight-decay "$wd" \
            --num-neighbors "$nh1" "$nh2" \
            --wandb-run-name "$wandb_name" \
            --checkpoint "$ckpt"
        done
      done
    done
  done
done

echo "[train_grid] finished ${N_GRID} runs"
