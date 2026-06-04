#!/usr/bin/env bash
#SBATCH --job-name=hetionet-grid
#SBATCH --output=outputs/slurm/grid_%j.out
#SBATCH --error=outputs/slurm/grid_%j.err
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Slurm copies the script to /var/spool/slurmd/... — use SLURM_SUBMIT_DIR, not BASH_SOURCE.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs/slurm

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/scripts/slurm/activate_env.sh"
  export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
  export GRID_PREFIX="${GRID_PREFIX:-${SLURM_JOB_ID}}"
  CKPT_DIR="${SCRATCH_BASE}/xai/outputs/checkpoints"
else
  CKPT_DIR="${REPO_ROOT}/outputs/checkpoints"
fi
mkdir -p "$CKPT_DIR"

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
            ckpt="${CKPT_DIR}/grid_${GRID_PREFIX}_${RUN}_${tag}.pth"
            wandb_name="grid-${GRID_PREFIX}_${RUN}_${tag}"
          else
            ckpt="${CKPT_DIR}/${tag}.pth"
            wandb_name="$tag"
          fi
          echo "=== [${RUN}/${N_GRID}] ${tag} ==="
          echo "checkpoint: $ckpt"
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

echo "[train_grid] finished ${N_GRID} runs; checkpoints in ${CKPT_DIR}"
