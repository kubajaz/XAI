#!/bin/bash
#SBATCH --job-name=hetionet-grid
#SBATCH --output=outputs/slurm/grid_%A_%a.out
#SBATCH --error=outputs/slurm/grid_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
source "$(dirname "$0")/activate_env.sh"

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
mkdir -p outputs/slurm outputs/checkpoints

# SLURM_ARRAY_TASK_ID → kombinacja hyperparametrów
case "${SLURM_ARRAY_TASK_ID}" in
  0) BS=256; DIM=64 ;;
  1) BS=512; DIM=64 ;;
  2) BS=256; DIM=128 ;;
  3) BS=512; DIM=128 ;;
  *) echo "Nieznany SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

python train.py \
  --batch-size "$BS" \
  --embed-dim "$DIM" \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --num-neighbors 15 10 \
  --wandb-run-name "grid-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
  --checkpoint "outputs/checkpoints/grid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pth"
