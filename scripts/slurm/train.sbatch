#!/bin/bash
#SBATCH --job-name=hetionet-train
#SBATCH --output=outputs/slurm/train_%j.out
#SBATCH --error=outputs/slurm/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

# Dostosuj do klastra:
# module load cuda/12.1
# source venv/bin/activate

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"
export DATA_PROCESSED="${DATA_PROCESSED:-$(pwd)/data/processed}"

CKPT="${SLURM_TMPDIR:-/tmp}/model_${SLURM_JOB_ID}.pth"
mkdir -p outputs/slurm

python train.py \
  --batch-size "${BATCH_SIZE:-256}" \
  --embed-dim "${EMBED_DIM:-64}" \
  --lr "${LR:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-5}" \
  --num-neighbors ${NUM_NEIGHBORS:-15 10} \
  --wandb-run-name "slurm-${SLURM_JOB_ID}" \
  --checkpoint "$CKPT"

# Opcjonalnie: kopiuj wynik do repozytorium
if [[ "${COPY_CHECKPOINT:-1}" == "1" ]]; then
  cp "$CKPT" "$(pwd)/model.pth"
fi
