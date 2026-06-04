#!/bin/bash
#SBATCH --job-name=hetionet-optuna
#SBATCH --output=outputs/slurm/optuna_%A_%a.out
#SBATCH --error=outputs/slurm/optuna_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
source "${REPO_ROOT}/scripts/slurm/activate_env.sh"

export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY}"

# SQLite na współdzielonym FS bywa problematyczny — na produkcji ustaw PostgreSQL:
# export OPTUNA_STORAGE="postgresql+psycopg2://user:pass@host/optuna"
export OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///$(pwd)/outputs/optuna/hetionet_ccse.db}"

mkdir -p outputs/slurm outputs/optuna outputs/checkpoints

python scripts/run_optuna_trial.py \
  --storage "$OPTUNA_STORAGE" \
  --study-name hetionet-ccse

# Po zakończeniu całego array (login node):
#   python scripts/run_optuna_trial.py --show-best --promote-best
