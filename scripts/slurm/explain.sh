#!/bin/bash
#SBATCH --job-name=xai-explain
#SBATCH --output=outputs/slurm/explain_%j.out
#SBATCH --error=outputs/slurm/explain_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# ---------------------------------------------------------------------------
# Run GNNExplainer for a trained model checkpoint.
#
# Usage examples:
#   # First positive CcSE pair (quick sanity check):
#   sbatch scripts/slurm/explain.sh
#
#   # First N positive CcSE pairs:
#   N_PAIRS=10 sbatch scripts/slurm/explain.sh
#
#   # Specific compound/side-effect pair:
#   COMPOUND=42 SIDE_EFFECT=7 sbatch scripts/slurm/explain.sh
#
#   # Custom checkpoint:
#   CHECKPOINT=/path/to/model.pth sbatch scripts/slurm/explain.sh
#
#   # Full custom run:
#   COMPOUND=42 SIDE_EFFECT=7 CHECKPOINT=model_v2.pth EPOCHS=200 TOP_K=20 \
#     sbatch scripts/slurm/explain.sh
#
# Overridable env vars:
#   CHECKPOINT   – path to .pth checkpoint       (default: model.pth in repo root)
#   COMPOUND     – compound node index            (default: empty → --example)
#   SIDE_EFFECT  – side-effect node index         (default: empty → --example)
#   N_PAIRS      – iterate over first N pairs     (default: empty → single pair)
#   SPLIT        – train / val / test             (default: test)
#   EPOCHS       – explainer optimisation epochs  (default: 100)
#   TOP_K        – top-k edges to visualise       (default: 15)
#   NO_WANDB     – set to 1 to disable W&B        (default: 0)
#   WANDB_PROJECT– W&B project name               (default: from src/paths.py)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO_ROOT"
source "${REPO_ROOT}/scripts/slurm/activate_env.sh"

# ── defaults ─────────────────────────────────────────────────────────────────
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/model.pth}"
COMPOUND="${COMPOUND:-}"
SIDE_EFFECT="${SIDE_EFFECT:-}"
N_PAIRS="${N_PAIRS:-}"
SPLIT="${SPLIT:-test}"
EPOCHS="${EPOCHS:-100}"
TOP_K="${TOP_K:-15}"
NO_WANDB="${NO_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

OUTPUT_DIR="${REPO_ROOT}/outputs/explanations"
mkdir -p "$OUTPUT_DIR" outputs/slurm

OUTPUT_FILE="${OUTPUT_DIR}/explanation_${SLURM_JOB_ID:-local}.png"

# ── W&B ──────────────────────────────────────────────────────────────────────
if [[ "$NO_WANDB" != "1" ]]; then
    export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY or pass NO_WANDB=1}"
fi

# ── build argument list ───────────────────────────────────────────────────────
ARGS=(
    --checkpoint "$CHECKPOINT"
    --split      "$SPLIT"
    --epochs     "$EPOCHS"
    --top-k      "$TOP_K"
    --output     "$OUTPUT_FILE"
)

if [[ -n "$N_PAIRS" ]]; then
    ARGS+=(--n-pairs "$N_PAIRS")
elif [[ -n "$COMPOUND" && -n "$SIDE_EFFECT" ]]; then
    ARGS+=(--compound "$COMPOUND" --side-effect "$SIDE_EFFECT")
else
    ARGS+=(--example)
fi

[[ "$NO_WANDB" == "1" ]] && ARGS+=(--no-wandb)
[[ -n "$WANDB_PROJECT" ]] && ARGS+=(--wandb-project "$WANDB_PROJECT")
ARGS+=(--wandb-run-name "slurm-${SLURM_JOB_ID:-local}")

# ── run ───────────────────────────────────────────────────────────────────────
echo "=== XAI Explain Job ==="
echo "  job-id      : ${SLURM_JOB_ID:-local}"
echo "  node        : $(hostname)"
echo "  checkpoint  : $CHECKPOINT"
echo "  compound    : ${COMPOUND:-<example>}"
echo "  side-effect : ${SIDE_EFFECT:-<example>}"
echo "  n-pairs     : ${N_PAIRS:-1}"
echo "  split       : $SPLIT"
echo "  epochs      : $EPOCHS"
echo "  top-k       : $TOP_K"
echo "  output      : $OUTPUT_FILE"
echo "  started at  : $(date)"
echo ""

python explain_gnn.py "${ARGS[@]}"

echo ""
echo "=== Done at $(date) ==="
