#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Download Hetionet raw TSVs and build processed .npz/.pkl on scratch.
#
# Run once on the login node (after setup_env.sh):
#   bash scripts/slurm/download_data.sh
#
# Optional:
#   DATA_DIR=/other/scratch/path bash scripts/slurm/download_data.sh
#   bash scripts/slurm/download_data.sh --sample 100000   # quick smoke test
#
# Output layout (default DATA_DIR=$SCRATCH_BASE/data/hetionet):
#   $DATA_DIR/raw/hetionet-v1.0-nodes.tsv
#   $DATA_DIR/raw/hetionet-v1.0-edges.sif.gz
#   $DATA_DIR/processed/hetionet_v1_baseline.npz
#   $DATA_DIR/processed/hetionet_v1_baseline_maps.pkl
#
# Slurm jobs read processed data via DATA_PROCESSED (set in activate_env.sh).
# ============================================================

log() { echo "[download_data] $*"; }
die() { echo "[download_data][ERROR] $*" >&2; exit 1; }

_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_SLURM_DIR}/../.." && pwd)"

log "Repo: $REPO_ROOT"
source "${_SLURM_DIR}/activate_env.sh"

export DATA_DIR="${DATA_DIR:-${SCRATCH_BASE}/data/hetionet}"
export DATA_PROCESSED="${DATA_PROCESSED:-${DATA_DIR}/processed}"

log "SCRATCH_BASE   : $SCRATCH_BASE"
log "DATA_DIR       : $DATA_DIR"
log "DATA_PROCESSED : $DATA_PROCESSED"

mkdir -p "${DATA_DIR}/raw" "${DATA_PROCESSED}"

log "Running prepare_hetionet.py (download + preprocess; may take a while)"
python "${REPO_ROOT}/scripts/prepare_hetionet.py" --data-dir "$DATA_DIR" "$@"

for f in hetionet_v1_baseline.npz hetionet_v1_baseline_maps.pkl; do
  [[ -f "${DATA_PROCESSED}/${f}" ]] || die "Missing ${DATA_PROCESSED}/${f}"
done

log "Done."
log "Processed data: $DATA_PROCESSED"
log "Slurm jobs use DATA_PROCESSED automatically after: source scripts/slurm/activate_env.sh"
