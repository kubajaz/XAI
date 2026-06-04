#!/usr/bin/env bash
# Source this file — do not execute directly.
#   source scripts/slurm/activate_env.sh
#
# Loads the scratch conda env created by setup_env.sh.
# Override via env vars (same defaults as setup_env.sh):
#   SCRATCH_BASE, ENV_PREFIX, DATA_PROCESSED
#
# For setup_env.sh only: ACTIVATE_CONDA=0 source ...  (paths + caches, no conda)

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && {
  echo "[activate_env][ERROR] Source this file instead of running it:" >&2
  echo "  source $(dirname "$0")/activate_env.sh" >&2
  exit 1
}

_activate_die() { echo "[activate_env][ERROR] $*" >&2; exit 1; }

_USERNAME="$(whoami)"
[[ -n "$_USERNAME" ]] || _activate_die "whoami returned empty username"

export SCRATCH_BASE="${SCRATCH_BASE:-/net/tscratch/people/${_USERNAME}}"
export ENV_PREFIX="${ENV_PREFIX:-${SCRATCH_BASE}/conda/py311_env}"
export DATA_DIR="${DATA_DIR:-${SCRATCH_BASE}/data/hetionet}"
export DATA_PROCESSED="${DATA_PROCESSED:-${DATA_DIR}/processed}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_BASE/.cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRATCH_BASE/.cache/conda/pkgs}"
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/.tmp}"
export PYTHONNOUSERSITE=1

mkdir -p \
  "$(dirname "$ENV_PREFIX")" \
  "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"   "$CONDA_PKGS_DIRS" \
  "$TMPDIR"

if [[ "${ACTIVATE_CONDA:-1}" != "1" ]]; then
  return 0
fi

[[ -d "$ENV_PREFIX" ]] || _activate_die \
  "Conda env not found at $ENV_PREFIX. Run: bash scripts/slurm/setup_env.sh"

if command -v module >/dev/null 2>&1; then
  module load Miniconda3 || _activate_die "Failed to load Miniconda3 module"
fi
command -v conda >/dev/null 2>&1 || _activate_die "conda not found (load Miniconda3?)"
eval "$(conda shell.bash hook)"

conda activate "$ENV_PREFIX"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$ENV_PREFIX}"
