#!/usr/bin/env bash
# Render and score a frozen 3DWS dynamics checkpoint on held-out validation episodes.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 3DWS_ROOT DYNAMICS_CHECKPOINT OUTPUT_DIR GPU [extra evaluator args...]" >&2
  exit 2
fi

THREEDWS_ROOT=$(cd "$1" && pwd)
DYNAMICS_CHECKPOINT=$2
OUTPUT_DIR=$3
GPU=$4
shift 4

[[ -f "$DYNAMICS_CHECKPOINT" ]] || { echo "missing dynamics checkpoint: $DYNAMICS_CHECKPOINT" >&2; exit 1; }
if [[ -e "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to mix with existing artifacts: $OUTPUT_DIR" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/data/gracexu/conda-envs/3dws/bin/python}
CACHE_ROOT=${CACHE_ROOT:-/data/gracexu/cache/3dws}
mkdir -p "$CACHE_ROOT/torch" "$CACHE_ROOT/matplotlib"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export TORCH_HOME="$CACHE_ROOT/torch"
export MPLCONFIGDIR="$CACHE_ROOT/matplotlib"

exec "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_3dws_bottle_dynamics.py" \
  --threedws-root "$THREEDWS_ROOT" \
  --dynamics-checkpoint "$DYNAMICS_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
