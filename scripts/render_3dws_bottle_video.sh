#!/usr/bin/env bash
# Make a comparison video from one held-out episode using frozen 3DWS weights.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 3DWS_ROOT DYNAMICS_CHECKPOINT OUTPUT.mp4 GPU [extra video args...]" >&2
  exit 2
fi

THREEDWS_ROOT=$(cd "$1" && pwd)
DYNAMICS_CHECKPOINT=$2
OUTPUT=$3
GPU=$4
shift 4

[[ -f "$DYNAMICS_CHECKPOINT" ]] || { echo "missing checkpoint: $DYNAMICS_CHECKPOINT" >&2; exit 1; }
[[ ! -e "$OUTPUT" ]] || { echo "refusing to overwrite prior video: $OUTPUT" >&2; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/data/gracexu/conda-envs/3dws/bin/python}
CACHE_ROOT=${CACHE_ROOT:-/data/gracexu/cache/3dws}
mkdir -p "$CACHE_ROOT/torch" "$CACHE_ROOT/matplotlib"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
export TORCH_HOME="$CACHE_ROOT/torch" MPLCONFIGDIR="$CACHE_ROOT/matplotlib"

exec "$PYTHON_BIN" "$SCRIPT_DIR/render_3dws_bottle_video.py" \
  --threedws-root "$THREEDWS_ROOT" \
  --dynamics-checkpoint "$DYNAMICS_CHECKPOINT" \
  --output "$OUTPUT" \
  "$@"
