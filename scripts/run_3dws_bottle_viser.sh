#!/usr/bin/env bash
# Launch the two-bottle 3DWS Viser viewer on one GPU. The service binds to
# loopback by default and is intended to be reached through an SSH tunnel.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/data/gracexu/conda-envs/3dws/bin/python}
CACHE_ROOT=${CACHE_ROOT:-/data/gracexu/cache/3dws}
THREEDWS_ROOT=${THREEDWS_ROOT:-/data/gracexu/robodojo-work/3dws}
DYNAMICS_DIR=${DYNAMICS_DIR:-/data/gracexu/robodojo-work/3dws-bottles/models/dynamics-two-hours}
DYNAMICS_CHECKPOINT=${DYNAMICS_CHECKPOINT:-$DYNAMICS_DIR/checkpoints/best.pth}
LATENT_DIR=${LATENT_DIR:-/data/gracexu/robodojo-work/3dws-bottles/latents/staged-two-hours/val}
PACKED_DIR=${PACKED_DIR:-/data/gracexu/robodojo-work/3dws-bottles/stage-two-hours/packed/val}
VISER_SITE=${VISER_SITE:-$CACHE_ROOT/python/viser-site}
GPU=${GPU:-0}

mkdir -p "$CACHE_ROOT/torch" "$CACHE_ROOT/xdg" "$CACHE_ROOT/python"
[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$THREEDWS_ROOT" ]] || { echo "missing 3DWS root: $THREEDWS_ROOT" >&2; exit 1; }
[[ -f "$DYNAMICS_CHECKPOINT" ]] || { echo "missing checkpoint: $DYNAMICS_CHECKPOINT" >&2; exit 1; }
[[ -d "$LATENT_DIR" ]] || { echo "missing latent directory: $LATENT_DIR" >&2; exit 1; }
[[ -d "$VISER_SITE/viser" ]] || {
  echo "Viser cache is missing at $VISER_SITE; install with:" >&2
  echo "  $PYTHON_BIN -m pip install --target $VISER_SITE viser==1.1.1" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export PYTHONPATH="$VISER_SITE${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/serve_3dws_bottle_viser.py" \
  --threedws-root "$THREEDWS_ROOT" \
  --dynamics-dir "$DYNAMICS_DIR" \
  --dynamics-checkpoint "$DYNAMICS_CHECKPOINT" \
  --latent-dir "$LATENT_DIR" \
  --packed-dirs "$PACKED_DIR" \
  "$@"
