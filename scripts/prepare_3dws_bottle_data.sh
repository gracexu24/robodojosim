#!/usr/bin/env bash
# Idempotently build imported-3DWS sidecars without changing packed episodes.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 3DWS_ROOT STAGE_ROOT [ROBOT_URDF] [extra prepare_robodojo args...]" >&2
  exit 2
fi
THREEDWS_ROOT=$(cd "$1" && pwd)
STAGE_ROOT=$(cd "$2" && pwd)
shift 2
ROBOT_URDF=/home/gracexu/robodojo-work/RoboDojo-arx-aloha-fixed-cameras/Assets/Robots/x5_aloha/X5A_Aloha.urdf
if [[ $# -gt 0 && "$1" != --* ]]; then ROBOT_URDF=$1; shift; fi
[[ -f "$STAGE_ROOT/training_manifest.json" ]] || { echo "missing training_manifest.json" >&2; exit 1; }
[[ -f "$ROBOT_URDF" ]] || { echo "missing robot URDF: $ROBOT_URDF" >&2; exit 1; }
cd "$THREEDWS_ROOT"
export PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
exec python -m scripts.prepare_robodojo \
  --data-root "$STAGE_ROOT" --dataset packed --robot-urdf "$ROBOT_URDF" "$@"
