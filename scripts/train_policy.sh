#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
    echo "Usage: $0 act|diffusion DATASET OUTPUT [GPU_ID] [CONDA_ENV]" >&2
    exit 2
fi
POLICY=$1
DATASET=$2
OUTPUT=$3
GPU_ID=${4:-0}
ENV_NAME=${5:-RoboDojoLeRobotTrain}

conda run -n "${ENV_NAME}" robodojosim train \
    --policy "${POLICY}" \
    --dataset "${DATASET}" \
    --output "${OUTPUT}" \
    --gpu "${GPU_ID}"
