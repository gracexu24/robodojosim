#!/bin/bash
set -euo pipefail

ENV_NAME=${1:-RoboDojoLeRobotTrain}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" python=3.11
fi
conda run -n "${ENV_NAME}" python -m pip install -e "${PROJECT_ROOT}[train]"
conda run -n "${ENV_NAME}" python -c \
    'import lerobot, torch; print(f"LeRobot {lerobot.__version__}; Torch {torch.__version__}; CUDA {torch.cuda.is_available()}")'
