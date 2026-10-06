#!/bin/bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 /path/to/XPolicyLab POLICY_SEED POLICY_CONDA_ENV ROBODOJO_CONDA_ENV OUTPUT_DIR" >&2
    exit 2
fi
XPL_ROOT="$(cd "$1" && pwd)"
POLICY_SEED=$2
POLICY_ENV=$3
EVAL_ENV=$4
OUTPUT_DIR="$(mkdir -p "$5" && cd "$5" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_GPU_ID="${ROBODOJOSIM_POLICY_GPU_ID:-0}"
ENV_GPU_ID="${ROBODOJOSIM_ENV_GPU_ID:-0}"

export ROBODOJOSIM_DATASET_DIR="${OUTPUT_DIR}"
export ROBODOJOSIM_CONFIG="${ROBODOJOSIM_CONFIG:-${PROJECT_ROOT}/configs/bottle_task.json}"
export EVAL_NUM="${ROBODOJOSIM_EVAL_NUM:-native}"

bash "${XPL_ROOT}/policy/bottle_scripted/eval.sh" \
    RoboDojo put_bottles_into_dustbin scripted arx_x5 ee "${POLICY_SEED}" \
    "${POLICY_GPU_ID}" "${ENV_GPU_ID}" "${POLICY_ENV}" "${EVAL_ENV}"
