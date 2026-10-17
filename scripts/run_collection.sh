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
ROBODOJO_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
WORLD_MODEL_CONFIG_ENABLED=0

restore_world_model_observations() {
    if [[ "${WORLD_MODEL_CONFIG_ENABLED}" == "1" ]]; then
        conda run -n "${EVAL_ENV}" robodojosim configure-world-model-observations \
          --robodojo-root "${ROBODOJO_ROOT}" --restore >/dev/null || true
    fi
}
trap restore_world_model_observations EXIT

export ROBODOJOSIM_DATASET_DIR="${OUTPUT_DIR}"
export ROBODOJOSIM_CONFIG="${ROBODOJOSIM_CONFIG:-${PROJECT_ROOT}/configs/bottle_task.json}"
export EVAL_NUM="${ROBODOJOSIM_EVAL_NUM:-native}"

if [[ "${ROBODOJOSIM_TWO_BOTTLE_TASK:-1}" == "1" ]]; then
    bash "${PROJECT_ROOT}/scripts/enable_two_bottle_task.sh" "${ROBODOJO_ROOT}"
else
    echo "[DEPRECATED] ROBODOJOSIM_TWO_BOTTLE_TASK=0 restores the unsupported four-bottle/handoff workflow." >&2
    echo "[DEPRECATED] Use the default brown/yellow two-bottle task for new collection and training." >&2
fi

# Asset 22 declares 0.22 kg in metadata, but RoboDojo's generated layouts
# encode it as 22 kg. Correct only that exact typo, retaining one backup per
# affected layout, so a normal X5 gripper can manipulate the plastic bottle.
if [[ "${ROBODOJOSIM_FIX_BOTTLE_MASS:-1}" == "1" ]]; then
    LAYOUT_ARGS=(fix-layouts --robodojo-root "${ROBODOJO_ROOT}")
    if [[ "${ROBODOJOSIM_CENTRALIZE_DUSTBIN:-1}" == "1" ]]; then
        LAYOUT_ARGS+=(--centralize-dustbin)
    fi
    if [[ "${ROBODOJOSIM_TWO_BOTTLE_TASK:-1}" == "1" ]]; then
        LAYOUT_ARGS+=(--two-bottle-brown-yellow)
        LAYOUT_ARGS+=(--bottle-side-order "${ROBODOJOSIM_BOTTLE_SIDE_ORDER:-alternating}")
    fi
    conda run -n "${EVAL_ENV}" robodojosim "${LAYOUT_ARGS[@]}"
fi

# 3DWS needs metric depth and calibrated camera poses. Enable those expensive
# annotators only for world-model collection, then restore the normal RGB-only
# policy configuration through the EXIT trap above.
if [[ "${ROBODOJOSIM_COLLECTION_PROFILE:-manual}" == world_model* ]]; then
    conda run -n "${EVAL_ENV}" robodojosim configure-world-model-observations \
      --robodojo-root "${ROBODOJO_ROOT}"
    WORLD_MODEL_CONFIG_ENABLED=1
else
    # Also recover from a previous SIGKILL/power loss, where an EXIT trap could
    # not restore the normal policy configuration.
    conda run -n "${EVAL_ENV}" robodojosim configure-world-model-observations \
      --robodojo-root "${ROBODOJO_ROOT}" --restore >/dev/null
fi

bash "${XPL_ROOT}/policy/bottle_scripted/eval.sh" \
    RoboDojo put_bottles_into_dustbin scripted arx_x5 ee "${POLICY_SEED}" \
    "${POLICY_GPU_ID}" "${ENV_GPU_ID}" "${POLICY_ENV}" "${EVAL_ENV}"
