#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
env_gpu_id=${8}
policy_conda_env=${9}
eval_env_conda_env=${10}

if [[ "${bench_name}" != "RoboDojo" || "${task_name}" != "put_bottles_into_dustbin" ]]; then
    echo "bottle_scripted supports only RoboDojo/put_bottles_into_dustbin" >&2
    exit 2
fi
if [[ "${action_type}" != "ee" ]]; then
    echo "bottle_scripted requires action_type=ee" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"
policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

bash "${SERVER_SCRIPT}" "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${policy_gpu_id}" "${policy_conda_env}" "${policy_server_port}" &
SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200
bash "${CLIENT_SCRIPT}" "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" \
    "${action_type}" "${seed}" "${env_gpu_id}" "${eval_env_conda_env}" "${additional_info}" \
    "${policy_server_port}" "${policy_server_ip}"
