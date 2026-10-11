#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 /path/to/XPolicyLab [eval_conda_env] [policy_conda_env]" >&2
    exit 2
fi
XPL_ROOT="$(cd "$1" && pwd)"
if [[ ! -f "${XPL_ROOT}/setup_policy_server.py" ]]; then
    echo "Not an XPolicyLab checkout: ${XPL_ROOT}" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
for policy_name in bottle_scripted bottle_lerobot; do
    SOURCE="${PROJECT_ROOT}/integration/xpolicylab/policy/${policy_name}"
    DESTINATION="${XPL_ROOT}/policy/${policy_name}"
    if [[ -e "${DESTINATION}" ]]; then
        cp -R "${SOURCE}/." "${DESTINATION}/"
        echo "Updated adapter at ${DESTINATION}"
        continue
    fi
    cp -R "${SOURCE}" "${DESTINATION}"
    echo "Installed adapter at ${DESTINATION}"
done

if [[ $# -ge 2 ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda run -n "$2" python -m pip install -e "${PROJECT_ROOT}"
fi
if [[ $# -ge 3 && "$3" != "$2" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda run -n "$3" python -m pip install -e "${PROJECT_ROOT}"
fi
