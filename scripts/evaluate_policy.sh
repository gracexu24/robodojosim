#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 6 ]]; then
    echo "Usage: $0 XPOLICYLAB CHECKPOINT OUTPUT [EPISODES] [POLICY_GPU] [ENV_GPU]" >&2
    exit 2
fi
XPOLICYLAB=$1
CHECKPOINT=$2
OUTPUT=$3
EPISODES=${4:-10}
POLICY_GPU=${5:-0}
ENV_GPU=${6:-0}

conda run -n RoboDojo robodojosim evaluate-policy \
    --xpolicylab "${XPOLICYLAB}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUTPUT}" \
    --episodes "${EPISODES}" \
    --policy-gpu "${POLICY_GPU}" \
    --env-gpu "${ENV_GPU}"
