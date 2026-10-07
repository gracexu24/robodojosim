#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/RoboDojo" >&2
    exit 2
fi

ROBODOJO_ROOT="$(cd "$1" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROBODOJO_ROOT}/scripts/eval_policy.sh"
LAUNCHER_PATCH="${PROJECT_ROOT}/patches/robodojo_fast_calibration.patch"
OBS_TARGET="${ROBODOJO_ROOT}/env/observation_manager/obs_manager.py"
OBS_PATCH="${PROJECT_ROOT}/patches/robodojo_state_only_observations.patch"

if [[ ! -f "${TARGET}" ]]; then
    echo "RoboDojo launcher not found: ${TARGET}" >&2
    exit 1
fi

if [[ ! -f "${OBS_TARGET}" ]]; then
    echo "RoboDojo observation manager not found: ${OBS_TARGET}" >&2
    exit 1
fi

if ! grep -q 'ROBODOJOSIM_CALIBRATION_FAST' "${TARGET}"; then
    git -C "${ROBODOJO_ROOT}" apply --check "${LAUNCHER_PATCH}"
    git -C "${ROBODOJO_ROOT}" apply "${LAUNCHER_PATCH}"
fi
if ! grep -q 'ROBODOJOSIM_CALIBRATION_FAST' "${OBS_TARGET}"; then
    git -C "${ROBODOJO_ROOT}" apply --check "${OBS_PATCH}"
    git -C "${ROBODOJO_ROOT}" apply "${OBS_PATCH}"
fi
echo "Installed fast calibration toggle. Set ROBODOJOSIM_CALIBRATION_FAST=1 for state-only checks."
