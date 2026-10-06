#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/RoboDojo" >&2
    exit 2
fi

ROBODOJO_ROOT="$(cd "$1" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${ROBODOJO_ROOT}/scripts/eval_policy.sh"
PATCH="${PROJECT_ROOT}/patches/robodojo_gui_toggle.patch"

if [[ ! -f "${TARGET}" ]]; then
    echo "RoboDojo launcher not found: ${TARGET}" >&2
    exit 1
fi

if grep -q 'ROBODOJO_HEADLESS' "${TARGET}"; then
    echo "RoboDojo GUI toggle is already installed: ${TARGET}"
    exit 0
fi

git -C "${ROBODOJO_ROOT}" apply --check "${PATCH}"
git -C "${ROBODOJO_ROOT}" apply "${PATCH}"
echo "Installed GUI toggle. Set ROBODOJO_HEADLESS=0 to show the Isaac Sim window."
