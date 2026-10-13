#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/RoboDojo" >&2
    exit 2
fi

ROBODOJO_ROOT="$(cd "$1" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE="${PROJECT_ROOT}/integration/robodojo/tasks/put_bottles_into_dustbin.py"
DESTINATION="${ROBODOJO_ROOT}/task/RoboDojo/tasks/put_bottles_into_dustbin.py"
BACKUP="${DESTINATION}.robodojosim-four-bottle-original"

if [[ ! -f "${DESTINATION}" ]]; then
    echo "RoboDojo bottle task not found: ${DESTINATION}" >&2
    exit 2
fi
if [[ ! -f "${BACKUP}" ]]; then
    cp -p "${DESTINATION}" "${BACKUP}"
fi
cp "${SOURCE}" "${DESTINATION}"
echo "Installed two-bottle reward overlay at ${DESTINATION}"
