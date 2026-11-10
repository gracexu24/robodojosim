#!/bin/bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: $0 [original-RoboDojo] [fixed-camera-RoboDojo]" >&2
    exit 2
fi

ORIGINAL="${1:-/home/gracexu/robodojo-work/RoboDojo}"
TARGET="${2:-/home/gracexu/robodojo-work/RoboDojo-arx-aloha-fixed-cameras}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OVERLAY="${PROJECT_ROOT}/integration/robodojo/arx_aloha_fixed_cameras"

# Build the third setup from the already isolated tactile-arm recipe. The
# target is its own git worktree and asset overlay; neither earlier setup is
# edited by this installer.
"${SCRIPT_DIR}/install_arx_aloha_tactile.sh" "${ORIGINAL}" "${TARGET}"

mkdir -p "${TARGET}/env_cfg/camera"
cp "${OVERLAY}/env_cfg/camera/camera_config_fixed.yml" \
    "${TARGET}/env_cfg/camera/camera_config_fixed.yml"
cp "${OVERLAY}/env_cfg/arx_x5_aloha_fixed_cameras.yml" \
    "${TARGET}/env_cfg/arx_x5_aloha_fixed_cameras.yml"
cp "${PROJECT_ROOT}/integration/xpolicylab/get_free_port_dualstack.sh" \
    "${TARGET}/XPolicyLab/utils/get_free_port.sh"
chmod +x "${TARGET}/XPolicyLab/utils/get_free_port.sh"

# RoboDojo indexes layouts by config_name. The arm geometry and workspace are
# unchanged, but the new config still needs its own independently editable
# layout directory instead of falling back to either earlier setup.
LAYOUT_ROOT="${TARGET}/Assets/Eval_Layout/RoboDojo"
if [[ ! -d "${LAYOUT_ROOT}/arx_x5_aloha_fixed_cameras" ]]; then
    cp -a "${LAYOUT_ROOT}/arx_x5_aloha" \
        "${LAYOUT_ROOT}/arx_x5_aloha_fixed_cameras"
fi

# Keep a recoverable copy of the pre-centered task layouts, then place the bin
# on the exact table centerline (the table center is [0.0, -0.05]). The two
# robot roots are already symmetric at x = +/-0.30, so their calibrated base
# geometry does not need to move.
FIXED_LAYOUT="${LAYOUT_ROOT}/arx_x5_aloha_fixed_cameras"
FIXED_LAYOUT_BACKUP="${LAYOUT_ROOT}/arx_x5_aloha_fixed_cameras.before_table_center"
if [[ ! -e "${FIXED_LAYOUT_BACKUP}" ]]; then
    cp -a "${FIXED_LAYOUT}" "${FIXED_LAYOUT_BACKUP}"
fi
python - "${FIXED_LAYOUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in root.rglob("put_bottles_into_dustbin_*.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for dustbin in data.get("Geometry", {}).get("dustbin", []):
        if dustbin.get("category_idx") != 0 or dustbin.get("label") != "dustbin":
            continue
        position = list(dustbin.get("default_pos", [0.0, -0.05, 0.8625]))
        position[0] = 0.0
        position[1] = -0.05
        dustbin["default_pos"] = position
        dustbin["xlim"] = [0.0, 0.0]
        dustbin["ylim"] = [-0.05, -0.05]
        changed = True
    if changed:
        path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
PY

# Wrist cameras are declared in the embodiment asset metadata and injected by
# RobotManager. Remove only that declaration in this third asset overlay; the
# arm, fingers, tactile meshes, controller, and 14D action interface remain.
python - "${TARGET}/Assets/Robots/x5_aloha/robot_config.yml" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8"))
data.pop("camera", None)
path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
PY

echo "Installed third ARX+ALOHA+tactile setup with three fixed cameras at ${TARGET}"
echo "Preserved original setup: ${ORIGINAL}"
echo "Preserved tactile wrist-camera setup: /home/gracexu/robodojo-work/RoboDojo-arx-aloha"
