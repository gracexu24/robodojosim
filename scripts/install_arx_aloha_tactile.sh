#!/bin/bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
    echo "Usage: $0 [original-RoboDojo] [hybrid-RoboDojo]" >&2
    exit 2
fi

ORIGINAL="${1:-/home/gracexu/robodojo-work/RoboDojo}"
HYBRID="${2:-/home/gracexu/robodojo-work/RoboDojo-arx-aloha}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OVERLAY_SOURCE="${PROJECT_ROOT}/integration/robodojo/arx_aloha"

if [[ ! -d "${ORIGINAL}/.git" && ! -f "${ORIGINAL}/.git" ]]; then
    echo "Original RoboDojo checkout not found: ${ORIGINAL}" >&2
    exit 2
fi
if [[ ! -f "${OVERLAY_SOURCE}/Assets/Robots/x5_aloha/X5A_Aloha.urdf" ]]; then
    echo "Hybrid overlay is incomplete: ${OVERLAY_SOURCE}" >&2
    exit 2
fi

if [[ ! -e "${HYBRID}" ]]; then
    git -C "${ORIGINAL}" worktree add --detach "${HYBRID}" "$(git -C "${ORIGINAL}" rev-parse HEAD)"

    # Reproduce the already validated local RoboDojo integration changes in
    # the copy, without altering or cleaning the original dirty checkout.
    while IFS= read -r relative; do
        [[ -f "${ORIGINAL}/${relative}" ]] || continue
        mkdir -p "${HYBRID}/$(dirname "${relative}")"
        cp -p "${ORIGINAL}/${relative}" "${HYBRID}/${relative}"
    done < <(git -C "${ORIGINAL}" diff --name-only --diff-filter=ACMRTUXB)
fi

if [[ ! -f "${HYBRID}/.git" && ! -d "${HYBRID}/.git" ]]; then
    echo "Hybrid target exists but is not a git worktree: ${HYBRID}" >&2
    exit 2
fi

# Reuse large pinned submodules read-only. The hybrid source files and assets
# themselves remain separate and versioned.
for relative in third_party/IsaacLab third_party/curobo; do
    destination="${HYBRID}/${relative}"
    source_path="${ORIGINAL}/${relative}"
    if [[ -L "${destination}" ]]; then
        continue
    fi
    if [[ -d "${destination}" ]] && [[ -z "$(find "${destination}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        rmdir "${destination}"
    fi
    if [[ ! -e "${destination}" ]]; then
        mkdir -p "$(dirname "${destination}")"
        ln -s "${source_path}" "${destination}"
    fi
done

# Keep policy integration independently editable in the hybrid. Earlier
# installer versions linked the entire XPolicyLab tree to the original; retain
# that link as provenance and materialize a copy before applying overlays.
if [[ -L "${HYBRID}/XPolicyLab" ]]; then
    if [[ ! -e "${HYBRID}/XPolicyLab.upstream" && ! -L "${HYBRID}/XPolicyLab.upstream" ]]; then
        mv "${HYBRID}/XPolicyLab" "${HYBRID}/XPolicyLab.upstream"
    else
        echo "Cannot replace ${HYBRID}/XPolicyLab; XPolicyLab.upstream already exists" >&2
        exit 2
    fi
fi
if [[ ! -d "${HYBRID}/XPolicyLab" ]]; then
    cp -a "${ORIGINAL}/XPolicyLab" "${HYBRID}/XPolicyLab"
fi
mkdir -p "${HYBRID}/XPolicyLab/policy/bottle_scripted"
cp "${PROJECT_ROOT}/integration/xpolicylab/policy/bottle_scripted/deploy.py" \
    "${HYBRID}/XPolicyLab/policy/bottle_scripted/deploy.py"

ASSET_OVERLAY="${HYBRID}/.cache/robodojo_assets_arx_aloha/Assets"
mkdir -p "${ASSET_OVERLAY}"

for source_path in "${ORIGINAL}/Assets"/*; do
    name="$(basename "${source_path}")"
    [[ "${name}" == "Robots" || "${name}" == "Eval_Layout" ]] && continue
    [[ -e "${ASSET_OVERLAY}/${name}" || -L "${ASSET_OVERLAY}/${name}" ]] || \
        ln -s "${source_path}" "${ASSET_OVERLAY}/${name}"
done

mkdir -p "${ASSET_OVERLAY}/Robots"
for source_path in "${ORIGINAL}/Assets/Robots"/*; do
    name="$(basename "${source_path}")"
    [[ "${name}" == "x5_aloha" ]] && continue
    [[ -e "${ASSET_OVERLAY}/Robots/${name}" || -L "${ASSET_OVERLAY}/Robots/${name}" ]] || \
        ln -s "${source_path}" "${ASSET_OVERLAY}/Robots/${name}"
done
if [[ ! -d "${ASSET_OVERLAY}/Robots/x5_aloha" ]]; then
    cp -a "${ORIGINAL}/Assets/Robots/x5" "${ASSET_OVERLAY}/Robots/x5_aloha"
fi
cp -a "${OVERLAY_SOURCE}/Assets/Robots/x5_aloha/." "${ASSET_OVERLAY}/Robots/x5_aloha/"

mkdir -p "${ASSET_OVERLAY}/Eval_Layout"
for source_path in "${ORIGINAL}/Assets/Eval_Layout"/*; do
    name="$(basename "${source_path}")"
    [[ "${name}" == "RoboDojo" ]] && continue
    [[ -e "${ASSET_OVERLAY}/Eval_Layout/${name}" || -L "${ASSET_OVERLAY}/Eval_Layout/${name}" ]] || \
        ln -s "${source_path}" "${ASSET_OVERLAY}/Eval_Layout/${name}"
done
mkdir -p "${ASSET_OVERLAY}/Eval_Layout/RoboDojo"
for source_path in "${ORIGINAL}/Assets/Eval_Layout/RoboDojo"/*; do
    name="$(basename "${source_path}")"
    [[ "${name}" == "arx_x5_aloha" ]] && continue
    [[ -e "${ASSET_OVERLAY}/Eval_Layout/RoboDojo/${name}" || -L "${ASSET_OVERLAY}/Eval_Layout/RoboDojo/${name}" ]] || \
        ln -s "${source_path}" "${ASSET_OVERLAY}/Eval_Layout/RoboDojo/${name}"
done
HYBRID_LAYOUT="${ASSET_OVERLAY}/Eval_Layout/RoboDojo/arx_x5_aloha"
if [[ -L "${HYBRID_LAYOUT}" ]]; then
    # Early development versions aliased the upstream layouts. Preserve that
    # link for provenance, but work on an independent copy so layout/mass
    # calibration can never mutate the original pipeline.
    [[ -e "${HYBRID_LAYOUT}.upstream" || -L "${HYBRID_LAYOUT}.upstream" ]] || \
        mv "${HYBRID_LAYOUT}" "${HYBRID_LAYOUT}.upstream"
fi
if [[ ! -d "${HYBRID_LAYOUT}" ]]; then
    cp -a "${ORIGINAL}/Assets/Eval_Layout/RoboDojo/arx_x5" "${HYBRID_LAYOUT}"
fi

# Preserve the worktree's upstream Assets link as a recoverable reference.
if [[ -L "${HYBRID}/Assets" ]] && [[ "$(readlink "${HYBRID}/Assets")" != "${ASSET_OVERLAY}" ]]; then
    if [[ ! -e "${HYBRID}/Assets.upstream" && ! -L "${HYBRID}/Assets.upstream" ]]; then
        mv "${HYBRID}/Assets" "${HYBRID}/Assets.upstream"
    else
        echo "Cannot replace ${HYBRID}/Assets; Assets.upstream already exists" >&2
        exit 2
    fi
fi
[[ -e "${HYBRID}/Assets" || -L "${HYBRID}/Assets" ]] || ln -s "${ASSET_OVERLAY}" "${HYBRID}/Assets"

cp "${OVERLAY_SOURCE}/env/robot_manager/robot_manager.py" \
    "${HYBRID}/env/robot_manager/robot_manager.py"
cp "${OVERLAY_SOURCE}/env/robot_manager/vt_refine_tactile.py" \
    "${HYBRID}/env/robot_manager/vt_refine_tactile.py"
cp "${OVERLAY_SOURCE}/EASYSIM_ENVS_LICENSE" \
    "${HYBRID}/EASYSIM_ENVS_LICENSE"
cp "${OVERLAY_SOURCE}/env/robot_manager/robot_class/x5_aloha.py" \
    "${HYBRID}/env/robot_manager/robot_class/x5_aloha.py"
cp "${OVERLAY_SOURCE}/env/robot_manager/robot_config/x5_aloha.py" \
    "${HYBRID}/env/robot_manager/robot_config/x5_aloha.py"
cp "${OVERLAY_SOURCE}/env/observation_manager/obs_manager.py" \
    "${HYBRID}/env/observation_manager/obs_manager.py"
cp "${OVERLAY_SOURCE}/env_cfg/robot/dual_x5_aloha.yml" \
    "${HYBRID}/env_cfg/robot/dual_x5_aloha.yml"
cp "${OVERLAY_SOURCE}/env_cfg/arx_x5_aloha.yml" \
    "${HYBRID}/env_cfg/arx_x5_aloha.yml"

# The hybrid retains the stock dual-X5 Cartesian action contract: each arm
# contributes six TCP dimensions and one gripper dimension. Register the
# copied config under its own name without changing the original entry.
ROBOT_INFO="${HYBRID}/env_cfg/robot/_robot_info.json"
python - "${ROBOT_INFO}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
data["dual_x5_aloha"] = dict(data["dual_x5"])
with open(path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, indent=4)
    handle.write("\n")
PY

USD="${ASSET_OVERLAY}/Robots/x5_aloha/ARX_ALOHA_calibrated.usd"
if [[ ! -s "${USD}" ]]; then
    source /home/gracexu/miniforge3/etc/profile.d/conda.sh
    conda activate "${ROBODOJO_CONDA_ENV:-RoboDojo}"
    "${HYBRID}/third_party/IsaacLab/isaaclab.sh" -p \
        "${HYBRID}/third_party/IsaacLab/scripts/tools/convert_urdf.py" \
        "${ASSET_OVERLAY}/Robots/x5_aloha/X5A_Aloha.urdf" \
        "${USD}" --fix-base --headless
fi

test -s "${USD}"
echo "Installed isolated ARX+ALOHA+tactile RoboDojo at ${HYBRID}"
echo "Original checkout was not modified: ${ORIGINAL}"
