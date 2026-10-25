#!/usr/bin/env bash
# Evaluate the registered ACT and Diffusion Policy checkpoints inside the
# frozen two-bottle 3DWS world model, one policy at a time.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
THREEDWS_PYTHON=${THREEDWS_PYTHON:-/data/gracexu/conda-envs/3dws/bin/python}
POLICY_PYTHON=${POLICY_PYTHON:-/home/gracexu/miniforge3/envs/RoboDojo/bin/python}
POLICY_EXTRA_SITE_PACKAGES=${POLICY_EXTRA_SITE_PACKAGES:-/home/gracexu/miniforge3/envs/RoboDojoLeRobot/lib/python3.11/site-packages}
THREEDWS_ROOT=${THREEDWS_ROOT:-/data/gracexu/robodojo-work/3dws}
DYNAMICS_CHECKPOINT=${DYNAMICS_CHECKPOINT:-/data/gracexu/robodojo-work/3dws-bottles/models/dynamics-two-hours/checkpoints/best.pth}
LATENT_DIR=${LATENT_DIR:-/data/gracexu/robodojo-work/3dws-bottles/latents/staged-two-hours/val}
PACKED_DIR=${PACKED_DIR:-/data/gracexu/robodojo-work/3dws-bottles/stage-two-hours/packed/val}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/gracexu/robodojo-work/3dws-bottles/policy-evaluations/act-dp-$(date +%Y%m%d-%H%M%S)}
WORLD_GPU=${WORLD_GPU:-0}
POLICY_GPU=${POLICY_GPU:-1}
EPISODES=${EPISODES:-10}
STEPS=${STEPS:-16}

ACT_CHECKPOINT=${ACT_CHECKPOINT:-/home/gracexu/robodojo-work/models/bottle-act-two-color-50/checkpoints/last/pretrained_model}
DIFFUSION_CHECKPOINT=${DIFFUSION_CHECKPOINT:-/home/gracexu/robodojo-work/models/bottle-diffusion-two-color-50/checkpoints/last/pretrained_model}

mkdir -p "$OUTPUT_ROOT"
[[ -x "$THREEDWS_PYTHON" ]] || { echo "missing 3DWS Python: $THREEDWS_PYTHON" >&2; exit 1; }
[[ -x "$POLICY_PYTHON" ]] || { echo "missing policy Python: $POLICY_PYTHON" >&2; exit 1; }
[[ -f "$DYNAMICS_CHECKPOINT" ]] || { echo "missing dynamics checkpoint: $DYNAMICS_CHECKPOINT" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$WORLD_GPU"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export TORCH_HOME=/data/gracexu/cache/3dws/torch
export XDG_CACHE_HOME=/data/gracexu/cache/3dws/xdg

for entry in "act:$ACT_CHECKPOINT" "diffusion:$DIFFUSION_CHECKPOINT"; do
  family=${entry%%:*}
  checkpoint=${entry#*:}
  output="$OUTPUT_ROOT/$family"
  [[ -d "$checkpoint" ]] || { echo "missing policy checkpoint: $checkpoint" >&2; exit 1; }
  "$THREEDWS_PYTHON" "$SCRIPT_DIR/evaluate_policy_in_3dws.py" \
    --repo-root "$REPO_ROOT" \
    --threedws-root "$THREEDWS_ROOT" \
    --dynamics-checkpoint "$DYNAMICS_CHECKPOINT" \
    --latent-dir "$LATENT_DIR" \
    --packed-dir "$PACKED_DIR" \
    --policy-checkpoint "$checkpoint" \
    --policy-python "$POLICY_PYTHON" \
    --policy-extra-site-packages "$POLICY_EXTRA_SITE_PACKAGES" \
    --policy-gpu "$POLICY_GPU" \
    --episodes "$EPISODES" \
    --steps "$STEPS" \
    --bf16 \
    --output-dir "$output" | tee "$OUTPUT_ROOT/$family.log"
done

printf '%s\n' "$OUTPUT_ROOT"
