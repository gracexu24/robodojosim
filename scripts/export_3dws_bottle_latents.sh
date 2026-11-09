#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
  echo "usage: $0 3DWS_ROOT STAGE_ROOT VAE_CHECKPOINT RAW_LATENT_ROOT STAGED_LATENT_ROOT GPU" >&2
  exit 2
fi
THREEDWS_ROOT=$(cd "$1" && pwd); STAGE_ROOT=$(cd "$2" && pwd)
VAE_CHECKPOINT=$3; RAW_LATENT_ROOT=$4; STAGED_LATENT_ROOT=$5; GPU=$6
[[ -f "$VAE_CHECKPOINT" ]] || { echo "missing VAE checkpoint" >&2; exit 1; }
[[ -f "$STAGE_ROOT/packed/preprocess_manifest.json" ]] || { echo "missing preprocessing manifest" >&2; exit 1; }
mkdir -p "$RAW_LATENT_ROOT/train" "$RAW_LATENT_ROOT/val" "$STAGED_LATENT_ROOT"
cd "$THREEDWS_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
for split in train val; do
  CMD=(python -m scripts.export_latents --vae-ckpt "$VAE_CHECKPOINT" --packed-dir "$STAGE_ROOT/packed/$split" --out-dir "$RAW_LATENT_ROOT/$split")
  if [[ ${ROBODOJOSIM_3DWS_DRY_RUN:-0} == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; else "${CMD[@]}"; fi
done
if [[ ${ROBODOJOSIM_3DWS_DRY_RUN:-0} != 1 ]]; then
  python /home/gracexu/robodojo-work/robodojosim/scripts/stage_imported_3dws_latents.py \
    --raw "$RAW_LATENT_ROOT" --output "$STAGED_LATENT_ROOT" --training-manifest "$STAGE_ROOT/training_manifest.json"
fi
