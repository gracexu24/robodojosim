#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 6 ]]; then
  echo "usage: $0 3DWS_ROOT STAGE_ROOT LATENT_ROOT VAE_CHECKPOINT OUTPUT_DIR GPU [extra args...]" >&2
  exit 2
fi
THREEDWS_ROOT=$(cd "$1" && pwd); STAGE_ROOT=$(cd "$2" && pwd); LATENT_ROOT=$(cd "$3" && pwd)
VAE_CHECKPOINT=$4; OUTPUT_DIR=$5; GPU=$6; shift 6
[[ "$GPU" != *,* ]] || { echo "this imported 3DWS trainer is single-process; select one GPU" >&2; exit 2; }
[[ -f "$LATENT_ROOT/latent_manifest.json" ]] || { echo "missing latent manifest" >&2; exit 1; }
[[ -f "$VAE_CHECKPOINT" ]] || { echo "missing VAE checkpoint" >&2; exit 1; }
cd "$THREEDWS_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
AR_MAX=${ROBODOJOSIM_3DWS_AR_MAX:-15}; STAGE_EPOCHS=${ROBODOJOSIM_3DWS_AR_STAGE_EPOCHS:-50}
mapfile -t AR < <(seq 1 "$AR_MAX"); mapfile -t ENDS < <(seq "$STAGE_EPOCHS" "$STAGE_EPOCHS" "$((AR_MAX * STAGE_EPOCHS))")
CMD=(python -m scripts.train_dynamics --latent-dir "$LATENT_ROOT/train" --val-latent-dir "$LATENT_ROOT/val"
  --out-dir "$OUTPUT_DIR" --n-history 8 --ar-schedule "${AR[@]}" --ar-schedule-epochs "${ENDS[@]}"
  --object-alpha 20 --tactile-alpha 5 --lr 8e-5 --batch-size 4 --iters 500 --val-iters 40 --num-workers 6 --save-interval 10)
if [[ ${ROBODOJOSIM_3DWS_SMOKE:-0} == 1 ]]; then
  CMD+=(--ar-schedule 1 2 --ar-schedule-epochs 1 2 --iters 3 --val-iters 1 --num-workers 0 --save-interval 1)
fi
[[ ! -f "$OUTPUT_DIR/checkpoints/latest.pth" ]] || CMD+=(--resume "$OUTPUT_DIR/checkpoints/latest.pth")
CMD+=("$@")
mkdir -p "$OUTPUT_DIR"
printf '%q ' "${CMD[@]}" > "$OUTPUT_DIR/command.txt"; printf '\n' >> "$OUTPUT_DIR/command.txt"
cp -n "$LATENT_ROOT/latent_manifest.json" "$OUTPUT_DIR/latent_manifest.json" || true
if [[ ${ROBODOJOSIM_3DWS_DRY_RUN:-0} == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
exec "${CMD[@]}"
