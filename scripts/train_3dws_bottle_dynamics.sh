#!/usr/bin/env bash
# Train action-conditioned 3DWS dynamics from the globally staged bottle latents.
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 3DWS_ROOT STAGE_ROOT LATENT_ROOT VAE_CHECKPOINT OUTPUT_DIR GPU[,GPU] [extra args...]" >&2
  exit 2
fi

THREEDWS_ROOT=$(cd "$1" && pwd)
STAGE_ROOT=$(cd "$2" && pwd)
LATENT_ROOT=$(cd "$3" && pwd)
VAE_CHECKPOINT=$4
OUTPUT_DIR=$5
GPU_LIST=$6
shift 6

[[ -f "$LATENT_ROOT/latent_manifest.json" ]] || { echo "missing latent manifest" >&2; exit 1; }
[[ -f "$VAE_CHECKPOINT" ]] || { echo "missing VAE checkpoint" >&2; exit 1; }
cd "$THREEDWS_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU_LIST
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}

GPU_COUNT=$(tr ',' '\n' <<<"$GPU_LIST" | wc -l | tr -d ' ')
if [[ $GPU_COUNT -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node "$GPU_COUNT" -m scripts.train_dynamics)
else
  LAUNCH=(python -m scripts.train_dynamics)
fi

AR_MAX=${ROBODOJOSIM_3DWS_AR_MAX:-15}
AR_STAGE_EPOCHS=${ROBODOJOSIM_3DWS_AR_STAGE_EPOCHS:-50}
N_HISTORY=${ROBODOJOSIM_3DWS_HISTORY:-8}
mapfile -t AR_SCHEDULE < <(seq 1 "$AR_MAX")
mapfile -t AR_EPOCHS < <(seq "$AR_STAGE_EPOCHS" "$AR_STAGE_EPOCHS" "$((AR_MAX * AR_STAGE_EPOCHS))")

CMD=("${LAUNCH[@]}"
  --latent-dir "$LATENT_ROOT/train"
  --val-latent-dir "$LATENT_ROOT/val"
  --vae-ckpt "$VAE_CHECKPOINT"
  --out-dir "$OUTPUT_DIR"
  --n-history "$N_HISTORY" --n-future 1
  --ar-schedule "${AR_SCHEDULE[@]}"
  --ar-schedule-epochs "${AR_EPOCHS[@]}"
  --batch-size 4 --num-workers 6 --iters-per-epoch 500 --val-iters 40
  --object-alpha 20 --obj-band-vox 0 --discount 0.95
  --max-hist-noise 0.4 --infer-hist-noise 0.1
  --cmp-horizons 10 30 75 150 --cmp-interval 50 --cmp-n-eps 10
  --cmp-infer-noises 0 0.075 0.15 --gap-every 50
  --lr 8e-5 --min-lr 8e-5 --dim 56 --n-levels 2 --film
  --bf16 --compile --save-interval 10 --vis-interval 30 --render-eps 2
  --packed-dirs "$STAGE_ROOT/packed/train" "$STAGE_ROOT/packed/val"
  --no-wandb)

if [[ ${ROBODOJOSIM_3DWS_WANDB:-0} == 1 ]]; then
  CMD+=(--wandb --wandb-project 3dws --wandb-name robodojo-two-color-bottle-dynamics)
fi
if [[ ${ROBODOJOSIM_3DWS_SMOKE:-0} == 1 ]]; then
  CMD+=(--ar-schedule 1 2 --ar-schedule-epochs 1 2 --iters-per-epoch 3
        --val-iters 1 --num-workers 0 --cmp-interval 1 --cmp-n-eps 2
        --gap-every 1 --gap-n-eps 2 --vis-interval 1 --save-interval 1)
fi
if [[ -f "$OUTPUT_DIR/checkpoints/latest.pth" ]]; then
  CMD+=(--resume "$OUTPUT_DIR/checkpoints/latest.pth")
fi
CMD+=("$@")

mkdir -p "$OUTPUT_DIR"
printf '%q ' "${CMD[@]}" > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"
cp "$LATENT_ROOT/latent_manifest.json" "$OUTPUT_DIR/latent_manifest.json"
if [[ ${ROBODOJOSIM_3DWS_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
exec "${CMD[@]}"
