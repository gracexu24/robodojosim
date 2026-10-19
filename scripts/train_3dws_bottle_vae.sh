#!/usr/bin/env bash
# Train the plain particle VAE on a staged RoboDojo two-bottle packed_v2 corpus.
# The caller must activate the 3DWS Python environment first. No ALOHA FK,
# robot-label cache, tactile stream, SAM3, or FoundationStereo sidecars are used.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 3DWS_ROOT STAGE_ROOT OUTPUT_DIR GPU[,GPU] [extra train_vae args...]" >&2
  exit 2
fi

THREEDWS_ROOT=$(cd "$1" && pwd)
STAGE_ROOT=$(cd "$2" && pwd)
OUTPUT_DIR=$3
GPU_LIST=$4
shift 4

[[ -f "$STAGE_ROOT/training_manifest.json" ]] || {
  echo "missing $STAGE_ROOT/training_manifest.json; run robodojosim stage-3dws-training first" >&2
  exit 1
}
[[ -d "$STAGE_ROOT/packed/train" && -d "$STAGE_ROOT/packed/val" ]] || {
  echo "staged train/val packed directories are missing" >&2
  exit 1
}

cd "$THREEDWS_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU_LIST
export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export NUMPY_MADVISE_HUGEPAGE=${NUMPY_MADVISE_HUGEPAGE:-0}

GPU_COUNT=$(tr ',' '\n' <<<"$GPU_LIST" | wc -l | tr -d ' ')
if [[ $GPU_COUNT -gt 1 ]]; then
  LAUNCH=(torchrun --standalone --nproc_per_node "$GPU_COUNT" -m scripts.train_vae)
else
  LAUNCH=(python -m scripts.train_vae)
fi

# This is an intentionally conservative baseline. The 32x20x20 latent grid
# keeps the first run and its latent cache tractable. Raise the grid only after
# the smoke-test reconstructions and an on-disk latent pilot pass.
NX=${ROBODOJOSIM_3DWS_NX:-32}
NY=${ROBODOJOSIM_3DWS_NY:-20}
NZ=${ROBODOJOSIM_3DWS_NZ:-20}
N_POINTS=${ROBODOJOSIM_3DWS_N_POINTS:-300000}
MIN_POINTS=${ROBODOJOSIM_3DWS_MIN_POINTS:-150000}
BG_BANK=${ROBODOJOSIM_3DWS_BG_BANK:-250000}

CMD=("${LAUNCH[@]}"
  --dataset robodojo_two_color_bottles
  --train-dir "$STAGE_ROOT/packed/train"
  --val-dir "$STAGE_ROOT/packed/val"
  --out-dir "$OUTPUT_DIR"
  --nx "$NX" --ny "$NY" --nz "$NZ"
  --x-range -0.85 0.75 --y-range -0.55 0.40 --z-range 0.30 1.35
  --latent-channels 8 --latent-split 4 --color-branch-rgb-only
  --encoder-channels 8 16 32 64 --encoder-residual --encoder-feat-skip
  --decoder-channels 256 128 64 --decoder-residual
  --batch-size 1 --num-workers 4 --lr 2e-4 --min-lr 1e-5
  --latent-noise 0.03 --latent-dropout 0 --channel-dropout 0 --kl-weight 0
  --n-points "$N_POINTS" --min-points "$MIN_POINTS" --max-points "$N_POINTS"
  --gaussian-aniso --fg-scale-max 0.04 --sh-degree 1
  --consistency-weight 100 --consistency-point-frac 0.5
  --object-alpha 20 --rgb-object-alpha 20 --seed-chamfer-object-alpha 20
  --render-weight 4 --render-ssim-weight 0.5 --render-lpips-weight 1 --lpips-net vgg
  --render-object-alpha 20 --render-obj-dilate 6 --render-band-valid-only --render-downscale 2
  --depth-carve-weight 2 --depth-carve-margin 0.03 --render-depth-weight 2
  --bg-bank "$BG_BANK" --bg-init-eps 16 --bg-near-grid 0.004 --bg-far-grid 0.012 --bg-lr 2e-3
  --wrist-cam-ids 1 2 --wrist-render-start-epoch 0
  --n-epochs 300 --phase-a-epochs 10 --phase-b-epochs 10 --iters-per-epoch 500
  --val-interval 10 --val-iters 20 --save-interval 10 --vis-interval 10 --render-vis-interval 5
  --detailed-metrics --percep-bf16 --no-r1-lpips --no-wandb)

if [[ ${ROBODOJOSIM_3DWS_WANDB:-0} == 1 ]]; then
  CMD+=(--wandb --wandb-project 3dws --wandb-name robodojo-two-color-bottle-vae)
fi
if [[ ${ROBODOJOSIM_3DWS_SMOKE:-0} == 1 ]]; then
  CMD+=(--n-epochs 1 --phase-a-epochs 0 --phase-b-epochs 0 --iters-per-epoch 3
        --val-interval 1 --val-iters 2 --warmup-epochs 0 --num-workers 0
        --vis-n-frames 1 --bg-init-eps 1)
fi
if [[ -f "$OUTPUT_DIR/checkpoints/latest.pth" ]]; then
  CMD+=(--resume "$OUTPUT_DIR/checkpoints/latest.pth")
fi
CMD+=("$@")

mkdir -p "$OUTPUT_DIR"
printf '%q ' "${CMD[@]}" > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"
cp "$STAGE_ROOT/training_manifest.json" "$OUTPUT_DIR/training_manifest.json"
if [[ ${ROBODOJOSIM_3DWS_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
exec "${CMD[@]}"
