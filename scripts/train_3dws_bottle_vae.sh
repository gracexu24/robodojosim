#!/usr/bin/env bash
# Train imported 3DWS on staged RoboDojo packed episodes.
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "usage: $0 3DWS_ROOT STAGE_ROOT OUTPUT_DIR GPU [extra train_vae args...]" >&2
  exit 2
fi
THREEDWS_ROOT=$(cd "$1" && pwd)
STAGE_ROOT=$(cd "$2" && pwd)
OUTPUT_DIR=$3
GPU=$4
shift 4
[[ "$GPU" != *,* ]] || { echo "this imported 3DWS trainer is single-process; select one GPU" >&2; exit 2; }
[[ -f "$STAGE_ROOT/packed/preprocess_manifest.json" ]] || {
  echo "missing imported-3DWS sidecars; run scripts/prepare_3dws_bottle_data.sh first" >&2
  exit 1
}
cd "$THREEDWS_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
CMD=(python -m scripts.train_vae
  --task robodojo_bottles --data-root "$STAGE_ROOT" --datasets packed
  --out-dir "$OUTPUT_DIR" --n-epochs 300 --iters 500 --val-iters 20
  --val-interval 10 --save-interval 10 --phase-a 10 --phase-b 10
  --wrist-start 0 --num-workers 4)
if [[ ${ROBODOJOSIM_3DWS_SMOKE:-0} == 1 ]]; then
  CMD+=(--n-epochs 1 --iters 3 --val-iters 1 --val-interval 1 --save-interval 1 --phase-a 0 --phase-b 0 --warmup 0 --num-workers 0 --bg-init-eps 1)
fi
[[ ! -f "$OUTPUT_DIR/checkpoints/latest.pth" ]] || CMD+=(--resume "$OUTPUT_DIR/checkpoints/latest.pth")
CMD+=("$@")
mkdir -p "$OUTPUT_DIR"
printf '%q ' "${CMD[@]}" > "$OUTPUT_DIR/command.txt"; printf '\n' >> "$OUTPUT_DIR/command.txt"
cp -n "$STAGE_ROOT/training_manifest.json" "$OUTPUT_DIR/training_manifest.json" || true
if [[ ${ROBODOJOSIM_3DWS_DRY_RUN:-0} == 1 ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
exec "${CMD[@]}"
