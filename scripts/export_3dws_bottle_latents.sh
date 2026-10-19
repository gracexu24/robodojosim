#!/usr/bin/env bash
# Export each original root independently (avoids repeated episode-ID
# collisions), then stage the latent files under the global IDs in the packed
# training manifest.
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 3DWS_ROOT STAGE_ROOT VAE_CHECKPOINT RAW_LATENT_ROOT STAGED_LATENT_ROOT GPU" >&2
  exit 2
fi

THREEDWS_ROOT=$(cd "$1" && pwd)
STAGE_ROOT=$(cd "$2" && pwd)
VAE_CHECKPOINT=$3
RAW_LATENT_ROOT=$4
STAGED_LATENT_ROOT=$5
GPU=$6
MANIFEST=$STAGE_ROOT/training_manifest.json
ROBODOJOSIM_BIN=${ROBODOJOSIM_BIN:-robodojosim}

[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST" >&2; exit 1; }
[[ -f "$VAE_CHECKPOINT" ]] || { echo "missing VAE checkpoint $VAE_CHECKPOINT" >&2; exit 1; }
mkdir -p "$RAW_LATENT_ROOT" "$STAGED_LATENT_ROOT"

cd "$THREEDWS_ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
N_ENCODINGS=${ROBODOJOSIM_3DWS_ENCODINGS:-1}
N_SUBSAMPLE=${ROBODOJOSIM_3DWS_SUBSAMPLE:-150000}

mapfile -t ROOTS < <(python - "$MANIFEST" <<'PY'
import json
import sys
for root in json.load(open(sys.argv[1], encoding="utf-8"))["roots"]:
    print(f'{root["label"]}\t{root["path"]}')
PY
)

for entry in "${ROOTS[@]}"; do
  IFS=$'\t' read -r label packed_root <<<"$entry"
  for split in train val; do
    source_split=$split
    if [[ $split == val && -d "$packed_root/validation" ]]; then
      source_split=validation
    fi
    output=$RAW_LATENT_ROOT/$label/$split
    mkdir -p "$output"
    CMD=(python -m scripts.export_latents
      --dataset robodojo_two_color_bottles
      --vae-ckpt "$VAE_CHECKPOINT"
      --packed-dir "$packed_root/$source_split"
      --out-dir "$output"
      --action cartesian
      --n-encodings "$N_ENCODINGS"
      --n-subsample "$N_SUBSAMPLE")
    printf '+ '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    if [[ ${ROBODOJOSIM_3DWS_DRY_RUN:-0} != 1 ]]; then
      "${CMD[@]}"
    fi
  done
done

if [[ ${ROBODOJOSIM_3DWS_DRY_RUN:-0} != 1 ]]; then
  "$ROBODOJOSIM_BIN" stage-3dws-latents \
    --manifest "$MANIFEST" \
    --source-root "$RAW_LATENT_ROOT" \
    --output "$STAGED_LATENT_ROOT"
fi
