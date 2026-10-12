#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 8 ]]; then
  echo "usage: $0 DATASET_ROOT CAMPAIGN_PID_FILE OUTPUT_DIR TRAIN_ENV [GPU] [STEPS] [BATCH_SIZE] [NUM_WORKERS]" >&2
  exit 2
fi

dataset_root=$(realpath "$1")
campaign_pid_file=$(realpath "$2")
output_dir=$(realpath -m "$3")
train_env=$(realpath "$4")
gpu=${5:-0}
steps=${6:-20000}
batch_size=${7:-8}
num_workers=${8:-4}
poll_seconds=${ROBODOJOSIM_WATCH_POLL_SECONDS:-30}

campaign_pid=$(tr -d '[:space:]' < "$campaign_pid_file")
if [[ ! "$campaign_pid" =~ ^[0-9]+$ ]]; then
  echo "invalid campaign PID in $campaign_pid_file" >&2
  exit 2
fi

echo "Waiting for collection campaign PID $campaign_pid to finish."
while kill -0 "$campaign_pid" 2>/dev/null; do
  sleep "$poll_seconds"
done

read -r campaign_status successful_episodes < <(
  "$train_env/bin/python" - "$dataset_root/campaign_state.json" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
print(state.get("status", "missing"), state.get("stats", {}).get("successful_episodes", 0))
PY
)
if [[ "$campaign_status" != "complete" || "$successful_episodes" -lt 50 ]]; then
  echo "collection did not complete successfully: status=$campaign_status successful_episodes=$successful_episodes" >&2
  exit 1
fi

lerobot_root="$dataset_root/lerobot"
"$train_env/bin/robodojosim" inspect-lerobot --dataset "$lerobot_root"
exec "$train_env/bin/robodojosim" train \
  --policy act \
  --dataset "$lerobot_root" \
  --output "$output_dir" \
  --gpu "$gpu" \
  --steps "$steps" \
  --batch-size "$batch_size" \
  --num-workers "$num_workers" \
  --executable "$train_env/bin/lerobot-train"
