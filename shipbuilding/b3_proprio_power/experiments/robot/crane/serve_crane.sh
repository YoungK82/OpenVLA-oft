#!/usr/bin/env bash
# Serve a magnet crane checkpoint for eval_server_on_rlds.py (FiLM on, x/y/z/load/power proprio, 3 exo cameras).
# Usage: CUDA_VISIBLE_DEVICES=0 bash experiments/robot/crane/serve_crane.sh /path/to/checkpoint   (VLA_PORT, VLA_HOST)
# The checkpoint must hold merged model weights (merge_lora_weights_and_save.py) plus its vision_backbone--*.pt.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 CHECKPOINT_DIRECTORY" >&2
  exit 2
fi
checkpoint_dir="$(cd -- "$1" && pwd -P)"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
# The action un-normalization key is the training dataset name, the only key in dataset_statistics.json
unnorm_key="$(python -c 'import json, sys; keys = list(json.load(open(sys.argv[1]))); print(keys[0] if len(keys) == 1 else "")' "$checkpoint_dir/dataset_statistics.json")"
if [[ -z "$unnorm_key" ]]; then
  echo "$checkpoint_dir/dataset_statistics.json does not hold exactly one dataset." >&2
  exit 2
fi
cd -- "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled
exec python vla-scripts/deploy.py \
  --robot_platform crane --pretrained_checkpoint "$checkpoint_dir" \
  --use_l1_regression True --use_diffusion False \
  --use_film True --use_proprio True --num_images_in_input 3 --center_crop True \
  --unnorm_key "$unnorm_key" \
  --host "${VLA_HOST:-127.0.0.1}" --port "${VLA_PORT:-8777}"
