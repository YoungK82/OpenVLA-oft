#!/usr/bin/env bash
# Run with: bash /path/to/repo/experiments/robot/so101/serve_portable.sh /path/to/checkpoint
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash $0 CHECKPOINT_DIRECTORY" >&2
  exit 2
fi
checkpoint_dir="$(cd -- "$1" && pwd -P)"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled
python - "$checkpoint_dir" <<'PY'
import json
import sys
from pathlib import Path
import torch

p = Path(sys.argv[1])
for name in ("config.json", "dataset_statistics.json", "model.safetensors.index.json", "preprocessor_config.json", "tokenizer_config.json"):
    if not (p / name).is_file():
        raise SystemExit(f"Missing checkpoint file: {p / name}")
index = json.loads((p / "model.safetensors.index.json").read_text())
for name in set(index["weight_map"].values()):
    if not (p / name).is_file():
        raise SystemExit(f"Missing model shard: {p / name}")
for pattern in ("action_head--*.pt", "proprio_projector--*.pt"):
    if len(list(p.glob(pattern))) != 1:
        raise SystemExit(f"Expected exactly one {pattern} in {p}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in this Python environment.")
# Exercise a CUDA kernel as well as device discovery.
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}")
PY
cd -- "$repo_dir"
exec python vla-scripts/deploy.py \
  --pretrained_checkpoint "$checkpoint_dir" \
  --robot_platform so101 \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 3 --use_proprio True --center_crop True \
  --unnorm_key so101_block_into_cup_50_v5 \
  --host "${VLA_HOST:-127.0.0.1}" --port "${VLA_PORT:-8777}"
