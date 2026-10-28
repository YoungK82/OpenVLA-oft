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
attn_implementation="$(python - "$checkpoint_dir" "${VLA_ATTN_IMPLEMENTATION:-}" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path
import torch

p, override = Path(sys.argv[1]), sys.argv[2]
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
print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}", file=sys.stderr)

# The moojink transformers fork makes sdpa bidirectional (official OpenVLA-OFT); stock transformers keeps it causal.
direct_url = importlib.metadata.distribution("transformers").read_text("direct_url.json")
current_source = json.loads(direct_url).get("url", "unknown") if direct_url else "pypi"
record_path = p / "training_attention.json"
if record_path.is_file():
    record = json.loads(record_path.read_text())
    attn = record["attn_implementation"]
    if attn == "sdpa" and record["transformers_source"] != current_source:
        raise SystemExit(
            f"Checkpoint trained with sdpa under transformers from {record['transformers_source']}, "
            f"but this env uses {current_source}; sdpa causality would differ."
        )
else:
    # Checkpoints without a record came from the old server's stock transformers + flash_attention_2 (causal).
    attn = "eager"
    print("No training_attention.json; assuming a causal-attention checkpoint (eager).", file=sys.stderr)
if override and override != attn:
    raise SystemExit(f"VLA_ATTN_IMPLEMENTATION={override} contradicts the checkpoint's attention ({attn}).")
print(f"Attention: {attn} (transformers from {current_source})", file=sys.stderr)
print(attn)
PY
)"
# The action un-normalization key is the training dataset name, the only key in dataset_statistics.json
unnorm_key="${VLA_UNNORM_KEY:-$(python -c 'import json, sys; keys = list(json.load(open(sys.argv[1]))); print(keys[0] if len(keys) == 1 else "")' "$checkpoint_dir/dataset_statistics.json")}"
if [[ -z "$unnorm_key" ]]; then
  echo "Set VLA_UNNORM_KEY: $checkpoint_dir/dataset_statistics.json does not hold exactly one dataset." >&2
  exit 2
fi
cd -- "$repo_dir"
exec python vla-scripts/deploy.py \
  --pretrained_checkpoint "$checkpoint_dir" \
  --robot_platform so101 \
  --use_l1_regression True --use_diffusion False --use_film False \
  --num_images_in_input 3 --use_proprio True --center_crop True \
  --unnorm_key "$unnorm_key" \
  --attn_implementation "$attn_implementation" \
  --diagnostic_log "${VLA_DIAGNOSTIC_LOG:-}" \
  --host "${VLA_HOST:-127.0.0.1}" --port "${VLA_PORT:-8777}"
