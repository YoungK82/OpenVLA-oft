#!/usr/bin/env bash
# Packs everything the SO-101 bidirectional training job needs on another (SLURM) cluster:
#   openvla-oft-code.tar.gz    committed code at HEAD (git archive; uncommitted edits are NOT included)
#   so101-rlds-1.1.0.tar       RLDS dataset version used by the recent trimmed runs (rlds/so101_block_into_cup_50_v5/1.1.0)
#   openvla-7b-47a0ec7.tar     base model (without config/modeling *.back.* backups)
#   requirements-cluster.txt   pinned Python packages of the env that trained/validated these runs
#   SO101_CLUSTER_TRAINING.md  setup + submission guide, SHA256SUMS
# Usage: bash experiments/robot/so101/pack_for_cluster.sh [OUT_DIR]
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
out_dir="${1:-$HOME/transfer/so101-cluster-$(date +%Y%m%d)}"
data_root="${DATA_ROOT:-/home/yhkim/datasets}"
dataset_rel="rlds/so101_block_into_cup_50_v5/1.1.0"
model_dir="${BASE_MODEL:-/home/shared/openvla-oft/models/openvla-7b-47a0ec7}"
python_bin="${PYTHON_BIN:-/home/yhkim/miniconda3/envs/openvla-oft/bin/python}"

test -f "$data_root/$dataset_rel/dataset_info.json"
test "$(find "$data_root/$dataset_rel" -maxdepth 1 -name '*-train.tfrecord-*' | wc -l)" -eq 16
test -f "$model_dir/model.safetensors.index.json"
if ! git -C "$repo_dir" diff --quiet HEAD -- vla-scripts/finetune.py experiments/robot/so101/slurm_train_bidir.sbatch; then
  echo "finetune.py or slurm_train_bidir.sbatch has uncommitted changes; commit them first." >&2
  exit 1
fi
mkdir -p "$out_dir"

commit="$(git -C "$repo_dir" rev-parse HEAD)"
git -C "$repo_dir" archive --format=tar.gz --prefix=openvla-oft/ -o "$out_dir/openvla-oft-code.tar.gz" HEAD
echo "$commit" > "$out_dir/CODE_COMMIT"

tar -C "$data_root" -cf "$out_dir/so101-rlds-1.1.0.tar" "$dataset_rel"
tar -C "$(dirname "$model_dir")" --exclude='*.back.*' -cf "$out_dir/openvla-7b-47a0ec7.tar" "$(basename "$model_dir")"

# Pin the packages the code was validated with; torch is installed separately to match the cluster's CUDA driver.
"$python_bin" - "$out_dir/requirements-cluster.txt" <<'PY'
import importlib.metadata as md, json, sys
pins = ["accelerate", "draccus", "einops", "huggingface_hub", "json-numpy", "jsonlines", "peft", "protobuf", "rich",
        "sentencepiece", "timm", "tokenizers", "wandb", "tensorflow", "tensorflow-datasets", "tensorflow-metadata",
        "tensorflow-graphics", "diffusers", "numpy", "safetensors"]
lines = ["# torch/torchvision: install first for the cluster's CUDA (see SO101_CLUSTER_TRAINING.md)"]
for name in ["transformers", "dlimp"]:
    dist = md.distribution(name)
    info = json.loads(dist.read_text("direct_url.json"))
    lines.append(f"{name} @ git+{info['url']}@{info['vcs_info']['commit_id']}")
lines += [f"{name}=={md.version(name)}" for name in pins]
open(sys.argv[1], "w").write("\n".join(lines) + "\n")
PY

cp "$repo_dir/SO101_CLUSTER_TRAINING.md" "$out_dir/"
(cd "$out_dir" && sha256sum openvla-oft-code.tar.gz so101-rlds-1.1.0.tar openvla-7b-47a0ec7.tar requirements-cluster.txt > SHA256SUMS)
du -sh "$out_dir"/*
echo "Packed commit $commit into $out_dir"
