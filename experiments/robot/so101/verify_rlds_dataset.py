"""Verify the SO-101 RLDS contract before launching expensive OpenVLA fine-tuning."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from experiments.robot.so101.constants import CAMERA_NAMES, JOINT_NAMES


def _builder_directory(data_root: Path, dataset_name: str) -> Path:
    candidates = sorted((data_root / dataset_name).glob("*/dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(f"No prepared TFDS dataset found under {data_root / dataset_name}")
    return candidates[-1].parent


def _numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _decode_text(value) -> str:
    value = _numpy(value)
    if isinstance(value, np.ndarray):
        value = value.item()
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="so101_block_into_cup_50_v5")
    args = parser.parse_args()

    builder_dir = _builder_directory(args.data_root, args.dataset_name)
    builder = tfds.builder_from_directory(str(builder_dir))
    split_names = set(builder.info.splits)
    if "train" not in split_names or not ({"val", "validation"} & split_names):
        raise ValueError(f"Expected train and validation splits, found {sorted(split_names)}")

    episode = next(iter(builder.as_dataset(split="train", shuffle_files=False)))
    steps = list(episode["steps"].take(30))
    if len(steps) < 30:
        raise ValueError("First training episode has fewer than 30 frames and cannot produce a 30-action chunk")

    action_values = []
    state_values = []
    instructions = set()
    expected_images = ("image", "top_image", "wrist_image")
    for step in steps:
        observation = step["observation"]
        for key in expected_images:
            image = _numpy(observation[key])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Invalid {key}: shape={image.shape}, dtype={image.dtype}")
        action = _numpy(step["action"])
        state = _numpy(observation["state"])
        if action.shape != (len(JOINT_NAMES),) or state.shape != (len(JOINT_NAMES),):
            raise ValueError(f"Expected 6D action/state, got action={action.shape}, state={state.shape}")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)):
            raise ValueError("Dataset contains non-finite action or state values")
        action_values.append(action)
        state_values.append(state)
        instructions.add(_decode_text(step["language_instruction"]))

    manifest_path = builder_dir / "so101_conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    print(f"Builder: {builder_dir}")
    print(f"Splits: { {name: split.num_examples for name, split in builder.info.splits.items()} }")
    print(f"Cameras: {CAMERA_NAMES}")
    print(f"Joint order: {JOINT_NAMES}")
    print(f"First 30 action min: {np.min(action_values, axis=0)}")
    print(f"First 30 action max: {np.max(action_values, axis=0)}")
    print(f"First 30 state min: {np.min(state_values, axis=0)}")
    print(f"First 30 state max: {np.max(state_values, axis=0)}")
    print(f"Instructions: {sorted(instructions)}")
    print(f"Manifest: {manifest if manifest is not None else 'not found (re-run conversion to create it)'}")
    print("SO-101 RLDS structural verification passed.")


if __name__ == "__main__":
    main()
