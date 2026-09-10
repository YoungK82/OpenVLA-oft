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
    parser.add_argument("--expected-version")
    parser.add_argument("--full-scan", action="store_true")
    args = parser.parse_args()

    builder_dir = _builder_directory(args.data_root, args.dataset_name)
    builder = tfds.builder_from_directory(str(builder_dir))
    if args.expected_version is not None and str(builder.info.version) != args.expected_version:
        raise ValueError(f"Expected dataset version {args.expected_version}, found {builder.info.version}")
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
    if args.full_scan:
        if manifest is None:
            raise ValueError("Full scan requires so101_conversion_manifest.json")
        expected_records = {record["episode_index"]: record for record in manifest["episodes"]}
        expected_splits = {
            "train": set(manifest["training_episode_indices"]),
            "validation": set(manifest["validation_episode_indices"]),
        }
        seen_episodes = set()
        total_steps = 0
        for split in ("train", "validation"):
            actual_split = split if split in split_names else "val"
            split_episodes = set()
            for full_episode in builder.as_dataset(split=actual_split, shuffle_files=False):
                episode_index = int(_numpy(full_episode["episode_metadata"]["episode_index"]))
                if episode_index in seen_episodes or episode_index not in expected_records:
                    raise ValueError(f"Unexpected or duplicate episode {episode_index}")
                seen_episodes.add(episode_index)
                split_episodes.add(episode_index)
                step_count = 0
                last_flags = None
                for step_count, full_step in enumerate(full_episode["steps"], start=1):
                    observation = full_step["observation"]
                    for key in expected_images:
                        image = _numpy(observation[key])
                        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                            raise ValueError(
                                f"Episode {episode_index} has invalid {key}: shape={image.shape}, dtype={image.dtype}"
                            )
                    action = _numpy(full_step["action"])
                    state = _numpy(observation["state"])
                    if action.shape != (len(JOINT_NAMES),) or state.shape != (len(JOINT_NAMES),):
                        raise ValueError(f"Episode {episode_index} has invalid action/state shape")
                    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)):
                        raise ValueError(f"Episode {episode_index} has non-finite action/state")
                    flags = tuple(bool(_numpy(full_step[name])) for name in ("is_first", "is_last", "is_terminal"))
                    if step_count == 1 and flags != (True, False, False):
                        raise ValueError(f"Episode {episode_index} has invalid first-step flags {flags}")
                    if step_count > 1 and flags[0]:
                        raise ValueError(f"Episode {episode_index} has multiple is_first steps")
                    last_flags = flags
                expected_length = int(expected_records[episode_index]["output_length"])
                if step_count != expected_length:
                    raise ValueError(
                        f"Episode {episode_index} has {step_count} RLDS steps, expected {expected_length}"
                    )
                if last_flags != (False, True, True):
                    raise ValueError(f"Episode {episode_index} has invalid final-step flags {last_flags}")
                total_steps += step_count
            if split_episodes != expected_splits[split]:
                raise ValueError(
                    f"{split} episode mismatch: expected={sorted(expected_splits[split])}, "
                    f"actual={sorted(split_episodes)}"
                )
        if seen_episodes != set(expected_records):
            raise ValueError("Not all manifest episodes were found in RLDS")
        print(f"Full scan: {len(seen_episodes)} episodes / {total_steps} steps passed")
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
