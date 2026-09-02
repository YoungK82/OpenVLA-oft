"""Send one real RLDS observation to a deployed SO-101 checkpoint and validate its output contract."""

import argparse
from pathlib import Path

import json_numpy
import numpy as np
import requests
import tensorflow_datasets as tfds

from experiments.robot.so101.constants import ACTION_CHUNK_SIZE, JOINT_NAMES

json_numpy.patch()


def _builder_directory(data_root: Path, dataset_name: str) -> Path:
    candidates = sorted((data_root / dataset_name).glob("*/dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(f"No prepared TFDS dataset found under {data_root / dataset_name}")
    return candidates[-1].parent


def _numpy(value) -> np.ndarray:
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _decode_text(value) -> str:
    value = _numpy(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="so101_block_into_cup_50_v5")
    parser.add_argument("--server-endpoint", default="http://127.0.0.1:8777/act")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    builder_dir = _builder_directory(args.data_root, args.dataset_name)
    builder = tfds.builder_from_directory(str(builder_dir))
    split_names = set(builder.info.splits)
    split = "val" if "val" in split_names else "validation"
    if split not in split_names:
        raise ValueError(f"Expected a validation split, found {sorted(split_names)}")

    episode = next(iter(builder.as_dataset(split=split, shuffle_files=False)))
    step = next(iter(episode["steps"]))
    observation = step["observation"]
    images = [_numpy(observation[key]) for key in ("image", "top_image", "wrist_image")]
    state = np.asarray(_numpy(observation["state"]), dtype=np.float32)
    instruction = _decode_text(step["language_instruction"])

    for name, image in zip(("front", "top", "wrist"), images):
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Invalid {name} image: shape={image.shape}, dtype={image.dtype}")
    if state.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(state)):
        raise ValueError(f"Invalid SO-101 state: shape={state.shape}, value={state}")

    payload = {
        "full_image": images[0],
        "additional_images": images[1:],
        "state": state,
        "instruction": instruction,
    }
    response = requests.post(args.server_endpoint, json=payload, timeout=args.timeout)
    response.raise_for_status()
    actions = np.asarray(response.json(), dtype=np.float32)

    expected_shape = (ACTION_CHUNK_SIZE, len(JOINT_NAMES))
    if actions.shape != expected_shape:
        raise ValueError(f"Expected server output {expected_shape}, got {actions.shape}: {response.text[:500]}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Server returned non-finite SO-101 actions")

    print(f"Builder: {builder_dir}")
    print(f"Validation instruction: {instruction!r}")
    print(f"Input images: front/top/wrist {[image.shape for image in images]}")
    print(f"Input state ({state.shape}): {state}")
    print(f"Output actions: shape={actions.shape}, dtype={actions.dtype}")
    print(f"First action: {actions[0]}")
    print(f"First-action max absolute change from input state: {np.max(np.abs(actions[0] - state)):.6f}")
    print("SO-101 checkpoint server smoke test passed.")


if __name__ == "__main__":
    main()
