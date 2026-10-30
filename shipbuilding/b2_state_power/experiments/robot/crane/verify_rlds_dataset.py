"""Verify the magnet crane RLDS contract before launching expensive OpenVLA fine-tuning."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds

from experiments.robot.crane.constants import ACTION_CHUNK_SIZE, CAMERA_FIELDS, DATASET_NAME, MAGNET_NAMES, STATE_NAMES


def _builder_directory(data_root: Path, dataset_name: str) -> Path:
    candidates = sorted((data_root / dataset_name).glob("*/dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(f"No prepared TFDS dataset found under {data_root / dataset_name}")
    return candidates[-1].parent


def _text(value) -> str:
    value = value.numpy()
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    builder_dir = _builder_directory(args.data_root, args.dataset_name)
    builder = tfds.builder_from_directory(str(builder_dir))
    manifest = json.loads((builder_dir / "crane_conversion_manifest.json").read_text())
    lengths = {e["tid"]: e["length"] for e in manifest["episodes"]}
    expected = {"train": set(manifest["training_tids"]), "validation": set(manifest["validation_tids"])}

    image_shape = (args.image_size, args.image_size, 3)
    total_steps, on_counts, seen = 0, np.zeros(len(MAGNET_NAMES)), set()
    for split in ("train", "validation"):
        tids, dates = set(), Counter()
        for episode in builder.as_dataset(split=split, shuffle_files=False):
            tid = _text(episode["episode_metadata"]["tid"])
            if tid in seen or tid not in lengths:
                raise ValueError(f"Unexpected or duplicate episode {tid}")
            seen.add(tid)
            tids.add(tid)
            dates[_text(episode["episode_metadata"]["date"])] += 1
            steps = 0
            for steps, step in enumerate(episode["steps"], start=1):
                observation = step["observation"]
                for key in CAMERA_FIELDS:
                    image = observation[key].numpy()
                    if image.shape != image_shape or image.dtype != np.uint8:
                        raise ValueError(f"{tid}: invalid {key} {image.shape} {image.dtype}")
                action, state = step["action"].numpy(), observation["state"].numpy()
                if action.shape != (len(MAGNET_NAMES),) or not np.all(np.isin(action, (0.0, 1.0))):
                    raise ValueError(f"{tid}: action must be 10 values of 0/1, got {action}")
                if state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(state)):
                    raise ValueError(f"{tid}: invalid state {state}")
                task, language = _text(step["episode_task"]), _text(step["language_instruction"])
                if not language.startswith(task):
                    raise ValueError(f"{tid}: instruction does not start with the episode task")
                if manifest["language_mode"] == "task" and language != task:
                    raise ValueError(f"{tid}: task-only dataset has extra instruction text")
                if manifest["language_mode"] == "power" and not language[len(task):].startswith(". 마그넷 세기 "):
                    raise ValueError(f"{tid}: instruction lacks the magnet power level: {language!r}")
                example = language
                flags = tuple(bool(step[name].numpy()) for name in ("is_first", "is_last", "is_terminal"))
                last = steps == lengths[tid]
                if flags != (steps == 1, last, last):
                    raise ValueError(f"{tid}: step {steps} has flags {flags}")
                on_counts += action
            if steps != lengths[tid] or steps < ACTION_CHUNK_SIZE:
                raise ValueError(f"{tid}: {steps} RLDS steps, expected {lengths[tid]}")
            total_steps += steps
        if tids != expected[split]:
            raise ValueError(f"{split} episodes differ from the manifest")
        print(f"{split}: {len(tids)} episodes, per date {dict(sorted(dates.items()))}")
    if seen != set(lengths):
        raise ValueError("Not all manifest episodes were found in RLDS")
    print(f"Builder: {builder_dir}")
    print(f"Language mode: {manifest['language_mode']}; last instruction seen: {example!r}")
    print(f"Cameras (cam1, cam2, cam3): {list(CAMERA_FIELDS.values())}")
    print(f"{len(seen)} episodes / {total_steps} steps; ON ratio per magnet: {np.round(on_counts / total_steps, 3).tolist()}")
    print("Magnet crane RLDS verification passed.")


if __name__ == "__main__":
    main()
