"""Evaluate a deployed SO-101 policy server on known RLDS trajectory frames."""

import argparse
import json
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
    value = _numpy(value)
    if isinstance(value, np.ndarray):
        value = value.item()
    return value.decode() if isinstance(value, bytes) else str(value)


def _sample_indices(length: int, samples_per_episode: int) -> np.ndarray:
    last_start = length - ACTION_CHUNK_SIZE
    if last_start < 0:
        raise ValueError(f"Episode has {length} steps, fewer than chunk size {ACTION_CHUNK_SIZE}")
    return np.unique(np.rint(np.linspace(0, last_start, samples_per_episode)).astype(int))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="so101_block_into_cup_50_v5")
    parser.add_argument("--server-endpoint", default="http://127.0.0.1:8777/act")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    builder_dir = _builder_directory(args.data_root, args.dataset_name)
    builder = tfds.builder_from_directory(str(builder_dir))
    split = args.split
    if split not in builder.info.splits and split == "validation" and "val" in builder.info.splits:
        split = "val"
    if split not in builder.info.splits:
        raise ValueError(f"Split {args.split!r} not found; available: {sorted(builder.info.splits)}")

    records = []
    dataset = builder.as_dataset(split=split, shuffle_files=False)
    for dataset_episode_index, episode in enumerate(dataset.take(args.num_episodes)):
        steps = list(episode["steps"])
        episode_index = int(_numpy(episode["episode_metadata"]["episode_index"]))
        for frame_index in _sample_indices(len(steps), args.samples_per_episode):
            step = steps[frame_index]
            observation = step["observation"]
            images = [_numpy(observation[key]) for key in ("image", "top_image", "wrist_image")]
            state = np.asarray(_numpy(observation["state"]), dtype=np.float32)
            instruction = _decode_text(step["language_instruction"])
            ground_truth = np.stack(
                [np.asarray(_numpy(item["action"]), dtype=np.float32) for item in steps[frame_index : frame_index + ACTION_CHUNK_SIZE]]
            )

            payload = {
                "full_image": images[0],
                "additional_images": images[1:],
                "state": state,
                "instruction": instruction,
            }
            response = requests.post(args.server_endpoint, json=payload, timeout=args.timeout)
            response.raise_for_status()
            predicted = np.asarray(response.json(), dtype=np.float32)
            expected_shape = (ACTION_CHUNK_SIZE, len(JOINT_NAMES))
            if predicted.shape != expected_shape:
                raise ValueError(f"Expected output {expected_shape}, got {predicted.shape}")

            record = {
                "dataset_episode_order": dataset_episode_index,
                "episode_index": episode_index,
                "frame_index": int(frame_index),
                "state": state.tolist(),
                "ground_truth_first": ground_truth[0].tolist(),
                "predicted_first": predicted[0].tolist(),
                "first_action_mae": float(np.mean(np.abs(predicted[0] - ground_truth[0]))),
                "chunk_mae": float(np.mean(np.abs(predicted - ground_truth))),
                "ground_truth_chunk_range": np.ptp(ground_truth, axis=0).tolist(),
                "predicted_chunk_range": np.ptp(predicted, axis=0).tolist(),
            }
            records.append(record)
            print(
                f"episode={episode_index} frame={frame_index} "
                f"first_mae={record['first_action_mae']:.4f} chunk_mae={record['chunk_mae']:.4f} "
                f"predicted_range={np.round(np.ptp(predicted, axis=0), 3).tolist()}"
            )

    predicted_first = np.asarray([record["predicted_first"] for record in records])
    ground_truth_first = np.asarray([record["ground_truth_first"] for record in records])
    summary = {
        "builder": str(builder_dir),
        "split": split,
        "server_endpoint": args.server_endpoint,
        "num_samples": len(records),
        "mean_first_action_mae": float(np.mean([record["first_action_mae"] for record in records])),
        "mean_chunk_mae": float(np.mean([record["chunk_mae"] for record in records])),
        "predicted_first_action_std_across_samples": np.std(predicted_first, axis=0).tolist(),
        "ground_truth_first_action_std_across_samples": np.std(ground_truth_first, axis=0).tolist(),
        "mean_predicted_chunk_range": np.mean(
            [record["predicted_chunk_range"] for record in records], axis=0
        ).tolist(),
        "mean_ground_truth_chunk_range": np.mean(
            [record["ground_truth_chunk_range"] for record in records], axis=0
        ).tolist(),
    }
    report = {"summary": summary, "samples": records}
    print("SUMMARY " + json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
