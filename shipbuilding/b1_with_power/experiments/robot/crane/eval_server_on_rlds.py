"""Replay RLDS validation frames through a deployed magnet crane checkpoint and score its on/off commands.

The server (vla-scripts/deploy.py --robot_platform crane --use_film True ...) must already be running. Each request
sends cam1 as `full_image`, cam2 and cam3 as `additional_images`, and the frame's stored instruction, which the
converter built with crane_utils.instruction_text (live clients must build it the same way).

`--max-requests 1` is a smoke test of the output contract; the default scores every `--frame-stride`-th frame.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import json_numpy
import numpy as np
import requests
import tensorflow_datasets as tfds

from experiments.robot.crane.constants import ACTION_CHUNK_SIZE, DATASET_NAME, MAGNET_NAMES

json_numpy.patch()
THRESHOLD = 0.5  # Unnormalized outputs are continuous; >= 0.5 means ON


def _builder_directory(data_root: Path, dataset_name: str) -> Path:
    candidates = sorted((data_root / dataset_name).glob("*/dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(f"No prepared TFDS dataset found under {data_root / dataset_name}")
    return candidates[-1].parent


def _text(value) -> str:
    value = value.numpy()
    return value.decode() if isinstance(value, bytes) else str(value)


def query(endpoint: str, step, timeout: float) -> np.ndarray:
    observation = step["observation"]
    payload = {
        "full_image": observation["cam1_image"].numpy(),
        "additional_images": [observation["cam2_image"].numpy(), observation["cam3_image"].numpy()],
        "state": observation["crane_pos"].numpy(),  # proprio = crane position x, y, z
        "instruction": _text(step["language_instruction"]),
    }
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    actions = np.asarray(response.json(), dtype=np.float32)
    expected = (ACTION_CHUNK_SIZE, len(MAGNET_NAMES))
    if actions.shape != expected:
        raise ValueError(f"Expected server output {expected}, got {actions.shape}: {response.text[:300]}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Server returned non-finite actions")
    return actions


def transition_errors(frames: list[int], truth: np.ndarray, pred: np.ndarray) -> tuple[list[int], int]:
    """For every true on/off change of each magnet, the frame distance to the nearest predicted change of the same
    direction in the episode (on the sampled frames). Returns (errors, number of true changes with no match)."""
    errors, missed = [], 0
    frames = np.asarray(frames)
    for magnet in range(truth.shape[1]):
        true_diff, pred_diff = np.diff(truth[:, magnet]), np.diff(pred[:, magnet])
        for i in np.flatnonzero(true_diff):
            same = np.flatnonzero(pred_diff == true_diff[i])
            if len(same) == 0:
                missed += 1
            else:
                errors.append(int(np.min(np.abs(frames[same + 1] - frames[i + 1]))))
    return errors, missed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--server-endpoint", default="http://127.0.0.1:8777/act")
    parser.add_argument("--frame-stride", type=int, default=15, help="Score every N-th frame (30 = once per second)")
    parser.add_argument("--max-requests", type=int, default=0, help="Stop after this many requests (0 = no limit)")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    builder = tfds.builder_from_directory(str(_builder_directory(args.data_root, args.dataset_name)))
    correct = np.zeros(len(MAGNET_NAMES))
    total, exact, requests_sent = 0, 0, 0
    all_errors, all_missed, all_changes = [], 0, 0
    per_episode = defaultdict(dict)
    for episode in builder.as_dataset(split=args.split, shuffle_files=False):
        tid = _text(episode["episode_metadata"]["tid"])
        frames, truth, pred = [], [], []
        for index, step in enumerate(episode["steps"]):
            if index % args.frame_stride:
                continue
            actions = query(args.server_endpoint, step, args.timeout)
            requests_sent += 1
            if requests_sent == 1:
                print(f"First request: instruction={_text(step['language_instruction'])!r}")
                print(f"  output shape {actions.shape}, first action {np.round(actions[0], 3).tolist()}")
            frames.append(index)
            truth.append(step["action"].numpy() >= THRESHOLD)
            pred.append(actions[0] >= THRESHOLD)
            if args.max_requests and requests_sent >= args.max_requests:
                break
        truth, pred = np.asarray(truth), np.asarray(pred)
        hits = truth == pred
        correct += hits.sum(axis=0)
        total += len(truth)
        exact += int(hits.all(axis=1).sum())
        errors, missed = transition_errors(frames, truth.astype(int), pred.astype(int))
        all_errors += errors
        all_missed += missed
        all_changes += len(errors) + missed
        per_episode[tid] = {"frames": len(truth), "accuracy": float(hits.mean())}
        if args.max_requests and requests_sent >= args.max_requests:
            break

    print(f"Scored {total} frames from {len(per_episode)} episodes (stride {args.frame_stride})")
    print("Per-magnet accuracy:", np.round(correct / max(total, 1), 3).tolist())
    print(f"Mean accuracy {correct.sum() / max(total * len(MAGNET_NAMES), 1):.3f}; all 10 magnets right {exact / max(total, 1):.3f}")
    if all_changes:
        median = f"{np.median(all_errors):.0f}" if all_errors else "n/a"
        print(f"On/off changes: {all_changes}; matched {len(all_errors)} (median error {median} frames), missed {all_missed}")
    for tid, result in sorted(per_episode.items()):
        print(f"  {tid}: {result['frames']} frames, accuracy {result['accuracy']:.3f}")


if __name__ == "__main__":
    main()
