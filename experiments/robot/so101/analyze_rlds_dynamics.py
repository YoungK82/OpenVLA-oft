"""Analyze SO-101 RLDS joint dynamics and their timing relative to camera motion."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw

from experiments.robot.so101.constants import JOINT_NAMES


CAMERA_KEYS = {"front": "image", "top": "top_image", "wrist": "wrist_image"}


def _builder_directory(data_root: Path, dataset_name: str) -> Path:
    candidates = sorted((data_root / dataset_name).glob("*/dataset_info.json"))
    if not candidates:
        raise FileNotFoundError(f"No prepared TFDS dataset found under {data_root / dataset_name}")
    return candidates[-1].parent


def _numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _best_lag(reference: np.ndarray, signal: np.ndarray, max_lag: int) -> tuple[int | None, float | None]:
    """Return the lag with maximum correlation; positive means the image signal occurs later."""
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left, right = reference[-lag:], signal[:lag]
        elif lag > 0:
            left, right = reference[:-lag], signal[lag:]
        else:
            left, right = reference, signal
        correlation = _correlation(left, right)
        if correlation is not None and np.isfinite(correlation):
            candidates.append((lag, correlation))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[1])


def _first_change(values: np.ndarray, epsilon: float) -> int | None:
    changed = np.flatnonzero(np.max(np.abs(np.diff(values, axis=0)), axis=1) > epsilon)
    return int(changed[0] + 1) if len(changed) else None


def _json_values(values: np.ndarray) -> list[float]:
    return [float(value) for value in values]


def _open_video_writer(output_dir: Path, episode_index: int, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None, None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode_{episode_index:06d}_annotated.mp4"
    writer = imageio.get_writer(path, fps=fps, codec="libx264")
    return writer, path


def _annotated_frame(images: dict[str, np.ndarray], frame_index: int, state: np.ndarray, action: np.ndarray) -> np.ndarray:
    camera_images = [Image.fromarray(images[name]).convert("RGB") for name in CAMERA_KEYS]
    width = sum(image.width for image in camera_images)
    image_height = max(image.height for image in camera_images)
    canvas = Image.new("RGB", (width, image_height + 64), "black")
    draw = ImageDraw.Draw(canvas)
    x_offset = 0
    for name, image in zip(CAMERA_KEYS, camera_images):
        canvas.paste(image, (x_offset, 0))
        draw.text((x_offset + 4, 4), name, fill="white", stroke_width=2, stroke_fill="black")
        x_offset += image.width
    draw.text((4, image_height + 4), f"frame={frame_index:04d} state={np.round(state, 2).tolist()}", fill="white")
    draw.text((4, image_height + 30), f"action={np.round(action, 2).tolist()}", fill="white")
    return np.asarray(canvas)


def _episode_index(episode) -> int:
    return int(_numpy(episode["episode_metadata"]["episode_index"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="so101_block_into_cup_50_v5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--joint-change-epsilon", type=float, default=1e-3)
    parser.add_argument("--max-lag-frames", type=int, default=5)
    parser.add_argument("--annotated-video-episodes", type=int, default=3)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--image-stride", type=int, default=4)
    args = parser.parse_args()

    if args.image_stride < 1:
        raise ValueError("--image-stride must be positive")

    # This diagnostic does not need the GPU and should not compete with model inference.
    tf.config.set_visible_devices([], "GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_dir / "annotated_videos"

    builder_dir = _builder_directory(args.data_root, args.dataset_name)
    builder = tfds.builder_from_directory(str(builder_dir))
    split_names = [name for name in ("train", "validation", "val") if name in builder.info.splits]

    global_actions = []
    global_states = []
    episode_reports = []
    transition_rows = []
    video_paths = []

    for split in split_names:
        for episode in builder.as_dataset(split=split, shuffle_files=False):
            episode_index = _episode_index(episode)
            actions = []
            states = []
            image_motion = {name: [] for name in CAMERA_KEYS}
            previous_images = None
            writer = None
            writer_path = None
            if episode_index < args.annotated_video_episodes:
                writer, writer_path = _open_video_writer(video_dir, episode_index, args.video_fps)

            try:
                for frame_index, step in enumerate(episode["steps"]):
                    observation = step["observation"]
                    action = np.asarray(_numpy(step["action"]), dtype=np.float32)
                    state = np.asarray(_numpy(observation["state"]), dtype=np.float32)
                    images = {
                        name: np.asarray(_numpy(observation[key]), dtype=np.uint8)
                        for name, key in CAMERA_KEYS.items()
                    }
                    actions.append(action)
                    states.append(state)

                    if previous_images is not None:
                        for name in CAMERA_KEYS:
                            current = images[name][:: args.image_stride, :: args.image_stride].astype(np.float32)
                            previous = previous_images[name][:: args.image_stride, :: args.image_stride].astype(np.float32)
                            image_motion[name].append(float(np.mean(np.abs(current - previous)) / 255.0))

                    if writer is not None:
                        writer.append_data(_annotated_frame(images, frame_index, state, action))
                    previous_images = images
            finally:
                if writer is not None:
                    writer.close()
                    video_paths.append(str(writer_path))

            actions_array = np.asarray(actions)
            states_array = np.asarray(states)
            action_speed = np.linalg.norm(np.diff(actions_array, axis=0), axis=1)
            state_speed = np.linalg.norm(np.diff(states_array, axis=0), axis=1)
            first_action_change = _first_change(actions_array, args.joint_change_epsilon)
            first_state_change = _first_change(states_array, args.joint_change_epsilon)

            alignment = {}
            for camera_name, motion_values in image_motion.items():
                motion = np.asarray(motion_values)
                state_lag, state_best_correlation = _best_lag(state_speed, motion, args.max_lag_frames)
                action_lag, action_best_correlation = _best_lag(action_speed, motion, args.max_lag_frames)
                alignment[camera_name] = {
                    "same_frame_state_correlation": _correlation(state_speed, motion),
                    "best_state_lag_frames": state_lag,
                    "best_state_correlation": state_best_correlation,
                    "same_frame_action_correlation": _correlation(action_speed, motion),
                    "best_action_lag_frames": action_lag,
                    "best_action_correlation": action_best_correlation,
                    "mean_pixel_change_fraction": float(np.mean(motion)) if len(motion) else None,
                }

            report = {
                "split": split,
                "episode_index": episode_index,
                "num_frames": len(actions_array),
                "first_action_change_frame_zero_based": first_action_change,
                "first_action_change_frame_one_based": None if first_action_change is None else first_action_change + 1,
                "first_state_change_frame_zero_based": first_state_change,
                "first_state_change_frame_one_based": None if first_state_change is None else first_state_change + 1,
                "first_30_action_range": _json_values(np.ptp(actions_array[:30], axis=0)),
                "first_30_state_range": _json_values(np.ptp(states_array[:30], axis=0)),
                "after_30_action_range": _json_values(np.ptp(actions_array[30:], axis=0)),
                "after_30_state_range": _json_values(np.ptp(states_array[30:], axis=0)),
                "action_min": _json_values(np.min(actions_array, axis=0)),
                "action_max": _json_values(np.max(actions_array, axis=0)),
                "action_std": _json_values(np.std(actions_array, axis=0)),
                "state_min": _json_values(np.min(states_array, axis=0)),
                "state_max": _json_values(np.max(states_array, axis=0)),
                "state_std": _json_values(np.std(states_array, axis=0)),
                "alignment": alignment,
            }
            episode_reports.append(report)
            global_actions.append(actions_array)
            global_states.append(states_array)

            for transition_index in range(len(action_speed)):
                transition_rows.append(
                    {
                        "split": split,
                        "episode_index": episode_index,
                        "from_frame": transition_index,
                        "to_frame": transition_index + 1,
                        "action_delta_l2": float(action_speed[transition_index]),
                        "state_delta_l2": float(state_speed[transition_index]),
                        **{
                            f"{name}_pixel_mad_fraction": image_motion[name][transition_index]
                            for name in CAMERA_KEYS
                        },
                    }
                )

    episode_reports.sort(key=lambda report: report["episode_index"])
    actions = np.concatenate(global_actions, axis=0)
    states = np.concatenate(global_states, axis=0)
    first_episode = episode_reports[0]

    summary = {
        "builder": str(builder_dir),
        "joint_names": JOINT_NAMES,
        "num_episodes": len(episode_reports),
        "num_frames": len(actions),
        "global": {
            "action_min": _json_values(np.min(actions, axis=0)),
            "action_max": _json_values(np.max(actions, axis=0)),
            "action_range": _json_values(np.ptp(actions, axis=0)),
            "action_std": _json_values(np.std(actions, axis=0)),
            "state_min": _json_values(np.min(states, axis=0)),
            "state_max": _json_values(np.max(states, axis=0)),
            "state_range": _json_values(np.ptp(states, axis=0)),
            "state_std": _json_values(np.std(states, axis=0)),
        },
        "first_episode": first_episode,
        "episodes_with_no_action_change": [
            report["episode_index"] for report in episode_reports if report["first_action_change_frame_zero_based"] is None
        ],
        "episodes_with_no_state_change": [
            report["episode_index"] for report in episode_reports if report["first_state_change_frame_zero_based"] is None
        ],
        "annotated_videos": video_paths,
        "episodes": episode_reports,
    }

    report_path = args.output_dir / "dynamics_report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = args.output_dir / "transition_metrics.csv"
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=transition_rows[0].keys())
        writer.writeheader()
        writer.writerows(transition_rows)

    print(f"Builder: {builder_dir}")
    print(f"Episodes / frames: {len(episode_reports)} / {len(actions)}")
    print(f"Joint order: {JOINT_NAMES}")
    print(f"Global action std: {np.std(actions, axis=0)}")
    print(f"Global state std: {np.std(states, axis=0)}")
    print(f"Global action range: {np.ptp(actions, axis=0)}")
    print(f"Global state range: {np.ptp(states, axis=0)}")
    print(
        "First episode change frames (one-based): "
        f"action={first_episode['first_action_change_frame_one_based']}, "
        f"state={first_episode['first_state_change_frame_one_based']}"
    )
    print(f"First episode action range after frame 30: {first_episode['after_30_action_range']}")
    print(f"First episode state range after frame 30: {first_episode['after_30_state_range']}")
    print(f"Episodes with no action change: {summary['episodes_with_no_action_change']}")
    print(f"Episodes with no state change: {summary['episodes_with_no_state_change']}")
    print(f"Detailed report: {report_path}")
    print(f"Transition metrics: {csv_path}")
    print(f"Annotated videos: {video_paths}")


if __name__ == "__main__":
    main()
