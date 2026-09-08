"""SO-101 observation, action, safety, and logging helpers."""

import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from experiments.robot.so101.constants import CAMERA_NAMES, JOINT_NAMES

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def get_joint_state(observation: Mapping) -> np.ndarray:
    """Read SO-101 joints in the same order used by the converted training dataset."""
    missing = [name for name in JOINT_NAMES if name not in observation]
    if missing:
        raise KeyError(f"SO-101 observation is missing joint fields: {missing}")
    state = np.asarray([observation[name] for name in JOINT_NAMES], dtype=np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"SO-101 returned a non-finite joint state: {state}")
    return state


def prepare_policy_observation(observation: Mapping, resize_size) -> dict:
    """Build the server payload in the training order: front, top, wrist, then state."""
    from experiments.robot.so101.vla_client_utils import resize_image_for_policy

    missing = [name for name in CAMERA_NAMES if name not in observation]
    if missing:
        raise KeyError(f"SO-101 observation is missing camera fields: {missing}")

    images = []
    for name in CAMERA_NAMES:
        image = np.asarray(observation[name])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Camera {name!r} must be uint8 HxWx3, got shape={image.shape}, dtype={image.dtype}")
        images.append(resize_image_for_policy(image, resize_size))

    return {
        "full_image": images[0],
        "additional_images": images[1:],
        "state": get_joint_state(observation),
    }


def validate_and_limit_action(
    action: Sequence[float], current_state: np.ndarray, max_relative_target: float
) -> np.ndarray:
    """Reject malformed predictions and limit each command relative to the measured pose."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (len(JOINT_NAMES),):
        raise ValueError(f"Expected a {len(JOINT_NAMES)}D SO-101 action, got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError(f"Model predicted a non-finite action: {action}")
    if max_relative_target <= 0:
        raise ValueError("max_relative_target must be positive")
    return np.clip(action, current_state - max_relative_target, current_state + max_relative_target)


def action_array_to_robot_dict(action: Sequence[float]) -> dict[str, float]:
    """Convert the model's ordered vector to LeRobot's named SO-101 action contract."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (len(JOINT_NAMES),):
        raise ValueError(f"Expected a {len(JOINT_NAMES)}D SO-101 action, got {action.shape}")
    return {name: float(value) for name, value in zip(JOINT_NAMES, action)}


def save_rollout_video(images, episode_index: int, success: bool, task: str, fps: int, output_dir: Path) -> Path:
    """Save the front-camera rollout using the same naming convention as ALOHA evaluation."""
    import imageio

    output_dir = Path(output_dir) / DATE
    os.makedirs(output_dir, exist_ok=True)
    safe_task = task.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    path = output_dir / (
        f"{DATE_TIME}--openvla_oft--so101--episode={episode_index}--success={success}--task={safe_task}.mp4"
    )
    writer = imageio.get_writer(path, fps=fps)
    try:
        for image in images:
            writer.append_data(np.asarray(image))
    finally:
        writer.close()
    return path
