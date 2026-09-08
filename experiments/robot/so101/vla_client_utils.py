"""Lightweight, robot-client-side copies of the VLA server helpers the SO-101 client needs.

Deliberately does NOT import `experiments.robot.openvla_utils` or `experiments.robot.robot_utils`: those modules
also import torch (and openvla_utils additionally imports transformers and prismatic), none of which the robot
machine needs, and torch has no bearing on what the client actually does (read cameras, call the VLA server).
`resize_image_for_policy` is only used server-side (in `prepare_images_for_vla`) when the client sends an image
that is not already the model's expected resolution; keeping the client's copy byte-for-byte identical to the
server's (`experiments/robot/openvla_utils.py`) preserves the training-time resize/JPEG contract described in
`SO101.md`.
"""

import os
import random
import time
from typing import Any, Dict, Tuple, Union

import numpy as np
import requests
import tensorflow as tf

# Mirrors experiments/robot/robot_utils.py::MODEL_IMAGE_SIZES for the one model family SO-101 clients use.
MODEL_IMAGE_SIZES = {
    "openvla": 224,
}

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching. Kept in sync with
    `experiments/robot/openvla_utils.py::resize_image_for_policy`.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act", timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server. Kept in sync with
    `experiments/robot/openvla_utils.py::get_action_from_server`.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_image_resize_size(cfg: Any) -> Union[int, Tuple[int, int]]:
    """Client-side equivalent of `experiments/robot/robot_utils.py::get_image_resize_size` (no torch import)."""
    if cfg.model_family not in MODEL_IMAGE_SIZES:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")
    return MODEL_IMAGE_SIZES[cfg.model_family]


def set_seed_everywhere(seed: int) -> None:
    """Seed the RNGs the robot client actually uses. No torch: the client runs no model, so no torch RNG exists."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
