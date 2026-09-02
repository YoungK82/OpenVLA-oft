"""Evaluate an OpenVLA-OFT checkpoint on a physical SO-101 through the VLA server."""

import logging
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np

# ruff: noqa: E402
sys.path.append(".")

from experiments.robot.openvla_utils import get_action_from_server
from experiments.robot.robot_utils import DATE_TIME, get_image_resize_size, set_seed_everywhere
from experiments.robot.so101.so101_utils import (
    JOINT_NAMES,
    action_array_to_robot_dict,
    get_joint_state,
    prepare_policy_observation,
    save_rollout_video,
    validate_and_limit_action,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class GenerateConfig:
    # Model/server configuration
    model_family: str = "openvla"
    vla_server_url: Union[str, Path] = "127.0.0.1"
    vla_server_port: int = 8777
    num_open_loop_steps: int = 1

    # SO-101 and camera configuration. Use the same robot id and camera names used while recording.
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "so101_follower"
    front_camera: str = "0"
    top_camera: str = "1"
    wrist_camera: str = "2"
    camera_width: int = 640
    camera_height: int = 480
    control_frequency: int = 30
    max_relative_target: float = 5.0

    # Evaluation and safety. Dry-run is intentionally the default.
    dry_run: bool = True
    num_rollouts_planned: int = 10
    max_steps: int = 900
    task_description: str = ""
    run_id_note: Optional[str] = None
    local_log_dir: Path = Path("experiments/logs")
    rollout_dir: Path = Path("rollouts")
    seed: int = 7


def _camera_identifier(value: str):
    return int(value) if value.isdigit() else Path(value)


def _server_endpoint(cfg: GenerateConfig) -> str:
    value = str(cfg.vla_server_url).rstrip("/")
    if value.startswith(("http://", "https://")):
        return value if value.endswith("/act") else f"{value}/act"
    host = socket.gethostbyname(value)
    return f"http://{host}:{cfg.vla_server_port}/act"


def _make_robot(cfg: GenerateConfig):
    try:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise ImportError(
            "SO-101 evaluation requires LeRobot. Install experiments/robot/so101/requirements_so101.txt "
            "in the robot-client environment."
        ) from exc

    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=_camera_identifier(cfg.front_camera),
            fps=cfg.control_frequency,
            width=cfg.camera_width,
            height=cfg.camera_height,
        ),
        "top": OpenCVCameraConfig(
            index_or_path=_camera_identifier(cfg.top_camera),
            fps=cfg.control_frequency,
            width=cfg.camera_width,
            height=cfg.camera_height,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=_camera_identifier(cfg.wrist_camera),
            fps=cfg.control_frequency,
            width=cfg.camera_width,
            height=cfg.camera_height,
        ),
    }
    robot_config = SO101FollowerConfig(
        port=cfg.robot_port,
        id=cfg.robot_id,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=cfg.max_relative_target,
    )
    return SO101Follower(robot_config)


def validate_config(cfg: GenerateConfig) -> None:
    if not 1 <= cfg.num_open_loop_steps <= 30:
        raise ValueError("num_open_loop_steps must be between 1 and the 30-action SO-101 chunk size")
    if cfg.control_frequency != 30:
        raise ValueError("This checkpoint recipe uses 30 Hz data and 30-action chunks; set control_frequency=30")
    if cfg.max_relative_target <= 0:
        raise ValueError("max_relative_target must be positive")
    if not cfg.dry_run and cfg.num_open_loop_steps > 3:
        raise ValueError(
            "Refusing more than 3 open-loop hardware actions; validate the policy before increasing this limit"
        )


def _next_task(previous: str, configured: str) -> str:
    if configured:
        return configured
    prompt = "Task description" if not previous else f"Task description [Enter repeats {previous!r}]"
    value = input(f"{prompt}: ").strip()
    return value or previous


def _run_episode(cfg: GenerateConfig, robot, task: str, endpoint: str, resize_size, log_file):
    action_queue = deque()
    replay_images = []
    query_time = 0.0
    step_period = 1.0 / cfg.control_frequency

    input("Prepare the SO-101 scene, clear the workspace, then press Enter to begin...")
    for step in range(cfg.max_steps):
        step_start = time.perf_counter()
        observation = robot.get_observation()
        replay_images.append(np.asarray(observation["front"]))

        if not action_queue:
            payload = prepare_policy_observation(observation, resize_size)
            payload["instruction"] = task
            query_start = time.perf_counter()
            actions = np.asarray(get_action_from_server(payload, endpoint), dtype=np.float32)
            query_time += time.perf_counter() - query_start
            if actions.ndim != 2 or actions.shape[1] != len(JOINT_NAMES):
                raise ValueError(f"Server returned actions with unexpected shape {actions.shape}")
            action_queue.extend(actions[: cfg.num_open_loop_steps])

        current_state = get_joint_state(observation)
        predicted_action = action_queue.popleft()
        safe_action = validate_and_limit_action(predicted_action, current_state, cfg.max_relative_target)
        log_file.write(f"step={step} predicted={predicted_action.tolist()} safe={safe_action.tolist()}\n")
        log_file.flush()

        if cfg.dry_run:
            logger.info("DRY RUN step=%d action=%s", step, safe_action)
        else:
            robot.send_action(action_array_to_robot_dict(safe_action))

        remaining = step_period - (time.perf_counter() - step_start)
        if remaining > 0:
            time.sleep(remaining)

    return replay_images, query_time


@draccus.wrap()
def eval_so101(cfg: GenerateConfig) -> None:
    """Run manually reset SO-101 episodes using the ALOHA server/client pattern."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    cfg.local_log_dir.mkdir(parents=True, exist_ok=True)
    note = f"--{cfg.run_id_note}" if cfg.run_id_note else ""
    log_path = cfg.local_log_dir / f"EVAL-SO101-{DATE_TIME}{note}.txt"
    endpoint = _server_endpoint(cfg)
    resize_size = get_image_resize_size(cfg)
    robot = _make_robot(cfg)

    successes = 0
    completed = 0
    task = ""
    with log_path.open("w") as log_file:
        log_file.write(f"dry_run={cfg.dry_run} endpoint={endpoint}\n")
        try:
            robot.connect()
            for episode_index in range(cfg.num_rollouts_planned):
                task = _next_task(task, cfg.task_description)
                replay_images, query_time = _run_episode(cfg, robot, task, endpoint, resize_size, log_file)
                success = input("Success? [y/N]: ").strip().lower() == "y"
                completed += 1
                successes += int(success)
                video_path = save_rollout_video(
                    replay_images,
                    episode_index=completed,
                    success=success,
                    task=task,
                    fps=cfg.control_frequency,
                    output_dir=cfg.rollout_dir,
                )
                message = (
                    f"episode={completed} success={success} total_successes={successes}/{completed} "
                    f"query_time={query_time:.3f}s video={video_path}"
                )
                logger.info(message)
                log_file.write(f"{message}\n")
                log_file.flush()
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted; no further actions will be sent.")
        finally:
            if robot.is_connected:
                robot.disconnect()

    logger.info("Finished %d episodes with %d successes. Log: %s", completed, successes, log_path)


if __name__ == "__main__":
    eval_so101()
