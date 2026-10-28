"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
    get_action_head,
    get_processor,
    get_proprio_projector,
    normalize_proprio,
)
from experiments.robot.robot_utils import (
    get_image_resize_size,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_TOKEN_BEGIN_IDX,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
    ROBOT_PLATFORM,
    STOP_INDEX,
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        # Load model
        self.vla = get_vla(cfg)

        # Load proprio projector
        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM)

        # Load continuous action head
        self.action_head = None
        if cfg.use_l1_regression or cfg.use_diffusion:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        # Check that the model contains the action un-normalization key
        assert cfg.unnorm_key in self.vla.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

        # Get Hugging Face processor
        self.processor = None
        self.processor = get_processor(cfg)

        # Get expected image dimensions
        self.resize_size = get_image_resize_size(cfg)


    def _validate_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Reject malformed observations before normalization can broadcast or clip invalid state."""
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a JSON object")
        instruction = observation.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")

        observation = dict(observation)
        if self.cfg.use_proprio:
            state = np.asarray(observation.get("state"))
            if (state.shape != (PROPRIO_DIM,) or not np.issubdtype(state.dtype, np.number)
                    or np.iscomplexobj(state) or not np.all(np.isfinite(state))):
                raise ValueError(f"state must contain exactly {PROPRIO_DIM} finite numeric values")
            observation["state"] = state

        images = [observation.get("full_image")]
        if self.cfg.num_images_in_input > 1:
            if "additional_images" in observation:
                additional = observation["additional_images"]
                if not isinstance(additional, (list, tuple)):
                    raise ValueError("additional_images must be an ordered list of images")
                images.extend(additional)
            else:
                images.extend(observation[key] for key in observation if "wrist" in key)
        if len(images) != self.cfg.num_images_in_input:
            raise ValueError(f"Expected {self.cfg.num_images_in_input} images, got {len(images)}")
        for index, image in enumerate(images):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
                    or image.shape[2] != 3 or min(image.shape[:2]) == 0):
                raise ValueError(f"Image {index} must be a non-empty uint8 HxWx3 NumPy array")
        return observation

    def get_server_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            if double_encode := "encoded" in payload:
                # Support the existing double-encoded NumPy JSON client contract.
                if len(payload) != 1 or not isinstance(payload["encoded"], str):
                    raise ValueError("encoded must be a JSON string and the only request field")
                payload = json.loads(payload["encoded"])
            observation = self._validate_observation(payload)
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            instruction = observation["instruction"]
            request_id = str(observation.get("request_id", uuid.uuid4().hex))
            started = time.perf_counter()

            action = get_vla_action(
                self.cfg, self.vla, self.processor, observation, instruction, action_head=self.action_head, proprio_projector=self.proprio_projector, use_film=self.cfg.use_film,
            )

            predicted = np.asarray(action, dtype=np.float64)
            if predicted.shape != (NUM_ACTIONS_CHUNK, ACTION_DIM) or not np.all(np.isfinite(predicted)):
                raise RuntimeError("Model returned malformed or non-finite actions")

            if self.cfg.diagnostic_log:
                state = np.asarray(observation["state"], dtype=np.float64)
                stats = self.vla.norm_stats[self.cfg.unnorm_key]["proprio"]
                record = {
                    "request_id": request_id,
                    "time_unix": time.time(),
                    "checkpoint": str(self.cfg.pretrained_checkpoint),
                    "instruction": instruction,
                    "state": state.tolist(),
                    "normalized_state": normalize_proprio(state, stats).tolist(),
                    "state_outside_training_min_max": ((state < np.asarray(stats["min"])) | (state > np.asarray(stats["max"]))).tolist(),
                    "actions": predicted.tolist(),
                    "first_target_minus_state": (predicted[0] - state).tolist(),
                    "inference_seconds": time.perf_counter() - started,
                }
                log_path = Path(self.cfg.diagnostic_log)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print("ACTION_DIAGNOSTIC " + json.dumps(record), flush=True)

            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except Exception:
            logging.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Inference failed; see the server log")

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8777                                                    # Host Port
    diagnostic_log: str = ""                        # Optional JSONL input/action trace

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    robot_platform: str = "auto"                     # Pass `so101` explicitly for SO-101 checkpoints

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)
    # The moojink transformers fork makes sdpa bidirectional; checkpoints trained with stock transformers need "eager".
    attn_implementation: Optional[str] = None        # None (HF default), eager, sdpa, or flash_attention_2

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    requested_platform = cfg.robot_platform.upper().replace("-", "")
    if requested_platform != "AUTO" and requested_platform != ROBOT_PLATFORM:
        raise ValueError(f"Requested {requested_platform} constants, but {ROBOT_PLATFORM} constants were loaded.")
    if ROBOT_PLATFORM == "SO101" and cfg.num_images_in_input != 3:
        raise ValueError("SO-101 checkpoints expect front, top, and wrist images; set --num_images_in_input 3.")
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
