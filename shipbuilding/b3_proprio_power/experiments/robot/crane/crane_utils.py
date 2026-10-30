"""Helpers shared by the magnet crane converter and its inference clients."""

import numpy as np

# b3_proprio_power: the instruction is the episode task only; load and magnet power level go in as proprio.
LANGUAGE_MODES = ("task",)
PROPRIO_NAMES = ("x_주행", "y_횡행", "z_권상", "load_total_t", "magnet_power_level")
POWER_OFF = 0.0  # Panel power off: NaN (stage off) and 0단 (dial turned to 0 just before it goes off) mean the same


def instruction_text(task: str, mode: str) -> str:
    """Language for one frame. Training data and inference requests must both build it with this function."""
    if mode not in LANGUAGE_MODES:
        raise ValueError(f"Unknown language mode {mode!r}; choose one of {LANGUAGE_MODES}")
    return task


def proprio_vector(state: np.ndarray, power_level: float) -> np.ndarray:
    """Proprio for one frame: [x, y, z, load, power]. `state` is observation.state = [load, x, y, z].

    The power level is the panel step (3, 4, 5); NaN and 0단 both become POWER_OFF, so no NaN reaches the normalization
    statistics. Training data and inference requests must both build it with this function.
    """
    state = np.asarray(state, dtype=np.float32)
    power = POWER_OFF if np.isnan(power_level) or power_level == 0 else float(power_level)
    return np.array([state[1], state[2], state[3], state[0], power], dtype=np.float32)
