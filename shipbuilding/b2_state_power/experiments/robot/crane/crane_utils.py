"""Helpers shared by the magnet crane converter and its inference clients."""

import numpy as np

# power: task + this frame's magnet power level; task: the episode task only. State goes in as the proprio token.
LANGUAGE_MODES = ("power", "task")


def instruction_text(task: str, power_level: float, mode: str) -> str:
    """Language for one frame. Training data and inference requests must both build it with this function.

    NaN power level means the panel's power stage was off (info.json note); 0단 is the dial turned to 0 just before that
    and means the same, so both are shown as "꺼짐".
    """
    if mode == "task":
        return task
    if mode != "power":
        raise ValueError(f"Unknown language mode {mode!r}; choose one of {LANGUAGE_MODES}")
    power = "꺼짐" if np.isnan(power_level) or power_level == 0 else f"{int(power_level)}단"
    return f"{task}. 마그넷 세기 {power}"
