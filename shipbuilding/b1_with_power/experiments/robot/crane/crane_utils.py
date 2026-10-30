"""Helpers shared by the magnet crane converter and its inference clients."""

import numpy as np

# load_power: task + this frame's load (t) and magnet power level; task: the episode task only.
# The crane position (x, y, z) goes in as the proprio token.
LANGUAGE_MODES = ("load_power", "task")


def instruction_text(task: str, load: float, power_level: float, mode: str) -> str:
    """Language for one frame. Training data and inference requests must both build it with this function.

    `load` is load_total_t (ERP CRNWGT held between events). NaN power level means the panel's power stage was off
    (info.json note); 0단 is the dial turned to 0 just before that and means the same, so both are shown as "꺼짐".
    """
    if mode == "task":
        return task
    if mode != "load_power":
        raise ValueError(f"Unknown language mode {mode!r}; choose one of {LANGUAGE_MODES}")
    power = "꺼짐" if np.isnan(power_level) or power_level == 0 else f"{int(power_level)}단"
    return f"{task}. 하중 {load:.1f}t, 마그넷 세기 {power}"
