"""Helpers shared by the magnet crane converter and its inference clients."""

# b4_state: the instruction is the episode task only. observation.state = [load, x, y, z] goes in as the proprio token,
# exactly as the data provider bundles it. The magnet power level is not used at all (not in language, proprio or RLDS).
LANGUAGE_MODES = ("task",)


def instruction_text(task: str, mode: str = "task") -> str:
    """Language for one frame. Training data and inference requests must both build it with this function."""
    if mode not in LANGUAGE_MODES:
        raise ValueError(f"Unknown language mode {mode!r}; choose one of {LANGUAGE_MODES}")
    return task
