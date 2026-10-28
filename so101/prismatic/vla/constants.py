"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

SO101_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 30,
    "ACTION_DIM": 6,
    "PROPRIO_DIM": 6,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


ROBOT_CONSTANTS = {
    "LIBERO": LIBERO_CONSTANTS,
    "ALOHA": ALOHA_CONSTANTS,
    "SO101": SO101_CONSTANTS,
    "BRIDGE": BRIDGE_CONSTANTS,
}


def _get_cli_arg_value(flag: str):
    """Return a CLI value passed as either ``--flag value`` or ``--flag=value``."""
    for index, argument in enumerate(sys.argv[1:]):
        if argument == flag and index + 2 <= len(sys.argv[1:]):
            return sys.argv[index + 2]
        if argument.startswith(f"{flag}="):
            return argument.split("=", maxsplit=1)[1]
    return None


def detect_robot_platform():
    """Resolve robot constants before modules that depend on their tensor shapes are imported."""
    explicit_platform = _get_cli_arg_value("--robot_platform")
    if explicit_platform is not None and explicit_platform.lower() != "auto":
        platform = explicit_platform.upper().replace("-", "")
        if platform not in ROBOT_CONSTANTS:
            raise ValueError(
                f"Unsupported --robot_platform={explicit_platform!r}; choose one of "
                f"{sorted(ROBOT_CONSTANTS)} or 'auto'."
            )
        return platform

    # Backward compatibility for existing commands. Prefer matching the dataset argument rather than arbitrary paths.
    dataset_name = _get_cli_arg_value("--dataset_name") or _get_cli_arg_value("--unnorm_key") or ""
    dataset_name = str(dataset_name).lower()
    for marker, platform in (("so101", "SO101"), ("aloha", "ALOHA"), ("libero", "LIBERO"), ("bridge", "BRIDGE")):
        if marker in dataset_name:
            return platform

    # Existing LIBERO commands rely on this default.
    return "LIBERO"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
constants = ROBOT_CONSTANTS[ROBOT_PLATFORM]

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("Use --robot_platform to select a platform explicitly when launching training or deployment.")
