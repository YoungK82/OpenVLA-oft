"""Shared magnet crane (shipbuilding) data contract."""

# The three exo cameras, in camera-number order: CAM1, CAM2, CAM3 (RLDS key -> LeRobot feature)
CAMERA_FIELDS = {
    "cam1_image": "observation.images.exo_1",  # CAM1
    "cam2_image": "observation.images.exo",  # CAM2
    "cam3_image": "observation.images.exo_3",  # CAM3
}
MAGNET_NAMES = tuple(f"magnet_{i}_onoff" for i in range(1, 11))  # POT1..POT10 crane excitation commands, 0/1
STATE_NAMES = ("load_total_t", "x_주행", "y_횡행", "z_권상")
CONTROL_FREQUENCY = 30
ACTION_CHUNK_SIZE = 8  # Keep equal to CRANE_CONSTANTS["NUM_ACTIONS_CHUNK"] in prismatic/vla/constants.py

# This folder's variant. b1_with_power: DATASET_NAME "crane_b_magnet_onoff_loadpower", LANGUAGE_MODE "load_power",
# proprio = crane_pos (x, y, z). b2_state_power: DATASET_NAME "crane_b_magnet_onoff_state_power", LANGUAGE_MODE "power",
# proprio = state (load, x, y, z).
# "_tvt" = train/validation/test 50/10/10 (2026-11-04). The earlier "crane_b_magnet_onoff_loadpower" (train 60 / val 10, no test)
# is kept on disk for the checkpoints trained on it; its train set includes the test episodes below.
DATASET_NAME = "crane_b_magnet_onoff_loadpower_tvt"
LANGUAGE_MODE = "load_power"
# Validation episodes picked at random within each date (YYYYMMDD -> count)
VALIDATION_PER_DATE = {"20261012": 1, "20261013": 6, "20261014": 3}
# Test episodes, drawn the same way from what is left after validation and EXCLUDED_TIDS (same seed).
# Never used for training or checkpoint selection: score it once at the end with eval_server_on_rlds.py --split test
TEST_PER_DATE = {"20261012": 1, "20261013": 6, "20261014": 3}
# Episodes left out of the RLDS dataset (bad recordings). The validation split is still drawn from all folders,
# so the same validation TIDs come out as before these were dropped.
EXCLUDED_TIDS = (
    "2026101310335401",
    "2026101315232601",
    "2026101316261101",  # recording cut mid-episode
    "2026101413483701",
    "2026101416333501",
)
