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

# This folder's variant. b1_with_power: DATASET_NAME "crane_b_magnet_onoff_power", LANGUAGE_MODE "power";
# b2_without_power: DATASET_NAME "crane_b_magnet_onoff_nopower", LANGUAGE_MODE "task". Both feed `state` as proprio.
DATASET_NAME = "crane_b_magnet_onoff_nopower"
LANGUAGE_MODE = "task"
# Validation episodes picked at random within each date (YYYYMMDD -> count)
VALIDATION_PER_DATE = {"20261012": 1, "20261013": 6, "20261014": 3}
