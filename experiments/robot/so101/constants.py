"""Shared SO-101 data and control contracts."""

JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
CAMERA_NAMES = ("front", "top", "wrist")
CAMERA_FIELDS = {
    "image": "observation.images.front",
    "top_image": "observation.images.top",
    "wrist_image": "observation.images.wrist",
}
CONTROL_FREQUENCY = 30
ACTION_CHUNK_SIZE = 30
