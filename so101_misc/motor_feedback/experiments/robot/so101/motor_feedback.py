"""Read-only motor telemetry; no LeRobot patching or motor register writes.

ST register 69 uses 6.5 mA/count. Torque is an uncalibrated magnitude estimate
assuming the configured constant describes output-shaft torque per reported amp.
Reference: https://docs.waveshare.net/Memory_Map_Explanation/ST_Servo_Memory_Map_Explanation/
"""

import csv
import math
import time

from experiments.robot.so101.constants import JOINT_NAMES

CURRENT_A_PER_COUNT = 0.0065
KGF_CM_TO_NM = 0.0980665
REGISTERS = {
    "present_current_raw": "Present_Current",
    "present_load": "Present_Load",
}
FIELDS = [
    "episode", "step", "motor", "timestamp_unix_s",
    "current_A", "load_pct", "errors", "torque_est_Nm",
]


class MotorFeedbackLogger:
    """Write one row per motor per observation step to an already-open CSV.

    Read failures leave blank values and an error; no stale values are reused.
    Load is sign-decoded by LeRobot even with normalize=False.
    Timestamp marks the start of the step's feedback reads on the host,
    not the servo's internal sensor sampling time. Raw values are not saved.
    """

    def __init__(self, stream, torque_constant_kgf_cm_per_A=11.0):
        if not math.isfinite(torque_constant_kgf_cm_per_A) or torque_constant_kgf_cm_per_A <= 0:
            raise ValueError("Torque constant must be finite and positive")
        self.stream = stream
        self.torque_constant = torque_constant_kgf_cm_per_A
        self.writer = csv.DictWriter(stream, fieldnames=FIELDS)
        self.writer.writeheader()
        self.stream.flush()

    def log_step(self, bus, *, episode, step):
        timestamp = time.time()
        readings, read_errors = {}, {}
        for field, register in REGISTERS.items():
            try:
                # Two sequential bus transactions, no position/velocity reads or retries.
                readings[field] = bus.sync_read(register, normalize=False, num_retry=0)
            except Exception as exc:
                readings[field] = {}
                read_errors[field] = f"{type(exc).__name__}: {exc}"

        common = dict(
            episode=episode, step=step, timestamp_unix_s=timestamp,
        )
        for joint in JOINT_NAMES:
            motor = joint.removesuffix(".pos")
            row = dict(common, motor=motor)
            raw_values = {}
            errors = []
            for field in REGISTERS:
                if field in read_errors:
                    errors.append(f"{field}: {read_errors[field]}")
                    continue
                value = readings[field].get(motor)
                if value is None:
                    errors.append(f"{field}: missing motor value")
                    continue
                try:
                    valid = math.isfinite(value)
                    if field == "present_current_raw":
                        valid = valid and value >= 0
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    errors.append(f"{field}: invalid value {value!r}")
                    continue
                raw_values[field] = value
            if "present_current_raw" in raw_values:
                row["current_A"] = raw_values["present_current_raw"] * CURRENT_A_PER_COUNT
                row["torque_est_Nm"] = row["current_A"] * self.torque_constant * KGF_CM_TO_NM
            if "present_load" in raw_values:
                row["load_pct"] = raw_values["present_load"] / 10.0
            row["errors"] = "; ".join(errors)
            self.writer.writerow(row)
        self.stream.flush()
