"""Offline telemetry checks. No hardware or VLA server is used."""

import csv
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from experiments.robot.so101.constants import JOINT_NAMES
from experiments.robot.so101.motor_feedback import MotorFeedbackLogger


class MotorFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.stream = io.StringIO()
        self.logger = MotorFeedbackLogger(self.stream)
        self.observation = {joint: index + 0.25 for index, joint in enumerate(JOINT_NAMES)}
        self.bus = Mock(spec=["sync_read"])
        self.bus.sync_read.side_effect = self.read

    def read(self, register, **kwargs):
        value = {"Present_Current": 100, "Present_Load": -300}[register]
        return {joint.removesuffix(".pos"): value for joint in JOINT_NAMES}

    def log(self, step=0):
        self.logger.log_step(self.bus, episode=2, step=step)
        return list(csv.DictReader(io.StringIO(self.stream.getvalue())))

    def test_conversion_units_and_read_only_calls(self):
        with patch("experiments.robot.so101.motor_feedback.time.time", return_value=1234.5):
            rows = self.log()
        self.assertEqual(len(rows), 6)
        row = rows[0]
        self.assertEqual(list(row), ["episode", "step", "motor", "timestamp_unix_s",
                                     "current_A", "load_pct", "errors", "torque_est_Nm"])
        self.assertTrue(all(float(r["timestamp_unix_s"]) == 1234.5 for r in rows))
        self.assertAlmostEqual(float(row["current_A"]), 0.65)
        self.assertAlmostEqual(float(row["torque_est_Nm"]), 0.701175475)
        self.assertEqual(float(row["load_pct"]), -30)
        self.assertEqual(row["episode"], "2")
        self.assertEqual(row["errors"], "")
        self.assertEqual(self.bus.sync_read.call_count, 2)
        self.bus.sync_read.assert_any_call("Present_Current", normalize=False, num_retry=0)
        self.bus.sync_read.assert_any_call("Present_Load", normalize=False, num_retry=0)

    def test_failed_read_does_not_reuse_prior_current(self):
        self.log()

        def fail_current(register, **kwargs):
            if register == "Present_Current":
                raise ConnectionError("packet timeout")
            return self.read(register)

        self.bus.sync_read.side_effect = fail_current
        rows = self.log(step=1)
        for row in rows[6:]:
            self.assertEqual(row["current_A"], "")
            self.assertEqual(row["torque_est_Nm"], "")
            self.assertEqual(row["load_pct"], "-30.0")
            self.assertIn("packet timeout", row["errors"])

    def test_missing_and_invalid_values_are_not_zero_measurements(self):
        def incomplete(register, **kwargs):
            values = self.read(register)
            if register == "Present_Current":
                del values["shoulder_pan"]
                values["shoulder_lift"] = float("nan")
                values["elbow_flex"] = -1
                values["wrist_flex"] = 0
            return values

        self.bus.sync_read.side_effect = incomplete
        rows = self.log()
        for row in rows[:3]:
            self.assertEqual(row["current_A"], "")
            self.assertTrue(row["errors"])
        self.assertEqual(float(rows[3]["current_A"]), 0)
        self.assertEqual(rows[3]["errors"], "")

    def test_constant_is_configurable_and_validated(self):
        self.stream = io.StringIO()
        self.logger = MotorFeedbackLogger(self.stream, 5)
        rows = self.log()
        self.assertAlmostEqual(float(rows[-1]["torque_est_Nm"]), 0.318716125)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MotorFeedbackLogger(io.StringIO(), value)

    def test_episode_samples_each_step_before_inference_or_action(self):
        from experiments.robot.so101 import run_so101_eval as evaluation

        events = []
        obs = dict(self.observation, front=np.zeros((2, 2, 3), dtype=np.uint8))
        robot = SimpleNamespace(bus=self.bus,
                                get_observation=lambda: (events.append("observe") or obs),
                                send_action=Mock(side_effect=lambda action: events.append("action")))
        actual_log_step = self.logger.log_step

        def record(*args, **kwargs):
            events.append("feedback")
            actual_log_step(*args, **kwargs)

        cfg = evaluation.GenerateConfig(max_steps=2, num_open_loop_steps=2, dry_run=False)
        actions = np.array([[obs[joint] for joint in JOINT_NAMES]] * 2)
        with patch("builtins.input", return_value=""), \
             patch.object(evaluation, "prepare_policy_observation", return_value={}), \
             patch.object(evaluation, "get_action_from_server",
                          side_effect=lambda *args: (events.append("inference") or actions)) as server, \
             patch.object(self.logger, "log_step", side_effect=record):
            evaluation._run_episode(cfg, robot, "test", "unused", 224, io.StringIO(),
                                    feedback_logger=self.logger, episode_index=3)
        self.assertEqual(events, ["observe", "feedback", "inference", "action",
                                  "observe", "feedback", "action"])
        self.assertEqual(server.call_count, 1)
        self.assertEqual(robot.send_action.call_count, 2)
        rows = list(csv.DictReader(io.StringIO(self.stream.getvalue())))
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[-1]["step"], "1")
        self.assertEqual(rows[-1]["episode"], "3")

    def test_disabled_logger_does_not_read_bus_and_dry_run_does_not_send(self):
        from experiments.robot.so101 import run_so101_eval as evaluation

        obs = dict(self.observation, front=np.zeros((2, 2, 3), dtype=np.uint8))
        robot = SimpleNamespace(bus=self.bus, get_observation=lambda: obs, send_action=Mock())
        cfg = evaluation.GenerateConfig(max_steps=1, dry_run=True, log_motor_feedback=False)
        actions = np.array([[obs[joint] for joint in JOINT_NAMES]])
        with patch("builtins.input", return_value=""), \
             patch.object(evaluation, "prepare_policy_observation", return_value={}), \
             patch.object(evaluation, "get_action_from_server", return_value=actions):
            evaluation._run_episode(cfg, robot, "test", "unused", 224, io.StringIO())
        self.bus.sync_read.assert_not_called()
        robot.send_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
