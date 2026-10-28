"""CPU regression checks for the server boundary; no model or robot is loaded."""

import copy
import json
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


class ServerValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["test", "--robot_platform", "so101"]):
            namespace = runpy.run_path(str(Path(__file__).resolve().parents[3] / "vla-scripts/deploy.py"))
        cls.server_class = namespace["OpenVLAServer"]
        cls.http_error = namespace["HTTPException"]

    def setUp(self):
        self.server = self.server_class.__new__(self.server_class)
        self.server.cfg = SimpleNamespace(use_proprio=True, num_images_in_input=3, use_film=False, diagnostic_log="")
        self.server.vla = self.server.processor = self.server.action_head = self.server.proprio_projector = None
        self.payload = {"instruction": "Pick up the block and place it into the cup.",
                        "state": np.zeros(6, dtype=np.float32),
                        "full_image": np.zeros((224, 224, 3), dtype=np.uint8),
                        "additional_images": [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]}
        self.actions = np.arange(180, dtype=np.float32).reshape(30, 6) / 100

    def reject_before_inference(self, payload):
        inference = Mock(return_value=list(self.actions))
        with patch.dict(self.server.get_server_action.__globals__, get_vla_action=inference):
            with self.assertRaises(self.http_error) as error:
                self.server.get_server_action(payload)
        self.assertEqual(error.exception.status_code, 422)
        inference.assert_not_called()

    def test_invalid_states_cannot_broadcast_or_clip_into_valid_inputs(self):
        for state in (None, [0], np.zeros((1, 6)), [0] * 5, [float("nan")] * 6,
                      [float("inf")] * 6, [float("-inf")] * 6, ["bad"] * 6):
            with self.subTest(state=repr(state)):
                self.reject_before_inference(dict(self.payload, state=state))

    def test_missing_inputs_and_invalid_images_do_not_call_the_model(self):
        for field in ("instruction", "state", "full_image"):
            payload = copy.deepcopy(self.payload)
            del payload[field]
            self.reject_before_inference(payload)
        for instruction in ("", "  ", None, 5):
            self.reject_before_inference(dict(self.payload, instruction=instruction))
        for image in (np.zeros((0, 224, 3), dtype=np.uint8), np.zeros((224, 224, 3)),
                      np.zeros((224, 224), dtype=np.uint8)):
            self.reject_before_inference(dict(self.payload, full_image=image))
        self.reject_before_inference(dict(self.payload, additional_images=[]))
        self.reject_before_inference({"encoded": "not JSON"})
        self.reject_before_inference({"encoded": "[]"})

    def test_valid_plain_and_double_encoded_clients_keep_their_response_format(self):
        for encoded in (False, True):
            with self.subTest(encoded=encoded):
                payload = {"encoded": json.dumps(self.payload)} if encoded else self.payload
                with patch.dict(self.server.get_server_action.__globals__,
                                get_vla_action=Mock(return_value=list(self.actions))):
                    response = self.server.get_server_action(payload)
                self.assertEqual(response.status_code, 200)
                body = json.loads(response.body)
                if encoded:
                    self.assertIsInstance(body, str)
                    body = json.loads(body)
                np.testing.assert_array_equal(np.asarray(body), self.actions)

    def test_nonfinite_or_partial_model_output_never_returns_actions(self):
        for action in (np.full((30, 6), np.nan), np.full((30, 6), np.inf), np.zeros((1, 6))):
            with patch.dict(self.server.get_server_action.__globals__,
                            get_vla_action=Mock(return_value=list(action))):
                with self.assertLogs(level="ERROR"), self.assertRaises(self.http_error) as error:
                    self.server.get_server_action(self.payload)
                self.assertEqual(error.exception.status_code, 500)

    def test_internal_failure_is_an_http_error(self):
        with patch.dict(self.server.get_server_action.__globals__,
                        get_vla_action=Mock(side_effect=RuntimeError("test failure"))):
            with self.assertLogs(level="ERROR"), self.assertRaises(self.http_error) as error:
                self.server.get_server_action(self.payload)
            self.assertEqual(error.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
