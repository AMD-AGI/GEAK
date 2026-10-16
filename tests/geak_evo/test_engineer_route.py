"""Regression checks for the documented Oct 15 Claude route forwarding defect."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import test_engineer_port as base


class EngineerRouteTests(unittest.TestCase):
    capture = base.EngineerTests.capture

    def setUp(self):
        base.EngineerTests.setUp(self)
        self.route = {"schema": "geak-native-model-route-v1", "workflow_backend": "claude",
                      "wire_protocol": "anthropic_messages", "requested_model": "fixture-model",
                      "accepted_reported_models": ["fixture-model", "fixture-reported-model"],
                      "requested_reasoning_effort": "ultracode"}
        self.options = {"executable": "/fixture/claude", "session_directory": str(self.root / "session"),
                        "model_route": self.route, "model_profile": base.profile(self.root, route=self.route, backend="claude")}

    def execute(self, *, settled=True):
        events = [{"type": "system", "subtype": "init", "session_id": base.THREAD,
                   "model": "fixture-model", "tools": ["Bash"]},
                  {"type": "result", "subtype": "success", "is_error": False,
                   "session_id": base.THREAD, "result": "Unreviewed fixture report.",
                   "modelUsage": {"fixture-reported-model": {"inputTokens": 2}},
                   "total_cost_usd": 0.125,
                   "usage": {"input_tokens": 2, "output_tokens": 3,
                             "cache_read_input_tokens": 5, "cache_creation_input_tokens": 0}}]
        raw = "\n".join(json.dumps(event) for event in events) + "\n"
        def launch(command, **kwargs):
            self.assertEqual(command[command.index("--model") + 1], "fixture-model")
            return subprocess.Popen([sys.executable, "-B", "-c", "print(" + repr(raw) + ", end='')"], **kwargs)
        def settle(process, attempt):
            process.wait(timeout=3)
            return {"settled": settled, "scope": "CPU child only"}
        return base.run_turn(self.request, self.attempts, backend="claude", backend_options=self.options,
                             launch_worker=launch, settle_worker=settle, capture_tip=self.capture)

    def test_explicit_claude_route_reaches_event_inspection(self):
        result = self.execute()
        self.assertTrue(result["receipt"]["transport_complete"], result["receipt"]["protocol_errors"])
        self.assertEqual(result["receipt"]["model_route"], self.route)
        self.assertEqual(result["receipt"]["reported_models"], ["fixture-model", "fixture-reported-model"])
        self.assertEqual(result["receipt"]["native_total_cost_usd"], 0.125)
        self.assertIsNone(result["receipt"]["monetary_cost"])
        self.assertEqual((Path(result["attempt_dir"]) / "report.md").read_text(), "Unreviewed fixture report.")

    def test_recovery_uses_saved_route_and_preserves_native_cost(self):
        result = self.execute(settled=False)
        original_bytes = Path(result["receipt_ref"]["path"]).read_bytes()
        self.route["accepted_reported_models"] = ["mutated-caller-value"]
        recovered = base.recover_turn(result["receipt_ref"], capture_tip=self.capture,
            verify_recovery=lambda *args: {"settled": True, "process_reaped": True, "scope": "CPU child only"})
        self.assertTrue(recovered["receipt"]["transport_complete"])
        self.assertEqual(recovered["receipt"]["native_total_cost_usd"], 0.125)
        self.assertEqual(recovered["receipt"]["model_route"]["accepted_reported_models"],
                         ["fixture-model", "fixture-reported-model"])
        self.assertEqual(Path(result["receipt_ref"]["path"]).read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
