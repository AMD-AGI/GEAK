import os
import sys
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import run_semantic_shape_capture as capture


class RunSemanticShapeCaptureTest(unittest.TestCase):
    def test_required_phases_are_inferred_from_capture_plan(self):
        plan = {
            "target_buckets": [
                {"phase": "decode"},
                {"phase": "prefill"},
                {"phase": "decode"},
            ]
        }
        self.assertEqual(
            capture._required_phases(plan), ["decode", "prefill"])

    def test_injects_eager_flag_for_inline_server_arguments(self):
        text = "launch --disable-radix-cache $EVAL_CONTEXT_ARGS"
        self.assertIn(
            "--disable-radix-cache --disable-cuda-graph",
            capture._with_disable_cuda_graph(text))

    def test_injects_eager_flag_for_multiline_server_arguments(self):
        text = (
            "launch \\\n"
            "  --disable-radix-cache \\\n"
            "  --max-prefill-tokens 32768")
        result = capture._with_disable_cuda_graph(text)
        self.assertIn(
            "--disable-radix-cache --disable-cuda-graph \\", result)

    def test_existing_eager_flag_is_idempotent(self):
        text = "--disable-radix-cache --disable-cuda-graph"
        self.assertEqual(capture._with_disable_cuda_graph(text), text)

    def test_stop_service_matches_space_and_equals_port_forms(self):
        with mock.patch.object(capture, "_docker") as docker:
            capture._stop_service("container", 8935)
        command = docker.call_args[0][1]
        self.assertIn("--port(=| )8935", command)
        self.assertIn("pkill -TERM", command)
        self.assertIn("pkill -KILL", command)


if __name__ == "__main__":
    unittest.main()
