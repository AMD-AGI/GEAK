import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import validate_kpu_model_pair as pair


class ValidateKpuModelPairTest(unittest.TestCase):
    def _manifest(self, root, name, reason=True):
        path = os.path.join(root, name)
        unavailable = [{
            "reason_code": "runtime_internal_buffer_operation",
            "reason": "runtime operation",
        }] if reason else [{"reason_code": "", "reason": ""}]
        with open(path, "w") as fh:
            json.dump({
                "status": "pass",
                "classification_complete": True,
                "row_count": 3,
                "evidence_counts": {"K": 1, "P": 1, "U": 1},
                "unavailable": unavailable,
            }, fh)
        return path

    def test_requires_both_complete_model_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pair.validate(
                self._manifest(tmp, "dsr1.json"),
                self._manifest(tmp, "qwen35.json"),
                os.path.join(tmp, "result.json"))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                [item["model"] for item in result["models"]],
                ["dsr1", "qwen35"])

    def test_fails_if_one_model_has_unexplained_u(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = pair.validate(
                self._manifest(tmp, "dsr1.json"),
                self._manifest(tmp, "qwen35.json", reason=False),
                os.path.join(tmp, "result.json"))
            self.assertEqual(result["status"], "fail")


if __name__ == "__main__":
    unittest.main()
