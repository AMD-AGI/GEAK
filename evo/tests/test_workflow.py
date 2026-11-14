import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node") or str(Path.home() / ".local/bin/node")


@unittest.skipUnless(Path(NODE).is_file() and shutil.which("git"), "Node and Git required for CPU witness")
class WorkflowWitness(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="evo-workflow-witness-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.canonical = self.folder / "canonical"
        self.canonical.mkdir()
        (self.folder / "eval").mkdir()
        (self.canonical / "kernel.py").write_text("value = 2\n")
        (self.canonical / "oracle.py").write_text("immutable oracle\n")
        self.environment = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        self.environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0")
        for arguments in (["init", "--quiet", "--template="], ["add", "kernel.py", "oracle.py"]):
            subprocess.run(["git", "-C", str(self.canonical), *arguments], check=True,
                           capture_output=True, env=self.environment)

    def run_witness(self, scenario):
        return subprocess.run([NODE, str(ROOT / "evo/tests/workflow_witness.cjs"),
                               str(self.folder), scenario], capture_output=True, text=True,
                              env=self.environment, timeout=90)

    def test_real_main_materializer_archive_and_older_parent(self):
        result = self.run_witness("reuse")
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["errors"], [])
        self.assertEqual(receipt["archive"], str(self.folder / "eval/evolution_archive"))
        self.assertIn("r1_d0-best", receipt["entries"])
        self.assertIn("r1_d0-development", receipt["entries"])
        self.assertIn("r2_d0-development", receipt["entries"])
        self.assertEqual(receipt["calls"].count("evo:init"), 1)
        self.assertEqual(receipt["calls"].count("evo:prepare"), 2)
        self.assertEqual(receipt["calls"].count("evo:check-patch"), 1)
        self.assertEqual(receipt["calls"].count("evo:record-batch"), 2)
        self.assertEqual(receipt["calls"].count("evo:query"), 1)

    def test_success_exit_cannot_hide_failed_receipt(self):
        result = self.run_witness("false-receipt")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid receipt", result.stderr)

    def test_real_helper_error_preserves_stdout_diagnostic(self):
        result = self.run_witness("helper-error")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EVO query did not complete: unknown_source: Unknown entry: missing-parent", result.stderr)

    def test_nonzero_stdout_diagnostic_takes_precedence_over_stderr(self):
        result = self.run_witness("structured-error")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EVO init did not complete: init_conflict: Initialization differs from captured founder",
                      result.stderr)

    def test_malformed_or_absent_result_still_fails(self):
        for scenario, diagnostic in (
            ("missing-result", "EVO init did not complete: no return"),
            ("missing-stdout", "EVO init did not complete: command failed"),
            ("non-string-stdout", "EVO init did not complete: command failed"),
            ("malformed-stdout", "EVO init returned an invalid receipt"),
            ("empty-stdout", "EVO init returned an invalid receipt"),
            ("null-receipt", "EVO init returned an invalid receipt"),
        ):
            with self.subTest(scenario=scenario):
                result = self.run_witness(scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_nonzero_without_valid_diagnostic_preserves_failure(self):
        for scenario, diagnostic in (
            ("nonzero-malformed-stdout", "helper crashed"),
            ("nonzero-missing-stdout", "helper crashed"),
            ("nonzero-invalid-diagnostic", "helper crashed"),
            ("nonzero-success-receipt", "command failed"),
        ):
            with self.subTest(scenario=scenario):
                result = self.run_witness(scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("EVO init did not complete: " + diagnostic, result.stderr)


if __name__ == "__main__":
    unittest.main()
