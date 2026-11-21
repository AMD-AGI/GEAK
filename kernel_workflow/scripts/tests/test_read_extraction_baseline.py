import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "read_extraction_baseline.py"
spec = importlib.util.spec_from_file_location("read_extraction_baseline", SCRIPT)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class ExtractionBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        (self.workspace / "run").mkdir(parents=True)
        self.rows = [
            {"sig": "verify_M6", "baseline_ms": 2.0, "optimized_ms": 1.9},
            {"sig": "draft_M1", "baseline_ms": 1.0, "optimized_ms": 1.1},
        ]
        self.write("meta.json", {"workload": {"cases": [{"sig": r["sig"]} for r in self.rows]}})

    def write(self, relative, value):
        (self.workspace / relative).write_text(json.dumps(value))

    def result(self, timing):
        self.write("run/ut_result.json", {
            "exit_code": 0, "ok": True,
            "gates": {"correctness": {"ok": True}, "timing": {"ok": True, **timing}},
        })

    def test_inline_and_referenced_extraction_formats(self):
        formats = [
            ({"per_case": self.rows}, None, None),
            ({"report": "run/ut_timing.json"}, "run/ut_timing.json", {"per_case": self.rows}),
            ({"record": "run/timing_per_case.json"}, "run/timing_per_case.json", self.rows),
        ]
        for timing, path, report in formats:
            with self.subTest(timing=timing):
                self.result(timing)
                if path:
                    self.write(path, report)
                out = reader.read_baseline(self.workspace)
                self.assertTrue(out["usable"])
                self.assertEqual(out["next_run"], "correctness")
                self.assertEqual(out["per_case"], self.rows)
                self.assertIn(path or "run/ut_result.json", out["source"])

    def test_missing_extraction_requires_full_run_then_accepts_fresh_result(self):
        out = reader.read_baseline(self.workspace)
        self.assertFalse(out["usable"])
        self.assertEqual(out["next_run"], "full")
        self.result({"per_case": self.rows})
        self.assertTrue(reader.read_baseline(self.workspace)["usable"])

    def test_missing_or_broken_reference_requires_full_run(self):
        self.result({"report": "run/ut_timing.json"})
        for text in (None, "invalid JSON", "{}", "[]"):
            with self.subTest(text=text):
                if text is not None:
                    (self.workspace / "run/ut_timing.json").write_text(text)
                self.assertEqual(reader.read_baseline(self.workspace)["next_run"], "full")

    def test_empty_inline_result_can_use_complete_sidecar(self):
        self.result({"per_case": [], "record": "run/timing_per_case.json"})
        self.write("run/timing_per_case.json", self.rows)
        self.assertEqual(reader.read_baseline(self.workspace)["per_case"], self.rows)

    def test_incomplete_duplicate_and_invalid_timings_require_full_run(self):
        invalid = [self.rows[:1], [self.rows[0], self.rows[0]]]
        for field in ("baseline_ms", "optimized_ms"):
            for value in (None, 0, -1, float("nan"), float("inf"), True, "1.0"):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                invalid.append(rows)
        for rows in invalid:
            with self.subTest(rows=rows):
                self.result({"per_case": rows})
                self.assertEqual(reader.read_baseline(self.workspace)["next_run"], "full")

    def test_failed_run_or_gate_cannot_supply_baseline(self):
        self.result({"per_case": self.rows})
        result = json.loads((self.workspace / "run/ut_result.json").read_text())
        for failure in ("exit", "correctness", "timing"):
            with self.subTest(failure=failure):
                bad = copy.deepcopy(result)
                if failure == "exit":
                    bad["exit_code"] = 1
                else:
                    bad["gates"][failure]["ok"] = False
                self.write("run/ut_result.json", bad)
                self.assertEqual(reader.read_baseline(self.workspace)["next_run"], "full")

    def test_preserved_baseline_survives_correctness_overwriting_ut_result(self):
        self.result({"per_case": self.rows})
        saved = self.workspace.parent / "extraction_baseline.json"
        subprocess.run([sys.executable, str(SCRIPT), "--workspace", str(self.workspace),
                        "--out", str(saved)], check=True, capture_output=True)
        self.write("run/ut_result.json", {
            "exit_code": 1, "gates": {"timing": {"ok": False, "skipped": True}},
        })
        self.assertEqual(json.loads(saved.read_text())["per_case"], self.rows)
        self.assertEqual(reader.read_baseline(self.workspace)["next_run"], "full")

    def test_reference_to_old_workspace_is_not_reused(self):
        stale = self.workspace.parent / "old_timings.json"
        stale.write_text(json.dumps(self.rows))
        self.result({"record": str(stale)})
        self.assertEqual(reader.read_baseline(self.workspace)["next_run"], "full")


if __name__ == "__main__":
    unittest.main()
