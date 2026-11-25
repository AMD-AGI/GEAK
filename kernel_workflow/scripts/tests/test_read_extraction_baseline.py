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


    def generated(self, rows=None, world=1, exit_code=0, **gates):
        """run/ut_result.json as the generated unittest.py (ut_runner.py) writes it: every gate is a
        string (a gate given as None is absent) and the timed rows are under the top-level "timing".
        The workload gives every timed case call mass, as an observed case has."""
        self.write("meta.json", {"workload": {"cases": [
            {"sig": r["sig"], "regime": "decode", "effective_calls": 100} for r in self.rows]}})
        gates = {"contract": "pass", "correctness": "pass", "timing": "pass", "receipt": "pass", **gates}
        self.write("run/ut_result.json", {
            "exit_code": exit_code, "ok": exit_code == 0, "generated_by": "ut_runner.py v1",
            "world_size": world, "required_gpu_count": world,
            "gates": {k: v for k, v in gates.items() if v is not None},
            "timing": {"per_case": self.rows if rows is None else rows, "weighted": 1.01},
        })

    def test_generated_ut_result_is_usable(self):
        self.generated()
        out = reader.read_baseline(self.workspace)
        self.assertTrue(out["usable"], out)
        self.assertEqual(out["next_run"], "correctness")
        self.assertEqual(out["source"], "run/ut_result.json:timing.per_case")
        self.assertEqual(out["per_case"], self.rows)
        self.assertNotIn("note", out)

    def test_generated_multi_rank_result_asks_for_one_full_run(self):
        # The saved rows are usable, but a multi-rank lane owes per-rank times of a run of its own.
        self.generated(world=4)
        out = reader.read_baseline(self.workspace)
        self.assertTrue(out["usable"], out)
        self.assertEqual(out["next_run"], "full")
        self.assertEqual(out["per_case"], self.rows)
        self.assertIn("per-rank", out["note"])

    def test_generated_ut_need_not_time_a_case_without_served_call_mass(self):
        # generate_ut.py asks for a timing case only where an observed case has call mass in a served regime.
        self.generated()
        meta = json.loads((self.workspace / "meta.json").read_text())
        meta["workload"]["served_regimes"] = ["decode"]
        meta["workload"]["cases"] += [
            {"sig": "idle_M9", "regime": "decode", "effective_calls": 0},
            {"sig": "eager_M2", "regime": "prefill_eager", "effective_calls": 7},
            {"sig": "unweighted_M3", "regime": "decode"},
        ]
        self.write("meta.json", meta)
        out = reader.read_baseline(self.workspace)
        self.assertTrue(out["usable"], out)
        self.assertEqual(out["per_case"], self.rows)

    def test_generated_ut_result_of_a_failed_run(self):
        shapes = [
            dict(exit_code=1, correctness="fail", timing=None, receipt=None),   # stopped at correctness
            dict(exit_code=3, timing="incomplete", receipt=None),               # timing did not finish
            dict(exit_code=2, timing="env", receipt=None),                      # environment fault while timing
            dict(exit_code=1, timing=None, receipt="fail"),                     # the receipt check failed
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                self.generated(**shape)
                out = reader.read_baseline(self.workspace)
                self.assertFalse(out["usable"])
                self.assertEqual(out["next_run"], "full")

    def test_generated_ut_result_with_a_failed_gate(self):
        # An inconsistent record (exit code 0 beside a gate that is not "pass") is not trusted either.
        for gate in ("contract", "correctness", "receipt"):
            for value in ("fail", "env", "incomplete"):
                with self.subTest(gate=gate, value=value):
                    self.generated(**{gate: value})
                    out = reader.read_baseline(self.workspace)
                    self.assertFalse(out["usable"])
                    self.assertEqual(out["next_run"], "full")

    def test_generated_ut_result_missing_a_workload_case(self):
        self.generated(rows=self.rows[:1])
        out = reader.read_baseline(self.workspace)
        self.assertFalse(out["usable"])
        self.assertEqual(out["next_run"], "full")

    def test_hand_written_result_with_a_string_among_its_gates_is_read_as_before(self):
        self.write("run/ut_result.json", {
            "exit_code": 0, "ok": True,
            "gates": {"correctness": {"ok": True}, "timing": {"ok": True, "per_case": self.rows},
                      "profile": "skipped"},
        })
        out = reader.read_baseline(self.workspace)
        self.assertTrue(out["usable"], out)
        self.assertEqual(out["source"], "run/ut_result.json:gates.timing.per_case")


if __name__ == "__main__":
    unittest.main()
