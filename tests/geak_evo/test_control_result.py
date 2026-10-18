"""CPU-only control readback; native evidence parsing stays in result.py."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from geak_evo import control_result
from geak_evo._util import sha256_hex
from geak_evo.evaluate.evaluator import ArenaEvaluator


class ControlResultTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.source_name = "kernel_src/module.py"
        self.source = self.workspace / self.source_name
        self.source.parent.mkdir(parents=True)
        self.before = b"value = 1\n"
        self.after = b"value = 2\n"
        self.source.write_bytes(self.before)
        self.source.chmod(0o644)
        self.evaluator = object.__new__(ArenaEvaluator)
        self.evaluator.contract = {
            "schema": "geak-arena-external-task-v1", "source_paths": [self.source_name],
            "benchmark_case_ids": ["prefill"], "report_contract": {"primary_regime": "extend"},
            "runtime": {"image": "example@sha256:" + "a" * 64, "config_digest": "sha256:" + "b" * 64}}
        self.record = {
            "request": {"contract": self.evaluator.contract, "operation_id": "evaluation",
                        "binding": {"candidate_id": "candidate"},
                        "source": {"files": [{"path": self.source_name,
                            "blob_id": sha256_hex(self.before), "executable": 0}]}},
            "receipt": {"settled": True, "native_suite_pass_observed": True,
                        "collection_errors": [], "host_errors": [],
                        "measurement_valid": False, "custody_proven": False, "reward_eligible": False,
                        "observed_runtime": {**self.evaluator.contract["runtime"],
                                             "physical_devices": ["synthetic-device"]}},
            "receipt_ref": {"path": "/retained/arena_evaluation.json", "sha256": "c" * 64},
            "timing_context": {"timing_basis": "external_whole_call_ut", "request_id": "evaluation",
                "evidence": {"observer/START.json": "d" * 64},
                "diagnostics": {"primary_regime": "extend", "primary_speedup": 1.01,
                    "native_reproducible_gain": False, "native_reproducible_regression": False,
                    "phase_speedups": {"extend": {"processes": [1.01, 1.02, 1.0], "median": 1.01}}},
                "paired_cases": {"prefill": {"baseline_ms": 2., "optimized_ms": 2., "speedup": 1.}}}}
        self.patch = self.root / "candidate.diff"
        self.patch.write_bytes(b"diff --git a/kernel_src/module.py b/kernel_src/module.py\n"
                               b"--- a/kernel_src/module.py\n+++ b/kernel_src/module.py\n"
                               b"@@ -1 +1 @@\n-value = 1\n+value = 2\n")

    def read(self, *, patch=None, record=None):
        argv = ["--candidate-id", "candidate", "--evaluation-id", "evaluation",
                "--workspace", str(self.workspace)]
        if patch is not None:
            argv.extend(["--patch", str(patch)])
        stdout = io.StringIO()
        with mock.patch.object(control_result.result, "read_evaluation", return_value=record or self.record) as reader:
            with contextlib.redirect_stdout(stdout):
                try:
                    rc = control_result.main(argv, evaluator=self.evaluator)
                except Exception:
                    self.assertEqual(stdout.getvalue(), "")
                    raise
        reader.assert_called_once_with(self.evaluator, "candidate", evaluation_id="evaluation")
        self.assertEqual(rc, 0)
        return json.loads(stdout.getvalue())

    def test_valid_no_gain_keeps_native_primary_and_no_qualification_claim(self):
        value = self.read()
        self.assertEqual(value["primary_speedup"], 1.01)
        self.assertFalse(value["native_reproducible_gain"])
        self.assertEqual(value["per_case"][0]["speedup"], 1.)
        self.assertEqual(value["objective"], {"kind": "external_primary", "primary_regime": "extend"})
        self.assertEqual(value["external_evaluation"], {"candidate_id": "candidate", "evaluation_id": "evaluation"})
        self.assertEqual(value["receipt_ref"], self.record["receipt_ref"])
        self.assertTrue(value["external_measurement_valid"])
        self.assertTrue({"measurement_valid", "custody_proven", "reward_eligible", "framework_final_pass"}.isdisjoint(value))

    def test_patch_matches_evaluated_candidate_without_changing_workspace(self):
        self.record["request"]["source"]["files"][0]["blob_id"] = sha256_hex(self.after)
        self.read(patch=self.patch)
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertEqual(list(self.workspace.rglob("*.py")), [self.source])

    def test_workspace_content_or_mode_mismatch_is_rejected(self):
        for raw, mode in ((self.after, 0o644), (self.before, 0o755)):
            with self.subTest(raw=raw, mode=mode):
                self.source.write_bytes(raw)
                self.source.chmod(mode)
                with self.assertRaisesRegex(ValueError, "source differs"):
                    self.read()

    def test_patch_result_mismatch_and_closure_deletion_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.read(patch=self.patch)
        self.patch.write_bytes(b"diff --git a/kernel_src/module.py b/kernel_src/module.py\n"
                               b"deleted file mode 100644\n--- a/kernel_src/module.py\n+++ /dev/null\n"
                               b"@@ -1 +0,0 @@\n-value = 1\n")
        with self.assertRaisesRegex(ValueError, "closure differs"):
            self.read(patch=self.patch)
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_patch_outside_closure_and_rename_source_are_rejected(self):
        for patch in (
            b"diff --git a/ut_main.py b/ut_main.py\nnew file mode 100644\n"
            b"--- /dev/null\n+++ b/ut_main.py\n@@ -0,0 +1 @@\n+changed judge\n",
            b"diff --git a/ut_main.py b/kernel_src/module.py\nsimilarity index 100%\n"
            b"rename from ut_main.py\nrename to kernel_src/module.py\n",
        ):
            with self.subTest(patch=patch):
                self.patch.write_bytes(patch)
                with self.assertRaisesRegex(ValueError, "outside the source closure"):
                    self.read(patch=self.patch)
        self.assertFalse((self.workspace / "ut_main.py").exists())

    def test_empty_patch_and_mode_only_patch(self):
        self.patch.write_bytes(b"")
        self.read(patch=self.patch)
        self.patch.write_bytes(b"diff --git a/kernel_src/module.py b/kernel_src/module.py\n"
                               b"old mode 100644\nnew mode 100755\n")
        self.record["request"]["source"]["files"][0]["executable"] = 0o111
        self.read(patch=self.patch)
        self.assertEqual(self.source.stat().st_mode & 0o111, 0)

    def test_missing_settlement_timing_or_runtime_fails_without_stdout(self):
        changes = (
            lambda r: r["receipt"].update(settled=False),
            lambda r: r["receipt"].update(native_suite_pass_observed=False),
            lambda r: r["receipt"].update(host_errors=["missing cleanup"]),
            lambda r: r.update(timing_context=None),
            lambda r: r["timing_context"].update(evidence={}),
            lambda r: r["receipt"]["observed_runtime"].update(image=None),
            lambda r: r["timing_context"]["diagnostics"].update(primary_regime="decode"),
        )
        for change in changes:
            record = copy.deepcopy(self.record)
            change(record)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read(record=record)

    def test_symlink_source_or_patch_is_rejected(self):
        link = self.root / "linked.diff"
        link.symlink_to(self.patch)
        with self.assertRaises(OSError):
            self.read(patch=link)
        self.source.unlink()
        self.source.symlink_to(self.patch)
        with self.assertRaises(OSError):
            self.read()


if __name__ == "__main__":
    unittest.main()
