# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for native report reduction, task contracts and runner evidence."""

import json
import os
import statistics
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from geak.native import external_runner
from geak.native.external import (
    metadata_view,
    project_external_task,
    read_external_report,
    validate_objective,
)
from tests.native.fixtures import (
    canonical,
    evaluation_request,
    fixture,
    raw_reports,
    sha256,
)


class ExternalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = self.root / "task"
        self.contract = fixture(self.task)

    def test_complete_process_fold_reduction_and_secondary_regression(self):
        request = evaluation_request(self.contract, self.task)
        observed = read_external_report(request, raw_reports(request))
        self.assertEqual(observed["primary_speedup"], statistics.median([101/51, 101/21, 200/70]))
        self.assertEqual(set(observed["phase_speedups"]), {"extend", "draft"})
        self.assertTrue(observed["native_reproducible_gain"])
        self.assertFalse(observed["native_reproducible_regression"])

        regressed = read_external_report(request, raw_reports(request, draft_regression=True))
        self.assertGreater(regressed["primary_speedup"], 1)
        self.assertFalse(regressed["native_reproducible_gain"])
        self.assertTrue(regressed["native_reproducible_regression"])

    def test_incomplete_or_wrong_operation_cannot_be_a_result(self):
        request = evaluation_request(self.contract, self.task)
        for change in (lambda a: a.pop("process_03/ut_result.json"),
                       lambda a: a.update({"START.json": canonical({"parent_evaluation": {}})})):
            artifacts = raw_reports(request)
            change(artifacts)
            with self.assertRaises((ValueError, KeyError)):
                read_external_report(request, artifacts)
        artifacts = raw_reports(request)
        receipt = json.loads(artifacts["process_03/RECEIPT.json"])
        receipt["artifact_errors"] = ["stdout.log: unstable or unsafe file"]
        artifacts["process_03/RECEIPT.json"] = canonical(receipt)
        with self.assertRaises(ValueError):
            read_external_report(request, artifacts)

    def test_rejects_a_blended_objective(self):
        with self.assertRaises(ValueError):
            validate_objective(self.contract, {"kind": "geomean"})

    def test_runner_forwards_only_recorded_runtime_values(self):
        request = evaluation_request(self.contract, self.task, evaluation_id="runner")
        raw = raw_reports(request)
        forwarded = []
        def process(argv, **options):
            forwarded.append(dict(options["env"]))
            prefix = f"process_{len(forwarded):02d}/"
            for name in ("ut_result.json", "_runs_v2.jsonl"):
                (options["cwd"]/name).write_bytes(raw[prefix+name])
            options["stdout"].write(b"synthetic stdout\n")
            options["stderr"].write(b"synthetic stderr\n")
            return subprocess.CompletedProcess(argv, 0)
        allowed = {"PATH": "/opt/venv/bin:/usr/bin", "LD_LIBRARY_PATH": "/opt/rocm/lib",
                   "HIP_VISIBLE_DEVICES": "0", "GEAK_PHYSICAL_DEVICE_MAP": '[{"local_ordinal":0}]',
                   "GEAK_GPU_SYSFS_BDFS": "0000:01:00.0", "AITER_ROOT_DIR": "/results/cache/aiter_root",
                   "CXX": ""}
        inherited = {**allowed, "OPENAI_API_KEY": "private-test-value", "HF_TOKEN": "private-test-value",
                     "HTTPS_PROXY": "private-test-value", "HOME": "private-test-value",
                     "UNRELATED_WORKFLOW_FIELD": "private-test-value", "GLM_UT_REQUIRE_REPRODUCIBLE_GAIN": "1"}
        output = self.root/"runner"
        with mock.patch.dict(os.environ, inherited, clear=True), mock.patch.object(external_runner.subprocess, "run", side_effect=process):
            self.assertEqual(external_runner.run(request, self.root/"task", output), 0)
        expected = {**allowed, "GLM_UT_REQUIRE_REPRODUCIBLE_GAIN": "0"}
        self.assertEqual(forwarded, [expected]*3)
        start = json.loads((output/"START.json").read_bytes())
        self.assertEqual(start["environment"], expected)
        self.assertNotIn("CC", start["environment"])
        self.assertNotIn(b"private-test-value", (output/"START.json").read_bytes())
        for index in range(1, 4):
            leg = output/f"process_{index:02d}"
            receipt = json.loads((leg/"RECEIPT.json").read_bytes())
            for name in ("ut_result.json", "_runs_v2.jsonl", "stdout.log", "stderr.log"):
                self.assertEqual(receipt[name+"_sha256"], sha256((leg/name).read_bytes()))

    def test_runner_rejects_symlinked_ut_reports_without_copying_target(self):
        target = self.root/"unrelated-private-file"
        target.write_bytes(b"private target must not be retained")
        for index, unsafe in enumerate(("ut_result.json", "_runs_v2.jsonl")):
            with self.subTest(unsafe=unsafe):
                request = evaluation_request(self.contract, self.task, evaluation_id=f"unsafe-{index}")
                raw = raw_reports(request)
                def process(argv, **options):
                    for name in ("ut_result.json", "_runs_v2.jsonl"):
                        path = options["cwd"]/name
                        if name == unsafe:
                            path.symlink_to(target)
                        else:
                            path.write_bytes(raw["process_01/"+name])
                    return subprocess.CompletedProcess(argv, 0)
                output = self.root/f"unsafe-{index}"
                with mock.patch.object(external_runner.subprocess, "run", side_effect=process) as run:
                    self.assertEqual(external_runner.run(request, self.root/"task", output), 1)
                self.assertEqual(run.call_count, 1)
                self.assertFalse((output/"process_01"/unsafe).exists())
                receipt = json.loads((output/"process_01/RECEIPT.json").read_bytes())
                self.assertTrue(any(error.startswith(unsafe+": ") for error in receipt["artifact_errors"]))
                self.assertNotIn(b"private target must not be retained", (output/"process_01/RECEIPT.json").read_bytes())

    def test_absent_tolerance_and_untimed_case_are_preserved(self):
        meta = json.loads((self.root/"task/meta.json").read_text())
        del meta["rms_tol"]
        meta["cases"].append({"name": "edge", "timed": False})
        view = metadata_view(meta)
        self.assertNotIn("rms_tol", view)
        self.assertEqual(view["cases"][-1], {"name": "edge", "timed": False})
        contract = project_external_task(meta, artifacts=self.contract["artifacts"],
            original_source_map=self.contract["original_source_map"], source=self.contract["source"],
            runtime=self.contract["runtime"], primary_regime="extend")
        self.assertIn("edge", contract["correctness_case_ids"])
        self.assertNotIn("edge", contract["benchmark_case_ids"])


if __name__ == "__main__":
    unittest.main()
