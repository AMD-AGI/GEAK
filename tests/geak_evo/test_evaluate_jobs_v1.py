"""CPU-only contract fixtures, never represented as target workload captures."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from geak_evo.evaluate.jobs_v1 import job_digest, load_job, validate_job


def fixture():
    files = {
        "source/baseline.py": b"def complete_call(x):\n    return x\n",
        "binding.json": b'{"entry":"source/baseline.py:complete_call"}',
        "framework.json": b'{"kind":"cpu_test_fixture"}',
        "workload.json": b'{"model":"cpu-fixture","tp":1}',
        "capture.json": b'{"fixture_events":["case-1","case-1"]}',
        "inputs.json": b'{"x":[1,2],"scale":1}',
        "numerical.json": b'{"comparison":"exact against captured baseline"}',
        "timing.json": b'{"timer":"fixture only","synchronize":"before and after"}',
        "graph.json": b'{"mode":"disabled"}',
        "state.json": b'{"counter":0}',
        "state_contract.json": b'{"reset":"before each call","compare":"output and counter"}',
        "inference.json": b'{"basis":"fixture extrapolation only"}',
    }
    job = {
        "schema_version": 1, "job_version": 1,
        "workload": {
            "model": "cpu-fixture/identity", "model_revision": "1" * 40,
            "framework_source": {"repository": "fixture://framework", "version": "fixture-v1",
                                 "revision": "2" * 40, "manifest_ref": "framework.json"},
            "device": "CPU fixture", "tp": 1, "isl_tokens": 2, "osl_tokens": 1,
            "concurrency": 1, "precision": "int64"},
        "accepted_stack_digest": "3" * 64,
        "callable": {"source_path": "source/baseline.py", "qualname": "complete_call",
                     "binding_ref": "binding.json", "stateful": False},
        "baseline": {"source_paths": ["source/baseline.py"], "support_paths": ["binding.json"]},
        "artifacts": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "provenance": {"kind": "test_fixture", "capture_ref": "capture.json",
                       "workload_ref": "workload.json"},
        "cases": [{"case_id": "case-1", "phase": "fixture", "rank": 0,
                   "tensors": {"x": {"shape": [2], "dtype": "int64", "strides": [1],
                                     "stride_unit": "elements", "layout": "contiguous"}},
                   "scalars": {"scale": 1}, "inputs_ref": "inputs.json", "state_refs": [],
                   "routing_refs": [],
                   "observed_frequency": {"count": 2, "unit": "calls", "provenance_ref": "capture.json"},
                   "inferred_frequency": None}],
        "numerical": {"reference": "baseline", "case_ids": ["case-1"],
                      "protocol_ref": "numerical.json", "state_contract_ref": None},
        "timing": {"case_ids": ["case-1"], "unit": "us", "protocol_ref": "timing.json",
                   "warmup_calls": 1, "measured_calls": 2, "aggregation": "median",
                   "scope": "complete_callable"},
        "graph": {"mode": "disabled", "protocol_ref": "graph.json"},
    }
    return job, files


class JobTests(unittest.TestCase):
    def setUp(self):
        self.job, self.files = fixture()

    def test_canonical_identity_and_detached_mapping(self):
        validated = validate_job(self.job)
        digest = job_digest(validated)
        self.assertEqual(digest, job_digest(dict(reversed(list(self.job.items())))))
        self.job["cases"][0]["tensors"]["x"]["shape"][0] = 3
        self.assertEqual(digest, job_digest(validated))
        self.assertNotEqual(digest, job_digest(self.job))

    def test_every_contract_component_changes_identity(self):
        edits = [
            (("workload", "model_revision"), "4" * 40),
            (("workload", "framework_source", "revision"), "5" * 40),
            (("workload", "tp"), 8), (("workload", "precision"), "fp32"),
            (("accepted_stack_digest",), "6" * 64),
            (("callable", "qualname"), "other_complete_call"),
            (("artifacts", "source/baseline.py"), "7" * 64),
            (("artifacts", "binding.json"), "8" * 64),
            (("cases", 0, "scalars", "scale"), 2),
            (("cases", 0, "observed_frequency", "count"), 3),
            (("cases", 0, "phase"), "decode"),
            (("cases", 0, "routing_refs"), ["inputs.json"]),
            (("numerical", "protocol_ref"), "state_contract.json"),
            (("timing", "unit"), "ns"), (("graph", "mode"), "required"),
        ]
        original = job_digest(self.job)
        for path, value in edits:
            with self.subTest(path=path):
                changed = copy.deepcopy(self.job)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                self.assertNotEqual(original, job_digest(changed))

    def test_candidate_identity_is_separate(self):
        self.job["candidate"] = {"source": "other.py"}
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            validate_job(self.job)

    def test_missing_reference_and_source_closure(self):
        del self.job["artifacts"]["inputs.json"]
        with self.assertRaisesRegex(ValueError, "missing artifact reference"):
            validate_job(self.job)
        self.job, _ = fixture()
        self.job["baseline"]["support_paths"] = []
        with self.assertRaisesRegex(ValueError, "binding missing"):
            validate_job(self.job)

    def test_inference_never_supplies_observed_frequency(self):
        frequency = {"count": 1000.5, "unit": "calls", "basis_ref": "inference.json"}
        self.job["cases"][0]["inferred_frequency"] = frequency
        validated = validate_job(self.job)
        self.assertEqual(validated["cases"][0]["observed_frequency"]["count"], 2)
        self.job["cases"][0]["observed_frequency"]["count"] = 0
        with self.assertRaises(ValueError):
            validate_job(self.job)

    def test_duplicate_case_and_missing_coverage(self):
        self.job["cases"].append(copy.deepcopy(self.job["cases"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate case ID"):
            validate_job(self.job)
        self.job, _ = fixture()
        self.job["numerical"]["case_ids"] = []
        with self.assertRaisesRegex(ValueError, "every observed case"):
            validate_job(self.job)

    def test_stateful_contract_requires_state_and_reset_reference(self):
        self.job["callable"]["stateful"] = True
        with self.assertRaises(ValueError):
            validate_job(self.job)
        self.job["cases"][0]["state_refs"] = ["state.json"]
        with self.assertRaises(ValueError):
            validate_job(self.job)
        self.job["numerical"]["state_contract_ref"] = "state_contract.json"
        validate_job(self.job)
        self.job["callable"]["stateful"] = False
        with self.assertRaisesRegex(ValueError, "state needs stateful"):
            validate_job(self.job)

    def test_units_ranks_and_strict_numbers(self):
        for path, value in [("unit", "cycles"), ("measured_calls", True)]:
            changed = copy.deepcopy(self.job)
            changed["timing"][path] = value
            with self.assertRaises(ValueError):
                validate_job(changed)
        for key, value in [("rank", 1), ("rank", False)]:
            changed = copy.deepcopy(self.job)
            changed["cases"][0][key] = value
            with self.assertRaises(ValueError):
                validate_job(changed)
        self.job["cases"][0]["scalars"]["scale"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_job(self.job)

    def test_scalar_and_empty_tensors_and_explicit_stride_unit(self):
        tensor = self.job["cases"][0]["tensors"]["x"]
        tensor.update(shape=[], strides=[])
        validate_job(self.job)
        tensor.update(shape=[0, 2], strides=[0, -1])
        validate_job(self.job)
        tensor["stride_unit"] = "bytes"
        with self.assertRaisesRegex(ValueError, "stride unit"):
            validate_job(self.job)

    def test_versioned_reformulation_requires_predecessor(self):
        previous = job_digest(self.job)
        self.job["job_version"] = 2
        with self.assertRaises(ValueError):
            validate_job(self.job)
        self.job["supersedes"] = previous
        self.assertNotEqual(previous, job_digest(self.job))

    def test_paths_bytes_and_bound_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in self.files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            job_path = root / "job.json"
            job_path.write_text(json.dumps(self.job), encoding="utf-8")
            digest = job_digest(self.job)
            self.assertEqual(load_job(job_path, root, digest), self.job)
            (root / "inputs.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact digest mismatch"):
                load_job(job_path, root, digest)
            self.job["workload"]["tp"] = 2
            job_path.write_text(json.dumps(self.job), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registered job digest mismatch"):
                load_job(job_path, expected_digest=digest)
            job_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_job(job_path)

    def test_path_traversal_and_symlink_escape(self):
        for path in ("../outside", "/outside", "./alias", "x//y", "x/../y", "x\\y"):
            changed = copy.deepcopy(self.job)
            changed["artifacts"][path] = "0" * 64
            with self.assertRaisesRegex(ValueError, "normalized and relative"):
                validate_job(changed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            for name, data in self.files.items():
                if name != "source/baseline.py":
                    (root / name).write_bytes(data)
            outside = Path(directory) / "outside"
            outside.mkdir()
            (root / "source").symlink_to(outside, target_is_directory=True)
            (outside / "baseline.py").write_bytes(self.files["source/baseline.py"])
            with self.assertRaisesRegex(ValueError, "symlink artifact"):
                validate_job(self.job, root)


if __name__ == "__main__":
    unittest.main()
