"""Unchanged legacy CPU job fixture shared by engineer parity tests."""

import hashlib


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

