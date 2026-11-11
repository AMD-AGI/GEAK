# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic task and complete three-process evidence for CPU contract tests."""

from geak.native._util import canonical_json as canonical
from geak.native._util import sha256_hex as sha256
from geak.native.external import project_external_task, validate_external_task


def fixture(root):
    cases = [{"name": name, "regime": regime, "calls": calls, "stat": "min", "legs": 5,
              "warmup": 5, "repeats": 20, "inner": 1, "graph": False, "timed": True}
             for name, regime, calls in (("a", "extend", 100), ("b", "extend", 100), ("c", "draft", 1))]
    meta = {"cases": cases, "candidate_module_file": "kernel_src/module.py", "candidate_attr": "run",
            "baseline_module_file": "baseline_src/module.py", "baseline_attr": "run",
            "tol": None, "rms_tol": None, "seed": 0, "random_draws": 3}
    bodies = {name: b"# synthetic source; never imported\n" for name in
              ("kernel_src/module.py", "baseline_src/module.py", "ut_main.py", "weight_lib.py", "harness_lib.py")}
    bodies["meta.json"] = canonical(meta)
    for name, body in bodies.items():
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    contract = project_external_task(meta, artifacts={name: sha256(body) for name, body in bodies.items()},
        original_source_map={"kernel_src/module.py": "baseline_src/module.py"}, primary_regime="extend",
        source={"repository": "external:fixture", "revision": "a"*64, "task_selector": "fixture"},
        runtime={"image": "example/image@sha256:"+"b"*64, "config_digest": "sha256:"+"c"*64})
    return validate_external_task(contract, root)


def raw_reports(request, *, original=False, draft_regression=False):
    source = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
    artifacts = {"START.json": canonical({"parent_evaluation": {"operation_id": request["operation_id"],
        "request_sha256": sha256(canonical(request)), "binding": request["binding"]},
        "source_files": {**request["contract"]["artifacts"], **source}, "declared_processes": 3,
        "gain_required_exit": False})}
    history = []
    # The median complete fold differs from a fold of separately reduced cases.
    series = [((100., 50.), (1., 1.)), ((1., 1.), (100., 20.)), ((100., 20.), (100., 50.))]
    for i, pairs in enumerate(series, 1):
        rows = []
        for c, (base, candidate) in zip(request["contract"]["metadata"]["value"]["cases"],
                                       [*pairs, (1., 2. if draft_regression else .5)]):
            if original:
                candidate = base
            rows.append({"name": c["name"], "regime": c["regime"], "calls": c["calls"],
                "graph_requested": False, "timer": "cuda_event", "measurement_valid": True,
                "measurement_errors": [], "legs_kept": 4,
                "arm_timers": {role: ["cuda_event"]*5 for role in ("baseline", "candidate", "control")},
                "baseline_legs": [base]*4, "optimized_legs": [candidate]*4, "control_legs": [base]*4,
                "baseline_ms": base, "optimized_ms": candidate, "control_ms": base,
                "speedup": base/candidate, "control_ratio": 1., "mde_frac": 0.})
        folds = {}
        for regime in ("draft", "extend"):
            selected = [row for row in rows if row["regime"] == regime]
            folds[regime] = sum(row["calls"]*row["baseline_ms"] for row in selected)/sum(row["calls"]*row["optimized_ms"] for row in selected)
        fp = sha256(canonical(source))
        runtime = {"packages": {"torch": "synthetic"}, "env": {}}
        runtime_id = sha256(canonical(runtime))
        history.append({"schema_version": 2, "process_id": f"synthetic:{i}:100", "fp": fp,
            "runtime_id": runtime_id, "measurement_valid": True, "correctness_pass": True,
            "weighted": folds["extend"], "weighted_mde": 0., "per_case": rows,
            "complete_regimes": ["draft", "extend"], "folds": folds,
            "buckets": {row["name"]: {"sp": row["speedup"], "mde": 0.} for row in rows}})
        gain = i >= 2 and not original and not draft_regression
        regression = i >= 2 and draft_regression and not original
        report = {"schema_version": 2, "candidate_fingerprint": fp, "runtime_fingerprint": runtime_id,
            "runtime": runtime, "correctness_pass": True, "measurement_valid": True,
            "performance_scope_complete": True, "measurement_errors": [], "missing_cases": [],
            "not_deployment_timed": [], "e2e_verified": False, "expected_case_count": 3,
            "measured_case_count": 3, "per_case": rows, "complete_regimes": ["draft", "extend"],
            "folds": {key: {"weighted": value} for key, value in folds.items()}, "primary_regime": "extend",
            "call_weighted": folds["extend"], "admissible": gain, "reproducible_gain": gain,
            "reproducible_regression": regression,
            "repro": {"runs": i, "enough": i >= 2, "reproducible_gain": gain, "reproducible_regression": regression}}
        prefix = f"process_{i:02d}/"
        artifacts[prefix+"ut_result.json"] = canonical(report)
        artifacts[prefix+"_runs_v2.jsonl"] = b"\n".join(canonical(row) for row in history)+b"\n"
        artifacts[prefix+"RECEIPT.json"] = canonical({"index": i, "returncode": 0, "sources_unchanged": True,
            **{name+"_sha256": sha256(artifacts[prefix+name]) for name in ("ut_result.json", "_runs_v2.jsonl")}})
    return artifacts


def evaluation_request(contract, task, workspace=None, evaluation_id="test"):
    from scripts.native_runtime.direct_external_command import create_request, snapshot

    workspace = task if workspace is None else workspace
    captured = snapshot(workspace, contract["source_paths"])
    return create_request(contract, task, "epoch", evaluation_id, captured, workspace)
