"""Pinned whole-call UTs, using the existing Arena owner and artifact envelope.

The first supported method is GLM's weight_lib v2. Its phase folds and native
gain gate are kept separate from graph and generated-task measurements. This
module reads bytes only; it never imports a task or starts a process.
"""

import math
from pathlib import Path
import statistics

from .jobs import _canonical, _decode, _ids, _keys, _path, _read, _revision, _sha, _text
from .._util import canonical_json, require, sha256_hex

SCHEMA = "geak-arena-external-task-v1"
METHOD = "glm-weight-lib-v2"


def metadata_view(meta):
    """Small contract view; the complete metadata file remains hash-protected."""
    view = {key: meta[key] for key in ("candidate_module_file", "baseline_module_file",
            "candidate_attr", "baseline_attr", "tol", "rms_tol", "seed", "random_draws") if key in meta}
    view["cases"] = [{key: row[key] for key in ("name", "regime", "calls", "stat", "legs",
                     "warmup", "repeats", "inner", "graph", "timed") if key in row} for row in meta["cases"]]
    return view


def need(value, message):
    require(value, message, ValueError)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_external_task(contract, artifact_root=None):
    contract = _decode(_canonical(contract))
    _keys(contract, ("schema", "source", "runtime", "artifacts", "source_paths",
                     "original_source_map", "metadata", "correctness_case_ids",
                     "benchmark_case_ids", "benchmark_report_ids", "case_bindings", "phases",
                     "timing_policy", "report_contract", "measurement_valid",
                     "custody_proven", "reward_eligible"))
    need(contract["schema"] == SCHEMA, "unsupported external task schema")
    for key in ("measurement_valid", "custody_proven", "reward_eligible"):
        need(contract[key] is False, "external UT does not establish " + key)
    _keys(contract["source"], ("repository", "revision", "task_selector"))
    _text(contract["source"]["repository"])
    _revision(contract["source"]["revision"])
    _path(contract["source"]["task_selector"])
    _keys(contract["runtime"], ("image", "config_digest"))
    image = contract["runtime"]["image"]
    need(type(image) is str and "@sha256:" in image, "pinned external runtime required")
    _sha(image.rsplit("@sha256:", 1)[1])
    config = contract["runtime"]["config_digest"]
    need(type(config) is str and config.startswith("sha256:"), "runtime config digest required")
    _sha(config[7:])
    files = contract["artifacts"]
    need(type(files) is dict and files, "external task artifacts required")
    for name, digest in files.items():
        _path(name)
        _sha(digest)
    paths = contract["source_paths"]
    need(type(paths) is list and paths and len(paths) == len(set(paths)), "source closure required")
    for name in paths:
        need(name in files, "source is outside pinned closure")
    original = contract["original_source_map"]
    need(type(original) is dict and set(original) == set(paths), "complete ORIGINAL source map required")
    for name, baseline in original.items():
        need(baseline in files and baseline not in paths and files[name] == files[baseline],
             "founding candidate must equal its protected ORIGINAL")
    metadata = contract["metadata"]
    _keys(metadata, ("path", "value"))
    need(metadata["path"] in files and metadata["path"] not in paths, "protected metadata required")
    meta = metadata["value"]
    cases = meta["cases"]
    ids = [row["name"] for row in cases]
    timed = [row["name"] for row in cases if row.get("timed")]
    _ids(ids)
    _ids(timed)
    need(contract["correctness_case_ids"] == ids and contract["benchmark_case_ids"] == timed
         and contract["benchmark_report_ids"] == {name: name for name in timed}, "case domain differs from UT")
    need(set(contract["case_bindings"]) == set(ids), "case bindings do not cover the UT")
    for i, row in enumerate(cases):
        binding = contract["case_bindings"][row["name"]]
        _keys(binding, ("artifact_ref", "json_pointer", "sha256"))
        need(binding["artifact_ref"] == metadata["path"] and binding["json_pointer"] == f"/cases/{i}",
             "case pointer differs from protected metadata")
        _sha(binding["sha256"])
    protocol = contract["report_contract"]
    _keys(protocol, ("method", "processes", "primary_regime"))
    need(protocol["method"] == METHOD and type(protocol["processes"]) is int
         and protocol["processes"] == 3, "unsupported external UT method")
    need(protocol["primary_regime"] in {row["regime"] for row in cases if row.get("timed")},
         "primary phase is outside timed cases")
    policy = {row["name"]: {key: row[key] for key in
              ("regime", "calls", "stat", "legs", "warmup", "repeats", "inner", "graph")}
              for row in cases if row.get("timed")}
    need(contract["timing_policy"] == policy, "timing policy differs from pinned metadata")
    for row in policy.values():
        need(row["stat"] in ("min", "median") and type(row["graph"]) is bool
             and all(type(row[key]) is int and row[key] > 0 for key in
                     ("calls", "legs", "warmup", "repeats", "inner"))
             and row["legs"] >= 2, "unsupported GLM timing settings")
    _keys(contract["phases"], ("evaluate",))
    phase = contract["phases"]["evaluate"]
    _keys(phase, ("argv", "timeout_seconds"))
    need(type(phase["argv"]) is list and phase["argv"] and positive(phase["timeout_seconds"]),
         "bounded combined UT command required")
    for arg in phase["argv"]:
        _text(arg)
    for name in ("ut_main.py", "weight_lib.py", "harness_lib.py"):
        need(name in files and name not in paths, "protected UT source missing: " + name)
    need(meta["candidate_module_file"] in paths and meta["baseline_module_file"] in original.values(),
         "callable binding differs from source map")
    if artifact_root is not None:
        for name, digest in files.items():
            raw = _read(Path(artifact_root) / name)
            need(sha256_hex(raw) == digest, "external artifact changed: " + name)
            if name == metadata["path"]:
                need(metadata_view(_decode(raw)) == meta, "effective metadata differs from protected file")
                for row in _decode(raw)["cases"]:
                    need(contract["case_bindings"][row["name"]]["sha256"] == sha256_hex(_canonical(row)),
                         "protected case descriptor digest differs")
    return contract


def project_external_task(meta, *, artifacts, original_source_map, source, runtime,
                          primary_regime, timeout_seconds=600):
    """Describe an already pinned external UT; the caller validates its files."""
    view = metadata_view(meta)
    timed = [row["name"] for row in view["cases"] if row["timed"]]
    return validate_external_task({
        "schema": SCHEMA, "source": source, "runtime": runtime, "artifacts": artifacts,
        "source_paths": sorted(original_source_map), "original_source_map": original_source_map,
        "metadata": {"path": "meta.json", "value": view},
        "correctness_case_ids": [row["name"] for row in view["cases"]],
        "benchmark_case_ids": timed, "benchmark_report_ids": {name: name for name in timed},
        "case_bindings": {row["name"]: {"artifact_ref": "meta.json", "json_pointer": f"/cases/{i}",
                "sha256": sha256_hex(_canonical(row))} for i, row in enumerate(meta["cases"])},
        "phases": {"evaluate": {"argv": ["python3", "-m", "geak_evo.evaluate.external_runner",
                    "--request", "request.json", "--task", "task", "--out", "observer"],
                    "timeout_seconds": timeout_seconds}},
        "timing_policy": {row["name"]: {key: row[key] for key in
            ("regime", "calls", "stat", "legs", "warmup", "repeats", "inner", "graph")}
            for row in view["cases"] if row["timed"]},
        "report_contract": {"method": METHOD, "processes": 3, "primary_regime": primary_regime},
        "measurement_valid": False, "custody_proven": False, "reward_eligible": False,
    })


def validate_objective(contract, objective):
    need(objective == {"kind": "external_primary",
                       "primary_regime": contract["report_contract"]["primary_regime"]},
         "external UT must retain its declared primary-phase objective")


def _near(a, b):
    return positive(a) and positive(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=0)


def _gate(history):
    """weight_lib.repro_gate: minimum two fresh processes, no bucket regression."""
    if len(history) < 2:
        return False, False
    regression = any(all(1-row["buckets"][case]["sp"] > row["buckets"][case]["mde"]
                         for row in history) for case in history[0]["buckets"])
    gain = all(row["weighted"]-1 > row["weighted_mde"] for row in history) and not regression
    return gain, regression


def read_external_report(request, artifacts):
    """Read one owned three-process collection, preserving all phase/case results.

    Artifacts use the wrapper's relative names. Positive owner settlement and
    physical-runtime evidence remain the existing Arena evaluator's job.
    """
    contract = validate_external_task(request["contract"])
    files, meta = contract["artifacts"], contract["metadata"]["value"]
    source = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
    need(set(source) == set(contract["source_paths"]) and all(source.values()), "candidate source closure differs")
    start = _decode(artifacts["START.json"])
    need(start["parent_evaluation"] == {"operation_id": request["operation_id"],
         "request_sha256": sha256_hex(canonical_json(request)), "binding": request["binding"]},
         "external report belongs to another owner request")
    expected_files = {**files, **source}
    need(start["source_files"] == expected_files and start["declared_processes"] == 3,
         "external report source/method binding differs")
    need(start["gain_required_exit"] is False, "first UT process cannot require prior gain history")
    cases = {row["name"]: row for row in meta["cases"] if row.get("timed")}
    reports, history, fingerprints, runtime = [], [], None, None
    evidence = ["START.json"]
    for index in range(1, 4):
        prefix = f"process_{index:02d}/"
        names = [prefix+name for name in ("RECEIPT.json", "ut_result.json", "_runs_v2.jsonl")]
        receipt, report = [_decode(artifacts[name]) for name in names[:2]]
        rows = [_decode(line) for line in artifacts[names[2]].splitlines() if line.strip()]
        need(receipt["index"] == index and receipt["returncode"] == 0
             and receipt["sources_unchanged"] is True, "external process failed or changed source")
        for name in ("ut_result.json", "_runs_v2.jsonl"):
            need(receipt[name+"_sha256"] == sha256_hex(artifacts[prefix+name]), "external result hash differs")
        need(len(rows) == index and rows[:-1] == history
             and len({row["process_id"] for row in rows}) == index, "fresh append-only process history required")
        for row in rows:
            _text(row["process_id"])
        history = rows
        fp = (report["candidate_fingerprint"], report["runtime_fingerprint"])
        for digest in fp:
            _sha(digest)
        need(fingerprints is None or fingerprints == fp, "candidate/runtime fingerprint changed between processes")
        fingerprints = fp
        need(runtime is None or runtime == report["runtime"], "runtime changed between processes")
        runtime = report["runtime"]
        need(report["schema_version"] == 2 and all(report[key] is True for key in
             ("correctness_pass", "measurement_valid", "performance_scope_complete"))
             and not report["measurement_errors"] and not report["missing_cases"]
             and not report["not_deployment_timed"] and report["e2e_verified"] is False,
             "external UT did not pass its complete native checks")
        need(report["expected_case_count"] == report["measured_case_count"] == len(cases)
             and len(report["per_case"]) == len(cases)
             and {row["name"] for row in report["per_case"]} == set(cases), "external result case coverage differs")
        for row in report["per_case"]:
            policy = cases[row["name"]]
            timer = "cuda_event_graph" if policy["graph"] else "cuda_event"
            need(row["regime"] == policy["regime"] and row["calls"] == policy["calls"]
                 and row["graph_requested"] is policy["graph"] and row["timer"] == timer
                 and row["measurement_valid"] is True and not row["measurement_errors"]
                 and row["legs_kept"] == policy["legs"]-1, "case timing policy differs")
            need(set(row["arm_timers"]) == {"baseline", "candidate", "control"}
                 and all(timers == [timer]*policy["legs"] for timers in row["arm_timers"].values()),
                 "timer fallback or missing arm")
            for role, field in (("baseline", "baseline"), ("candidate", "optimized"), ("control", "control")):
                values = row[field+"_legs"]
                need(type(values) is list and len(values) == policy["legs"]-1 and all(positive(v) for v in values),
                     "raw retained timing legs incomplete")
                reduced = min(values) if policy["stat"] == "min" else statistics.median(values)
                need(_near(reduced, row[field+"_ms"]), "UT time differs from retained legs")
            need(_near(row["speedup"], row["baseline_ms"]/row["optimized_ms"])
                 and _near(row["control_ratio"], row["baseline_ms"]/row["control_ms"]), "case ratio differs")
            nulls = row["baseline_legs"] + row["control_legs"]
            floor = 2*(max(nulls)-min(nulls))/max(max(nulls)+min(nulls), 1e-12)
            need(type(row["mde_frac"]) in (int, float) and math.isfinite(row["mde_frac"]) and row["mde_frac"] >= 0
                 and math.isclose(row["mde_frac"], floor, rel_tol=1e-12, abs_tol=1e-15), "native null floor differs")
        regimes = sorted({row["regime"] for row in cases.values()})
        need(report["complete_regimes"] == regimes and set(report["folds"]) == set(regimes), "phase coverage differs")
        totals = {}
        for regime in regimes:
            selected = [row for row in report["per_case"] if row["regime"] == regime]
            before = sum(row["calls"]*row["baseline_ms"] for row in selected)
            after = sum(row["calls"]*row["baseline_ms"]/row["speedup"] for row in selected)
            totals[regime] = before
            need(_near(report["folds"][regime]["weighted"], before/after), "phase fold differs")
        primary = contract["report_contract"]["primary_regime"]
        need(max(totals, key=totals.get) == primary == report["primary_regime"]
             and _near(report["call_weighted"], report["folds"][primary]["weighted"]), "UT primary phase changed")
        selected = [row for row in report["per_case"] if row["regime"] == primary]
        floor = sum(row["mde_frac"]*row["baseline_ms"]*row["calls"] for row in selected)/totals[primary]
        current = history[-1]
        need(current["schema_version"] == 2 and current["measurement_valid"] is True
             and current["correctness_pass"] is True and (current["fp"], current["runtime_id"]) == fp
             and current["per_case"] == report["per_case"] and current["complete_regimes"] == regimes
             and current["folds"] == {key: report["folds"][key]["weighted"] for key in regimes}
             and _near(current["weighted"], report["call_weighted"])
             and type(current["weighted_mde"]) in (int, float) and current["weighted_mde"] >= 0
             and math.isclose(current["weighted_mde"], floor, rel_tol=1e-12, abs_tol=1e-15)
             and current["buckets"] == {row["name"]: {"sp": row["speedup"], "mde": row["mde_frac"]}
                                       for row in report["per_case"]}, "history differs from current UT result")
        gain, regression = _gate(history)
        need(report["repro"]["runs"] == index and report["repro"]["enough"] is (index >= 2)
             and report["repro"]["reproducible_gain"] is gain
             and report["repro"]["reproducible_regression"] is regression
             and report["reproducible_gain"] is gain and report["admissible"] is gain
             and report["reproducible_regression"] is regression, "native gain gate differs")
        reports.append(report)
        evidence.extend(names)
    # Per-case medians are descriptive. The headline is the median of complete
    # process folds, never a fold reconstructed from independently reduced cases.
    paired = {name: {"baseline_ms": statistics.median([next(row["baseline_ms"] for row in r["per_case"]
                if row["name"] == name) for r in reports]),
                    "optimized_ms": statistics.median([next(row["optimized_ms"] for row in r["per_case"]
                if row["name"] == name) for r in reports])} for name in cases}
    for row in paired.values():
        row["speedup"] = row["baseline_ms"]/row["optimized_ms"]
    folds = {regime: [r["folds"][regime]["weighted"] for r in reports] for regime in regimes}
    return {"method": METHOD, "primary_regime": primary, "primary_speedup": statistics.median(folds[primary]),
            "native_reproducible_gain": gain, "native_reproducible_regression": regression,
            "phase_speedups": {key: {"processes": values, "median": statistics.median(values)}
                               for key, values in folds.items()},
            "paired_cases": paired, "runtime": runtime, "runtime_fingerprint": fingerprints[1],
            "per_process": reports, "evidence": evidence,
            "observations": {"correctness": {"status": "passed", "case_ids": contract["correctness_case_ids"], "evidence": evidence},
                             "benchmark": {"status": "passed", "times_ms": {name: row["optimized_ms"] for name, row in paired.items()},
                                           "evidence": evidence}}}
