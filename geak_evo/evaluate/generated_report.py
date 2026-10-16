"""CPU reader for a prospective, admitted generated-task paired report.

The combined native evaluation owns both measurement legs.  This reader never
creates operations, reruns a leg, or grants source/runtime custody or scientific
acceptance.  Raw harness evidence remains separate from its derived lane score.
"""
import json
import math
from pathlib import PurePosixPath
import re

from .._util import canonical_json, sha256_hex
from .generated_task import validate_generated_task, _bound_metadata


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def upper_middle(samples, expected_count=50):
    """The pinned in-process timer's sorted[n//2], including even sample counts."""
    require(type(samples) is list and len(samples) == expected_count
            and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in samples),
            "incomplete or invalid raw timer samples")
    return sorted(samples)[len(samples) // 2]


def conventional_median(values):
    """The pinned across-process reducer averages the two middle values."""
    require(type(values) is list and values and all(_positive(value) for value in values),
            "incomplete or invalid replicate times")
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def summarize_replicates(baseline, candidate, *, max_reps=3, undecided=(0.95, 1.10)):
    """Recompute the actual interleaved sweep's separate medians and stop rule.

    Ordering/identity joins belong to the raw-record validator.  Missing or
    untimable legs are retained there and must not enter this complete scorer.
    """
    require(type(max_reps) is int and max_reps >= 1,
            "paired repetition ceiling must be a positive integer")
    require(type(undecided) in (list, tuple) and len(undecided) == 2
            and all(_positive(value) for value in undecided) and undecided[0] <= undecided[1],
            "paired stopping interval differs")
    require(type(baseline) is list and type(candidate) is list
            and len(baseline) == len(candidate) and 1 <= len(baseline) <= max_reps,
            "paired replicate coverage differs")
    for n in range(1, len(baseline) + 1):
        ratio = conventional_median(baseline[:n]) / conventional_median(candidate[:n])
        decisive = not undecided[0] <= ratio <= undecided[1]
        require(not decisive or n == len(baseline), "measurement continued after decisive pair")
    require(len(baseline) == max_reps or decisive, "measurement stopped before declared rule")
    bm, cm = conventional_median(baseline), conventional_median(candidate)
    pair_ratios = [b / c for b, c in zip(baseline, candidate)]
    return {"baseline_ms": bm, "optimized_ms": cm, "speedup": bm / cm,
            "reps": len(baseline),
            "speedup_spread": ((max(pair_ratios) - min(pair_ratios))
                               / conventional_median(pair_ratios) if len(pair_ratios) > 1 else 0.0)}


def canonical(value):
    return canonical_json(value, ensure_ascii=False)


def source_canonical(value):
    """ResearchStore source identity uses ASCII escaping; job identity does not."""
    return canonical_json(value)


def digest(raw):
    return sha256_hex(raw)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def _relative(name):
    require(type(name) is str and name and "\\" not in name and "\0" not in name,
            "invalid artifact path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and path.as_posix() == name and name != "."
            and ".." not in path.parts, "canonical relative artifact required")
    return name


class _Artifacts:
    """Read only bytes already retained in the owner's authenticated envelope."""
    def __init__(self, values):
        require(type(values) is dict and all(type(raw) is bytes for raw in values.values()),
                "retained generated artifacts must map names to bytes")
        for name in values:
            _relative(name)
        self.values, self.retained = values, {}

    def read(self, name, expected=None):
        raw = self.values[_relative(name)]
        self.retained[name] = raw
        require(expected is None or digest(raw) == expected, "artifact digest differs: " + name)
        return raw

    def document(self, name, expected=None):
        return strict_json(self.read(name, expected))


def parse_records(stdout, report_contract, parent_pid):
    """Parse only the exact producer prefix; retain the complete raw stream outside."""
    require(type(parent_pid) is int and parent_pid > 0, "native parent PID missing")
    prefix = report_contract["stdout_prefix"]
    records = []
    for line in stdout.decode("utf-8").splitlines():
        if not line.startswith(prefix):
            continue
        row = strict_json(line[len(prefix):])
        require(type(row) is dict and set(row) == {"schema", "kind", "parent_pid", "sequence", "monotonic_ns", "payload"}
                and row["schema"] == report_contract["evidence_schema"], "raw evidence schema differs")
        require(type(row["sequence"]) is int and row["sequence"] == len(records) + 1
                and type(row["parent_pid"]) is int and row["parent_pid"] == parent_pid,
                "raw evidence sequence/native parent differs")
        require(type(row["monotonic_ns"]) is int and row["monotonic_ns"] > 0
                and (not records or row["monotonic_ns"] >= records[-1]["monotonic_ns"]),
                "raw evidence monotonic order differs")
        require(row["kind"] in {"leg_start", "leg", "measurement_start", "measurement", "correctness",
                                "overlay_build_start", "overlay_build"}
                and type(row["payload"]) is dict, "unsupported raw evidence record")
        records.append(row)
    require(records, "no generated harness evidence")
    return records


def _timing(row, graph_requested):
    evidence, detail = row["timing_evidence"], row["timing_detail"]
    require(evidence["status"] == "ok" and evidence["error"] is None and type(detail) is dict,
            "timing failed or remains unresolved")
    require(canonical(evidence["requested"]) == canonical({"warmup": 10, "repeats": 50,
            "inner": 1, "graph": graph_requested})
            and canonical(evidence["effective"]) == canonical({"warmup": 10, "repeats": 50, "inner": 1}),
            "actual child timer policy differs")
    timer = evidence["timer"]
    require(timer in ("cuda_event", "cuda_event_graph", "wall")
            and detail["timer"] == timer == row["timer"], "actual timer missing or inconsistent")
    wall = upper_middle(evidence["wall_ms_samples"])
    if timer == "wall":
        require(evidence["device_ms_samples"] == [], "wall fallback claims device samples")
        measured = wall
    else:
        measured = upper_middle(evidence["device_ms_samples"])
    require(_positive(detail["ms"]) and row["ms"] == detail["ms"] == measured
            and row["wall_ms"] == detail["wall_ms"] == wall,
            "child timing reduction differs from ordered raw samples")
    require(type(detail.get("cache_condition")) is dict
            and row["cache_condition"] == detail["cache_condition"], "cache condition missing or inconsistent")
    capture = evidence["graph_capture"]
    require(capture["status"] in ("not_requested", "not_available", "ok", "error"),
            "graph capture disposition missing")
    require((not graph_requested and capture["status"] == "not_requested")
            or (graph_requested and capture["status"] != "not_requested"), "graph request/capture differs")
    require((timer == "cuda_event_graph") == (capture["status"] == "ok"), "graph timer/capture differs")
    require(("host_ms" in detail) == ("primed" in detail), "partial dispatch diagnostic")
    if "host_ms" in detail:
        require(type(detail["host_ms"]) in (int, float) and math.isfinite(detail["host_ms"])
                and detail["host_ms"] >= 0 and type(detail["primed"]) is bool
                and detail["primed"] == (detail["host_ms"] < measured), "dispatch diagnostic differs")
    require(type(row["compile_evidence"]) is dict, "compile diagnostic missing")
    return measured


def _identity(result, role, contract, task_dir, baseline_closure, expected_source_sha256):
    identity, observed = result["identity"], result["identity_file_observation"]
    require(type(identity) is dict and all(type(identity.get(key)) is str and identity[key]
            for key in ("module", "file", "qualname")), "resolved callable identity missing")
    require(observed["status"] == "ok" and observed["error"] is None
            and observed["path"] == identity["file"]
            and observed["claim"] == "resolved_file_bytes_observed_after_timing_not_loaded_code_proof",
            "resolved callable file observation missing")
    path = PurePosixPath(identity["file"])
    require(path.is_absolute() and path.as_posix() == identity["file"] and ".." not in path.parts,
            "noncanonical resolved callable file")
    if role == "baseline":
        allowed = {row["path"]: row["sha256"] for row in baseline_closure["external_files"]}
        # The actual harness rejects baseline callables resolving inside task_dir.
        require(not identity["file"].startswith(task_dir + "/")
                and allowed.get(identity["file"]) == observed["sha256"],
                "baseline callable file is outside its admitted external closure")
    else:
        bind = contract["runtime_requirements"]["candidate_bind"]
        require(observed["sha256"] == expected_source_sha256[bind["file"]],
                "candidate callable file differs from bound source")
        module = bind["module"] if bind["kind"] == "module" else bind["impl_module"]
        # Overlay setup chooses its generated filename. Bind the actual bytes,
        # admitted implementation module, and generated overlay/source scope.
        require((identity["file"].startswith(task_dir + "/_cand_overlay/")
                 or identity["file"] == task_dir + "/" + bind["file"])
                and identity["module"] == module,
                "candidate callable route differs from admitted overlay")
        if bind["kind"] == "rebind":
            require(identity["qualname"] == bind["impl_attr"], "candidate rebound callable differs")


def _correctness(records, case_rows):
    correctness = [record["payload"] for record in records if record["kind"] == "correctness"]
    require(len(correctness) == 1, "exactly one complete correctness report required")
    verdict = correctness[0]
    require(verdict["status"] == "ok" and verdict["error"] is None and verdict["ok"] is True,
            "generated correctness did not pass")
    report = verdict["report"]
    require(type(report) is dict and type(report.get("eager")) is list, "eager correctness evidence missing")
    signatures = [row["signature"] for row in case_rows]
    eager = [row["case"] for row in report["eager"] if row.get("case") != "output_independence"]
    require(eager == signatures, "eager correctness signature coverage differs")
    for group, rows in report.items():
        require(type(rows) is list and all(type(row) is dict and row.get("correct") is True for row in rows),
                "correctness group contains failed or unresolved observations: " + group)
    return verdict


def validate_records(contract, records, *, task_dir, baseline_closure, case_rows,
                     expected_source_sha256, meta):
    """Validate self records only inside externally bound native stream custody."""
    starts, legs, measurements, measurement_starts = {}, {}, [], []
    for record in records:
        kind, payload, sequence = record["kind"], record["payload"], record["sequence"]
        if kind in ("leg_start", "leg"):
            key = payload["leg_id"]
            require(type(key) is str and re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", key), "invalid raw leg ID")
            table = starts if kind == "leg_start" else legs
            require(key not in table and payload["task_dir"] == task_dir, "duplicate or foreign raw leg")
            table[key] = (sequence, payload)
        elif kind == "measurement_start":
            measurement_starts.append((sequence, payload))
        elif kind == "measurement":
            measurements.append((sequence, payload))
        # Correctness is checked separately. Overlay-build diagnostics are
        # retained verbatim; their original harness failure flows into the
        # measurement and native phase result, not a new scoring policy.
    require(set(starts) == set(legs), "started leg lacks its terminal evidence")
    for key, (end, leg) in legs.items():
        start, initial = starts[key]
        require(start < end and initial["status"] == "started", "leg start/terminal order differs")
        for field in ("task_dir", "overlay", "mode", "bucket", "seed", "draws", "argv", "timeout_s"):
            require(initial[field] == leg[field], "leg start/terminal binding differs: " + field)
        require(leg["status"] == "ok" and leg["returncode"] == 0 and leg["error"] is None
                and type(leg["child_pid"]) is int and leg["child_pid"] > 0,
                "child leg failed, timed out or lacks a process")
        require(type(leg["stdout"]) is str and type(leg["stderr"]) is str, "child streams missing")
        lines = [line for line in leg["stdout"].strip().splitlines() if line.startswith("{")]
        require(lines and canonical(strict_json(lines[-1])) == canonical(leg["result"]),
                "decoded child result differs from retained stdout")
        argv = leg["argv"]
        expected = [task_dir + "/leg_runner.py", "--task", task_dir, "--mode", leg["mode"],
                    "--seed", str(leg["seed"])]
        if leg["bucket"]:
            expected += ["--bucket", leg["bucket"]]
        if leg["mode"] != "oracle":
            if leg["draws"]:
                expected += ["--draws", str(leg["draws"])]
            require(type(argv) is list and len(argv) > 1 and argv[1:] == expected,
                    "child argv differs from declared raw leg")
    verdict = _correctness(records, case_rows)
    require(len(measurements) == len(measurement_starts) == 1, "exactly one complete paired sweep required")
    begin, initial = measurement_starts[0]
    end, measured = measurements[0]
    require(begin < end and initial["measurement_id"] == measured["measurement_id"]
            and initial["status"] == "started" and measured["status"] == "ok"
            and measured["error"] is None and measured["task_dir"] == initial["task_dir"] == task_dir
            and measured["policy"] == initial["policy"], "paired sweep start/terminal binding differs")
    replication = contract["timing_policy"]["replication"]
    policy = measured["policy"]
    require(canonical(policy) == canonical({"max_reps_requested": replication["max_reps"],
            "max_reps_effective": replication["max_reps"], "undecided": replication["undecided"],
            "child_timer_reducer": "upper_middle", "replicate_reducer": "median",
            "pair_order": ["baseline", "candidate"]}), "paired sweep policy differs")
    timed = [row for row in case_rows if row["benchmark"]]
    require([row["sig"] for row in measured["buckets"]] == [row["signature"] for row in timed]
            and [row["sig"] for row in measured["per_case"]] == [row["signature"] for row in timed],
            "paired sweep missing, duplicate or unexpected case signatures")
    controls = sorted((sequence, key, leg) for key, (sequence, leg) in legs.items()
                      if begin < sequence < end and leg["mode"] != "time")
    require([(leg["mode"], leg["overlay"]) for _, _, leg in controls] == [
        ("resolve", task_dir + "/baseline_overlay"), ("resolve", task_dir + "/_cand_overlay"),
        ("list", task_dir + "/baseline_overlay")], "paired sweep resolve/list coverage differs")
    previous = begin
    for sequence, key, leg in controls:
        require(previous < starts[key][0] < sequence, "paired sweep setup child order differs")
        previous = sequence
    resolved = {role: controls[index][2]["result"] for index, role in enumerate(("baseline", "candidate"))}
    require(resolved["baseline"] != resolved["candidate"]
            and not resolved["baseline"].get("error") and not resolved["candidate"].get("error"),
            "paired sweep callable resolution did not distinguish legs")
    require(controls[2][2]["result"] == {"sigs": [row["signature"] for row in timed]},
            "native list output differs from admitted timed signatures")
    used, per_case = set(), []
    regime = meta.get("regime", {})
    graph = False if regime.get("enforce_eager") else bool(regime.get("cuda_graph", True))
    for admitted, bucket, aggregate in zip(timed, measured["buckets"], measured["per_case"]):
        baseline, candidate, details = [], [], []
        for n, pair in enumerate(bucket["pairs"], 1):
            require(type(pair["replicate"]) is int and pair["replicate"] == n, "replicate order differs")
            pair_details = {}
            for role, values in (("baseline", baseline), ("candidate", candidate)):
                key = pair[role + "_leg_id"]
                require(key in legs and key not in used, "missing or reused paired child")
                used.add(key)
                seq, leg = legs[key]
                require(previous < starts[key][0] < seq < end and leg["mode"] == "time"
                        and leg["bucket"] == bucket["sig"]
                        and leg["overlay"] == task_dir + ("/baseline_overlay" if role == "baseline" else "/_cand_overlay"),
                        "B/C child order, bucket or overlay differs")
                previous = seq
                result = leg["result"]
                require(type(result["cases"]) is list and len(result["cases"]) == 1
                        and result["cases"][0]["sig"] == bucket["sig"], "timed child case coverage differs")
                row = result["cases"][0]
                require(result["identity"] == resolved[role], "timed callable identity differs from resolved leg")
                _identity(result, role, contract, task_dir, baseline_closure, expected_source_sha256)
                values.append(_timing(row, graph))
                pair_details[role] = {"leg_id": key, "child_pid": leg["child_pid"],
                    "identity": result["identity"], "identity_file_observation": result["identity_file_observation"],
                    "timing_detail": row["timing_detail"], "timing_evidence": row["timing_evidence"],
                    "compile_evidence": row["compile_evidence"]}
            require(pair_details["baseline"]["identity"] != pair_details["candidate"]["identity"],
                    "both timed legs resolved to the same callable identity")
            details.append(pair_details)
        require(bucket["baseline_ms"] == baseline and bucket["candidate_ms"] == candidate,
                "retained replicate arrays differ from child measurements")
        summary = summarize_replicates(baseline, candidate, max_reps=replication["max_reps"],
                                       undecided=replication["undecided"])
        for key, value in summary.items():
            require(type(aggregate.get(key)) is type(value) and aggregate[key] == value,
                    "reported paired aggregate differs: " + key)
        decisive = not replication["undecided"][0] <= summary["speedup"] <= replication["undecided"][1]
        require(bucket["stop_reason"] == ("outside_undecided" if decisive else "max_reps"),
                "reported pair stopping reason differs")
        per_case.append({"case_id": admitted["case_id"], "signature": admitted["signature"],
                         "descriptor": admitted["descriptor"], **summary, "pairs": details})
    require(used == {key for key, (_, leg) in legs.items() if leg["mode"] == "time"},
            "unaccounted timed children in native evaluation")
    return {"correctness_report": verdict, "per_case": per_case,
            "geomean_speedup": math.exp(math.fsum(math.log(row["speedup"]) for row in per_case) / len(per_case)),
            "measurement_id": measured["measurement_id"], "raw_records": records}


def read_generated_report(contract, retained_artifacts, expected_source_sha256):
    """Return existing Arena observation groups and a separately named paired score.

    The caller supplies the admitted contract/source map and the existing owner
    envelope's archived bytes, with the ``observer/`` prefix removed. Native
    settlement, runtime/device identity, alias and fixture preparation remain
    the existing owner's responsibility. No task code is imported or executed.
    No separate operation ID is fabricated for either leg.
    """
    contract = validate_generated_task(contract)
    require(set(expected_source_sha256) == set(contract["source_paths"]), "candidate source closure differs")
    require(all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in expected_source_sha256.values()), "invalid candidate source hash")
    artifacts = _Artifacts(retained_artifacts)
    result = {"correctness": {"status": "unknown", "case_ids": [], "evidence": []},
              "benchmark": {"status": "unknown", "times_ms": {}, "evidence": []}}
    issues, records, assessed, parent = [], [], None, None
    candidate_source_id = None
    phase, binding, task_dir, case_rows, baseline, meta = None, None, None, None, None, None
    common_ok = False
    try:
        identity = artifacts.document("identity.json")
        require(identity["task"] == contract["source"]["task_selector"][6:]
                and identity["source_commit"] == contract["source"]["revision"], "native task/revision differs")
        require(identity["generated_protocol"] == {"schema": "geak-generated-task-protocol-v1", "contract": contract}
                and identity["public_image"] == contract["runtime"]["image"]
                and identity["public_config_digest"] == contract["runtime"]["config_digest"],
                "native generated protocol/runtime differs")
        effective = strict_json(canonical(contract))
        effective["artifacts"].update(expected_source_sha256)
        for name in expected_source_sha256:
            mode = identity["task_file_manifest"][name]["executable"]
            require(type(mode) is int and mode in (0, 0o111), "invalid current candidate executable mode")
            effective["artifact_modes"][name] = mode
        # Archived transport retains bytes, not executable modes or symlinks.
        # Recheck all input bytes and metadata here; recorded source modes enter
        # the source ID below. Native preparation owns alias/fixture validation.
        contents = {name: artifacts.read("task/" + name, expected)
                    for name, expected in effective["artifacts"].items()}
        _bound_metadata(effective, contents)
        for name, expected in effective["artifacts"].items():
            recorded = identity["task_file_manifest"][name]
            require(recorded["sha256"] == expected
                    and recorded.get("executable", 0) == effective["artifact_modes"][name],
                    "native input manifest differs: " + name)
        for name, ref in contract["fixtures"].items():
            recorded = identity["task_file_manifest"][name]
            require(recorded["sha256"] == ref["sha256"] and recorded["size_bytes"] == ref["size_bytes"],
                    "native fixture manifest differs")
        metadata = contract["task_metadata"]
        case_rows = artifacts.document("task/" + metadata["case_manifest_path"])["cases"]
        baseline = artifacts.document("task/" + metadata["baseline_closure_path"])
        meta = artifacts.document("task/" + metadata["meta_path"])
        phase = artifacts.document("evaluate/phase.json")
        binding = artifacts.document("evaluate/protocol-binding.json", phase["protocol_binding_sha256"])
        require(binding["schema"] == "geak-generated-task-protocol-v1" and binding["phase"] == "evaluate"
                and binding["contract_sha256"] == digest(canonical(contract))
                and binding["source_sha256"] == expected_source_sha256 and binding["runtime"] == contract["runtime"]
                and binding["protected_artifacts_sha256"] == {name: value for name, value in contract["artifacts"].items()
                                                              if name not in contract["source_paths"]},
                "native phase source/runtime/contract binding differs")
        parent = binding["parent_evaluation"]
        require(parent == identity["generated_evaluation_binding"]
                and set(parent) == {"operation_id", "request_sha256", "binding"}
                and type(parent["operation_id"]) is str
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", parent["operation_id"])
                and re.fullmatch(r"[0-9a-f]{64}", parent["request_sha256"]),
                "native parent evaluation binding differs")
        source_manifest = {"kind": "source", "files": [{"path": name, "blob_id": expected_source_sha256[name],
            "executable": effective["artifact_modes"][name]} for name in sorted(expected_source_sha256)]}
        request_binding = parent["binding"]
        require(set(request_binding) == {"job_id", "measurement_epoch_id", "candidate_id", "candidate_source_id"}
                and request_binding["job_id"] == digest(canonical(contract))
                and request_binding["candidate_source_id"] == digest(source_canonical(source_manifest))
                and type(request_binding["measurement_epoch_id"]) is str and request_binding["measurement_epoch_id"]
                and re.fullmatch(r"[0-9a-f]{64}", request_binding["candidate_id"]),
                "native candidate/source/epoch binding differs")
        candidate_source_id = request_binding["candidate_source_id"]
        launch_ref = phase["launch_settings_ref"]
        launch = artifacts.document("evaluate/" + _relative(launch_ref["path"]), launch_ref["sha256"])
        task_dir = launch["cwd"]
        require(type(task_dir) is str and PurePosixPath(task_dir).is_absolute()
                and PurePosixPath(task_dir).as_posix() == task_dir and ".." not in PurePosixPath(task_dir).parts,
                "native launch task directory missing")
        require(launch["executed_argv"] == phase["executed_argv"]
                and phase["declared_command"] == contract["phases"]["evaluate"]["argv"]
                and phase["inner_timeout_seconds"] == phase["outer_timeout_seconds"]
                == contract["phases"]["evaluate"]["timeout_seconds"], "native command/allowance differs")
        require(phase["phase"] == "evaluate" and phase["timed_out"] is False and phase["returncode"] == 0
                and phase["settlement"]["empty"] is True and phase["tracked_file_changes"] == [],
                "native evaluate failed, changed inputs, timed out or remains unsettled")
        common_ok = True
    except (ValueError, KeyError, OSError, TypeError, AttributeError, IndexError) as exc:
        issues.append("evaluate binding: " + str(exc))
    # Always retain the available complete raw parent streams, including failures.
    streams = {}
    for stream in ("stdout", "stderr"):
        name = "evaluate/" + stream + ".bin"
        try:
            streams[stream] = artifacts.read(name, phase.get(stream + "_sha256") if phase else None)
            require(phase is not None and phase.get(stream + "_sha256") == digest(streams[stream]),
                    "native stream has no phase-bound digest")
        except (ValueError, KeyError, OSError, TypeError) as exc:
            common_ok = False
            issues.append(stream + ": " + str(exc))
    if "stdout" in streams and phase is not None:
        try:
            records = parse_records(streams["stdout"], contract["report_contract"], phase["process_pid"])
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            common_ok = False
            issues.append("generated stream: " + str(exc))
    if b"GEAK_MEASUREMENT_EVIDENCE_INCOMPLETE " in streams.get("stderr", b""):
        common_ok = False
        issues.append("generated stream: producer reported incomplete evidence")
    if common_ok:
        try:
            _correctness(records, case_rows)
            result["correctness"].update(status="passed", case_ids=contract["correctness_case_ids"])
            assessed = validate_records(contract, records, task_dir=task_dir, baseline_closure=baseline,
                case_rows=case_rows, expected_source_sha256=expected_source_sha256, meta=meta)
            result["benchmark"].update(status="passed", times_ms={
                contract["benchmark_report_ids"][row["case_id"]]: row["optimized_ms"] for row in assessed["per_case"]})
        except (ValueError, KeyError, OSError, TypeError, AttributeError, IndexError, UnicodeError) as exc:
            issues.append("generated report: " + str(exc))
    for group in result.values():
        if group["status"] != "passed":
            group["status"] = "failed" if records else "unknown"
    baseline_manifest = {"kind": "source", "files": [
        {"path": name, "blob_id": contract["artifacts"][name], "executable": contract["artifact_modes"][name]}
        for name in sorted(contract["source_paths"])]}
    paired = {"schema": "geak-generated-task-paired-score-v1", "status": result["benchmark"]["status"],
              "parent_evaluation": parent, "per_case": assessed["per_case"] if assessed else [],
              "baseline_source_id": digest(source_canonical(baseline_manifest)),
              "candidate_source_id": candidate_source_id,
              "baseline_closure_sha256": contract["artifacts"][contract["task_metadata"]["baseline_closure_path"]],
              "job_id": digest(canonical(contract)),
              "comparison_contract_sha256": digest(source_canonical(contract)),
              "runtime": contract["runtime"], "case_bindings": contract["case_bindings"],
              "benchmark_report_ids": contract["benchmark_report_ids"],
              "geomean_speedup": assessed["geomean_speedup"] if assessed else None,
              "raw_records": records, "aggregation": contract["timing_policy"]["aggregation"],
              "scope": "source-bound generated-task lane score; no workload acceptance or loaded-code proof"}
    # This is a derived artifact, regenerated from the retained native bundle.
    # The existing transport prefixes it as raw/observer/paired-score.json. It
    # contains neither its own hash nor its observation evidence references.
    artifacts.retained["paired-score.json"] = canonical(paired) + b"\n"
    evidence = list(artifacts.retained)
    for group in result.values():
        group["evidence"] = evidence
    return {"observations": result, "artifacts": artifacts.retained, "issues": issues, "paired_score": paired,
            "methods": {"report_format": contract["report_contract"]["protocol"],
                        "aggregation": contract["timing_policy"]["aggregation"]},
            "phase_validated": {"evaluate": common_ok and assessed is not None and not issues},
            "measurement_valid": False, "custody_proven": False, "reward_eligible": False}
