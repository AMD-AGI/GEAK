"""Explicit generated-task declaration; no task imports or native execution.

Artifact identities are streamed; semantically decoded metadata remains bounded.
Fixture bodies remain separately streamed at admission. External baseline source/build
identities are declared here and must be rechecked in the admitted runtime by
the owner; validating their metadata is not proof of their bytes or closure.
The original task_root identifies extraction, while artifact_root may be a
relocated immutable snapshot. Neither parsing nor hashing grants task authority.
"""
import functools
import hashlib
import os
from pathlib import Path, PurePosixPath

from .jobs import (_canonical, _decode, _ids, _keys, _path, _pointer,
                   _revision, _sha, _text)
from .native_contract import _read, regular, SMALL_TOTAL_LIMIT
from .._util import require


def _need(condition, reason):
    require(condition, reason, ValueError)


SCHEMA = "geak-arena-generated-task-v1"
REPORT_PROTOCOL = "geak-generated-task-paired-report-v1"
REQUIRED_HARNESS = {"cases.py", "harness_lib.py", "leg_runner.py", "overlay_setup.py", "unittest.py"}


def _shapes(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, IndexError, AttributeError, OSError) as exc:
            raise ValueError("malformed or unavailable generated-task declaration") from exc
    return wrapped


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def streamed_identity(path, expected_size, expected_sha256):
    """Bounded-memory whole-file verification; retain inode/change identity."""
    _sha(expected_sha256)
    _need(type(expected_size) is int and expected_size >= 0, "invalid expected size")
    with regular(path) as stream:
        before = os.fstat(stream.fileno())
        _need(before.st_size == expected_size, "fixture size mismatch: " + str(path))
        hasher, count = hashlib.sha256(), 0
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
            count += len(block)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
    _need(all(getattr(before, key) == getattr(after, key) for key in fields),
          "fixture changed while hashing")
    _need(count == expected_size and hasher.hexdigest() == expected_sha256,
          "fixture SHA256 mismatch: " + str(path))
    return {"sha256": hasher.hexdigest(), "size_bytes": count,
            "stat": {key: getattr(after, key) for key in fields}}

def _absolute(value):
    _text(value)
    path = PurePosixPath(value)
    _need(path.is_absolute() and str(path) == value and ".." not in path.parts and "\\" not in value,
          "canonical absolute task/runtime path required")
    return value


def _mode(value):
    _need(type(value) is int and value in (0, 0o111), "Git-representable executable bits required")


def _within(path, root):
    return path == root or root in PurePosixPath(path).parents


def _protected(names, artifacts, source_paths, required=()):
    _need(type(names) is list and names and len(set(names)) == len(names), "unique protected source list required")
    _need(set(required) <= set(names) and set(names) <= set(artifacts) - set(source_paths),
          "required harness/metadata must be protected artifacts")


def _file_rows(rows):
    _need(type(rows) is list and rows, "external baseline source/build closure required")
    paths = []
    for row in rows:
        _keys(row, ("path", "sha256", "executable"))
        paths.append(_absolute(row["path"]))
        _sha(row["sha256"]); _mode(row["executable"])
    _need(len(paths) == len(set(paths)), "duplicate baseline runtime file")


def _timing(value, artifacts, sources):
    _keys(value, ("kind", "in_process", "replication", "aggregation", "fallback", "source_refs"))
    _need(value["kind"] == "geak-generated-task-paired-timing-v1", "generated timing declaration required")
    _need(value["in_process"] == {"warmup_calls": 10, "measured_calls": 50, "inner_calls": 1,
          "statistic": "upper_middle", "unit": "ms"}, "actual 10/50 upper-middle timer required")
    # Exact numeric types avoid True == 1 in a JSON declaration.
    _need(all(type(value["in_process"][name]) is int for name in ("warmup_calls", "measured_calls", "inner_calls")),
          "integer timer counts required")
    repeat = value["replication"]
    _keys(repeat, ("order", "fresh_process", "max_reps", "undecided", "stop_rule"))
    interval = repeat["undecided"]
    _need(repeat["order"] == ["baseline", "candidate"] and repeat["fresh_process"] is True
          and type(repeat["max_reps"]) is int and repeat["max_reps"] > 0
          and type(interval) is list and len(interval) == 2
          and all(type(bound) in (int, float) for bound in interval) and 0 < interval[0] <= interval[1]
          and repeat["stop_rule"] == "median_ratio_outside_inclusive_interval",
          "actual ordered fresh-process pair policy required")
    _need(value["aggregation"] == {"per_leg": "median", "even_count": "mean_of_middle_pair",
          "per_case": "baseline_median_over_candidate_median", "lane": "geometric_mean_of_case_ratios"},
          "ratio of separate conventional medians required")
    _need(value["fallback"] == "retain_actual_timer_and_detail", "actual fallback/timer detail must remain visible")
    _protected(value["source_refs"], artifacts, sources, ("harness_lib.py", "leg_runner.py", "unittest.py"))


def _bound_metadata(value, contents):
    task, artifacts, modes = value["task_metadata"], value["artifacts"], value["artifact_modes"]
    meta = _decode(contents[task["meta_path"]])
    routing = value["runtime_requirements"]
    _need(meta.get("target_callable") == routing["target_callable"]
          and meta.get("candidate_bind") == routing["candidate_bind"], "actual generated callable/routing differs")
    _need(meta.get("reference_io", "reference_io.pt") == task["oracle_path"]
          and meta.get("reference_io_sha256") == value["fixtures"][task["oracle_path"]]["sha256"],
          "actual root oracle path or hash differs")
    case_doc = _decode(contents[task["case_manifest_path"]])
    _keys(case_doc, ("schema", "cases"))
    _need(case_doc["schema"] == "geak-generated-task-cases-v1" and type(case_doc["cases"]) is list,
          "protected generated case manifest required")
    rows = case_doc["cases"]
    _need([row["case_id"] for row in rows] == value["correctness_case_ids"], "complete ordered correctness mapping differs")
    signatures, benchmark = [], []
    for index, row in enumerate(rows):
        _keys(row, ("case_id", "signature", "descriptor", "correctness", "benchmark"))
        _text(row["signature"])
        _need(type(row["descriptor"]) is dict and row["descriptor"] and row["correctness"] is True
              and type(row["benchmark"]) is bool, "complete case descriptor and coverage flags required")
        signatures.append(row["signature"])
        binding = value["case_bindings"][row["case_id"]]
        _need(binding == {"artifact_ref": task["case_manifest_path"], "json_pointer": "/cases/" + str(index),
                          "sha256": digest(row)}, "case descriptor/signature binding differs")
        _need(_pointer(case_doc, binding["json_pointer"]) == row, "case binding pointer differs")
        if row["benchmark"]:
            benchmark.append(row["case_id"])
            _need(value["benchmark_report_ids"][row["case_id"]] == row["signature"],
                  "native report ID differs from exact generated signature")
    _need(len(signatures) == len(set(signatures)) and benchmark == value["benchmark_case_ids"],
          "duplicate signatures or incomplete timed case mapping")
    closure = _decode(contents[task["baseline_closure_path"]])
    _keys(closure, ("schema", "runtime", "target_callable", "overlay_root", "task_files", "external_files"))
    _need(closure["schema"] == "geak-generated-task-baseline-closure-v1"
          and closure["runtime"] == value["runtime"] and closure["target_callable"] == routing["target_callable"]
          and closure["overlay_root"] == "baseline_overlay", "baseline runtime/callable/overlay binding differs")
    files = closure["task_files"]
    _need(type(files) is dict and files, "baseline overlay/source/build task files required")
    overlay = {name for name in artifacts if name.startswith("baseline_overlay/")}
    _need(overlay and overlay <= set(files) and set(files) <= set(artifacts) - set(value["source_paths"]),
          "baseline overlay closure omitted or overlaps editable source")
    for name, row in files.items():
        _keys(row, ("sha256", "executable")); _sha(row["sha256"]); _mode(row["executable"])
        _need(row == {"sha256": artifacts[name], "executable": modes[name]}, "baseline task-file identity differs")
    _file_rows(closure["external_files"])
    _need(all(not _within(PurePosixPath(row["path"]), PurePosixPath(task["task_root"]))
              for row in closure["external_files"]), "baseline runtime file must resolve outside original task directory")


@_shapes
def validate_generated_task(contract, artifact_root=None):
    value = _decode(_canonical(contract))
    _keys(value, ("schema", "source", "runtime", "runtime_requirements", "provenance", "task_metadata",
          "artifacts", "artifact_modes", "aliases", "source_paths", "fixtures", "correctness_case_ids",
          "benchmark_case_ids", "benchmark_report_ids", "case_bindings", "phases", "numerical_policy",
          "timing_policy", "report_contract", "measurement_valid", "custody_proven", "reward_eligible"))
    _need(value["schema"] == SCHEMA, "unsupported generated task schema")
    _need(all(value[name] is False for name in ("measurement_valid", "custody_proven", "reward_eligible")),
          "declaration cannot claim qualification")
    source = value["source"]
    _keys(source, ("repository", "revision", "task_selector", "snapshot_manifest_sha256"))
    _text(source["repository"]); _revision(source["revision"]); _path(source["task_selector"])
    _sha(source["snapshot_manifest_sha256"])
    _need(source["task_selector"].startswith("tasks/"), "snapshot task selector must name tasks/ entry")
    runtime = value["runtime"]
    _keys(runtime, ("image", "config_digest"))
    _text(runtime["image"]); _text(runtime["config_digest"])
    _need("@sha256:" in runtime["image"] and runtime["config_digest"].startswith("sha256:"), "pinned runtime required")
    _sha(runtime["image"].rsplit("@sha256:", 1)[1]); _sha(runtime["config_digest"][7:])
    artifacts, modes, aliases, fixtures = (value[k] for k in ("artifacts", "artifact_modes", "aliases", "fixtures"))
    _need(type(artifacts) is dict and artifacts and type(modes) is dict and set(modes) == set(artifacts),
          "exact artifact byte and executable-mode inventory required")
    for name, sha in artifacts.items():
        _path(name); _sha(sha); _mode(modes[name])
    sources = value["source_paths"]
    _need(type(sources) is list and sources and len(sources) == len(set(sources)) and set(sources) <= set(artifacts)
          and all(name.startswith("kernel_src/") for name in sources), "exact editable kernel_src closure required")
    _need(type(aliases) is dict and type(fixtures) is dict and fixtures, "explicit alias and oracle fixture inventories required")
    for name, target in aliases.items():
        _path(name); _path(target)
        _need(target in artifacts and name not in artifacts, "alias must target exact artifact")
    for name, row in fixtures.items():
        _path(name); _keys(row, ("manifest_path", "sha256", "size_bytes")); _path(row["manifest_path"]); _sha(row["sha256"])
        _need(type(row["size_bytes"]) is int and row["size_bytes"] > 0
              and name not in artifacts and name not in aliases, "fixture must be separate from small files")
    all_files = set(artifacts) | set(aliases) | set(fixtures)
    _need(all(not any(str(parent) in all_files for parent in PurePosixPath(name).parents) for name in all_files),
          "file paths cannot overlap parent directories")
    task = value["task_metadata"]
    _keys(task, ("task_root", "meta_path", "oracle_path", "case_manifest_path", "baseline_closure_path", "shared_harness_paths"))
    _absolute(task["task_root"])
    _need(task["meta_path"] == "meta.json" and task["oracle_path"] in fixtures
          and len(PurePosixPath(task["oracle_path"]).parts) == 1, "actual root oracle fixture required")
    _protected([task[k] for k in ("meta_path", "case_manifest_path", "baseline_closure_path")], artifacts, sources)
    _protected(task["shared_harness_paths"], artifacts, sources, REQUIRED_HARNESS)
    routing = value["runtime_requirements"]
    _keys(routing, ("target_callable", "candidate_bind", "task_run_root"))
    _text(routing["target_callable"])
    _need(":" in routing["target_callable"] and routing["task_run_root"] == ".", "actual callable and task-root execution required")
    bind = routing["candidate_bind"]
    _need(type(bind) is dict and bind.get("kind") in ("module", "rebind"), "explicit candidate routing required")
    _keys(bind, ("kind", "module", "file") if bind["kind"] == "module" else ("kind", "target", "impl_module", "impl_attr", "file"))
    for item in bind.values(): _text(item)
    _need(bind["file"] in sources, "candidate routing must name an editable source")
    provenance = value["provenance"]
    _keys(provenance, ("kind", "evidence_ref"))
    _need(provenance["kind"] == "hyperloom_kernel_extraction" and provenance["evidence_ref"] in artifacts
          and provenance["evidence_ref"] not in sources, "protected generated-task provenance required")
    _ids(value["correctness_case_ids"]); _ids(value["benchmark_case_ids"])
    _need(set(value["benchmark_case_ids"]) <= set(value["correctness_case_ids"]), "timed cases lack declared correctness")
    _keys(value["benchmark_report_ids"], value["benchmark_case_ids"])
    for report in value["benchmark_report_ids"].values(): _text(report)
    _need(len(set(value["benchmark_report_ids"].values())) == len(value["benchmark_report_ids"]), "duplicate report signature")
    _keys(value["case_bindings"], value["correctness_case_ids"])
    for row in value["case_bindings"].values():
        _keys(row, ("artifact_ref", "json_pointer", "sha256")); _sha(row["sha256"])
        _need(row["artifact_ref"] == task["case_manifest_path"] and type(row["json_pointer"]) is str
              and row["json_pointer"].startswith("/cases/"), "protected full-descriptor case binding required")
    _keys(value["phases"], ("evaluate",))
    phase = value["phases"]["evaluate"]
    _keys(phase, ("argv", "timeout_seconds"))
    _need(type(phase["argv"]) is list and phase["argv"][:2] == ["python3", "unittest.py"]
          and all(type(arg) is str and arg and "\x00" not in arg for arg in phase["argv"])
          and type(phase["timeout_seconds"]) is int and phase["timeout_seconds"] > 0,
          "explicit sealed unittest evaluate command and allowance required")
    _timing(value["timing_policy"], artifacts, sources)
    numerical = value["numerical_policy"]
    _keys(numerical, ("kind", "meta_ref", "source_refs", "result", "compile_parity_exception"))
    _need(numerical["kind"] == "geak-generated-task-correctness-v1" and numerical["meta_ref"] == task["meta_path"]
          and numerical["result"] == "run_correctness_ok_and_report"
          and numerical["compile_parity_exception"] == "retain_soft_degraded", "actual harness correctness semantics required")
    _protected(numerical["source_refs"], artifacts, sources, ("cases.py", "harness_lib.py", "unittest.py"))
    report = value["report_contract"]
    _keys(report, ("protocol", "evidence_schema", "stdout_prefix", "writable_paths"))
    _need(report["protocol"] == REPORT_PROTOCOL
          and report["evidence_schema"] == "geak-extracted-measurement-evidence-v1"
          and report["stdout_prefix"] == "GEAK_MEASUREMENT_EVIDENCE ",
          "explicit generated stdout paired-report protocol required")
    roots = report["writable_paths"]
    _need(type(roots) is list and roots and len(set(roots)) == len(roots), "explicit writable output paths required")
    for name in roots: _path(name)
    _need(all(not (_within(PurePosixPath(name), PurePosixPath(root)) or _within(PurePosixPath(root), PurePosixPath(name)))
              for name in all_files for root in roots), "writable output overlaps protected task/source/fixture")
    if artifact_root is not None:
        root, contents, total = Path(artifact_root).absolute(), {}, 0
        metadata = {task[key] for key in ("meta_path", "case_manifest_path", "baseline_closure_path")}
        for name, expected in artifacts.items():
            path = root / name
            if name in metadata:
                raw = _read(path); total += len(raw)
                _need(total <= SMALL_TOTAL_LIMIT and hashlib.sha256(raw).hexdigest() == expected
                      and path.stat().st_mode & 0o111 == modes[name], "artifact bytes or mode differ: " + name)
                contents[name] = raw
            else:
                identity = streamed_identity(path, path.lstat().st_size, expected)
                _need(identity["stat"]["st_mode"] & 0o111 == modes[name], "artifact mode differs: " + name)
        for name, target in aliases.items():
            path = root / name
            _need(path.is_symlink() and path.resolve(strict=True) == root / target, "declared alias differs")
        _bound_metadata(value, contents)
    return value


@_shapes
def fixture_admission(contract, fixture_root, task_destination="/work/task"):
    """Stream fixture bytes separately; key is task path, manifest_path is input.

    This creates a declarative read-only mount plan, never mounts or qualifies it.
    The owner must reverify or hold immutable bodies through worker admission.
    """
    value = validate_generated_task(contract)
    destination = _absolute(task_destination)
    root, verified, mounts = Path(fixture_root).absolute(), {}, []
    for name, row in value["fixtures"].items():
        path = root / row["manifest_path"]
        verified[name] = dict(streamed_identity(path, row["size_bytes"], row["sha256"]), host_path=str(path))
        mounts.append({"type": "bind", "source": str(path), "destination": str(Path(destination) / name),
                       "readonly": True, "sha256": row["sha256"], "size_bytes": row["size_bytes"]})
    return {"schema": "geak-generated-task-fixture-admission-v1", "job_id": digest(value),
            "fixtures": verified, "mount_plan": mounts, "mounts_applied": False,
            "immutable_snapshot_proven": False, "measurement_valid": False, "custody_proven": False}


@_shapes
def project_generated_task(*, task, observation, policy, provenance_ref, artifact_rows, meta):
    """Project an authenticated extraction and host policy into a declaration.

    The caller must authenticate the completed extraction and its observation,
    supply the admitted harness pins and timeout, and materialize artifact_rows
    including generated_cases.json, baseline_closure.json and provenance_ref.
    Validate those files with validate_generated_task before host admission.
    This projection grants no measurement, custody or execution authority.
    """
    _need(observation.get("schema") == "geak-generated-task-admission-observation-v1"
          and observation["task_dir"] == str(task), "exact generated-task observation required")
    _need(observation["runtime"] == policy["runtime"], "observed runtime differs from policy")
    files = observation["files"]
    generated = {"generated_cases.json", "baseline_closure.json", provenance_ref}
    _need(type(files) is dict and files and len(generated) == 3
          and not set(files).intersection(generated)
          and set(artifact_rows) == set(files) | generated,
          "artifact inventory differs from observed task and protected metadata")
    for name, row in files.items():
        _keys(row, ("sha256", "size_bytes", "executable"))
        _sha(row["sha256"]); _mode(row["executable"])
        _need(type(row["size_bytes"]) is int and row["size_bytes"] >= 0
              and row == artifact_rows[name], "copied artifact differs from observed file: " + name)
    _need(all(name not in files and target in files for name, target in observation["aliases"].items()),
          "alias must target an observed task file")
    pins = policy["shared_harness_sha256"]
    _need({"harness_lib.py", "leg_runner.py"} <= set(pins), "paired harness pins required")
    for name, expected in pins.items():
        _need(observation["files"][name]["sha256"] == artifact_rows[name]["sha256"] == expected,
              "shared harness differs from supplied policy")
    case_doc = observation["case_manifest"]
    _need(case_doc.get("schema") == "geak-generated-task-cases-v1", "actual generated case manifest required")
    cases = case_doc["cases"]
    _need(type(cases) is list and cases, "no observed correctness cases")
    sources = sorted(name for name in artifact_rows if name.startswith("kernel_src/"))
    _need(sources, "no editable kernel source closure")
    oracle = meta.get("reference_io", "reference_io.pt")
    _need(oracle in observation["files"] and observation["files"][oracle]["size_bytes"] > 0,
         "root immutable oracle is absent or empty")
    oracle_row = observation["files"][oracle]
    _need(meta.get("reference_io_sha256") == oracle_row["sha256"], "actual oracle differs from capture metadata")
    repeat = observation["effective_measurement"]
    _need(type(repeat["max_reps"]) is int and repeat["max_reps"] > 0
         and type(repeat["undecided"]) is list and len(repeat["undecided"]) == 2
         and all(type(x) in (int, float) for x in repeat["undecided"])
         and 0 < repeat["undecided"][0] <= repeat["undecided"][1], "actual repeat policy unavailable")
    revision = digest({"files": observation["files"], "aliases": observation["aliases"]})
    correctness = [row["case_id"] for row in cases]
    timed = [row for row in cases if row["benchmark"]]
    _need(timed, "no observed timed cases")
    contract = {
        "schema": "geak-arena-generated-task-v1",
        "source": {"repository": "hyperloom-generated-task", "revision": revision,
                   "task_selector": "tasks/generated/" + revision, "snapshot_manifest_sha256": revision},
        "runtime": policy["runtime"],
        "runtime_requirements": {"target_callable": meta["target_callable"],
                                 "candidate_bind": meta["candidate_bind"], "task_run_root": "."},
        "provenance": {"kind": "hyperloom_kernel_extraction", "evidence_ref": provenance_ref},
        "task_metadata": {"task_root": str(task), "meta_path": "meta.json", "oracle_path": oracle,
                          "case_manifest_path": "generated_cases.json", "baseline_closure_path": "baseline_closure.json",
                          "shared_harness_paths": sorted(set(policy["shared_harness_sha256"]) |
                              {"cases.py", "harness_lib.py", "leg_runner.py", "overlay_setup.py", "unittest.py"})},
        "artifacts": {name: row["sha256"] for name, row in artifact_rows.items() if name != oracle},
        "artifact_modes": {name: row["executable"] for name, row in artifact_rows.items() if name != oracle},
        "aliases": observation["aliases"], "source_paths": sources,
        "fixtures": {oracle: {"manifest_path": oracle, "sha256": oracle_row["sha256"],
                               "size_bytes": oracle_row["size_bytes"]}},
        "correctness_case_ids": correctness, "benchmark_case_ids": [row["case_id"] for row in timed],
        "benchmark_report_ids": {row["case_id"]: row["signature"] for row in timed},
        "case_bindings": {row["case_id"]: {"artifact_ref": "generated_cases.json", "json_pointer": "/cases/" + str(index),
                                           "sha256": digest(row)} for index, row in enumerate(cases)},
        "phases": {"evaluate": {"argv": ["python3", "unittest.py"], "timeout_seconds": policy["evaluation_seconds"]}},
        "numerical_policy": {"kind": "geak-generated-task-correctness-v1", "meta_ref": "meta.json",
                             "source_refs": ["cases.py", "harness_lib.py", "unittest.py"],
                             "result": "run_correctness_ok_and_report", "compile_parity_exception": "retain_soft_degraded"},
        "timing_policy": {"kind": "geak-generated-task-paired-timing-v1",
            "in_process": {"warmup_calls": 10, "measured_calls": 50, "inner_calls": 1, "statistic": "upper_middle", "unit": "ms"},
            "replication": {"order": ["baseline", "candidate"], "fresh_process": True,
                            "max_reps": repeat["max_reps"], "undecided": repeat["undecided"],
                            "stop_rule": "median_ratio_outside_inclusive_interval"},
            "aggregation": {"per_leg": "median", "even_count": "mean_of_middle_pair",
                            "per_case": "baseline_median_over_candidate_median", "lane": "geometric_mean_of_case_ratios"},
            "fallback": "retain_actual_timer_and_detail", "source_refs": ["harness_lib.py", "leg_runner.py", "unittest.py"]},
        "report_contract": {"protocol": "geak-generated-task-paired-report-v1",
                            "evidence_schema": "geak-extracted-measurement-evidence-v1",
                            "stdout_prefix": "GEAK_MEASUREMENT_EVIDENCE ", "writable_paths": policy["writable_paths"]},
        "measurement_valid": False, "custody_proven": False, "reward_eligible": False,
    }
    return validate_generated_task(contract)
