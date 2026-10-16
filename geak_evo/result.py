"""Export a chosen program and matched provisional Arena measurements.

The common Report/Director/workload stages still own final acceptance. This
module replays selection and rereads retained evaluations; it starts no work.
"""

import copy
import difflib
import hashlib
import json
import math
from pathlib import Path
import re

from ._util import EvoError, canonical_json, require
from .archive.refs import read_ref
from .archive.store import _read
from .evaluate.evaluator import ArenaEvaluator, _parse, canonical, sha256
from .selection.final import validate_final_recommendation
from .selection.outcome import final_outcome, measured_outcome


def _need(value, reason):
    require(value, reason, ValueError)


def _published_timing(request, host, artifacts, assessed):
    """Read observed task-report-v1 graph samples and native cache receipts.

    Field meanings come from the published evaluation_contract.validate_report
    and published_protocol.prepare_phase/prepare_private_caches. Other routes
    need their own actual timing reader; declarations cannot substitute for it.
    """
    contract = request["contract"]
    protocol = contract["report_contract"].get("published_protocol", {})
    _need(contract["schema"] == "geak-arena-native-operator-v2"
          and protocol.get("schema") == "geak-published-task-protocol-v1"
          and protocol.get("report_format") == "task-report-v1",
          "actual timing proof is unsupported or unavailable for this report route")
    report_name = "observer/performance/raw-reports/build/performance_report.json"
    prefix = "observer/performance/"
    names = [report_name] + [prefix + name for name in ("phase.json", "protocol-binding.json", "protocol-request.json")]
    _need(set(names) <= set(host["observations"]["benchmark"]["evidence"]), "actual performance report binding is missing")
    report = _parse(artifacts[report_name])
    phase, binding, native_request = [_parse(artifacts[name]) for name in names[1:]]
    source = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
    _need(type(report["schema_version"]) is int and report["schema_version"] == 1 and report["status"] == "ok"
          and report["request"]["phase"] == "performance"
          and report["request"]["source_sha256"] == source
          and report["request"] == native_request == binding["request"], "actual performance source/request differs")
    info = phase["reports"]["build/performance_report.json"]
    _need(binding["schema"] == protocol["schema"]
          and re.fullmatch(r"[0-9a-f]{48}", native_request["request_id"])
          and phase["phase"] == binding["phase"] == "performance" and phase["returncode"] == 0
          and phase["timed_out"] is False and phase["settlement"]["empty"] is True
          and phase["tracked_file_changes"] == []
          and phase["protocol_binding_sha256"] == sha256(artifacts[names[2]])
          and info["changed_since_phase_start"] is True
          and info["sha256"] == sha256(artifacts[report_name]) and info["size_bytes"] == len(artifacts[report_name]),
          "actual performance phase/report binding differs")
    manifest_ref = protocol["case_manifest"]
    manifest_raw = _read(Path(request["artifact_root"]), manifest_ref["path"])[0]
    _need(sha256(manifest_raw) == manifest_ref["sha256"], "published case manifest differs")
    manifest = _parse(manifest_raw)
    _need(native_request["manifest_sha256"] == binding["manifest_sha256"] == sha256(canonical(manifest))
          and binding["protocol_sha256"] == sha256(canonical(protocol)) and binding["source_sha256"] == source,
          "actual performance manifest/protocol binding differs")
    policy = manifest["measurement"]
    primary = contract["timing_policy"]["primary"]
    _need(policy["method"] == "cuda_graph" and primary["kind"] == "cuda_event_graph_replay"
          and primary["graph"] is True and primary["statistic"] == "mean" and primary["unit"] == "ms"
          and policy["benchmark_iterations"] == primary["measured_calls"]
          and policy["warmup_iterations"] == primary["warmup_calls"]
          and all(policy[key] == "each_replay" for key in ("refresh_inputs", "initialize_outputs", "validate_outputs")),
          "published graph/reset measurement contract differs")
    count, warmup = policy["benchmark_iterations"], policy["warmup_iterations"]
    expected = {row["case_id"]: row for row in manifest["cases"]}
    rows = report["cases"]
    _need(set(expected) == set(contract["benchmark_case_ids"])
          and len(rows) == len(expected) and {row["case"]["case_id"] for row in rows} == set(expected),
          "actual timing case coverage differs")
    contexts = {}
    for row in rows:
        case = row["case"]["case_id"]
        _need(row["case"] == expected[case] and row["correct"] is True
              and row["benchmark_method"] == "cuda_graph"
              and type(row["warmup_iterations"]) is int and row["warmup_iterations"] == warmup,
              "actual timer, warmup or case differs; fallback is not comparable")
        samples = row["samples_ms"]
        _need(type(samples) is list and len(samples) == count
              and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in samples),
              "actual graph samples are incomplete")
        for key in ("fresh_input_resets", "output_initializations", "oracle_checks"):
            _need(type(row[key]) is int and row[key] == len(samples), "actual replay state checks are incomplete")
        # The published validate_report producer uses fsum/count. The host's
        # validate_phase and observation readers copy that scalar unchanged.
        _need(math.fsum(samples) / count == assessed["per_case_times_ms"][case],
              "native timing scalar differs from actual graph samples")
        contexts[case] = {key: row[key] for key in
                          ("benchmark_method", "warmup_iterations", "fresh_input_resets",
                           "output_initializations", "oracle_checks")}
        contexts[case]["sample_count"] = len(samples)
    scope = protocol["private_cache_scope"]
    _need(scope == "phase", "actual phase cache proof is required for this exporter")
    seeded = _parse(artifacts["native/performance/private-cache-receipt.json"])
    empty = _parse(artifacts["native/performance/empty-cache-receipt.json"])
    declarations = protocol["private_image_caches"]
    _need(set(seeded) == set(declarations) == set(protocol["preserve_cache_vars"]),
          "actual seeded cache inventory differs")
    for variable, row in seeded.items():
        _need(row["seed_manifest_sha256"] == declarations[variable]["manifest_sha256"]
              and row["private_path"] == "protocol-cache/" + variable, "actual cache seed differs")
    _need(type(empty) is dict and not set(empty).intersection(seeded)
          and all(type(path) is str and path.startswith("empty-cache/") and ".." not in Path(path).parts
                  for path in empty.values()), "actual empty-cache preparation differs")
    return {"timing_basis": "cuda_graph", "request_id": native_request["request_id"],
            "per_case": contexts, "cache_scope": scope,
            "cache_seeds": {key: row["seed_manifest_sha256"] for key, row in seeded.items()},
            "empty_cache_paths": empty,
            "evidence": {name: host["artifacts_sha256"][name] for name in
                         (report_name, "native/performance/private-cache-receipt.json",
                          "native/performance/empty-cache-receipt.json")}}


def _generated_timing(request, host, artifacts, assessed):
    """Recompute the generated harness's internal baseline/candidate pair."""
    from .evaluate.generated_report import read_generated_report

    prefix = "observer/"
    source = {row["path"]: row["blob_id"] for row in request["source"]["files"]}
    observed = read_generated_report(request["contract"],
        {name[len(prefix):]: raw for name, raw in artifacts.items() if name.startswith(prefix)}, source)
    _need(not observed["issues"] and observed["phase_validated"] == {"evaluate": True},
          "generated raw report failed validation: " + "; ".join(observed["issues"]))
    for name, raw in observed["artifacts"].items():
        _need(artifacts.get(prefix + name) == raw, "generated retained artifact differs: " + name)
    groups = copy.deepcopy(observed["observations"])
    for group in groups.values():
        group["evidence"] = [prefix + name for name in group["evidence"]]
    _need(groups == host["observations"], "generated report and owner observations differ")
    paired = observed["paired_score"]
    _need(paired["status"] == "passed" and paired["parent_evaluation"] == {
        "operation_id": request["operation_id"], "request_sha256": sha256(canonical(request)),
        "binding": request["binding"]}, "generated report belongs to another owner request")
    _need(paired["candidate_source_id"] == request["binding"]["candidate_source_id"]
          and {row["case_id"]: row["optimized_ms"] for row in paired["per_case"]}
          == assessed["per_case_times_ms"], "generated source or candidate timing scalar differs")
    contexts, diagnostics, cases = {}, {}, {}
    for row in paired["per_case"]:
        case, settings, details = row["case_id"], [], []
        for pair in row["pairs"]:
            detail = {}
            for role in ("baseline", "candidate"):
                leg = pair[role]
                timer = leg["timing_evidence"]
                settings.append({"timer": timer["timer"], "requested": timer["requested"],
                    "effective": timer["effective"], "cache_condition": leg["timing_detail"]["cache_condition"]})
                detail[role] = {key: leg[key] for key in ("leg_id", "timing_detail", "compile_evidence")}
                detail[role]["graph_capture"] = timer["graph_capture"]
            details.append(detail)
        _need(settings and all(value == settings[0] for value in settings),
              "generated baseline/candidate actual timer or cache conditions differ")
        contexts[case], diagnostics[case] = settings[0], details
        cases[case] = {key: row[key] for key in
                       ("baseline_ms", "optimized_ms", "speedup", "reps", "speedup_spread")}
    return {"timing_basis": "generated_task_paired", "request_id": request["operation_id"],
            "admitted_source_id": paired["baseline_source_id"],
            "baseline_implementation": "external_callable_in_generated_baseline_closure",
            "baseline_closure_sha256": paired["baseline_closure_sha256"],
            "per_case": contexts, "paired_cases": cases, "diagnostics": diagnostics,
            "evidence": {name: host["artifacts_sha256"][name] for name in
                         ("observer/paired-score.json", "observer/evaluate/stdout.bin",
                          "observer/evaluate/protocol-binding.json")}}


def read_evaluation(evaluator, candidate_id, *, evaluation_id):
    """Recheck one owner collection under its lock without changing evaluator pins.

    The existing owner root is the trust boundary. An incomplete latest collection
    never falls back to an older pass; transport failures retain their status.
    """
    with evaluator._lock():
        request = evaluator._request(candidate_id, evaluation_id)
        root = evaluator.root / evaluation_id
        _need(_parse(_read(root, "request.json")[0]) == request, "retained evaluation request differs")
        collections = sorted((path for path in root.glob("collection-*")
                              if re.fullmatch(r"collection-[0-9]+", path.name)),
                             key=lambda path: int(path.name[11:]))
        _need(bool(collections), "evaluation has no retained collection")
        collection = collections[-1]
        raw_result = _read(collection, "arena_evaluation.json")[0]
        result = _parse(raw_result)
        _need(result["schema"] == "geak-arena-evaluation-v1"
              and result["evaluation_id"] == evaluation_id and result["binding"] == request["binding"]
              and result["request_sha256"] == sha256(canonical(request))
              and result["collection"] == str(collection), "retained evaluation identity differs")
        _need(all(result[name] is False for name in
                  ("measurement_valid", "custody_proven", "reward_eligible", "framework_final_pass")),
              "Arena observation cannot claim qualification")
        timing = None
        if result["collection_errors"]:
            _need(result["native_suite_pass_observed"] is False, "failed collection cannot report a pass")
        else:
            host = _parse(_read(collection, "host_receipt.json")[0])
            artifacts = {name: _read(collection / "raw", name)[0] for name in host["artifacts_sha256"]}
            settled = evaluator._envelope(request, host, artifacts)
            assessed = evaluator._observations(request, host, artifacts)
            if not settled:
                assessed["native_suite_pass_observed"] = False
            _need(result["settled"] == settled and all(result[key] == value for key, value in assessed.items()),
                  "retained evaluation assessment differs")
            if settled and assessed["native_suite_pass_observed"]:
                reader = (_generated_timing if request["contract"]["schema"] == "geak-arena-generated-task-v1"
                          else _published_timing)
                timing = reader(request, host, artifacts, assessed)
        return {"receipt": result, "request": request,
                "timing_context": timing,
                "receipt_ref": {"path": str(collection / "arena_evaluation.json"), "sha256": sha256(raw_result)}}


def _source(store, source_id):
    manifest = store.source(source_id)
    result = {}
    for row in manifest["files"]:
        name, identity, mode = row["path"], row["blob_id"], row["executable"]
        _need(mode in (0, 0o111) and not any(ord(c) < 32 for c in name),
              "source path or executable mode cannot be represented in a Git patch")
        body = None if identity is None else read_ref(
            {"path": str(store.root / "objects" / identity), "sha256": identity})
        result[name] = (body, mode)
    return result


def _patch(original, selected):
    """Git text patch, retaining absent/empty files, modes and final newlines."""
    _need(set(original) == set(selected), "selected editable source closure differs")
    lines = []
    for name in sorted(original):
        before, before_mode = original[name]
        after, after_mode = selected[name]
        if (before, before_mode) == (after, after_mode):
            continue
        # Binary source is retained in the store, but this text exporter rejects
        # it explicitly rather than writing a patch that changes its bytes.
        old = "" if before is None else before.decode("utf-8")
        new = "" if after is None else after.decode("utf-8")
        _need("\x00" not in old + new, "binary source needs a binary patch exporter")
        a, b = (json.dumps(prefix + name, ensure_ascii=False) for prefix in ("a/", "b/"))
        mode_a, mode_b = ("100755" if mode else "100644" for mode in (before_mode, after_mode))
        lines.append(f"diff --git {a} {b}\n")
        if before is None:
            lines.append(f"new file mode {mode_b}\n")
        elif after is None:
            lines.append(f"deleted file mode {mode_a}\n")
        elif mode_a != mode_b:
            lines.extend([f"old mode {mode_a}\n", f"new mode {mode_b}\n"])
        def git_hash(raw):
            return "0" * 40 if raw is None else hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        lines.append(f"index {git_hash(before)}..{git_hash(after)}\n")
        def text_lines(text):
            parts = text.split("\n")
            return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
        for line in difflib.unified_diff(text_lines(old), text_lines(new),
                                        fromfile="/dev/null" if before is None else a,
                                        tofile="/dev/null" if after is None else b):
            lines.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(lines).encode("utf-8")


def _matched(evaluator, baseline_id, selected_id, baseline_evaluation_id, selected_evaluation_id):
    records = {role: read_evaluation(evaluator, candidate, evaluation_id=evaluation)
               for role, candidate, evaluation in
               (("baseline", baseline_id, baseline_evaluation_id),
                ("candidate", selected_id, selected_evaluation_id))}
    _need(baseline_evaluation_id != selected_evaluation_id, "distinct baseline and candidate evaluations required")
    runtime = None
    for role, record in records.items():
        result = record["receipt"]
        _need(result["settled"] is True and result["native_suite_pass_observed"] is True
              and not result["collection_errors"] and not result["host_errors"],
              role + " evaluation did not complete a passing native suite")
        observed = result["observed_runtime"]
        actual = {key: observed[key] for key in ("image", "config_digest", "physical_devices")}
        _need({key: actual[key] for key in ("image", "config_digest")} == evaluator.contract["runtime"]
              and actual["physical_devices"], role + " runtime/device identity is unavailable or differs")
        _need(runtime is None or actual == runtime, "baseline and selected runtime/device differ")
        runtime = actual
        _need(set(result["per_case_times_ms"]) == set(evaluator.contract["benchmark_case_ids"]),
              role + " measurements do not cover the complete benchmark domain")
    _need(records["baseline"]["timing_context"]["request_id"] != records["candidate"]["timing_context"]["request_id"],
          "baseline and selected reused a performance request")
    timing = [{key: value for key, value in records[role]["timing_context"].items()
               if key not in ("evidence", "request_id", "paired_cases", "diagnostics")}
              for role in ("baseline", "candidate")]
    _need(timing[0] == timing[1], "baseline and selected actual timer/cache conditions differ")
    return records, runtime


def export_final_result(store, snapshot, *, evaluator, selection_options, baseline_candidate_id,
                        baseline_evaluation_id, selected_evaluation_id, output_dir,
                        objective=None):
    """Write the existing kernel-result fields and exact selected text patch.

    ``selection_options`` is passed to ``validate_final_recommendation``: the
    existing expected binding, clock ref, trusted reader and choice authenticator.
    The explicitly named baseline must equal this task's admitted source bytes.
    Measurements come from this evaluator's retained owner collections, never
    from caller-supplied ratios or the population development score.

    ``objective`` is the common lane's mapping: ``kind`` is ``geomean`` (the
    default) or ``time_weighted``. Its optional ``weights`` are positive workload
    time shares covering every case, used as sum(w) / sum(w / paired_speedup).
    The mapping, including workload provenance, is retained exactly as supplied;
    this does not authenticate a workload capture. Numeric export rereads native
    graph samples or generated-task raw paired evidence. A generated task uses
    BOTH legs from the selected evaluation; the separate ORIGINAL evaluation
    remains a reference and its timings are not pooled. Actual timer/cache
    conditions must match. "improved" means only a ratio above one;
    the command/lane must apply its incumbent and minimum-improvement threshold.
    """
    _need(isinstance(evaluator, ArenaEvaluator) and evaluator.store is store,
          "the selected store's existing Arena evaluator is required")
    objective = {"kind": "geomean"} if objective is None else copy.deepcopy(objective)
    _need(type(objective) is dict and objective.get("kind") in ("geomean", "time_weighted"),
          "unsupported lane objective")
    kind, weights = objective["kind"], objective.get("weights")
    cases = evaluator.contract["benchmark_case_ids"]
    _need(weights is None or (type(weights) is dict and set(weights) == set(cases)
          and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in weights.values())),
          "complete positive lane workload weights required")
    _need(kind != "time_weighted" or weights is not None,
          "weighted objective requires explicit lane workload weights")
    out = Path(output_dir)
    _need(out.is_absolute() and out.parent.resolve(strict=True) == out.parent
          and not out.exists() and not out.is_symlink(), "new canonical result directory required")
    result = {"eval_dir": str(out), "final_patch": "", "final_geomean": None,
              "final_weighted": None, "final_arithmetic": None, "final_speedup": None,
              "workload_aligned": kind == "time_weighted", "timing_basis": "unknown",
              "validation_status": "selection_rejected", "outcome": "error",
              "objective": objective,
              "timing_policy": copy.deepcopy(evaluator.contract["timing_policy"]),
              "measurement_valid": False, "custody_proven": False, "reward_eligible": False,
              "framework_final_pass": False, "campaign_id": snapshot["campaign_id"],
              "campaign_revision": snapshot["revision"], "campaign_phase": snapshot["state"]["phase"],
              "scope": "Provisional native comparison against admitted task source; final Director and workload validation remain required."}
    selection, records, patch, selected_source_id = None, None, b"", None
    try:
        selection = validate_final_recommendation(store, snapshot, **selection_options)
        outcome = final_outcome(selection)
        result.update(outcome=outcome["outcome"], detail=outcome["detail"],
                      validation_status="selection_missing")
        if outcome["outcome"] == "no_improvement":
            result["validation_status"] = "kept_original_unmeasured"
        if outcome["outcome"] == "selected":
            choice = outcome["best"]
            result.update(validation_status="measurement_rejected", selected_candidate_id=choice["candidate_id"],
                          selected_source_id=choice["source_id"], baseline_evaluation_id=baseline_evaluation_id,
                          selected_evaluation_id=selected_evaluation_id)
            selected = store.get_candidate(choice["candidate_id"])
            baseline = store.get_candidate(baseline_candidate_id)
            _need(selected["source_id"] == choice["source_id"]
                  and choice["binding"]["campaign_id"] == snapshot["campaign_id"]
                  and all(choice["binding"][key] == selected[key] == baseline[key]
                          for key in ("job_id", "measurement_epoch_id")), "selected/baseline domain differs")
            original = _source(store, baseline["source_id"])
            _need(set(original) == set(evaluator.contract["source_paths"]), "baseline source closure differs")
            for name, identity in original.items():
                _need(identity == _read(evaluator.artifact_root, name),
                      "baseline is not the exact admitted task source: " + name)
            selected_source_id = selected["source_id"]
            selected_source = _source(store, selected_source_id)
            records, runtime = _matched(evaluator, baseline_candidate_id, choice["candidate_id"],
                                        baseline_evaluation_id, selected_evaluation_id)
            paired = records["candidate"]["timing_context"].get("paired_cases")
            if paired is None:
                before, after = (records[role]["receipt"]["per_case_times_ms"] for role in ("baseline", "candidate"))
            else:
                before = {case: paired[case]["baseline_ms"] for case in cases}
                after = {case: paired[case]["optimized_ms"] for case in cases}
                result["comparison_basis"] = "selected_evaluation_internal_baseline_candidate_pair"
                result["baseline_closure_sha256"] = records["candidate"]["timing_context"]["baseline_closure_sha256"]
                result["scope"] = ("Provisional generated-harness comparison against the external baseline closure. "
                                   "baseline_source_id names the admitted editable source, not that external callable. "
                                   "Final Director and workload validation remain required.")
            ratios = {case: before[case] / after[case] for case in cases}
            geometric = math.exp(math.fsum(math.log(ratios[case]) for case in cases) / len(cases))
            weighted = (math.fsum(weights.values()) /
                        math.fsum(weights[case] / ratios[case] for case in cases)) if weights is not None else None
            primary = weighted if kind == "time_weighted" else geometric
            _need(all(math.isfinite(value) and value > 0 for value in [geometric, primary, *ratios.values()]),
                  "nonfinite measured ratio")
            measured = measured_outcome(outcome, primary)
            result.update(outcome=measured["outcome"], detail=measured["detail"], final_geomean=geometric,
                          final_weighted=weighted, final_arithmetic=math.fsum(ratios.values()) / len(cases),
                          final_speedup=primary, runtime=runtime, baseline_candidate_id=baseline_candidate_id,
                          timing_basis=records["candidate"]["timing_context"]["timing_basis"],
                          timing_context={role: record["timing_context"] for role, record in records.items()},
                          baseline_source_id=baseline["source_id"], job_id=evaluator.contract_id,
                          measurement_epoch_id=evaluator.epoch,
                          measurements={role: record["receipt_ref"] for role, record in records.items()},
                          per_case=[{"case_id": case, "baseline_ms": before[case], "candidate_ms": after[case],
                                     "speedup": ratios[case]} for case in cases],
                          validation_status="export_rejected")
            patch = _patch(original, selected_source)
            result["validation_status"] = "provisional_native_observation"
            if not patch:
                result.update(outcome="no_improvement", detail="selected_source_unchanged")
    except (EvoError, ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        result.update(outcome="error", error_kind=type(exc).__name__, error=str(exc))
        result.update(final_geomean=None, final_weighted=None, final_arithmetic=None, final_speedup=None)
        patch, selected_source_id = b"", None
    out.mkdir(mode=0o700)
    if selection is not None:
        (out / "selection.json").write_bytes(canonical_json(selection) + b"\n")
    if records is not None:
        (out / "measurements.json").write_bytes(canonical_json(records) + b"\n")
    if selected_source_id is not None:
        store.materialize(selected_source_id, out / "selected-source")
    if patch:
        patch_path = out / "selected_patch.diff"; patch_path.write_bytes(patch)
        result["selected_patch"] = str(patch_path)
        if result["outcome"] == "improved":
            result["final_patch"] = str(patch_path)
    (out / "result.json").write_bytes(canonical_json(result) + b"\n")
    return result
