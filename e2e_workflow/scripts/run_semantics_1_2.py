#!/usr/bin/env python3
"""Run the complete Semantics Mapping 1.2 pipeline using GEAK scripts only."""
import argparse
import hashlib
import json
import os
import shutil
import sys

import semantic_kernel_mapping
import semantic_evidence_ledger
import semantic_layer_boundary_transfer
import semantic_runtime_marker_mapping
import semantic_shape_merge
import semantic_source_mapping
import semantic_targeted_shape_plan
import validate_structural_patterns
import run_semantic_shape_capture

# Both model phases are required by default. See the require_phases block in
# run() for why this is not an env-var opt-in any more.
DEFAULT_REQUIRE_PHASES = ("prefill", "decode")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representative_hints(table_path):
    with open(table_path) as fh:
        document = json.load(fh)
    hints = {}
    for table in document.get("tables", []):
        pattern_id = table.get("pattern_id")
        layer_id = table.get("representative_layer_id")
        if pattern_id is not None and layer_id is not None:
            hints.setdefault(pattern_id, set()).add(int(layer_id))
    return {key: sorted(values) for key, values in hints.items()}


def run(config_path, trace_path, shape_log_path, out_dir,
        config_key="", runtime_sources=None, capture_setup_path="",
        capture_result_path="", capture_result_paths=None,
        structural_patterns_path="", require_phases=None,
        targeted_probe_plan_path="", targeted_capture_setup_path=""):
    os.makedirs(out_dir, exist_ok=True)
    runtime_sources = list(runtime_sources or [])
    if not structural_patterns_path:
        raise ValueError(
            "structural_patterns_path is required; Layer Patterns must be "
            "defined by the semantics_mapper Agent from config and runtime "
            "source before Semantics 1.2")
    structural_patterns_input = os.path.abspath(
        structural_patterns_path)
    structural_patterns_input_sha256 = _sha256(
        structural_patterns_input)
    callable_kernel_map = []
    source_wrapper_map = []
    for source_path in runtime_sources:
        if not source_path.endswith(".json"):
            continue
        try:
            with open(source_path) as fh:
                source_config = json.load(fh)
            callable_kernel_map.extend(
                source_config.get("callable_kernel_map", []))
            source_wrapper_map.extend(
                source_config.get("source_wrapper_map", []))
        except (OSError, ValueError):
            pass
    patterns_path = os.path.join(out_dir, "STRUCTURAL_LAYER_PATTERNS.json")
    structural_validation = validate_structural_patterns.validate(
        structural_patterns_input, config_path, runtime_sources,
        patterns_path)

    # trace_path may be a single trace or a list; the mapper auto-adopts the
    # EXTEND/DECODE phase sibling so a run cannot silently cover one phase.
    #
    # Both phases are required BY DEFAULT. This used to read an env var that
    # defaulted to EMPTY, i.e. "require no phase at all" — so unless a caller
    # happened to export GEAK_SEMANTICS_REQUIRE_PHASES, a prefill-only capture
    # sailed through and every downstream fusion decision was made on half the
    # model's execution. A decode-blind semantic table is not a lighter table,
    # it is a wrong one: decode is where the small-tensor fusions actually pay.
    #
    # GEAK_SEMANTICS_REQUIRE_PHASES may still NARROW the requirement (e.g. a
    # deliberate prefill-only probe run sets it to "prefill"), but that is now
    # an explicit, logged act rather than the silent default.
    if require_phases is None:
        configured = os.environ.get("GEAK_SEMANTICS_REQUIRE_PHASES")
        if configured is None:
            require_phases = list(DEFAULT_REQUIRE_PHASES)
        else:
            require_phases = [
                value.strip() for value in configured.split(",")
                if value.strip()]
            narrowed = [
                phase for phase in DEFAULT_REQUIRE_PHASES
                if phase not in require_phases]
            if narrowed:
                print(
                    "[semantics] GEAK_SEMANTICS_REQUIRE_PHASES=%r NARROWS the "
                    "default two-phase requirement; dropping %s. Fusion "
                    "candidates for the dropped phase(s) will not be "
                    "trustworthy." % (configured, narrowed),
                    file=sys.stderr)
    semantic = semantic_kernel_mapping.build(
        trace_path, patterns_path, out_dir, require_phases=require_phases)

    # --- Initial layer-boundary evidence -------------------------------------
    # semantic_kernel_mapping resolves per-layer boundaries only from complete
    # independent scopes. A graph-replayed phase without Python module spans
    # stays unresolved until the graph-construction all-layer marker transfer
    # below supplies validated cuts.
    with open(semantic["layer_instance_audit_json"]) as fh:
        module_scope_count = int(
            json.load(fh).get("module_scope_count", 0) or 0)
    boundary_evidence = (
        "module_span" if module_scope_count > 0
        else "degraded_no_module_span")

    phase_1_1_json = os.path.join(
        out_dir, "pattern_layer_kernel_table_1_1.json")
    phase_1_1_md = os.path.join(
        out_dir, "ORDERED_UNIQUE_LAYER_TABLES_1_1.md")
    shutil.copyfile(semantic["semantic_table_json"], phase_1_1_json)
    shutil.copyfile(semantic["semantic_table_md"], phase_1_1_md)
    semantic["semantic_table_json"] = phase_1_1_json
    semantic["semantic_table_md"] = phase_1_1_md
    source_plan_path = os.path.join(
        out_dir, "SHAPE_CAPTURE_PLAN_SOURCE_MAPPED.json")
    semantic_source_mapping.map_plan(
        semantic["shape_capture_plan_json"], runtime_sources,
        source_plan_path)

    capture_results = []
    if capture_setup_path:
        capture_results.append(run_semantic_shape_capture.capture(
            capture_setup_path, source_plan_path,
            os.path.join(out_dir, "capture")))
    result_paths = list(capture_result_paths or [])
    if capture_result_path:
        if isinstance(capture_result_path, (list, tuple)):
            result_paths.extend(capture_result_path)
        else:
            result_paths.append(capture_result_path)
    for path in result_paths:
        with open(path) as fh:
            capture_results.append(json.load(fh))
    if not shape_log_path and not capture_results:
        raise ValueError(
            "shape_log_path, capture_setup_path, or capture_result_path "
            "is required")

    # A graph-replayed Clean Trace has correct device timing but may have no
    # Python layer scopes.  The graph-construction replay now emits one
    # lightweight marker for every main layer.  Transfer only those validated
    # cuts, then rebuild the Clean Trace table before any Shape evidence is
    # merged.  Shape completion is never allowed to upgrade a bad boundary.
    boundary_transfers = []
    boundary_map_paths = []
    initial_hints = _representative_hints(phase_1_1_json)
    for index, capture_result in enumerate(capture_results):
        if (capture_result.get("shape_capture_execution") != "graph_capture"
                or not capture_result.get("capture_trace")):
            continue
        boundary_dir = os.path.join(
            out_dir, "boundary_transfers", "run_%02d" % index)
        boundary_path = os.path.join(
            boundary_dir, "LAYER_BOUNDARY_TRANSFER.json")
        transfer = semantic_layer_boundary_transfer.transfer(
            capture_result["capture_trace"], trace_path,
            patterns_path, boundary_path)
        capture_result["layer_boundary_transfer"] = transfer
        boundary_transfers.append(transfer)
        if transfer.get("status") in ("pass", "partial"):
            boundary_map_paths.append(boundary_path)

    if boundary_map_paths:
        semantic = semantic_kernel_mapping.build(
            trace_path, patterns_path, out_dir,
            require_phases=require_phases,
            boundary_map_paths=boundary_map_paths,
            representative_layer_hints=initial_hints)
        shutil.copyfile(semantic["semantic_table_json"], phase_1_1_json)
        shutil.copyfile(semantic["semantic_table_md"], phase_1_1_md)
        semantic["semantic_table_json"] = phase_1_1_json
        semantic["semantic_table_md"] = phase_1_1_md
        semantic_source_mapping.map_plan(
            semantic["shape_capture_plan_json"], runtime_sources,
            source_plan_path)

    probe_tables = []
    probe_runs = []
    if shape_log_path:
        direct_dir = os.path.join(out_dir, "probe_runs", "direct")
        direct = semantic_shape_merge.merge(
            phase_1_1_json, source_plan_path, shape_log_path, direct_dir)
        probe_tables.append(direct["semantic_table_json"])
        probe_runs.append({
            "kind": "direct_shape_log",
            "shape_log": os.path.abspath(shape_log_path),
            "shape_merge": direct,
        })
    for index, capture_result in enumerate(capture_results):
        run_dir = os.path.join(out_dir, "probe_runs", "run_%02d" % index)
        os.makedirs(run_dir, exist_ok=True)
        merge_plan_path = os.path.join(
            run_dir, "SHAPE_CAPTURE_PLAN_RUNTIME_MAPPED.json")
        marker_mapping = semantic_runtime_marker_mapping.map_plan(
            source_plan_path, capture_result["capture_trace"],
            merge_plan_path, capture_result.get("shape_log", ""),
            capture_result.get(
                "callable_kernel_map", callable_kernel_map),
            capture_result.get(
                "source_wrapper_map", source_wrapper_map),
            clean_table_path=phase_1_1_json,
            required_phases=capture_result.get("capture_phases"),
            operator_schema_manifest_path=capture_result.get(
                "operator_schema_manifest", ""))
        capture_result["runtime_marker_mapping"] = marker_mapping
        merged_probe = semantic_shape_merge.merge(
            phase_1_1_json, merge_plan_path,
            capture_result["shape_log"], run_dir)
        probe_tables.append(merged_probe["semantic_table_json"])
        probe_runs.append({
            "kind": "runtime_capture",
            "capture": capture_result,
            "mapped_plan": merge_plan_path,
            "shape_merge": merged_probe,
        })

    merged_dir = os.path.join(out_dir, "semantics_1_2")
    merged = semantic_evidence_ledger.merge(
        phase_1_1_json, probe_tables, merged_dir)
    targeted_shape_retry = {"status": "not_requested"}
    if targeted_probe_plan_path:
        runtime_probe_runs = [
            run for run in probe_runs if run.get("kind") == "runtime_capture"]
        if not runtime_probe_runs:
            raise ValueError(
                "targeted Shape retry requires a completed second runtime capture")
        retry_setup_input = (
            targeted_capture_setup_path or capture_setup_path)
        if not retry_setup_input:
            raise ValueError(
                "targeted Shape retry requires --targeted-capture-setup "
                "or --capture-setup")
        retry_plan_path = os.path.join(
            out_dir, "TARGETED_SHAPE_CAPTURE_PLAN.json")
        retry_setup_path = os.path.join(
            out_dir, "TARGETED_SHAPE_CAPTURE_SETUP.json")
        targeted_shape_retry = semantic_targeted_shape_plan.build(
            merged["semantic_table_json"],
            runtime_probe_runs[-1]["mapped_plan"],
            targeted_probe_plan_path, retry_setup_input,
            runtime_sources, retry_plan_path, retry_setup_path)
        targeted_shape_retry.update({
            "capture_plan": retry_plan_path,
            "capture_setup": retry_setup_path,
        })
        if targeted_shape_retry["status"] == "ready":
            retry_capture = run_semantic_shape_capture.capture(
                retry_setup_path, retry_plan_path,
                os.path.join(out_dir, "capture_targeted"),
                phases=targeted_shape_retry["capture_phases"])
            retry_run_dir = os.path.join(
                out_dir, "probe_runs", "targeted")
            os.makedirs(retry_run_dir, exist_ok=True)
            retry_mapped_plan = os.path.join(
                retry_run_dir, "TARGETED_SHAPE_CAPTURE_PLAN_RUNTIME_MAPPED.json")
            retry_mapping = semantic_runtime_marker_mapping.map_plan(
                retry_plan_path, retry_capture["capture_trace"],
                retry_mapped_plan, retry_capture.get("shape_log", ""),
                retry_capture.get("callable_kernel_map", []),
                retry_capture.get("source_wrapper_map", []),
                clean_table_path=phase_1_1_json,
                required_phases=retry_capture.get("capture_phases"),
                operator_schema_manifest_path=retry_capture.get(
                    "operator_schema_manifest", ""))
            retry_capture["runtime_marker_mapping"] = retry_mapping
            retry_probe = semantic_shape_merge.merge(
                phase_1_1_json, retry_mapped_plan,
                retry_capture["shape_log"], retry_run_dir)
            probe_tables.append(retry_probe["semantic_table_json"])
            probe_runs.append({
                "kind": "targeted_runtime_capture",
                "capture": retry_capture,
                "mapped_plan": retry_mapped_plan,
                "shape_merge": retry_probe,
            })
            capture_results.append(retry_capture)
            merged = semantic_evidence_ledger.merge(
                phase_1_1_json, probe_tables, merged_dir)
            targeted_shape_retry.update({
                "status": "completed",
                "capture": retry_capture,
                "runtime_marker_mapping": retry_mapping,
                "shape_merge": retry_probe,
            })
    published_json = os.path.join(
        out_dir, "pattern_layer_kernel_table.json")
    published_md = os.path.join(
        out_dir, "ORDERED_UNIQUE_LAYER_TABLES.md")
    shutil.copyfile(merged["semantic_table_json"], published_json)
    shutil.copyfile(merged["semantic_table_md"], published_md)
    capture_phase_coverage_complete = all(
        capture.get("runtime_marker_mapping", {}).get(
            "phase_coverage_complete", False)
        for capture in capture_results)
    graph_capture_verified_phases = sorted({
        str(group.get("phase") or "").lower()
        for transfer in boundary_transfers
        if transfer.get("status") in ("pass", "partial")
        for group in transfer.get("mapped_groups", [])
        if group.get("phase")})
    boundary_match_rules = sorted({
        str(group.get("match_rule") or "")
        for transfer in boundary_transfers
        if transfer.get("status") in ("pass", "partial")
        for group in transfer.get("mapped_groups", [])
        if group.get("match_rule")})
    boundary_rebuild = {
        "applied": bool(boundary_map_paths),
        "status": (
            "validated_graph_capture_layer_scopes_applied"
            if boundary_map_paths else "unavailable"),
        "reason": (
            "complete 0..N-1 graph-construction layer markers supplied cuts; "
            "the donor-to-Clean-Trace transfer preserved every validated layer "
            "core for the same phase and workload bucket; any unsupported "
            "inter-layer gaps remain explicit residual rows"),
        "match_rules": boundary_match_rules,
        "verified_phases": graph_capture_verified_phases,
        "transfers": boundary_transfers,
        "residual_range_count": sum(
            int(transfer.get("residual_range_count", 0) or 0)
            for transfer in boundary_transfers
            if transfer.get("status") in ("pass", "partial")),
        "traces": [
            os.path.abspath(transfer["donor"]["path"])
            for transfer in boundary_transfers
            if transfer.get("donor", {}).get("path")],
    }
    # --- Per-phase boundary evidence -----------------------------------------
    # `boundary_evidence` above is an AGGREGATE over the whole run: prefill's 61
    # module spans can set it to "module_span" even when another phase has no
    # authoritative boundary at all. Grade each phase separately and require
    # every published table to carry module or validated donor scope evidence.
    phase_boundary_evidence = {}
    try:
        with open(phase_1_1_json) as fh:
            for table in json.load(fh).get("tables", []):
                phase = table.get("phase")
                if not phase:
                    continue
                levels = {row.get("layer_evidence")
                          for row in table.get("rows", [])}
                degraded = {level for level in levels
                            if not str(level).startswith(
                                ("module_span", "python_module_span",
                                 "validated_graph_capture_layer_scope"))}
                phase_boundary_evidence[phase] = (
                    "degraded:" + ",".join(sorted(str(x) for x in degraded))
                    if degraded else "authoritative_scope")
    except Exception as exc:  # pragma: no cover - diagnostics only
        phase_boundary_evidence = {"error": str(exc)}
    degraded_phases = sorted(
        phase for phase, level in phase_boundary_evidence.items()
        if level != "authoritative_scope")
    blocking_degraded_phases = list(degraded_phases)

    status = "pass" if (
        semantic["status"] == "pass"
        and merged["status"] == "pass"
        and capture_phase_coverage_complete
        and not blocking_degraded_phases
    ) else "fail"
    result = {
        "schema_version": 1,
        "pipeline": "geak_semantics_1_2",
        "evidence_policy": {
            "levels": ["K", "P", "U"],
            "K": "clean trace Input Dims via External id",
            "P": (
                "graph-capture trace External-ID shape or runtime "
                "shape_logger probe (kernel or wrapper scope)"),
            "U": "unavailable after probes with mandatory reason_code",
            "priority": ["K", "P(kernel)", "P(wrapper)", "U"],
            "additive_across_probe_runs": True,
        },
        "status": status,
        "capture_phase_coverage_complete": (
            capture_phase_coverage_complete),
        "boundary_evidence": boundary_evidence,
        "phase_boundary_evidence": phase_boundary_evidence,
        "degraded_boundary_phases": degraded_phases,
        "blocking_degraded_boundary_phases": blocking_degraded_phases,
        "boundary_rebuild": boundary_rebuild,
        "targeted_shape_retry": targeted_shape_retry,
        "module_scope_count": module_scope_count,
        "inputs": {
            "config": {
                "path": os.path.abspath(config_path),
                "sha256": _sha256(config_path),
            },
            "trace": {
                "path": os.path.abspath(trace_path),
                "sha256": _sha256(trace_path),
            },
            "shape_log": {
                "path": os.path.abspath(shape_log_path),
                "sha256": _sha256(shape_log_path),
            } if shape_log_path else None,
            "runtime_sources": [
                os.path.abspath(path) for path in runtime_sources],
            "agent_structural_patterns": {
                "path": structural_patterns_input,
                "sha256": structural_patterns_input_sha256,
            },
        },
        "structural_patterns_json": patterns_path,
        "structural_pattern_validation": (
            structural_validation.get("validation", {})),
        "semantic_mapping": semantic,
        "shape_merge": merged,
        "probe_runs": probe_runs,
        "runtime_captures": capture_results,
        "published_semantic_table_json": published_json,
        "published_semantic_table_md": published_md,
    }
    result_path = os.path.join(out_dir, "SEMANTICS_1_2_RUN.json")
    result["result_json"] = result_path
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-key", default="")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--shape-log", default="")
    parser.add_argument("--capture-setup", default="")
    parser.add_argument("--capture-result", action="append", default=[])
    parser.add_argument("--runtime-source", action="append", default=[])
    parser.add_argument("--structural-patterns", required=True)
    parser.add_argument("--targeted-probe-plan", default="")
    parser.add_argument("--targeted-capture-setup", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = run(
        args.config, args.trace, args.shape_log, args.out_dir,
        args.config_key, args.runtime_source,
        args.capture_setup, capture_result_paths=args.capture_result,
        structural_patterns_path=args.structural_patterns,
        targeted_probe_plan_path=args.targeted_probe_plan,
        targeted_capture_setup_path=args.targeted_capture_setup)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
