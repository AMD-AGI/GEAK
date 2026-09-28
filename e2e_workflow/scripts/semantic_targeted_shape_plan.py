#!/usr/bin/env python3
"""Build a third-pass Shape capture plan from unresolved Semantic rows.

The second capture remains the source of graph-construction boundaries and
broad runtime metadata.  This planner selects only rows that are still ``U``
after that merge and for which the Semantics Agent supplied a source-backed
callable or wrapper mapping.  It never derives ownership from a Kernel name by
itself: the Kernel regular expression is only a selector inside an explicitly
reviewed source mapping.
"""
import argparse
import copy
import json
import os
import re


_PERMANENTLY_INELIGIBLE = {
    "runtime_copy_without_unique_tensor",
    "runtime_internal_buffer_operation",
    "unnamed_runtime_kernel",
}

_RUNTIME_MAPPING_FIELDS = (
    "candidate_op_instance_id",
    "runtime_marker_evidence",
    "source_callable_evidence",
    "source_wrapper_evidence",
    "shape_log_layer_evidence",
    "targeted_region_evidence",
    "kernel_trace_shape",
    "runtime_marker_mapping_status",
    "runtime_marker_candidate_count",
)


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _source_allowed(path, runtime_sources):
    absolute = os.path.abspath(path)
    for source in runtime_sources:
        root = os.path.abspath(source)
        if os.path.isfile(root) and absolute == root:
            return True
        if os.path.isdir(root) and os.path.commonpath(
                (absolute, root)) == root:
            return True
    return False


def _validate_source_evidence(spec, runtime_sources, label):
    evidence = spec.get("source_evidence") or spec.get("source")
    if isinstance(evidence, dict):
        evidence = [evidence]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("%s requires source_evidence" % label)
    validated = []
    for item in evidence:
        path = os.path.abspath(str(item.get("path") or ""))
        if not _source_allowed(path, runtime_sources):
            raise ValueError(
                "%s cites an unapproved runtime source: %s" % (label, path))
        start = int(item.get("line_start", 0) or 0)
        end = int(item.get("line_end", 0) or 0)
        symbol = str(item.get("symbol") or "").strip()
        claim = str(item.get("claim") or "").strip()
        if start <= 0 or end < start or not symbol or not claim:
            raise ValueError(
                "%s source evidence requires line range, symbol, and claim"
                % label)
        with open(path, errors="replace") as fh:
            lines = fh.readlines()
        if end > len(lines):
            raise ValueError("%s source evidence exceeds file" % label)
        excerpt = "".join(lines[start - 1:end])
        if symbol not in excerpt:
            raise ValueError(
                "%s symbol %r is absent from cited lines" % (label, symbol))
        value = copy.deepcopy(item)
        value["path"] = path
        validated.append(value)
    return validated


def _selector_matches(spec, row):
    if spec.get("row_ids") and row.get("row_id") not in spec["row_ids"]:
        return False
    if spec.get("phase") and str(spec["phase"]).lower() != str(
            row.get("phase") or "").lower():
        return False
    if spec.get("pattern_id") and spec["pattern_id"] != row.get("pattern_id"):
        return False
    pos = int(row.get("pos", -1))
    if spec.get("pos_start") is not None and pos < int(spec["pos_start"]):
        return False
    if spec.get("pos_end") is not None and pos > int(spec["pos_end"]):
        return False
    pattern = re.compile(str(spec.get("kernel_pattern") or ""))
    return bool(pattern.search(str(row.get("raw_name") or "")))


def _validate_mappings(agent_plan, runtime_sources):
    if agent_plan.get("producer") != "semantics_mapper_agent":
        raise ValueError(
            "targeted Shape mappings must be produced by semantics_mapper_agent")
    if agent_plan.get("mapping_claim") not in (
            "source_call_path_candidates", "source_verified_targets"):
        raise ValueError(
            "targeted probe plan requires a source-backed mapping_claim")

    callables = []
    for index, raw in enumerate(agent_plan.get("callable_kernel_map") or []):
        spec = copy.deepcopy(raw)
        label = "callable_kernel_map[%d]" % index
        target = str(spec.get("target") or "").strip()
        if ":" not in target:
            raise ValueError("%s target must be module:attribute" % label)
        pattern = str(spec.get("kernel_pattern") or "").strip()
        if not pattern:
            raise ValueError("%s requires kernel_pattern" % label)
        re.compile(pattern)
        scope = str(spec.get("scope") or "wrapper")
        if scope not in ("kernel", "wrapper"):
            raise ValueError("%s scope must be kernel or wrapper" % label)
        spec["target"] = target
        spec["kernel_pattern"] = pattern
        spec["scope"] = scope
        spec["source_evidence"] = _validate_source_evidence(
            spec, runtime_sources, label)
        spec["source"] = spec["source_evidence"]
        callables.append(spec)

    wrappers = []
    for index, raw in enumerate(agent_plan.get("source_wrapper_map") or []):
        spec = copy.deepcopy(raw)
        label = "source_wrapper_map[%d]" % index
        if not str(spec.get("op_path") or "").strip():
            raise ValueError("%s requires op_path" % label)
        pattern = str(spec.get("kernel_pattern") or "").strip()
        if not pattern:
            raise ValueError("%s requires kernel_pattern" % label)
        re.compile(pattern)
        spec["source_evidence"] = _validate_source_evidence(
            spec, runtime_sources, label)
        spec["source"] = spec["source_evidence"]
        wrappers.append(spec)
    return callables, wrappers


def _callable_targets(agent_plan, callable_maps):
    result = []
    seen = set()
    supplied = list(agent_plan.get("callable_targets") or [])
    supplied.extend({"callable": spec["target"],
                     "evidence": "source_backed_targeted_retry",
                     "mapping_claim": "none"}
                    for spec in callable_maps)
    for item in supplied:
        value = copy.deepcopy(item) if isinstance(item, dict) else {
            "callable": str(item),
            "evidence": "source_backed_targeted_retry",
            "mapping_claim": "none",
        }
        name = str(value.get("callable") or "").strip()
        if not name or name in seen:
            continue
        if ":" not in name:
            raise ValueError(
                "targeted callable must be module:attribute: %s" % name)
        value["callable"] = name
        seen.add(name)
        result.append(value)
    return result


def build(semantic_table_path, mapped_plan_path, agent_plan_path,
          capture_setup_path, runtime_sources, out_plan_path,
          out_setup_path):
    table = _load(semantic_table_path)
    source_plan = _load(mapped_plan_path)
    agent_plan = _load(agent_plan_path)
    setup = _load(capture_setup_path)
    callable_maps, wrapper_maps = _validate_mappings(
        agent_plan, runtime_sources)

    unresolved = []
    permanently_ineligible = []
    for semantic_table in table.get("tables", []):
        for row in semantic_table.get("rows", []):
            evidence = row.get("semantic_evidence") or {}
            if evidence.get("level") != "U":
                continue
            value = copy.deepcopy(row)
            value.update({
                "phase": semantic_table.get("phase"),
                "pattern_id": semantic_table.get("pattern_id"),
                "representative_layer_id": semantic_table.get(
                    "representative_layer_id"),
            })
            if evidence.get("reason_code") in _PERMANENTLY_INELIGIBLE:
                permanently_ineligible.append({
                    "row_id": value.get("row_id"),
                    "reason_code": evidence.get("reason_code"),
                })
            else:
                unresolved.append(value)

    source_targets = {
        target.get("row_id"): target
        for target in source_plan.get("capture_targets", [])
        if target.get("row_id")}
    selected = []
    unmatched = []
    used_callable_ids = set()
    used_wrapper_ids = set()
    for row in unresolved:
        matches = []
        for index, spec in enumerate(callable_maps):
            if _selector_matches(spec, row):
                matches.append(("callable", index))
        for index, spec in enumerate(wrapper_maps):
            if _selector_matches(spec, row):
                matches.append(("wrapper", index))
        if len(matches) > 1:
            raise ValueError(
                "multiple targeted source mappings match unresolved row %s"
                % row.get("row_id"))
        if not matches:
            unmatched.append({
                "row_id": row.get("row_id"),
                "reason_code": "no_source_backed_targeted_probe",
            })
            continue
        target = source_targets.get(row.get("row_id"))
        if target is None:
            unmatched.append({
                "row_id": row.get("row_id"),
                "reason_code": "missing_second_capture_target",
            })
            continue
        target = copy.deepcopy(target)
        for field in _RUNTIME_MAPPING_FIELDS:
            target.pop(field, None)
        kind, index = matches[0]
        target["targeted_retry_mapping"] = {
            "kind": kind,
            "index": index,
            "agent_plan": os.path.abspath(agent_plan_path),
        }
        selected.append(target)
        if kind == "callable":
            used_callable_ids.add(index)
        else:
            used_wrapper_ids.add(index)

    selected_keys = {
        (target.get("phase"), target.get("pattern_id"),
         int(target.get("representative_layer_id", -1)))
        for target in selected}
    phases = sorted({str(target.get("phase") or "").lower()
                     for target in selected if target.get("phase")})
    representatives = sorted({int(target["representative_layer_id"])
                              for target in selected})
    retry_plan = copy.deepcopy(source_plan)
    retry_plan["scope"] = "targeted_unresolved_shapes_only"
    retry_plan["capture_targets"] = selected
    retry_plan["target_count"] = len(selected)
    retry_plan["representative_layer_filter"] = representatives
    retry_plan["target_buckets"] = [
        bucket for bucket in source_plan.get("target_buckets", [])
        if (bucket.get("phase"), bucket.get("pattern_id"),
            int(bucket.get("representative_layer_id", -1))) in selected_keys]
    retry_plan["patterns"] = [
        pattern for pattern in source_plan.get("patterns", [])
        if any(pattern.get("pattern_id") == key[1]
               and int(pattern.get("representative_layer_id", -1)) == key[2]
               for key in selected_keys)]
    retry_plan["phase_coverage"] = {
        **(source_plan.get("phase_coverage") or {}),
        "required_phases": phases,
    }
    retry_plan["operator_probe_plan"] = {
        "schema_version": 1,
        "producer": "semantics_mapper_agent_targeted_retry",
        "status": "observational_prior_only",
        "mapping_claim": "none",
        "callable_targets": _callable_targets(
            agent_plan, [callable_maps[i] for i in sorted(used_callable_ids)]),
        "operator_targets": copy.deepcopy(
            agent_plan.get("operator_targets") or []),
        "note": (
            "Third-pass observation targets are restricted to rows still U "
            "after the second capture and backed by reviewed runtime source."),
    }
    summary = {
        "status": "ready" if selected else "not_needed",
        "semantic_table": os.path.abspath(semantic_table_path),
        "second_capture_plan": os.path.abspath(mapped_plan_path),
        "agent_probe_plan": os.path.abspath(agent_plan_path),
        "unresolved_row_count": len(unresolved) + len(permanently_ineligible),
        "eligible_unresolved_row_count": len(unresolved),
        "targeted_row_count": len(selected),
        "capture_phases": phases,
        "representative_layers": representatives,
        "permanently_ineligible": permanently_ineligible,
        "not_targeted": unmatched,
    }
    retry_plan["targeted_shape_retry"] = summary
    retry_plan["status"] = summary["status"]

    targeted_setup = copy.deepcopy(setup)
    targeted_setup.pop("operator_probe_plan", None)
    targeted_setup["capture_phases"] = phases
    targeted_setup["callable_kernel_map"] = [
        callable_maps[i] for i in sorted(used_callable_ids)]
    targeted_setup["source_wrapper_map"] = [
        wrapper_maps[i] for i in sorted(used_wrapper_ids)]
    targeted_setup["targeted_shape_retry"] = {
        "source_semantic_table": os.path.abspath(semantic_table_path),
        "agent_probe_plan": os.path.abspath(agent_plan_path),
        "targeted_row_count": len(selected),
    }

    for path, document in (
            (out_plan_path, retry_plan),
            (out_setup_path, targeted_setup)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(document, fh, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-table", required=True)
    parser.add_argument("--mapped-capture-plan", required=True)
    parser.add_argument("--agent-probe-plan", required=True)
    parser.add_argument("--capture-setup", required=True)
    parser.add_argument("--runtime-source", action="append", default=[])
    parser.add_argument("--out-plan", required=True)
    parser.add_argument("--out-setup", required=True)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = build(
        args.semantic_table, args.mapped_capture_plan,
        args.agent_probe_plan, args.capture_setup,
        args.runtime_source, args.out_plan, args.out_setup)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
