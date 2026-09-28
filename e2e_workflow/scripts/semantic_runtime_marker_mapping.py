#!/usr/bin/env python3
"""Map unresolved Clean Trace rows to GEAK second-replay wrapper markers."""
import argparse
import collections
import gzip
import json
import os
import re


MARKER_PREFIX = "GEAK_SEMANTICS|"
LAYER_SCOPE_PREFIX = "GEAK_LAYER_SCOPE|"
RUNTIME_CATEGORIES = {"cuda_runtime", "hip_runtime"}
DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def _load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        document = json.load(fh)
    return document.get("traceEvents", document)


def _marker_fields(name):
    result = {}
    for token in str(name).split("|")[1:]:
        key, separator, value = token.partition("=")
        if separator:
            result[key] = value
    return result


def _phase(value):
    value = str(value or "").lower()
    return {
        "extend": "prefill",
        "prompt": "prefill",
        "generation": "decode",
    }.get(value, value)


def _integer(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_schema_manifest(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _schema_index(manifest):
    result = collections.defaultdict(list)
    for schema in (manifest or {}).get("schemas", []):
        name = str(schema.get("name") or "")
        qualified = str(schema.get("qualified_name") or "")
        if name:
            result[name].append(schema)
        if qualified and qualified != name:
            result[qualified].append(schema)
    return result


def _schema_type_is_tensor(value):
    return "tensor" in str(value or "").lower()


def _schema_score(schema, dims, types):
    arguments = schema.get("arguments") or []
    observed_count = max(len(dims), len(types))
    required_count = sum(
        not bool(argument.get("has_default"))
        for argument in arguments)
    if observed_count < required_count or observed_count > len(arguments):
        return None
    score = 0
    if observed_count == len(arguments):
        score += 20
    score -= len(arguments) - observed_count
    scalar_names = {
        "bool", "int", "int64", "int64_t", "long", "long int",
        "float", "double", "str", "string", "device", "layout",
        "scalar", "symint", "none",
    }
    for index in range(observed_count):
        dim = dims[index] if index < len(dims) else []
        observed_type = str(
            types[index] if index < len(types) else "").strip().lower()
        tensor_argument = _schema_type_is_tensor(
            arguments[index].get("type"))
        if isinstance(dim, list) and dim:
            if not tensor_argument:
                return None
            score += 4
        elif observed_type:
            scalar_type = observed_type in scalar_names
            if tensor_argument and not scalar_type:
                score += 2
            elif not tensor_argument and scalar_type:
                score += 2
    return score


def _resolve_operator_schema(op_name, dims, types, schema_by_name):
    candidates = list(schema_by_name.get(str(op_name or ""), []))
    if not candidates:
        return {
            "status": "not_registered",
            "operator": op_name,
            "candidate_count": 0,
            "schema": None,
        }
    scored = []
    for schema in candidates:
        score = _schema_score(schema, dims, types)
        if score is not None:
            scored.append((score, schema))
    if not scored:
        return {
            "status": "no_compatible_schema",
            "operator": op_name,
            "candidate_count": len(candidates),
            "schema": None,
        }
    best_score = max(score for score, _ in scored)
    best = [schema for score, schema in scored if score == best_score]
    if len(best) != 1:
        return {
            "status": "ambiguous",
            "operator": op_name,
            "candidate_count": len(best),
            "compatible_schema_count": len(scored),
            "schema": None,
            "candidate_schemas": [
                schema.get("schema") for schema in best[:8]],
        }
    return {
        "status": "matched_unique",
        "operator": op_name,
        "candidate_count": 1,
        "compatible_schema_count": len(scored),
        "match_rule": (
            "unique registered schema compatible with observed argument "
            "count and tensor/scalar positions"),
        "schema": best[0],
    }


def _raw_operands(dims, types, strides=None):
    strides = strides or []
    count = max(len(dims), len(types), len(strides))
    return [
        {
            "arg_index": index,
            "shape": dims[index] if index < len(dims) else None,
            "dtype": types[index] if index < len(types) else None,
            "stride": strides[index] if index < len(strides) else None,
        }
        for index in range(count)]


def _kernel_key(name):
    value = str(name or "")
    value = re.sub(r"GRID_MN_\d+", "GRID_MN_*", value)
    value = re.sub(r"(_grid_)\d+(?=_|$)", r"\1*", value, flags=re.I)
    return value


def _runtime_entries(events, schema_manifest=None):
    markers = []
    schema_by_name = _schema_index(schema_manifest)
    device_by_correlation = {}
    cpu_shapes_by_external_id = collections.defaultdict(list)
    runtime_launch_count_by_external_id = collections.Counter()
    for index, event in enumerate(events):
        name = str(event.get("name", ""))
        args = event.get("args") or {}
        if (event.get("cat") == "user_annotation"
                and name.startswith(MARKER_PREFIX)
                and event.get("ts") is not None
                and event.get("dur") is not None):
            fields = _marker_fields(name)
            markers.append({
                "index": index,
                "name": name,
                "op_instance_id": fields.get("op"),
                "phase": _phase(fields.get("phase")),
                "layer_id": int(fields.get("layer", -1)),
                "batch_size": _integer(fields.get("bs")),
                "input_tokens": _integer(fields.get("toks")),
                "op_path": fields.get("path"),
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "ts": float(event["ts"]),
                "end": float(event["ts"]) + float(event["dur"]),
                "duration_us": float(event["dur"]),
                "external_id": args.get("External id"),
            })
        if event.get("cat") == "cpu_op":
            external_id = args.get("External id")
            input_dims = args.get("Input Dims") or []
            if external_id is not None and any(
                    isinstance(dim, list) and dim for dim in input_dims):
                input_types = args.get("Input type") or []
                input_strides = args.get("Input Strides") or []
                schema_resolution = _resolve_operator_schema(
                    name, input_dims, input_types, schema_by_name)
                cpu_shapes_by_external_id[external_id].append({
                    "op_name": name,
                    "external_id": external_id,
                    "cpu_event_index": index,
                    "input_dims": input_dims,
                    "input_types": input_types,
                    "input_strides": input_strides,
                    "raw_operands": _raw_operands(
                        input_dims, input_types, input_strides),
                    "operator_schema_resolution": {
                        key: value for key, value in schema_resolution.items()
                        if key != "schema"},
                    "operator_schema": schema_resolution.get("schema"),
                })
        if event.get("cat") in RUNTIME_CATEGORIES:
            external_id = args.get("External id")
            correlation = args.get(
                "correlation", args.get("Correlation ID"))
            if (external_id is not None and args.get("kernel")
                    and correlation is not None):
                runtime_launch_count_by_external_id[external_id] += 1
        if event.get("cat") in DEVICE_CATEGORIES:
            correlation = args.get(
                "correlation", args.get("Correlation ID"))
            if correlation is not None:
                device_by_correlation.setdefault(correlation, []).append(event)

    entries = []
    for index, event in enumerate(events):
        if event.get("cat") not in RUNTIME_CATEGORIES:
            continue
        args = event.get("args") or {}
        kernel = args.get("kernel")
        correlation = args.get("correlation", args.get("Correlation ID"))
        if not kernel or correlation is None or event.get("ts") is None:
            continue
        timestamp = float(event["ts"])
        containing = [
            marker for marker in markers
            if marker["pid"] == event.get("pid")
            and marker["tid"] == event.get("tid")
            and marker["ts"] <= timestamp <= marker["end"]]
        if not containing:
            continue
        marker = min(containing, key=lambda item: item["duration_us"])
        device_matches = device_by_correlation.get(correlation, [])
        device = device_matches[0] if len(device_matches) == 1 else None
        external_id = args.get("External id")
        cpu_shapes = cpu_shapes_by_external_id.get(external_id, [])
        kernel_trace_shape = None
        if (external_id is not None and len(cpu_shapes) == 1
                and runtime_launch_count_by_external_id[external_id] == 1):
            kernel_trace_shape = dict(cpu_shapes[0])
            kernel_trace_shape.update({
                "source": "graph_capture_trace_external_id",
                "mapping_cardinality": "1:1",
                "runtime_launch_count": 1,
            })
        entries.append({
            "runtime_event_index": index,
            "runtime_name": event.get("name"),
            "runtime_external_id": external_id,
            "correlation": correlation,
            "raw_name": kernel,
            "kernel_key": _kernel_key(kernel),
            "marker": marker,
            "kernel_trace_shape": kernel_trace_shape,
            "device_event": {
                "name": device.get("name"),
                "ts": device.get("ts"),
                "dur": device.get("dur"),
            } if device else None,
        })
    entries.sort(key=lambda item: (
        item["marker"]["phase"], item["marker"]["layer_id"],
        item["runtime_event_index"]))
    return markers, entries


def _target_bucket(target):
    bucket = target.get("selected_bucket") or {}
    if not bucket:
        return None
    return (
        _phase(bucket.get("phase", target.get("phase"))),
        int(target.get("representative_layer_id", -1)),
        _integer(bucket.get("batch_size")),
        _integer(bucket.get("input_tokens")),
    )


def _clean_sequence_audit(plan, clean_table_path, active_phases):
    """Verify captured target positions against the authoritative clean table."""
    if not clean_table_path:
        return None, set()
    with open(clean_table_path) as fh:
        clean_doc = json.load(fh)
    clean_tables = {
        (_phase(table.get("phase")), table.get("pattern_id")): table
        for table in clean_doc.get("tables", [])}
    grouped = {}
    for target in plan.get("capture_targets", []):
        phase = _phase(target.get("phase"))
        if phase not in active_phases:
            continue
        key = (phase, target.get("pattern_id"))
        grouped.setdefault(key, []).append(target)
    groups = []
    failed = set()
    for key, targets in sorted(grouped.items()):
        table = clean_tables.get(key)
        clean_by_pos = {
            int(row.get("pos", -1)): row
            for row in (table or {}).get("rows", [])}
        mismatches = []
        for target in sorted(targets, key=lambda item: item.get("pos", -1)):
            pos = int(target.get("pos", -1))
            clean_row = clean_by_pos.get(pos)
            if clean_row is None:
                mismatches.append({"pos": pos, "reason": "missing_clean_row"})
                continue
            if _kernel_key(target.get("raw_name")) != _kernel_key(
                    clean_row.get("raw_name")):
                mismatches.append({
                    "pos": pos,
                    "reason": "kernel_identity_mismatch",
                    "capture_plan_kernel": target.get("raw_name"),
                    "clean_table_kernel": clean_row.get("raw_name"),
                })
        status = "pass" if table is not None and not mismatches else "fail"
        if status == "fail":
            failed.add(key)
        groups.append({
            "phase": key[0],
            "pattern_id": key[1],
            "status": status,
            "capture_target_count": len(targets),
            "clean_row_count": len(clean_by_pos),
            "capture_representative_layers": sorted({
                int(target.get("representative_layer_id", -1))
                for target in targets}),
            "clean_representative_layer": (
                table.get("representative_layer_id") if table else None),
            "matched_position_count": len(targets) - len(mismatches),
            "mismatches": mismatches,
        })
    return {
        "status": "fail" if failed else "pass",
        "clean_table": os.path.abspath(clean_table_path),
        "active_phases": sorted(active_phases),
        "groups": groups,
    }, failed


def _marker_matches_bucket(marker, bucket):
    phase, layer_id, batch_size, input_tokens = bucket
    if marker["phase"] != phase or marker["layer_id"] != layer_id:
        return False
    if batch_size >= 0 and marker["batch_size"] != batch_size:
        return False
    # Clean decode steps use input_tokens=0, while runtime markers describe
    # the one-token-per-sequence tensor and therefore report toks=bs.
    if phase != "decode" and input_tokens >= 0:
        return marker["input_tokens"] == input_tokens
    return True


def _layer_scope_markers(events):
    """Return explicit full-layer invocation ranges from the capture trace."""
    scopes = []
    for index, event in enumerate(events):
        name = str(event.get("name", ""))
        if (event.get("cat") != "user_annotation"
                or not name.startswith(LAYER_SCOPE_PREFIX)
                or event.get("ts") is None
                or event.get("dur") is None):
            continue
        fields = _marker_fields(name)
        scopes.append({
            "index": index,
            "name": name,
            "phase": _phase(fields.get("phase")),
            "layer_id": _integer(fields.get("layer")),
            "batch_size": _integer(fields.get("bs")),
            "input_tokens": _integer(fields.get("toks")),
            "op_path": fields.get("path"),
            "pid": event.get("pid"),
            "tid": event.get("tid"),
            "ts": float(event["ts"]),
            "end": float(event["ts"]) + float(event["dur"]),
        })
    return scopes


def _module_path(op_path):
    return str(op_path or "").split("::launcher:", 1)[0]


def _common_module_path(markers):
    paths = [
        _module_path(marker.get("op_path")).split(".")
        for marker in markers if marker.get("op_path")]
    if not paths:
        return ""
    common = []
    for components in zip(*paths):
        if len(set(components)) != 1:
            break
        common.append(components[0])
    return ".".join(common)


def _markers_in_scope(markers, scope):
    selected = []
    for marker in markers:
        if (marker["pid"] != scope["pid"]
                or marker["tid"] != scope["tid"]):
            continue
        contained = (
            scope["ts"] <= marker["ts"]
            and marker["end"] <= scope["end"])
        # The module hook may enter immediately before the dedicated layer
        # scope and exit immediately after it.  Keep that one outer module
        # marker as part of the same invocation too.
        wraps_scope = (
            _module_path(marker.get("op_path")) == scope.get("op_path")
            and marker["ts"] <= scope["ts"]
            and scope["end"] <= marker["end"])
        if contained or wraps_scope:
            selected.append(marker)
    return selected


def _first_forward_marker_ids(markers, bucket, layer_scopes=None):
    """Select one complete layer invocation for a clean-trace bucket.

    The capture emits an explicit ``GEAK_LAYER_SCOPE`` range around each main
    decoder-layer invocation.  That range, rather than repeated inner wrapper
    names, is the authoritative boundary: one wrapper may legitimately execute
    multiple times inside a layer (for example pre-attention and pre-MLP norm).

    Older/synthetic traces may not contain layer scopes.  For those, use the
    earliest outer module marker that spans the common module path.  This is a
    conservative compatibility fallback; it never treats an arbitrary repeated
    child path as proof that a new forward started.
    """
    effective_bucket = bucket
    candidates = sorted(
        (marker for marker in markers
         if _marker_matches_bucket(marker, bucket)),
        key=lambda marker: marker["index"])
    if not candidates:
        phase, layer_id, batch_size, input_tokens = bucket
        compatible = [
            marker for marker in markers
            if marker["phase"] == phase and marker["layer_id"] == layer_id]
        if compatible:
            target_tokens = (
                input_tokens if input_tokens > 0 else batch_size)
            available_buckets = {
                (marker["batch_size"], marker["input_tokens"])
                for marker in compatible}
            selected_bucket = min(
                available_buckets,
                key=lambda value: (
                    abs(value[1] - target_tokens),
                    abs(value[0] - batch_size),
                    value))
            effective_bucket = (
                phase, layer_id, selected_bucket[0], selected_bucket[1])
            candidates = sorted(
                (marker for marker in compatible
                 if (marker["batch_size"], marker["input_tokens"])
                 == selected_bucket),
                key=lambda marker: marker["index"])
    if not candidates:
        return set()

    scopes = sorted(
        (scope for scope in (layer_scopes or [])
         if _marker_matches_bucket(scope, effective_bucket)),
        key=lambda scope: scope["index"])
    selected = []
    for scope in scopes:
        selected = _markers_in_scope(candidates, scope)
        if selected:
            break

    if not selected:
        common_path = _common_module_path(candidates)
        envelope = next((
            marker for marker in candidates
            if marker.get("op_path") == common_path), None)
        if envelope is not None:
            selected = _markers_in_scope(candidates, {
                "pid": envelope["pid"],
                "tid": envelope["tid"],
                "ts": envelope["ts"],
                "end": envelope["end"],
                "op_path": common_path,
            })
        else:
            # No structural invocation envelope exists.  Fail narrowly by
            # selecting only the first observed marker; selecting every replay
            # would make repeated kernels look uniquely attributable.
            selected = candidates[:1]
    return {
        marker["op_instance_id"] for marker in selected
        if marker.get("op_instance_id")
    }


def _apply_mapping(
        target, candidate, capture_trace_path, rule, marker_launch_count):
    marker = candidate["marker"]
    kernel_trace_shape = candidate.get("kernel_trace_shape")
    target["candidate_op_path"] = marker["op_path"]
    target["candidate_op_instance_id"] = marker["op_instance_id"]
    target["candidate_wrapper"] = marker["op_path"]
    target["candidate_terminal_launcher"] = (
        kernel_trace_shape.get("op_name")
        if kernel_trace_shape else candidate["runtime_name"])
    targeted_launcher = "::launcher:" in str(marker.get("op_path") or "")
    target["mapping_cardinality"] = (
        "1:1" if kernel_trace_shape
        or (targeted_launcher and marker_launch_count == 1) else "1:N")
    target["source_mapping_status"] = (
        "runtime_kernel_trace_shape"
        if kernel_trace_shape else "runtime_marker_contained")
    if kernel_trace_shape:
        kernel_trace_shape = dict(kernel_trace_shape)
        bucket = _target_bucket(target)
        kernel_trace_shape["bucket_match"] = (
            "exact" if bucket and _marker_matches_bucket(marker, bucket)
            else "compatible")
        kernel_trace_shape["marker_context"] = {
            "phase": marker["phase"],
            "layer_id": marker["layer_id"],
            "batch_size": marker["batch_size"],
            "input_tokens": marker["input_tokens"],
        }
        target["kernel_trace_shape"] = kernel_trace_shape
    target["runtime_marker_mapping_status"] = "matched"
    target.pop("runtime_marker_candidate_count", None)
    target["runtime_marker_evidence"] = {
        "capture_trace": os.path.abspath(capture_trace_path),
        "marker_name": marker["name"],
        "marker_external_id": marker["external_id"],
        "runtime_event_index": candidate["runtime_event_index"],
        "runtime_name": candidate["runtime_name"],
        "runtime_external_id": candidate.get("runtime_external_id"),
        "runtime_correlation": candidate["correlation"],
        "captured_kernel": candidate["raw_name"],
        "targeted_launcher_probe": targeted_launcher,
        "kernel_trace_shape_exact": bool(kernel_trace_shape),
        "marker_launch_count": marker_launch_count,
        "capture_device_event": candidate["device_event"],
        "rule": rule,
    }


def _shape_log_records(path, targeted_only=False):
    records = {}
    if not path or not os.path.exists(path):
        return records
    with open(path) as fh:
        for line in fh:
            record = json.loads(line)
            if (targeted_only
                    and record.get("op_type")
                    != "targeted_python_launcher"):
                continue
            key = (
                _phase(record.get("phase")),
                int(record.get("layer_id", -1)))
            records.setdefault(key, []).append(record)
    return records


def _shape_log_first_forward(path, targeted_only=False):
    selected = {}
    seen_paths = {}
    for key, records in _shape_log_records(path, targeted_only).items():
        for record in records:
            op_path = record.get("op_path")
            if op_path in seen_paths.setdefault(key, set()):
                continue
            seen_paths[key].add(op_path)
            selected.setdefault(key, []).append(record)
    return selected


def _apply_source_callable_mapping(
        targets, shape_log_path, callable_kernel_map):
    records_by_key = _shape_log_first_forward(
        shape_log_path, targeted_only=True)
    matched = 0
    for spec in callable_kernel_map or []:
        pattern = re.compile(spec["kernel_pattern"])
        launcher = spec["target"]
        scope = spec.get("scope", "wrapper")
        cardinality = "1:1" if scope == "kernel" else "1:N"
        for (phase, layer_id), records in records_by_key.items():
            matching_targets = [
                target for target in targets
                if target.get("runtime_marker_mapping_status") != "matched"
                and _phase(target.get("phase")) == phase
                and int(target.get("representative_layer_id", -1)) == layer_id
                and pattern.search(str(target.get("raw_name") or ""))]
            matching_records = [
                record for record in records
                if str(record.get("op_path") or "").endswith(
                    "::launcher:" + launcher)]
            if not matching_targets or not matching_records:
                continue
            if cardinality == "1:1":
                pairs = zip(matching_targets, matching_records)
            elif len(matching_records) == 1:
                pairs = (
                    (target, matching_records[0])
                    for target in matching_targets)
            else:
                pairs = zip(matching_targets, matching_records)
            for target, record in pairs:
                target["candidate_op_path"] = record["op_path"]
                target["candidate_op_instance_id"] = record[
                    "op_instance_id"]
                target["candidate_wrapper"] = record["op_path"]
                target["candidate_terminal_launcher"] = launcher
                target["mapping_cardinality"] = cardinality
                target["source_mapping_status"] = (
                    "source_targeted_launcher_probe")
                target["runtime_marker_mapping_status"] = "matched"
                target.pop("runtime_marker_candidate_count", None)
                target["source_callable_evidence"] = {
                    "shape_log": os.path.abspath(shape_log_path),
                    "launcher": launcher,
                    "kernel_pattern": spec["kernel_pattern"],
                    "source": spec.get("source"),
                    "scope": scope,
                    "rule": (
                        "kernel-to-launcher association verified from source; "
                        "shape captured by monkeypatched launcher"),
                }
                matched += 1
    return matched


def _apply_source_wrapper_mapping(
        targets, shape_log_path, source_wrapper_map):
    records_by_key = _shape_log_first_forward(shape_log_path)
    matched = 0
    for spec in source_wrapper_map or []:
        kernel_pattern = re.compile(
            spec.get("kernel_pattern", ".*"))
        for (phase, layer_id), records in records_by_key.items():
            expected_path = spec["op_path"].format(layer=layer_id)
            record = next((
                item for item in records
                if item.get("op_path") == expected_path), None)
            if record is None:
                continue
            for target in targets:
                pos = int(target.get("pos", -1))
                if target.get("runtime_marker_mapping_status") == "matched":
                    continue
                if spec.get("phase") not in (
                        None, _phase(target.get("phase"))):
                    continue
                if _phase(target.get("phase")) != phase:
                    continue
                if int(target.get("representative_layer_id", -1)) != layer_id:
                    continue
                if spec.get("pattern_id") not in (
                        None, target.get("pattern_id")):
                    continue
                if not (
                        int(spec.get("pos_start", pos)) <= pos
                        <= int(spec.get("pos_end", pos))):
                    continue
                if not kernel_pattern.search(
                        str(target.get("raw_name") or "")):
                    continue
                target["candidate_op_path"] = record["op_path"]
                target["candidate_op_instance_id"] = record[
                    "op_instance_id"]
                target["candidate_wrapper"] = record["op_path"]
                target["candidate_terminal_launcher"] = (
                    record.get("op_type"))
                target["mapping_cardinality"] = "1:N"
                target["source_mapping_status"] = (
                    "source_verified_wrapper_probe")
                target["runtime_marker_mapping_status"] = "matched"
                target.pop("runtime_marker_candidate_count", None)
                target["source_wrapper_evidence"] = {
                    "shape_log": os.path.abspath(shape_log_path),
                    "wrapper": record["op_path"],
                    "source": spec.get("source"),
                    "rule": (
                        "kernel assigned to an enclosing wrapper verified "
                        "from the model/runtime source call path"),
                }
                matched += 1
    return matched


def _layer_fallback_eligible(target):
    kernel = str(target.get("short_name") or target.get("raw_name") or "")
    return (
        kernel not in ("kentry", "Memcpy")
        and "__amd_rocclr_fillBufferAligned" not in kernel)


def _shape_mapping_exclusion_reason(target):
    """Classify clean rows that do not own a meaningful tensor schema."""
    if str(target.get("stage") or "").lower() == "communication":
        return "cross_device_communication"
    if str(target.get("event_type") or "").lower() in (
            "gpu_memcpy", "gpu_memset"):
        return "runtime_copy_or_memset"
    kernel = str(target.get("short_name") or target.get("raw_name") or "")
    if "__amd_rocclr_fillBufferAligned" in kernel:
        return "runtime_buffer_fill"
    return None


def _annotate_decode_semantic_regions(targets):
    """Partition MLA Decode rows using anchors in the clean graph trace.

    The result is region evidence, never Kernel-exact ownership.  It prevents
    a whole-layer fallback from conflating the pre-attention K-absorb BMM with
    the post-attention V-absorb BMM.
    """
    grouped = {}
    for target in targets:
        if _phase(target.get("phase")) != "decode":
            continue
        key = (target.get("pattern_id"),
               int(target.get("representative_layer_id", -1)))
        grouped.setdefault(key, []).append(target)
    for (_, layer_id), rows in grouped.items():
        rows.sort(key=lambda item: int(item.get("pos", -1)))
        attn = [i for i, row in enumerate(rows)
                if str(row.get("stage") or "").lower() == "attn"]
        if not attn:
            continue
        core = attn[0]
        reduce_idx = next((
            i for i in attn[1:]
            if "reduce" in str(rows[i].get("raw_name") or "").lower()), core)
        v_gemm = next((
            i for i in range(reduce_idx + 1, len(rows))
            if str(rows[i].get("stage") or "").lower() == "gemm"), None)
        next_collective = next((
            i for i in range((v_gemm + 1) if v_gemm is not None else core + 1,
                             len(rows))
            if str(rows[i].get("stage") or "").lower() == "communication"),
            len(rows))
        for i, row in enumerate(rows):
            raw = str(row.get("raw_name") or "").lower()
            region = None
            if i < core and "qk" in raw and "norm" in raw:
                region = "qk_norm"
            elif i < core and "batched_gemm" in raw:
                region = "k_absorb"
            elif i < core and (row.get("stage") == "kv_cache"
                               or "rope" in raw or "catarray" in raw):
                region = "rope_kv"
            elif core <= i <= reduce_idx:
                region = "mla_core"
            elif v_gemm is not None and reduce_idx < i <= v_gemm:
                region = "v_absorb"
            elif v_gemm is not None and v_gemm < i < next_collective:
                region = "o_proj"
            if not region:
                continue
            row["semantic_region"] = region
            row["semantic_region_path"] = (
                "model.layers.%d.self_attn.%s" % (layer_id, region))
            row["semantic_region_evidence"] = {
                "level": "P",
                "source": "clean_graph_anchor_partition",
                "rule": "ordered Decode anchors in the CUDA-graph trace",
            }


def _apply_vabsorb_bmm_probe(targets, shape_log_path):
    """Attach a unique graph-construction torch.bmm probe to V-absorb."""
    if not shape_log_path or not os.path.exists(shape_log_path):
        return 0
    records = []
    with open(shape_log_path) as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if (record.get("op_type") == "targeted_python_launcher"
                    and str(record.get("op_path") or "").endswith(
                        "::launcher:torch:bmm")):
                records.append(record)
    matched = 0
    for target in targets:
        if target.get("runtime_marker_mapping_status") == "matched":
            continue
        if (target.get("semantic_region") != "v_absorb"
                or str(target.get("stage") or "").lower() != "gemm"):
            continue
        candidates = [record for record in records
                      if _phase(record.get("phase")) == "decode"
                      and int(record.get("layer_id", -1)) == int(
                          target.get("representative_layer_id", -1))]
        if len(candidates) != 1:
            continue
        record = candidates[0]
        target["candidate_op_path"] = target["semantic_region_path"]
        target["candidate_op_instance_id"] = record["op_instance_id"]
        target["candidate_wrapper"] = record["op_path"]
        target["candidate_terminal_launcher"] = "torch:bmm"
        target["mapping_cardinality"] = "1:1"
        target["source_mapping_status"] = "vabsorb_targeted_bmm_probe"
        target["runtime_marker_mapping_status"] = "matched"
        target.pop("runtime_marker_candidate_count", None)
        target["targeted_region_evidence"] = {
            "shape_log": os.path.abspath(shape_log_path),
            "semantic_region": "v_absorb",
            "launcher": "torch:bmm",
            "rule": (
                "unique graph-construction bmm inside the anchored "
                "V-absorb region"),
        }
        matched += 1
    return matched


def _semantic_wrapper_candidates(records, stage, layer_id):
    rules = {
        "norm": r"norm",
        "rope": r"rope|rotary",
        "attn": r"attention|attn|mha|mla|radix",
        "linear_attn": (
            r"linear.?attn|linear.?attention|gated.?delta|"
            r"conv1d|recurrent|gdn"),
        "gemm": r"linear|gemm|matmul|projection|proj",
        "topk": r"top.?k|router|gate",
        "moe": r"moe|expert|mlp",
        "activation": r"activation|silu|gelu|swiglu",
        "quant": r"quant|fp8|scale",
        "communication": r"all.?reduce|communicat|collective",
        "kv_cache": r"cache",
    }
    pattern = rules.get(str(stage or "").lower())
    if not pattern:
        return []
    layer_pattern = re.compile(
        r"(?:^|\.)layers\.%d$" % layer_id)
    candidates = []
    for record in records:
        path = str(record.get("op_path") or "")
        if layer_pattern.search(path):
            continue
        path_leaf = path.rsplit(".", 1)[-1]
        text = " ".join(str(record.get(key) or "") for key in (
            "op_name", "op_type")) + " " + path_leaf
        if re.search(pattern, text, flags=re.I):
            candidates.append(record)
    primary_rules = {
        "rope": r"rope|rotary",
        "attn": r"attention|attn|mha|mla|radix",
        "linear_attn": (
            r"linear.?attn|linear.?attention|gated.?delta|recurrent|gdn"),
        "topk": r"top.?k",
        "kv_cache": r"cache",
    }
    primary = primary_rules.get(str(stage or "").lower())
    if primary:
        typed = [
            record for record in candidates
            if re.search(
                primary, str(record.get("op_type") or ""), flags=re.I)]
        if typed:
            candidates = typed
    if str(stage or "").lower() in ("attn", "linear_attn"):
        by_depth = sorted(
            candidates,
            key=lambda item: len(str(item.get("op_path") or "").split(".")))
        if by_depth:
            deepest = str(by_depth[-1].get("op_path") or "")
            if all(
                    deepest.startswith(
                        str(item.get("op_path") or "") + ".")
                    for item in by_depth[:-1]):
                return [by_depth[-1]]
    return candidates


def _apply_shape_log_semantic_wrapper_mapping(
        targets, shape_log_path, eligible_bucket_keys):
    """Use a unique executed semantic wrapper without positional guessing."""
    # Keep duplicate invocations visible here.  A path may execute twice in one
    # layer (for example the same norm launcher before attention and before the
    # MLP); collapsing those records would turn an ambiguous fallback into a
    # falsely unique one.
    records_by_key = _shape_log_records(shape_log_path)
    matched = 0
    for phase, layer_id in eligible_bucket_keys:
        records = records_by_key.get((phase, layer_id), [])
        for target in targets:
            if target.get("runtime_marker_mapping_status") == "matched":
                continue
            if not _layer_fallback_eligible(target):
                continue
            if _shape_mapping_exclusion_reason(target) is not None:
                continue
            if _phase(target.get("phase")) != phase:
                continue
            if int(target.get("representative_layer_id", -1)) != layer_id:
                continue
            candidates = _semantic_wrapper_candidates(
                records, target.get("stage"), layer_id)
            if len(candidates) != 1:
                continue
            record = candidates[0]
            target["candidate_op_path"] = record["op_path"]
            target["candidate_op_instance_id"] = record["op_instance_id"]
            target["candidate_wrapper"] = record["op_path"]
            target["candidate_terminal_launcher"] = record.get("op_type")
            target["mapping_cardinality"] = "1:N"
            target["source_mapping_status"] = (
                "runtime_shape_log_unique_semantic_wrapper")
            target["runtime_marker_mapping_status"] = "matched"
            target.pop("runtime_marker_candidate_count", None)
            target["shape_log_layer_evidence"] = {
                "shape_log": os.path.abspath(shape_log_path),
                "wrapper": record["op_path"],
                "scope": "phase_layer_semantic_wrapper",
                "stage": target.get("stage"),
                "candidate_count": 1,
                "rule": (
                    "same phase/layer replay exposed exactly one executed "
                    "wrapper matching the kernel semantic stage; no launch "
                    "order or kernel-level containment claimed"),
            }
            matched += 1
    return matched


def _apply_shape_log_layer_fallback(
        targets, shape_log_path, missing_bucket_keys):
    """Attribute an unresolved row only to its proven phase/layer wrapper.

    Decode execution can run on a thread where record_function ranges are not
    exported by torch.profiler.  The clean trace still proves the row's
    phase/layer, and the same replay's shape log proves that layer wrapper's
    tensor schema.  This is deliberately P(wrapper), never P(kernel).
    """
    records_by_key = _shape_log_first_forward(shape_log_path)
    matched = 0
    covered_keys = set()
    for phase, layer_id in missing_bucket_keys:
        records = records_by_key.get((phase, layer_id), [])
        layer_pattern = re.compile(
            r"(?:^|\.)layers\.%d$" % layer_id)
        record = next((
            item for item in records
            if layer_pattern.search(str(item.get("op_path") or ""))), None)
        if record is None:
            continue
        covered_keys.add((phase, layer_id))
        for target in targets:
            if target.get("runtime_marker_mapping_status") == "matched":
                continue
            if not _layer_fallback_eligible(target):
                continue
            if _phase(target.get("phase")) != phase:
                continue
            if int(target.get("representative_layer_id", -1)) != layer_id:
                continue
            region_path = target.get("semantic_region_path")
            target["candidate_op_path"] = region_path or record["op_path"]
            target["candidate_op_instance_id"] = record["op_instance_id"]
            target["candidate_wrapper"] = record["op_path"]
            target["candidate_terminal_launcher"] = record.get("op_type")
            target["mapping_cardinality"] = "1:N"
            target["source_mapping_status"] = (
                "runtime_shape_log_semantic_region"
                if region_path else "runtime_shape_log_layer_wrapper")
            target["runtime_marker_mapping_status"] = "matched"
            target.pop("runtime_marker_candidate_count", None)
            target["shape_log_layer_evidence"] = {
                "shape_log": os.path.abspath(shape_log_path),
                "wrapper": record["op_path"],
                "scope": (
                    "phase_layer_semantic_region"
                    if region_path else "phase_layer_wrapper"),
                "semantic_region": target.get("semantic_region"),
                "rule": (
                    "clean trace assigns kernel to phase/layer; same replay "
                    "captures that layer wrapper shape; no kernel-level "
                    "profiler containment claimed"),
            }
            matched += 1
    return matched, covered_keys


def _apply_shape_log_region_fallback(targets, shape_log_path):
    """Bind anchored Decode regions to the layer probe without claiming shape.

    This runs even when some profiler markers exist for the bucket: partial
    marker export is common on Decode worker threads.  Only targets already
    assigned a clean-graph semantic region are eligible.
    """
    records_by_key = _shape_log_first_forward(shape_log_path)
    matched = 0
    for (phase, layer_id), records in records_by_key.items():
        if phase != "decode":
            continue
        layer_pattern = re.compile(r"(?:^|\.)layers\.%d$" % layer_id)
        record = next((item for item in records
                       if layer_pattern.search(
                           str(item.get("op_path") or ""))), None)
        if record is None:
            continue
        for target in targets:
            if target.get("runtime_marker_mapping_status") == "matched":
                continue
            if not target.get("semantic_region_path"):
                continue
            if (_phase(target.get("phase")) != phase
                    or int(target.get("representative_layer_id", -1))
                    != layer_id):
                continue
            target["candidate_op_path"] = target["semantic_region_path"]
            target["candidate_op_instance_id"] = record["op_instance_id"]
            target["candidate_wrapper"] = record["op_path"]
            target["candidate_terminal_launcher"] = record.get("op_type")
            target["mapping_cardinality"] = "1:N"
            target["source_mapping_status"] = (
                "runtime_shape_log_semantic_region")
            target["runtime_marker_mapping_status"] = "matched"
            target.pop("runtime_marker_candidate_count", None)
            target["shape_log_layer_evidence"] = {
                "shape_log": os.path.abspath(shape_log_path),
                "wrapper": record["op_path"],
                "scope": "phase_layer_semantic_region",
                "semantic_region": target.get("semantic_region"),
                "rule": (
                    "clean graph anchor partition identifies the narrow "
                    "region; graph-construction layer wrapper supplies "
                    "context only"),
            }
            matched += 1
    return matched


def map_plan(
        plan_path, capture_trace_path, out_path, shape_log_path="",
        callable_kernel_map=None, source_wrapper_map=None,
        clean_table_path="", required_phases=None,
        operator_schema_manifest_path=""):
    with open(plan_path) as fh:
        plan = json.load(fh)
    _annotate_decode_semantic_regions(plan.get("capture_targets", []))
    events = _load(capture_trace_path)
    schema_manifest = _load_schema_manifest(operator_schema_manifest_path)
    markers, entries = _runtime_entries(events, schema_manifest)
    layer_scopes = _layer_scope_markers(events)
    marker_launch_counts = collections.Counter(
        entry["marker"]["op_instance_id"] for entry in entries)

    grouped_targets = {}
    for target in plan.get("capture_targets", []):
        key = (
            _phase(target.get("phase")),
            int(target.get("representative_layer_id", -1)),
            _kernel_key(target.get("raw_name")),
            _target_bucket(target))
        grouped_targets.setdefault(key, []).append(target)

    bucket_marker_ids = {}
    for key in grouped_targets:
        bucket = key[3]
        if bucket is not None and bucket not in bucket_marker_ids:
            bucket_marker_ids[bucket] = _first_forward_marker_ids(
                markers, bucket, layer_scopes)
    required_phases = {
        _phase(value) for value in (required_phases or [])}
    active_phases = {
        bucket[0] for bucket, marker_ids in bucket_marker_ids.items()
        if marker_ids}
    audited_phases = (
        active_phases & required_phases if required_phases else active_phases)
    clean_sequence_audit, failed_sequence_groups = _clean_sequence_audit(
        plan, clean_table_path, audited_phases)

    matched = 0
    ambiguous = 0
    unmatched = 0
    ambiguous_candidates = {}
    for key, targets in grouped_targets.items():
        targets.sort(key=lambda item: item.get("pos", -1))
        phase, layer_id, kernel_key, bucket = key
        if any(
                (phase, target.get("pattern_id")) in failed_sequence_groups
                for target in targets):
            for target in targets:
                target["runtime_marker_mapping_status"] = (
                    "clean_sequence_mismatch")
            continue
        marker_ids = bucket_marker_ids.get(bucket) if bucket else None
        candidates = [
            entry for entry in entries
            if entry["marker"]["phase"] == phase
            and entry["marker"]["layer_id"] == layer_id
            and entry["kernel_key"] == kernel_key
            and (marker_ids is None
                 or entry["marker"]["op_instance_id"] in marker_ids)
        ]
        if len(candidates) != len(targets):
            for target in targets:
                target["runtime_marker_mapping_status"] = (
                    "ambiguous_count" if candidates else "not_found")
                target["runtime_marker_candidate_count"] = len(candidates)
                if candidates:
                    ambiguous_candidates[id(target)] = candidates
            if candidates:
                ambiguous += len(targets)
            else:
                unmatched += len(targets)
            continue
        candidates.sort(key=lambda item: item["runtime_event_index"])
        for target, candidate in zip(targets, candidates):
            _apply_mapping(
                target, candidate, capture_trace_path,
                (
                    "same phase/layer/normalized kernel identity with "
                    "unique group cardinality; runtime launch timestamp "
                    "contained by deepest GEAK wrapper marker"),
                marker_launch_counts[candidate["marker"]["op_instance_id"]])
            matched += 1

    # A repeated generic kernel name may occur under two wrappers in the same
    # layer.  Once unique-name rows are mapped, surrounding launch positions
    # provide a strict interval that can disambiguate it without guessing.
    ordered_groups = {}
    for target in plan.get("capture_targets", []):
        key = (
            _phase(target.get("phase")),
            int(target.get("representative_layer_id", -1)),
            _target_bucket(target))
        ordered_groups.setdefault(key, []).append(target)
    for targets in ordered_groups.values():
        targets.sort(key=lambda item: item.get("pos", -1))
        for index, target in enumerate(targets):
            candidates = ambiguous_candidates.get(id(target))
            if not candidates:
                continue
            previous = next((
                item for item in reversed(targets[:index])
                if item.get("runtime_marker_mapping_status") == "matched"),
                None)
            following = next((
                item for item in targets[index + 1:]
                if item.get("runtime_marker_mapping_status") == "matched"),
                None)
            lower = (
                previous["runtime_marker_evidence"]["runtime_event_index"]
                if previous else float("-inf"))
            upper = (
                following["runtime_marker_evidence"]["runtime_event_index"]
                if following else float("inf"))
            inside = [
                candidate for candidate in candidates
                if lower < candidate["runtime_event_index"] < upper]
            if len(inside) != 1:
                continue
            _apply_mapping(
                target, inside[0], capture_trace_path,
                (
                    "same phase/layer/normalized kernel identity; unique "
                    "candidate between already-mapped neighboring clean "
                    "kernel positions inside one graph-construction forward"),
                marker_launch_counts[
                    inside[0]["marker"]["op_instance_id"]])
            matched += 1
            ambiguous -= 1

    source_callable_matched = _apply_source_callable_mapping(
        plan.get("capture_targets", []), shape_log_path,
        callable_kernel_map)
    vabsorb_probe_matched = _apply_vabsorb_bmm_probe(
        plan.get("capture_targets", []), shape_log_path)
    source_wrapper_matched = _apply_source_wrapper_mapping(
        plan.get("capture_targets", []), shape_log_path,
        source_wrapper_map)
    missing_bucket_keys = {
        (bucket[0], bucket[1])
        for bucket, marker_ids in bucket_marker_ids.items()
        if not marker_ids
    }
    semantic_fallback_bucket_keys = {
        (bucket[0], bucket[1])
        for bucket in bucket_marker_ids
    }
    semantic_wrapper_matched = _apply_shape_log_semantic_wrapper_mapping(
        plan.get("capture_targets", []), shape_log_path,
        semantic_fallback_bucket_keys)
    region_fallback_matched = _apply_shape_log_region_fallback(
        plan.get("capture_targets", []), shape_log_path)
    layer_fallback_matched, layer_fallback_keys = (
        _apply_shape_log_layer_fallback(
            plan.get("capture_targets", []), shape_log_path,
            missing_bucket_keys))
    for target in plan.get("capture_targets", []):
        sequence_key = (
            _phase(target.get("phase")), target.get("pattern_id"))
        if sequence_key not in failed_sequence_groups:
            continue
        for key in (
                "candidate_op_path", "candidate_op_instance_id",
                "candidate_wrapper", "candidate_terminal_launcher",
                "mapping_cardinality", "source_mapping_status",
                "runtime_marker_evidence", "source_callable_evidence",
                "source_wrapper_evidence", "shape_log_layer_evidence",
                "kernel_trace_shape"):
            target.pop(key, None)
        target["runtime_marker_mapping_status"] = "clean_sequence_mismatch"
    statuses = [
        target.get("runtime_marker_mapping_status")
        for target in plan.get("capture_targets", [])]
    kernel_trace_shape_matched = sum(
        bool(target.get("kernel_trace_shape"))
        for target in plan.get("capture_targets", []))
    operator_schema_statuses = collections.Counter(
        (target.get("kernel_trace_shape") or {}).get(
            "operator_schema_resolution", {}).get("status", "unavailable")
        for target in plan.get("capture_targets", [])
        if target.get("kernel_trace_shape"))
    matched = statuses.count("matched")
    ambiguous = statuses.count("ambiguous_count")
    unmatched = statuses.count("not_found")
    sequence_rejected = statuses.count("clean_sequence_mismatch")
    eligibility = collections.Counter()
    eligible_targets = []
    for target in plan.get("capture_targets", []):
        reason = _shape_mapping_exclusion_reason(target)
        if reason:
            eligibility[reason] += 1
        else:
            eligible_targets.append(target)
    source_only_covered_buckets = set()
    for bucket, marker_ids in bucket_marker_ids.items():
        if marker_ids:
            continue
        bucket_targets = [
            target for target in plan.get("capture_targets", [])
            if _target_bucket(target) == bucket
            and _shape_mapping_exclusion_reason(target) is None]
        if (bucket_targets and all(
                target.get("runtime_marker_mapping_status") == "matched"
                and (target.get("source_callable_evidence")
                     or target.get("source_wrapper_evidence"))
                for target in bucket_targets)):
            # A targeted retry may intentionally expose only a monkeypatched
            # source callable in the Shape log.  In that case a missing
            # profiler range is not missing evidence: every shape-bearing row
            # in this bucket already has a reviewed source path and an
            # observed invocation.  This exception is deliberately narrower
            # than the normal layer fallback and cannot assign ownership from
            # a Kernel name alone.
            source_only_covered_buckets.add(bucket)
    missing_marker_buckets = [
        "|".join(str(value) for value in bucket)
        for bucket, marker_ids in bucket_marker_ids.items()
        if (not marker_ids
            and (not required_phases or bucket[0] in required_phases)
            and (bucket[0], bucket[1]) not in layer_fallback_keys
            and bucket not in source_only_covered_buckets)
    ]
    eligible_matched = sum(
        target.get("runtime_marker_mapping_status") == "matched"
        for target in eligible_targets)
    mapping_by_phase = {}
    for phase in sorted({
            _phase(target.get("phase"))
            for target in plan.get("capture_targets", [])}):
        phase_targets = [
            target for target in plan.get("capture_targets", [])
            if _phase(target.get("phase")) == phase]
        phase_eligible = [
            target for target in phase_targets
            if _shape_mapping_exclusion_reason(target) is None]
        phase_matched = sum(
            target.get("runtime_marker_mapping_status") == "matched"
            for target in phase_eligible)
        mapping_by_phase[phase] = {
            "target_count": len(phase_targets),
            "shape_eligible_target_count": len(phase_eligible),
            "shape_eligible_matched_target_count": phase_matched,
            "shape_eligible_match_fraction": round(
                phase_matched / len(phase_eligible), 4)
                if phase_eligible else 1.0,
        }
    plan["runtime_marker_mapping"] = {
        "schema_version": 1,
        "capture_trace": os.path.abspath(capture_trace_path),
        "marker_count": len(markers),
        "contained_runtime_kernel_count": len(entries),
        "matched_target_count": matched,
        "kernel_trace_shape_matched_target_count": (
            kernel_trace_shape_matched),
        "operator_schema_manifest": (
            os.path.abspath(operator_schema_manifest_path)
            if operator_schema_manifest_path else None),
        "operator_schema_resolution_counts": dict(sorted(
            operator_schema_statuses.items())),
        "source_callable_matched_target_count": source_callable_matched,
        "vabsorb_probe_matched_target_count": vabsorb_probe_matched,
        "shape_log_region_fallback_matched_target_count": (
            region_fallback_matched),
        "source_wrapper_matched_target_count": source_wrapper_matched,
        "shape_log_semantic_wrapper_matched_target_count": (
            semantic_wrapper_matched),
        "shape_log_layer_fallback_matched_target_count": (
            layer_fallback_matched),
        "ambiguous_target_count": ambiguous,
        "unmatched_target_count": unmatched,
        "sequence_rejected_target_count": sequence_rejected,
        "shape_eligible_target_count": len(eligible_targets),
        "shape_eligible_matched_target_count": eligible_matched,
        "shape_eligible_match_fraction": round(
            eligible_matched / len(eligible_targets), 4)
            if eligible_targets else 1.0,
        "shape_ineligible_target_count": sum(eligibility.values()),
        "shape_ineligible_reasons": dict(sorted(eligibility.items())),
        "shape_mapping_by_phase": mapping_by_phase,
        "clean_table_sequence_audit": clean_sequence_audit,
        "required_phases": sorted(required_phases),
        "phase_coverage_complete": not missing_marker_buckets,
        "missing_marker_buckets": missing_marker_buckets,
        "source_only_covered_buckets": [
            "|".join(str(value) for value in bucket)
            for bucket in sorted(source_only_covered_buckets)],
        "selected_forward_marker_counts": {
            "|".join(str(value) for value in bucket):
                len(marker_ids)
            for bucket, marker_ids in bucket_marker_ids.items()
        },
    }
    if operator_schema_manifest_path:
        plan["operator_schema_manifest"] = os.path.abspath(
            operator_schema_manifest_path)
    with open(out_path, "w") as fh:
        json.dump(plan, fh, indent=2)
    return plan["runtime_marker_mapping"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--capture-trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--shape-log", default="",
        help="optional runtime shape JSONL used only for explicit fallbacks")
    parser.add_argument(
        "--clean-table", default="",
        help="optional current clean-trace table used to gate cross-trace mapping")
    parser.add_argument(
        "--phase", action="append", default=[],
        help="phase required from this capture; repeat for multiple phases")
    parser.add_argument("--operator-schema-manifest", default="")
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = map_plan(
        args.capture_plan, args.capture_trace, args.out,
        shape_log_path=args.shape_log,
        clean_table_path=args.clean_table,
        required_phases=args.phase,
        operator_schema_manifest_path=args.operator_schema_manifest)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
