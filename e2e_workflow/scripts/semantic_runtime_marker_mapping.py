#!/usr/bin/env python3
"""Map unresolved Clean Trace rows to GEAK second-replay wrapper markers."""
import argparse
import collections
import gzip
import json
import os
import re


MARKER_PREFIX = "GEAK_SEMANTICS|"
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


def _kernel_key(name):
    value = str(name or "")
    value = re.sub(r"GRID_MN_\d+", "GRID_MN_*", value)
    value = re.sub(r"(_grid_)\d+(?=_|$)", r"\1*", value, flags=re.I)
    return value


def _runtime_entries(events):
    markers = []
    device_by_correlation = {}
    for index, event in enumerate(events):
        name = str(event.get("name", ""))
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
                "external_id": (event.get("args") or {}).get("External id"),
            })
        if event.get("cat") in DEVICE_CATEGORIES:
            correlation = (event.get("args") or {}).get(
                "correlation", (event.get("args") or {}).get("Correlation ID"))
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
        entries.append({
            "runtime_event_index": index,
            "runtime_name": event.get("name"),
            "correlation": correlation,
            "raw_name": kernel,
            "kernel_key": _kernel_key(kernel),
            "marker": marker,
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


def _first_forward_marker_ids(markers, bucket):
    """Select one complete wrapper-marker pass for a clean-trace bucket.

    Shape replays may execute the same decode bucket many times.  Matching all
    of them against the single representative clean layer makes every repeated
    kernel ambiguous.  Within one layer forward, registered module paths occur
    once; the first repeated path therefore starts the next forward.
    """
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
            candidates = sorted(
                (marker for marker in compatible
                 if (marker["batch_size"], marker["input_tokens"])
                 == selected_bucket),
                key=lambda marker: marker["index"])
    selected = []
    seen_paths = set()
    for marker in candidates:
        path = marker.get("op_path")
        if selected and path and path in seen_paths:
            break
        selected.append(marker)
        if path:
            seen_paths.add(path)
    return {
        marker["op_instance_id"] for marker in selected
        if marker.get("op_instance_id")
    }


def _apply_mapping(
        target, candidate, capture_trace_path, rule, marker_launch_count):
    marker = candidate["marker"]
    target["candidate_op_path"] = marker["op_path"]
    target["candidate_op_instance_id"] = marker["op_instance_id"]
    target["candidate_wrapper"] = marker["op_path"]
    target["candidate_terminal_launcher"] = candidate["runtime_name"]
    targeted_launcher = "::launcher:" in str(marker.get("op_path") or "")
    target["mapping_cardinality"] = (
        "1:1" if targeted_launcher and marker_launch_count == 1 else "1:N")
    target["source_mapping_status"] = "runtime_marker_contained"
    target["runtime_marker_mapping_status"] = "matched"
    target.pop("runtime_marker_candidate_count", None)
    target["runtime_marker_evidence"] = {
        "capture_trace": os.path.abspath(capture_trace_path),
        "marker_name": marker["name"],
        "marker_external_id": marker["external_id"],
        "runtime_event_index": candidate["runtime_event_index"],
        "runtime_name": candidate["runtime_name"],
        "runtime_correlation": candidate["correlation"],
        "captured_kernel": candidate["raw_name"],
        "targeted_launcher_probe": targeted_launcher,
        "marker_launch_count": marker_launch_count,
        "capture_device_event": candidate["device_event"],
        "rule": rule,
    }


def _shape_log_first_forward(path, targeted_only=False):
    selected = {}
    seen_paths = {}
    if not path or not os.path.exists(path):
        return selected
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
        targets, shape_log_path, missing_bucket_keys):
    """Use a unique executed semantic wrapper without positional guessing."""
    records_by_key = _shape_log_first_forward(shape_log_path)
    matched = 0
    for phase, layer_id in missing_bucket_keys:
        records = records_by_key.get((phase, layer_id), [])
        for target in targets:
            if target.get("runtime_marker_mapping_status") == "matched":
                continue
            if not _layer_fallback_eligible(target):
                continue
            if _phase(target.get("phase")) != phase:
                continue
            if int(target.get("representative_layer_id", -1)) != layer_id:
                continue
            candidates = _semantic_wrapper_candidates(
                records, target.get("stage"), layer_id)
            paths = {item.get("op_path") for item in candidates}
            if len(paths) != 1:
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
            target["candidate_op_path"] = record["op_path"]
            target["candidate_op_instance_id"] = record["op_instance_id"]
            target["candidate_wrapper"] = record["op_path"]
            target["candidate_terminal_launcher"] = record.get("op_type")
            target["mapping_cardinality"] = "1:N"
            target["source_mapping_status"] = (
                "runtime_shape_log_layer_wrapper")
            target["runtime_marker_mapping_status"] = "matched"
            target.pop("runtime_marker_candidate_count", None)
            target["shape_log_layer_evidence"] = {
                "shape_log": os.path.abspath(shape_log_path),
                "wrapper": record["op_path"],
                "scope": "phase_layer_wrapper",
                "rule": (
                    "clean trace assigns kernel to phase/layer; same replay "
                    "captures that layer wrapper shape; no kernel-level "
                    "profiler containment claimed"),
            }
            matched += 1
    return matched, covered_keys


def map_plan(
        plan_path, capture_trace_path, out_path, shape_log_path="",
        callable_kernel_map=None, source_wrapper_map=None):
    with open(plan_path) as fh:
        plan = json.load(fh)
    events = _load(capture_trace_path)
    markers, entries = _runtime_entries(events)
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
                markers, bucket)

    matched = 0
    ambiguous = 0
    unmatched = 0
    ambiguous_candidates = {}
    for key, targets in grouped_targets.items():
        targets.sort(key=lambda item: item.get("pos", -1))
        phase, layer_id, kernel_key, bucket = key
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
                    "kernel positions inside one eager forward"),
                marker_launch_counts[
                    inside[0]["marker"]["op_instance_id"]])
            matched += 1
            ambiguous -= 1

    source_callable_matched = _apply_source_callable_mapping(
        plan.get("capture_targets", []), shape_log_path,
        callable_kernel_map)
    source_wrapper_matched = _apply_source_wrapper_mapping(
        plan.get("capture_targets", []), shape_log_path,
        source_wrapper_map)
    missing_bucket_keys = {
        (bucket[0], bucket[1])
        for bucket, marker_ids in bucket_marker_ids.items()
        if not marker_ids
    }
    semantic_wrapper_matched = _apply_shape_log_semantic_wrapper_mapping(
        plan.get("capture_targets", []), shape_log_path,
        missing_bucket_keys)
    layer_fallback_matched, layer_fallback_keys = (
        _apply_shape_log_layer_fallback(
            plan.get("capture_targets", []), shape_log_path,
            missing_bucket_keys))
    statuses = [
        target.get("runtime_marker_mapping_status")
        for target in plan.get("capture_targets", [])]
    matched = statuses.count("matched")
    ambiguous = statuses.count("ambiguous_count")
    unmatched = statuses.count("not_found")
    missing_marker_buckets = [
        "|".join(str(value) for value in bucket)
        for bucket, marker_ids in bucket_marker_ids.items()
        if (not marker_ids
            and (bucket[0], bucket[1]) not in layer_fallback_keys)
    ]
    plan["runtime_marker_mapping"] = {
        "schema_version": 1,
        "capture_trace": os.path.abspath(capture_trace_path),
        "marker_count": len(markers),
        "contained_runtime_kernel_count": len(entries),
        "matched_target_count": matched,
        "source_callable_matched_target_count": source_callable_matched,
        "source_wrapper_matched_target_count": source_wrapper_matched,
        "shape_log_semantic_wrapper_matched_target_count": (
            semantic_wrapper_matched),
        "shape_log_layer_fallback_matched_target_count": (
            layer_fallback_matched),
        "ambiguous_target_count": ambiguous,
        "unmatched_target_count": unmatched,
        "phase_coverage_complete": not missing_marker_buckets,
        "missing_marker_buckets": missing_marker_buckets,
        "selected_forward_marker_counts": {
            "|".join(str(value) for value in bucket):
                len(marker_ids)
            for bucket, marker_ids in bucket_marker_ids.items()
        },
    }
    with open(out_path, "w") as fh:
        json.dump(plan, fh, indent=2)
    return plan["runtime_marker_mapping"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-plan", required=True)
    parser.add_argument("--capture-trace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = map_plan(args.capture_plan, args.capture_trace, args.out)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
