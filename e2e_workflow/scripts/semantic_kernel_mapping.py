#!/usr/bin/env python3
"""Build Pattern/Phase/Layer ordered device-event tables from one clean trace."""
import argparse
import bisect
import difflib
import gzip
import hashlib
import json
import math
import os
import re
import statistics

import parse_profile


DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")
LAYER_RE = re.compile(r"(?:layers?|h|blocks?)[./_\[](\d+)", re.IGNORECASE)
MODULE_LAYER_RE = re.compile(
    r"^nn\.Module:\s+.*DecoderLayer_(\d+)$", re.IGNORECASE)
SGLANG_STEP_RE = re.compile(
    r"^step\[(EXTEND|DECODE)\s+bs=(\d+)(?:\s+toks=(\d+))?\]$")


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _layer_id(event):
    args = event.get("args") or {}
    fields = [event.get("name", "")]
    fields.extend(str(args.get(key, "")) for key in (
        "Module Hierarchy", "Call stack", "Python parent id", "op_path"))
    text = " ".join(fields)
    match = LAYER_RE.search(text)
    return int(match.group(1)) if match else None


def _phase_name(tag):
    return "prefill" if tag == "P" else "decode"


def _collect_step_spans(events):
    """Recognize both legacy execute_* and current SGLang step[...] spans."""
    spans = []
    legacy = parse_profile._collect_step_spans(events)
    for index, span in enumerate(legacy):
        spans.append((span[0], span[1], span[2], span[3], span[4],
                      "legacy-%d" % index, "legacy_execute"))
    for raw_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("cat") != "gpu_user_annotation":
            continue
        name = event.get("name")
        match = SGLANG_STEP_RE.match(name) if isinstance(name, str) else None
        if not match or event.get("ts") is None or event.get("dur") is None:
            continue
        kind, batch, tokens = match.groups()
        tag = "P" if kind == "EXTEND" else "D"
        spans.append((
            event["ts"], event["ts"] + event["dur"], tag,
            int(tokens or 0), int(batch), "step-%d" % raw_index,
            "sglang_step_annotation"))
    spans.sort(key=lambda item: (item[0], item[1], item[5]))
    return spans


STAGE_RULESET_VERSION = "semantic-stage-v2"
STAGE_RULES = (
    ("communication.collective", "communication",
     r"all.?reduce|reduce.?scatter|all.?gather|nccl|rccl|quickreduce|cross_device"),
    ("norm.layer", "norm", r"rms.?norm|layer.?norm|l2norm"),
    ("position.rope", "rope", r"rope|rotary"),
    ("attention.linear", "linear_attn",
     r"fused_qkvzba|causal_conv1d|gdn_gating|gated_delta|"
     r"recompute_w_u|chunk_local_cumsum|chunk_fwd_kernel_o|"
     r"state_passing|wv_splitk_small"),
    ("attention.full_or_mla", "attn", r"fmha|attention|attn|paged|mla_"),
    ("router.topk", "topk", r"topk|routing|router|gate_kernel"),
    ("experts.moe", "moe", r"\bmoe\b|expert|sorting|fmoe"),
    ("linear.gemm", "gemm", r"gemm|cijk|tensile|matmul|\bmm\b"),
    ("cache.kv", "kv_cache", r"cache|index_put"),
    ("activation", "activation", r"silu|gelu|swiglu|act_and_mul"),
    ("quantization", "quant", r"quant|dequant|float8|fp8"),
    ("elementwise", "elementwise", r"elementwise|copy|cast|fill|add|mul"),
)


def _stage_detail(name, category, parent_name=""):
    if category in ("gpu_memcpy", "gpu_memset"):
        return "memory", "event.memory", "event_category"
    value = name.lower()
    for rule_id, stage, regex in STAGE_RULES:
        if re.search(regex, value):
            return stage, rule_id, "kernel_name"
    parent_value = parent_name.lower()
    if re.search(r"gated.?delta|linear.?attention|causal.?conv", parent_value):
        return "linear_attn", "attention.linear.parent", "parent_operator"
    return "unknown", "unresolved", "unresolved"


def _stage(name, category):
    return _stage_detail(name, category)[0]


def _load_events(path):
    with _open(path) as fh:
        data = json.load(fh)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def _cpu_evidence(events):
    by_ext = {}
    scopes = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("cat") != "cpu_op":
            continue
        args = event.get("args") or {}
        evidence = {
            "name": str(event.get("name", "unresolved")),
            "external_id": args.get("External id"),
            "input_dims": args.get("Input Dims") or [],
            "input_types": args.get("Input type") or [],
            "layer_id": _layer_id(event),
            "ts": event.get("ts"),
            "end": ((event.get("ts") or 0) + (event.get("dur") or 0)),
            "event_index": index,
        }
        ext = evidence["external_id"]
        if ext is not None:
            old = by_ext.get(ext)
            if old is None or (not old["input_dims"] and evidence["input_dims"]):
                by_ext[ext] = evidence
        if evidence["layer_id"] is not None and event.get("ts") is not None and event.get("dur") is not None:
            scopes.append(evidence)
    scopes.sort(key=lambda item: (item["ts"], item["end"]))
    return by_ext, scopes, [item["ts"] for item in scopes]


def _step_at(ts, spans, starts):
    if ts is None or not spans:
        return None
    pos = bisect.bisect_right(starts, ts) - 1
    if pos >= 0 and ts < spans[pos][1]:
        return spans[pos]
    return None


def _expected_layer_type(pattern):
    pid = str((pattern or {}).get("pattern_id", "")).lower()
    attention = str((pattern or {}).get("attention_type", "")).lower()
    if "full" in pid or "full" in attention:
        return "full"
    if "linear" in pid or "linear" in attention:
        return "linear"
    return ""


def _module_layer_scopes(events, spans, pattern_doc):
    """Resolve outer DecoderLayer python spans to global layer ordinals.

    Numeric suffixes are class-local for some hybrid model implementations, so
    execution order inside each full N-layer pass is authoritative. The suffix
    remains validation evidence only.
    """
    expected_count = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)
    if expected_count <= 0:
        return [], []
    pattern_by_layer = _pattern_index(pattern_doc)
    span_starts = [span[0] for span in spans]
    candidates = []
    for raw_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("cat") != "python_function":
            continue
        name = str(event.get("name", ""))
        match = MODULE_LAYER_RE.match(name)
        if not match or event.get("ts") is None or event.get("dur") is None:
            continue
        step = _step_at(event["ts"], spans, span_starts)
        if step is None:
            continue
        candidates.append({
            "name": name,
            "class_local_id": int(match.group(1)),
            "ts": event["ts"],
            "end": event["ts"] + event["dur"],
            "event_index": raw_index,
            "step_id": step[5],
            "phase": _phase_name(step[2]),
        })
    by_step = {}
    for candidate in candidates:
        by_step.setdefault(candidate["step_id"], []).append(candidate)
    scopes = []
    diagnostics = []
    for step_id, values in sorted(by_step.items()):
        values.sort(key=lambda item: (item["ts"], item["end"]))
        full_passes = len(values) // expected_count
        remainder = len(values) % expected_count
        diagnostics.append({
            "step_id": step_id,
            "candidate_count": len(values),
            "expected_layer_count": expected_count,
            "full_passes": full_passes,
            "remainder": remainder,
        })
        for pass_index in range(full_passes):
            chunk = values[pass_index * expected_count:(pass_index + 1) * expected_count]
            for layer_id, item in enumerate(chunk):
                item = dict(item)
                expected_type = _expected_layer_type(pattern_by_layer.get(layer_id))
                actual_type = ("full" if "AttentionDecoderLayer" in item["name"]
                               else "linear" if "LinearDecoderLayer" in item["name"]
                               else "")
                item.update({
                    "layer_id": layer_id,
                    "pattern_id": (pattern_by_layer.get(layer_id) or {}).get("pattern_id"),
                    "pass_index": pass_index,
                    "layer_instance_id": "%s:pass-%d:layer-%d" % (
                        step_id, pass_index, layer_id),
                    "type_validation": (
                        "pass" if not expected_type or expected_type == actual_type
                        else "mismatch"),
                })
                scopes.append(item)
    scopes.sort(key=lambda item: (item["ts"], item["end"]))
    return scopes, diagnostics


def _module_scope_at(ts, scopes, starts):
    if ts is None or not scopes:
        return None
    pos = bisect.bisect_right(starts, ts)
    matches = []
    for item in reversed(scopes[max(0, pos - 256):pos]):
        if item["ts"] <= ts < item["end"]:
            matches.append(item)
    if not matches:
        return None
    # Prefer a type-validated outer decoder span, then the narrowest interval.
    return min(matches, key=lambda item: (
        item["type_validation"] != "pass", item["end"] - item["ts"]))


def _scope_at(ts, scopes, starts):
    if ts is None or not scopes:
        return None
    pos = bisect.bisect_right(starts, ts)
    best = None
    # Torch CPU scopes are nested and the relevant layer scope is normally among
    # the most recently opened scopes. The bound avoids O(device*cpu) traces.
    for item in reversed(scopes[max(0, pos - 256):pos]):
        if item["ts"] <= ts < item["end"]:
            if best is None or (item["end"] - item["ts"]) < (best["end"] - best["ts"]):
                best = item
    return best


def _flow_layer_index(events, module_scopes, module_starts):
    """Map GPU flow-marker timestamps from CPU-side DecoderLayer spans."""
    flows = {}
    for event in events:
        if not isinstance(event, dict) or event.get("ph") not in ("s", "t", "f"):
            continue
        flow_id = event.get("id", (event.get("args") or {}).get("id"))
        if flow_id is not None and event.get("ts") is not None:
            flows.setdefault(str(flow_id), []).append(event.get("ts"))
    result = {}
    for timestamps in flows.values():
        sources = [_module_scope_at(ts, module_scopes, module_starts)
                   for ts in timestamps]
        sources = [source for source in sources if source]
        instance_ids = set(source["layer_instance_id"] for source in sources)
        if len(instance_ids) != 1:
            continue
        for ts in timestamps:
            result[ts] = sources[0]
    return result


def _phase_at(ts, spans, starts):
    span = _step_at(ts, spans, starts)
    return _phase_name(span[2]) if span else None


def _pattern_index(pattern_doc):
    result = {}
    for pattern in pattern_doc.get("patterns", []):
        for layer_id in pattern.get("layer_ids", []):
            result[int(layer_id)] = pattern
    return result


def _event_rows(events, pattern_doc):
    patterns = _pattern_index(pattern_doc)
    spans = _collect_step_spans(events)
    span_starts = [span[0] for span in spans]
    cpu_by_ext, scopes, scope_starts = _cpu_evidence(events)
    module_scopes, module_diagnostics = _module_layer_scopes(
        events, spans, pattern_doc)
    module_starts = [scope["ts"] for scope in module_scopes]
    flow_layers = _flow_layer_index(events, module_scopes, module_starts)
    rows = []
    out_of_scope = {"count": 0, "duration_us": 0.0}
    device_sequence = 0
    device_events = [(raw_index, event) for raw_index, event in enumerate(events)
                     if isinstance(event, dict)
                     and event.get("cat") in DEVICE_CATEGORIES]
    device_events.sort(key=lambda item: (
        item[1].get("ts") is None, item[1].get("ts") or 0, item[0]))
    external_id_launch_count = {}
    for _, device_event in device_events:
        device_args = device_event.get("args") or {}
        device_external_id = device_args.get("External id")
        if device_external_id is not None:
            external_id_launch_count[device_external_id] = (
                external_id_launch_count.get(device_external_id, 0) + 1)
    for raw_index, event in device_events:
        ts = event.get("ts")
        step = _step_at(ts, spans, span_starts)
        phase = _phase_name(step[2]) if step else None
        if spans and phase is None:
            out_of_scope["count"] += 1
            out_of_scope["duration_us"] += float(event.get("dur", 0) or 0)
            continue
        device_sequence += 1
        args = event.get("args") or {}
        ext = args.get("External id")
        parent = cpu_by_ext.get(ext)
        scope = _scope_at(ts, scopes, scope_starts)
        module_scope = _module_scope_at(
            (parent or {}).get("ts"), module_scopes, module_starts)
        flow_scope = flow_layers.get(ts)
        layer_instance_id = None
        if module_scope is not None:
            layer_id = module_scope["layer_id"]
            layer_instance_id = module_scope["layer_instance_id"]
            layer_evidence = "python_module_span_external_id"
        elif flow_scope is not None:
            layer_id = flow_scope["layer_id"]
            layer_instance_id = flow_scope["layer_instance_id"]
            module_scope = flow_scope
            layer_evidence = "python_module_span_ac2g_flow"
        elif (parent or {}).get("layer_id") is not None:
            layer_id = parent["layer_id"]
            layer_evidence = "cpu_op_external_id"
        elif (scope or {}).get("layer_id") is not None:
            layer_id = scope["layer_id"]
            layer_evidence = "cpu_op_scope"
        elif ts in flow_layers:
            layer_id = flow_layers[ts]
            layer_evidence = "trace_flow"
        else:
            layer_id = None
            layer_evidence = "unresolved"
        parent = parent or scope
        pattern = patterns.get(layer_id)
        assignment = "layer_body" if pattern else (
            "transition_global" if parent else "concurrent_unresolved")
        name = str(event.get("name", "?"))
        classification, provider, _, _ = parse_profile.classify(name)
        dims = (parent or {}).get("input_dims") or []
        types = (parent or {}).get("input_types") or []
        one_to_one_launch = (
            ext is not None and external_id_launch_count.get(ext) == 1)
        shape_source = "kernel_exact" if dims and one_to_one_launch else (
            "parent_context" if parent else "unresolved")
        stage, stage_rule_id, stage_source = _stage_detail(
            name, event.get("cat"), (parent or {}).get("name", ""))
        rows.append({
            "row_id": "event-%d" % raw_index,
            "raw_event_index": raw_index,
            "device_seq_index": device_sequence,
            "timestamp": ts,
            "duration_us": float(event.get("dur", 0) or 0),
            "stream": args.get("stream", args.get("Stream")),
            "external_id": ext,
            "correlation": args.get("correlation", args.get("Correlation ID")),
            "event_type": event.get("cat"),
            "phase": phase or "unresolved",
            "phase_source": step[6] if step else "unresolved",
            "step_id": step[5] if step else None,
            "step_batch_size": step[4] if step else None,
            "step_input_tokens": step[3] if step else None,
            "assignment": assignment,
            "layer_id": layer_id,
            "layer_instance_id": layer_instance_id,
            "layer_evidence": layer_evidence,
            "layer_region": None,
            "boundary_role": None,
            "pattern_id": pattern.get("pattern_id") if pattern else None,
            "raw_name": name,
            "short_name": parse_profile.short_name(name),
            "classification": classification,
            "stage": stage,
            "stage_rule_id": stage_rule_id,
            "stage_source": stage_source,
            "stage_ruleset_version": STAGE_RULESET_VERSION,
            "provider": provider,
            "parent_operator": {
                "op_instance_id": "ext-%s" % ext if ext is not None else None,
                "canonical_op": (parent or {}).get("name", "unresolved"),
                "mapping_level": "external_id" if ext in cpu_by_ext else (
                    "cpu_scope" if scope else "unresolved"),
                "mapping_cardinality": (
                    "1:1" if ext in cpu_by_ext and one_to_one_launch else
                    "1:N" if ext in cpu_by_ext else "unresolved"),
                "device_launch_count": external_id_launch_count.get(ext),
                "confidence": "high" if ext in cpu_by_ext else (
                    "medium" if scope else "low"),
                "evidence_event_index": (parent or {}).get("event_index"),
            },
            "shape": {
                "source": shape_source,
                "input_dims": dims,
                "input_types": types,
            },
        })
    out_of_scope["duration_us"] = round(out_of_scope["duration_us"], 6)
    return rows, spans, out_of_scope, module_scopes, module_diagnostics


def _complete_module_ranges(rows, module_scopes, patterns):
    """Fill ext-less launches between module-backed layer launch ranges.

    The interpolation never crosses a measured module pass and only fills the
    interval between its first and last module-backed GPU launches.
    """
    backed = {}
    for row in rows:
        if row.get("layer_instance_id"):
            backed.setdefault(row["layer_instance_id"], []).append(row)
    by_pass = {}
    for scope in module_scopes:
        key = (scope["step_id"], scope["pass_index"])
        if scope["layer_instance_id"] in backed:
            group = backed[scope["layer_instance_id"]]
            by_pass.setdefault(key, []).append({
                "scope": scope,
                "first": min(row["device_seq_index"] for row in group),
                "last": max(row["device_seq_index"] for row in group),
            })
    filled = 0
    for (step_id, _), intervals in by_pass.items():
        intervals.sort(key=lambda item: item["scope"]["layer_id"])
        if len(intervals) < 2:
            continue
        centers = [(item["first"] + item["last"]) / 2.0 for item in intervals]
        bounds = [(centers[index] + centers[index + 1]) / 2.0
                  for index in range(len(centers) - 1)]
        pass_start = intervals[0]["first"]
        pass_end = intervals[-1]["last"]
        for row in rows:
            if (row.get("step_id") != step_id or row.get("layer_instance_id")
                    or not (pass_start <= row["device_seq_index"] <= pass_end)):
                continue
            pos = bisect.bisect_right(bounds, row["device_seq_index"])
            target = intervals[min(pos, len(intervals) - 1)]["scope"]
            row["layer_id"] = target["layer_id"]
            row["layer_instance_id"] = target["layer_instance_id"]
            row["pattern_id"] = (patterns.get(target["layer_id"]) or {}).get("pattern_id")
            row["assignment"] = "layer_body"
            row["layer_evidence"] = "module_sequence_interpolation"
            filled += 1
    return filled


def _deduped_stage_sequence(group):
    sequence = []
    for row in sorted(group, key=lambda item: item["device_seq_index"]):
        stage = row["stage"]
        if not sequence or sequence[-1] != stage:
            sequence.append(stage)
    return sequence


def _sequence_ratio(left, right):
    if not left and not right:
        return 1.0
    return difflib.SequenceMatcher(
        None, tuple(left), tuple(right), autojunk=False).ratio()


def _sequence_medoid(sequences):
    """Choose one observed sequence with minimum total edit-distance proxy."""
    counts = {}
    for sequence in sequences:
        key = tuple(sequence)
        if key:
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return []
    scored = []
    for candidate in sorted(counts):
        distance = sum(
            (1.0 - _sequence_ratio(candidate, other)) * count
            for other, count in counts.items())
        scored.append((round(distance, 12), len(candidate), candidate))
    return list(min(scored)[2])


def _module_pattern_templates(rows):
    """Learn full Pattern medoids plus exact External-ID core medoids."""
    full_groups = {}
    core_groups = {}
    for row in rows:
        instance_id = row.get("layer_instance_id")
        if not instance_id or row.get("assignment") != "layer_body":
            continue
        evidence = row.get("layer_evidence", "")
        if (evidence.startswith("python_module_span")
                or evidence == "module_sequence_interpolation"):
            full_groups.setdefault(instance_id, []).append(row)
        if evidence == "python_module_span_external_id":
            core_groups.setdefault(instance_id, []).append(row)

    def collect(groups):
        result = {}
        for group in groups.values():
            sequence = _deduped_stage_sequence(group)
            if sequence:
                key = (group[0].get("phase"), group[0].get("pattern_id"))
                result.setdefault(key, []).append(sequence)
        return result

    full_by_pattern = collect(full_groups)
    core_by_pattern = collect(core_groups)
    templates, core_templates, core_prefixes = {}, {}, {}
    evidence = {}
    pattern_ids = sorted(set(
        pid for source in (full_by_pattern, core_by_pattern)
        for _, pid in source if pid))
    for pattern_id in pattern_ids:
        full_prefill = full_by_pattern.get(("prefill", pattern_id), [])
        full_values = full_prefill or [
            sequence for (phase, pid), sequences in full_by_pattern.items()
            if pid == pattern_id for sequence in sequences]
        core_prefill = core_by_pattern.get(("prefill", pattern_id), [])
        core_values = core_prefill or [
            sequence for (phase, pid), sequences in core_by_pattern.items()
            if pid == pattern_id for sequence in sequences]
        full_template = _sequence_medoid(full_values)
        core_template = _sequence_medoid(core_values)
        if full_template:
            templates[pattern_id] = full_template
        if core_template:
            core_templates[pattern_id] = core_template
            prefixes = []
            for sequence in full_values:
                candidates = [
                    position for position, stage in enumerate(sequence)
                    if stage == core_template[0]]
                if not candidates:
                    continue
                position = min(candidates, key=lambda value: (
                    -_sequence_ratio(core_template, sequence[value:]),
                    value,
                ))
                if position:
                    prefixes.append(sequence[max(0, position - 4):position])
            prefix = _sequence_medoid(prefixes)
            if prefix:
                core_prefixes[pattern_id] = prefix
        if full_template or core_template:
            evidence[pattern_id] = {
                "source": "module_full_and_external_core_medoids",
                "full_instance_count": len(full_values),
                "full_stage_count": len(full_template),
                "core_instance_count": len(core_values),
                "core_stage_count": len(core_template),
                "learned_core_prefix": core_prefixes.get(pattern_id, []),
            }
    return templates, core_templates, core_prefixes, evidence


def _stage_runs(step_rows):
    """Return lossless row ranges for continuously deduplicated stages."""
    runs = []
    for index, row in enumerate(step_rows):
        if not runs or runs[-1]["stage"] != row["stage"]:
            runs.append({"stage": row["stage"], "start": index, "end": index})
        else:
            runs[-1]["end"] = index
    return runs


def _bootstrap_missing_templates(runs, patterns, templates, layer_count):
    """Build deterministic medoids when no module-backed template exists."""
    if not runs or layer_count <= 0:
        return dict(templates), {}
    provisional = {}
    run_count = len(runs)
    for layer_id in range(layer_count):
        start = int(round(float(layer_id) * run_count / layer_count))
        end = int(round(float(layer_id + 1) * run_count / layer_count))
        if end <= start:
            end = min(run_count, start + 1)
        pattern_id = (patterns.get(layer_id) or {}).get("pattern_id")
        sequence = [run["stage"] for run in runs[start:end]]
        if pattern_id and sequence:
            provisional.setdefault(pattern_id, []).append(sequence)
    result = dict(templates)
    evidence = {}
    for pattern_id, sequences in sorted(provisional.items()):
        if pattern_id not in result:
            result[pattern_id] = _sequence_medoid(sequences)
            evidence[pattern_id] = {
                "source": "self_bootstrap_medoid",
                "instance_count": len(sequences),
                "stage_count": len(result[pattern_id]),
            }
    return result, evidence


def _rough_alignment_bounds(observed, expected):
    """Map expected positions to observed positions using matching blocks."""
    observed_count, expected_count = len(observed), len(expected)
    if not observed_count or not expected_count:
        mapped = [
            int(round(float(index) * observed_count / max(expected_count, 1)))
            for index in range(expected_count + 1)]
        return 0, observed_count, mapped
    matcher = difflib.SequenceMatcher(
        None, tuple(expected), tuple(observed), autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        mapped = [
            int(round(float(index) * observed_count / expected_count))
            for index in range(expected_count + 1)]
        return 0, observed_count, mapped
    scale = float(observed_count) / expected_count
    first, last = blocks[0], blocks[-1]
    start = max(0, int(round(first.b - first.a * scale)))
    expected_tail = expected_count - (last.a + last.size)
    end = min(observed_count, int(round(
        last.b + last.size + expected_tail * scale)))
    if end - start < 1:
        start, end = 0, observed_count
    anchors = [(0, start), (expected_count, end)]
    for block in blocks:
        anchors.extend([
            (block.a, block.b),
            (block.a + block.size, block.b + block.size),
        ])
    anchors = sorted(set(anchors))
    expected_anchors = [item[0] for item in anchors]
    mapped = []
    for expected_pos in range(expected_count + 1):
        right = bisect.bisect_left(expected_anchors, expected_pos)
        if right <= 0:
            observed_pos = anchors[0][1]
        elif right >= len(anchors):
            observed_pos = anchors[-1][1]
        else:
            left_item, right_item = anchors[right - 1], anchors[right]
            width = right_item[0] - left_item[0]
            fraction = (
                float(expected_pos - left_item[0]) / width if width else 0.0)
            observed_pos = int(round(
                left_item[1] + fraction * (right_item[1] - left_item[1])))
        mapped.append(max(start, min(end, observed_pos)))
    return start, end, mapped


def _segment_alignment_score(observed, template, target_length):
    ratio = _sequence_ratio(observed, template)
    length_penalty = (
        abs(len(observed) / float(target_length) - 1.0)
        if target_length else 0.0)
    return ratio - 0.15 * length_penalty, ratio


def _align_pattern_chain(runs, patterns, templates, layer_count):
    """Globally align one full step to the config-declared Pattern chain."""
    observed = [run["stage"] for run in runs]
    chain = []
    expected = []
    cumulative = [0]
    for layer_id in range(layer_count):
        pattern_id = (patterns.get(layer_id) or {}).get("pattern_id")
        template = list(templates.get(pattern_id) or ["unknown"])
        chain.append((layer_id, pattern_id, template))
        expected.extend(template)
        cumulative.append(len(expected))
    start, end, mapped = _rough_alignment_bounds(observed, expected)
    if end - start < layer_count:
        start, end = 0, len(observed)
    available = max(end - start, layer_count)
    scale = float(available) / max(len(expected), 1)
    average = float(available) / max(layer_count, 1)
    # Rough matching blocks already provide the global position. A narrow,
    # deterministic refinement band keeps long graph-replay traces linear-ish.
    radius = max(2, int(round(average * 0.15)))
    candidates = [[start]]
    for layer_index in range(1, layer_count):
        center = mapped[cumulative[layer_index]]
        low = max(start + layer_index, center - radius)
        high = min(end - (layer_count - layer_index), center + radius)
        values = list(range(low, high + 1))
        if not values:
            values = [max(start + layer_index, min(
                end - (layer_count - layer_index), center))]
        candidates.append(values)
    candidates.append([end])
    states = {start: [(0.0, [], [])]}
    for layer_index, (_, _, template) in enumerate(chain):
        next_states = {}
        target = max(1.0, len(template) * scale)
        for stop in candidates[layer_index + 1]:
            options = []
            for begin, records in states.items():
                if stop <= begin:
                    continue
                sequence = observed[begin:stop]
                segment_score, ratio = _segment_alignment_score(
                    sequence, template, target)
                for score, path, ratios in records:
                    options.append((
                        score + segment_score,
                        path + [stop],
                        ratios + [ratio],
                    ))
            if options:
                options.sort(key=lambda item: (-item[0], item[1]))
                next_states[stop] = options[:2]
        states = next_states
    records = states.get(end, [])
    if not records:
        cuts = [start]
        for index in range(1, layer_count):
            cuts.append(int(round(
                start + float(index) * (end - start) / layer_count)))
        cuts.append(end)
        for index in range(1, len(cuts)):
            cuts[index] = max(cuts[index], cuts[index - 1] + 1)
        return cuts, float("-inf"), None, 0.0, chain
    best = records[0]
    second_score = records[1][0] if len(records) > 1 else None
    mean_ratio = sum(best[2]) / len(best[2]) if best[2] else 0.0
    return ([start] + best[1], best[0], second_score, mean_ratio, chain)


def _refine_cuts_with_core_medoids(
        runs, cuts, chain, core_templates, core_prefixes, allow_lookback):
    """Split bridge runs once between adjacent aligned Pattern cores."""
    core_ranges = []
    average_width = float(cuts[-1] - cuts[0]) / max(len(chain), 1)
    lookback = (
        max(3, int(round(average_width * 0.6))) if allow_lookback else 0)
    for index, (_, pattern_id, full_template) in enumerate(chain):
        start, end = cuts[index], cuts[index + 1]
        core = core_templates.get(pattern_id) or []
        search_end = min(
            end, start + max(6, int(round((end - start) * 0.35))))
        pattern_transition = (
            index > 0 and chain[index - 1][1] != pattern_id)
        effective_lookback = (
            lookback if allow_lookback else
            max(3, int(round(average_width * 0.6)))
            if pattern_transition else 0)
        search_start = max(0, start - effective_lookback)
        candidates = [
            position for position in range(search_start, search_end)
            if core and runs[position]["stage"] == core[0]]
        if candidates:
            prefix = core_prefixes.get(pattern_id) or []

            def prefix_match_width(position):
                for width in range(min(len(prefix), position), 0, -1):
                    observed_prefix = [
                        run["stage"] for run in runs[position - width:position]]
                    if observed_prefix == prefix[-width:]:
                        return width
                return 0

            core_start = min(candidates, key=lambda position: (
                -prefix_match_width(position),
                -_sequence_ratio(
                    core, [run["stage"] for run in runs[position:end]]),
                abs(position - start),
                position,
            ))
        else:
            core_start = start
        observed = [run["stage"] for run in runs[core_start:end]]
        blocks = [
            block for block in difflib.SequenceMatcher(
                None, tuple(core), tuple(observed),
                autojunk=False).get_matching_blocks()
            if block.size]
        if blocks:
            core_ranges.append((
                core_start,
                core_start + blocks[-1].b + blocks[-1].size,
            ))
        else:
            core_ranges.append((start, end))
    if not core_ranges:
        return cuts
    refined = []
    for index, (core_start, _) in enumerate(core_ranges):
        boundary = core_start
        prefix = core_prefixes.get(chain[index][1]) or []
        for width in range(min(len(prefix), core_start), 0, -1):
            observed_prefix = [
                run["stage"] for run in runs[core_start - width:core_start]]
            if observed_prefix == prefix[-width:]:
                # The closest learned predecessor belongs to the current layer;
                # earlier bridge runs remain owned by the preceding layer.
                boundary -= 1
                break
        minimum = refined[-1] + 1 if refined else 0
        maximum = cuts[-1] - (len(core_ranges) - index)
        refined.append(max(minimum, min(maximum, boundary)))
    refined.append(cuts[-1])
    return refined


def _transition_stability(cuts, runs, chain):
    by_transition = {}
    for index in range(1, len(cuts) - 1):
        left = runs[cuts[index] - 1]["stage"]
        right = runs[cuts[index]]["stage"]
        key = (chain[index - 1][1], chain[index][1])
        by_transition.setdefault(key, []).append((left, right))
    coverages = []
    for values in by_transition.values():
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        coverages.append(float(max(counts.values())) / len(values))
    return sum(coverages) / len(coverages) if coverages else 1.0


def _refine_cuts_with_stable_transition_context(runs, cuts, chain):
    """Resolve repeated within-layer transitions from the dominant Pattern."""
    pattern_counts = {}
    for _, pattern_id, _ in chain:
        pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1
    if not pattern_counts:
        return cuts
    dominant_pattern = min(
        pattern_counts,
        key=lambda pattern_id: (-pattern_counts[pattern_id], str(pattern_id)))

    def context(position):
        return [
            run["stage"] for run in
            runs[max(0, position - 2):min(len(runs), position + 6)]]

    contexts = [
        context(cuts[index])
        for index in range(1, len(chain))
        if chain[index][1] == dominant_pattern]
    template = _sequence_medoid(contexts)
    if not template:
        return cuts
    average = float(cuts[-1] - cuts[0]) / max(len(chain), 1)
    radius = max(3, int(round(average * 0.6)))
    refined = [cuts[0]]
    for index in range(1, len(cuts) - 1):
        low = max(refined[-1] + 1, cuts[index] - radius)
        high = min(cuts[-1] - (len(cuts) - index - 1),
                   cuts[index] + radius)
        candidates = range(low, high + 1)
        chosen = min(candidates, key=lambda position: (
            -_sequence_ratio(template, context(position)),
            abs(position - cuts[index]),
            position,
        ))
        refined.append(chosen)
    refined.append(cuts[-1])
    return refined


def _module_guided_segments(
        step_rows, runs, patterns, core_templates, core_prefixes, layer_count):
    row_to_run = {}
    for run_index, run in enumerate(runs):
        for row_index in range(run["start"], run["end"] + 1):
            row_to_run[step_rows[row_index]["row_id"]] = run_index
    groups = {}
    for row in step_rows:
        instance_id = row.get("layer_instance_id")
        if (instance_id and
                row.get("layer_evidence") == "python_module_span_external_id"):
            groups.setdefault(instance_id, []).append(row)
    candidates = []
    for instance_id, group in groups.items():
        match = re.search(r":pass-(\d+):layer-(\d+)$", instance_id)
        if not match:
            continue
        pass_index, layer_id = int(match.group(1)), int(match.group(2))
        pattern_id = (patterns.get(layer_id) or {}).get("pattern_id")
        core = core_templates.get(pattern_id) or []
        positions = [
            row_to_run[row["row_id"]] for row in group
            if row["row_id"] in row_to_run]
        core_positions = [
            row_to_run[row["row_id"]] for row in group
            if core and row["row_id"] in row_to_run
            and row["stage"] == core[0]]
        if not positions:
            continue
        start = min(core_positions or positions)
        prefix = core_prefixes.get(pattern_id) or []
        if (prefix and start > 0
                and runs[start - 1]["stage"] == prefix[-1]):
            start -= 1
        candidates.append({
            "pass_index": pass_index,
            "layer_id": layer_id,
            "pattern_id": pattern_id,
            "start": start,
        })
    candidates.sort(key=lambda item: (item["pass_index"], item["layer_id"]))
    if len(candidates) < layer_count:
        return []
    previous = -1
    for index, candidate in enumerate(candidates):
        minimum = previous + 1
        maximum = len(runs) - (len(candidates) - index)
        candidate["start"] = max(minimum, min(maximum, candidate["start"]))
        previous = candidate["start"]
    for index, candidate in enumerate(candidates):
        candidate["end"] = (
            candidates[index + 1]["start"]
            if index + 1 < len(candidates) else len(runs))
    return candidates


def _stage_sequence_partition(rows, pattern_doc):
    """Partition module-less steps without operator/backend boundary names."""
    layer_count = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)
    patterns = _pattern_index(pattern_doc)
    if layer_count <= 0:
        return [], {}
    (templates, core_templates, core_prefixes,
     template_evidence) = _module_pattern_templates(rows)
    alignment_cache = {}
    by_step = {}
    for row in rows:
        if row.get("step_id"):
            by_step.setdefault(row["step_id"], []).append(row)
    diagnostics = []
    for step_id, step_rows in sorted(by_step.items()):
        step_rows.sort(key=lambda row: row["device_seq_index"])
        module_instances = set(
            row["layer_instance_id"] for row in step_rows
            if row.get("layer_instance_id") and row.get("layer_evidence") and (
                row["layer_evidence"].startswith("python_module_span")
                or row["layer_evidence"] == "module_sequence_interpolation"))
        runs = _stage_runs(step_rows)
        if len(runs) < layer_count:
            # This is physically impossible to split into non-empty layers
            # without duplicating a device event, so preserve existing evidence.
            diagnostics.append({
                "step_id": step_id,
                "status": "insufficient_physical_events",
                "partition_method": "forced_best_alignment",
                "configured_layer_count": layer_count,
                "observed_stage_run_count": len(runs),
            })
            continue
        if len(module_instances) >= layer_count:
            guided = _module_guided_segments(
                step_rows, runs, patterns, core_templates, core_prefixes,
                layer_count)
            if guided:
                for row in step_rows:
                    row["assignment"] = "transition_global"
                    row["layer_id"] = None
                    row["layer_instance_id"] = None
                    row["pattern_id"] = None
                    row["layer_evidence"] = "sequence_outside_layer"
                    row["layer_region"] = "transition_global"
                    row["boundary_role"] = None
                mapped_count = 0
                cut_events = []
                for item in guided:
                    first_run, last_run = item["start"], item["end"] - 1
                    first_row = runs[first_run]["start"]
                    last_row = runs[last_run]["end"]
                    segment = step_rows[first_row:last_row + 1]
                    instance_id = (
                        "%s:module-sequence:pass-%d:layer-%d" % (
                            step_id, item["pass_index"], item["layer_id"]))
                    pattern = patterns.get(item["layer_id"]) or {}
                    for index, row in enumerate(segment):
                        row["layer_id"] = item["layer_id"]
                        row["layer_instance_id"] = instance_id
                        row["pattern_id"] = pattern.get("pattern_id")
                        row["assignment"] = "layer_body"
                        row["layer_evidence"] = "module_span_sequence_medoid"
                        row["layer_region"] = "layer_body"
                        row["boundary_role"] = (
                            "body_start_kernel" if index == 0
                            else "end_kernel"
                            if index == len(segment) - 1 else None)
                    mapped_count += len(segment)
                    cut_events.append({
                        "pass_index": item["pass_index"],
                        "layer_id": item["layer_id"],
                        "body_start_event": segment[0]["row_id"],
                        "body_end_event": segment[-1]["row_id"],
                    })
                diagnostics.append({
                    "step_id": step_id,
                    "status": "mapped",
                    "partition_method": "module_span_sequence_medoid",
                    "configured_layer_count": layer_count,
                    "module_instance_count": len(module_instances),
                    "mapped_pass_count": len(guided) // layer_count,
                    "observed_stage_run_count": len(runs),
                    "mapped_event_count": mapped_count,
                    "template_evidence": template_evidence,
                    "layer_boundaries": cut_events,
                })
                continue
        step_templates, bootstrap_evidence = _bootstrap_missing_templates(
            runs, patterns, templates, layer_count)
        cache_key = (
            tuple(run["stage"] for run in runs),
            tuple(
                ((patterns.get(layer_id) or {}).get("pattern_id"),
                 tuple(step_templates.get(
                     (patterns.get(layer_id) or {}).get("pattern_id"), [])))
                for layer_id in range(layer_count)),
        )
        if cache_key not in alignment_cache:
            alignment_cache[cache_key] = _align_pattern_chain(
                runs, patterns, step_templates, layer_count)
        cuts, score, second_score, mean_ratio, chain = alignment_cache[cache_key]
        cuts = _refine_cuts_with_core_medoids(
            runs, cuts, chain, core_templates, core_prefixes,
            allow_lookback=len(module_instances) >= layer_count)
        if len(module_instances) < layer_count:
            cuts = _refine_cuts_with_stable_transition_context(
                runs, cuts, chain)
        stability = _transition_stability(cuts, runs, chain)
        method = (
            "module_span_sequence_medoid"
            if len(module_instances) >= layer_count
            else "repeated_sequence_medoid"
            if mean_ratio >= 0.45 and stability >= 0.5
            else "forced_best_alignment")
        for row in step_rows:
            row["assignment"] = "transition_global"
            row["layer_id"] = None
            row["layer_instance_id"] = None
            row["pattern_id"] = None
            row["layer_evidence"] = "sequence_outside_layer"
            row["layer_region"] = "transition_global"
            row["boundary_role"] = None
        mapped_count = 0
        cut_events = []
        for layer_id in range(layer_count):
            first_run, last_run = cuts[layer_id], cuts[layer_id + 1] - 1
            first_row = runs[first_run]["start"]
            last_row = runs[last_run]["end"]
            segment = step_rows[first_row:last_row + 1]
            instance_id = "%s:sequence:layer-%d" % (step_id, layer_id)
            pattern = patterns.get(layer_id) or {}
            for index, row in enumerate(segment):
                row["layer_id"] = layer_id
                row["layer_instance_id"] = instance_id
                row["pattern_id"] = pattern.get("pattern_id")
                row["assignment"] = "layer_body"
                row["layer_evidence"] = method
                row["layer_region"] = "layer_body"
                row["boundary_role"] = (
                    "body_start_kernel" if index == 0
                    else "end_kernel" if index == len(segment) - 1 else None)
            mapped_count += len(segment)
            cut_events.append({
                "layer_id": layer_id,
                "body_start_event": segment[0]["row_id"],
                "body_end_event": segment[-1]["row_id"],
            })
        diagnostics.append({
            "step_id": step_id,
            "status": "mapped",
            "partition_method": method,
            "configured_layer_count": layer_count,
            "module_instance_count": len(module_instances),
            "observed_stage_run_count": len(runs),
            "mapped_event_count": mapped_count,
            "transition_stability": round(stability, 6),
            "mean_pattern_similarity": round(mean_ratio, 6),
            "best_score": None if score == float("-inf") else round(score, 6),
            "second_best_score": (
                round(second_score, 6) if second_score is not None else None),
            "score_margin": (
                round(score - second_score, 6)
                if second_score is not None and score != float("-inf") else None),
            "template_evidence": {
                **template_evidence,
                **bootstrap_evidence,
            },
            "layer_boundaries": cut_events,
        })
    return diagnostics, template_evidence


def _layer_instances(rows):
    explicit = {}
    groups = []
    current = []
    current_key = None
    for row in sorted(rows, key=lambda item: item["device_seq_index"]):
        if row["assignment"] == "layer_body" and row.get("layer_instance_id"):
            explicit.setdefault(row["layer_instance_id"], []).append(row)
            if current:
                groups.append(current)
                current = []
                current_key = None
            continue
        key = (row["phase"], row["layer_id"]) if row["assignment"] == "layer_body" else None
        if key != current_key:
            if current:
                groups.append(current)
            current = []
            current_key = key
        if key is not None:
            current.append(row)
    if current:
        groups.append(current)
    groups.extend(explicit[key] for key in sorted(
        explicit, key=lambda key: min(
            row["device_seq_index"] for row in explicit[key])))
    instances = []
    occurrence = {}
    for group in groups:
        phase, layer_id = group[0]["phase"], group[0]["layer_id"]
        key = (phase, layer_id)
        occurrence[key] = occurrence.get(key, 0) + 1
        positions = [row["device_seq_index"] for row in group]
        explicit_instance = group[0].get("layer_instance_id")
        contiguous = positions == list(range(min(positions), max(positions) + 1))
        evidence_sources = sorted(set(row["layer_evidence"] for row in group))
        boundary_complete = contiguous
        duration = sum(row["duration_us"] for row in group)
        signature = [row["short_name"] for row in group]
        instances.append({
            "phase": phase,
            "layer_id": layer_id,
            "pattern_id": group[0]["pattern_id"],
            "step_id": group[0].get("step_id"),
            "layer_instance_id": explicit_instance,
            "occurrence": occurrence[key],
            "body_start_event": group[0]["row_id"],
            "body_end_event": group[-1]["row_id"],
            "first_device_seq_index": min(positions),
            "last_device_seq_index": max(positions),
            "event_count": len(group),
            "duration_us": round(duration, 6),
            "sequence_signature": hashlib.sha256(
                json.dumps(signature).encode()).hexdigest()[:16],
            "boundary_complete": boundary_complete,
            "boundary_evidence": {
                "sources": evidence_sources,
                "continuity": "contiguous" if contiguous else "interleaved_or_unresolved",
                "end_anchor_required": False,
                "end_anchor_valid": True,
                "end_kernel": group[-1]["row_id"],
                "end_stage": group[-1]["stage"],
            },
        })
    return instances


def _representatives(pattern_doc, instances):
    by_pattern = {}
    for instance in instances:
        if instance["boundary_complete"]:
            by_pattern.setdefault(instance["pattern_id"], []).append(instance)
    selected = {}
    for pattern in pattern_doc.get("patterns", []):
        pid = pattern["pattern_id"]
        values = by_pattern.get(pid, [])
        phases = sorted({item["phase"] for item in values})
        candidates = sorted(set(pattern.get("layer_ids", [])) &
                            set(item["layer_id"] for item in values))
        if phases:
            candidates = [layer_id for layer_id in candidates
                          if all(any(item["phase"] == phase and item["layer_id"] == layer_id
                                     for item in values) for phase in phases)]
        medians = {}
        layer_phase_medians = {}
        for phase in phases:
            phase_values = [item for item in values if item["phase"] == phase]
            for layer_id in candidates:
                durations = [item["duration_us"] for item in phase_values
                             if item["layer_id"] == layer_id]
                if durations:
                    layer_phase_medians[(phase, layer_id)] = statistics.median(durations)
            medians[phase] = statistics.median(
                list(layer_phase_medians[(phase, layer_id)]
                     for layer_id in candidates
                     if (phase, layer_id) in layer_phase_medians)) if candidates else 0
        scored = []
        for layer_id in candidates:
            deviations = []
            for phase in phases:
                duration = layer_phase_medians.get((phase, layer_id))
                median = medians[phase]
                if duration is not None and median:
                    deviations.append(abs(duration / median - 1.0))
            if deviations:
                scored.append((max(deviations), sum(deviations), layer_id))
        selected_layer = min(scored)[2] if scored else None
        selected_instances = {}
        if selected_layer is not None:
            for phase in phases:
                phase_instances = [item for item in values
                                   if item["phase"] == phase and item["layer_id"] == selected_layer]
                target = layer_phase_medians.get((phase, selected_layer), 0)
                if phase_instances:
                    chosen = min(phase_instances, key=lambda item: (
                        abs(item["duration_us"] - target), item["first_device_seq_index"]))
                    selected_instances[phase] = {
                        "body_start_event": chosen["body_start_event"],
                        "body_end_event": chosen["body_end_event"],
                        "first_device_seq_index": chosen["first_device_seq_index"],
                        "last_device_seq_index": chosen["last_device_seq_index"],
                        "occurrence": chosen["occurrence"],
                    }
        selected[pid] = {
            "layer_id": selected_layer,
            "phases": phases,
            "phase_median_duration_us": medians,
            "selected_instances": selected_instances,
            "selection_confidence": "high" if phases and phases != ["unresolved"] else "low",
        }
    return selected


def _table(pattern_doc, rows, representatives, table_phases=None):
    pattern_meta = {
        pattern["pattern_id"]: pattern
        for pattern in pattern_doc.get("patterns", [])
    }
    grouped = {}
    for row in rows:
        if table_phases and row["phase"] not in table_phases:
            continue
        rep = representatives.get(row["pattern_id"], {})
        selected = (rep.get("selected_instances") or {}).get(row["phase"])
        if (row["assignment"] != "layer_body" or row["layer_id"] != rep.get("layer_id")
                or not selected
                or not (selected["first_device_seq_index"] <= row["device_seq_index"]
                        <= selected["last_device_seq_index"])):
            continue
        grouped.setdefault((row["phase"], row["pattern_id"]), []).append(row)
    tables = []
    phase_order = {"prefill": 0, "decode": 1, "unresolved": 2}
    for (phase, pattern_id), group in sorted(
            grouped.items(),
            key=lambda item: (
                phase_order.get(item[0][0], 99), item[0][1])):
        total = sum(row["duration_us"] for row in group)
        output_rows = []
        for pos, row in enumerate(group):
            item = dict(row)
            item["pos"] = pos
            item["layer_total_pct"] = round(
                100.0 * row["duration_us"] / total, 6) if total else 0.0
            output_rows.append(item)
        tables.append({
            "phase": phase,
            "pattern_id": pattern_id,
            "pattern_display_name": pattern_meta.get(
                pattern_id, {}).get("pattern_display_name", pattern_id),
            "representative_layer_id": representatives[pattern_id]["layer_id"],
            "selected_step_id": group[0].get("step_id"),
            "selected_bucket": {
                "phase": phase,
                "batch_size": group[0].get("step_batch_size"),
                "input_tokens": group[0].get("step_input_tokens"),
            },
            "structural_context": pattern_meta.get(
                pattern_id, {}).get(
                    "structural_context",
                    pattern_meta.get(pattern_id, {}).get(
                        "structural_signature", {})),
            "event_count": len(group),
            "layer_total_us": round(total, 6),
            "rows": output_rows,
        })
    return tables


def _representative_integrity(rows, tables, representatives):
    """Verify only the Pattern representatives exported for downstream fusion."""
    by_position = {
        row["device_seq_index"]: row
        for row in rows
    }
    audits = []
    for table in tables:
        pattern_id = table["pattern_id"]
        phase = table["phase"]
        selected = (
            representatives.get(pattern_id, {}).get(
                "selected_instances", {}).get(phase) or {})
        first = selected.get("first_device_seq_index")
        last = selected.get("last_device_seq_index")
        expected = [
            by_position[position]
            for position in range(first, last + 1)
            if first is not None and last is not None and position in by_position
        ]
        actual = table.get("rows", [])
        expected_ids = [row["row_id"] for row in expected]
        actual_ids = [row["row_id"] for row in actual]
        duration_sum = round(sum(
            float(row.get("duration_us", 0) or 0) for row in actual), 6)
        exact_once = len(actual_ids) == len(set(actual_ids))
        ordered = [
            row["device_seq_index"] for row in actual
        ] == sorted(row["device_seq_index"] for row in actual)
        interval_complete = actual_ids == expected_ids
        duration_matches = math.isclose(
            duration_sum, float(table.get("layer_total_us", 0) or 0),
            rel_tol=0, abs_tol=1e-6)
        passed = (
            first is not None and last is not None and exact_once and ordered
            and interval_complete and duration_matches)
        audits.append({
            "pattern_id": pattern_id,
            "phase": phase,
            "representative_layer_id": table["representative_layer_id"],
            "body_start_device_seq_index": first,
            "body_end_device_seq_index": last,
            "expected_event_count": len(expected_ids),
            "actual_event_count": len(actual_ids),
            "dropped_row_ids": [
                row_id for row_id in expected_ids if row_id not in set(actual_ids)],
            "duplicate_row_ids": sorted({
                row_id for row_id in actual_ids if actual_ids.count(row_id) > 1}),
            "exact_once": exact_once,
            "ordered": ordered,
            "interval_complete": interval_complete,
            "duration_sum_us": duration_sum,
            "declared_layer_total_us": table.get("layer_total_us"),
            "duration_matches": duration_matches,
            "status": "pass" if passed else "fail",
        })
    return {
        "status": "pass" if tables and all(
            item["status"] == "pass" for item in audits) else "fail",
        "table_count": len(tables),
        "tables": audits,
    }


def _quality(
        pattern_doc, rows, instances, representatives, spans, out_of_scope,
        partition_diagnostics, tables):
    input_count = len(rows)
    assigned_count = sum(1 for row in rows if row["assignment"] in (
        "layer_body", "transition_global", "concurrent_unresolved"))
    input_duration = sum(row["duration_us"] for row in rows)
    assigned_duration = sum(row["duration_us"] for row in rows
                            if row["assignment"] in (
                                "layer_body", "transition_global", "concurrent_unresolved"))
    pattern_missing = [pid for pid, rep in representatives.items()
                       if rep["layer_id"] is None]
    incomplete = [item for item in instances if not item["boundary_complete"]]
    instances_by_step = {}
    for instance in instances:
        instances_by_step.setdefault(instance.get("step_id"), []).append(instance)
    step_audits = []
    for diagnostic in partition_diagnostics:
        if diagnostic.get("status") != "mapped":
            continue
        step_instances = sorted(
            instances_by_step.get(diagnostic["step_id"], []),
            key=lambda item: item["first_device_seq_index"])
        layer_count = diagnostic["configured_layer_count"]
        pass_count = int(diagnostic.get("mapped_pass_count", 1) or 1)
        expected_order = list(range(layer_count)) * pass_count
        actual_order = [item["layer_id"] for item in step_instances]
        non_overlapping = all(
            left["last_device_seq_index"] < right["first_device_seq_index"]
            for left, right in zip(step_instances, step_instances[1:]))
        step_audits.append({
            "step_id": diagnostic["step_id"],
            "expected_instance_count": len(expected_order),
            "actual_instance_count": len(step_instances),
            "layer_order_valid": actual_order == expected_order,
            "non_overlapping": non_overlapping,
            "status": "pass" if (
                actual_order == expected_order and non_overlapping) else "fail",
        })
    mechanical_pass = (not partition_diagnostics or bool(step_audits)) and all(
        item["status"] == "pass" for item in step_audits)
    phase_status = "pass" if spans else "partial"
    representative_integrity = _representative_integrity(
        rows, tables, representatives)
    status = "fail" if (
        pattern_missing or representative_integrity["status"] == "fail") else (
        "partial" if phase_status == "partial" or
        pattern_doc.get("quality", {}).get("status") == "partial" else "pass")
    return {
        "schema_version": 1,
        "status": status,
        "gates": {
            "pattern_coverage": pattern_doc.get("coverage_check", {}),
            "phase": {
                "status": phase_status,
                "annotation_spans": len(spans),
                "reason": "" if spans else "no measured phase annotations",
            },
            "analysis_window_conservation": {
                "status": "pass" if (
                    input_count == assigned_count and
                    math.isclose(input_duration, assigned_duration, rel_tol=0, abs_tol=1e-6)
                ) else "fail",
                "input_event_count": input_count,
                "assigned_event_count": assigned_count,
                "input_duration_us": round(input_duration, 6),
                "assigned_duration_us": round(assigned_duration, 6),
                "out_of_scope": out_of_scope,
            },
            "layer_boundaries": {
                "status": "pass" if not incomplete and not pattern_missing else "fail",
                "gating": False,
                "scope": "all_layer_instances_diagnostic",
                "incomplete_instances": len(incomplete),
                "patterns_without_representative": pattern_missing,
            },
            "step_layer_order": {
                "status": "pass" if mechanical_pass else "fail",
                "gating": False,
                "scope": "all_layer_instances_diagnostic",
                "steps": step_audits,
            },
            "representative_layer_integrity": representative_integrity,
        },
    }


def _shape_capture_plan(tables, pattern_doc, trace_path):
    needs = []
    target_layers = sorted({
        int(table["representative_layer_id"]) for table in tables
        if table.get("representative_layer_id") is not None})
    target_buckets = []
    for table in tables:
        bucket = dict(table.get("selected_bucket") or {})
        bucket.update({
            "pattern_id": table["pattern_id"],
            "representative_layer_id": table["representative_layer_id"],
            "step_id": table.get("selected_step_id"),
        })
        target_buckets.append(bucket)
        for row in table["rows"]:
            if row["shape"]["source"] == "kernel_exact":
                continue
            needs.append({
                "phase": table["phase"],
                "pattern_id": table["pattern_id"],
                "representative_layer_id": table["representative_layer_id"],
                "pos": row["pos"],
                "row_id": row["row_id"],
                "raw_event_index": row["raw_event_index"],
                "device_seq_index": row["device_seq_index"],
                "event_type": row["event_type"],
                "raw_name": row["raw_name"],
                "short_name": row["short_name"],
                "provider": row["provider"],
                "classification": row["classification"],
                "stage": row["stage"],
                "external_id": row.get("external_id"),
                "parent_operator": row["parent_operator"]["canonical_op"],
                "parent_mapping_level": row["parent_operator"]["mapping_level"],
                "parent_mapping_cardinality": row[
                    "parent_operator"].get("mapping_cardinality"),
                "parent_device_launch_count": row[
                    "parent_operator"].get("device_launch_count"),
                "candidate_op_path": None,
                "candidate_wrapper": None,
                "candidate_terminal_launcher": None,
                "mapping_cardinality": "unresolved",
                "source_evidence": [],
                "selected_bucket": table.get("selected_bucket", {}),
                "missing_fields": ["tensor roles", "input/output shapes", "dtype"],
                "current_source": row["shape"]["source"],
            })
    return {
        "schema_version": 2,
        "scope": "representative_layers_only",
        "analysis_rank": 0,
        "trace_path": os.path.abspath(trace_path),
        "trace_sha256": _sha(trace_path),
        "representative_layer_filter": target_layers,
        "target_buckets": target_buckets,
        "patterns": [{
            "pattern_id": table["pattern_id"],
            "representative_layer_id": table["representative_layer_id"],
            "structural_context": table.get("structural_context", {}),
        } for table in tables],
        "capture_policy": {
            "rank": 0,
            "max_matched_forwards_per_bucket": 1,
            "metadata_only": True,
            "stdout": False,
            "unresolved_targets_only": True,
            "decode_capture_windows": ["graph_capture", "warmup", "enforce_eager_probe"],
        },
        "capture_targets": needs,
        "target_count": len(needs),
    }


def _markdown(tables, quality):
    lines = ["# Ordered Unique Layer Kernel Tables", "",
             "Semantic mapping status: `%s`." % quality["status"], ""]
    for table in tables:
        lines.extend([
            "## %s — %s" % (table["phase"].upper(), table["pattern_display_name"]),
            "",
            "- representative layer: `L%s`" % table["representative_layer_id"],
            "- selected bucket: `%s`" % json.dumps(
                table.get("selected_bucket", {}), sort_keys=True),
            "- complete-layer device event count: `%s`" % table["event_count"],
            "- raw one-layer device event total us: `%.3f`" % table["layer_total_us"],
            "- structural context: `%s`" % json.dumps(
                table.get("structural_context", {}), sort_keys=True),
            "",
            "| pos | stage | kernel | parent operator | shape source | duration us | layer total % |",
            "|---:|---|---|---|---|---:|---:|",
        ])
        for row in table["rows"]:
            lines.append("| %d | %s | `%s` | %s | %s | %.3f | %.3f |" % (
                row["pos"], row["stage"], row["short_name"],
                row["parent_operator"]["canonical_op"].replace("|", "\\|"),
                row["shape"]["source"], row["duration_us"], row["layer_total_pct"]))
        lines.append("")
    return "\n".join(lines) + "\n"


def build(trace_path, pattern_path, out_dir, table_phases=None):
    with open(pattern_path) as fh:
        pattern_doc = json.load(fh)
    events = _load_events(trace_path)
    patterns = _pattern_index(pattern_doc)
    rows, spans, out_of_scope, module_scopes, module_diagnostics = _event_rows(
        events, pattern_doc)
    module_interpolated = _complete_module_ranges(rows, module_scopes, patterns)
    partition_diagnostics, pattern_templates = _stage_sequence_partition(
        rows, pattern_doc)
    instances = _layer_instances(rows)
    representative_instances = [
        instance for instance in instances
        if not table_phases or instance["phase"] in table_phases]
    representatives = _representatives(pattern_doc, representative_instances)
    tables = _table(pattern_doc, rows, representatives, table_phases)
    quality = _quality(
        pattern_doc, rows, instances, representatives, spans, out_of_scope,
        partition_diagnostics, tables)
    capture_plan = _shape_capture_plan(tables, pattern_doc, trace_path)
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "semantic_event_audit_jsonl": os.path.join(out_dir, "semantic_event_audit.jsonl"),
        "layer_instance_audit_json": os.path.join(out_dir, "layer_instance_audit.json"),
        "semantic_table_json": os.path.join(out_dir, "pattern_layer_kernel_table.json"),
        "semantic_table_md": os.path.join(out_dir, "ORDERED_UNIQUE_LAYER_TABLES.md"),
        "shape_capture_plan_json": os.path.join(out_dir, "SHAPE_CAPTURE_PLAN.json"),
        "quality_json": os.path.join(out_dir, "semantic_mapping_quality.json"),
    }
    with open(paths["semantic_event_audit_jsonl"], "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    docs = (
        (paths["layer_instance_audit_json"], {
            "schema_version": 1, "trace_sha256": _sha(trace_path),
            "module_scope_diagnostics": module_diagnostics,
            "module_scope_count": len(module_scopes),
            "module_interpolated_event_count": module_interpolated,
            "boundary_partition_diagnostics": partition_diagnostics,
            "pattern_stage_templates": pattern_templates,
            "instances": instances, "representatives": representatives}),
        (paths["semantic_table_json"], {
            "schema_version": 2, "trace_path": os.path.abspath(trace_path),
            "trace_sha256": _sha(trace_path), "patterns_path": os.path.abspath(pattern_path),
            "table_phases": sorted(table_phases) if table_phases else ["all"],
            "tables": tables}),
        (paths["shape_capture_plan_json"], capture_plan),
        (paths["quality_json"], quality),
    )
    for path, doc in docs:
        with open(path, "w") as fh:
            json.dump(doc, fh, indent=2)
    with open(paths["semantic_table_md"], "w") as fh:
        fh.write(_markdown(tables, quality))
    return {"schema_version": 1, "status": quality["status"], **paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result-json", default="")
    parser.add_argument("--table-phases", default="all",
                        help="comma-separated representative-table phases; default all, ordered prefill then decode")
    args = parser.parse_args()
    requested_phases = set(
        value.strip() for value in args.table_phases.split(",") if value.strip())
    table_phases = None if "all" in requested_phases else requested_phases or None
    result = build(args.trace, args.patterns, args.out_dir, table_phases)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0 if result["status"] != "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
