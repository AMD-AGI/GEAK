#!/usr/bin/env python3
"""Validate per-layer body descriptors and derive structural Patterns.

The semantics Agent interprets arbitrary model config/runtime source and emits
one descriptor per main decoder layer. It does not decide Pattern identity.
This validator groups identical canonical ``body_signature`` objects and keeps
position/cross-layer effects in ``instance_context`` so first/last-layer
epilogues cannot manufacture singleton Patterns.
"""
import argparse
import copy
import hashlib
import json
import os
import re


_CONTEXT_ONLY_KEYS = {
    "boundary_dispatch",
    "entry_handoff",
    "exit_handoff",
    "is_first_layer",
    "is_first_main_layer",
    "is_last_layer",
    "is_last_main_layer",
    "model_entry",
    "model_epilogue",
    "model_exit",
    "post_layer_region",
    "pre_layer_region",
    "special_layer_role",
}


def _reject_trace_derived_definition(value, path="definition"):
    forbidden_keys = (
        "trace_evidence",
        "trace_signature",
        "kernel_sequence",
        "kernel_cluster",
        "kernel_names",
        "duration_signature",
        "timing_signature",
    )
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in forbidden_keys):
                raise ValueError(
                    "Trace/kernel-derived Pattern definition is forbidden: "
                    "%s.%s" % (path, key))
            if (lowered in ("source", "evidence_source")
                    and str(item).lower() in (
                        "trace", "profiler_trace", "kernel_trace")):
                raise ValueError(
                    "Trace may validate but not define Pattern structure")
            _reject_trace_derived_definition(item, "%s.%s" % (path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_trace_derived_definition(item, "%s[%d]" % (path, index))


def _reject_context_in_body(value, path="body_signature"):
    """Keep positional/cross-layer decoration out of Pattern identity."""
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).strip().lower()
            if lowered in _CONTEXT_ONLY_KEYS:
                raise ValueError(
                    "%s.%s is instance context and may not define a Pattern"
                    % (path, key))
            _reject_context_in_body(item, "%s.%s" % (path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_context_in_body(item, "%s[%d]" % (path, index))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_config(config):
    value = config.get("text_config")
    return value if isinstance(value, dict) else config


_CONFIG_PATH_PART = re.compile(r"([^\[\]]+)|\[(\d+)\]")


def _config_value(config, dotted_path):
    """Resolve dotted config paths, including optional ``field[3]`` indices."""
    value = config
    for component in str(dotted_path).split("."):
        parts = list(_CONFIG_PATH_PART.finditer(component))
        if not parts:
            raise ValueError(
                "invalid config evidence path: %s" % dotted_path)
        for match in parts:
            key, index = match.groups()
            if key is not None:
                if not isinstance(value, dict) or key not in value:
                    raise ValueError(
                        "config evidence path does not exist: %s"
                        % dotted_path)
                value = value[key]
            else:
                position = int(index)
                if (not isinstance(value, list)
                        or position < 0 or position >= len(value)):
                    raise ValueError(
                        "config evidence index does not exist: %s"
                        % dotted_path)
                value = value[position]
    return value


def _signature_hash(signature):
    payload = json.dumps(
        signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _validate_config_evidence(owner, config):
    evidence = owner.get("config_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "layer %s requires non-empty config_evidence list"
            % owner.get("layer_id"))
    for item in evidence:
        path = item.get("config_path")
        if not path or "value" not in item or not item.get("claim"):
            raise ValueError("config evidence requires path, value, and claim")
        actual = _config_value(config, path)
        if actual != item["value"]:
            raise ValueError(
                "config evidence mismatch at %s" % path)


def _validate_source_evidence(owner, source_by_path):
    evidence = owner.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "layer %s requires runtime source evidence"
            % owner.get("layer_id"))
    for item in evidence:
        path = os.path.abspath(str(item.get("path") or ""))
        if path not in source_by_path:
            raise ValueError("unapproved runtime source evidence: %s" % path)
        start = int(item.get("line_start", 0) or 0)
        end = int(item.get("line_end", 0) or 0)
        if start <= 0 or end < start:
            raise ValueError("invalid source evidence line range")
        if end > source_by_path[path]["line_count"]:
            raise ValueError("source evidence line range exceeds file")
        if not item.get("symbol") or not item.get("claim"):
            raise ValueError("source evidence requires symbol and claim")
        item["path"] = path
        item["sha256"] = source_by_path[path]["sha256"]


def _is_contextual_representative(layer):
    context = layer.get("instance_context") or {}
    return any(bool(context.get(key)) for key in (
        "is_first_layer", "is_first_main_layer", "is_last_layer",
        "is_last_main_layer", "model_entry", "model_exit",
        "model_epilogue"))


def _semantic_label(signature, name):
    labels = signature.get("semantic_labels") or {}
    value = labels.get(name)
    return value if value is not None else signature.get(name)


def validate(pattern_path, config_path, runtime_sources, out_path=""):
    with open(pattern_path) as fh:
        draft = json.load(fh)
    with open(config_path) as fh:
        config = json.load(fh)

    definition = draft.get("pattern_definition") or {}
    if definition.get("producer") != "semantics_mapper_agent":
        raise ValueError(
            "layer descriptors must be produced by semantics_mapper_agent")
    if definition.get("method") != "config_runtime_body_analysis":
        raise ValueError(
            "unsupported Agent definition method; expected "
            "config_runtime_body_analysis")
    if definition.get("trace_used_for_definition") is not False:
        raise ValueError("Trace may not define structural Patterns")
    if not definition.get("analysis_summary"):
        raise ValueError("Agent analysis_summary is required")

    source_by_path = {}
    for path in runtime_sources or []:
        absolute = os.path.abspath(path)
        if not os.path.isfile(absolute):
            raise ValueError("runtime source does not exist: %s" % absolute)
        with open(absolute) as fh:
            line_count = sum(1 for _ in fh)
        source_by_path[absolute] = {
            "path": absolute,
            "sha256": _sha256(absolute),
            "line_count": line_count,
        }
    if not source_by_path:
        raise ValueError(
            "Agent structural analysis requires current runtime source")

    text_config = _text_config(config)
    count = int(text_config.get("num_hidden_layers", 0) or 0)
    if count <= 0:
        raise ValueError("config has no positive num_hidden_layers")
    scope = draft.get("main_layer_scope") or {}
    declared_count = int(scope.get("num_hidden_layers", 0) or 0)
    if declared_count != count:
        raise ValueError(
            "main_layer_scope.num_hidden_layers=%d does not match config=%d"
            % (declared_count, count))

    layers = draft.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(
            "Agent must emit one body descriptor per main layer")

    by_id = {}
    grouped = {}
    for raw_layer in layers:
        layer = copy.deepcopy(raw_layer)
        layer_id = layer.get("layer_id")
        if not isinstance(layer_id, int):
            raise ValueError("layer_id must be an integer")
        if layer_id < 0 or layer_id >= count:
            raise ValueError("layer_id outside main model: %s" % layer_id)
        if layer_id in by_id:
            raise ValueError("duplicate layer_id: %s" % layer_id)
        signature = layer.get("body_signature")
        if not isinstance(signature, dict) or not signature:
            raise ValueError(
                "layer %d requires non-empty body_signature" % layer_id)
        context = layer.get("instance_context", {})
        if not isinstance(context, dict):
            raise ValueError(
                "layer %d instance_context must be an object" % layer_id)
        _reject_trace_derived_definition(layer, "layers[%d]" % layer_id)
        _reject_context_in_body(signature)
        _validate_config_evidence(layer, config)
        _validate_source_evidence(layer, source_by_path)
        signature_hash = _signature_hash(signature)
        layer["body_signature_hash"] = signature_hash
        layer["instance_context"] = context
        by_id[layer_id] = layer
        grouped.setdefault(signature_hash, []).append(layer)

    covered = sorted(by_id)
    if covered != list(range(count)):
        missing = sorted(set(range(count)) - set(covered))
        raise ValueError(
            "Agent layer descriptors must cover every main layer exactly once; "
            "missing=%s" % missing)

    groups = sorted(grouped.values(), key=lambda values: min(
        layer["layer_id"] for layer in values))
    patterns = []
    for pattern_index, values in enumerate(groups):
        values.sort(key=lambda layer: layer["layer_id"])
        signature = values[0]["body_signature"]
        display_names = sorted(set(
            str(layer.get("body_display_name") or "").strip()
            for layer in values))
        non_empty_names = [name for name in display_names if name]
        if len(non_empty_names) > 1:
            raise ValueError(
                "identical body_signature has inconsistent display names: %s"
                % non_empty_names)
        layer_ids = [layer["layer_id"] for layer in values]
        ordinary = [
            layer["layer_id"] for layer in values
            if not _is_contextual_representative(layer)]
        # Prefer layers without model-entry/model-exit context.  Do not remove
        # an edge-only Pattern, though: when its sole implementation is the
        # first or final layer, that contextual layer is the required fallback.
        # Kernel mapping still keeps outer global prefix/suffix rows outside
        # the representative layer body.
        candidates = ordinary if ordinary else layer_ids
        config_evidence = []
        source_evidence = []
        seen_config = set()
        seen_source = set()
        for layer in values:
            for item in layer["config_evidence"]:
                key = json.dumps(item, sort_keys=True)
                if key not in seen_config:
                    seen_config.add(key)
                    config_evidence.append(item)
            for item in layer["source_evidence"]:
                key = json.dumps(item, sort_keys=True)
                if key not in seen_source:
                    seen_source.add(key)
                    source_evidence.append(item)
        pattern = {
            "pattern_id": "P%d" % pattern_index,
            "pattern_display_name": (
                non_empty_names[0] if non_empty_names
                else "Body pattern %d" % pattern_index),
            "body_signature": signature,
            # Compatibility alias for existing table/report readers. Pattern
            # identity is nevertheless defined only by body_signature.
            "structural_signature": signature,
            "signature_hash": values[0]["body_signature_hash"],
            "layer_ids": layer_ids,
            "layer_count": len(layer_ids),
            "representative_candidates": candidates,
            "config_evidence": config_evidence,
            "source_evidence": source_evidence,
            "structural_context": {
                "body_signature": signature,
                "shape_parameters": signature.get("shape_parameters", {}),
                "semantic_labels": signature.get("semantic_labels", {}),
            },
        }
        attention_type = _semantic_label(signature, "attention_type")
        ffn_type = _semantic_label(signature, "ffn_type")
        if attention_type is not None:
            pattern["attention_type"] = attention_type
        if ffn_type is not None:
            pattern["ffn_type"] = ffn_type
        patterns.append(pattern)

    result = copy.deepcopy(draft)
    result["schema_version"] = 3
    result["config_path"] = os.path.abspath(config_path)
    result["config_sha256"] = _sha256(config_path)
    result["model_type"] = config.get("model_type")
    result["num_hidden_layers_main"] = count
    result["layers"] = [by_id[layer_id] for layer_id in range(count)]
    result["layer_contexts"] = {
        str(layer_id): by_id[layer_id].get("instance_context", {})
        for layer_id in range(count)
    }
    result["patterns"] = patterns
    result["coverage_check"] = {
        "total_main_layers": count,
        "covered": len(covered),
        "mutually_exclusive": True,
        "full_coverage": True,
    }
    result["quality"] = {
        "status": "pass",
        "confidence": "high",
        "reason": (
            "Per-layer body descriptors passed deterministic evidence, "
            "coverage, context-separation, and canonical grouping checks"),
    }
    result["validation"] = {
        "validator": "validate_structural_patterns.py",
        "agent_layer_descriptors_preserved": True,
        "definition_preserved": True,
        "patterns_derived_deterministically": True,
        "checks": [
            "agent_provenance",
            "trace_not_used_for_definition",
            "config_evidence_values",
            "runtime_source_hashes_and_line_ranges",
            "main_layer_exact_coverage",
            "context_excluded_from_body_identity",
            "canonical_body_signature_grouping",
            "contextual_representatives_deprioritized",
        ],
        "runtime_sources": list(source_by_path.values()),
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-source", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    validate(args.input, args.config, args.runtime_source, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
