#!/usr/bin/env python3
"""Validate Agent-defined structural Layer Patterns without defining them."""
import argparse
import copy
import hashlib
import json
import os


REQUIRED_SIGNATURE_FIELDS = (
    "attention_type",
    "model_native_attention_name",
    "attention_config_fields",
    "runtime_attention_module_class",
    "ffn_type",
    "is_moe",
    "num_experts",
    "topk",
    "shared_expert",
    "router_family",
    "special_layer_role",
    "runtime_dispatch_branch",
)


def _reject_trace_derived_definition(value, path="pattern"):
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
            _reject_trace_derived_definition(
                item, "%s.%s" % (path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_trace_derived_definition(
                item, "%s[%d]" % (path, index))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_config(config):
    value = config.get("text_config")
    return value if isinstance(value, dict) else config


def _config_value(config, dotted_path):
    value = config
    for part in str(dotted_path).split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(
                "config evidence path does not exist: %s" % dotted_path)
        value = value[part]
    return value


def _signature_hash(signature):
    payload = json.dumps(
        signature, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _validate_config_evidence(pattern, config):
    evidence = pattern.get("config_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "%s requires non-empty config_evidence list"
            % pattern.get("pattern_id"))
    for item in evidence:
        path = item.get("config_path")
        if not path or "value" not in item or not item.get("claim"):
            raise ValueError("config evidence requires path, value, and claim")
        actual = _config_value(config, path)
        if actual != item["value"]:
            raise ValueError(
                "config evidence mismatch at %s" % path)


def _validate_source_evidence(pattern, source_by_path):
    evidence = pattern.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "%s requires runtime source evidence"
            % pattern.get("pattern_id"))
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


def validate(pattern_path, config_path, runtime_sources, out_path=""):
    with open(pattern_path) as fh:
        draft = json.load(fh)
    with open(config_path) as fh:
        config = json.load(fh)
    definition = draft.get("pattern_definition") or {}
    if definition.get("producer") != "semantics_mapper_agent":
        raise ValueError(
            "patterns must be produced by semantics_mapper_agent")
    if definition.get("method") != "config_runtime_source_analysis":
        raise ValueError("unsupported Agent pattern definition method")
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

    count = int(_text_config(config).get("num_hidden_layers", 0) or 0)
    if count <= 0:
        raise ValueError("config has no positive num_hidden_layers")
    patterns = draft.get("patterns")
    if not isinstance(patterns, list) or not patterns:
        raise ValueError("Agent must define at least one Pattern")

    pattern_ids = set()
    signature_hashes = set()
    covered = []
    for pattern in patterns:
        pattern_id = str(pattern.get("pattern_id") or "")
        if not pattern_id or pattern_id in pattern_ids:
            raise ValueError("pattern_id must be non-empty and unique")
        pattern_ids.add(pattern_id)
        _reject_trace_derived_definition(pattern)
        if not pattern.get("pattern_display_name"):
            raise ValueError("%s requires pattern_display_name" % pattern_id)
        signature = pattern.get("structural_signature")
        if not isinstance(signature, dict):
            raise ValueError("%s requires structural_signature" % pattern_id)
        missing = [
            key for key in REQUIRED_SIGNATURE_FIELDS
            if key not in signature]
        if missing:
            raise ValueError(
                "%s missing signature fields: %s"
                % (pattern_id, ", ".join(missing)))
        if not isinstance(signature["attention_config_fields"], dict):
            raise ValueError("attention_config_fields must be an object")
        signature_hash = _signature_hash(signature)
        if signature_hash in signature_hashes:
            raise ValueError(
                "identical structural signatures must be merged")
        signature_hashes.add(signature_hash)
        pattern["signature_hash"] = signature_hash
        layer_ids = pattern.get("layer_ids")
        if (not isinstance(layer_ids, list)
                or any(not isinstance(value, int) for value in layer_ids)):
            raise ValueError("%s layer_ids must be integers" % pattern_id)
        if layer_ids != sorted(set(layer_ids)):
            raise ValueError("%s layer_ids must be sorted and unique" % pattern_id)
        if any(value < 0 or value >= count for value in layer_ids):
            raise ValueError("%s layer_id outside main model" % pattern_id)
        if not layer_ids:
            raise ValueError("%s has no layers" % pattern_id)
        if pattern.get("representative_candidates") != layer_ids:
            raise ValueError(
                "%s representative_candidates must equal layer_ids"
                % pattern_id)
        pattern["layer_count"] = len(layer_ids)
        pattern["attention_type"] = signature["attention_type"]
        pattern["ffn_type"] = signature["ffn_type"]
        _validate_config_evidence(pattern, config)
        _validate_source_evidence(pattern, source_by_path)
        covered.extend(layer_ids)

    mutually_exclusive = len(covered) == len(set(covered))
    full_coverage = sorted(covered) == list(range(count))
    if not mutually_exclusive or not full_coverage:
        raise ValueError(
            "Agent Patterns must cover every main layer exactly once")

    result = copy.deepcopy(draft)
    result["schema_version"] = 2
    result["config_path"] = os.path.abspath(config_path)
    result["config_sha256"] = _sha256(config_path)
    result["model_type"] = config.get("model_type")
    result["num_hidden_layers_main"] = count
    result["patterns"] = patterns
    result["coverage_check"] = {
        "total_main_layers": count,
        "covered": len(covered),
        "mutually_exclusive": mutually_exclusive,
        "full_coverage": full_coverage,
    }
    result["quality"] = {
        "status": "pass",
        "confidence": "high",
        "reason": "Agent definition passed deterministic evidence and coverage validation",
    }
    result["validation"] = {
        "validator": "validate_structural_patterns.py",
        "definition_preserved": True,
        "checks": [
            "agent_provenance",
            "trace_not_used_for_definition",
            "no_trace_or_kernel_derived_pattern_evidence",
            "config_evidence_values",
            "runtime_source_hashes_and_line_ranges",
            "required_structural_signature",
            "identical_signatures_merged",
            "layer_ids_mutually_exclusive",
            "all_main_layers_covered",
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
    validate(
        args.input, args.config, args.runtime_source, args.out)
    print(args.out)


if __name__ == "__main__":
    main()
