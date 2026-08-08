#!/usr/bin/env python3
"""Resolve model structural layer patterns from config, with optional source evidence."""
import argparse
import hashlib
import json
import os


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pattern(pattern_id, display, signature, layer_ids, config_evidence,
             source_evidence):
    return {
        "pattern_id": pattern_id,
        "pattern_display_name": display,
        "structural_signature": signature,
        "signature_hash": _hash(signature),
        "layer_ids": layer_ids,
        "layer_count": len(layer_ids),
        "attention_type": signature.get("attention_type"),
        "ffn_type": signature.get("ffn_type"),
        "config_evidence": config_evidence,
        "source_evidence": source_evidence,
        "representative_candidates": list(layer_ids),
    }


ATTENTION_FIELDS = (
    "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim",
    "qk_rope_head_dim", "v_head_dim", "num_attention_heads",
    "num_key_value_heads", "head_dim", "full_attention_interval",
    "linear_num_key_heads", "linear_key_head_dim",
    "linear_num_value_heads", "linear_value_head_dim",
    "linear_conv_kernel_dim",
)

ROUTER_FIELDS = (
    "n_routed_experts", "num_experts", "num_experts_per_tok",
    "n_shared_experts", "shared_expert_intermediate_size", "topk_method",
    "scoring_func", "n_group", "topk_group", "norm_topk_prob",
)


def _text_config(cfg):
    text = cfg.get("text_config")
    return text if isinstance(text, dict) and "num_hidden_layers" in text else cfg


def _present(cfg, fields):
    return {key: cfg[key] for key in fields if cfg.get(key) is not None}


def _first(cfg, fields):
    for key in fields:
        if cfg.get(key) is not None:
            return cfg[key]
    return None


def _slug(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).upper()


def _attention_type(text, declared=None):
    if declared is not None:
        return str(declared)
    if any(text.get(key) is not None for key in (
            "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim",
            "qk_rope_head_dim")):
        return "MLA"
    return str(text.get("attention_type") or "homogeneous_attention")


def _is_moe_config(text):
    return _first(text, ("n_routed_experts", "num_experts")) is not None


def _attention_display(attention_type, fields):
    lowered = attention_type.lower()
    if lowered == "mla":
        return "MLA(q=%s,kv=%s,nope=%s,rope=%s)" % (
            fields.get("q_lora_rank"), fields.get("kv_lora_rank"),
            fields.get("qk_nope_head_dim"), fields.get("qk_rope_head_dim"))
    if "linear" in lowered:
        return "%s(kh=%s,kd=%s,vh=%s,vd=%s,conv=%s)" % (
            attention_type, fields.get("linear_num_key_heads"),
            fields.get("linear_key_head_dim"),
            fields.get("linear_num_value_heads"),
            fields.get("linear_value_head_dim"),
            fields.get("linear_conv_kernel_dim"))
    if "full" in lowered:
        return "%s(h=%s,kv=%s,d=%s)" % (
            attention_type, fields.get("num_attention_heads"),
            fields.get("num_key_value_heads"), fields.get("head_dim"))
    return attention_type


def _ffn_display(text, ffn_type):
    if ffn_type == "dense":
        return "DenseFFN(i=%s)" % text.get("intermediate_size")
    experts = _first(text, ("n_routed_experts", "num_experts"))
    topk = _first(text, ("num_experts_per_tok", "num_experts_per_token"))
    shared = _first(text, ("n_shared_experts", "shared_expert_intermediate_size"))
    return "MoE(%se,top%s,shared=%s)" % (experts, topk, shared or 0)


def _make_pattern(text, attention_type, ffn_type, layer_ids, dialect,
                  source_evidence, dialect_evidence):
    attention_fields = _present(text, ATTENTION_FIELDS)
    router = _present(text, ROUTER_FIELDS) if ffn_type == "moe" else {}
    signature = {
        "attention_type": attention_type,
        "model_native_attention_name": attention_type,
        "attention_config_fields": attention_fields,
        "ffn_type": ffn_type,
        "is_moe": ffn_type == "moe",
        "num_experts": (
            _first(text, ("n_routed_experts", "num_experts"))
            if ffn_type == "moe" else None),
        "topk": (
            _first(text, ("num_experts_per_tok", "num_experts_per_token"))
            if ffn_type == "moe" else None),
        "shared_expert": (
            bool(_first(
                text, ("n_shared_experts", "shared_expert_intermediate_size")))
            if ffn_type == "moe" else False),
        "router_family": router,
        "runtime_dispatch_branch": "config_dialect.%s" % dialect,
    }
    display = "%s / %s" % (
        _attention_display(attention_type, attention_fields),
        _ffn_display(text, ffn_type))
    pattern_id = "P_%s" % _slug(attention_type)
    if dialect == "dense_moe_formula":
        pattern_id += "_%s" % ffn_type.upper()
    evidence = {
        "config_dialect": dialect,
        **dialect_evidence,
        **attention_fields,
    }
    if ffn_type == "moe":
        evidence.update(router)
        if text.get("moe_intermediate_size") is not None:
            evidence["moe_intermediate_size"] = text["moe_intermediate_size"]
    elif text.get("intermediate_size") is not None:
        evidence["intermediate_size"] = text["intermediate_size"]
    return _pattern(
        pattern_id, display, signature, layer_ids, evidence, source_evidence)


def _by_explicit_layer_types(text, count, source_evidence):
    layer_types = text.get("layer_types")
    if len(layer_types) != count:
        raise ValueError("layer_types must describe every main layer")
    groups = {}
    for layer_id, layer_type in enumerate(layer_types):
        groups.setdefault(str(layer_type), []).append(layer_id)
    ffn_type = "moe" if _is_moe_config(text) else "dense"
    return [
        _make_pattern(
            text, _attention_type(text, layer_type), ffn_type, layer_ids,
            "per_layer_list", source_evidence, {"layer_type": layer_type})
        for layer_type, layer_ids in groups.items()
    ]


def _by_dense_moe_formula(text, count, source_evidence):
    first_dense = int(text.get("first_k_dense_replace", count))
    moe_freq = int(text.get("moe_layer_freq", 1) or 1)
    if moe_freq <= 0:
        raise ValueError("moe_layer_freq must be positive")
    layer_groups = {"dense": [], "moe": []}
    for layer_id in range(count):
        ffn_type = (
            "moe" if layer_id >= first_dense and layer_id % moe_freq == 0
            else "dense")
        layer_groups[ffn_type].append(layer_id)
    attention_type = _attention_type(text)
    evidence = {
        "first_k_dense_replace": first_dense,
        "moe_layer_freq": moe_freq,
    }
    return [
        _make_pattern(
            text, attention_type, ffn_type, layer_ids, "dense_moe_formula",
            source_evidence, evidence)
        for ffn_type, layer_ids in layer_groups.items() if layer_ids
    ]


def _homogeneous(text, count, source_evidence):
    ffn_type = "moe" if _is_moe_config(text) else "dense"
    return [_make_pattern(
        text, _attention_type(text), ffn_type, list(range(count)),
        "homogeneous", source_evidence, {"num_hidden_layers": count})]


def _patterns_from_config_dialect(cfg, source_evidence):
    text = _text_config(cfg)
    if text.get("num_hidden_layers") is None:
        raise ValueError("config has no num_hidden_layers")
    count = int(text["num_hidden_layers"])
    layer_types = text.get("layer_types")
    if layer_types is not None:
        if not isinstance(layer_types, list):
            raise ValueError("layer_types must be a list")
        dialect = "per_layer_list"
        patterns = _by_explicit_layer_types(text, count, source_evidence)
    elif "first_k_dense_replace" in text or "moe_layer_freq" in text:
        dialect = "dense_moe_formula"
        patterns = _by_dense_moe_formula(text, count, source_evidence)
    else:
        dialect = "homogeneous"
        patterns = _homogeneous(text, count, source_evidence)
    excluded = {
        key: text[key] for key in (
            "num_nextn_predict_layers", "mtp_num_hidden_layers")
        if text.get(key) is not None
    }
    if excluded:
        excluded["reason"] = "excluded unless speculative decoding is active"
    return count, patterns, excluded, dialect


def build(config_path, config_key="", runtime_sources=None):
    with open(config_path) as fh:
        cfg = json.load(fh)
    sources = []
    for path in runtime_sources or []:
        if os.path.isfile(path):
            sources.append({"path": os.path.abspath(path), "sha256": _file_sha(path)})
    source_evidence = {
        "status": "available" if sources else "unavailable",
        "files": sources,
    }
    model_type = cfg.get("model_type")
    count, patterns, excluded, dialect = _patterns_from_config_dialect(
        cfg, source_evidence)
    covered = [layer_id for pattern in patterns for layer_id in pattern["layer_ids"]]
    mutually_exclusive = len(covered) == len(set(covered))
    full_coverage = sorted(covered) == list(range(count))
    if not mutually_exclusive or not full_coverage:
        raise ValueError("structural patterns must cover each main layer exactly once")
    confidence = "high" if sources else "medium"
    return {
        "schema_version": 1,
        "config_key": config_key,
        "model_type": model_type,
        "config_path": os.path.abspath(config_path),
        "config_sha256": _file_sha(config_path),
        "pattern_dialect": dialect,
        "num_hidden_layers_main": count,
        "patterns": patterns,
        "coverage_check": {
            "total_main_layers": count,
            "covered": len(covered),
            "mutually_exclusive": mutually_exclusive,
            "full_coverage": full_coverage,
        },
        "excluded_layers_note": excluded,
        "quality": {
            "status": "pass" if confidence == "high" else "partial",
            "confidence": confidence,
            "reason": "" if sources else "runtime source unavailable; config-only patterns",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-key", default="")
    parser.add_argument("--runtime-source", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    doc = build(args.config, args.config_key, args.runtime_source)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(args.out)


if __name__ == "__main__":
    main()
