#!/usr/bin/env python3
"""Resolve model structural layer patterns from config, with optional source evidence."""
import argparse
import hashlib
import json
import os
import re


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

# Canonical Semantics Mapping 1.2 fields.  Aliases are intentionally about
# config dialects, never model names.
STATIC_FIELD_SPECS = {
    "common": {
        "hidden_size": ("hidden_size", "d_model", "n_embd"),
        "num_hidden_layers": ("num_hidden_layers", "n_layer", "num_layers"),
        "vocab_size": ("vocab_size", "n_vocab"),
        "max_position_embeddings": (
            "max_position_embeddings", "max_seq_len", "seq_length"),
        "model_dtype": ("torch_dtype", "dtype"),
        "norm_type": ("norm_type", "normalization_type"),
        "norm_epsilon": (
            "rms_norm_eps", "layer_norm_eps", "norm_epsilon", "norm_eps"),
        "hidden_activation": ("hidden_act", "activation_function"),
        "tie_word_embeddings": ("tie_word_embeddings",),
    },
    "full_attention": {
        "num_attention_heads": (
            "num_attention_heads", "n_head", "n_heads", "attention_heads"),
        "num_key_value_heads": (
            "num_key_value_heads", "n_kv_heads", "num_kv_heads"),
        "head_dim": ("head_dim", "attention_head_dim"),
        "rope_theta": ("rope_theta", "rotary_emb_base"),
        "rope_scaling": ("rope_scaling",),
        "rope_section": ("mrope_section", "rope_section"),
        "partial_rotary_factor": ("partial_rotary_factor",),
        "attention_bias": ("attention_bias", "use_attention_bias"),
        "attention_output_gate": (
            "attention_output_gate", "use_attention_output_gate"),
    },
    "mla": {
        "q_lora_rank": ("q_lora_rank",),
        "kv_lora_rank": ("kv_lora_rank",),
        "qk_nope_head_dim": ("qk_nope_head_dim",),
        "qk_rope_head_dim": ("qk_rope_head_dim",),
        "v_head_dim": ("v_head_dim",),
    },
    "linear_attention": {
        "key_heads": ("linear_num_key_heads", "num_linear_key_heads"),
        "key_head_dim": ("linear_key_head_dim",),
        "value_heads": ("linear_num_value_heads",),
        "value_head_dim": ("linear_value_head_dim",),
        "conv_kernel_dim": (
            "linear_conv_kernel_dim", "linear_conv_kernel_size"),
        "state_dtype": ("linear_state_dtype", "ssm_state_dtype"),
        "chunk_size": ("linear_chunk_size",),
        "state_layout": ("linear_state_layout", "ssm_state_layout"),
    },
    "dense_ffn": {
        "intermediate_size": (
            "intermediate_size", "ffn_hidden_size", "n_inner"),
        "activation": ("hidden_act", "activation_function"),
        "gated": ("is_gated_act", "gated_ffn"),
    },
    "moe": {
        "num_experts": ("n_routed_experts", "num_experts", "num_local_experts"),
        "experts_per_token": (
            "num_experts_per_tok", "num_experts_per_token", "top_k"),
        "moe_intermediate_size": (
            "moe_intermediate_size", "expert_intermediate_size"),
        "shared_expert_intermediate_size": (
            "shared_expert_intermediate_size",),
        "num_shared_experts": ("n_shared_experts", "num_shared_experts"),
        "router_scoring": ("scoring_func", "router_scoring_function"),
        "topk_method": ("topk_method",),
        "normalize_topk": ("norm_topk_prob", "normalize_topk_prob"),
        "expert_groups": ("n_group", "num_expert_groups"),
        "topk_group": ("topk_group",),
    },
    "quantization": {
        "quant_method": ("quant_method", "quantization_method"),
        "weight_dtype": ("weight_dtype", "weight_quant_dtype"),
        "activation_dtype": ("activation_dtype", "activation_quant_dtype"),
        "weight_block_size": ("weight_block_size", "block_size"),
        "group_size": ("group_size",),
        "quant_scheme": ("quant_scheme", "scheme"),
        "dynamic": ("dynamic", "is_dynamic"),
        "activation_scheme": ("activation_scheme",),
        "modules_to_not_convert": (
            "modules_to_not_convert", "ignored_modules", "exclude_modules"),
    },
    "parallelism": {
        "pretraining_tp": ("pretraining_tp",),
        "sequence_parallel": ("sequence_parallel",),
        "expert_tensor_parallel_size": ("expert_tensor_parallel_size",),
    },
}

RUNTIME_FIELD_SPECS = {
    "tensor_parallel_size": (
        "tensor_parallel_size", "tensor_parallel_degree", "tp_size", "tp-size"),
    "expert_parallel_size": (
        "expert_parallel_size", "expert_parallel_degree", "ep_size", "ep-size"),
    "pipeline_parallel_size": (
        "pipeline_parallel_size", "pipeline_parallel_degree", "pp_size", "pp-size"),
    "data_parallel_size": (
        "data_parallel_size", "data_parallel_degree", "dp_size", "dp-size"),
    "kv_cache_dtype": ("kv_cache_dtype",),
    "page_size": ("page_size", "block_size"),
    "chunk_size": ("chunk_size", "chunked_prefill_size"),
    "batch_size": ("batch_size",),
    "sequence_length": ("sequence_length", "seq_len"),
    "num_tokens": ("num_tokens", "token_count"),
}


def _text_config(cfg):
    text = cfg.get("text_config")
    return text if isinstance(text, dict) and "num_hidden_layers" in text else cfg


def _config_containers(cfg):
    """Yield searchable config objects with their original JSON paths."""
    seen = set()

    def add(obj, path):
        if isinstance(obj, dict) and id(obj) not in seen:
            seen.add(id(obj))
            containers.append((obj, path))

    containers = []
    text = cfg.get("text_config")
    add(text, "text_config")
    add(cfg, "")
    for obj, path in list(containers):
        for name in ("quantization_config", "quant_config"):
            add(obj.get(name), ("%s.%s" % (path, name)).strip("."))
    return containers


def _field_from_containers(containers, aliases):
    for obj, prefix in containers:
        for alias in aliases:
            if obj.get(alias) is not None:
                path = ("%s.%s" % (prefix, alias)).strip(".")
                return {
                    "value": obj[alias],
                    "scope": "global",
                    "evidence": {
                        "source": "config",
                        "config_field": alias,
                        "config_path": path,
                        "raw_value": obj[alias],
                    },
                }
    return None


def _extract_fields(containers, specs):
    result = {}
    for canonical, aliases in specs.items():
        field = _field_from_containers(containers, aliases)
        if field is not None:
            result[canonical] = field
    return result


def _runtime_documents(runtime_sources):
    """Read JSON or command-line-like runtime evidence without executing it."""
    documents = []
    for path in runtime_sources or []:
        if not os.path.isfile(path):
            continue
        abspath = os.path.abspath(path)
        try:
            with open(path) as fh:
                raw = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            value = None
        if isinstance(value, dict):
            documents.extend(_flatten_runtime_dict(value, abspath))
        documents.append((raw, abspath, "text"))
    return documents


def _flatten_runtime_dict(value, source_path, prefix=""):
    rows = [(value, source_path, prefix)]
    for key, child in value.items():
        if isinstance(child, dict):
            child_prefix = ("%s.%s" % (prefix, key)).strip(".")
            rows.extend(_flatten_runtime_dict(child, source_path, child_prefix))
    return rows


def _runtime_field(documents, aliases):
    for document, source_path, prefix in documents:
        if isinstance(document, dict):
            for alias in aliases:
                if document.get(alias) is not None:
                    field_path = ("%s.%s" % (prefix, alias)).strip(".")
                    return {
                        "value": document[alias],
                        "evidence": {
                            "source": "runtime_source",
                            "path": source_path,
                            "field": field_path,
                            "raw_value": document[alias],
                        },
                    }
        elif isinstance(document, str):
            for alias in aliases:
                flag = alias.replace("_", "-")
                match = re.search(
                    r"(?:^|\s)--%s(?:=|\s+)([^\s]+)" % re.escape(flag),
                    document)
                if match:
                    raw = match.group(1).strip("'\"")
                    try:
                        value = int(raw)
                    except ValueError:
                        value = raw
                    return {
                        "value": value,
                        "evidence": {
                            "source": "runtime_source",
                            "path": source_path,
                            "field": "--%s" % flag,
                            "raw_value": raw,
                        },
                    }
    return None


def _partition(global_field, parallel_field, policy="sharded"):
    if global_field is None:
        return {"status": "unresolved", "reason": "global_value_unavailable"}
    if parallel_field is None:
        return {"status": "unresolved", "reason": "parallel_size_unavailable"}
    value = global_field["value"]
    size = parallel_field["value"]
    evidence = {
        "global": global_field["evidence"],
        "parallel_size": parallel_field["evidence"],
    }
    if not isinstance(value, int) or not isinstance(size, int) or size <= 0:
        return {
            "status": "unresolved",
            "reason": "non_integer_or_invalid_partition_inputs",
            "evidence": evidence,
        }
    if size == 1:
        return {
            "status": "resolved", "value": value, "scope": "rank_local",
            "rule": "parallel_size_is_one", "evidence": evidence,
        }
    if policy == "replication_possible" and value < size:
        return {
            "status": "replicated_policy_required",
            "reason": "global_value_smaller_than_parallel_size",
            "evidence": evidence,
        }
    if value % size:
        return {
            "status": "unresolved",
            "reason": "global_value_not_divisible_by_parallel_size",
            "evidence": evidence,
        }
    return {
        "status": "resolved", "value": value // size, "scope": "rank_local",
        "rule": "exact_integer_partition", "evidence": evidence,
    }


def _structural_context(cfg, runtime_sources):
    containers = _config_containers(cfg)
    static = {
        category: _extract_fields(containers, specs)
        for category, specs in STATIC_FIELD_SPECS.items()
    }
    # head_dim has a universally valid architectural derivation when both
    # operands are explicit and divisible.
    full = static["full_attention"]
    common = static["common"]
    if "norm_type" not in common:
        norm_eps = common.get("norm_epsilon")
        if (norm_eps and
                norm_eps["evidence"]["config_field"] == "rms_norm_eps"):
            common["norm_type"] = {
                "value": "rms_norm",
                "scope": "global",
                "evidence": {
                    "source": "config_derived",
                    "rule": "rms_norm_eps field semantics",
                    "inputs": [norm_eps["evidence"]],
                },
            }
    if "head_dim" not in full:
        hidden = common.get("hidden_size")
        heads = full.get("num_attention_heads")
        if (hidden and heads and isinstance(hidden["value"], int)
                and isinstance(heads["value"], int) and heads["value"] > 0
                and hidden["value"] % heads["value"] == 0):
            full["head_dim"] = {
                "value": hidden["value"] // heads["value"],
                "scope": "global",
                "evidence": {
                    "source": "config_derived",
                    "rule": "hidden_size / num_attention_heads (exact)",
                    "inputs": [hidden["evidence"], heads["evidence"]],
                },
            }

    documents = _runtime_documents(runtime_sources)
    runtime = {}
    for canonical, aliases in RUNTIME_FIELD_SPECS.items():
        field = _runtime_field(documents, aliases)
        if field is not None:
            runtime[canonical] = field
    tp = runtime.get("tensor_parallel_size")
    ep = runtime.get("expert_parallel_size")
    rank_local = {
        "attention_heads": _partition(
            full.get("num_attention_heads"), tp),
        "key_value_heads": _partition(
            full.get("num_key_value_heads"), tp,
            policy="replication_possible"),
        "linear_key_heads": _partition(
            static["linear_attention"].get("key_heads"), tp),
        "linear_value_heads": _partition(
            static["linear_attention"].get("value_heads"), tp),
        "experts": _partition(static["moe"].get("num_experts"), ep),
    }
    runtime["rank_local_derivations"] = rank_local
    return {
        "schema_version": "1.2",
        "static_model_context": static,
        "runtime_context": runtime,
    }


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
    structural_context = _structural_context(cfg, runtime_sources)
    for pattern in patterns:
        pattern["structural_context"] = dict(
            structural_context,
            pattern_scope={
                "attention_type": pattern["attention_type"],
                "ffn_type": pattern["ffn_type"],
            })
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
