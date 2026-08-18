#!/usr/bin/env python3
"""Build stable executable-task identities from profiling candidates.

Profilers are allowed to aggregate several device kernels into one framework
operation for Amdahl accounting.  Such an aggregate is not automatically an
executable extraction target.  This module preserves the aggregate as a
``profiling_entity`` and materializes separate ``executable_task_candidates``
only when the available evidence identifies a deployable optimization unit.

The implementation is deliberately framework- and model-agnostic.  It consumes
the structural fields emitted by profiling tools (operation name, device kernel
names, category, dispatch kind, and source hints); it does not match model names
or historical run identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import OrderedDict
from typing import Any, Iterable


SCHEMA = "op-identity-v1"
_ATTENTION_RE = re.compile(r"attention|attn|paged|flash|fmha|mha|mla", re.I)
_MOE_RE = re.compile(r"(?:^|[^a-z])moe(?:[^a-z]|$)|fmoe", re.I)
_FUSION_RE = re.compile(r"(?:^|[^a-z])fused(?:[^a-z]|$)|fmoe|group(?:ed)?[_ -]?gemm", re.I)
_FULL_MOE_RE = re.compile(
    r"fmoe|fused[_ :>-]*(?:moe|experts?)|(?:moe)[_ :>-]*fused|asm[_ :>-]*moe|moe[_ :>-]*dispatch",
    re.I,
)
_COMM_RE = re.compile(
    r"collective|nccl|rccl|all[_ -]?(?:reduce|gather)|reduce[_ -]?scatter|"
    r"cross[_ -]?device|custom[_ -]?all[_ -]?reduce|communication|comm[_ -]",
    re.I,
)
_PREFILL_RE = re.compile(r"prefill|context|prompt", re.I)
_DECODE_RE = re.compile(r"decode", re.I)
_PAGED_RE = re.compile(r"paged", re.I)


def _text(*values: Any) -> str:
    return " ".join(str(v) for v in values if v not in (None, "", [], {}))


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _casefold_sort_key(value: Any) -> tuple[str, str]:
    text = str(value)
    return text.casefold(), text


def slug(value: str) -> str:
    """Return a filesystem-safe, human-readable identity component."""
    value = re.sub(r"^void\s+", "", str(value or "").strip())
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return value[:80] or "unknown"


def stable_id(prefix: str, *parts: Any) -> str:
    """Return a stable key whose suffix prevents collisions after slugging."""
    normalized = "|".join(str(part or "").strip().lower() for part in parts)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    stem = slug(next((str(part) for part in reversed(parts) if part), prefix))
    return f"{prefix}_{stem}_{digest}"


def device_kernel_names(candidate: dict[str, Any]) -> list[str]:
    """Collect device kernel names without discarding singular-form producers."""
    values: list[Any] = []
    plural = candidate.get("device_kernel_names")
    if isinstance(plural, list):
        values.extend(plural)
    elif plural:
        values.append(plural)
    values.append(candidate.get("device_kernel_name"))
    return _unique_strings(values)


def _source_hints(candidate: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in (
        "source_file",
        "source_path",
        "kernel_path",
        "launcher_source_file",
        "op_to_source_matched_route",
    ):
        value = candidate.get(key)
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
    return _unique_strings(values)


def _category_text(candidate: dict[str, Any]) -> str:
    return _text(
        candidate.get("kernel_category"),
        candidate.get("tracelens_category"),
        candidate.get("classification"),
        candidate.get("kernel_kind"),
        candidate.get("kernel_contract"),
        candidate.get("identity_evidence"),
        candidate.get("name"),
    )


def is_attention(candidate: dict[str, Any]) -> bool:
    return bool(_ATTENTION_RE.search(_category_text(candidate)))


def is_fused_moe(candidate: dict[str, Any]) -> bool:
    text = _category_text(candidate)
    if re.search(r"(?:^|[^a-z])unfused(?:[^a-z]|$)", text, re.I):
        return False
    full_moe = bool(_FULL_MOE_RE.search(text))
    if re.search(r"fused[_ :>-]*experts?[_ :>-]*(?:matmul|gemm)", text, re.I):
        full_moe = False
    explicit_moe = (
        str(candidate.get("op_kind") or "").lower() == "moe"
        or full_moe
        or (
            candidate.get("is_fused_kernel") is True
            and bool(_MOE_RE.search(text))
        )
    )
    return bool(
        explicit_moe
        and (candidate.get("is_fused_kernel") is True or _FUSION_RE.search(text))
    )


def infer_op_kind(candidate: dict[str, Any]) -> str:
    if is_attention(candidate):
        return "attn"
    if is_fused_moe(candidate):
        return "moe"
    text = _category_text(candidate)
    if re.search(r"gemm|matmul|linear", text, re.I):
        return "gemm"
    return str(candidate.get("op_kind") or "kernel").lower()


def infer_regimes(
    kernel_name: str,
    hints: Iterable[str],
    explicit: Iterable[str] = (),
) -> list[str]:
    """Infer only when structural names identify a regime; otherwise keep both."""
    explicit_regimes = _unique_strings(
        str(value).lower() for value in explicit if str(value).lower() in ("prefill", "decode")
    )
    if explicit_regimes:
        return [name for name in ("prefill", "decode") if name in explicit_regimes]
    # The device symbol is stronger evidence than an aggregate source-hint list:
    # a decode leaf may share an outer launcher with a prefill leaf.
    prefill = bool(_PREFILL_RE.search(kernel_name))
    decode = bool(_DECODE_RE.search(kernel_name))
    if prefill and not decode:
        return ["prefill"]
    if decode and not prefill:
        return ["decode"]
    evidence = _text(*hints)
    prefill = bool(_PREFILL_RE.search(evidence))
    decode = bool(_DECODE_RE.search(evidence))
    if prefill and not decode:
        return ["prefill"]
    if decode and not prefill:
        return ["decode"]
    if _PAGED_RE.search(kernel_name):
        return ["decode"]
    return ["prefill", "decode"]


def infer_execution_scope(candidate: dict[str, Any], kernels: list[str]) -> str:
    """Classify a profiling entity without assuming aggregate always means leaf."""
    category = _category_text(candidate)
    if candidate.get("e2e_transferable") is False or _COMM_RE.search(category):
        return "config_only"
    if (
        re.search(r"synthetic op", _candidate_name(candidate), re.I)
        and not candidate.get("op_to_source_patchable")
    ):
        return "blocked"
    if is_fused_moe(candidate):
        # A fused operation owns its full production contract.  Decomposing it
        # into constituent kernels would recreate the standalone-GEMM bug.
        return "executable_op"
    if candidate.get("op_to_source_patchable") is False:
        return "config_only"
    if is_attention(candidate):
        # Attention aggregates frequently combine prefill and decode kernels.
        # The aggregate remains useful for Amdahl accounting, while extraction
        # must happen at a leaf with a separately verified consumer seam.
        return "expand_leaves" if kernels else "config_only"
    if len(kernels) > 1 or str(candidate.get("op_to_source_kind") or "").lower() == "dispatch":
        return "expand_leaves" if kernels else "blocked"
    return "executable_op"


def _candidate_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("name") or candidate.get("short_name") or "unknown").strip()


def _base_operation_name(name: str) -> str:
    """Collapse profiler-added regime suffixes while preserving the operation."""
    return re.sub(r"\s*\((?:prefill|decode|mixed)\)\s*$", "", str(name or "").strip(), flags=re.I)


def _explicit_regimes(
    rows: Iterable[dict[str, Any]],
    leaf: str | None = None,
) -> list[str]:
    regimes: list[str] = []
    for row in rows:
        # Regime labels on a multi-device aggregate belong to the parent op, not
        # necessarily to each leaf.  Use them for a leaf only when the row is
        # structurally leaf-specific.
        if (
            leaf is not None
            and len(device_kernel_names(row)) > 1
            and row.get("profiling_kind") != "device_leaf"
        ):
            continue
        served = row.get("served_regimes")
        if isinstance(served, list):
            regimes.extend(str(value).lower() for value in served)
        phase = str(row.get("phase") or "").lower()
        if phase in ("prefill", "decode"):
            regimes.append(phase)
        elif phase == "both":
            regimes.extend(["prefill", "decode"])
        suffix = re.search(
            r"\((prefill|decode|mixed)\)\s*$", _candidate_name(row), re.I
        )
        if suffix:
            if suffix.group(1).lower() == "mixed":
                regimes.extend(["prefill", "decode"])
            else:
                regimes.append(suffix.group(1).lower())
    unique = _unique_strings(regimes)
    return [name for name in ("prefill", "decode") if name in unique]


def _choose_classification(values: Iterable[str], op_kind: str) -> str:
    ordered = sorted(_unique_strings(values), key=_casefold_sort_key)
    patterns = {
        "attn": _ATTENTION_RE,
        "moe": re.compile(r"moe|expert", re.I),
        "gemm": re.compile(r"gemm|matmul|linear", re.I),
    }
    pattern = patterns.get(op_kind)
    if pattern:
        matching = [value for value in ordered if pattern.search(value)]
        if matching:
            return matching[0]
    return ordered[0] if ordered else ""


def _allocate_leaf_percentages(
    kernels: Iterable[str],
    measured_weights: dict[str, float],
    total_pct: float,
) -> tuple[dict[str, float], dict[str, str]]:
    names = sorted(_unique_strings(kernels))
    if not names:
        return {}, {}
    total_units = int(round(float(total_pct) * 1_000_000))
    weighted = [name for name in names if float(measured_weights.get(name, 0.0)) > 0]
    active = weighted or names
    weight_total = (
        sum(float(measured_weights[name]) for name in active)
        if weighted
        else float(len(active))
    )
    units: dict[str, int] = {name: 0 for name in names}
    exact: dict[str, float] = {}
    for name in active:
        weight = float(measured_weights[name]) if weighted else 1.0
        exact[name] = total_units * weight / weight_total
        units[name] = math.floor(exact[name])
    remainder = total_units - sum(units.values())
    residual_order = sorted(
        active,
        key=lambda name: (-(exact[name] - units[name]), name),
    )
    for name in residual_order[:remainder]:
        units[name] += 1
    values = {name: units[name] / 1_000_000.0 for name in names}
    if weighted:
        sources = {
            name: ("leaf_attributed" if name in weighted else "unattributed_leaf")
            for name in names
        }
    else:
        sources = {name: "equal_split_parent" for name in names}
    return values, sources


def _merge_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge repeated signatures of one framework operation for Amdahl use."""
    rows = [row for row in candidates if isinstance(row, dict)]
    variant_bases = {
        _base_operation_name(_candidate_name(row)).lower()
        for row in rows
        if _candidate_name(row) != _base_operation_name(_candidate_name(row))
    }
    merged: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for raw in rows:
        raw_name = _candidate_name(raw)
        name = _base_operation_name(raw_name)
        key = name.lower()
        if key not in merged:
            merged[key] = {
                **raw,
                "name": name,
                "_device_kernel_names": [],
                "_source_hints": [],
                "_raw_candidates": [],
                "_source_kinds": [],
                "_op_kind_values": [],
                "_patchable_values": [],
                "_category_values": [],
                "_transferable_values": [],
                "_fused_values": [],
                "_leaf_gpu_pct": OrderedDict(),
                "_gpu_pct_sum": 0.0,
                "_duration_us_sum": 0.0,
                "_detail_call_count_sum": 0,
                "_summary_gpu_pct": None,
                "_summary_duration_us": None,
                "_summary_call_count": None,
            }
        entry = merged[key]
        entry["_device_kernel_names"] = _unique_strings(
            [*entry["_device_kernel_names"], *device_kernel_names(raw)]
        )
        entry["_source_hints"] = _unique_strings(
            [*entry["_source_hints"], *_source_hints(raw)]
        )
        entry["_raw_candidates"].append(raw)
        entry["_source_kinds"] = _unique_strings(
            [*entry["_source_kinds"], raw.get("op_to_source_kind")]
        )
        entry["_op_kind_values"] = _unique_strings(
            [*entry["_op_kind_values"], raw.get("op_kind")]
        )
        if raw.get("op_to_source_patchable") is not None:
            entry["_patchable_values"].append(bool(raw.get("op_to_source_patchable")))
        entry["_category_values"] = _unique_strings(
            [
                *entry["_category_values"],
                raw.get("kernel_category"),
                raw.get("tracelens_category"),
                raw.get("classification"),
                raw.get("kernel_kind"),
                raw.get("kernel_contract"),
            ]
        )
        if raw.get("e2e_transferable") is not None:
            entry["_transferable_values"].append(bool(raw.get("e2e_transferable")))
        if raw.get("is_fused_kernel") is not None:
            entry["_fused_values"].append(bool(raw.get("is_fused_kernel")))
        raw_kernels = device_kernel_names(raw)
        raw_calls = raw.get("call_count") or raw.get("calls") or 0
        is_parent_summary = (
            str(raw.get("profiling_kind") or "").lower() == "aggregate"
            or (raw_name == name and key in variant_bases)
        )
        try:
            raw_gpu_pct = float(raw.get("gpu_pct") or raw.get("pct_gpu_time") or 0.0)
            entry["_gpu_pct_sum"] += raw_gpu_pct
            if is_parent_summary and raw_gpu_pct:
                entry["_summary_gpu_pct"] = max(
                    float(entry["_summary_gpu_pct"] or 0.0), raw_gpu_pct
                )
            primary = str(raw.get("device_kernel_name") or "").strip()
            if not primary and len(raw_kernels) == 1:
                primary = raw_kernels[0]
            if primary and raw_gpu_pct and not is_parent_summary:
                entry["_leaf_gpu_pct"][primary] = (
                    float(entry["_leaf_gpu_pct"].get(primary, 0.0)) + raw_gpu_pct
                )
        except (TypeError, ValueError):
            pass
        try:
            raw_duration_us = float(
                raw.get("duration_us")
                or (float(raw.get("total_ms") or 0.0) * 1000.0)
            )
            entry["_duration_us_sum"] += raw_duration_us
            if is_parent_summary and raw_duration_us:
                entry["_summary_duration_us"] = max(
                    float(entry["_summary_duration_us"] or 0.0), raw_duration_us
                )
        except (TypeError, ValueError):
            pass
        try:
            count = int(raw.get("call_count") or raw.get("calls") or 0)
            if is_parent_summary:
                entry["_summary_call_count"] = max(
                    int(entry["_summary_call_count"] or 0), count
                )
            else:
                entry["_detail_call_count_sum"] += count
        except (TypeError, ValueError):
            pass
    return [merged[key] for key in sorted(merged)]


def _task(
    entity: dict[str, Any],
    short_name: str,
    regimes: list[str],
    *,
    resolution_status: str,
    source_hints: list[str] | None = None,
) -> dict[str, Any]:
    task_key = stable_id(
        "task",
        entity["entity_id"],
        short_name,
    )
    leaf_pct = entity.get("leaf_pct_gpu_time", {}).get(short_name)
    if entity["execution_scope"] == "executable_op":
        task_pct = entity["pct_gpu_time"]
        pct_source = "executable_entity"
    elif leaf_pct is not None:
        task_pct = leaf_pct
        pct_source = entity.get("leaf_pct_source", {}).get(
            short_name, "leaf_attributed"
        )
    elif entity.get("leaf_pct_gpu_time"):
        task_pct = 0.0
        pct_source = "unattributed_leaf"
    else:
        divisor = max(len(entity.get("device_kernel_names") or []), 1)
        task_pct = round(float(entity["pct_gpu_time"]) / divisor, 6)
        pct_source = "equal_split_parent"
    return {
        "stable_task_key": task_key,
        "profiling_entity_id": entity["entity_id"],
        "profiling_entity_name": entity["display_name"],
        "short_name": short_name,
        "op_kind": entity["op_kind"],
        "device_kernel_names": [short_name] if short_name else [],
        "served_regimes": regimes,
        "pct_gpu_time": task_pct,
        "parent_pct_gpu_time": entity["pct_gpu_time"],
        "pct_gpu_time_source": pct_source,
        "source_hints": list(source_hints if source_hints is not None else entity["source_hints"]),
        "execution_scope": entity["execution_scope"],
        "capture_policy": "require_live" if entity["op_kind"] in ("attn", "moe") else "allow_value_independent",
        "baseline_policy": "require_live",
        "consumer_callable": "",
        "rebind_callable": "",
        "resolution_status": resolution_status,
    }


def normalize_candidates(document: Any) -> dict[str, Any]:
    """Normalize a TraceLens-like document into identity entities and tasks."""
    profile_source = ""
    from_top_kernels = False
    if isinstance(document, dict):
        candidates = document.get("hot_kernels")
        if candidates is None:
            candidates = document.get("top_kernels")
            from_top_kernels = candidates is not None
        if candidates is None:
            candidates = [document]
        framework = str(document.get("framework") or "")
        profile_source = str(document.get("source") or "")
    elif isinstance(document, list):
        candidates = document
        framework = ""
    else:
        raise TypeError("identity input must be an object or array")

    if from_top_kernels and profile_source in ("torch-trace", "rocprofv3", "merged"):
        normalized_top = []
        for row in candidates or []:
            row = dict(row)
            if (
                not device_kernel_names(row)
                and row.get("profiling_kind") != "aggregate"
            ):
                row["device_kernel_name"] = str(
                    row.get("short_name") or row.get("name") or ""
                )
                row["device_kernel_names"] = [row["device_kernel_name"]]
                row["profiling_kind"] = "device_leaf"
            normalized_top.append(row)
        candidates = normalized_top

    entities: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for candidate in _merge_candidates(candidates or []):
        name = _candidate_name(candidate)
        kernels = sorted(candidate.pop("_device_kernel_names"))
        hints = sorted(candidate.pop("_source_hints"))
        raw_candidates = sorted(
            candidate.pop("_raw_candidates"),
            key=lambda row: (
                _candidate_name(row).lower(),
                str(row.get("device_kernel_name") or "").lower(),
                _text(*_source_hints(row)).lower(),
            ),
        )
        source_kinds = sorted(candidate.pop("_source_kinds"), key=_casefold_sort_key)
        op_kind_values = sorted(candidate.pop("_op_kind_values"), key=_casefold_sort_key)
        patchable_values = list(candidate.pop("_patchable_values"))
        category_values = sorted(candidate.pop("_category_values"), key=_casefold_sort_key)
        transferable_values = list(candidate.pop("_transferable_values"))
        fused_values = list(candidate.pop("_fused_values"))
        leaf_gpu_pct_raw = dict(candidate.pop("_leaf_gpu_pct"))
        if "dispatch" in [kind.lower() for kind in source_kinds]:
            candidate["op_to_source_kind"] = "dispatch"
        elif source_kinds:
            candidate["op_to_source_kind"] = source_kinds[0]
        if op_kind_values:
            priority = {"attn": 0, "attention": 0, "moe": 1, "gemm": 2, "kernel": 3}
            candidate["op_kind"] = min(
                op_kind_values,
                key=lambda value: (priority.get(value.lower(), 4), value.lower()),
            )
        if patchable_values:
            candidate["op_to_source_patchable"] = any(patchable_values)
        candidate["identity_evidence"] = _text(
            *category_values,
            *kernels,
            *sorted(
                _unique_strings(raw.get("kernel_contract") for raw in raw_candidates),
                key=_casefold_sort_key,
            ),
        )
        if transferable_values:
            candidate["e2e_transferable"] = all(transferable_values)
        if fused_values:
            candidate["is_fused_kernel"] = any(fused_values)
        gpu_pct_sum = float(candidate.pop("_gpu_pct_sum"))
        duration_us_sum = float(candidate.pop("_duration_us_sum"))
        summary_gpu_pct = candidate.pop("_summary_gpu_pct")
        summary_duration_us = candidate.pop("_summary_duration_us")
        summary_call_count = candidate.pop("_summary_call_count")
        detail_call_count = candidate.pop("_detail_call_count_sum")
        gpu_pct = round(float(summary_gpu_pct if summary_gpu_pct is not None else gpu_pct_sum), 6)
        duration_us = round(
            float(summary_duration_us if summary_duration_us is not None else duration_us_sum), 3
        )
        call_count = int(
            summary_call_count
            if summary_call_count not in (None, 0)
            else detail_call_count
        )
        op_kind = infer_op_kind(candidate)
        scope = infer_execution_scope(candidate, kernels)
        entity_id = stable_id("entity", framework, name)
        leaf_pct_gpu_time, leaf_pct_source = _allocate_leaf_percentages(
            kernels, leaf_gpu_pct_raw, gpu_pct
        ) if scope == "expand_leaves" else ({}, {})
        entity = {
            "entity_id": entity_id,
            "display_name": name,
            "op_name": name,
            "framework": framework or str(candidate.get("framework") or ""),
            "op_kind": op_kind,
            "classification": _choose_classification(category_values, op_kind),
            "pct_gpu_time": gpu_pct,
            "total_ms": round(duration_us / 1000.0, 6),
            "calls": call_count,
            "device_kernel_names": kernels,
            "leaf_pct_gpu_time": leaf_pct_gpu_time,
            "leaf_pct_source": leaf_pct_source,
            "source_hints": hints,
            "execution_scope": scope,
            "op_to_source_kind": str(candidate.get("op_to_source_kind") or ""),
            "op_to_source_patchable": candidate.get("op_to_source_patchable"),
        }
        entities.append(entity)

        if scope == "expand_leaves":
            for kernel in kernels:
                primary_hints: list[str] = []
                related_hints: list[str] = []
                primary_rows: list[dict[str, Any]] = []
                related_rows: list[dict[str, Any]] = []
                for raw in raw_candidates:
                    raw_names = device_kernel_names(raw)
                    if str(raw.get("device_kernel_name") or "").strip() == kernel:
                        primary_rows.append(raw)
                        primary_hints.extend(_source_hints(raw))
                    elif kernel in raw_names:
                        related_rows.append(raw)
                        related_hints.extend(_source_hints(raw))
                kernel_hints = sorted(
                    _unique_strings([*primary_hints, *related_hints]),
                    key=_casefold_sort_key,
                )
                explicit = _explicit_regimes(primary_rows or related_rows, kernel)
                inference_hints = primary_hints or related_hints or hints
                regimes = infer_regimes(kernel, inference_hints, explicit)
                tasks.append(
                    _task(
                        entity,
                        kernel,
                        regimes,
                        resolution_status="needs_seam",
                        source_hints=kernel_hints or hints,
                    )
                )
        elif scope == "executable_op":
            tasks.append(
                _task(
                    entity,
                    name,
                    infer_regimes(name, hints, _explicit_regimes(raw_candidates)),
                    resolution_status="needs_seam",
                )
            )
        else:
            blocked.append(
                {
                    "profiling_entity_id": entity_id,
                    "display_name": name,
                    "execution_scope": scope,
                    "reason": (
                        "configuration-only profiling entity"
                        if scope == "config_only"
                        else "no executable device leaf or verified operation contract"
                    ),
                }
            )

    return {
        "schema": SCHEMA,
        "profiling_entities": entities,
        "executable_task_candidates": tasks,
        "blocked_entities": blocked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Separate profiling aggregates from executable task identities."
    )
    parser.add_argument("--input", required=True, help="TraceLens/profile candidate JSON")
    parser.add_argument("--output", required=True, help="Identity JSON output")
    args = parser.parse_args(argv)

    with open(args.input, encoding="utf-8") as fh:
        document = json.load(fh)
    result = normalize_candidates(document)

    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(
        json.dumps(
            {
                "output": args.output,
                "profiling_entities": len(result["profiling_entities"]),
                "executable_task_candidates": len(result["executable_task_candidates"]),
                "blocked_entities": len(result["blocked_entities"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
