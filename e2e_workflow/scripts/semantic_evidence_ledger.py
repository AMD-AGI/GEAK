#!/usr/bin/env python3
"""Accumulate Semantics K/P/U evidence from multiple runtime probe runs."""
import argparse
import copy
import json
import os

import semantic_shape_merge


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _rows(document):
    return {
        row["row_id"]: row
        for table in document.get("tables", [])
        for row in table.get("rows", [])
    }


def _table_key(table):
    return (
        table.get("phase"),
        table.get("pattern_id"),
        int(table.get("representative_layer_id", -1)),
    )


def _probe_quality(row):
    evidence = row.get("semantic_evidence") or {}
    if evidence.get("level") not in ("P", "C"):
        return None
    scope = evidence.get("probe_scope")
    if not scope:
        scope = "kernel" if evidence.get("level") == "P" else "wrapper"
    bucket = evidence.get("bucket_match")
    schema = evidence.get("schema") or {}
    return (
        2 if scope == "kernel" else 1,
        1 if bucket == "exact" else 0,
        len(schema.get("tensors") or []),
    )


def _normalise_probe(row, source_path):
    value = copy.deepcopy(row)
    evidence = copy.deepcopy(value.get("semantic_evidence") or {})
    scope = evidence.get("probe_scope")
    if not scope:
        scope = "kernel" if evidence.get("level") == "P" else "wrapper"
    graph_trace_shape = (
        evidence.get("source") == "graph_capture_trace_external_id")
    evidence.update({
        "level": "P",
        "probe_scope": scope,
        "evidence_origin": (
            "graph_capture_trace" if graph_trace_shape else "shape_logger"),
        "probe_table": os.path.abspath(source_path),
    })
    value["semantic_evidence"] = evidence
    shape = value.setdefault("shape", {})
    if not graph_trace_shape:
        shape["source"] = "runtime_probe_%s" % scope
    return value


def _reason_from_attempts(row, attempts):
    kernel = str(row.get("short_name") or row.get("raw_name") or "")
    if "__amd_rocclr_fillBufferAligned" in kernel:
        return (
            "runtime_internal_buffer_operation",
            "runtime buffer operation has no model tensor shape to attribute")
    if not kernel:
        return (
            "unnamed_runtime_kernel",
            "trace event has no stable kernel identity for probe correlation")
    codes = [
        attempt.get("reason_code") for attempt in attempts
        if attempt.get("reason_code")]
    if "multiple_wrapper_candidates" in codes:
        return (
            "multiple_wrapper_candidates",
            "all probes left multiple possible wrapper owners")
    if kernel == "kentry":
        return (
            "non_unique_native_kernel_name",
            "generic native kernel name remained ambiguous across all probes")
    if "runtime_copy_without_unique_tensor" in codes:
        return (
            "runtime_copy_without_unique_tensor",
            "runtime copy has no unique source-confirmed model tensor attribution")
    if "kernel_not_observed_in_probe" in codes:
        return (
            "kernel_not_observed_in_probe",
            "kernel was not observed inside a matching graph-capture marker")
    return (
        codes[-1] if codes else "probe_exhausted_without_unique_mapping",
        "available probe runs did not produce a unique shape attribution")


def merge(clean_table_path, probe_table_paths, out_dir):
    clean = _load(clean_table_path)
    output = copy.deepcopy(clean)
    probe_documents = [
        (path, _load(path)) for path in probe_table_paths]
    probe_rows = [
        (path, _rows(document)) for path, document in probe_documents]
    probe_tables = [
        (path, {_table_key(table): table
                for table in document.get("tables", [])})
        for path, document in probe_documents]

    audits = []
    counts = {"K": 0, "P": 0, "U": 0}
    for table in output.get("tables", []):
        key = _table_key(table)
        layer_candidates = [
            candidate_tables[key]
            for _, candidate_tables in probe_tables
            if key in candidate_tables
            and (candidate_tables[key].get("layer_io") or {}).get(
                "source") == "shape_logger"]
        if layer_candidates:
            layer_candidates.sort(
                key=lambda candidate: (
                    (candidate.get("layer_io") or {}).get(
                        "bucket_match") == "exact"),
                reverse=True)
            table["layer_io"] = copy.deepcopy(
                layer_candidates[0]["layer_io"])

        for row in table.get("rows", []):
            identity = {
                name: row.get(name) for name in (
                    "row_id", "raw_event_index", "device_seq_index",
                    "raw_name", "short_name", "duration_us")}
            history = []
            if row.get("shape", {}).get("source") == "kernel_exact":
                selected = copy.deepcopy(row)
                schema_candidates = []
                for path, rows_by_id in probe_rows:
                    candidate = rows_by_id.get(row["row_id"])
                    if not candidate:
                        continue
                    candidate_evidence = candidate.get(
                        "semantic_evidence") or {}
                    kernel_shape = (candidate.get("shape") or {}).get(
                        "kernel_shape")
                    if (candidate_evidence.get("level") == "K"
                            and kernel_shape):
                        resolution = kernel_shape.get(
                            "operator_schema_resolution") or {}
                        quality = (
                            resolution.get("status") == "matched_unique",
                            bool(kernel_shape.get("operator_schema")),
                            len(kernel_shape.get("operands") or []),
                        )
                        schema_candidates.append((
                            quality, path, candidate, kernel_shape))
                selected_evidence = {
                    "level": "K",
                    "status": "preserved",
                    "source": "clean_trace_external_id",
                    "evidence_origin": "trace_input_dims",
                }
                if schema_candidates:
                    _, path, candidate, kernel_shape = max(
                        schema_candidates, key=lambda item: item[0])
                    selected.setdefault("shape", {})["kernel_shape"] = (
                        copy.deepcopy(kernel_shape))
                    candidate_evidence = copy.deepcopy(
                        candidate.get("semantic_evidence") or {})
                    selected_evidence.update({
                        "operator_schema_manifest": candidate_evidence.get(
                            "operator_schema_manifest"),
                        "schema": copy.deepcopy(kernel_shape),
                    })
                    history.append({
                        "probe_table": os.path.abspath(path),
                        "level": "K",
                        "purpose": "operator_schema_enrichment_only",
                        "shape_source": "clean_trace_external_id",
                        "operator_schema_resolution": kernel_shape.get(
                            "operator_schema_resolution"),
                    })
                selected["semantic_evidence"] = selected_evidence
            else:
                candidates = []
                unavailable_attempts = []
                for path, rows_by_id in probe_rows:
                    candidate = rows_by_id.get(row["row_id"])
                    if not candidate:
                        continue
                    evidence = copy.deepcopy(
                        candidate.get("semantic_evidence") or {})
                    history.append({
                        "probe_table": os.path.abspath(path),
                        "level": evidence.get("level", "U"),
                        "probe_scope": evidence.get("probe_scope"),
                        "reason_code": evidence.get("reason_code"),
                        "reason": evidence.get("reason"),
                    })
                    quality = _probe_quality(candidate)
                    if quality is not None:
                        candidates.append((quality, path, candidate))
                    else:
                        unavailable_attempts.append(evidence)
                if candidates:
                    _, path, candidate = max(
                        candidates, key=lambda item: item[0])
                    selected = _normalise_probe(candidate, path)
                else:
                    selected = copy.deepcopy(row)
                    reason_code, reason = _reason_from_attempts(
                        row, unavailable_attempts)
                    selected["semantic_evidence"] = {
                        "level": "U",
                        "status": "unavailable",
                        "source": "probe_ledger",
                        "reason_code": reason_code,
                        "reason": reason,
                        "probe_attempt_count": len(unavailable_attempts),
                    }
            selected["evidence_history"] = history
            if any(selected.get(name) != value
                   for name, value in identity.items()):
                raise ValueError(
                    "probe changed clean trace identity for %s" %
                    row["row_id"])
            row.clear()
            row.update(selected)
            level = row["semantic_evidence"]["level"]
            counts[level] += 1
            audits.append({
                "row_id": row["row_id"],
                "phase": table.get("phase"),
                "pattern_id": table.get("pattern_id"),
                "representative_layer_id": table.get(
                    "representative_layer_id"),
                "pos": row.get("pos"),
                "kernel": row.get("short_name"),
                "evidence": row["semantic_evidence"],
                "evidence_history": history,
            })

    unexplained = [
        audit for audit in audits
        if audit["evidence"]["level"] == "U"
        and not audit["evidence"].get("reason_code")]
    classified = sum(counts.values()) == len(audits) and not unexplained
    # Probe merges change row-level shape evidence.  Refresh the embedded phase
    # summary so downstream consumers do not see the pre-probe "0/N" beside
    # rows that now carry resolved shapes.
    phase_stats = {}
    for table in output.get("tables", []):
        phase = table.get("phase")
        if not phase:
            continue
        stat = phase_stats.setdefault(phase, {"rows": 0, "resolved": 0})
        for row in table.get("rows", []):
            stat["rows"] += 1
            if (row.get("shape") or {}).get("input_dims"):
                stat["resolved"] += 1
    for stat in phase_stats.values():
        stat["resolved_fraction"] = (
            round(stat["resolved"] / float(stat["rows"]), 4)
            if stat["rows"] else 0.0)
    phase_coverage = output.setdefault("phase_coverage", {})
    phase_coverage["shape_resolution_by_phase"] = phase_stats
    decode = phase_stats.get("decode", {"resolved": 0})
    decode_sequence = bool(
        phase_coverage.get("decode_sequence_covered")
        or "decode" in phase_stats)
    phase_coverage["decode_shapes_covered"] = decode["resolved"] > 0
    phase_coverage["decode_covered"] = bool(
        decode_sequence and decode["resolved"] > 0)
    phase_coverage["decode_requires_graph_capture"] = bool(
        "decode" in phase_stats and decode["resolved"] == 0)
    if "decode" in phase_stats:
        phase_coverage["decode_evidence"] = (
            "sequence_and_shapes" if phase_coverage["decode_covered"]
            else "sequence_only_shapes_unresolved"
            if decode_sequence else "no_decode_trace_analysed")
    os.makedirs(out_dir, exist_ok=True)
    table_out = os.path.join(out_dir, "pattern_layer_kernel_table.json")
    markdown_out = os.path.join(out_dir, "ORDERED_UNIQUE_LAYER_TABLES.md")
    ledger_out = os.path.join(out_dir, "KPU_EVIDENCE_LEDGER.jsonl")
    coverage_out = os.path.join(out_dir, "KPU_COVERAGE_MANIFEST.json")
    verification_out = os.path.join(
        out_dir, "SHAPE_TYPE_VERIFICATION.json")
    with open(table_out, "w") as fh:
        json.dump(output, fh, indent=2)
    with open(markdown_out, "w") as fh:
        fh.write(semantic_shape_merge._markdown(output))
    with open(ledger_out, "w") as fh:
        for audit in audits:
            fh.write(json.dumps(audit, sort_keys=True) + "\n")
    unavailable = [
        {
            "row_id": audit["row_id"],
            "phase": audit["phase"],
            "pattern_id": audit["pattern_id"],
            "representative_layer_id": audit["representative_layer_id"],
            "pos": audit["pos"],
            "kernel": audit["kernel"],
            "reason_code": audit["evidence"]["reason_code"],
            "reason": audit["evidence"]["reason"],
            "probe_attempt_count": audit["evidence"].get(
                "probe_attempt_count", 0),
        }
        for audit in audits if audit["evidence"]["level"] == "U"]
    probe_scope_counts = {}
    for audit in audits:
        evidence = audit["evidence"]
        if evidence["level"] != "P":
            continue
        label = "P(%s)" % evidence.get("probe_scope", "wrapper")
        probe_scope_counts[label] = probe_scope_counts.get(label, 0) + 1
    unavailable_reason_counts = {}
    for item in unavailable:
        code = item["reason_code"]
        unavailable_reason_counts[code] = (
            unavailable_reason_counts.get(code, 0) + 1)
    coverage = {
        "schema_version": 1,
        "status": "pass" if classified else "fail",
        "classification_complete": classified,
        "priority": ["K", "P(kernel)", "P(wrapper)", "U"],
        "row_count": len(audits),
        "evidence_counts": counts,
        "probe_scope_counts": probe_scope_counts,
        "unavailable_reason_counts": unavailable_reason_counts,
        "probe_tables": [
            os.path.abspath(path) for path in probe_table_paths],
        "unavailable": unavailable,
    }
    with open(coverage_out, "w") as fh:
        json.dump(coverage, fh, indent=2)
    with open(verification_out, "w") as fh:
        json.dump({
            "schema_version": 2,
            "status": coverage["status"],
            "classification_complete": classified,
            "evidence_counts": counts,
            "row_count": len(audits),
            "unexplained_u_count": len(unexplained),
        }, fh, indent=2)
    return {
        "status": coverage["status"],
        "semantic_table_json": table_out,
        "semantic_table_md": markdown_out,
        "evidence_ledger_jsonl": ledger_out,
        "coverage_manifest": coverage_out,
        "shape_type_verification_json": verification_out,
        "evidence_counts": counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-table", required=True)
    parser.add_argument(
        "--probe-table", action="append", default=[], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = merge(args.clean_table, args.probe_table, args.out_dir)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
