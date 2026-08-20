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
    evidence.update({
        "level": "P",
        "probe_scope": scope,
        "evidence_origin": "shape_logger",
        "probe_table": os.path.abspath(source_path),
    })
    value["semantic_evidence"] = evidence
    value.setdefault("shape", {})["source"] = "runtime_probe_%s" % scope
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
            "kernel was not observed inside a matching graph/eager probe marker")
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
                selected["semantic_evidence"] = {
                    "level": "K",
                    "status": "preserved",
                    "source": "clean_trace_external_id",
                    "evidence_origin": "trace_input_dims",
                }
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
