#!/usr/bin/env python3
"""Build a deterministic trace manifest and semantic-capability report.

The manifest is the additive contract between the Profiler and Semantics Mapper.
Only the selected rank trace is inspected; no cross-rank merge is attempted.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter


TRACE_SUFFIXES = (".json", ".json.gz", ".pt.trace.json", ".pt.trace.json.gz")
SGLANG_STEP_RE = re.compile(
    r"^step\[(EXTEND|DECODE)\s+bs=(\d+)(?:\s+toks=(\d+))?\]$")
MODULE_LAYER_RE = re.compile(
    r"^nn\.Module:\s+.*DecoderLayer_(\d+)$", re.IGNORECASE)


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rank(path):
    name = os.path.basename(path)
    for rx in (r"(?:^|[_-])rank[_-]?(\d+)", r"(?:^|[_-])tp[_-]?(\d+)"):
        match = re.search(rx, name, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def discover(trace_dir):
    files = []
    if os.path.isfile(trace_dir):
        files = [os.path.abspath(trace_dir)]
    elif os.path.isdir(trace_dir):
        for name in os.listdir(trace_dir):
            path = os.path.join(trace_dir, name)
            if os.path.isfile(path) and name.endswith(TRACE_SUFFIXES):
                files.append(os.path.abspath(path))
    files.sort(key=lambda p: (_rank(p) is None, _rank(p) or 0, os.path.basename(p)))
    return files


def inspect_trace(path):
    with _open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    cats = Counter()
    device = 0
    external = 0
    dims = 0
    cpu_parent = 0
    flow = 0
    correlation = 0
    phase_spans = 0
    phase_dialects = Counter()
    module_layer_spans = 0
    module_layer_names = set()
    boundary_anchors = Counter()
    streams = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        cat = str(event.get("cat", ""))
        cats[cat] += 1
        args = event.get("args") or {}
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            device += 1
            if args.get("External id") is not None:
                external += 1
            stream = args.get("stream") if args.get("stream") is not None else args.get("Stream")
            if stream is not None:
                streams.add(str(stream))
            if any(args.get(key) is not None for key in (
                    "correlation", "correlation_id", "Correlation ID")):
                correlation += 1
            name = str(event.get("name", "")).lower()
            for anchor, regex in (
                    ("collective", r"all.?reduce|reduce.?scatter|all.?gather|nccl|rccl"),
                    ("norm", r"rms.?norm|layer.?norm"),
                    ("router", r"topk|router|routing")):
                if re.search(regex, name):
                    boundary_anchors[anchor] += 1
        if cat == "cpu_op":
            if args.get("Input Dims"):
                dims += 1
            if event.get("dur") is not None and event.get("ts") is not None:
                cpu_parent += 1
        if cat == "gpu_user_annotation":
            name = str(event.get("name", ""))
            if name.startswith("execute_") and "context_" in name and "generation_" in name:
                phase_spans += 1
                phase_dialects["legacy_execute"] += 1
            elif SGLANG_STEP_RE.match(name):
                phase_spans += 1
                phase_dialects["sglang_step"] += 1
        if cat == "python_function":
            name = str(event.get("name", ""))
            if MODULE_LAYER_RE.match(name):
                module_layer_spans += 1
                module_layer_names.add(name)
        if event.get("ph") in ("s", "t", "f") or "flow" in cat.lower():
            flow += 1
    return {
        "event_count": len(events),
        "categories": dict(sorted(cats.items())),
        "device_event_count": device,
        "device_external_id_coverage": (external / device if device else 0.0),
        "cpu_op_with_input_dims": dims,
        "cpu_op_interval_count": cpu_parent,
        "phase_annotation_count": phase_spans,
        "phase_annotation_dialects": dict(sorted(phase_dialects.items())),
        "module_layer_span_count": module_layer_spans,
        "distinct_module_layer_names": len(module_layer_names),
        "flow_event_count": flow,
        "device_correlation_count": correlation,
        "layer_boundary_anchor_candidates": dict(sorted(boundary_anchors.items())),
        "streams": sorted(streams),
        "capabilities": {
            "phase_annotations": phase_spans > 0,
            "cpu_op_scopes": cpu_parent > 0,
            "external_id": external > 0,
            "input_dims_types": dims > 0,
            "flow_or_correlation": flow > 0 or correlation > 0,
            "stable_layer_anchor_candidates": bool(boundary_anchors),
            "module_layer_spans": module_layer_spans > 0,
        },
        "recommended_layer_mapping": (
            "module_span_plus_flow" if module_layer_spans and (flow or correlation)
            else "module_span" if module_layer_spans
            else "ordered_anchor_fallback" if boundary_anchors
            else "unresolved"),
    }


def build_manifest(trace_dir, analysis_rank=0):
    files = discover(trace_dir)
    entries = [{"path": path, "rank": _rank(path), "sha256": _sha256(path)}
               for path in files]
    selected = next((e for e in entries if e["rank"] == analysis_rank), None)
    if selected is None and entries:
        selected = entries[0]
    capabilities = inspect_trace(selected["path"]) if selected else {
        "event_count": 0,
        "categories": {},
        "device_event_count": 0,
        "capabilities": {},
        "error": "no top-level torch trace found",
    }
    return {
        "schema_version": 1,
        "trace_dir": os.path.abspath(trace_dir),
        "trace_files": entries,
        "analysis_rank": analysis_rank,
        "analysis_rank_trace": selected["path"] if selected else "",
        "analysis_rank_sha256": selected["sha256"] if selected else "",
        "cross_rank_merge": False,
        "capability": capabilities,
        "status": "pass" if selected else "failed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--analysis-rank", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    doc = build_manifest(args.trace_dir, args.analysis_rank)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(args.out)
    return 0 if doc["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
