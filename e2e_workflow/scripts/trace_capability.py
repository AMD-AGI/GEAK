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

import sglang_step_modes


TRACE_SUFFIXES = (".json", ".json.gz", ".pt.trace.json", ".pt.trace.json.gz")
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
    step_modes = Counter()
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
            else:
                step = sglang_step_modes.parse_step(name)
                if step is not None:
                    phase_spans += 1
                    phase_dialects["sglang_step"] += 1
                    step_modes[step["mode"]] += 1
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
        "step_mode_counts": dict(sorted(step_modes.items())),
        "speculative_decoding": any(
            mode in sglang_step_modes.SPECULATIVE_MODES for mode in step_modes),
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


def _stage_device_coverage(path):
    """Return per-step device coverage for each GPU stage annotation."""
    with _open(path) as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    device = [
        event for event in events
        if isinstance(event, dict)
        and event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
        and event.get("ts") is not None
    ]
    coverage = Counter()
    duration = Counter()
    annotation_count = Counter()
    modes = Counter()
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "gpu_user_annotation":
            continue
        step = sglang_step_modes.parse_step(event.get("name"))
        if step is None or event.get("ts") is None or event.get("dur") is None:
            continue
        # Speculative decoding runs target generation as TARGET_VERIFY (phase
        # "verify"). Draft steps are reported under their own "draft" phase and
        # never compete for generation-step selection.
        phase = step["phase"]
        modes[step["mode"]] += 1
        start = float(event["ts"])
        end = start + float(event["dur"])
        # A retry may contain several steps. Use the best single step rather
        # than summing them, otherwise three truncated steps can look complete.
        coverage[phase] = max(coverage[phase], sum(
            1 for item in device if start <= float(item["ts"]) <= end))
        duration[phase] = max(duration[phase], float(event["dur"]))
        annotation_count[phase] += 1
    return {
        "device_events_by_phase": dict(sorted(coverage.items())),
        "annotation_duration_us_by_phase": dict(sorted(duration.items())),
        "annotation_count_by_phase": dict(sorted(annotation_count.items())),
        "step_mode_counts": dict(sorted(modes.items())),
    }


_SERVER_ARGS_RE = re.compile(r"server_args=ServerArgs\((?P<body>.*)\)")
_ACCEPT_RE = re.compile(
    r"#running-req:\s*(?P<running>\d+).*?accept len:\s*(?P<accept>[0-9.]+)")


def _server_log_facts(path):
    """Optional speculative facts from an SGLang server log (never a gate).

    The log only enriches what the trace already decided: the runtime-resolved
    speculative config (e.g. NEXTN is reported as EAGLE) and the steady-state
    mean accept length (decode-stat rows whose running-req count is at least
    90% of the largest seen, so warmup/ramp/tail rows do not bias it).
    """
    facts = {"algorithm_runtime": None, "verify_tokens_per_request": None,
             "topk": None, "num_steps": None, "accept_len_mean": None,
             "accept_len_rows": 0, "server_log": None, "warnings": []}
    if not path:
        facts["warnings"].append("no server log supplied; speculative "
                                 "parameters left unset")
        return facts
    facts["server_log"] = os.path.abspath(path)
    try:
        with open(path, errors="ignore") as fh:
            text = fh.read()
    except OSError as exc:
        facts["warnings"].append("server log unreadable: %s" % exc)
        return facts
    match = _SERVER_ARGS_RE.search(text)
    if match:
        body = match.group("body")
        for key, field, cast in (
                ("speculative_algorithm", "algorithm_runtime", str),
                ("speculative_num_draft_tokens", "verify_tokens_per_request", int),
                ("speculative_eagle_topk", "topk", int),
                ("speculative_num_steps", "num_steps", int)):
            value = re.search(r"\b%s=('?)([^,')]*)\1" % key, body)
            if value and value.group(2) not in ("", "None"):
                try:
                    facts[field] = cast(value.group(2))
                except ValueError:
                    facts["warnings"].append("unparsable %s" % key)
    else:
        facts["warnings"].append("server_args line not found in server log")
    rows = []
    for line in text.splitlines():
        found = _ACCEPT_RE.search(line)
        if found:
            try:
                rows.append((int(found.group("running")),
                             float(found.group("accept"))))
            except ValueError:
                continue
    if rows:
        peak = max(running for running, _ in rows)
        steady = [accept for running, accept in rows if running >= 0.9 * peak]
        facts["accept_len_mean"] = round(sum(steady) / len(steady), 4)
        facts["accept_len_rows"] = len(steady)
    elif "accept len" in text:
        facts["warnings"].append("accept len rows present but unparsable")
    return facts


def _speculative_block(entries, server_log):
    """Decide speculative decoding from the trace; enrich from the log."""
    modes = {mode for entry in entries
             for mode in entry.get("step_mode_counts", {})}
    verify_modes = sorted(
        mode for mode in modes
        if sglang_step_modes.PHASE_BY_MODE.get(mode) == "verify")
    enabled = bool(verify_modes)
    block = {"enabled": enabled,
             "evidence": ("trace:" + ",".join(verify_modes)) if enabled
             else "trace:no_verify_steps"}
    facts = _server_log_facts(server_log) if (enabled or server_log) else {
        "warnings": []}
    if enabled:
        block.update({key: facts.get(key) for key in (
            "algorithm_runtime", "verify_tokens_per_request", "topk",
            "num_steps", "accept_len_mean", "accept_len_rows", "server_log")})
    warnings = list(facts.get("warnings", []))
    if not enabled and facts.get("algorithm_runtime"):
        warnings.append(
            "server log reports speculative_algorithm=%s but the trace has no "
            "verify steps; the trace decides (generation phase = decode)"
            % facts["algorithm_runtime"])
    if not enabled:
        warnings = [w for w in warnings if "trace" in w]
    block["warnings"] = warnings
    return block


def build_manifest(trace_dir, analysis_rank=0, auto_select_rank=False,
                   server_log=None):
    files = discover(trace_dir)
    entries = []
    for path in files:
        entry = {"path": path, "rank": _rank(path), "sha256": _sha256(path)}
        if entry["rank"] is not None:
            entry.update(_stage_device_coverage(path))
        entries.append(entry)
    selected = None
    selection_reason = "requested_rank"
    rank_candidates = []
    if auto_select_rank:
        # EXTEND and DECODE are separate files. Keep the strongest DECODE
        # trace for each TP rank, then compare rank-local device coverage.
        by_rank = {}
        for entry in entries:
            rank = entry["rank"]
            if rank is None:
                continue
            # Generation is DECODE, or TARGET_VERIFY ("verify") when
            # speculative decoding is on.
            events_by_phase = entry.get("device_events_by_phase", {})
            duration_by_phase = entry.get(
                "annotation_duration_us_by_phase", {})
            score = (
                max(int(events_by_phase.get(p, 0))
                    for p in sglang_step_modes.GENERATION_PHASES),
                max(float(duration_by_phase.get(p, 0.0))
                    for p in sglang_step_modes.GENERATION_PHASES),
            )
            if score > by_rank.get(rank, ((-1, -1.0), None))[0]:
                by_rank[rank] = (score, entry)
        max_decode_events = max(
            (score[0] for score, _entry in by_rank.values()), default=0)
        for rank in sorted(by_rank):
            entry = by_rank[rank][1]
            device_events, duration_us = by_rank[rank][0]
            if not duration_us:
                eligible = False
                reason = "missing_decode_annotation"
            elif not device_events:
                eligible = False
                reason = "decode_annotation_has_no_device_events"
            elif device_events < max_decode_events:
                eligible = False
                reason = "decode_device_events_below_rank_maximum"
            else:
                eligible = True
                reason = "decode_device_events_match_rank_maximum"
            rank_candidates.append({
                "rank": rank,
                "decode_trace": entry["path"],
                "decode_device_events": device_events,
                "decode_duration_us": duration_us,
                "eligible": eligible,
                "reason": reason,
            })
        eligible_ranks = [item for item in rank_candidates
                          if item["eligible"] is True]
        if eligible_ranks:
            winner = max(eligible_ranks, key=lambda item: (
                item["decode_device_events"], -item["rank"]))
            selected = by_rank[winner["rank"]][1]
            analysis_rank = winner["rank"]
            selection_reason = "max_decode_step_device_coverage"
    else:
        selected = next(
            (e for e in entries if e["rank"] == analysis_rank), None)
    if selected is None and entries and not auto_select_rank:
        selected = entries[0]
        analysis_rank = selected["rank"]
        selection_reason = "first_available_trace"
    capabilities = inspect_trace(selected["path"]) if selected else {
        "event_count": 0,
        "categories": {},
        "device_event_count": 0,
        "capabilities": {},
        "error": "no top-level torch trace found",
    }
    speculative = _speculative_block(entries, server_log)
    observed_phases = {phase for entry in entries
                       for phase in entry.get("device_events_by_phase", {})}
    if speculative["enabled"]:
        observed_phases.add("verify")
    return {
        "schema_version": 1,
        "trace_dir": os.path.abspath(trace_dir),
        "trace_files": entries,
        "analysis_rank": analysis_rank,
        "analysis_rank_trace": selected["path"] if selected else "",
        "analysis_rank_sha256": selected["sha256"] if selected else "",
        "analysis_rank_selection_reason": selection_reason,
        "clean_trace_selection": selection_reason,
        "selected_analysis_rank": analysis_rank if selected else None,
        "rank_candidates": rank_candidates,
        "cross_rank_merge": False,
        "speculative_decoding": any(
            mode in sglang_step_modes.SPECULATIVE_MODES
            for entry in entries
            for mode in entry.get("step_mode_counts", {})),
        "decode_step_modes": sorted({
            mode for entry in entries
            for mode in entry.get("step_mode_counts", {})
            if sglang_step_modes.is_generation_phase(
                sglang_step_modes.PHASE_BY_MODE.get(mode))}),
        "generation_phase": sglang_step_modes.generation_phase(observed_phases),
        "target_phases": sglang_step_modes.target_phases(observed_phases),
        "speculative": speculative,
        "capability": capabilities,
        "status": "pass" if selected else "failed",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--analysis-rank", type=int, default=0)
    parser.add_argument("--auto-select-rank", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--server-log", default="",
        help="optional server log; only enriches speculative parameters")
    args = parser.parse_args()
    doc = build_manifest(
        args.trace_dir, args.analysis_rank,
        auto_select_rank=args.auto_select_rank,
        server_log=args.server_log or None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(args.out)
    return 0 if doc["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
