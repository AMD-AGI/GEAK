#!/usr/bin/env python3
"""Verify that an extracted callable is the live seam for a profiled GPU kernel.

The extractor marks every structured candidate with ``seam_trace`` and captures
the selected callable with ``capture_shapes``. ``seam_trace`` profiles a bounded
warmup/call window and emits install proof plus nested live-call markers. This tool
verifies, from machine evidence, that:

* the target is an exact import spec rather than a prose call chain;
* the target was called by the live server;
* the expected profiled GPU kernel ran while the target marker was active.
* no deeper marked live candidate launches the same kernel.

The JSON verdict is consumed by the workflow.  A rejected outer wrapper is not
selection success; the extractor must descend and repeat this probe until both
this contract and the binding contract pass.
"""

import argparse
import gzip
import json
import os
import re
import sys


MARKER_PREFIX = "GEAK_TARGET::"
INSTALL_PREFIX = "GEAK_INSTALLED::"
_SPEC_RE = re.compile(
    r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)


def valid_callable_spec(value):
    return bool(_SPEC_RE.fullmatch(str(value or "").strip()))


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def load_events(path):
    with _open(path) as fh:
        doc = json.load(fh)
    return doc.get("traceEvents", doc if isinstance(doc, list) else [])


def merge_process_traces(paths):
    """Merge call traces without allowing pid or External-id collisions across files."""
    merged = []
    for trace_index, path in enumerate(paths):
        prefix = f"trace-{trace_index}:"
        for original in load_events(path):
            if not isinstance(original, dict):
                continue
            event = dict(original)
            event["pid"] = prefix + str(original.get("pid"))
            event["tid"] = prefix + str(original.get("tid"))
            args = dict(original.get("args") or {})
            if args.get("External id") is not None:
                args["External id"] = prefix + str(args["External id"])
            if args.get("correlation") is not None:
                args["correlation"] = prefix + str(args["correlation"])
            event["args"] = args
            merged.append(event)
    return merged


def _strip_template_args(text):
    """Remove balanced ``<...>`` groups innermost-first until the symbol stops changing.

    One greedy pass over ``<.*>`` spans from the FIRST ``<`` to the LAST ``>``, so ``k<a>(t<b>)``
    loses the ``(`` that marks the end of the name, and a nested ``k<pair<a,b>>`` leaves the leftover
    delimiter behind. This must stay identical to ``canonicalDeviceKernel`` in e2e_workflow.js: the JS
    gate and this verdict compare the same two symbols, and a canonicalization that drifted between
    them would let a kernel pass one side and be refused by the other.
    """
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"<[^<>]*>", "", text)
    return text


def canonical_kernel_name(value):
    """Return a stable token for matching mangled/demangled kernel symbols."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\[clone[^\]]*\]", "", text)
    text = _strip_template_args(text)
    text = text.split("(", 1)[0]
    text = text.rsplit("::", 1)[-1]
    return re.sub(r"[^a-z0-9_]+", "", text)


def kernel_matches(expected, observed):
    want = canonical_kernel_name(expected)
    got = canonical_kernel_name(observed)
    if not want or not got:
        return False
    return want == got or (
        len(want) >= 6
        and re.search(r"(?:^|_)" + re.escape(want) + r"(?:$|_)", got) is not None
    )


def _device_projection(event):
    """True for the GPU-timeline copy of a host annotation (``gpu_user_annotation``).

    The profiler re-emits every ``record_function`` marker on the device timeline, where the span
    covers the kernels it launched rather than the Python call. Those projections do not preserve
    host call nesting -- a short device span for an OUTER seam routinely lands inside the device span
    of an INNER one -- so only host-side spans may establish nesting or launch causality.
    """
    return str(event.get("cat") or "").startswith("gpu_")


def _complete_spans(events, name):
    spans = []
    stacks = {}
    for event in events:
        if not isinstance(event, dict) or event.get("name") != name:
            continue
        if _device_projection(event):
            continue
        phase = event.get("ph", "X")
        if phase == "X" and event.get("ts") is not None:
            start = float(event["ts"])
            spans.append((start, start + float(event.get("dur") or 0), event))
        elif phase == "B":
            stacks.setdefault((event.get("pid"), event.get("tid")), []).append(event)
        elif phase == "E":
            key = (event.get("pid"), event.get("tid"))
            if stacks.get(key):
                begin = stacks[key].pop()
                start = float(begin.get("ts") or 0)
                spans.append((start, float(event.get("ts") or start), begin))
    return spans


def _within_any_span(event, spans, same_thread=False):
    if event.get("ts") is None:
        return False
    ts = float(event["ts"])
    return any(
        start <= ts <= end
        and (not same_thread or (
            event.get("pid") == marker.get("pid") and event.get("tid") == marker.get("tid")))
        for start, end, marker in spans
    )


def _marker_kernel_evidence(trace_events, target_callable, device_kernel):
    marker = MARKER_PREFIX + target_callable
    spans = _complete_spans(trace_events or [], marker)
    # Only CPU events on the marker's own thread can establish launch causality. Global timestamp
    # overlap is unsafe under concurrent serving. The resulting External ids bridge to async GPU events.
    related_external_ids = set()
    # Triton (and any raw hipModuleLaunchKernel) device rows carry only `correlation`, never an
    # `External id`, so the External-id bridge alone silently misses them. The host-side launch
    # runtime event inside the marker span carries BOTH ids, so its `correlation` is an equally
    # strict launch-causality bridge: same thread, inside the marker span, same launch.
    related_correlations = set()
    for event in trace_events or []:
        if not isinstance(event, dict) or not _within_any_span(event, spans, same_thread=True):
            continue
        if event.get("cat") == "kernel":
            continue
        args = event.get("args") or {}
        ext = args.get("External id")
        if ext is not None:
            related_external_ids.add(ext)
        corr = args.get("correlation")
        if corr is not None:
            related_correlations.add(corr)
    matched = []
    for event in trace_events or []:
        if not isinstance(event, dict) or event.get("cat") != "kernel":
            continue
        args = event.get("args") or {}
        ext = args.get("External id")
        corr = args.get("correlation")
        linked = (ext is not None and ext in related_external_ids) or (
            corr is not None and corr in related_correlations)
        if linked and kernel_matches(device_kernel, event.get("name")):
            matched.append(str(event.get("name") or ""))
    return {
        "target": target_callable,
        "marker": marker,
        "spans": spans,
        "related_external_ids": related_external_ids,
        "related_correlations": related_correlations,
        "matched": matched,
    }


def _is_nested(inner_spans, outer_spans):
    for inner_start, inner_end, inner_event in inner_spans:
        for outer_start, outer_end, outer_event in outer_spans:
            same_thread = (
                inner_event.get("pid") == outer_event.get("pid")
                and inner_event.get("tid") == outer_event.get("tid")
            )
            if same_thread and outer_start <= inner_start and inner_end <= outer_end and (
                    outer_start < inner_start or inner_end < outer_end):
                return True
    return False


def verify(target_callable, device_kernel, capture_meta, trace_events, candidate_targets=None):
    target_callable = str(target_callable or "").strip()
    device_kernel = str(device_kernel or "").strip()
    failed = []

    if not valid_callable_spec(target_callable):
        failed.append("invalid_target_callable")
    if not device_kernel:
        failed.append("missing_device_kernel")

    meta_target = ""
    observed_calls = 0
    if isinstance(capture_meta, dict):
        module = str(capture_meta.get("module") or "").strip()
        attr = str(capture_meta.get("attr") or "").strip()
        meta_target = f"{module}:{attr}" if module and attr else ""
        observed_calls = int(capture_meta.get("total_calls_observed") or 0)
    if meta_target != target_callable:
        failed.append("capture_target_mismatch")
    if observed_calls <= 0:
        failed.append("target_not_observed")

    candidates = []
    for candidate in list(candidate_targets or []) + [target_callable]:
        candidate = str(candidate or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    invalid_candidates = [candidate for candidate in candidates if not valid_callable_spec(candidate)]
    if invalid_candidates:
        failed.append("invalid_candidate_target")
    evidence = {
        candidate: _marker_kernel_evidence(trace_events, candidate, device_kernel)
        for candidate in candidates if valid_callable_spec(candidate)
    }
    installed_candidates = [
        candidate for candidate in candidates
        if _complete_spans(trace_events, INSTALL_PREFIX + candidate)
    ]
    missing_candidate_markers = sorted(set(candidates) - set(installed_candidates))
    if missing_candidate_markers:
        failed.append("candidate_marker_not_installed")
    selected_evidence = evidence.get(target_callable) or {
        "marker": MARKER_PREFIX + target_callable, "spans": [],
        "related_external_ids": set(), "related_correlations": set(), "matched": [],
    }
    marker = selected_evidence["marker"]
    spans = selected_evidence["spans"]
    if not spans:
        failed.append("target_marker_missing")

    related_external_ids = selected_evidence["related_external_ids"]
    matched = selected_evidence["matched"]
    if not matched:
        failed.append("device_kernel_not_under_target")
    deeper = [
        candidate for candidate, candidate_evidence in evidence.items()
        if candidate != target_callable and candidate_evidence["matched"]
        and _is_nested(candidate_evidence["spans"], spans)
    ]
    if deeper:
        failed.append("deeper_live_candidate_exists")
    # A candidate whose marker was installed and never fired is NOT evidence that the callable is
    # off the live path. Installation rebinds one module attribute; a callable reached through the
    # torch dispatcher (`torch.ops.*`) or through an alias a caller imported before installation
    # still runs with the wrapper bypassed, and is therefore invisible here. `deeper` can only see
    # what the markers saw, so report the unobserved candidates separately instead of letting them
    # be read as "not deeper": whoever holds the declared depths decides what an unobserved
    # declared-deeper candidate means.
    unobserved_candidates = sorted(
        candidate for candidate in installed_candidates
        if candidate != target_callable and not (evidence.get(candidate) or {}).get("spans"))
    deepest_verified = bool(
        spans and matched and not deeper and not invalid_candidates
        and not missing_candidate_markers)

    return {
        "contract": "kernel_selection",
        "ok": not failed,
        "target_callable": target_callable,
        "device_kernel": device_kernel,
        "capture_target": meta_target,
        "total_calls_observed": observed_calls,
        "target_marker": marker,
        "target_marker_calls": len(spans),
        "matched_kernel_calls": len(matched),
        "matched_kernel_names": sorted(set(matched)),
        "correlated_external_ids": len(related_external_ids),
        "correlated_launch_correlations": len(selected_evidence.get("related_correlations") or ()),
        "candidate_targets_tested": installed_candidates,
        "live_candidate_targets": sorted(
            candidate for candidate, candidate_evidence in evidence.items()
            if candidate_evidence["matched"]),
        "deeper_live_candidates": sorted(deeper),
        "missing_candidate_markers": missing_candidate_markers,
        "installed_but_never_live_candidates": unobserved_candidates,
        "deepest_verified": deepest_verified,
        "evidence": "installed+live_nested_candidate_markers+torch_profiler_external_id_or_launch_correlation",
        "failed": failed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="exact module:attr selected seam")
    parser.add_argument("--device-kernel", required=True, help="GPU kernel selected from profile")
    parser.add_argument("--capture-meta", required=True, nargs="+",
                        help="one or more process-local capture_shapes meta.json files")
    parser.add_argument("--torch-trace", required=True, nargs="+",
                        help="one or more process-local capture traces (.json or .json.gz)")
    parser.add_argument("--candidate-target", action="append", default=[],
                        help="exact module:attr candidate marked in the same trace; repeat for all candidates")
    parser.add_argument("--out", default="", help="optional verdict JSON path")
    args = parser.parse_args(argv)

    trace_paths = list(args.torch_trace)
    attempts = []
    for meta_path in args.capture_meta:
        with open(meta_path) as fh:
            meta = json.load(fh)
        capture_pid = str(meta.get("process_id") or "")
        matching_traces = trace_paths
        if capture_pid:
            matching_traces = [
                path for path in trace_paths
                if re.search(rf"\.pid-{re.escape(capture_pid)}(?:\.|$)",
                             os.path.basename(path))
            ]
        if not matching_traces:
            verdict = verify(
                args.target, args.device_kernel, meta, [], args.candidate_target)
            verdict["failed"].append("capture_process_trace_missing")
            verdict["ok"] = False
            verdict["deepest_verified"] = False
            attempts.append((meta_path, [], verdict))
            continue
        attempts.append((
            meta_path,
            matching_traces,
            verify(args.target, args.device_kernel, meta,
                   merge_process_traces(matching_traces), args.candidate_target),
        ))
    selected_meta_path, selected_paths, verdict = max(
        attempts,
        key=lambda item: (
            item[2]["matched_kernel_calls"],
            item[2]["target_marker_calls"],
            -len(item[2]["failed"]),
        ),
    )
    verdict = dict(verdict)
    all_verdicts = [item[2] for item in attempts]
    verdict["ok"] = all(item["ok"] for item in all_verdicts)
    verdict["deepest_verified"] = all(
        item["deepest_verified"] for item in all_verdicts)
    verdict["failed"] = sorted({
        failure for item in all_verdicts for failure in item["failed"]
    })
    verdict["live_candidate_targets"] = sorted({
        target for item in all_verdicts
        for target in item["live_candidate_targets"]
    })
    verdict["deeper_live_candidates"] = sorted({
        target for item in all_verdicts
        for target in item["deeper_live_candidates"]
    })
    verdict["missing_candidate_markers"] = sorted({
        target for item in all_verdicts
        for target in item["missing_candidate_markers"]
    })
    tested_sets = [
        set(item["candidate_targets_tested"]) for item in all_verdicts
    ]
    verdict["candidate_targets_tested"] = sorted(
        set.intersection(*tested_sets) if tested_sets else set())
    # Never-live has to hold on every capture process: one rank observing the candidate is enough
    # to make it live, so this intersects rather than unions.
    unobserved_sets = [
        set(item["installed_but_never_live_candidates"]) for item in all_verdicts
    ]
    verdict["installed_but_never_live_candidates"] = sorted(
        set.intersection(*unobserved_sets) if unobserved_sets else set())
    for field in (
            "total_calls_observed", "target_marker_calls", "matched_kernel_calls",
            "correlated_external_ids", "correlated_launch_correlations"):
        verdict[field] = sum(int(item.get(field) or 0) for item in all_verdicts)
    verdict["matched_kernel_names"] = sorted({
        name for item in all_verdicts for name in item["matched_kernel_names"]
    })
    verdict["process_verdicts"] = [
        {
            "capture_meta_file": meta_path,
            "trace_files": paths,
            "ok": item["ok"],
            "failed": item["failed"],
            "live_candidate_targets": item["live_candidate_targets"],
            "deeper_live_candidates": item["deeper_live_candidates"],
        }
        for meta_path, paths, item in attempts
    ]
    selected_paths = [path for _, paths, _ in attempts for path in paths]
    verdict["capture_meta_file"] = selected_meta_path
    verdict["trace_file"] = selected_paths[0] if len(selected_paths) == 1 else ""
    verdict["trace_files"] = selected_paths
    verdict["trace_files_considered"] = trace_paths
    payload = json.dumps(verdict, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(payload + "\n")
    print(payload)
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
