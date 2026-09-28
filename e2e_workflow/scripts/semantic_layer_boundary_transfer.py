#!/usr/bin/env python3
"""Transfer all-layer graph-construction scopes onto a Clean Trace.

The graph-construction replay executes Python once and therefore can emit one
``GEAK_LAYER_SCOPE`` record_function range around every main decoder layer.
The production Clean Trace remains the sole timing/order source; this module
copies only the cuts between layer bodies.

Transfer is deliberately fail-closed.  A donor pass must contain exactly the
configured layer order ``0..N-1``.  The mapper first requires the complete donor
device sequence to occur as one exact, contiguous normalized sequence inside a
workload-compatible Clean Trace step.  When graph construction and graph replay
expose different backend events, it may instead use an exact stable projection:
retain only raw identities with equal total multiplicity on both sides, require
the complete projected sequences to be identical, and require multiple distinct
anchors in every layer plus majority coverage.  A one-sided unmatched internal
gap follows the marker-proven side.  A two-sided or otherwise unsupported gap
remains an explicit inter-layer residual while both proven layer cores remain
eligible as representatives.  Outer prefix/suffix work remains global. Stage
names, model names, attention kinds and recurring subsequences are never used to
invent a boundary.
"""
import argparse
import bisect
import collections
import hashlib
import json
import os

import semantic_kernel_mapping


MARKER_PREFIX = "GEAK_LAYER_SCOPE|"
RUNTIME_CATEGORIES = {"cuda_runtime", "hip_runtime"}
STABLE_MIN_ANCHOR_EVENTS_PER_LAYER = 2
STABLE_MIN_DISTINCT_IDENTITIES_PER_LAYER = 2
STABLE_MIN_DONOR_EVENT_FRACTION = 0.5
STABLE_MIN_RECIPIENT_BODY_FRACTION = 0.5


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase(value):
    value = str(value or "").strip().lower()
    return {
        "extend": "prefill", "prompt": "prefill",
        "generation": "decode",
    }.get(value, value)


def _integer(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _marker_fields(name):
    fields = {}
    for token in str(name).split("|")[1:]:
        key, separator, value = token.partition("=")
        if separator:
            fields[key] = value
    return fields


def _kernel_identity(value, event_type=None):
    """Normalize launch-geometry decorations, not semantic operator names."""
    return semantic_kernel_mapping._boundary_identity(value, event_type)


def _sequence_sha(keys):
    payload = json.dumps(list(keys), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _scope_markers(events):
    markers = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        name = str(event.get("name", ""))
        if (event.get("cat") != "user_annotation"
                or not name.startswith(MARKER_PREFIX)
                or event.get("ts") is None or event.get("dur") is None):
            continue
        fields = _marker_fields(name)
        markers.append({
            "event_index": event_index,
            "name": name,
            "phase": _phase(fields.get("phase")),
            "batch_size": _integer(fields.get("bs")),
            "input_tokens": _integer(fields.get("toks")),
            "layer_id": _integer(fields.get("layer")),
            "op_path": fields.get("path"),
            "pid": event.get("pid"),
            "tid": event.get("tid"),
            "ts": float(event["ts"]),
            "end": float(event["ts"]) + float(event["dur"]),
            "entries": [],
        })
    markers.sort(key=lambda item: (item["event_index"], item["ts"]))
    return markers


def _attach_device_entries(events, markers):
    """Attach correlated device launches to their enclosing layer scope."""
    by_thread = {}
    for marker in markers:
        by_thread.setdefault((marker["pid"], marker["tid"]), []).append(marker)
    starts = {}
    for key, values in by_thread.items():
        values.sort(key=lambda item: (item["ts"], item["end"]))
        starts[key] = [item["ts"] for item in values]

    device_by_correlation = {}
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if event.get("cat") not in semantic_kernel_mapping.DEVICE_CATEGORIES:
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation", args.get("Correlation ID"))
        if correlation is not None:
            device_by_correlation.setdefault(correlation, []).append(
                (event_index, event))

    attached = set()
    for runtime_index, event in enumerate(events):
        if not isinstance(event, dict) or event.get("cat") not in RUNTIME_CATEGORIES:
            continue
        if event.get("ts") is None:
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation", args.get("Correlation ID"))
        device_matches = device_by_correlation.get(correlation, [])
        if correlation is None or len(device_matches) != 1:
            continue
        key = (event.get("pid"), event.get("tid"))
        candidates = by_thread.get(key, [])
        if not candidates:
            continue
        timestamp = float(event["ts"])
        position = bisect.bisect_right(starts[key], timestamp) - 1
        if position < 0:
            continue
        marker = candidates[position]
        if not (marker["ts"] <= timestamp <= marker["end"]):
            continue
        device_index, device = device_matches[0]
        attachment = (marker["event_index"], device_index)
        if attachment in attached:
            continue
        attached.add(attachment)
        raw_name = str(device.get("name") or args.get("kernel") or "")
        if not raw_name:
            continue
        marker["entries"].append({
            "runtime_event_index": runtime_index,
            "device_event_index": device_index,
            "raw_name": raw_name,
            "identity": _kernel_identity(raw_name, device.get("cat")),
            "event_type": device.get("cat"),
        })
    for marker in markers:
        marker["entries"].sort(key=lambda item: (
            item["runtime_event_index"], item["device_event_index"]))


def _complete_donor_passes(events, expected_layers):
    markers = _scope_markers(events)
    _attach_device_entries(events, markers)
    passes = []
    by_bucket = {}
    for marker in markers:
        key = (marker["phase"], marker["batch_size"], marker["input_tokens"])
        by_bucket.setdefault(key, []).append(marker)
    for bucket, values in sorted(by_bucket.items()):
        current = []
        for marker in values:
            layer_id = marker["layer_id"]
            if layer_id == 0:
                current = [marker]
            elif current and layer_id == len(current):
                current.append(marker)
            else:
                current = []
            if len(current) == expected_layers:
                if all(item["entries"] for item in current):
                    entries = []
                    layer_starts = []
                    for item in current:
                        layer_starts.append(len(entries))
                        entries.extend(item["entries"])
                    passes.append({
                        "phase": bucket[0],
                        "batch_size": bucket[1],
                        "input_tokens": bucket[2],
                        "markers": list(current),
                        "entries": entries,
                        "layer_starts": layer_starts,
                        "sequence": [item["identity"] for item in entries],
                    })
                current = []
    return markers, passes


def _complete_authoritative_step(step_rows, expected_layers):
    groups = {}
    for row in step_rows:
        instance_id = row.get("layer_instance_id")
        evidence = str(row.get("layer_evidence") or "")
        if instance_id and evidence.startswith("python_module_span"):
            groups.setdefault(instance_id, []).append(row)
    ordered = sorted(groups.values(), key=lambda group: min(
        row["device_seq_index"] for row in group))
    return ([group[0].get("layer_id") for group in ordered]
            == list(range(expected_layers)))


def _bucket_matches(donor, phase, batch_size, input_tokens):
    if donor["phase"] != phase or donor["batch_size"] != batch_size:
        return False
    if phase == "decode":
        return True
    return (input_tokens < 0 or donor["input_tokens"] < 0
            or donor["input_tokens"] == input_tokens)


def _subsequence_starts(sequence, needle):
    """KMP exact-subsequence search; returns every contiguous match start."""
    if not needle or len(needle) > len(sequence):
        return []
    prefix = [0] * len(needle)
    length = 0
    for index in range(1, len(needle)):
        while length and needle[index] != needle[length]:
            length = prefix[length - 1]
        if needle[index] == needle[length]:
            length += 1
            prefix[index] = length
    starts = []
    length = 0
    for index, value in enumerate(sequence):
        while length and value != needle[length]:
            length = prefix[length - 1]
        if value == needle[length]:
            length += 1
            if length == len(needle):
                starts.append(index - len(needle) + 1)
                length = prefix[length - 1]
    return starts


def _stable_projection_map(sequence, donor, expected_layers):
    """Map a marker-labelled donor when graph capture omits replay kernels.

    HIP/CUDA graph construction and graph replay do not necessarily expose the
    same device events to the profiler.  In particular, a collective recorded
    into a graph may appear only when the graph is replayed, while temporary
    fill/memset work may appear only during construction.  Requiring the full
    raw sequences to be identical therefore rejects a valid donor.

    This fallback is still exact rather than fuzzy:

    * retain only raw kernel identities whose total multiplicity is identical
      in donor and recipient;
    * require the two complete projected sequences to be byte-for-byte equal;
    * require multiple distinct matched anchors inside every marker-labelled
      layer and majority event coverage on both sides;
    * place an unmatched boundary gap on one side only when the donor marker
      proves that only that side has unmatched boundary-local events.

    There is no LCS, similarity score, stage taxonomy, model name, kernel-name
    allow-list, proportional partition, or best-effort layer completion here.
    """
    donor_sequence = list(donor.get("sequence") or [])
    donor_counts = collections.Counter(donor_sequence)
    recipient_counts = collections.Counter(sequence)
    stable_identities = {
        identity for identity, count in donor_counts.items()
        if count > 0 and recipient_counts.get(identity) == count
    }
    donor_projection = [
        (identity, position)
        for position, identity in enumerate(donor_sequence)
        if identity in stable_identities
    ]
    recipient_projection = [
        (identity, position)
        for position, identity in enumerate(sequence)
        if identity in stable_identities
    ]
    donor_values = [item[0] for item in donor_projection]
    recipient_values = [item[0] for item in recipient_projection]
    if not donor_values or donor_values != recipient_values:
        return None, {
            "reason": "stable_identity_projection_mismatch",
            "stable_identity_count": len(stable_identities),
            "donor_projected_event_count": len(donor_values),
            "recipient_projected_event_count": len(recipient_values),
        }

    layer_starts = [int(value) for value in donor.get(
        "layer_starts", [])]
    if (len(layer_starts) != expected_layers
            or layer_starts != sorted(layer_starts)
            or len(set(layer_starts)) != expected_layers):
        return None, {"reason": "donor_layer_starts_invalid"}

    donor_anchor_positions = [[] for _ in range(expected_layers)]
    recipient_anchor_positions = [[] for _ in range(expected_layers)]
    donor_anchor_identities = [set() for _ in range(expected_layers)]
    for (identity, donor_position), (_, recipient_position) in zip(
            donor_projection, recipient_projection):
        layer_id = bisect.bisect_right(
            layer_starts, donor_position) - 1
        if layer_id < 0 or layer_id >= expected_layers:
            return None, {
                "reason": "stable_anchor_outside_donor_layer_pass",
                "donor_position": donor_position,
            }
        donor_anchor_positions[layer_id].append(donor_position)
        recipient_anchor_positions[layer_id].append(recipient_position)
        donor_anchor_identities[layer_id].add(identity)

    weak_layers = []
    for layer_id in range(expected_layers):
        event_count = len(donor_anchor_positions[layer_id])
        identity_count = len(donor_anchor_identities[layer_id])
        if (event_count < STABLE_MIN_ANCHOR_EVENTS_PER_LAYER
                or identity_count < STABLE_MIN_DISTINCT_IDENTITIES_PER_LAYER):
            weak_layers.append({
                "layer_id": layer_id,
                "stable_anchor_event_count": event_count,
                "stable_anchor_identity_count": identity_count,
            })
    if weak_layers:
        return None, {
            "reason": "insufficient_stable_anchors_per_layer",
            "weak_layers": weak_layers,
        }

    if any(
            recipient_anchor_positions[layer_id][-1]
            >= recipient_anchor_positions[layer_id + 1][0]
            for layer_id in range(expected_layers - 1)):
        return None, {"reason": "stable_anchor_layer_order_overlap"}

    first_recipient_anchor = recipient_anchor_positions[0][0]
    last_recipient_anchor = recipient_anchor_positions[-1][-1]
    donor_fraction = float(len(donor_values)) / max(1, len(donor_sequence))
    recipient_body_span = last_recipient_anchor - first_recipient_anchor + 1
    recipient_fraction = float(len(recipient_values)) / max(
        1, recipient_body_span)
    if (donor_fraction < STABLE_MIN_DONOR_EVENT_FRACTION
            or recipient_fraction < STABLE_MIN_RECIPIENT_BODY_FRACTION):
        return None, {
            "reason": "stable_projection_coverage_too_low",
            "donor_event_fraction": round(donor_fraction, 6),
            "recipient_body_fraction": round(recipient_fraction, 6),
        }

    starts = [None] * expected_layers
    ends = [None] * expected_layers
    starts[0] = first_recipient_anchor
    boundary_gaps = []
    residual_ranges = []
    for layer_id in range(1, expected_layers):
        previous_last_donor = donor_anchor_positions[layer_id - 1][-1]
        current_first_donor = donor_anchor_positions[layer_id][0]
        previous_donor_stop = layer_starts[layer_id]
        current_donor_start = layer_starts[layer_id]
        previous_unstable_suffix = (
            previous_donor_stop - previous_last_donor - 1)
        current_unstable_prefix = (
            current_first_donor - current_donor_start)

        previous_last_recipient = recipient_anchor_positions[
            layer_id - 1][-1]
        current_first_recipient = recipient_anchor_positions[layer_id][0]
        recipient_gap = current_first_recipient - previous_last_recipient - 1
        if recipient_gap < 0:
            return None, {"reason": "negative_recipient_boundary_gap"}
        if recipient_gap == 0:
            start = current_first_recipient
            previous_end = start
            side = "none"
        elif previous_unstable_suffix and current_unstable_prefix:
            # Both marker-labelled donor layers contain boundary-local events
            # that did not survive the strict stable projection.  The old
            # implementation rejected the complete 0..N-1 Decode pass here,
            # even when every layer still had a large, ordered stable core.
            # Preserve the two proven cores and leave only the recipient gap
            # unassigned.  The consumer records it as transition_global.  The
            # gap does not invalidate either independently anchored layer core.
            previous_end = previous_last_recipient + 1
            start = current_first_recipient
            side = "inter_layer_residual"
            residual_ranges.append({
                "layer_boundary": [layer_id - 1, layer_id],
                "start_position": previous_end,
                "end_position": start,
                "previous_donor_unstable_suffix": previous_unstable_suffix,
                "current_donor_unstable_prefix": current_unstable_prefix,
                "recipient_unmatched_gap": recipient_gap,
                "previous_donor_suffix_identities": donor_sequence[
                    previous_last_donor + 1:previous_donor_stop],
                "current_donor_prefix_identities": donor_sequence[
                    current_donor_start:current_first_donor],
                "recipient_gap_identities": sequence[
                    previous_end:start],
                "reason": "ambiguous_unmatched_events_on_both_boundary_sides",
            })
        elif current_unstable_prefix:
            # The donor marker proves its unmatched boundary-local work is a
            # prefix of the current layer, so include the recipient gap there.
            start = previous_last_recipient + 1
            previous_end = start
            side = "current_layer_prefix"
        elif previous_unstable_suffix:
            start = current_first_recipient
            previous_end = start
            side = "previous_layer_suffix"
        else:
            # The gap is real production work, but neither donor layer contains
            # a corresponding unstable edge.  Keep it explicit and global
            # instead of fabricating ownership or deleting the whole phase.
            previous_end = previous_last_recipient + 1
            start = current_first_recipient
            side = "inter_layer_residual"
            residual_ranges.append({
                "layer_boundary": [layer_id - 1, layer_id],
                "start_position": previous_end,
                "end_position": start,
                "previous_donor_unstable_suffix": 0,
                "current_donor_unstable_prefix": 0,
                "recipient_unmatched_gap": recipient_gap,
                "previous_donor_suffix_identities": [],
                "current_donor_prefix_identities": [],
                "recipient_gap_identities": sequence[
                    previous_end:start],
                "reason": "recipient_boundary_gap_has_no_donor_side_evidence",
            })
        ends[layer_id - 1] = previous_end
        starts[layer_id] = start
        boundary_gaps.append({
            "layer_boundary": [layer_id - 1, layer_id],
            "recipient_unmatched_gap": recipient_gap,
            "assigned_side": side,
            "previous_donor_unstable_suffix": previous_unstable_suffix,
            "current_donor_unstable_prefix": current_unstable_prefix,
        })

    donor_last_suffix = (
        len(donor_sequence) - donor_anchor_positions[-1][-1] - 1)
    recipient_suffix = len(sequence) - last_recipient_anchor - 1
    # Outer unmatched regions are intentionally not forced into L0/LN-1.
    # They may be graph-construction scratch work, model setup, or terminal
    # postprocessing.  Keeping them global is conservative and prevents the
    # exact first/final-layer contamination this transfer is meant to remove.
    end = last_recipient_anchor + 1
    ends[-1] = end
    layer_ranges = [{
        "layer_id": layer_id,
        "start_position": starts[layer_id],
        "end_position": ends[layer_id],
        # An unresolved inter-layer transition is deliberately excluded from
        # both layer bodies.  It therefore does not make either stable body an
        # invalid representative.  Representative eligibility is revoked only
        # when the layer body itself cannot be mapped authoritatively.
        "representative_eligible": True,
    } for layer_id in range(expected_layers)]
    widths = [item["end_position"] - item["start_position"]
              for item in layer_ranges]
    if any(width <= 0 for width in widths):
        return None, {
            "reason": "non_positive_stable_projection_layer_width",
            "layer_widths": widths,
        }

    return {
        "body_start_position": starts[0],
        "body_end_position": end,
        "layer_start_positions": starts,
        "layer_widths": widths,
        "layer_ranges": layer_ranges,
        "residual_ranges": residual_ranges,
        "prefix_row_count": starts[0],
        "suffix_row_count": len(sequence) - end,
        "match_rule": (
            "exact_equal_multiplicity_stable_identity_projection_with_residuals"
            if residual_ranges else
            "exact_equal_multiplicity_stable_identity_projection"),
        "stable_projection": {
            "stable_identity_count": len(stable_identities),
            "stable_event_count": len(donor_values),
            "donor_event_fraction": round(donor_fraction, 6),
            "recipient_body_fraction": round(recipient_fraction, 6),
            "projected_sequence_sha256": _sequence_sha(donor_values),
            "per_layer_anchor_event_counts": [
                len(values) for values in donor_anchor_positions],
            "per_layer_anchor_identity_counts": [
                len(values) for values in donor_anchor_identities],
            "boundary_gaps": boundary_gaps,
            "residual_boundary_count": len(residual_ranges),
            "first_layer_donor_unstable_prefix": (
                donor_anchor_positions[0][0] - layer_starts[0]),
            "last_layer_donor_unstable_suffix": donor_last_suffix,
            "recipient_unmatched_suffix_kept_global": recipient_suffix,
        },
    }, None


def _map_step(step_rows, donor_passes, expected_layers):
    sequence = [
        _kernel_identity(row.get("raw_name"), row.get("event_type"))
        for row in step_rows]
    phase = _phase(step_rows[0].get("phase"))
    batch_size = _integer(step_rows[0].get("step_batch_size"))
    input_tokens = _integer(step_rows[0].get("step_input_tokens"))
    mappings = {}
    compatible_donors = []
    considered = 0
    for donor in donor_passes:
        if not _bucket_matches(
                donor, phase, batch_size, input_tokens):
            continue
        considered += 1
        compatible_donors.append(donor)
        for match_start in _subsequence_starts(sequence, donor["sequence"]):
            starts = tuple(
                match_start + position for position in donor["layer_starts"])
            end = match_start + len(donor["sequence"])
            mappings[(starts, end)] = donor
    if len(mappings) > 1:
        return None, {
            "reason": "ambiguous_exact_contiguous_donor_sequence",
            "compatible_donor_pass_count": considered,
            "exact_match_count": len(mappings),
            "recipient_sequence_sha256": _sequence_sha(sequence),
        }
    if len(mappings) == 1:
        (starts, end), donor = next(iter(mappings.items()))
        widths = [
            (starts[index + 1] if index + 1 < expected_layers else end)
            - starts[index]
            for index in range(expected_layers)]
        if len(starts) != expected_layers or any(width <= 0 for width in widths):
            return None, {
                "reason": "non_positive_or_incomplete_layer_width",
                "layer_widths": widths,
            }
        return {
            "phase": phase,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
            "recipient_step_id": step_rows[0].get("step_id"),
            "recipient_row_count": len(step_rows),
            "recipient_sequence_sha256": _sequence_sha(sequence),
            "body_start_position": starts[0],
            "body_end_position": end,
            "layer_start_positions": list(starts),
            "layer_widths": widths,
            "layer_ranges": [{
                "layer_id": layer_id,
                "start_position": starts[layer_id],
                "end_position": (
                    starts[layer_id + 1]
                    if layer_id + 1 < expected_layers else end),
                "representative_eligible": True,
            } for layer_id in range(expected_layers)],
            "residual_ranges": [],
            "prefix_row_count": starts[0],
            "suffix_row_count": len(step_rows) - end,
            "donor": {
                "phase": donor["phase"],
                "batch_size": donor["batch_size"],
                "input_tokens": donor["input_tokens"],
                "sequence_sha256": _sequence_sha(donor["sequence"]),
                "event_count": len(donor["sequence"]),
            },
            "match_rule": "exact_contiguous_normalized_device_sequence",
        }, None

    stable_mappings = {}
    stable_failures = []
    for donor in compatible_donors:
        stable, failure = _stable_projection_map(
            sequence, donor, expected_layers)
        if stable is None:
            stable_failures.append({
                "donor_phase": donor["phase"],
                "donor_batch_size": donor["batch_size"],
                "donor_input_tokens": donor["input_tokens"],
                **(failure or {}),
            })
            continue
        key = tuple(
            (item["start_position"], item["end_position"])
            for item in stable.get("layer_ranges", []))
        stable_mappings[key] = (stable, donor)
    if len(stable_mappings) != 1:
        return None, {
            "reason": (
                "no_validated_stable_projection"
                if not stable_mappings
                else "ambiguous_validated_stable_projection"),
            "compatible_donor_pass_count": considered,
            "exact_match_count": 0,
            "stable_projection_count": len(stable_mappings),
            "stable_projection_failures": stable_failures,
            "recipient_sequence_sha256": _sequence_sha(sequence),
        }

    stable, donor = next(iter(stable_mappings.values()))
    stable.update({
        "phase": phase,
        "batch_size": batch_size,
        "input_tokens": input_tokens,
        "recipient_step_id": step_rows[0].get("step_id"),
        "recipient_row_count": len(step_rows),
        "recipient_sequence_sha256": _sequence_sha(sequence),
        "donor": {
            "phase": donor["phase"],
            "batch_size": donor["batch_size"],
            "input_tokens": donor["input_tokens"],
            "sequence_sha256": _sequence_sha(donor["sequence"]),
            "event_count": len(donor["sequence"]),
        },
    })
    return stable, None


def transfer(donor_trace, recipient_traces, pattern_path, out_path=""):
    recipient_traces, adopted_siblings = (
        semantic_kernel_mapping._resolve_trace_paths(
            recipient_traces, auto_sibling=True))
    with open(pattern_path) as fh:
        pattern_doc = json.load(fh)
    expected_layers = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)
    if expected_layers <= 0:
        raise ValueError("structural patterns have no positive main layer count")

    donor_events = semantic_kernel_mapping._load_events(donor_trace)
    markers, donor_passes = _complete_donor_passes(
        donor_events, expected_layers)
    recipient_events = semantic_kernel_mapping._load_events_multi(
        recipient_traces)
    rows, _, _, _, _ = semantic_kernel_mapping._event_rows(
        recipient_events, pattern_doc)
    by_step = {}
    for row in rows:
        if row.get("step_id"):
            by_step.setdefault(row["step_id"], []).append(row)

    groups = []
    failures = []
    skipped_authoritative = []
    for step_id, step_rows in sorted(
            by_step.items(), key=lambda item: min(
                row["device_seq_index"] for row in item[1])):
        step_rows.sort(key=lambda row: row["device_seq_index"])
        if _complete_authoritative_step(step_rows, expected_layers):
            skipped_authoritative.append(step_id)
            continue
        group, failure = _map_step(
            step_rows, donor_passes, expected_layers)
        if group is None:
            failures.append({
                "step_id": step_id,
                "phase": step_rows[0].get("phase"),
                "batch_size": step_rows[0].get("step_batch_size"),
                "input_tokens": step_rows[0].get("step_input_tokens"),
                **(failure or {}),
            })
        else:
            groups.append(group)

    if not markers:
        failures.append({
            "reason": "donor_has_no_geak_layer_scope_markers"})
    elif not donor_passes:
        failures.append({
            "reason": "donor_has_no_complete_nonempty_layer_pass",
            "marker_count": len(markers),
            "expected_layers": expected_layers,
        })
    has_residuals = any(
        group.get("residual_ranges") for group in groups)
    status = (
        "fail" if failures or not groups else
        "partial" if has_residuals else
        "pass")
    if skipped_authoritative and not groups and not failures:
        status = "not_needed"
    document = {
        "schema_version": 2,
        "transfer": "graph_construction_main_layer_boundaries",
        "status": status,
        "evidence_policy": (
            "Boundary cuts only; Clean Trace rows, order, timestamps and "
            "durations are never copied or replaced."),
        "expected_main_layers": expected_layers,
        "patterns": {
            "path": os.path.abspath(pattern_path),
            "sha256": _sha256(pattern_path),
        },
        "donor": {
            "path": os.path.abspath(donor_trace),
            "sha256": _sha256(donor_trace),
            "layer_scope_marker_count": len(markers),
            "complete_pass_count": len(donor_passes),
            "buckets": sorted({
                "%s|bs=%s|toks=%s" % (
                    item["phase"], item["batch_size"], item["input_tokens"])
                for item in donor_passes}),
        },
        "recipient": {
            "traces": [{
                "path": os.path.abspath(path), "sha256": _sha256(path),
            } for path in recipient_traces],
            "adopted_phase_siblings": adopted_siblings,
            "step_count": len(by_step),
        },
        "mapped_groups": groups,
        "mapped_step_count": len(groups),
        "residual_range_count": sum(
            len(group.get("residual_ranges") or []) for group in groups),
        "skipped_authoritative_steps": skipped_authoritative,
        "failures": failures,
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(document, fh, indent=2)
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-trace", required=True)
    parser.add_argument("--recipient-trace", action="append", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = transfer(
        args.donor_trace, args.recipient_trace, args.patterns, args.out)
    print(json.dumps({
        "status": result["status"],
        "mapped_step_count": result["mapped_step_count"],
        "failures": result["failures"],
    }, indent=2))
    return 0 if result["status"] in (
        "pass", "partial", "not_needed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
