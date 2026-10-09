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
the complete projected sequences to be identical -- or, when concurrent streams
reorder work inside a layer, each donor layer's projected multiset to occupy the
matching consecutive chunk of the recipient projection -- and require multiple
distinct anchors in every layer plus majority coverage.  A one-sided unmatched internal
gap follows the marker-proven side.  A two-sided or otherwise unsupported gap
remains an explicit inter-layer residual while both proven layer cores remain
eligible as representatives.  Outer prefix/suffix work remains global.  A step
no donor pass validates stays unresolved and makes the map ``partial``; it does
not void the steps that did transfer.  Stage
names, model names, attention kinds and recurring subsequences are never used to
invent a boundary.

Preferred donor: the graph-capture pass (``src=capture`` markers).  Graph
capture launches no device work, so its kernel identities come from the
runtime launch records (``args.kernel``), and they are exactly what graph
replay -- the Clean Trace -- executes.  Eager warmups may additionally run
one-shot work (overlay self-checks, lazy init) that never reaches the graph.
A capture donor is matched as an ordered subsequence of the Clean Trace step:
Clean-Trace-only kernels (input preparation, overlapped scheduler work) are
allowed anywhere, and the eager-warmup rules above remain the fallback.
"""
import argparse
import bisect
import collections
import hashlib
import json
import os

import semantic_kernel_mapping
import sglang_step_modes


MARKER_PREFIX = "GEAK_LAYER_SCOPE|"
RUNTIME_CATEGORIES = {"cuda_runtime", "hip_runtime"}
STABLE_MIN_ANCHOR_EVENTS_PER_LAYER = 2
STABLE_MIN_DISTINCT_IDENTITIES_PER_LAYER = 2
STABLE_MIN_DONOR_EVENT_FRACTION = 0.5
STABLE_MIN_RECIPIENT_BODY_FRACTION = 0.5
STABLE_RULE = "exact_equal_multiplicity_stable_identity_projection"
STABLE_PER_LAYER_RULE = "equal_multiplicity_stable_identity_per_layer_multiset"
# Capture donor: share of its kernels that must be found, in order, in the
# Clean Trace step, and how far ahead one missing kernel may be searched for
# before it is skipped instead of consuming later work.
CAPTURE_MIN_MATCHED_FRACTION = 0.9
CAPTURE_LOOKAHEAD = 64


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
        # Speculative decoding: the target model's generation step.
        "target_verify": "verify",
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
            "source": fields.get("src") or "eager",
            "op_path": fields.get("path"),
            "pid": event.get("pid"),
            "tid": event.get("tid"),
            "ts": float(event["ts"]),
            "end": float(event["ts"]) + float(event["dur"]),
            "entries": [],
        })
    markers.sort(key=lambda item: (item["event_index"], item["ts"]))
    return markers


def _dispatch_scope_markers(events, pattern_doc):
    """Donor layer scopes from the Pattern-declared dispatch ops.

    Used when the donor carries no GEAK_LAYER_SCOPE markers. On vLLM the module
    hooks that emit those markers cannot run inside torch.compile, but a capture
    with CUDA-graph replay off walks the model on every step, so each layer's
    declared dispatch op is present and semantic_kernel_mapping cuts exact
    per-layer scopes from them. They are shaped like markers so every matching
    rule below applies unchanged.
    """
    spans = semantic_kernel_mapping._collect_step_spans(events)
    scopes, _ = semantic_kernel_mapping._dispatch_anchor_scopes(
        events, spans, pattern_doc)
    markers = []
    for index, scope in enumerate(scopes):
        markers.append({
            "event_index": index,
            "name": scope["name"],
            "phase": _phase(scope.get("phase")),
            "batch_size": _integer(scope.get("batch_size")),
            "input_tokens": _integer(scope.get("input_tokens")),
            "layer_id": scope["layer_id"],
            # A dispatch-op donor walks the model eagerly (graph replay off).
            "source": "eager",
            "op_path": scope["name"],
            "pid": scope.get("pid"),
            "tid": scope.get("tid"),
            "ts": float(scope["ts"]),
            "end": float(scope["end"]),
            "entries": [],
        })
    return markers


def _attach_device_entries(events, markers):
    """Attach the kernels launched inside each layer scope.

    Eager scopes take the correlated device kernel.  Capture scopes take the
    kernel named by the launch record itself: nothing executes on the device
    while a graph is being recorded.
    """
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
        if marker["source"] == "capture":
            # A launch recorded into a graph runs nothing yet: the runtime
            # event names its kernel and has no device row.
            raw_name = str(args.get("kernel") or "")
            if device_matches or not raw_name:
                continue
            marker["entries"].append({
                "runtime_event_index": runtime_index,
                "device_event_index": -1,
                "raw_name": raw_name,
                "identity": _kernel_identity(raw_name, "kernel"),
                "event_type": "kernel",
            })
            continue
        if correlation is None or len(device_matches) != 1:
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


def _complete_donor_passes(events, expected_layers, markers=None):
    if markers is None:
        markers = _scope_markers(events)
    _attach_device_entries(events, markers)
    passes = []
    by_bucket = {}
    for marker in markers:
        key = (marker["phase"], marker["batch_size"], marker["input_tokens"],
               marker["source"])
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
                        "source": bucket[3],
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
        if instance_id and (evidence.startswith("python_module_span")
                            or evidence == "declared_dispatch_op_span"):
            groups.setdefault(instance_id, []).append(row)
    ordered = sorted(groups.values(), key=lambda group: min(
        row["device_seq_index"] for row in group))
    return ([group[0].get("layer_id") for group in ordered]
            == list(range(expected_layers)))


def _bucket_matches(donor, phase, batch_size, input_tokens):
    if donor["phase"] != phase or donor["batch_size"] != batch_size:
        return False
    if sglang_step_modes.is_generation_phase(phase):
        return True
    return (input_tokens < 0 or donor["input_tokens"] < 0
            or donor["input_tokens"] == input_tokens)


def _compatible_donors(donor_passes, phase, batch_size, input_tokens):
    """Donor passes for one clean step.

    Generation (decode/verify) steps replay the smallest captured CUDA graph
    whose batch is >= the live batch, so a bs=9 step runs the bs=10 graph.
    Match exactly first, then that padded bucket.  Prefill stays exact.
    """
    exact = [donor for donor in donor_passes
             if _bucket_matches(donor, phase, batch_size, input_tokens)]
    if exact or not sglang_step_modes.is_generation_phase(phase):
        return exact
    larger = sorted({donor["batch_size"] for donor in donor_passes
                     if donor["phase"] == phase
                     and donor["batch_size"] > batch_size})
    if not larger:
        return []
    return [donor for donor in donor_passes
            if _bucket_matches(donor, phase, larger[0], input_tokens)]


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


def stable_projection_pairs(sequence, donor_sequence, layer_starts):
    """Pair the equal-multiplicity projections of a recipient step and a donor pass.

    Returns ([(identity, donor_position, recipient_position)], info) or
    (None, failure). The projections pair in order when they are byte-equal.
    Otherwise they still pair when the recipient projection, cut into consecutive
    chunks sized by the donor's per-layer stable counts, holds exactly each donor
    layer's multiset; inside a chunk the k-th occurrence of an identity pairs with
    the k-th. That admits reordering WITHIN a layer and nothing across one.

    Why: a model that runs two streams concurrently inside a layer (MiniMax-M3
    TP8, default config: routed experts on one stream, the shared expert on
    another) interleaves them by device timing, and eager and graph replay place
    the work on different streams, so neither the full order nor a per-stream
    order is reproducible between donor and recipient -- but each layer's kernel
    multiset is. Measured: 0/13 decode steps byte-equal, 12/12 per-layer equal.
    """
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
    info = {
        "stable_identity_count": len(stable_identities),
        "donor_projected_event_count": len(donor_projection),
        "recipient_projected_event_count": len(recipient_projection),
    }
    if not donor_projection or len(donor_projection) != len(recipient_projection):
        return None, dict(info, reason="stable_identity_projection_mismatch")
    if [item[0] for item in donor_projection] == [
            item[0] for item in recipient_projection]:
        return [
            (identity, donor_position, recipient_position)
            for (identity, donor_position), (_, recipient_position)
            in zip(donor_projection, recipient_projection)
        ], dict(info, order_rule="exact_sequence",
                reordered_layer_count=0, reordered_event_count=0)

    chunks = []
    for identity, position in donor_projection:
        layer_id = bisect.bisect_right(layer_starts, position) - 1
        if not chunks or chunks[-1][0] != layer_id:
            chunks.append((layer_id, []))
        chunks[-1][1].append((identity, position))
    pairs = []
    cursor = 0
    reordered_layers = 0
    reordered_events = 0
    for layer_id, donor_chunk in chunks:
        recipient_chunk = recipient_projection[cursor:cursor + len(donor_chunk)]
        cursor += len(donor_chunk)
        donor_ids = [item[0] for item in donor_chunk]
        recipient_ids = [item[0] for item in recipient_chunk]
        if collections.Counter(donor_ids) != collections.Counter(recipient_ids):
            return None, dict(
                info, reason="stable_identity_projection_mismatch",
                order_rule="per_layer_multiset",
                first_mismatch_layer=layer_id)
        queues = collections.defaultdict(collections.deque)
        for identity, position in recipient_chunk:
            queues[identity].append(position)
        for identity, position in donor_chunk:
            pairs.append((identity, position, queues[identity].popleft()))
        moved = sum(1 for left, right in zip(donor_ids, recipient_ids)
                    if left != right)
        if moved:
            reordered_layers += 1
            reordered_events += moved
    return pairs, dict(info, order_rule="per_layer_multiset",
                       reordered_layer_count=reordered_layers,
                       reordered_event_count=reordered_events)


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
    * require the two complete projected sequences to be byte-for-byte equal,
      or failing that, each donor layer's projected multiset to fill the matching
      consecutive chunk of the recipient projection (reordering inside a layer,
      never across one -- see stable_projection_pairs);
    * require multiple distinct matched anchors inside every marker-labelled
      layer and majority event coverage on both sides;
    * place an unmatched boundary gap on one side only when the donor marker
      proves that only that side has unmatched boundary-local events.

    There is no LCS, similarity score, stage taxonomy, model name, kernel-name
    allow-list, proportional partition, or best-effort layer completion here.
    """
    donor_sequence = list(donor.get("sequence") or [])
    layer_starts = [int(value) for value in donor.get(
        "layer_starts", [])]
    if (len(layer_starts) != expected_layers
            or layer_starts != sorted(layer_starts)
            or len(set(layer_starts)) != expected_layers):
        return None, {"reason": "donor_layer_starts_invalid"}

    pairs, projection = stable_projection_pairs(
        sequence, donor_sequence, layer_starts)
    if pairs is None:
        return None, projection
    donor_values = [item[0] for item in pairs]

    donor_anchor_positions = [[] for _ in range(expected_layers)]
    recipient_anchor_positions = [[] for _ in range(expected_layers)]
    donor_anchor_identities = [set() for _ in range(expected_layers)]
    for identity, donor_position, recipient_position in pairs:
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
    # A per-layer multiset pairing leaves a layer's recipient positions out of order.
    for positions in recipient_anchor_positions:
        positions.sort()

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
    recipient_fraction = float(len(donor_values)) / max(
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
            (STABLE_RULE if projection["order_rule"] == "exact_sequence"
             else STABLE_PER_LAYER_RULE)
            + ("_with_residuals" if residual_ranges else "")),
        "stable_projection": {
            "stable_identity_count": projection["stable_identity_count"],
            "order_rule": projection["order_rule"],
            "reordered_layer_count": projection["reordered_layer_count"],
            "reordered_event_count": projection["reordered_event_count"],
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


def _capture_subsequence_map(sequence, donor, expected_layers):
    """Map a graph-capture donor as an ordered subsequence of one step.

    Walk the donor kernels in order and take the next equal identity in the
    step within ``CAPTURE_LOOKAHEAD`` events.  Step-only kernels are passed
    over; a donor kernel not found nearby is skipped without moving on.  A
    layer starts at its first found kernel and runs to the next layer's
    start, so every step kernel in the body keeps a layer.  Kernels the
    capture did not name (overlapped scheduler work, copy/memset nodes,
    side-stream work) are only listed in ``capture_unmatched_positions``.
    The only requirements are that every layer finds a kernel and that most
    donor kernels are found.
    """
    donor_sequence = donor["sequence"]
    layer_starts = donor["layer_starts"]
    if len(layer_starts) != expected_layers or not donor_sequence:
        return None, {"reason": "donor_layer_starts_invalid"}
    first_found = [None] * expected_layers
    matched_positions = set()
    skipped = []
    found = 0
    cursor = 0
    last = -1
    for layer_id in range(expected_layers):
        stop = (layer_starts[layer_id + 1]
                if layer_id + 1 < expected_layers else len(donor_sequence))
        for donor_position in range(layer_starts[layer_id], stop):
            identity = donor_sequence[donor_position]
            limit = min(len(sequence), cursor + CAPTURE_LOOKAHEAD)
            hit = next((position for position in range(cursor, limit)
                        if sequence[position] == identity), None)
            if hit is None:
                skipped.append({"layer_id": layer_id, "identity": identity})
                continue
            if first_found[layer_id] is None:
                first_found[layer_id] = hit
            matched_positions.add(hit)
            found += 1
            cursor = hit + 1
            last = hit
    matched_fraction = float(found) / len(donor_sequence)
    empty_layers = [layer_id for layer_id, start in enumerate(first_found)
                    if start is None]
    if empty_layers or matched_fraction < CAPTURE_MIN_MATCHED_FRACTION:
        return None, {
            "reason": "capture_donor_not_found_in_step",
            "matched_fraction": round(matched_fraction, 6),
            "layers_without_match": empty_layers[:16],
            "skipped_donor_kernels": skipped[:16],
        }
    end = last + 1
    widths = [
        (first_found[layer_id + 1] if layer_id + 1 < expected_layers else end)
        - first_found[layer_id]
        for layer_id in range(expected_layers)]
    unmatched = [position for position in range(first_found[0], end)
                 if position not in matched_positions]
    return {
        "body_start_position": first_found[0],
        "body_end_position": end,
        "layer_start_positions": first_found,
        "layer_widths": widths,
        "layer_ranges": [{
            "layer_id": layer_id,
            "start_position": first_found[layer_id],
            "end_position": first_found[layer_id] + widths[layer_id],
            "representative_eligible": True,
        } for layer_id in range(expected_layers)],
        "residual_ranges": [],
        "capture_unmatched_positions": unmatched,
        "prefix_row_count": first_found[0],
        "suffix_row_count": len(sequence) - end,
        "match_rule": "capture_launch_ordered_subsequence",
        "capture_subsequence": {
            "donor_kernel_count": len(donor_sequence),
            "matched_kernel_count": found,
            "matched_fraction": round(matched_fraction, 6),
            "skipped_donor_kernels": skipped[:32],
            "capture_unmatched_kernels_in_body": len(unmatched),
        },
    }, None


def _map_step(step_rows, donor_passes, expected_layers):
    sequence = [
        _kernel_identity(row.get("raw_name"), row.get("event_type"))
        for row in step_rows]
    phase = _phase(step_rows[0].get("phase"))
    batch_size = _integer(step_rows[0].get("step_batch_size"))
    input_tokens = _integer(step_rows[0].get("step_input_tokens"))
    candidates = _compatible_donors(
        donor_passes, phase, batch_size, input_tokens)
    capture_failures = []
    for donor in candidates:
        if donor.get("source") != "capture":
            continue
        group, failure = _capture_subsequence_map(
            sequence, donor, expected_layers)
        if group is None:
            capture_failures.append({
                "donor_batch_size": donor["batch_size"], **failure})
            continue
        group.update({
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
                "source": "capture",
                "sequence_sha256": _sequence_sha(donor["sequence"]),
                "event_count": len(donor["sequence"]),
            },
        })
        return group, None

    mappings = {}
    compatible_donors = []
    considered = 0
    for donor in candidates:
        if donor.get("source") == "capture":
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
            "capture_donor_failures": capture_failures,
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
            "capture_donor_failures": capture_failures,
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


def _split_step_failures(failures, analysis_step_ids):
    """Only the steps semantics builds tables from may fail the transfer.

    A capture holds several steps; a non-analysis step (e.g. an extra prefill
    batch, or a ramp-up verify step) that cannot be mapped is reported but
    does not discard the validated cuts of the analysis steps.
    """
    blocking, ignored = [], []
    for failure in failures:
        step_id = failure.get("step_id")
        if step_id is not None and step_id not in analysis_step_ids:
            ignored.append(failure)
        else:
            blocking.append(failure)
    return blocking, ignored


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
    donor_scope_source = "geak_layer_scope_markers"
    if not markers:
        markers, donor_passes = _complete_donor_passes(
            donor_events, expected_layers,
            _dispatch_scope_markers(donor_events, pattern_doc))
        donor_scope_source = "declared_dispatch_op_span"
    recipient_events = semantic_kernel_mapping._load_events_multi(
        recipient_traces)
    rows, spans, _, _, _ = semantic_kernel_mapping._event_rows(
        recipient_events, pattern_doc)
    analysis_step_ids = {
        item["step_id"] for item in
        semantic_kernel_mapping._select_analysis_steps(spans).values()}
    by_step = {}
    for row in rows:
        if row.get("step_id"):
            by_step.setdefault(row["step_id"], []).append(row)

    groups = []
    failures = []
    skipped_authoritative = []
    # A step whose phase has no donor pass at all cannot be transferred, and that
    # says nothing about the other phases: the graph-construction replay captures
    # decode buckets only, while a Profile trace can hold a prefill step inside its
    # DECODE file. Such steps are reported, not counted as transfer failures, so
    # they cannot veto the steps that did map.
    donor_phases = {str(donor.get("phase") or "") for donor in donor_passes}
    untransferable = []
    for step_id, step_rows in sorted(
            by_step.items(), key=lambda item: min(
                row["device_seq_index"] for row in item[1])):
        step_rows.sort(key=lambda row: row["device_seq_index"])
        if _complete_authoritative_step(step_rows, expected_layers):
            skipped_authoritative.append(step_id)
            continue
        group, failure = _map_step(
            step_rows, donor_passes, expected_layers)
        step_phase = str(step_rows[0].get("phase") or "")
        if group is None and donor_passes and step_phase not in donor_phases:
            untransferable.append({
                "step_id": step_id,
                "phase": step_rows[0].get("phase"),
                "batch_size": step_rows[0].get("step_batch_size"),
                "input_tokens": step_rows[0].get("step_input_tokens"),
                "reason": "no_donor_pass_for_phase",
                "donor_phases": sorted(donor_phases),
            })
        elif group is None:
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
            "reason": "donor_has_no_layer_scopes",
            "detail": ("no GEAK_LAYER_SCOPE markers, and no step carries a "
                       "complete, Pattern-ordered set of declared dispatch ops")})
    elif not donor_passes:
        failures.append({
            "reason": "donor_has_no_complete_nonempty_layer_pass",
            "marker_count": len(markers),
            "expected_layers": expected_layers,
        })
    failures, non_analysis_failures = _split_step_failures(
        failures, analysis_step_ids)
    has_residuals = any(
        group.get("residual_ranges") for group in groups)
    # A step no donor pass validates (typically a batch size the donor never ran)
    # stays unresolved; it does not void the steps that did transfer. Only a
    # donor-wide failure, or nothing transferred at all, fails the map.
    unmapped_steps = [item for item in failures if item.get("step_id")]
    donor_failures = [item for item in failures if not item.get("step_id")]
    status = (
        "fail" if donor_failures or not groups else
        "partial" if has_residuals or unmapped_steps else
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
        "analysis_step_ids": sorted(analysis_step_ids),
        "non_analysis_step_failures": non_analysis_failures,
        "expected_main_layers": expected_layers,
        "patterns": {
            "path": os.path.abspath(pattern_path),
            "sha256": _sha256(pattern_path),
        },
        "donor": {
            "path": os.path.abspath(donor_trace),
            "sha256": _sha256(donor_trace),
            "scope_source": donor_scope_source,
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
        "untransferable_steps": untransferable,
        "unmapped_step_count": len(unmapped_steps),
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
