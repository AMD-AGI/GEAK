#!/usr/bin/env python3
"""Project Shape evidence from a boundary donor trace onto a Clean Trace table.

On vLLM the Clean Trace replays decode under a CUDA graph, so its decode rows
carry no ``Input Dims`` and no parent operator.  The CUDA-graph-off donor that
already supplied the layer cuts (``semantic_layer_boundary_transfer.py``) runs
the same compiled graph with the CPU walking every op, so its rows do carry
both.  This module copies them across the SAME event correspondence the
transfer validated -- nothing looser:

* ``exact_contiguous_normalized_device_sequence``: donor position i is recipient position
  ``body_start + i``;
* ``exact_equal_multiplicity_stable_identity_projection``: only identities with
  equal multiplicity on both sides, paired in order, and the pair count must
  equal the transfer's own ``stable_event_count``;
* ``equal_multiplicity_stable_identity_per_layer_multiset``: the same identities,
  paired within each layer by occurrence (the k-th ``X`` with the k-th ``X``)
  where concurrent streams reordered the layer; for both stable rules, inside each transferred
  layer cut, positional pairing over the longest prefix on which the donor and
  recipient identity sequences agree (the two rules must agree wherever both
  apply).

A row is projected only when every donor pass the transfer could have used
(same sequence hash, same bucket) gives it the same shape and parent operator;
anything else stays unresolved with a reason.  Projected evidence is level
``P`` (cross-trace), never ``K``.  Row identity, order, counts and durations
are never changed -- only ``shape``, ``parent_operator`` (when absent) and
``semantic_evidence`` of rows that were unresolved.
"""
import argparse
import collections
import copy
import json
import os

import semantic_evidence_ledger
import semantic_kernel_mapping
import semantic_layer_boundary_transfer as transfer
import semantic_shape_merge

SOURCE = "donor_trace_stable_projection"
EXACT_RULE = "exact_contiguous_normalized_device_sequence"
STABLE_RULE = transfer.STABLE_RULE
STABLE_RULES = tuple(rule + suffix
                     for rule in (transfer.STABLE_RULE, transfer.STABLE_PER_LAYER_RULE)
                     for suffix in ("", "_with_residuals"))


def _stable_pairs(recipient_sequence, donor_sequence, layer_starts):
    """(donor_position, recipient_position) pairs of the stable projection.

    The transfer's own pairing, so a per-layer multiset match pairs each row with
    the donor row of the same identity, never merely the one at the same index.
    """
    pairs, _ = transfer.stable_projection_pairs(
        recipient_sequence, donor_sequence, layer_starts)
    if pairs is None:
        return None
    return [(donor_position, recipient_position)
            for _, donor_position, recipient_position in pairs]


# The same device-to-device copy is a gpu_memcpy event when the CPU issues it and a
# rocclr copy kernel when a HIP graph replays it. Only for the layer comparison
# below are the two spellings one identity; nothing else is normalised.
_GRAPH_REPLAY_COPY_ALIASES = {
    "__amd_rocclr_copyBuffer": "device_to_device_copy",
    "Memcpy DtoD (Device -> Device)": "device_to_device_copy",
}


def _layer_identity(identity):
    return _GRAPH_REPLAY_COPY_ALIASES.get(identity, identity)


def _forced_subsequence_pairs(short, long):
    """{short_index: long_index} fixed by EVERY embedding of `short` in `long`.

    None when `short` is not a subsequence of `long`. Any embedding E satisfies
    leftmost[i] <= E[i] <= rightmost[i], so where the greedy leftmost and
    rightmost embeddings agree the pairing is forced; elsewhere it is a guess
    and stays unpaired.
    """
    leftmost, cursor = [], 0
    for identity in short:
        while cursor < len(long) and long[cursor] != identity:
            cursor += 1
        if cursor == len(long):
            return None
        leftmost.append(cursor)
        cursor += 1
    rightmost, cursor = [None] * len(short), len(long) - 1
    for index in range(len(short) - 1, -1, -1):
        while long[cursor] != short[index]:
            cursor -= 1
        rightmost[index] = cursor
        cursor -= 1
    return {index: left for index, (left, right) in enumerate(zip(leftmost, rightmost))
            if left == right}


def _layer_identical_pairs(group, donor, recipient_sequence):
    """Positional pairs inside each validated layer cut (donor vs recipient).

    The stable projection drops any identity whose step-wide multiplicity differs
    -- typically GEMM/quant kernels that also run outside the layer stack (lm_head,
    sampler), which are exactly the rows fusion cares about. Inside one layer cut,
    when one side's identity sequence is a subsequence of the other's (the eager
    donor adds work graph replay drops, e.g. the copy into a custom all-reduce's
    registered buffer), the positions every embedding agrees on are forced pairs.
    Otherwise only the agreeing prefix from the cut is paired.
    """
    # layer_ranges already exclude any inter-layer residual; a layer whose width
    # then differs from the donor's simply is not paired.
    recipient_ranges = [(int(item["start_position"]), int(item["end_position"]))
                        for item in group["layer_ranges"]]
    donor_starts = [int(value) for value in donor["layer_starts"]]
    donor_end = len(donor["sequence"])
    pairs = {}
    for layer, ((r_start, r_stop), d_start) in enumerate(
            zip(recipient_ranges, donor_starts)):
        d_stop = donor_starts[layer + 1] if layer + 1 < len(donor_starts) else donor_end
        recipient_layer = [_layer_identity(identity)
                           for identity in recipient_sequence[r_start:r_stop]]
        donor_layer = [_layer_identity(identity)
                       for identity in donor["sequence"][d_start:d_stop]]
        forced = _forced_subsequence_pairs(recipient_layer, donor_layer)
        if forced is not None:
            for r_offset, d_offset in forced.items():
                pairs[d_start + d_offset] = r_start + r_offset
            continue
        forced = _forced_subsequence_pairs(donor_layer, recipient_layer)
        if forced is not None:
            for d_offset, r_offset in forced.items():
                pairs[d_start + d_offset] = r_start + r_offset
            continue
        # Both sides start at the same validated cut, so while their identities
        # agree position by position the pairing is forced; stop at the first
        # divergence (e.g. the last layer's tail runs into different post-model work).
        for offset in range(min(r_stop - r_start, d_stop - d_start)):
            if (_layer_identity(recipient_sequence[r_start + offset])
                    != _layer_identity(donor["sequence"][d_start + offset])):
                break
            pairs[d_start + offset] = r_start + offset
    return pairs


def _pairs_for(group, donor, recipient_sequence):
    """{donor_position: recipient_position}, or None if it contradicts the map."""
    rule = group.get("match_rule")
    if rule == EXACT_RULE:
        start = int(group["body_start_position"])
        return {position: start + position
                for position in range(len(donor["sequence"]))}
    if rule not in STABLE_RULES:
        return None
    stable = _stable_pairs(recipient_sequence, donor["sequence"],
                           [int(value) for value in donor["layer_starts"]])
    expected = (group.get("stable_projection") or {}).get("stable_event_count")
    if stable is None or len(stable) != expected:
        return None
    pairs = dict(stable)
    for donor_position, recipient_position in _layer_identical_pairs(
            group, donor, recipient_sequence).items():
        # Both rules must agree wherever both speak; otherwise trust neither.
        if pairs.setdefault(donor_position, recipient_position) != recipient_position:
            return None
    return pairs


def _shape_key(row):
    shape = row.get("shape") or {}
    parent = (row.get("parent_operator") or {}).get("canonical_op")
    return json.dumps([shape.get("input_dims"), shape.get("input_types"), parent],
                      sort_keys=True)


def _resolved(row):
    # Dims, not the source label: a parent_context row can name its scope op and
    # still carry no dims, and must neither block nor supply a projection.
    return bool((row.get("shape") or {}).get("input_dims"))


def project(table_path, boundary_map_path, recipient_trace, donor_trace,
            pattern_path, out_dir):
    with open(boundary_map_path) as fh:
        boundary = json.load(fh)
    if boundary.get("status") not in ("pass", "partial"):
        raise ValueError("refusing a non-passing boundary map: %s"
                         % boundary.get("status"))
    if boundary["donor"]["sha256"] != transfer._sha256(donor_trace):
        raise ValueError("donor trace does not match the boundary map")
    recipient_shas = {item["sha256"] for item in boundary["recipient"]["traces"]}
    if transfer._sha256(recipient_trace) not in recipient_shas:
        raise ValueError("recipient trace does not match the boundary map")
    if boundary["patterns"]["sha256"] != transfer._sha256(pattern_path):
        raise ValueError("structural patterns do not match the boundary map")
    with open(pattern_path) as fh:
        pattern_doc = json.load(fh)
    with open(table_path) as fh:
        table_doc = json.load(fh)
    expected_layers = int(pattern_doc.get("num_hidden_layers_main", 0) or 0)

    donor_events = semantic_kernel_mapping._load_events(donor_trace)
    if boundary["donor"].get("scope_source") == "declared_dispatch_op_span":
        markers = transfer._dispatch_scope_markers(donor_events, pattern_doc)
    else:
        markers = None
    _, donor_passes = transfer._complete_donor_passes(
        donor_events, expected_layers, markers)
    passes_by_sha = collections.defaultdict(list)
    for donor in donor_passes:
        passes_by_sha[transfer._sequence_sha(donor["sequence"])].append(donor)
    donor_rows, _, _, _, _ = semantic_kernel_mapping._event_rows(
        donor_events, pattern_doc)
    donor_row_by_event = {row["raw_event_index"]: row for row in donor_rows}

    recipient_rows, _, _, _, _ = semantic_kernel_mapping._event_rows(
        semantic_kernel_mapping._load_events(recipient_trace), pattern_doc)
    by_step = collections.defaultdict(list)
    for row in recipient_rows:
        if row.get("step_id"):
            by_step[row["step_id"]].append(row)
    position_of = {}
    sequences = {}
    for step_id, rows in by_step.items():
        rows.sort(key=lambda row: row["device_seq_index"])
        sequences[step_id] = [
            transfer._kernel_identity(row.get("raw_name"), row.get("event_type"))
            for row in rows]
        for position, row in enumerate(rows):
            position_of[row["row_id"]] = (step_id, position)

    # (step_id, recipient_position) -> the donor row every candidate agrees on
    projected, conflicts, step_failures = {}, set(), []
    for group in boundary.get("mapped_groups") or []:
        step_id = group["recipient_step_id"]
        sequence = sequences.get(step_id)
        candidates = [
            donor for donor in passes_by_sha.get(
                group["donor"]["sequence_sha256"], [])
            if transfer._bucket_matches(donor, group.get("phase"),
                                        group.get("batch_size"),
                                        group.get("input_tokens"))]
        if sequence is None or not candidates:
            step_failures.append({"step_id": step_id,
                                  "reason": "donor_pass_not_reconstructed"})
            continue
        seen = collections.defaultdict(dict)
        for donor in candidates:
            pairs = _pairs_for(group, donor, sequence)
            if pairs is None:
                step_failures.append({"step_id": step_id,
                                      "reason": "pairing_disagrees_with_boundary_map"})
                seen = None
                break
            for donor_position, recipient_position in pairs.items():
                entry = donor["entries"][donor_position]
                donor_row = donor_row_by_event.get(entry["device_event_index"])
                if donor_row is not None:
                    seen[recipient_position][_shape_key(donor_row)] = donor_row
        if seen is None:
            continue
        for recipient_position, variants in seen.items():
            key = (step_id, recipient_position)
            if len(variants) == 1:
                projected[key] = next(iter(variants.values()))
            else:
                conflicts.add(key)

    stats = collections.defaultdict(collections.Counter)
    out_doc = copy.deepcopy(table_doc)
    for table in out_doc.get("tables", []):
        phase = table.get("phase")
        for row in table.get("rows", []):
            stats[phase]["rows"] += 1
            if _resolved(row):
                stats[phase]["already_resolved"] += 1
                continue
            key = position_of.get(row["row_id"])
            donor_row = projected.get(key) if key else None
            if key in conflicts:
                reason = "donor_passes_disagree_on_shape"
            elif donor_row is None:
                reason = "no_stable_donor_counterpart"
            elif not _resolved(donor_row):
                reason = "donor_counterpart_shape_unresolved"
            else:
                reason = None
            if reason:
                stats[phase][reason] += 1
                continue
            donor_shape = donor_row["shape"]
            # Projection moves dims across traces; it cannot make an operator's
            # inputs into the kernel's own, so the donor row's granularity travels.
            granularity = semantic_evidence_ledger.shape_granularity(donor_row)
            row["shape"] = {
                "source": SOURCE,
                "input_dims": donor_shape.get("input_dims") or [],
                "input_types": donor_shape.get("input_types") or [],
                "donor_shape_source": donor_shape.get("source"),
                "granularity": granularity,
            }
            donor_parent = donor_row.get("parent_operator") or {}
            if (row.get("parent_operator") or {}).get("canonical_op") in (
                    None, "", "unresolved"):
                row["parent_operator"] = {
                    **donor_parent,
                    "mapping_level": SOURCE,
                    "evidence_event_index": donor_parent.get("evidence_event_index"),
                }
            row["semantic_evidence"] = {
                "level": "P",
                "probe_scope": "kernel" if granularity == "kernel" else "wrapper",
                "status": "matched",
                "source": SOURCE,
                "mapping_basis": (
                    "boundary-map event correspondence (%s) between the Clean "
                    "Trace step and a same-bucket CUDA-graph-off donor pass"
                    % boundary_rule_of(boundary, key[0])),
                "donor_row_id": donor_row["row_id"],
                "donor_trace_sha256": boundary["donor"]["sha256"],
                "schema": {"input_dims": row["shape"]["input_dims"],
                           "input_types": row["shape"]["input_types"]},
            }
            stats[phase]["projected"] += 1

    summary = {
        "schema_version": 1,
        "source": SOURCE,
        "boundary_map": os.path.abspath(boundary_map_path),
        "donor_trace": os.path.abspath(donor_trace),
        "recipient_trace": os.path.abspath(recipient_trace),
        "by_phase": {phase: dict(counter) for phase, counter in sorted(stats.items())},
        "step_failures": step_failures,
    }
    out_doc["donor_shape_projection"] = summary
    # Rows now carry shapes; the embedded coverage summary and the markdown table
    # must say so, or readers see decode "0/N" beside resolved rows.
    semantic_evidence_ledger.refresh_phase_coverage(out_doc)
    os.makedirs(out_dir, exist_ok=True)
    table_out = os.path.join(out_dir, "pattern_layer_kernel_table.json")
    with open(table_out, "w") as fh:
        json.dump(out_doc, fh, indent=2)
    markdown_out = os.path.join(out_dir, "ORDERED_UNIQUE_LAYER_TABLES.md")
    with open(markdown_out, "w") as fh:
        fh.write(semantic_shape_merge._markdown(out_doc))
    summary["table_json"] = table_out
    summary["table_md"] = markdown_out
    with open(os.path.join(out_dir, "DONOR_SHAPE_PROJECTION.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def boundary_rule_of(boundary, step_id):
    for group in boundary.get("mapped_groups") or []:
        if group["recipient_step_id"] == step_id:
            return group.get("match_rule")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True,
                        help="pattern_layer_kernel_table.json rebuilt with --layer-boundary-map")
    parser.add_argument("--boundary-map", required=True)
    parser.add_argument("--recipient-trace", required=True)
    parser.add_argument("--donor-trace", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    summary = project(args.table, args.boundary_map, args.recipient_trace,
                      args.donor_trace, args.patterns, args.out_dir)
    print(json.dumps({k: summary[k] for k in ("by_phase", "step_failures")}, indent=2))
    return 0 if not summary["step_failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
