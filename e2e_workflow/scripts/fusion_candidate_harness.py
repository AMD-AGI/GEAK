#!/usr/bin/env python3
"""Validate Fusion 2.1 facts/coverage and render the mandatory total table."""
import argparse
import json
import math
import os
import re
import sys


PHASE_ORDER = {"prefill": 0, "decode": 1}
READINESS = {
    "ready_for_api_validation",
    "needs_source_dependency_proof",
    "blocked_shape",
    "blocked_evidence",
    "research_only",
}
EXACT_STATUS = {"yes", "no"}
# Author-track classes (no existing kernel -> 现成算子=无).
AUTHOR_CLASSES = {"new_helper_kernel", "main_kernel_or_algorithmic"}
# Generic tokens that indicate a quantization output/arg in a routed call
# signature. Used to check that a flag actually fuses quant before an A-tier
# *_quant candidate may claim it — model/kernel-agnostic, no hard-coded answer.
_QUANT_SIG_TOKENS = (
    "quant", "scale", "fp8", "fp4", "amax", "e4m3", "e5m2", "mxfp")
API_COVERAGE = {"full", "partial", "similar"}
API_SOURCE_KIND = {
    "runtime_environment", "runtime_source", "perf_knowledge"}

# Only the main compute/communication bodies are donors. Every other stage is a
# removable helper (elementwise/layout/copy/norm/quant/activation/kv-cache
# write) and must not be silently dropped into a fusion_opportunity=false stage.
DONOR_STAGES = {
    "gemm", "attn", "attention", "communication", "collective",
    "moe", "expert_gemm"}
# Helper rows at or above this duration may not vanish: each must be a candidate
# member or a deferred required_followups[].row_ids entry.
DEFAULT_HELPER_FLOOR_US = 5.0
# Helper rows at or above this (higher) duration must be an actual CANDIDATE
# (author-track), not merely deferred to a followup — completeness escalation.
DEFAULT_ESCALATE_FLOOR_US = 15.0
# Sub-floor helpers in one (phase, pattern) whose per-layer durations sum to at
# least this must also become a candidate (cluster fusion) — catches many small
# helpers that individually escape the per-row escalate floor but add up.
DEFAULT_AGG_ESCALATE_FLOOR_US = 20.0

# --- Roofline reuse: borrow GEAK's existing roofline peaks + launch constant to
# turn "which rows fusion removes" into a physically-grounded savings estimate.
# Defensive: a broken/absent roofline helper only drops the estimate to None; it
# never fails validation (mirrors the roofline SKILL's own doctrine).
_RF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "knowledge", "analysis_skills", "roofline")
_PEAKS_MD = os.path.join(_RF_DIR, "peaks.md")
try:
    if _RF_DIR not in sys.path:
        sys.path.insert(0, _RF_DIR)
    import roofline_tools as _roofline
except Exception:
    _roofline = None
LAUNCH_OVERHEAD_US = 5.0  # matches roofline_tools.LAUNCH_OVERHEAD_S (5e-6 s)
# Per-gfx HBM bandwidth (bytes/us) fallback if peaks.md/roofline is unavailable.
# Values are per-gfx HBM bandwidth in bytes/us (peak B/s ÷ 1e6). Extend per gfx.
_FALLBACK_HBM_BW_BYTES_PER_US = {"gfx942": 5.3e6, "gfx950": 8.0e6}
_DTYPE_BYTES = {
    "float8_e4m3fnuz": 1, "float8": 1, "fp8": 1, "int8": 1,
    "bfloat16": 2, "c10::bfloat16": 2, "float16": 2, "c10::half": 2, "fp16": 2,
    "bf16": 2, "float": 4, "float32": 4, "fp32": 4, "int32": 4, "int64": 8,
    "float64": 8}


def _gfx_key(environment):
    """Extract a gfxNNN key from anywhere in the environment inventory."""
    match = re.search(r"gfx\d+", json.dumps(environment or {}))
    return match.group(0) if match else None


def _hbm_bw_bytes_per_us(environment):
    gfx = _gfx_key(environment)
    if _roofline is not None and gfx:
        try:
            peaks = _roofline.load_peaks(_PEAKS_MD, gfx)
            if peaks and peaks.get("hbm_bw_bytes_s"):
                return float(peaks["hbm_bw_bytes_s"]) / 1e6
        except Exception:
            pass
    return _FALLBACK_HBM_BW_BYTES_PER_US.get(gfx)


def _dtype_bytes(name):
    key = str(name or "").strip().lower()
    return _DTYPE_BYTES.get(key, 2)


def _row_bytes(row):
    """Bytes the kernel moves through HBM, from the semantic table's shape."""
    shape = row.get("shape") or {}
    dims = shape.get("input_dims")
    types = shape.get("input_types") or []
    if not isinstance(dims, list) or not dims:
        return None
    total = 0
    for index, dim in enumerate(dims):
        if not isinstance(dim, list) or not dim:
            continue
        count = 1
        for value in dim:
            if isinstance(value, (int, float)) and value > 0:
                count *= int(value)
        dtype = types[index] if index < len(types) else None
        total += count * _dtype_bytes(dtype)
    return total if total > 0 else None


def _estimate_savings(removable_ids, source_rows, key, bw_per_us, ceiling_us):
    """Roofline-grounded savings for the rows a fusion removes.

    Per removed helper the recoverable time is the launch it saves plus the HBM
    round-trip it eliminates (bytes / bandwidth), capped by its own measured
    duration — so a memory-bound helper recovers ~all of its time while a
    compute-bound one recovers only the launch. Total is capped at the
    duration-based ceiling (Σ removable). Returns estimate=None when bandwidth
    or shapes are unavailable rather than guessing.
    """
    if not removable_ids:
        return {"ceiling_us": round(ceiling_us, 3), "estimate_us": 0.0,
                "basis": "empty"}
    if not bw_per_us:
        return {"ceiling_us": round(ceiling_us, 3), "estimate_us": None,
                "basis": "no_bandwidth"}
    estimate = 0.0
    modeled_all = True
    for row_id in removable_ids:
        row = source_rows.get((key[0], key[1], row_id))
        if not row:
            continue
        duration = float(row.get("duration_us", 0.0) or 0.0)
        num_bytes = _row_bytes(row)
        if num_bytes is None:
            modeled_all = False
            estimate += duration  # shape unknown -> optimistic (memory-bound)
            continue
        recoverable = LAUNCH_OVERHEAD_US + num_bytes / bw_per_us
        estimate += min(duration, recoverable)
    estimate = min(estimate, ceiling_us)
    return {"ceiling_us": round(ceiling_us, 3),
            "estimate_us": round(estimate, 3),
            "basis": "roofline" if modeled_all else "roofline_partial"}

# Per-aiter-commit fused all-reduce (custom_all_reduce) size-guard registry.
# Consumed only by this harness to cross-check the analyst-declared
# environment_api_inventory.collective_fused_ar_guard.threshold_bytes, so the
# threshold number itself cannot drift/hallucinate between runs. Schema:
#   {aiter_commit: {fused_ar_threshold_bytes:int, expr:str, ref:str, notes:str}}
# Add a new entry when a new aiter build is inspected; verify the expr against
# the installed source before adding. Kept inline (not a sidecar JSON) to match
# GEAK's convention for small versioned single-consumer tables (see STAGE_RULES
# in semantic_kernel_mapping.py) — e2e_workflow has no data-file convention.
FUSED_AR_GUARD_REGISTRY = {
    "a6bb499375849eec45d68c5ccaebc8865fd422c0": {
        "fused_ar_threshold_bytes": 67108864,
        "expr": "n <= 16384 and total_bytes < 8 * 1024 * 8192 and world_size != 6",
        "ref": ("aiter/dist/device_communicators/communicator_cuda.py::"
                "fused_allreduce_rmsnorm (can_use_fuse_ar_rms)"),
        "notes": ("8*1024*8192 = 67108864 bytes = 64 MiB. Tensor >= this falls "
                  "back to split all-reduce + separate rmsnorm. 1-stage vs "
                  "2-stage is a separate switch (total_bytes <= 128*1024) and "
                  "does not affect availability."),
    },
}


def _load_guard_registry():
    """Per-aiter-commit fused-AR guard registry (inline; see above)."""
    return FUSED_AR_GUARD_REGISTRY


def _short_ref(ref):
    ref = str(ref or "")
    return ref.split("/")[-1] or ref


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _close(left, right, tolerance=1e-3):
    return math.isclose(
        float(left or 0.0), float(right or 0.0),
        rel_tol=1e-7, abs_tol=tolerance)


def _escape(value):
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def _numbered(values):
    markers = "①②③④⑤⑥⑦⑧⑨⑩"
    return "<br>".join(
        "%s %s" % (markers[index] if index < len(markers)
                   else "%d." % (index + 1), value)
        for index, value in enumerate(values))


def _api_summary(plan):
    apis = plan.get("existing_apis") or []
    rendered = [
        "`%s`（%s）" % (
            api.get("name", "unknown"),
            "完整覆盖" if api.get("coverage") == "full" else "部分覆盖")
        for api in apis
    ]
    if plan.get("exact_kernel_status") == "no":
        reason = plan.get("exact_reason", "无现成算子，需自写")
        rendered.append("无现成算子（%s）" % reason)
    return "；".join(rendered) if rendered else "无现成算子"


def _api_detail(plan):
    apis = plan.get("existing_apis") or []
    if not apis:
        return "无已确认 API"
    rendered = []
    for api in apis:
        text = "`%s`（%s，%s）" % (
            api.get("name", "unknown"), api.get("coverage", "unknown"),
            api.get("source_kind", "unknown"))
        if api.get("constraints"):
            text += "；" + "；".join(map(str, api["constraints"]))
        rendered.append(text)
    return "；".join(rendered)


def _savings_text(plan, layer_total_us, candidate=None):
    # Total table shows the single roofline engineering estimate. The
    # duration-based ceiling stays in the per-candidate detail. When the estimate
    # is unavailable (missing shape/bandwidth) fall back to the ceiling, flagged.
    savings = (candidate or {}).get("_savings") or {}
    est = savings.get("estimate_us")
    if est is not None:
        pct = est / layer_total_us * 100.0 if layer_total_us else 0.0
        note = "" if savings.get("basis") == "roofline" else "*"
        return "%.2f µs/层（%.2f%%）%s" % (est, pct, note)
    ceiling = float(
        savings.get("ceiling_us",
                    plan.get("addressable_us_per_layer", 0.0)) or 0.0)
    pct = ceiling / layer_total_us * 100.0 if layer_total_us else 0.0
    return "≤%.2f µs/层（%.2f%%，估算不可用）" % (ceiling, pct)


def _forward_savings_text(candidate):
    # Whole-forward projection: per-layer estimate × recurrence count, as a
    # fraction of this phase's total forward time.
    savings = (candidate or {}).get("_savings") or {}
    stack = savings.get("stack_estimate_us")
    if stack is None:
        return "估算不可用"
    pct = savings.get("total_forward_pct")
    note = "" if savings.get("basis") == "roofline" else "*"
    pct_text = ("%.3f%%" % pct) if pct is not None else "n/a"
    return "%.1f µs（%s）%s" % (stack, pct_text, note)


def _canonical_plan(value):
    return "+".join(
        part.strip().lower() for part in str(value or "").split("+"))


def _collective_requirements(table):
    requirements = []
    for item in table.get("tables", []):
        rows = sorted(
            item.get("rows", []), key=lambda row: int(row.get("pos", 0)))
        for index in range(len(rows) - 1):
            communication, norm = rows[index:index + 2]
            if (communication.get("stage"), norm.get("stage")) != (
                    "communication", "norm"):
                continue
            if (communication.get("stream") != norm.get("stream")
                    or int(norm.get("device_seq_index", -1))
                    != int(communication.get("device_seq_index", -2)) + 1):
                continue
            key = (item.get("phase"), item.get("pattern_id"))
            quant = rows[index + 2] if index + 2 < len(rows) else {}
            has_quant = (
                quant.get("stage") == "quant"
                    and quant.get("stream") == norm.get("stream")
                    and int(quant.get("device_seq_index", -1))
                    == int(norm.get("device_seq_index", -2)) + 1)
            # A post-collective residual norm is a norm position with the full
            # narrow-to-broad family. The quant member is the fp8 quant that
            # consumes the normed activation: the immediately-following row when
            # present (dense FFN), otherwise the first later same-stream quant in
            # the table (MoE expert-input quant; the router consumes the same
            # normed activation in bf16 via the fused-AR emit_bf16 dual output).
            # Only a norm whose activation is never quantized later collapses to
            # allreduce + norm alone.
            if has_quant:
                quant_row = quant
            else:
                quant_row = None
                for later in rows[index + 2:]:
                    if (later.get("stage") == "quant"
                            and later.get("stream") == norm.get("stream")):
                        quant_row = later
                        break
            if quant_row is not None:
                requirements.append((
                    key, 1, "norm+quant",
                    (norm.get("row_id"), quant_row.get("row_id"))))
                requirements.append((
                    key, 2, "allreduce+norm",
                    (communication.get("row_id"), norm.get("row_id"))))
                requirements.append((
                    key, 3, "allreduce+norm+quant",
                    (communication.get("row_id"), norm.get("row_id"),
                     quant_row.get("row_id"))))
            else:
                requirements.append((
                    key, 1, "allreduce+norm",
                    (communication.get("row_id"), norm.get("row_id"))))
    return requirements


def _semantic_index(table):
    tables = {}
    rows = {}
    table_order = {}
    for index, item in enumerate(table.get("tables", [])):
        key = (item.get("phase"), item.get("pattern_id"))
        tables[key] = item
        table_order[key] = index
        for row in item.get("rows", []):
            rows[(key[0], key[1], row.get("row_id"))] = row
    return tables, rows, table_order


def _validate_api_list(owner, path, errors):
    status = owner.get("exact_kernel_status")
    if status not in EXACT_STATUS:
        errors.append("%s exact_kernel_status must be %s" % (
            path, sorted(EXACT_STATUS)))
    apis = owner.get("existing_apis")
    if not isinstance(apis, list):
        errors.append("%s existing_apis must be a list" % path)
        return
    for index, api in enumerate(apis):
        api_path = "%s.existing_apis[%d]" % (path, index)
        if not api.get("name"):
            errors.append("%s missing name" % api_path)
        if api.get("coverage") not in API_COVERAGE:
            errors.append("%s coverage must be %s" % (
                api_path, sorted(API_COVERAGE)))
        if api.get("source_kind") not in API_SOURCE_KIND:
            errors.append("%s source_kind must be %s" % (
                api_path, sorted(API_SOURCE_KIND)))
        if not api.get("evidence"):
            errors.append("%s missing evidence" % api_path)
    if status == "yes" and not any(
            api.get("coverage") == "full"
            and api.get("source_kind") in {
                "runtime_environment", "runtime_source"}
            for api in apis):
        errors.append(
            "%s exact=yes requires full current-environment/source evidence"
            % path)
    if status == "no":
        reason = owner.get("exact_reason", "")
        if not reason:
            errors.append("%s exact=no requires exact_reason" % path)
        elif len(reason) > 160:
            errors.append("%s exact_reason must stay concise" % path)


def validate(semantic_table_path, candidates_path,
             helper_floor=DEFAULT_HELPER_FLOOR_US,
             escalate_floor=DEFAULT_ESCALATE_FLOOR_US,
             agg_escalate_floor=DEFAULT_AGG_ESCALATE_FLOOR_US):
    table = _load(semantic_table_path)
    payload = _load(candidates_path)
    tables, source_rows, table_order = _semantic_index(table)
    errors = []
    warnings = []
    bw_per_us = None  # HBM bytes/us for the roofline savings estimate

    # Provenance: the candidates must have been built from exactly this table.
    declared = payload.get("source_semantic_table") or {}
    declared_sha = declared.get("trace_sha256")
    table_sha = table.get("trace_sha256")
    if declared_sha and table_sha and declared_sha != table_sha:
        errors.append(
            "source_semantic_table.trace_sha256 %s does not match semantic "
            "table trace_sha256 %s" % (declared_sha, table_sha))

    if payload.get("phase") != "generate_plans":
        errors.append("top-level phase must be generate_plans")
    if not isinstance(payload.get("stage_inventory"), list):
        errors.append("stage_inventory must be a list")
    if not isinstance(payload.get("summary_rows"), list):
        errors.append("summary_rows must be a list")
    if not isinstance(payload.get("candidates"), list):
        errors.append("candidates must be a list")
    collective_guard = None
    model_dims = None
    environment_path = payload.get("environment_api_inventory_json", "")
    if not environment_path or not os.path.exists(environment_path):
        errors.append(
            "environment_api_inventory_json must reference an inspected "
            "runtime environment artifact")
    else:
        environment = _load(environment_path)
        if not environment.get("image"):
            errors.append("environment API inventory must record image")
        if not environment.get("inspection_evidence"):
            errors.append(
                "environment API inventory must record inspection_evidence")
        bw_per_us = _hbm_bw_bytes_per_us(environment)
        # Collective fused-AR size guard becomes a machine-checked fact so the
        # prefill=no / decode=yes Exact decision is deterministic instead of
        # re-derived (and mis-numbered) by the model each run.
        collective_guard = environment.get("collective_fused_ar_guard")
        if (not isinstance(collective_guard, dict)
                or not isinstance(collective_guard.get("threshold_bytes"), int)
                or not collective_guard.get("source_expr")
                or not collective_guard.get("source_ref")):
            errors.append(
                "environment API inventory must record collective_fused_ar_guard "
                "{threshold_bytes:int, source_expr, source_ref}")
            collective_guard = None
        model_dims = environment.get("model_dims")
        if (not isinstance(model_dims, dict)
                or not isinstance(model_dims.get("hidden_size"), int)
                or not isinstance(model_dims.get("dtype_bytes"), int)):
            errors.append(
                "environment API inventory must record model_dims "
                "{hidden_size:int, dtype_bytes:int}")
            model_dims = None
        # Cross-check the declared threshold against the guard registry keyed by
        # the recorded aiter commit, so the threshold number itself cannot drift.
        if collective_guard is not None:
            commit = (environment.get("toolchain", {}) or {}).get(
                "aiter_git_commit") or environment.get("aiter_git_commit")
            entry = _load_guard_registry().get(commit) if commit else None
            if entry is not None:
                expected = entry.get("fused_ar_threshold_bytes")
                if (expected is not None
                        and int(collective_guard["threshold_bytes"])
                        != int(expected)):
                    errors.append(
                        "collective_fused_ar_guard.threshold_bytes %s disagrees "
                        "with guard registry %s for aiter %s" % (
                            collective_guard["threshold_bytes"], expected,
                            commit))
            else:
                warnings.append(
                    "no fused-AR guard registry entry for aiter commit %s; "
                    "threshold not cross-checked" % commit)
    if errors:
        return payload, table, errors, warnings, {
            "source_row_count": len(source_rows),
            "covered_source_row_count": 0,
            "source_row_coverage_pct": 0.0,
        }

    covered = set()
    opportunity_candidate_ids = set()
    for index, stage in enumerate(payload["stage_inventory"]):
        path = "stage_inventory[%d]" % index
        key = (stage.get("phase"), stage.get("pattern_id"))
        if key not in tables:
            errors.append("%s references unknown table %s" % (path, key))
            continue
        row_ids = stage.get("row_ids")
        if not isinstance(row_ids, list) or not row_ids:
            errors.append("%s row_ids must be a non-empty list" % path)
            continue
        for row_id in row_ids:
            row_key = (key[0], key[1], row_id)
            if row_key not in source_rows:
                errors.append("%s references unknown row_id %s" % (
                    path, row_id))
            else:
                covered.add(row_key)
        if stage.get("fusion_opportunity") is True:
            ids = stage.get("candidate_ids")
            if not isinstance(ids, list) or not ids:
                errors.append(
                    "%s fusion opportunity requires candidate_ids" % path)
            else:
                opportunity_candidate_ids.update(ids)
        elif not stage.get("reason"):
            errors.append(
                "%s non-opportunity stage requires a reason" % path)

    missing_rows = sorted(set(source_rows) - covered)
    if missing_rows:
        errors.append("stage inventory misses %d/%d source rows; first=%s" % (
            len(missing_rows), len(source_rows), missing_rows[:5]))

    # Per-phase total forward time = Σ over that phase's tables of
    # (representative layer_total_us × pattern_layer_count). Prefill and decode
    # are separate forwards, so each candidate's stack savings is expressed as a
    # fraction of ITS phase's whole forward, never a prefill+decode mix.
    phase_total_forward_us = {}
    for item in table.get("tables", []):
        phase = item.get("phase")
        contribution = (float(item.get("layer_total_us", 0.0) or 0.0)
                        * int(item.get("pattern_layer_count", 0) or 0))
        phase_total_forward_us[phase] = (
            phase_total_forward_us.get(phase, 0.0) + contribution)

    candidates_by_id = {}
    candidate_member_ids = {}
    referenced_candidate_ids = set()
    collective_guard_checks = []
    for index, candidate in enumerate(payload["candidates"]):
        path = "candidates[%d]" % index
        candidate_id = candidate.get("candidate_id")
        if not candidate_id:
            errors.append("%s missing candidate_id" % path)
            continue
        if candidate_id in candidates_by_id:
            errors.append("duplicate candidate_id %s" % candidate_id)
        candidates_by_id[candidate_id] = candidate
        key = (candidate.get("phase"), candidate.get("pattern_id"))
        if key not in tables:
            errors.append("%s references unknown table %s" % (path, key))
            continue
        if candidate.get("readiness") not in READINESS:
            errors.append("%s invalid readiness" % path)
        members = candidate.get("members")
        if not isinstance(members, list) or len(members) < 2:
            errors.append("%s members must contain at least two rows" % path)
            continue
        member_ids = []
        member_total = 0.0
        previous_pos = None
        is_boundary = bool(candidate.get("boundary"))
        for member_index, member in enumerate(members):
            member_path = "%s.members[%d]" % (path, member_index)
            row_id = member.get("row_id")
            row_key = (key[0], key[1], row_id)
            source = source_rows.get(row_key)
            if source is None:
                errors.append("%s unknown row_id %s" % (
                    member_path, row_id))
                continue
            member_ids.append(row_id)
            member_total += float(source.get("duration_us", 0.0) or 0.0)
            for field in ("pos", "device_seq_index", "stream", "duration_us"):
                if field == "duration_us":
                    matches = _close(
                        member.get(field), source.get(field))
                else:
                    matches = member.get(field) == source.get(field)
                if not matches:
                    errors.append("%s %s differs from semantic table" % (
                        member_path, field))
            # Boundary (cross-layer) candidates list the previous-layer tail
            # all-reduce alongside this layer's head norm/quant, so their in-table
            # pos wraps around; ordering is not enforced for them.
            if (not is_boundary and previous_pos is not None
                    and member.get("pos", -1) <= previous_pos):
                errors.append("%s members are not in execution order" % path)
            previous_pos = member.get("pos")
        candidate_member_ids[candidate_id] = tuple(member_ids)
        if not _close(
                candidate.get("current_chain_us_per_layer"), member_total):
            errors.append("%s current_chain_us_per_layer must equal member sum" % path)
        removable = candidate.get("removable_row_ids")
        if not isinstance(removable, list):
            errors.append("%s removable_row_ids must be a list" % path)
            removable = []
        if not set(removable).issubset(set(member_ids)):
            errors.append("%s removable rows must be candidate members" % path)
        removable_total = sum(
            float(source_rows[(key[0], key[1], row_id)].get(
                "duration_us", 0.0) or 0.0)
            for row_id in removable
            if (key[0], key[1], row_id) in source_rows)
        if not _close(
                candidate.get("addressable_us_per_layer"), removable_total):
            errors.append("%s addressable_us_per_layer must equal removable sum" % path)
        # Merge ceiling: a fusion replaces N kernels with ONE fused kernel that
        # still runs, so it must keep an anchor — the heaviest member cannot be
        # counted as savings. This kills the "removable = every member -> save
        # 100%" over-claim (e.g. norm+quant marked fully removable).
        member_durations = [
            float(source_rows[(key[0], key[1], rid)].get("duration_us", 0.0)
                  or 0.0)
            for rid in member_ids if (key[0], key[1], rid) in source_rows]
        max_member = max(member_durations) if member_durations else 0.0
        merge_ceiling = member_total - max_member
        if removable_total > merge_ceiling + 1e-3:
            errors.append(
                "%s addressable %.3f exceeds merge ceiling %.3f (Σmembers − "
                "heaviest member %.3f): a fused kernel is at least as costly as "
                "its heaviest constituent, which cannot be counted as savings — "
                "keep it as the anchor (not removable)" % (
                    path, removable_total, merge_ceiling, max_member))
        # A fused kernel cannot cross a main donor. An aggregate "cluster"
        # candidate (family *_cluster, from the sub-floor escalation) must be a
        # CONTIGUOUS run — no donor-stage row (GEMM/Attention/MoE/Collective) may
        # sit strictly between its members' device_seq span unless it is a member;
        # scattered small helpers on both sides of an attention/GEMM are separate
        # clusters. (Single producer→consumer folds like norm→downstream-quant,
        # and collective dual-output fusions, are NOT clusters and are exempt —
        # their non-adjacent member is the same data path emitted in one kernel.)
        cand_family = str(candidate.get("family") or "")
        if not is_boundary and cand_family.endswith("_cluster") and members:
            member_seqs = [
                int(source_rows[(key[0], key[1], rid)].get(
                    "device_seq_index", -1))
                for rid in member_ids if (key[0], key[1], rid) in source_rows]
            member_seq_set = set(member_seqs)
            if member_seqs:
                lo, hi = min(member_seqs), max(member_seqs)
                spanned = [
                    (row.get("device_seq_index"), row.get("stage"),
                     row.get("row_id"))
                    for row in tables[key].get("rows", [])
                    if str(row.get("stage") or "").lower() in DONOR_STAGES
                    and lo < int(row.get("device_seq_index", -1)) < hi
                    and int(row.get("device_seq_index", -1))
                    not in member_seq_set]
                if spanned:
                    errors.append(
                        "%s non-boundary candidate spans %d main donor row(s) "
                        "(a fused kernel cannot cross GEMM/Attention/MoE/"
                        "Collective); split into contiguous sub-fusions; "
                        "first=%s" % (path, len(spanned), spanned[:3]))
        layer_count = int(candidate.get("pattern_layer_count", 0) or 0)
        pattern_layers = int(tables[key].get("pattern_layer_count", 0) or 0)
        if layer_count != pattern_layers:
            errors.append("%s pattern_layer_count differs from semantic table" % path)
        # A boundary candidate recurs once per homogeneous layer boundary, not
        # once per pattern layer, so its ceiling uses a declared
        # boundary_occurrences instead of pattern_layer_count.
        if is_boundary:
            occ = candidate.get("boundary_occurrences")
            if (not isinstance(occ, int) or occ < 1
                    or (pattern_layers and occ > pattern_layers)):
                errors.append(
                    "%s boundary candidate requires boundary_occurrences int in "
                    "[1, pattern_layer_count]" % path)
                occ = 0
            ceiling_count = occ
        else:
            ceiling_count = layer_count
        expected_stack = removable_total * ceiling_count
        if not _close(
                candidate.get("stack_addressable_ceiling_us"),
                expected_stack):
            errors.append(
                "%s stack_addressable_ceiling_us formula mismatch" % path)
        # Roofline savings (harness-computed) + whole-forward projection: scale
        # per-layer savings by the recurrence count and express as a fraction of
        # this phase's total forward time (all layers of all its patterns).
        candidate["_savings"] = _estimate_savings(
            removable, source_rows, key, bw_per_us, removable_total)
        forward_total = float(phase_total_forward_us.get(key[0], 0.0) or 0.0)
        _sv = candidate["_savings"]
        _sv["ceiling_count"] = ceiling_count
        _sv["stack_ceiling_us"] = round(removable_total * ceiling_count, 3)
        _sv["stack_estimate_us"] = (
            round(_sv["estimate_us"] * ceiling_count, 3)
            if _sv.get("estimate_us") is not None else None)
        _sv["phase_total_forward_us"] = round(forward_total, 3)
        _sv["total_forward_pct"] = (
            round(_sv["stack_estimate_us"] / forward_total * 100.0, 4)
            if _sv.get("stack_estimate_us") is not None and forward_total
            else None)
        # Deterministic collective size-guard: a candidate that includes a
        # communication-stage member is a collective (allreduce) fusion; the
        # harness computes the AR tensor bytes from the selected bucket and
        # model dims and enforces Exact=no when it exceeds the fused-AR guard.
        if collective_guard is not None and model_dims is not None:
            is_collective = any(
                source_rows.get((key[0], key[1], rid), {}).get("stage")
                == "communication" for rid in member_ids)
            bucket = tables[key].get("selected_bucket", {}) or {}
            tokens = (bucket.get("input_tokens") if key[0] == "prefill"
                      else bucket.get("batch_size"))
            if is_collective and tokens:
                tensor_bytes = (int(tokens) * int(model_dims["hidden_size"])
                                * int(model_dims["dtype_bytes"]))
                # Single fused-AR size guard (from the env-recorded threshold).
                # NOTE: the AR+norm+quant path is variant-dependent — a per-TOKEN
                # variant may add a tighter byte + shape-whitelist guard, while a
                # per-GROUP variant (hidden % group == 0) shares this AR guard. Do
                # NOT blanket the tighter guard onto the _quant family — that
                # wrongly excludes the per-group path. Variant selection is an
                # env/source fact recorded by the analyst, not a constant here.
                threshold = int(collective_guard["threshold_bytes"])
                exceeds = tensor_bytes >= threshold
                ref = _short_ref(collective_guard.get("source_ref", ""))
                candidate["_collective_guard"] = {
                    "tensor_bytes": tensor_bytes, "threshold_bytes": threshold,
                    "verdict": "exceeds" if exceeds else "fits",
                    "source_ref": collective_guard.get("source_ref", "")}
                collective_guard_checks.append({
                    "candidate_id": candidate_id, "phase": key[0],
                    "pattern_id": key[1], "tensor_bytes": tensor_bytes,
                    "threshold_bytes": threshold,
                    "verdict": "exceeds" if exceeds else "fits"})
                # A size-guard exceed does NOT change 现成算子(exact): the fused
                # kernel still exists (exact stays 有). It只 means the fused path
                # falls back to split at THIS shape, so the candidate is not
                # applicable here — Top-K reads collective_guard_checks and drops
                # it from the actionable board. Keep a note for the rendered MD.
                if exceeds:
                    candidate["_collective_guard"]["note"] = (
                        "collective tensor~%.0fMiB >= fused-AR guard %.0fMiB "
                        "(%s) -> split fallback at this shape" % (
                            tensor_bytes / 1048576.0, threshold / 1048576.0,
                            ref))
        _validate_api_list(candidate, path, errors)
        # 现成算子 invariant: exact_kernel_status is a binary "is there a ready
        # kernel". If the implementation cites a flag/existing API (A or B) there
        # IS one -> exact must be 有(yes). If it is author-track (new kernel / main
        # rewrite) there is NONE -> exact must be 无(no). This makes 现成算子 a
        # clean function of the class (no adapter sub-shades) and stops A/B being
        # filed as 无 (or C as 有).
        _impl = candidate.get("implementation_class")
        _exact = candidate.get("exact_kernel_status")
        _author = _impl in ("new_helper_kernel", "main_kernel_or_algorithmic")
        if _impl:
            if _author and _exact != "no":
                errors.append(
                    "%s author-track (%s) has no existing kernel -> "
                    "exact_kernel_status(现成算子) must be no" % (path, _impl))
            elif not _author and _exact != "yes":
                errors.append(
                    "%s %s cites an existing kernel -> "
                    "exact_kernel_status(现成算子) must be yes" % (path, _impl))
        # Gate A — a flag-only (A) win must record the flag's ACTUAL routed call
        # signature, and may only claim the ops that signature carries. A *_quant
        # family cannot be A on a flag whose routed signature has no scale/quant/
        # fp8 arg — that flag does not fuse quant (→ B, integrate the quant
        # variant). Generic: checks the recorded signature tokens, not any model.
        if _impl == "existing_flag_or_env":
            sig = candidate.get("flag_routed_signature")
            if (not isinstance(sig, dict)
                    or not sig.get("routed_call_ref")
                    or not sig.get("fused_fn")
                    or not sig.get("arg_signature")):
                errors.append(
                    "%s existing_flag_or_env (A) must record flag_routed_signature "
                    "{routed_call_ref, fused_fn, arg_signature, covers_ops}: the "
                    "actual fused fn the flag routes to and its argument signature "
                    "(so claimed ops are backed by the call, not assumed)" % path)
            else:
                fam = str(candidate.get("family") or "")
                argsig = str(sig.get("arg_signature") or "").lower()
                if "quant" in fam and not any(
                        tok in argsig for tok in _QUANT_SIG_TOKENS):
                    errors.append(
                        "%s family %s implies quant but flag_routed_signature."
                        "arg_signature has no scale/quant/fp8 arg -> the flag does "
                        "not fuse quant; classify B (integrate the quant variant), "
                        "not A" % (path, fam))
        # Gate B — declaring 现成算子=无 (author-track / C) must be an EVIDENCED
        # search conclusion, not an opinion: record the exhaustive symbol search
        # (queries + installed locations + null results) proving no installed
        # kernel implements this op combination. Finding one makes it B.
        if _author:
            searches = candidate.get("absence_search")
            if not isinstance(searches, list) or not searches:
                errors.append(
                    "%s author-track (现成算子=无) must record absence_search "
                    "[{query, location, result}]: the exhaustive symbol search "
                    "over the installed libs proving no kernel does this fusion "
                    "(finding one makes it B, not C)" % path)
            else:
                for si, srch in enumerate(searches):
                    if not (isinstance(srch, dict) and srch.get("query")
                            and srch.get("location")):
                        errors.append(
                            "%s absence_search[%d] must record query + location "
                            "(the installed path searched)" % (path, si))
        # Activation evidence: an exact=yes candidate is realized by config or by
        # code, and which one decides its Top-K tier (A ConfigSweep vs B adapter)
        # and 3.1 routing. Force that decision to be evidence-backed so a
        # flag-only win (e.g. --enable-aiter-allreduce-fusion) is never silently
        # filed as "needs adapter". Cheap and stops the A/B mislabel at the source.
        if candidate.get("exact_kernel_status") == "yes":
            impl = candidate.get("implementation_class")
            seam = (candidate.get("live_call_seam") or "").strip()
            if impl == "existing_flag_or_env" and not seam:
                errors.append(
                    "%s existing_flag_or_env must record the enabling flag/env "
                    "in live_call_seam (e.g. --enable-aiter-allreduce-fusion)"
                    % path)
            elif impl in ("existing_api_needs_adapter",
                          "reference_path_port") and not seam:
                errors.append(
                    "%s exact=yes + %s must record live_call_seam (where the "
                    "code wires in); if instead a flag/env engages it with no "
                    "code, classify existing_flag_or_env (A / ConfigSweep)"
                    % (path, impl))

    summary_order = []
    reported_collective_plans = set()
    for index, summary in enumerate(payload["summary_rows"]):
        path = "summary_rows[%d]" % index
        key = (summary.get("phase"), summary.get("pattern_id"))
        if key not in tables:
            errors.append("%s references unknown table %s" % (path, key))
            continue
        if not summary.get("pattern_short_name"):
            errors.append("%s missing pattern_short_name" % path)
        elif len(summary["pattern_short_name"]) > 40:
            errors.append("%s pattern_short_name is not concise" % path)
        if not summary.get("stage"):
            errors.append("%s missing stage" % path)
        elif len(summary["stage"]) > 80:
            errors.append("%s stage is not concise" % path)
        summary_order.append((
            PHASE_ORDER.get(key[0], 99), table_order[key],
            int(summary.get("order", 0) or 0)))
        row_ids = summary.get("source_row_ids")
        if not isinstance(row_ids, list) or not row_ids:
            errors.append("%s source_row_ids must be non-empty" % path)
            row_ids = []
        chain_total = 0.0
        for row_id in row_ids:
            source = source_rows.get((key[0], key[1], row_id))
            if source is None:
                errors.append("%s unknown source row %s" % (path, row_id))
            else:
                chain_total += float(source.get("duration_us", 0.0) or 0.0)
        if not _close(
                summary.get("current_chain_us_per_layer"), chain_total):
            errors.append("%s current chain duration differs from source rows" % path)
        plans = summary.get("plans")
        if not isinstance(plans, list) or not plans:
            errors.append("%s plans must be non-empty" % path)
            continue
        if len(row_ids) >= 3 and len(plans) < 2 and not summary.get(
                "single_plan_reason"):
            errors.append(
                "%s chain with >=3 rows requires narrow/broad alternatives "
                "or single_plan_reason" % path)
        plan_orders = [
            int(plan.get("order", 0) or 0) for plan in plans]
        if sorted(plan_orders) != list(range(1, len(plans) + 1)):
            errors.append("%s plan order must be consecutive 1..N" % path)
        for plan_index, plan in enumerate(plans):
            plan_path = "%s.plans[%d]" % (path, plan_index)
            candidate_id = plan.get("candidate_id")
            if candidate_id not in candidates_by_id:
                errors.append("%s references unknown candidate %s" % (
                    plan_path, candidate_id))
                continue
            referenced_candidate_ids.add(candidate_id)
            plan_title = plan.get("plan")
            if not plan_title:
                errors.append("%s missing plan text" % plan_path)
            elif (len(plan_title) > 80 or " + " not in plan_title
                  or any(word in plan_title for word in (
                      "融合", "合并", "折入", "把"))):
                errors.append(
                    "%s plan must be a concise canonical chain joined by ' + '"
                    % plan_path)
            if not plan.get("plan_detail"):
                errors.append("%s missing plan_detail" % plan_path)
            _validate_api_list(plan, plan_path, errors)
            candidate = candidates_by_id[candidate_id]
            if not _close(
                    plan.get("current_chain_us_per_layer"),
                    candidate.get("current_chain_us_per_layer")):
                errors.append(
                    "%s current chain duration differs from candidate"
                    % plan_path)
            if plan.get("exact_kernel_status") != candidate.get(
                    "exact_kernel_status"):
                errors.append(
                    "%s exact status differs from candidate" % plan_path)
            if not _close(
                    plan.get("addressable_us_per_layer"),
                    candidate.get("addressable_us_per_layer")):
                errors.append(
                    "%s addressable duration differs from candidate" % plan_path)
            estimate = plan.get("estimated_savings_us", [])
            if estimate and (
                    not isinstance(estimate, list)
                    or len(estimate) not in (1, 2)
                    or any(float(value) < 0 for value in estimate)):
                errors.append(
                    "%s estimated_savings_us must be [] or 1-2 numbers"
                    % plan_path)
            reported_collective_plans.add((
                key, int(plan.get("order", 0) or 0),
                _canonical_plan(plan_title),
                candidate_member_ids.get(candidate_id, ())))

    if summary_order != sorted(summary_order):
        errors.append("summary_rows must be ordered Prefill -> Decode and by table/stage")
    unreported = sorted(set(candidates_by_id) - referenced_candidate_ids)
    if unreported:
        errors.append("summary table omits candidate_ids: %s" % unreported[:10])
    dangling_opportunities = sorted(
        opportunity_candidate_ids - set(candidates_by_id))
    if dangling_opportunities:
        errors.append("stage inventory references unknown candidates: %s" % (
            dangling_opportunities[:10]))
    uncovered_opportunities = sorted(
        opportunity_candidate_ids - referenced_candidate_ids)
    if uncovered_opportunities:
        errors.append("summary table omits inventory opportunities: %s" % (
            uncovered_opportunities[:10]))
    for key, order, plan_title, member_ids in _collective_requirements(table):
        if (key, order, plan_title, member_ids) not in reported_collective_plans:
            errors.append(
                "%s contiguous collective chain %s requires plan %d '%s'" % (
                    key, list(member_ids), order,
                    plan_title.replace("+", " + ")))

    # No silent drop of author-track helpers: a non-donor helper row at or above
    # the floor must be a candidate member or a deferred followup row. Absence of
    # a ready API routes it to kernel authoring, never to omission.
    member_row_ids = set()
    for candidate in payload["candidates"]:
        for member in candidate.get("members", []) or []:
            if member.get("row_id"):
                member_row_ids.add(member["row_id"])
    deferred_row_ids = set()
    for followup in payload.get("required_followups", []) or []:
        for row_id in followup.get("row_ids", []) or []:
            deferred_row_ids.add(row_id)
    dropped_helpers = []
    for (_phase, _pattern, row_id), row in source_rows.items():
        stage = str(row.get("stage") or "").lower()
        duration = float(row.get("duration_us", 0.0) or 0.0)
        if stage in DONOR_STAGES or duration < helper_floor:
            continue
        if row_id in member_row_ids or row_id in deferred_row_ids:
            continue
        dropped_helpers.append((row_id, stage, duration))
    dropped_helpers.sort(key=lambda item: -item[2])
    dropped_helper_us = round(
        sum(duration for _, _, duration in dropped_helpers), 3)
    if dropped_helpers:
        errors.append(
            "%d helper rows >= %.1f us dropped without candidate or "
            "required_followups.row_ids (total %.1f us); first=%s" % (
                len(dropped_helpers), helper_floor, dropped_helper_us,
                [(rid, stage, round(dur, 1))
                 for rid, stage, dur in dropped_helpers[:5]]))

    # Completeness escalation: a big non-donor helper must be an actual
    # CANDIDATE (author-track), not merely deferred to a followup. A deferral
    # (row_ids in required_followups) is NOT enough above the escalate floor.
    escalated_missing = []
    for (_phase, _pattern, row_id), row in source_rows.items():
        stage = str(row.get("stage") or "").lower()
        duration = float(row.get("duration_us", 0.0) or 0.0)
        if stage in DONOR_STAGES or duration < escalate_floor:
            continue
        if row_id in member_row_ids:
            continue
        escalated_missing.append((row_id, stage, duration))
    escalated_missing.sort(key=lambda item: -item[2])
    escalated_missing_us = round(
        sum(duration for _, _, duration in escalated_missing), 3)
    if escalated_missing:
        errors.append(
            "%d helper rows >= %.1f us must be CANDIDATES (author-track), not "
            "deferred to followups (total %.1f us); a deferral does not satisfy "
            "escalation — emit a candidate even when blocked "
            "(readiness=research_only/needs_source_dependency_proof); first=%s"
            % (len(escalated_missing), escalate_floor, escalated_missing_us,
               [(rid, stage, round(dur, 1))
                for rid, stage, dur in escalated_missing[:5]]))

    # Aggregate escalation: small helpers each below the per-row escalate floor
    # can still add up to a big fusion opportunity within one (phase, pattern) —
    # especially in high-layer-count patterns. If the
    # non-candidate sub-floor helpers in one table sum to >= the aggregate floor
    # per layer, they must be surfaced as a candidate (a cluster fusion), not
    # left as scattered followup rows.
    agg_by_group = {}
    for (phase, pattern, row_id), row in source_rows.items():
        stage = str(row.get("stage") or "").lower()
        duration = float(row.get("duration_us", 0.0) or 0.0)
        if stage in DONOR_STAGES or duration >= escalate_floor:
            continue
        if row_id in member_row_ids:
            continue
        agg_by_group.setdefault((phase, pattern), [0.0, 0])
        agg_by_group[(phase, pattern)][0] += duration
        agg_by_group[(phase, pattern)][1] += 1
    agg_violations = sorted(
        [(key, total, count) for key, (total, count) in agg_by_group.items()
         if total >= agg_escalate_floor],
        key=lambda item: -item[1])
    if agg_violations:
        errors.append(
            "deferred sub-floor helpers cluster to >= %.1f us/layer and must "
            "become a candidate (cluster fusion), not scattered followup rows; "
            "groups=%s" % (
                agg_escalate_floor,
                [{"phase": k[0], "pattern_id": k[1],
                  "us_per_layer": round(total, 1), "rows": count}
                 for k, total, count in agg_violations[:5]]))

    # Every collective (all-reduce / reduce-scatter) is a fusion anchor — either
    # its in-place AR+norm(+quant), or the cross-layer tail-AR -> next-layer head
    # boundary. It must be a CANDIDATE member (as donor), not merely deferred to a
    # followup. This is separate from the escalate check, which skips donors.
    collective_not_candidate = []
    for (_phase, _pattern, row_id), row in source_rows.items():
        if str(row.get("stage") or "").lower() != "communication":
            continue
        if row_id in member_row_ids:
            continue
        collective_not_candidate.append(
            (row_id, float(row.get("duration_us", 0.0) or 0.0)))
    collective_not_candidate.sort(key=lambda item: -item[1])
    if collective_not_candidate:
        errors.append(
            "%d collective (all-reduce) rows are not candidate members (only "
            "covered/deferred); every collective is a fusion anchor incl. the "
            "cross-layer tail-AR -> next head boundary and must be a candidate; "
            "first=%s" % (
                len(collective_not_candidate),
                [(rid, round(dur, 1))
                 for rid, dur in collective_not_candidate[:5]]))

    metrics = {
        "source_table_count": len(tables),
        "covered_table_count": len(set(
            (key[0], key[1]) for key in covered)),
        "source_row_count": len(source_rows),
        "covered_source_row_count": len(covered),
        "source_row_coverage_pct": round(
            len(covered) / len(source_rows) * 100.0, 6)
        if source_rows else 0.0,
        "candidate_count": len(candidates_by_id),
        "summary_row_count": len(payload["summary_rows"]),
        "collective_candidate_count": len(collective_guard_checks),
        "collective_guard_exceeds_count": sum(
            1 for c in collective_guard_checks if c["verdict"] == "exceeds"),
        "collective_guard_checks": collective_guard_checks,
        "helper_floor_us": helper_floor,
        "dropped_helper_row_count": len(dropped_helpers),
        "dropped_helper_us": dropped_helper_us,
        "dropped_helper_rows": [
            {"row_id": rid, "stage": stage, "duration_us": round(dur, 3)}
            for rid, stage, dur in dropped_helpers[:20]],
        "collective_not_candidate_count": len(collective_not_candidate),
        "escalate_floor_us": escalate_floor,
        "escalated_missing_row_count": len(escalated_missing),
        "agg_escalate_floor_us": agg_escalate_floor,
        "agg_escalate_violation_count": len(agg_violations),
        "agg_escalate_violations": [
            {"phase": k[0], "pattern_id": k[1],
             "us_per_layer": round(total, 3), "rows": count}
            for k, total, count in agg_violations[:20]],
        "escalated_missing_us": escalated_missing_us,
        "escalated_missing_rows": [
            {"row_id": rid, "stage": stage, "duration_us": round(dur, 3)}
            for rid, stage, dur in escalated_missing[:20]],
        "hbm_bw_bytes_per_us": bw_per_us,
        "savings_ceiling_us_total": round(sum(
            (c.get("_savings") or {}).get("ceiling_us", 0.0) or 0.0
            for c in candidates_by_id.values()), 3),
        "savings_estimate_us_total": round(sum(
            ((c.get("_savings") or {}).get("estimate_us") or 0.0)
            for c in candidates_by_id.values()), 3),
        # Per-candidate savings, machine-readable for Phase 2.2 Top-K ranking.
        "phase_total_forward_us": {
            k: round(v, 3) for k, v in phase_total_forward_us.items()},
        "candidate_savings": [
            {"candidate_id": cid,
             "phase": c.get("phase"), "pattern_id": c.get("pattern_id"),
             "pattern_layer_count": c.get("pattern_layer_count"),
             "ceiling_us": (c.get("_savings") or {}).get("ceiling_us"),
             "estimate_us": (c.get("_savings") or {}).get("estimate_us"),
             "ceiling_count": (c.get("_savings") or {}).get("ceiling_count"),
             "stack_ceiling_us": (c.get("_savings") or {}).get(
                 "stack_ceiling_us"),
             "stack_estimate_us": (c.get("_savings") or {}).get(
                 "stack_estimate_us"),
             "total_forward_pct": (c.get("_savings") or {}).get(
                 "total_forward_pct"),
             "basis": (c.get("_savings") or {}).get("basis"),
             "exact_kernel_status": c.get("exact_kernel_status"),
             "readiness": c.get("readiness"),
             "implementation_class": c.get("implementation_class")}
            for cid, c in candidates_by_id.items()],
    }
    return payload, table, errors, warnings, metrics


def render_markdown(payload, table):
    tables, _, table_order = _semantic_index(table)
    summaries = sorted(payload["summary_rows"], key=lambda item: (
        PHASE_ORDER.get(item.get("phase"), 99),
        table_order.get((item.get("phase"), item.get("pattern_id")), 99),
        int(item.get("order", 0) or 0)))
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in payload["candidates"]}
    lines = [
        "# Kernel Fusion Candidate Analysis",
        "",
        "## Fusion 总表（Prefill → Decode）",
        "",
        "| Phase | Pattern | Stage（时间顺序） | Fusion 方案（按建议顺序） | "
        "当前链耗时 µs/层 | 现成 fusion kernel / API | 现成算子 | "
        "预期节省 µs/层（roofline 估算，单层比例） | "
        "整 forward 预期节省 µs（占该 phase 全层比例） |",
        "|---|---|---|---|---:|---|---|---:|---:|",
    ]
    for summary in summaries:
        plans = sorted(
            summary["plans"], key=lambda plan: int(plan.get("order", 0) or 0))
        lines.append(
            "| {phase} | {pattern} | {stage} | {plans} | {current} | "
            "{apis} | {exact} | {savings} | {fwd} |".format(
                phase=_escape(summary["phase"].capitalize()),
                pattern=_escape(summary["pattern_short_name"]),
                stage=_escape(summary["stage"]),
                plans=_escape(_numbered([
                    plan["plan"] for plan in plans])),
                current=_escape(_numbered([
                    "%.3f" % float(plan["current_chain_us_per_layer"])
                    for plan in plans])),
                apis=_escape(_numbered([
                    _api_summary(plan) for plan in plans])),
                exact=_escape(_numbered([
                    "有" if plan["exact_kernel_status"] == "yes" else "无"
                    for plan in plans])),
                savings=_escape(_numbered([
                    _savings_text(
                        plan, float(tables[(
                            summary["phase"], summary["pattern_id"])].get(
                                "layer_total_us", 0.0) or 0.0),
                        candidates.get(plan.get("candidate_id")))
                    for plan in plans])),
                fwd=_escape(_numbered([
                    _forward_savings_text(
                        candidates.get(plan.get("candidate_id")))
                    for plan in plans]))))
    lines.extend([
        "",
        "说明：预期节省列 = `roofline 估算`——复用 GEAK roofline 的每-gfx HBM 带宽，"
        "把「融合消除的访存往返 + launch」折算成 µs，是工程估算非实测，最终以 "
        "benchmark 为准（`*` 表示部分行缺 shape、估算偏乐观；`≤X 估算不可用` 表示无 "
        "shape/带宽时退回可寻址上限）。可寻址上限（Clean Trace 实测、扣除锚点/donor "
        "的硬天花板）见每条候选明细。主 GEMM/Attention/MoE/Collective 及每条链最重的"
        "成分作为锚点，不计入收益。",
        "",
        "## 候选证据明细",
        "",
    ])
    for summary in summaries:
        lines.extend([
            "### %s / %s / %s" % (
                summary["phase"].upper(),
                summary.get("pattern_display_name") or summary["pattern_id"],
                summary["stage"]),
            "",
        ])
        for plan in sorted(
                summary["plans"],
                key=lambda item: int(item.get("order", 0) or 0)):
            candidate = candidates[plan["candidate_id"]]
            lines.extend([
                "#### `%s`" % candidate["candidate_id"],
                "",
                "- plan: `%s`" % plan["plan"],
                "- detail: " + plan["plan_detail"],
                "- readiness: `%s`" % candidate["readiness"],
                "- implementation: `%s`" % candidate["implementation_class"],
                "- exact kernel: `%s`" % (
                    "有" if candidate["exact_kernel_status"] == "yes"
                    else "无"),
                "- chain/addressable: `%.3f / %.3f µs/层`" % (
                    float(candidate["current_chain_us_per_layer"]),
                    float(candidate["addressable_us_per_layer"])),
                "- savings: 上限 `%.3f` / roofline 估算 `%s` µs/层（basis=%s）" % (
                    float((candidate.get("_savings") or {}).get(
                        "ceiling_us", 0.0) or 0.0),
                    ("%.3f" % (candidate.get("_savings") or {})["estimate_us"])
                    if (candidate.get("_savings") or {}).get(
                        "estimate_us") is not None else "n/a",
                    (candidate.get("_savings") or {}).get("basis", "n/a")),
                "- 整 forward: ×%s 层 → stack 估算 `%s` µs（占该 phase 全 forward `%s`）" % (
                    (candidate.get("_savings") or {}).get("ceiling_count", "?"),
                    ("%.1f" % (candidate.get("_savings") or {})[
                        "stack_estimate_us"])
                    if (candidate.get("_savings") or {}).get(
                        "stack_estimate_us") is not None else "n/a",
                    ("%.3f%%" % (candidate.get("_savings") or {})[
                        "total_forward_pct"])
                    if (candidate.get("_savings") or {}).get(
                        "total_forward_pct") is not None else "n/a"),
                "- members: " + ", ".join(
                    "`pos%s %s %s (%s, %.3fµs)`" % (
                        member["pos"], member.get("stage", ""),
                        member["row_id"], member.get("evidence_level", "?"),
                        float(member["duration_us"]))
                    for member in candidate["members"]),
                "- existing API: " + _api_detail(candidate),
                "- exact reason: " + candidate.get(
                    "exact_reason", "current environment fully verified"),
            ] + ([
                "- collective guard: tensor≈%.1f MiB vs fused-AR 阈值 %.1f MiB"
                "（%s）→ %s" % (
                    candidate["_collective_guard"]["tensor_bytes"] / 1048576.0,
                    candidate["_collective_guard"]["threshold_bytes"]
                    / 1048576.0,
                    _short_ref(candidate["_collective_guard"]["source_ref"]),
                    candidate["_collective_guard"]["verdict"])
            ] if candidate.get("_collective_guard") else []) + [
                "- risks: " + (
                    "；".join(map(str, candidate.get("risks", []))) or "none"),
                "- validation: " + (
                    "；".join(map(str, candidate.get(
                        "validation_requirements", []))) or "none"),
                "",
            ])
    return "\n".join(lines) + "\n"


def run(semantic_table_path, candidates_path, out_md, result_json,
        helper_floor=DEFAULT_HELPER_FLOOR_US,
        escalate_floor=DEFAULT_ESCALATE_FLOOR_US,
        agg_escalate_floor=DEFAULT_AGG_ESCALATE_FLOOR_US):
    payload, table, errors, warnings, metrics = validate(
        semantic_table_path, candidates_path, helper_floor, escalate_floor,
        agg_escalate_floor)
    result = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "semantic_table_json": os.path.abspath(semantic_table_path),
        "fusion_candidates_json": os.path.abspath(candidates_path),
        "fusion_candidates_md": os.path.abspath(out_md),
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(result_json)), exist_ok=True)
    with open(result_json, "w") as fh:
        json.dump(result, fh, indent=2)
    if not errors:
        with open(out_md, "w") as fh:
            fh.write(render_markdown(payload, table))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-table", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument(
        "--helper-floor", type=float, default=DEFAULT_HELPER_FLOOR_US,
        help="min duration (us) a non-donor helper row must clear before it "
             "must be a candidate member or a deferred followup row")
    parser.add_argument(
        "--escalate-floor", type=float, default=DEFAULT_ESCALATE_FLOOR_US,
        help="min duration (us) a non-donor helper row must clear before it "
             "must be an actual candidate (a followup deferral is not enough)")
    parser.add_argument(
        "--agg-escalate-floor", type=float,
        default=DEFAULT_AGG_ESCALATE_FLOOR_US,
        help="min per-layer sum (us) of sub-floor non-candidate helpers in one "
             "(phase, pattern) before they must become a cluster candidate")
    args = parser.parse_args()
    result = run(
        args.semantic_table, args.candidates,
        args.out_md, args.result_json, args.helper_floor, args.escalate_floor,
        args.agg_escalate_floor)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
