#!/usr/bin/env python3
"""Validate Fusion 2.1 facts/coverage and render the mandatory total table."""
import argparse
import json
import math
import os
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
API_COVERAGE = {"full", "partial", "similar"}
API_SOURCE_KIND = {
    "runtime_environment", "runtime_source", "perf_knowledge"}


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
        reason = plan.get("exact_reason", "当前环境无完整覆盖")
        rendered.append("无 exact API（%s）" % reason)
    return "；".join(rendered) if rendered else "无 exact API"


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


def _savings_text(plan, layer_total_us):
    estimate = plan.get("estimated_savings_us") or []
    if len(estimate) == 2:
        low, high = map(float, estimate)
        pct_low = low / layer_total_us * 100.0 if layer_total_us else 0.0
        pct_high = high / layer_total_us * 100.0 if layer_total_us else 0.0
        return "工程预估 %.2f–%.2f µs/层（%.2f%%–%.2f%%）" % (
            low, high, pct_low, pct_high)
    if len(estimate) == 1:
        value = float(estimate[0])
        pct = value / layer_total_us * 100.0 if layer_total_us else 0.0
        return "工程预估 %.2f µs/层（%.2f%%）" % (value, pct)
    ceiling = float(plan.get("addressable_us_per_layer", 0.0) or 0.0)
    pct = ceiling / layer_total_us * 100.0 if layer_total_us else 0.0
    return "最高 %.2f µs/层（%.2f%%）" % (ceiling, pct)


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


def validate(semantic_table_path, candidates_path):
    table = _load(semantic_table_path)
    payload = _load(candidates_path)
    tables, source_rows, table_order = _semantic_index(table)
    errors = []
    warnings = []

    if payload.get("phase") != "generate_plans":
        errors.append("top-level phase must be generate_plans")
    if not isinstance(payload.get("stage_inventory"), list):
        errors.append("stage_inventory must be a list")
    if not isinstance(payload.get("summary_rows"), list):
        errors.append("summary_rows must be a list")
    if not isinstance(payload.get("candidates"), list):
        errors.append("candidates must be a list")
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

    candidates_by_id = {}
    candidate_member_ids = {}
    referenced_candidate_ids = set()
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
            if previous_pos is not None and member.get("pos", -1) <= previous_pos:
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
        layer_count = int(candidate.get("pattern_layer_count", 0) or 0)
        if layer_count != int(
                tables[key].get("pattern_layer_count", 0) or 0):
            errors.append("%s pattern_layer_count differs from semantic table" % path)
        expected_stack = removable_total * layer_count
        if not _close(
                candidate.get("stack_addressable_ceiling_us"),
                expected_stack):
            errors.append(
                "%s stack_addressable_ceiling_us formula mismatch" % path)
        _validate_api_list(candidate, path, errors)

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
        "当前链耗时 µs/层 | 现成 fusion kernel / API | Exact Kernel | "
        "预期节省 µs/层（单层比例） |",
        "|---|---|---|---|---:|---|---|---:|",
    ]
    for summary in summaries:
        plans = sorted(
            summary["plans"], key=lambda plan: int(plan.get("order", 0) or 0))
        lines.append(
            "| {phase} | {pattern} | {stage} | {plans} | {current} | "
            "{apis} | {exact} | {savings} |".format(
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
                                "layer_total_us", 0.0) or 0.0))
                    for plan in plans]))))
    lines.extend([
        "",
        "说明：当前链耗时与可寻址上限来自 Clean Trace。预期节省若出现，"
        "必须明确标记为工程估算而非实测；主 GEMM/Attention/MoE/Collective "
        "donor 本体默认不计入可消除收益。",
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
                "- members: " + ", ".join(
                    "`pos%s %s %s (%s, %.3fµs)`" % (
                        member["pos"], member.get("stage", ""),
                        member["row_id"], member.get("evidence_level", "?"),
                        float(member["duration_us"]))
                    for member in candidate["members"]),
                "- existing API: " + _api_detail(candidate),
                "- exact reason: " + candidate.get(
                    "exact_reason", "current environment fully verified"),
                "- risks: " + (
                    "；".join(map(str, candidate.get("risks", []))) or "none"),
                "- validation: " + (
                    "；".join(map(str, candidate.get(
                        "validation_requirements", []))) or "none"),
                "",
            ])
    return "\n".join(lines) + "\n"


def run(semantic_table_path, candidates_path, out_md, result_json):
    payload, table, errors, warnings, metrics = validate(
        semantic_table_path, candidates_path)
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
    args = parser.parse_args()
    result = run(
        args.semantic_table, args.candidates,
        args.out_md, args.result_json)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
