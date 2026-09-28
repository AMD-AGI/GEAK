#!/usr/bin/env python3
"""Phase 1: render the semantic layer table as a human report.

Phase 1 produced `pattern_layer_kernel_table.json` and nothing readable. Everything
a reader needed -- which phases were actually analysed, whether decode resolved its
shapes, where the fusion surface is -- lived inside a 5 MB JSON, so in practice
nobody looked, and a table with `decode 0/64 shapes resolved` travelled all the way
to apply-back without anyone noticing (DSR1, 2026-08-26).

This renders the parts that decide what happens downstream:

  * provenance -- which traces, which sha, which phases, how shapes were resolved
  * phase coverage, MEASURED off the rows rather than read off the declared record
    (a later step can graft shapes onto the table and leave the record stale)
  * per-pattern cost, the stage mix inside each layer, and EVERY row of every
    layer table in trace order with its measured `input_dims` / `input_types`.
    Not a top-N, and donor rows included. This report is the only human-readable
    artifact that travels between phases, so whatever it drops is, in practice,
    dropped: it used to render `shape` as a bare `✓` and never touch
    `input_types` at all, and downstream then rebuilt shapes by multiplying
    config fields (`input_tokens` x `kv_lora_rank`). That silently loses the
    batch axis and the 64 rope columns, the kernel falls into a degenerate path,
    and the resulting parity failure plus distorted timing is indistinguishable
    from "this fusion is not worth it" (DSR1, 2026-08-31: two fusions worth
    +20.6% and +14.0% e2e were blocked this way).
  * the FUSIBLE REGIONS -- contiguous non-donor runs, i.e. the fusion surface the
    Phase 2.1 harness will hold the candidate set against. Seeing them here, one
    phase before candidates exist, is the point: it turns "did we find everything?"
    into a number you can check.

It computes no new facts; it is a view of the table. Nothing here gates anything.

Usage:
  python3 semantic_report.py --semantic-table pattern_layer_kernel_table.json \
      --out-md 01_SEMANTIC.md [--out-json semantic_report.json] [--helper-floor 5.0]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_candidate_harness import DONOR_STAGES, _fusible_regions

# A decode-only or prefill-only table is a legitimate artifact, but it is never a
# legitimate SILENCE. The DSR1 failure was a table with no decode tables at all and no
# declared record saying so -- so the report showed one tidy prefill row and read as
# complete. Both halves are always named; the one with no tables is named as absent.
EXPECTED_PHASES = ("prefill", "decode")
PHASE_ORDER = {phase: index for index, phase in enumerate(EXPECTED_PHASES)}


def _pattern_order(pattern):
    phase = str(pattern.get("phase") or "")
    return (PHASE_ORDER.get(phase, len(PHASE_ORDER)), phase,
            str(pattern.get("pattern_id")))


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _measure_phases(table):
    """Shape resolution per phase, off the ROWS. See the module docstring: the
    declared `phase_coverage` can legitimately go stale under a later graft."""
    measured = {}
    for item in table.get("tables", []):
        phase = str(item.get("phase") or "")
        stat = measured.setdefault(phase, {"rows": 0, "resolved": 0, "us": 0.0,
                                           "patterns": 0})
        stat["patterns"] += 1
        for row in item.get("rows", []):
            stat["rows"] += 1
            stat["us"] += float(row.get("duration_us", 0.0) or 0.0)
            if (row.get("shape") or {}).get("input_dims"):
                stat["resolved"] += 1
    return measured


def _stage_mix(item):
    mix = {}
    for row in item.get("rows", []):
        stage = str(row.get("stage") or "?")
        slot = mix.setdefault(stage, {"count": 0, "us": 0.0})
        slot["count"] += 1
        slot["us"] += float(row.get("duration_us", 0.0) or 0.0)
    return mix


def _row_order(row):
    """Trace order. Fusion is an adjacency question, so `pos` is the only order
    that lets a reader see which rows sit next to which."""
    for key in ("pos", "device_seq_index", "raw_event_index"):
        val = row.get(key)
        if isinstance(val, (int, float)):
            return (0, float(val))
    return (1, 0.0)


def _op_name(parent_operator):
    """`parent_operator` is either a plain aten/aiter name or a mapping record
    carrying it under `canonical_op`. The name is what tells a reader what the
    row actually is (`aten::bmm`, `aten::cat`); the rest of the record is
    provenance and belongs in the JSON, not in a table cell."""
    if isinstance(parent_operator, dict):
        return parent_operator.get("canonical_op") or parent_operator.get("op") or ""
    return parent_operator or ""


def _fmt_dims(dims):
    """`input_dims` verbatim, one operand per group, in operand order.

    A TensorList operand (aten::cat) nests one level and is rendered `{a; b}` so
    that `[[4,16,512],[4,16,64]]` does not read as two separate operands -- those
    two are the halves that concatenate to 576, and losing that grouping is
    exactly how a 576-wide kernel ends up being fed 512.
    A dimensionless operand (a Scalar) is `·`; it holds its slot so the dims line
    up positionally with `input_types`.
    """
    if dims is None:
        return "—"
    if not isinstance(dims, list):
        return _esc(dims)
    if not dims:
        return "—"

    def one(operand):
        if not isinstance(operand, list):
            return str(operand)
        if not operand:
            return "·"
        if all(isinstance(v, (int, float)) for v in operand):
            return "[%s]" % ",".join(str(int(v)) for v in operand)
        return "{%s}" % "; ".join(one(v) for v in operand)

    return _esc(" ".join(one(d) for d in dims))


def _fmt_types(types):
    """`input_types` verbatim minus the `c10::` noise. The fp8 variant (`fnuz` vs
    `fn`) rides in this column and decides parity on its own."""
    if not types:
        return "—"
    if not isinstance(types, list):
        return _esc(types)
    return _esc(",".join(str(t).replace("c10::", "") or "·" for t in types))


def _fmt_named_operands(operands):
    """Render schema-backed tensor operands without replacing raw dimensions."""
    values = []
    for operand in operands or []:
        shape = operand.get("shape")
        if not isinstance(shape, list) or not shape:
            continue
        name = operand.get("schema_name") or (
            "arg_%s" % operand.get("arg_index", "?"))
        dtype = str(operand.get("dtype") or "Tensor").replace("c10::", "")
        direction = str(operand.get("direction") or "unresolved")
        suffix = "{%s}" % direction if direction not in (
            "input", "output") else ""
        values.append("%s%s=%s[%s]" % (
            name, suffix, dtype,
            "×".join(str(value) for value in shape)))
    return _esc("; ".join(values)) if values else "—"


def build(table_path, helper_floor=5.0, top_rows=8):
    table = _load(table_path)
    measured = _measure_phases(table)
    declared = ((table.get("phase_coverage") or {})
                .get("shape_resolution_by_phase") or {})

    stale = []
    for phase, stat in sorted(measured.items()):
        d = declared.get(phase) or {}
        if d and int(d.get("resolved", -1)) != stat["resolved"]:
            stale.append("%s: record says %s/%s, rows say %d/%d"
                         % (phase, d.get("resolved"), d.get("rows"),
                            stat["resolved"], stat["rows"]))

    patterns = []
    for item in table.get("tables", []):
        phase = str(item.get("phase") or "")
        rows = item.get("rows", [])
        total = float(item.get("layer_total_us")
                      or sum(float(r.get("duration_us", 0.0) or 0.0) for r in rows))
        donor_us = sum(float(r.get("duration_us", 0.0) or 0.0) for r in rows
                       if str(r.get("stage") or "").lower() in DONOR_STAGES)
        patterns.append({
            "phase": phase,
            "pattern_id": item.get("pattern_id"),
            "display_name": item.get("pattern_display_name"),
            "layer_count": item.get("pattern_layer_count"),
            "representative_layer_id": item.get("representative_layer_id"),
            "body_signature": (
                (item.get("structural_context") or {}).get(
                    "body_signature")
                or item.get("structural_context") or {}),
            "representative_instance_context": item.get(
                "representative_instance_context") or {},
            "boundary_evidence": item.get("boundary_evidence") or {},
            "rows": len(rows),
            "layer_total_us": round(total, 3),
            "donor_us": round(donor_us, 3),
            "non_donor_us": round(total - donor_us, 3),
            "non_donor_pct": round(100.0 * (total - donor_us) / total, 2) if total else 0.0,
            "phase_layer_us": round(total * float(item.get("pattern_layer_count") or 1), 3),
            "stage_mix": {k: {"count": v["count"], "us": round(v["us"], 3)}
                          for k, v in sorted(_stage_mix(item).items(),
                                             key=lambda kv: -kv[1]["us"])},
            "top_non_donor_rows": [
                {"row_id": r.get("row_id"), "stage": r.get("stage"),
                 "short_name": r.get("short_name"),
                 "duration_us": round(float(r.get("duration_us", 0.0) or 0.0), 3),
                 "provider": r.get("provider"),
                 "shape_resolved": bool((r.get("shape") or {}).get("input_dims"))}
                for r in sorted(rows,
                                key=lambda r: -float(r.get("duration_us", 0.0) or 0.0))
                if str(r.get("stage") or "").lower() not in DONOR_STAGES][:top_rows],
            # EVERY row, in trace order, carrying the shapes and dtypes verbatim.
            # Not a top-N and not non-donor-only: see the module docstring.
            "row_detail": [
                {"pos": r.get("pos"),
                 "row_id": r.get("row_id"),
                 "stage": r.get("stage"),
                 "donor": str(r.get("stage") or "").lower() in DONOR_STAGES,
                 "short_name": r.get("short_name"),
                 "parent_operator": r.get("parent_operator"),
                 "duration_us": round(float(r.get("duration_us", 0.0) or 0.0), 3),
                 "provider": r.get("provider"),
                 "shape_source": (r.get("shape") or {}).get("source"),
                 "input_dims": (r.get("shape") or {}).get("input_dims"),
                 "input_types": (r.get("shape") or {}).get("input_types"),
                 "named_operands": (((r.get("shape") or {}).get(
                     "kernel_shape") or {}).get("operands") or [])}
                for r in sorted(rows, key=_row_order)],
        })

    regions = []
    for (phase, pattern), row_ids, stages, total_us in _fusible_regions(
            table, helper_floor):
        regions.append({"phase": phase, "pattern_id": pattern,
                        "row_ids": list(row_ids), "stages": stages,
                        "rows": len(row_ids), "total_us": total_us})
    regions.sort(key=lambda r: -r["total_us"])

    # Per-phase forward totals: a layer table is ONE layer, so the phase cost is
    # the layer cost times how many layers share that pattern.
    phase_forward = {}
    for p in patterns:
        phase_forward[p["phase"]] = round(
            phase_forward.get(p["phase"], 0.0) + p["phase_layer_us"], 3)

    declared_absent = ((table.get("phase_coverage") or {})
                       .get("phases_absent_from_tables") or [])
    absent = [ph for ph in EXPECTED_PHASES if ph not in measured]
    for ph in declared_absent:
        if ph not in absent:
            absent.append(str(ph))

    return {
        "schema_version": 1,
        "phase": "semantic_report",
        "phases_absent": absent,
        "semantic_table": os.path.abspath(table_path),
        "trace_paths": table.get("trace_paths") or (
            [table["trace_path"]] if table.get("trace_path") else []),
        "trace_sha256_by_path": table.get("trace_sha256_by_path") or {},
        "patterns_path": table.get("patterns_path"),
        "table_phases": table.get("table_phases") or sorted(measured),
        "table_phases_requested": table.get("table_phases_requested"),
        "semantics_provenance": table.get("semantics_provenance") or {},
        "phase_coverage_declared": table.get("phase_coverage") or {},
        "phase_coverage_measured": {
            k: {"rows": v["rows"], "resolved": v["resolved"],
                "resolved_fraction": round(v["resolved"] / v["rows"], 4) if v["rows"] else 0.0,
                "patterns": v["patterns"], "layer_us": round(v["us"], 3)}
            for k, v in sorted(measured.items())},
        "stale_record_notes": stale,
        "phase_forward_us": phase_forward,
        "patterns": patterns,
        "helper_floor_us": helper_floor,
        "fusible_regions": regions,
        "fusible_region_count": len(regions),
        "fusible_region_us": round(sum(r["total_us"] for r in regions), 3),
    }


def _esc(value):
    return str(value if value is not None else "").replace("|", "\\|")


def render_markdown(rep):
    lines = ["# 语义层报告 (Phase 1) — pattern × layer × kernel", ""]
    lines.append(
        "本报告是 `pattern_layer_kernel_table.json` 的**可读视图**：它不产生任何新事实，"
        "只把决定下游走向的部分摆出来——分析了哪些阶段、shape 解出来没有、"
        "以及**可融合面（fusible regions）在哪里**。")
    lines.append("")

    lines.append("## 1. 覆盖与证据（先看这里）")
    lines.append("")
    meas = rep["phase_coverage_measured"]
    lines.append("| 阶段 | pattern 数 | 行数 | shape 已解析 | 解析率 | 单层耗时 | 该阶段 forward |")
    lines.append("|:--:|---:|---:|---:|---:|---:|---:|")
    for phase, v in meas.items():
        lines.append("| **%s** | %d | %d | %d | %.0f%% | %.1f µs | %.0f µs |" % (
            phase, v["patterns"], v["rows"], v["resolved"],
            100.0 * v["resolved_fraction"], v["layer_us"],
            rep["phase_forward_us"].get(phase, 0.0)))
    lines.append("")
    absent = rep.get("phases_absent") or []
    if absent:
        lines.append(
            "> 🔴 **表里没有的阶段**：%s。这些阶段完全没被分析——"
            "它们的融合机会一条也不会被发现，而下游会把「零候选」读成「没机会」。"
            "先确认 trace 是否分开抓、decode 那份有没有进表。" % "、".join(absent))
        lines.append("")
    for phase, v in meas.items():
        if v["rows"] and not v["resolved"]:
            lines.append(
                "> 🔴 **%s 的 shape 一条都没解析出来**（%d 行）。CUDA-graph replay 下 "
                "decode replay 几乎不发 `nn.Module` span，需要 graph-construction "
                "capture 补 shape，否则该阶段"
                "只有序列、没有形状，无法出候选。" % (phase, v["rows"]))
            lines.append("")
    if rep.get("stale_record_notes"):
        lines.append("> ⚠️ 声明的 `phase_coverage` 与行内实测不一致（以**实测**为准）：%s。"
                     "通常是后续步骤把 shape 嫁接进了表、但没更新记录。"
                     % "；".join(rep["stale_record_notes"]))
        lines.append("")

    lines.append("## 2. 证据来源（provenance）")
    lines.append("")
    for path in rep.get("trace_paths") or []:
        sha = (rep.get("trace_sha256_by_path") or {}).get(path, "")
        lines.append("- trace `%s`%s" % (path, ("  \n  sha256 `%s`" % sha) if sha else ""))
    if rep.get("patterns_path"):
        lines.append("- 结构 pattern：`%s`" % rep["patterns_path"])
    prov = rep.get("semantics_provenance") or {}
    for key, val in prov.items():
        lines.append("- %s：%s" % (key, val))
    lines.append("")

    lines.append("## 3. 层结构与成本")
    lines.append("")
    lines.append("| 阶段 | pattern | 名称 | 层数 | 单层 µs | 其中非-donor | 非-donor 占比 | 该 pattern forward |")
    lines.append("|:--:|---|---|---:|---:|---:|---:|---:|")
    for p in sorted(rep["patterns"], key=_pattern_order):
        lines.append("| %s | `%s` | %s | %s | %.1f | %.1f | %.1f%% | %.0f µs |" % (
            _esc(p["phase"]), _esc(p["pattern_id"]), _esc(p["display_name"]),
            _esc(p["layer_count"]), p["layer_total_us"], p["non_donor_us"],
            p["non_donor_pct"], p["phase_layer_us"]))
    lines.append("")
    lines.append(
        "**非-donor** = 不是 GEMM / attention / MoE / collective 的那部分（donor 阶段：`%s`）。"
        "融合只能发生在 donor 之间的空隙里，所以这一列就是融合能触及的上限。"
        % "、".join(sorted(DONOR_STAGES)))
    lines.append("")

    for p in sorted(rep["patterns"], key=_pattern_order):
        lines.append("### %s / `%s` — stage 分布" % (p["phase"], p["pattern_id"]))
        lines.append("")
        lines.append("- body signature：`%s`" % _esc(json.dumps(
            p.get("body_signature") or {}, sort_keys=True,
            ensure_ascii=False)))
        lines.append("- representative instance context：`%s`" % _esc(
            json.dumps(p.get("representative_instance_context") or {},
                       sort_keys=True, ensure_ascii=False)))
        lines.append("- layer boundary evidence：`%s`" % _esc(json.dumps(
            p.get("boundary_evidence") or {}, sort_keys=True,
            ensure_ascii=False)))
        lines.append("")
        lines.append("| stage | 次数 | µs/层 | 占单层 |")
        lines.append("|---|---:|---:|---:|")
        for stage, v in p["stage_mix"].items():
            pct = (100.0 * v["us"] / p["layer_total_us"]) if p["layer_total_us"] else 0.0
            mark = " *(donor)*" if stage.lower() in DONOR_STAGES else ""
            lines.append("| %s%s | %d | %.2f | %.1f%% |"
                         % (_esc(stage), mark, v["count"], v["us"], pct))
        lines.append("")
        detail = p.get("row_detail") or []
        if detail:
            unresolved = [r for r in detail if not r.get("input_dims")]
            lines.append(
                "这一层的**全部 %d 行**，按 trace 顺序（`pos`），donor 行也在内。"
                "`input_dims` / `dtypes` 是 trace 里量到的原文，"
                "`具名 operands` 只在 operator schema 唯一匹配时显示参数名；"
                "**下游构造融合参考侧和单侧 microbench 时必须从这里抄，不要从配置字段拼**。%s"
                % (len(detail),
                   ("其中 **%d 行没有解出 shape**（`—`）——它们对下游是盲区。"
                    % len(unresolved)) if unresolved else "全部 %d 行都解出了 shape。" % len(detail)))
            lines.append("")
            lines.append("| pos | row | stage | kernel | 算子 | µs/层 | input_dims | dtypes | 具名 operands |")
            lines.append("|---:|---|---|---|---|---:|---|---|---|")
            for r in detail:
                mark = " *(donor)*" if r.get("donor") else ""
                lines.append("| %s | `%s` | %s%s | `%s` | `%s` | %.2f | %s | %s | %s |" % (
                    _esc(r.get("pos")), _esc(r.get("row_id")),
                    _esc(r.get("stage")), mark,
                    _esc(r.get("short_name")), _esc(_op_name(r.get("parent_operator"))),
                    r.get("duration_us") or 0.0,
                    _fmt_dims(r.get("input_dims")), _fmt_types(r.get("input_types")),
                    _fmt_named_operands(r.get("named_operands"))))
            lines.append("")

    lines.append("## 4. 可融合面：fusible regions")
    lines.append("")
    lines.append(
        "一次融合跨不过 donor，所以 donor 把每张层表切成若干**独立区间**。"
        "每段 ≥2 行、同 stream 连续、合计 ≥ %.1f µs/层 的非-donor 连续段就是一个"
        "**可融合区间**，它的最宽形态就是整段。共 **%d** 个区间，合计 **%.1f µs/层**。"
        % (rep["helper_floor_us"], rep["fusible_region_count"], rep["fusible_region_us"]))
    lines.append("")
    lines.append(
        "Phase 2.1 会按这张表核对候选集：每个区间都必须有一条覆盖**整段**的候选，"
        "或一条写明理由的延后。这就是让「候选集」变成表的函数、而不是每次跑各凭判断的地方。")
    lines.append("")
    lines.append(
        "**每个 `row_id` 的 shape 和 dtype 在第 3 节该 pattern 的逐行表里，按 `pos` 排好。**"
        "构造候选、构造 split 参考、跑单侧 microbench，用的都必须是那里的 `input_dims` 原文。"
        "从 `structural_context` 的配置字段（`kv_lora_rank`、`input_tokens`、`num_attention_heads`）"
        "拼一个 shape 出来，会拼掉 batch 轴、拼掉 rope 那 64 列——kernel 于是走退化路径，"
        "parity 挂、计时失真，而失真后的数字长得和「这个融合不划算」一模一样。")
    lines.append("")
    lines.append("| # | 阶段 | pattern | 行数 | stages | µs/层 | row_ids |")
    lines.append("|---:|:--:|---|---:|---|---:|---|")
    for i, r in enumerate(rep["fusible_regions"], 1):
        lines.append("| %d | %s | `%s` | %d | %s | %.2f | %s |" % (
            i, _esc(r["phase"]), _esc(r["pattern_id"]), r["rows"],
            _esc(r["stages"]), r["total_us"],
            _esc(", ".join(r["row_ids"]))))
    lines.append("")
    lines.append(
        "说明：本报告只是表的视图，不 gate 任何东西。真正的把关在 Phase 2.1 "
        "(`fusion_candidate_harness.py`)——它用同一份区间定义核对候选集是否完整。")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(table_path, out_md, out_json=None, helper_floor=5.0):
    rep = build(table_path, helper_floor)
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write(render_markdown(rep))
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(rep, fh, indent=2, ensure_ascii=False)
    return rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-table", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--helper-floor", type=float, default=5.0,
                        help="µs/layer floor for a fusible region (match the value "
                             "passed to fusion_candidate_harness.py)")
    args = parser.parse_args()
    rep = run(args.semantic_table, args.out_md, args.out_json, args.helper_floor)
    print(json.dumps({
        "phases": {k: "%d/%d shapes" % (v["resolved"], v["rows"])
                   for k, v in rep["phase_coverage_measured"].items()},
        "patterns": len(rep["patterns"]),
        "fusible_regions": rep["fusible_region_count"],
        "fusible_region_us": rep["fusible_region_us"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
