#!/usr/bin/env python3
"""Apply-back coverage gate for the KernelFusion stage.

Deterministic validator of what the `fusion_integrator` role actually did with the
Phase 2.2 Top-K board. It runs no server and measures nothing of its own; it answers
one question the rest of the pipeline could not:

    every row the board put on the table -- did it end somewhere explicit?

WHY THIS EXISTS
---------------
Phase 2.1 gates the candidate SET against the table. Phase 3.0 gates the 单侧
verdicts against the candidate set. Both are recall gates now. Apply-back had
neither: `FUSION_APPLY_SCHEMA` requires `accepted_fusions` and nothing else, so a run
that integrated 3 fusions and quietly walked past the other 9 returned
`accepted_fusions: [3]` and read as a success. On DSR1 2026-08-26, 26 of 42
candidates reached the end of the pipeline with no disposition of any kind -- not
applied, not rejected, not deferred. Just absent.

So the Top-K `execution_list` is the denominator, and every `exec_id` must end in
exactly one of:

  * `applied`              -- integrated, gated, kept
  * `blocked`              -- tried or ruled out, WITH a reason (wire failure, accuracy
                              gate, no 单侧 win, kernel unavailable...)
  * `blocked_by_exclusion` -- a conflicting entry in its exclusive group was applied.
                              Derived, not asserted: only entries that genuinely
                              conflict with an APPLIED entry get this.
  * `deferred_with_reason` -- knowingly left for next round, WITH a reason. This is
                              INVALID for an in-budget tier-A/B row whose unit-side
                              status passed: that row must be applied or blocked by an
                              actual apply-back attempt.
  * `deferred_budget`      -- past `--budget` in board rank order. Legitimate, but
                              recorded and named. A budget is a decision to stop, and
                              a decision to stop is a thing the reader must SEE.
  * `unaccounted`          -- nobody said anything → the gate FAILS.

`unaccounted` is the whole point. "Not mentioned" is not "skipped"; it is a hole in
the report, and it must be red.

BUDGET TRUNCATION
-----------------
`--budget N` is not a licence to ignore the tail. It only excuses entries whose board
RANK is beyond N. If a rank-2 entry is unaccounted while rank-9 was applied, the
budget is not the explanation and the gate still fails -- the loop skipped something
inside its own budget.

Usage:
  python3 fusion_applyback_harness.py --topk fusion_topk.json \
      --apply apply_result.json [--unitside fusion_unitside.json] \
      --out-md FUSION_APPLYBACK.md --out-json fusion_applyback.json \
      [--budget 6] [--allow-partial-coverage] [--waive <exec_id>=<reason> ...]
"""
import argparse
import json
import os
import sys


ACCOUNTED = {"applied", "blocked", "blocked_by_exclusion",
             "deferred_with_reason", "deferred_budget"}
EXPLICIT = {"applied", "blocked", "deferred_with_reason"}
APPLYBACK_ELIGIBLE = {"pass", "equivalent_pass", "subsumed_pass"}


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _norm(text):
    """Lowercase, collapse everything that is not alnum/underscore to a separator."""
    out = []
    for ch in str(text or "").lower():
        out.append(ch if (ch.isalnum() or ch == "_") else " ")
    return "_".join(t for t in "".join(out).split() if t)


def _keys(exec_id=None, candidate_ids=(), handle=None, action=None):
    """The identity set an execution row and an apply record are matched on.

    Deliberately NOT a fuzzy name match. An apply record has to name an exec_id, a
    candidate_id, or the exact handle; anything else is reported as unmatched rather
    than credited to whichever row it happens to resemble. Crediting the wrong row is
    worse than failing to credit: it turns a hole into a false 'applied'.
    """
    keys = set()
    if exec_id:
        keys.add("exec:%s" % _norm(exec_id))
    for cid in candidate_ids or ():
        if cid:
            keys.add("cand:%s" % _norm(cid))
    if handle:
        keys.add("handle:%s" % _norm(handle))
    if action:
        keys.add("handle:%s" % _norm(action))
    return keys


def _rec_keys(rec):
    cids = rec.get("candidate_ids")
    if cids is None:
        cids = [rec.get("candidate_id")] if rec.get("candidate_id") else []
    return _keys(rec.get("exec_id"),
                 cids,
                 rec.get("handle") or rec.get("fusion") or rec.get("name"),
                 rec.get("action"))


def _collect_records(apply_payload, errors):
    """Normalize the integrator's return into [(disposition, reason, record)].

    Three accepted shapes, so an existing role return still works:
      * `dispositions: [{exec_id, disposition, reason}]`  -- the explicit form
      * `accepted_fusions: [...]`                          -- → applied
      * `rejected: [...]` / `deferred: [...]`              -- → blocked / deferred
    """
    records = []

    def _reason(rec):
        for field in ("reason", "why", "notes", "note"):
            val = str(rec.get(field) or "").strip()
            if val:
                return val
        return ""

    for rec in apply_payload.get("dispositions") or []:
        disp = str(rec.get("disposition") or "").strip().lower()
        if disp in ("deferred", "defer"):
            disp = "deferred_with_reason"
        if disp not in EXPLICIT:
            errors.append(
                "dispositions[] entry %r has disposition %r; must be one of %s"
                % (rec.get("exec_id") or rec.get("fusion"), rec.get("disposition"),
                   sorted(EXPLICIT)))
            continue
        records.append((disp, _reason(rec), rec))
    for rec in apply_payload.get("accepted_fusions") or []:
        records.append(("applied", _reason(rec) or "integrated and kept", rec))
    for rec in apply_payload.get("rejected") or []:
        records.append(("blocked", _reason(rec), rec))
    for rec in apply_payload.get("deferred") or []:
        records.append(("deferred_with_reason", _reason(rec), rec))

    # A blocked/deferred row without a reason is the failure mode this gate exists to
    # stop, just one level down: "we looked at it" with nothing behind it.
    for disp, reason, rec in records:
        if disp != "applied" and not reason:
            errors.append(
                "apply record %r is %s with no reason; a disposition without a reason "
                "is not a disposition" % (rec.get("exec_id") or rec.get("fusion")
                                          or rec.get("candidate_id"), disp))
    return records


def _unitside_index(unitside_payload):
    """candidate_id -> (unit_side_status, reason)."""
    index = {}
    for row in (unitside_payload or {}).get("results") or []:
        cid = row.get("candidate_id")
        if cid:
            index[cid] = (str(row.get("unit_side_status") or ""),
                          str(row.get("reason") or ""))
    return index


def _conflict_map(topk):
    """exec_id -> set(exec_ids it conflicts with), read off the exclusive groups.

    Uses the pairwise `conflict_edges` when present (the Top-K harness emits them for
    a `compatible_subset` group). A `choose: 1` group is a clique, so every pair in it
    conflicts. Falling back to "everyone in the group conflicts" for a non-clique group
    would auto-block entries that could legally have landed together.
    """
    conflicts = {}

    def _link(a, b):
        conflicts.setdefault(a, set()).add(b)
        conflicts.setdefault(b, set()).add(a)

    for group in topk.get("exclusive_groups") or []:
        ids = [m.get("exec_id") for m in group.get("members") or [] if m.get("exec_id")]
        edges = group.get("conflict_edges")
        if edges:
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    _link(edge[0], edge[1])
        elif group.get("choose") == 1 or group.get("choose") == "1":
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    _link(a, b)
        else:
            for member in group.get("members") or []:
                for other in member.get("conflicts_with") or []:
                    if member.get("exec_id"):
                        _link(member["exec_id"], other)
    return conflicts


def validate(topk_path, apply_path, unitside_path=None, budget=None,
             require_coverage=True, waivers=None):
    topk = _load(topk_path)
    apply_payload = _load(apply_path)
    unitside = _load(unitside_path) if unitside_path else None
    waivers = dict(waivers or {})

    errors = []
    warnings = []

    execution_list = topk.get("execution_list") or []
    if not execution_list:
        errors.append(
            "Top-K json carries no execution_list — this gate has no denominator to "
            "check against. Regenerate FUSION_TOPK with fusion_topk_harness.py "
            "(schema_version >= 3).")

    records = _collect_records(apply_payload, errors)
    unit_index = _unitside_index(unitside)
    conflicts = _conflict_map(topk)

    # ---- pass 1: attach explicit records to rows -----------------------------
    rows = []
    matched_records = set()
    for entry in execution_list:
        exec_id = entry.get("exec_id")
        keys = _keys(exec_id, entry.get("candidate_ids"),
                     entry.get("handle"), entry.get("action"))
        hits = [(i, d, r, rec) for i, (d, r, rec) in enumerate(records)
                if _rec_keys(rec) & keys]
        for i, _d, _r, _rec in hits:
            matched_records.add(i)
        rows.append({"entry": entry, "keys": keys, "hits": hits})

    unmatched = [i for i in range(len(records)) if i not in matched_records]
    for i in unmatched:
        disp, _reason, rec = records[i]
        errors.append(
            "apply record %r (%s) matches no execution_list entry — it names no "
            "exec_id / candidate_id / handle on the board, so it cannot be credited "
            "to any row"
            % (rec.get("exec_id") or rec.get("fusion") or rec.get("candidate_id"),
               disp))

    applied_ids = set()
    for row in rows:
        if any(d == "applied" for _i, d, _r, _rec in row["hits"]):
            applied_ids.add(row["entry"].get("exec_id"))

    # ---- pass 2: derive the disposition of every row -------------------------
    results = []
    for row in rows:
        entry = row["entry"]
        exec_id = entry.get("exec_id")
        rank = entry.get("rank")
        hits = row["hits"]
        unit_states = [unit_index.get(cid) for cid in entry.get("candidate_ids") or []]
        unit_states = [u for u in unit_states if u]
        unit_statuses = sorted({u[0] for u in unit_states})
        within_budget = (
            budget is None or rank is None or int(rank) <= int(budget)
        )
        applyback_required = bool(
            str(entry.get("tier") or "").upper() in ("A", "B") and
            within_budget and
            any(status in APPLYBACK_ELIGIBLE for status in unit_statuses)
        )
        disposition, reason, source = None, "", ""

        if hits:
            # An explicit record wins. If the loop said several things about one row,
            # 'applied' is the strongest claim and the rest are its history.
            order = {"applied": 0, "blocked": 1, "deferred_with_reason": 2}
            _i, disposition, reason, rec = sorted(
                hits, key=lambda h: order.get(h[1], 9))[0]
            source = "integrator"
            if len({d for _i, d, _r, _rec in hits}) > 1:
                warnings.append(
                    "%s carries several apply records (%s); taking '%s'"
                    % (exec_id, sorted({d for _i, d, _r, _rec in hits}), disposition))
        elif exec_id in waivers:
            disposition, reason, source = ("deferred_with_reason",
                                           waivers[exec_id], "waiver")
        else:
            # No claim was made. Before calling it a hole, check the two places a
            # disposition legitimately comes from without the integrator saying so.
            if unit_states and all(u[0] in ("fail", "blocked") for u in unit_states):
                disposition = "blocked"
                reason = "单侧 gate: " + "; ".join(
                    "%s (%s)" % (u[0], u[1]) for u in unit_states[:2])
                source = "unitside"
            elif unit_states and all(
                    u[0] in ("budget_skipped", "not_validated", "fail", "blocked")
                    for u in unit_states):
                # NEVER MEASURED is not a verdict. It must not inherit `blocked`
                # (which reads as "we looked and it lost") and it must not be
                # laundered into the tidy `deferred_budget` row either -- the
                # unit-side budget ran out, so nobody knows what this fusion does.
                disposition = "unaccounted"
                reason = ("单侧 never measured it (%s) — no evidence either way; "
                          "raise fusion_unitside_budget or hang it off a passing "
                          "ladder top" % "; ".join(
                              "%s: %s" % (u[0], u[1]) for u in unit_states[:2]))
                source = "unitside_gap"
            elif conflicts.get(exec_id, set()) & applied_ids:
                disposition = "blocked_by_exclusion"
                reason = ("conflicts with applied %s"
                          % sorted(conflicts[exec_id] & applied_ids))
                source = "exclusion"
            elif (budget is not None and rank is not None
                  and int(rank) > int(budget)):
                disposition = "deferred_budget"
                reason = ("rank %s beyond --budget %s — not attempted this round"
                          % (rank, budget))
                source = "budget"
            else:
                disposition = "unaccounted"
                unit_note = ""
                if entry.get("candidate_ids") and not unit_states:
                    unit_note = (" (its candidates have no 单侧 verdict either — the "
                                 "gap starts at Phase 3.0)")
                reason = ("no disposition anywhere: the integrator did not report it, "
                          "no exclusion covers it, and it is inside the budget"
                          + unit_note)
                source = "none"

        if disposition == "deferred_with_reason" and applyback_required:
            errors.append(
                "%s is an in-budget tier-%s row with passing unit-side evidence (%s), "
                "so deferred_with_reason is not a valid terminal state. Run live "
                "apply-back and report applied, or blocked with the attempted "
                "engagement/A/B/accuracy evidence."
                % (exec_id, entry.get("tier"), ", ".join(unit_statuses)))

        results.append({
            "exec_id": exec_id,
            "rank": rank,
            "phase": entry.get("phase"),
            "tier": entry.get("tier"),
            "action": entry.get("action"),
            "handle": entry.get("handle"),
            "candidate_ids": entry.get("candidate_ids") or [],
            "unit_side_statuses": unit_statuses,
            "applyback_required": applyback_required,
            "forward_us": entry.get("forward_us"),
            "forward_pct": entry.get("forward_pct"),
            "workload_weight": entry.get("workload_weight"),
            "workload_savings_us": entry.get("workload_savings_us"),
            "workload_forward_pct": entry.get("workload_forward_pct"),
            "disposition": disposition,
            "reason": reason,
            "source": source,
        })

    unaccounted = [r["exec_id"] for r in results if r["disposition"] == "unaccounted"]
    waived_unknown = sorted(
        set(waivers) - {e.get("exec_id") for e in execution_list})
    for exec_id in waived_unknown:
        errors.append("--waive references unknown exec_id '%s'" % exec_id)

    counts = {}
    for r in results:
        counts[r["disposition"]] = counts.get(r["disposition"], 0) + 1

    required_applyback = [r for r in results if r["applyback_required"]]
    incomplete_applyback = [
        r for r in required_applyback
        if r["disposition"] not in ("applied", "blocked", "blocked_by_exclusion")
    ]

    truncated = [r["exec_id"] for r in results if r["disposition"] == "deferred_budget"]
    budget_record = {
        "declared": budget,
        "execution_list_size": len(execution_list),
        "beyond_budget": len(truncated),
        "beyond_budget_exec_ids": truncated,
    }
    if budget is not None and len(execution_list) > int(budget):
        # Count the rows the budget ACTUALLY excused, not len-budget: a row past
        # the budget that the integrator deferred on its own merits was attempted
        # as a decision, and folding it into the budget number overstates the cut.
        warnings.append(
            "--budget %s truncates a %d-row board: %d row(s) rank beyond it, "
            "%d of which were left entirely unattempted"
            % (budget, len(execution_list), len(execution_list) - int(budget),
               len(truncated)))

    coverage_fail = bool(require_coverage and unaccounted)
    accounted = sum(1 for r in results if r["disposition"] in ACCOUNTED)
    result = {
        "schema_version": 1,
        "phase": "applyback_gate",
        "status": "fail" if (errors or coverage_fail) else "pass",
        "topk_json": os.path.abspath(topk_path),
        "apply_json": os.path.abspath(apply_path),
        "unitside_json": os.path.abspath(unitside_path) if unitside_path else "",
        "errors": errors,
        "warnings": warnings,
        "coverage": {
            "required": require_coverage,
            "execution_list_size": len(execution_list),
            "accounted": accounted,
            "unaccounted": len(unaccounted),
            "unaccounted_exec_ids": unaccounted,
            "complete": not unaccounted,
        },
        "coverage_ok": not coverage_fail,
        "applyback_completion": {
            "required": len(required_applyback),
            "completed": len(required_applyback) - len(incomplete_applyback),
            "incomplete": len(incomplete_applyback),
            "incomplete_exec_ids": [r["exec_id"] for r in incomplete_applyback],
            "complete": not incomplete_applyback,
        },
        "budget": budget_record,
        "counts": counts,
        "results": results,
        # The board's own coverage travels one more hop, so the FINAL report carries
        # the whole chain: which phases were analysed → which regions were covered →
        # which candidates made the board → which of those actually landed.
        "phase_coverage": topk.get("phase_coverage"),
        "region_coverage": topk.get("region_coverage"),
        "family_coverage": topk.get("family_coverage"),
        "candidate_total": topk.get("candidate_total"),
        "candidates_on_board": topk.get("candidates_on_board"),
        "truncated_count": topk.get("truncated_count"),
        "e2e": {
            "throughput_tok_s": apply_payload.get("e2e_throughput_tok_s"),
            "final_overlay": apply_payload.get("final_overlay"),
            "notes": apply_payload.get("notes"),
        },
        "accepted_fusions": apply_payload.get("accepted_fusions") or [],
    }
    return result


def _esc(value):
    return str(value if value is not None else "").replace("|", "\\|")


_DISP_LABEL = {
    "applied": "✅ 已落地",
    "blocked": "⛔ 被挡",
    "blocked_by_exclusion": "⛔ 互斥被挡",
    "deferred_with_reason": "⏸ 延后",
    "deferred_budget": "⏸ 超预算",
    "unaccounted": "🔴 无交代",
}


def render_markdown(result):
    lines = ["# KernelFusion Apply-back 结果", ""]
    lines.append(
        "融合的**最终结果报告**：Phase 2.2 执行清单上的每一条，在这里都必须有一个明确去向。"
        "「没提到」不是「跳过」，是报告的漏洞——它会被标成 🔴 无交代并让本 gate fail。")
    lines.append("")

    cov = result.get("coverage") or {}
    lines.append("## 交代覆盖率（先看这里）")
    lines.append("")
    lines.append("执行清单 **%d** 条：已交代 **%d** / 无交代 **%d**。"
                 % (cov.get("execution_list_size", 0), cov.get("accounted", 0),
                    cov.get("unaccounted", 0)))
    completion = result.get("applyback_completion") or {}
    if completion.get("required"):
        lines.append("")
        lines.append("预算内且单侧通过、必须完成 Applyback **%d** 条：已完成 **%d** / "
                     "**未完成 %d**。"
                     % (completion.get("required", 0),
                        completion.get("completed", 0),
                        completion.get("incomplete", 0)))
    c = result.get("counts") or {}
    lines.append("")
    lines.append("去向分布：" + " / ".join(
        "%s %d" % (_DISP_LABEL.get(k, k), v) for k, v in sorted(c.items())) + "。")
    bud = result.get("budget") or {}
    if bud.get("declared") is not None:
        lines.append("")
        lines.append("预算 `--budget %s`：清单 %d 条，超预算未尝试 **%d** 条%s。"
                     % (bud.get("declared"), bud.get("execution_list_size", 0),
                        bud.get("beyond_budget", 0),
                        ("（%s）" % "、".join("`%s`" % e for e in
                                             bud.get("beyond_budget_exec_ids") or []))
                        if bud.get("beyond_budget") else ""))
    pc = result.get("phase_coverage") or {}
    rc = result.get("region_coverage") or {}
    if pc or rc:
        lines.append("")
        chain = []
        if pc.get("shape_resolution_by_phase"):
            chain.append("语义阶段：" + "、".join(
                "%s %d/%d" % (ph, v.get("resolved", 0), v.get("rows", 0))
                for ph, v in sorted(pc["shape_resolution_by_phase"].items())))
        if rc:
            chain.append("可融合区间 %d 个：已覆盖 %d / 已延后 %d / 未覆盖 %d"
                         % (rc.get("regions_total", rc.get("regions", 0)),
                            rc.get("covered", 0), rc.get("deferred", 0),
                            rc.get("uncovered", 0)))
        if result.get("candidate_total"):
            chain.append("候选 %d 条 → 进板 %s 条 → 清单 %d 条"
                         % (result.get("candidate_total"),
                            result.get("candidates_on_board"),
                            cov.get("execution_list_size", 0)))
        lines.append("链路：" + "；".join(chain) + "。")
    lines.append("")
    if cov.get("unaccounted"):
        lines.append(
            "> 🔴 **%d 条无交代**：%s。这些融合既没落地、也没被挡、也没延后——"
            "报告里根本没提。补一条明确处置（已落地/被挡+原因/延后+原因）或 "
            "`--waive <exec_id>=<理由>`。"
            % (cov.get("unaccounted"),
               "、".join("`%s`" % e for e in cov.get("unaccounted_exec_ids") or [])))
        lines.append("")
    if completion.get("incomplete"):
        lines.append(
            "> 🔴 **预算内单侧通过项不得延期**：%s。它们必须完成 live Applyback，"
            "最终落到 applied，或在实际接入/engagement/A/B/accuracy 尝试后 blocked。"
            % "、".join("`%s`" % e for e in
                         completion.get("incomplete_exec_ids") or []))
        lines.append("")

    e2e = result.get("e2e") or {}
    if e2e.get("throughput_tok_s"):
        lines.append("## 端到端结果")
        lines.append("")
        lines.append("融合后吞吐 **%s tok/s**；stacked overlay：`%s`。"
                     % (e2e.get("throughput_tok_s"), e2e.get("final_overlay") or "-"))
        if e2e.get("notes"):
            lines.append("")
            lines.append(str(e2e["notes"]))
        lines.append("")

    lines.append("## 逐条去向")
    lines.append("")
    lines.append("| # | exec | 阶段 | 难度 | 融合 | 对应候选 | 预期 forward 收益 | 去向 | 依据 | 说明 |")
    lines.append("|---:|---|:--:|:--:|---|---|---:|:--:|:--:|---|")
    for r in result["results"]:
        if (r.get("workload_savings_us") is not None and
                r.get("workload_forward_pct") is not None):
            gain = "%.0f µs（%.2f%% workload；单次 %.0f µs / phase %.2f%%）" % (
                r["workload_savings_us"], r["workload_forward_pct"],
                r.get("forward_us") or 0.0, r.get("forward_pct") or 0.0)
        else:
            gain = ("%d µs（%.2f%%）" % (r["forward_us"], r["forward_pct"])
                    if r.get("forward_us") is not None
                    and r.get("forward_pct") is not None else "-")
        lines.append("| %s | `%s` | %s | %s | %s | %s | %s | **%s** | %s | %s |" % (
            _esc(r.get("rank")), _esc(r.get("exec_id")), _esc(r.get("phase")),
            _esc(r.get("tier")), _esc(r.get("action")),
            _esc("、".join("`%s`" % c for c in r.get("candidate_ids") or [])),
            gain, _DISP_LABEL.get(r["disposition"], r["disposition"]),
            _esc(r.get("source")), _esc(r.get("reason"))))
    lines.append("")

    accepted = result.get("accepted_fusions") or []
    if accepted:
        lines.append("## 已落地融合明细")
        lines.append("")
        lines.append("| 融合 | rung | overlay | TPOT Δ% | 吞吐 Δ% | gsm8k base→cand | engaged |")
        lines.append("|---|---|---|---:|---:|---|:--:|")
        for f in accepted:
            lines.append("| %s | %s | `%s` | %s | %s | %s → %s | %s |" % (
                _esc(f.get("fusion") or f.get("exec_id") or f.get("candidate_id")),
                _esc(f.get("rung")), _esc(f.get("overlay_path")),
                _esc(f.get("tpot_delta_pct")), _esc(f.get("throughput_delta_pct")),
                _esc(f.get("gsm8k_base")), _esc(f.get("gsm8k_cand")),
                _esc(f.get("engaged"))))
        lines.append("")

    if result.get("warnings"):
        lines.append("## 提示")
        lines.append("")
        for w in result["warnings"]:
            lines.append("- %s" % w)
        lines.append("")
    if result.get("errors"):
        lines.append("## 错误（必须修，不得当作通过）")
        lines.append("")
        for e in result["errors"]:
            lines.append("- %s" % e)
        lines.append("")
    lines.append(
        "说明：本 gate 不跑 server、不产生自己的性能数字；它只校验 apply-back 对 "
        "Phase 2.2 执行清单的**交代完整性**。`applied` 的收益数字来自 fusion_integrator "
        "的实测 A/B，`blocked_by_exclusion` 由互斥冲突图自动推导（只有与**已落地**条目"
        "真正冲突的才会被自动挡），`deferred_budget` 由 `--budget` 按板上排名推导——"
        "预算只能解释排名在预算之外的条目，预算之内漏掉的仍然是 🔴 无交代。")
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_waivers(pairs):
    out = {}
    for item in pairs or []:
        key, sep, reason = str(item).partition("=")
        key, reason = key.strip(), reason.strip()
        if not sep or not key or not reason:
            raise SystemExit(
                "--waive must be <exec_id>=<reason> with a non-empty reason; got %r"
                % item)
        out[key] = reason
    return out


def run(topk_path, apply_path, out_md, out_json, unitside_path=None, budget=None,
        require_coverage=True, waivers=None):
    result = validate(topk_path, apply_path, unitside_path, budget,
                      require_coverage=require_coverage, waivers=waivers)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write(render_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", required=True,
                        help="fusion_topk.json (schema_version >= 3, with execution_list)")
    parser.add_argument("--apply", required=True,
                        help="the fusion_integrator return json")
    parser.add_argument("--unitside", default=None,
                        help="fusion_unitside.json — lets a 单侧 fail/blocked stand as "
                             "a disposition without the integrator repeating it")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--budget", type=int, default=None,
                        help="FUSION_BUDGET actually applied; excuses rows ranked "
                             "beyond it (recorded and named, never silent)")
    parser.add_argument("--allow-partial-coverage", action="store_true")
    parser.add_argument("--waive", action="append", metavar="EXEC_ID=REASON", default=[])
    args = parser.parse_args()
    result = run(args.topk, args.apply, args.out_md, args.out_json,
                 unitside_path=args.unitside, budget=args.budget,
                 require_coverage=not args.allow_partial_coverage,
                 waivers=_parse_waivers(args.waive))
    cov = result.get("coverage") or {}
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "errors": len(result["errors"]),
                      "coverage": {k: cov.get(k) for k in
                                   ("execution_list_size", "accounted",
                                    "unaccounted", "complete")}},
                     indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
