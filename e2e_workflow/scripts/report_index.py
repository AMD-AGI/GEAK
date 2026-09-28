#!/usr/bin/env python3
"""Publish the fusion pipeline's reports at the EVAL_DIR root and index them.

Every phase produced its markdown somewhere inside its own working directory, so
reading the run meant knowing five paths (and which of the five semantics_* dirs was
the live one). The reports are the thing a person actually opens; the JSON, traces,
overlays and scratch are the thing a harness opens. So the reports live at the root
under fixed, ordered names, and everything else stays where it was written:

    <EVAL_DIR>/00_INDEX.md                <- this file's output
    <EVAL_DIR>/01_SEMANTIC.md             <- Phase 1    semantic_report.py
    <EVAL_DIR>/02_FUSION_CANDIDATES.md    <- Phase 2.1  fusion_candidate_harness.py
    <EVAL_DIR>/03_FUSION_TOPK.md          <- Phase 2.2  fusion_topk_harness.py
    <EVAL_DIR>/04_FUSION_UNITSIDE.md      <- Phase 3.0  fusion_unitside_harness.py
    <EVAL_DIR>/05_FUSION_APPLYBACK.md     <- Apply-back fusion_applyback_harness.py

The index is generated from what is actually on disk, never from what was supposed to
run. A phase whose report is missing is listed as missing rather than omitted -- an
index that silently drops the phases that did not happen is the same failure as a
report that silently drops the candidates that were not tried.

Usage:
  python3 report_index.py --eval-dir <EVAL_DIR> [--title "DSR1 20260826"]
"""
import argparse
import glob
import json
import os
import sys


# (report basename, phase label, sidecar json basenames, summariser)
#
# More than one sidecar name per phase on purpose: the candidate harness's JSON has
# been written as both fusion_candidate_result.json and fusion_candidate_validation.json
# by different callers. The index resolves whichever is on disk instead of declaring a
# phase report-less because the caller picked the other spelling.
REPORTS = [
    ("01_SEMANTIC.md", "Phase 1 · 语义层",
     ["semantic_report.json"], "semantic"),
    ("02_FUSION_CANDIDATES.md", "Phase 2.1 · 融合候选",
     ["fusion_candidate_result.json", "fusion_candidate_validation.json"],
     "candidates"),
    ("03_FUSION_TOPK.md", "Phase 2.2 · Top-K 执行清单",
     ["fusion_topk.json"], "topk"),
    ("04_FUSION_UNITSIDE.md", "Phase 3.0 · 单侧 gate",
     ["fusion_unitside.json"], "unitside"),
    ("05_FUSION_APPLYBACK.md", "Apply-back · 融合最终结果",
     ["fusion_applyback.json"], "applyback"),
]

INDEX_NAME = "00_INDEX.md"


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _find_sidecar(eval_dir, name):
    """Root first, then anywhere under EVAL_DIR, newest wins.

    The JSON is an intermediate and legitimately lives in the phase's own working
    dir; the index should not force it to the root just to be found.
    """
    at_root = os.path.join(eval_dir, name)
    if os.path.isfile(at_root):
        return at_root
    hits = [p for p in glob.glob(os.path.join(eval_dir, "**", name), recursive=True)
            if os.path.isfile(p)]
    if not hits:
        return None
    return max(hits, key=lambda p: os.path.getmtime(p))


def _summarise(kind, data):
    """One line per phase: the numbers that decide whether to open it."""
    if not data:
        return "", ""
    if kind == "semantic":
        meas = data.get("phase_coverage_measured") or {}
        parts = ["%s %d/%d shape" % (ph, v.get("resolved", 0), v.get("rows", 0))
                 for ph, v in sorted(meas.items())]
        parts.append("可融合区间 %d 个" % data.get("fusible_region_count", 0))
        blind = [ph for ph, v in sorted(meas.items())
                 if v.get("rows") and not v.get("resolved")]
        return "；".join(parts), ("🔴 %s 无 shape" % "/".join(blind)) if blind else "—"
    if kind == "candidates":
        rc = data.get("region_coverage") or {}
        pc = data.get("phase_coverage") or {}
        count = (data.get("candidate_count")
                 or (data.get("metrics") or {}).get("candidate_count")
                 or len(data.get("candidates") or []))
        parts = ["候选 %s 条" % (count or "?")]
        if rc:
            parts.append("区间 %d：覆盖 %d / 延后 %d / 未覆盖 %d"
                         % (rc.get("regions_total", 0), rc.get("covered", 0),
                            rc.get("deferred", 0), rc.get("uncovered", 0)))
        # Known-fusion priors: the run-to-run stability number. It belongs on
        # the index line because its failure mode is silence -- a prior that was
        # never proposed shows up nowhere else on this page.
        pr = data.get("prior_coverage") or (
            data.get("metrics") or {}).get("prior_coverage") or {}
        if pr:
            parts.append("已知先验 %d：带入 %d / 已生效 %d / 不适用 %d / 无交代 %d"
                         % (pr.get("priors_total", 0), pr.get("carried", 0),
                            pr.get("already_engaged", 0),
                            pr.get("not_applicable", 0),
                            pr.get("undisposed", 0)))
        status = data.get("status") or ("fail" if data.get("errors") else "pass")
        flag = "🔴 fail" if status != "pass" else "✅ pass"
        if pc and pc.get("problems"):
            flag += "（阶段覆盖有问题）"
        return "；".join(parts), flag
    if kind == "topk":
        parts = ["候选 %s → 进板 %s → 清单 %s 条"
                 % (data.get("candidate_total", "?"),
                    data.get("candidates_on_board", "?"),
                    len(data.get("execution_list") or []))]
        if data.get("truncated_count"):
            parts.append("截断 %d 条" % data["truncated_count"])
        groups = data.get("exclusive_groups") or []
        if groups:
            parts.append("互斥组 %d 个" % len(groups))
        return "；".join(parts), "—"
    if kind == "unitside":
        cov = data.get("coverage") or {}
        c = data.get("counts") or {}
        parts = ["在范围内 %d：已验证 %d / 未验证 %d / 豁免 %d"
                 % (cov.get("in_scope", 0), cov.get("validated", 0),
                    cov.get("not_validated", 0), cov.get("waived", 0)),
                 "pass %d / fail %d / blocked %d"
                 % (c.get("pass", 0), c.get("fail", 0), c.get("blocked", 0))]
        flag = "✅ pass" if data.get("status") == "pass" else "🔴 fail"
        return "；".join(parts), flag
    if kind == "applyback":
        cov = data.get("coverage") or {}
        c = data.get("counts") or {}
        e2e = data.get("e2e") or {}
        parts = ["清单 %d 条：已交代 %d / 无交代 %d"
                 % (cov.get("execution_list_size", 0), cov.get("accounted", 0),
                    cov.get("unaccounted", 0)),
                 "已落地 %d" % c.get("applied", 0)]
        if e2e.get("throughput_tok_s"):
            parts.append("吞吐 %s tok/s" % e2e["throughput_tok_s"])
        flag = "✅ pass" if data.get("status") == "pass" else "🔴 fail"
        return "；".join(parts), flag
    return "", ""


def build(eval_dir):
    rows = []
    for name, label, sidecars, kind in REPORTS:
        path = os.path.join(eval_dir, name)
        present = os.path.isfile(path)
        side = None
        for cand in sidecars:
            side = _find_sidecar(eval_dir, cand)
            if side:
                break
        data = _load(side) if side else None
        summary, flag = _summarise(kind, data)
        rows.append({
            "report": name, "label": label, "present": present,
            "path": path if present else "",
            "sidecar": side or "", "summary": summary, "flag": flag,
            "workdir": os.path.dirname(side) if side else "",
        })
    return {"eval_dir": os.path.abspath(eval_dir), "reports": rows,
            "missing": [r["report"] for r in rows if not r["present"]]}


def render_markdown(index, title=""):
    lines = ["# 融合流程报告索引%s" % (" — %s" % title if title else ""), ""]
    lines.append("根目录：`%s`" % index["eval_dir"])
    lines.append("")
    lines.append(
        "五个阶段各出一份报告，按顺序读即可。**中间产物（JSON / trace / overlay / "
        "microbench 脚手架）留在各自的工作目录**，报告放在根目录——要看结论看这里，"
        "要复核证据再进工作目录。")
    lines.append("")
    lines.append("| 阶段 | 报告 | 状态 | 关键数字 | 中间产物目录 |")
    lines.append("|---|---|:--:|---|---|")
    for r in index["reports"]:
        if r["present"]:
            report = "[`%s`](./%s)" % (r["report"], r["report"])
        else:
            report = "`%s` — **未生成**" % r["report"]
        lines.append("| %s | %s | %s | %s | %s |" % (
            r["label"], report, r["flag"] or "—", r["summary"] or "—",
            ("`%s`" % r["workdir"]) if r["workdir"] else "—"))
    lines.append("")
    if index["missing"]:
        lines.append(
            "> ⚠️ 未生成的报告：%s。索引按**磁盘上实际有什么**生成，不按计划生成——"
            "没跑的阶段列成「未生成」，不会从索引里消失。"
            % "、".join("`%s`" % m for m in index["missing"]))
        lines.append("")
    lines.append("## 读法")
    lines.append("")
    lines.append(
        "1. **01 语义层** — 分析了哪些阶段、shape 解出来没有、可融合面（fusible "
        "regions）有多大。decode 一列若是 0 shape，后面全部阶段的 decode 结论都不成立。")
    lines.append(
        "2. **02 融合候选** — 候选集是否覆盖了 01 里的每一个可融合区间。"
        "「未覆盖」不是「没机会」，是漏了。")
    lines.append(
        "3. **03 Top-K 执行清单** — 排名 + 每条的 `exec_id`。这张清单是 3.0/3.1 的**分母**。")
    lines.append(
        "4. **04 单侧 gate** — 每条候选的隔离 microbench 结论；未验证 = 没测，不是通过。")
    lines.append(
        "5. **05 apply-back** — 清单上每个 `exec_id` 的最终去向 + 端到端结果。"
        "🔴 无交代 = 报告漏洞。")
    lines.append("")
    return "\n".join(lines) + "\n"


def run(eval_dir, title=""):
    index = build(eval_dir)
    out = os.path.join(eval_dir, INDEX_NAME)
    os.makedirs(eval_dir, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(render_markdown(index, title))
    index["index_path"] = out
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    index = run(args.eval_dir, args.title)
    print(json.dumps({"index": index["index_path"],
                      "present": [r["report"] for r in index["reports"]
                                  if r["present"]],
                      "missing": index["missing"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
