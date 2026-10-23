"""Offline checks on a finished multi-node leg: no cluster, no ssh.

    mn.py trace-gate   validate the per-rank trace set before parsing
    mn.py topn-merge   fold a pd_disagg run's per-engine Top-Ns into one

mn.py is the entry point for both; this module reads files and writes reports only.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys

# mn imports this module lazily (its _offline), so importing mn back is no cycle.
from mn import _utc, _write_json


# --------------------------------------------------------------------------
# TRACE GATE  (mn.py trace-gate)
# --------------------------------------------------------------------------
# Every rank's trace lands in one directory, where an `ls -t | head -1` pick is an
# NFS flush race. The gate VERIFIES every expected TP rank, both nodes, complete
# files and collectives, NAMES the trace the profiler parses (parsed_trace), and
# REPORTS per-rank work volume and node imbalance.
# pd_disagg: each engine has its own TP-0..N in its own subdirectory (from
# --launch-json); each is gated separately and ranks/mn_trace_roles.json names
# the trace per role. Traces are streamed, never json.load-ed.

COLLECTIVE_PATTERNS = {
    "nccl_dev_kernel": rb"ncclDevKernel",
    "nccl_kernel": rb"nccl[A-Za-z]*Kernel",
    "rccl": rb"rccl",
    "all_reduce": rb"[Aa]ll[_]?[Rr]educe",
    "all_gather": rb"[Aa]ll[_]?[Gg]ather",
    "reduce_scatter": rb"[Rr]educe[_]?[Ss]catter",
}
KERNEL_EVENT = rb'"cat": ?"kernel"'

TRACE_RE = re.compile(r"TP-(\d+)")
DP_RE = re.compile(r"DP-(\d+)")


def discover(profile_dir: str) -> list[str]:
    files = [f for pattern in ("*.trace.json.gz", "*.trace.json", "*.json.gz")
             for f in glob.glob(os.path.join(profile_dir, pattern))]
    return sorted(set(f for f in files if not os.path.basename(f).startswith("merged-")))


def scan(path: str) -> dict:
    """Stream one trace to EOF: a truncated gzip raises here."""
    info = {
        "file": os.path.basename(path), "path": path, "bytes": os.path.getsize(path),
        "kernel_events": 0, "collectives": {}, "readable": False, "error": "",
    }
    opener = gzip.open if path.endswith(".gz") else open
    tail = b""
    try:
        with opener(path, "rb") as handle:
            while True:
                chunk = handle.read(8 << 20)
                if not chunk:
                    break
                window = tail + chunk
                info["kernel_events"] += len(re.findall(KERNEL_EVENT, window))
                for name, pattern in COLLECTIVE_PATTERNS.items():
                    hits = len(re.findall(pattern, window))
                    if hits:
                        info["collectives"][name] = info["collectives"].get(name, 0) + hits
                tail = window[-64:]
        info["readable"] = True
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def pd_lanes(launch_json: str) -> list:
    """(role, node_rank, engine TP or 0, node ranks it spans) per profiled engine (replica 0 of
    each role: the only one the window opens), from the launch record."""
    try:
        with open(launch_json, encoding="utf-8") as handle:
            pd = (json.load(handle) or {}).get("pd") or {}
    except (OSError, ValueError, AttributeError):
        return []
    tps = pd.get("engine_tp_by_role") or {}
    out = []
    for i, e in enumerate(pd.get("engines") or []):
        if e.get("replica"):
            continue
        role = e.get("role") or f"engine{i}"
        rank = int(e.get("rank", i))
        out.append((role, rank, int(tps.get(role) or 0),
                    [int(r) for r in e.get("node_ranks") or [rank]]))
    return out


def lane_node(span: list, tp_rank: int, engine_tp: int) -> int:
    """The node a spanning engine's TP rank runs on (ranks fill node-rank 0 first)."""
    per = max(engine_tp // max(len(span), 1), 1)
    return span[min(tp_rank // per, len(span) - 1)]


def cmd_trace_gate(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", default=os.environ.get("PROFILE_DIR", ""))
    ap.add_argument("--expect-ranks", type=int, default=int(os.environ.get("TP", 0) or 0))
    ap.add_argument("--nodes", type=int, default=1, help="nodes the TP group spans")
    ap.add_argument("--select-rank", type=int, default=0, help="the TP rank the profiler parses")
    ap.add_argument("--report-path", default="",
                    help="where to write the report (default: <profile-dir>/ranks/)")
    ap.add_argument("--launch-json", default="",
                    help="mn_launch.json of a pd_disagg launch: gate each engine separately")
    args = ap.parse_args(argv)

    if not args.profile_dir or not os.path.isdir(args.profile_dir):
        sys.stderr.write(f"!!! mn trace-gate: no profile dir at '{args.profile_dir}'\n")
        return 2
    per_node = max(args.expect_ranks // max(args.nodes, 1), 1)

    # A LANE is one directory whose TP ranks form one world. An engine whose
    # directory is missing is kept: that is the failure worth reporting.
    pd_engines = pd_lanes(args.launch_json) if args.launch_json else []
    spans = {}
    if pd_engines:
        fallback = max(args.expect_ranks // max(len(pd_engines), 1), 1)
        lanes = [(role, os.path.join(args.profile_dir, role), node, etp or fallback)
                 for role, node, etp, _span in pd_engines]
        spans = {role: (span, etp or fallback) for role, _n, etp, span in pd_engines}
    else:
        lanes = [("", args.profile_dir, None, args.expect_ranks)]

    if not any(discover(directory) for _r, directory, _n, _e in lanes):
        sys.stderr.write(
            f"!!! mn trace-gate: no traces in {args.profile_dir}.\n    The capture produced nothing; a Top-N built on this would be\n    invented rather than measured.\n")
        return 2

    per_rank, unranked = {}, []
    for role, directory, node, _exp in lanes:
        for path in discover(directory):
            name = os.path.basename(path)
            match, dp = TRACE_RE.search(name), DP_RE.search(name)
            scanned = scan(path)
            scanned["dp_rank"] = int(dp.group(1)) if dp else None
            if role:
                scanned["role"] = role
            if not match:
                unranked.append(scanned)
                continue
            rank = int(match.group(1))
            scanned["tp_rank"] = rank
            scanned["node_index"] = (lane_node(spans[role][0], rank, spans[role][1])
                                     if role in spans and len(spans[role][0]) > 1
                                     else node if node is not None
                                     else rank // per_node)
            per_rank[(role, rank)] = scanned

    ranks = sorted({k[1] for k in per_rank})
    expected = args.expect_ranks or (len(per_rank) or 1)

    by_lane, missing = {}, []
    for role, directory, node, exp in lanes:
        found = sorted(k[1] for k in per_rank if k[0] == role)
        gap = [r for r in range(exp) if r not in found] if exp else []
        missing.extend(f"{role}:{r}" if role else r for r in gap)
        by_lane[role or "-"] = {
            "role": role, "node_index": node, "dir": directory,
            "expected_ranks": exp, "found_ranks": found, "missing_ranks": gap,
            "kernel_events": sum(per_rank[(role, r)]["kernel_events"] for r in found),
            "ranks_with_collectives": [r for r in found if per_rank[(role, r)]["collectives"]],
        }

    nodes = sorted({v["node_index"] for v in per_rank.values()})
    expected_nodes = (sorted({n for span, _t in spans.values() for n in span}) if pd_engines
                      else list(range(max(args.nodes, 1))))
    unreadable = [v["file"] for v in per_rank.values() if not v["readable"]]
    no_kernels = [v["file"] for v in per_rank.values()
                  if v["readable"] and v["kernel_events"] == 0]
    with_collectives = sorted({k[1] for k, v in per_rank.items() if v["collectives"]})

    kernel_by_node = {}
    for value in per_rank.values():
        kernel_by_node[value["node_index"]] = (kernel_by_node.get(value["node_index"], 0)
                                               + value["kernel_events"])
    imbalance_pct = None
    if len(kernel_by_node) > 1 and max(kernel_by_node.values()) > 0:
        low, high = min(kernel_by_node.values()), max(kernel_by_node.values())
        imbalance_pct = round((high - low) / high * 100.0, 2)

    total_expected = sum(lane[3] for lane in lanes)
    problems = []
    silent = [name for name, lane in by_lane.items() if not lane["found_ranks"]]
    if pd_engines and silent:
        problems.append(
            f"the {', '.join(silent)} engine(s) produced NO traces: the phase they serve has no profile, and a Top-N from the other engine alone looks complete")
    if missing:
        problems.append(
            f"missing TP rank(s) {missing}: the capture covers {len(per_rank)}/{total_expected} ranks, so the set is not the server")
    if sorted(nodes) != expected_nodes:
        problems.append(
            f"only node(s) {nodes} produced traces (expected {expected_nodes}) -- a set missing a whole node looks like a healthy single-node capture")
    if unreadable:
        problems.append(
            f"unreadable/truncated trace(s) {unreadable}: most likely the server was torn down before the export flushed")
    if no_kernels:
        problems.append(f"trace(s) with zero kernel events {no_kernels}")
    if not with_collectives:
        problems.append(
            "no RCCL/NCCL kernels in ANY rank: the cross-node traffic was not captured, so the Top-N would describe local compute only")
    selected = {}
    for role, _directory, _node, _exp in lanes:
        if (role, args.select_rank) in per_rank:
            selected[role or "-"] = per_rank[(role, args.select_rank)]["path"]
        elif not (role and not by_lane[role]["found_ranks"]):
            problems.append(f"selected rank {args.select_rank} has no trace"
                            + (f" for the {role} engine" if role else ""))

    report = {
        "profile_dir": args.profile_dir,
        "checked_at": _utc(),
        "expected_ranks": expected,
        "found_ranks": ranks,
        "nodes_represented": nodes,
        "ranks_per_node": per_node,
        "selected_rank": args.select_rank,
        "ranks_with_collectives": with_collectives,
        "kernel_events_by_node": kernel_by_node,
        "node_kernel_imbalance_pct": imbalance_pct,
        "per_rank": [per_rank[k] for k in sorted(per_rank)],
        "unranked_files": unranked,
        "parsed_trace": selected.get("-", ""),
        "problems": problems,
        "verdict": "pass" if not problems else "fail",
    }
    if pd_engines:
        report["topology"] = "pd_disagg"
        report["per_role"] = [by_lane[r or "-"] for r, _d, _n, _e in lanes]
        report["kernel_events_by_role"] = {
            name: lane["kernel_events"] for name, lane in by_lane.items()}
        report["parsed_trace_by_role"] = selected

    # In ranks/, never beside the traces: the parser's glob matches .json and
    # would pick the report as the trace.
    report_path = args.report_path or os.path.join(args.profile_dir, "ranks", "mn_trace_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    _write_json(report_path, report)
    report["report_path"] = report_path

    # A separate input file for the profiler, written even on fail: one profiled
    # engine is still worth tuning.
    roles_path = ""
    if pd_engines:
        roles_path = os.path.join(args.profile_dir, "ranks", "mn_trace_roles.json")
        _write_json(roles_path, {
            "verdict": report["verdict"],
            "report": report_path,
            "roles": [
                {"role": role, "node_index": node, "dir": directory,
                 "trace": selected.get(role, ""),
                 "ranks": len(by_lane[role]["found_ranks"]),
                 "kernel_events": by_lane[role]["kernel_events"]}
                for role, directory, node, _exp in lanes
            ],
        })
        report["roles_path"] = roles_path

    print(f">>> mn trace-gate: {len(per_rank)}/{total_expected or expected} rank trace(s), node(s) {nodes}, collectives in {len(with_collectives)} rank(s)")
    for key in sorted(per_rank):
        value = per_rank[key]
        label = f"{key[0]}/TP-{key[1]}" if key[0] else f"TP-{key[1]}"
        print(f"    {label:<14} node{value['node_index']} {value['bytes']/2**20:8.1f} MiB  kernels={value['kernel_events']:<8} collectives={sum(value['collectives'].values())}")
    if imbalance_pct is not None:
        print(f">>> kernel-event imbalance between nodes: {imbalance_pct}% ({kernel_by_node})")
    if roles_path:
        print(f">>> mn trace-gate: per-engine traces indexed at {roles_path}")
    print(f">>> mn trace-gate: report at {report_path}")
    for problem in report["problems"]:
        sys.stderr.write(f"!!! mn trace-gate: {problem}\n")
    return 0 if report["verdict"] == "pass" else 1


# --------------------------------------------------------------------------
# TOPN MERGE  (mn.py topn-merge)
# --------------------------------------------------------------------------
# Fold the per-engine Top-Ns into the ONE profile_topN.json downstream routes on,
# as a WEIGHTED UNION: a kernel at 30% of prefill, on an engine holding 40% of
# the deployment's GPU time, is 12%. Rows carry their `engine`; a kernel on both engines
# stays two rows. Weights are comparable because multinode.sh opens and closes
# both windows together.
#
# The bench of the same leg gives the Amdahl weight on throughput:
#     E2E = TTFT + (OSL - 1) * TPOT
# and each phase's share of E2E names the bottleneck engine. kv_handoff reads the
# engines' scheduler lines to split TTFT into compute vs KV-transfer wait.

def latency_split(bench_summary: str, osl: int) -> dict:
    """{prefill_ms, decode_ms, share} from one bench leg, or {} without TTFT/TPOT."""
    if not bench_summary or osl < 1:
        return {}
    try:
        with open(bench_summary, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return {}
    ttft, tpot = doc.get("ttft_ms_median"), doc.get("tpot_ms_median")
    if not isinstance(ttft, (int, float)) or not isinstance(tpot, (int, float)):
        return {}
    prefill_ms = float(ttft)
    decode_ms = float(tpot) * max(osl - 1, 0)
    total = prefill_ms + decode_ms
    if total <= 0:
        return {}
    return {
        "source": bench_summary, "osl": osl,
        "ttft_ms": round(prefill_ms, 3), "tpot_ms": round(float(tpot), 3),
        "prefill_ms": round(prefill_ms, 3), "decode_ms": round(decode_ms, 3),
        "share": {"prefill": round(prefill_ms / total, 4),
                  "decode": round(decode_ms / total, 4)},
    }


_BATCH_FIELD = re.compile(r"#([a-z-]+): (\d+)")


def _batch_lines(path: str, marker: str) -> list:
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if marker in line:
                    out.append({k: int(v) for k, v in _BATCH_FIELD.findall(line)})
    except OSError:
        return []
    return out


def _mean(rows: list, key: str) -> float:
    return round(sum(r.get(key, 0) for r in rows) / len(rows), 2)


def kv_handoff(prefill_log: str, decode_log: str) -> dict:
    """decode: running-req vs transfer-req (waiting for KV) per step;
    prefill: new-seq / new-token / cached-token / inflight-req per batch."""
    dec = [d for d in _batch_lines(decode_log, "Decode batch") if "running-req" in d]
    pre = [p for p in _batch_lines(prefill_log, "Prefill batch") if "new-seq" in p]
    if not dec and not pre:
        return {}
    out: dict = {}
    if dec:
        running = sum(d.get("running-req", 0) for d in dec)
        transfer = sum(d.get("transfer-req", 0) for d in dec)
        out["decode"] = {
            "source": decode_log, "steps": len(dec),
            "running_req_mean": _mean(dec, "running-req"),
            "transfer_req_mean": _mean(dec, "transfer-req"),
            "transfer_req_max": max(d.get("transfer-req", 0) for d in dec),
            "wait_share": round(transfer / (transfer + running), 4) if transfer + running else None,
        }
    if pre:
        seqs = sum(p.get("new-seq", 0) for p in pre)
        new_tok = sum(p.get("new-token", 0) for p in pre)
        cached = sum(p.get("cached-token", 0) for p in pre)
        out["prefill"] = {
            "source": prefill_log, "batches": len(pre),
            "new_seq_mean": _mean(pre, "new-seq"),
            "single_seq_batch_share": round(
                sum(1 for p in pre if p.get("new-seq", 0) <= 1) / len(pre), 4),
            "computed_tokens_per_seq": round(new_tok / seqs, 1) if seqs else None,
            "prefix_cache_share": round(cached / (cached + new_tok), 4) if cached + new_tok else None,
            "inflight_req_mean": _mean(pre, "inflight-req"),
        }
    verdict = handoff_verdict(out)
    if verdict:
        out["verdict"] = verdict
    return out


HANDOFF_WAIT_SHARE = 0.5


def handoff_verdict(kv: dict) -> dict:
    """{handoff_bound, reason}, or {} without a decode side. Prefill's in-flight
    count breaks the tie: requests finished but not yet delivered."""
    dec, pre = kv.get("decode"), kv.get("prefill")
    if not dec or dec.get("wait_share") is None:
        return {}
    share = dec["wait_share"]
    running = dec.get("running_req_mean") or 0.0
    inflight = (pre or {}).get("inflight_req_mean")
    if share < HANDOFF_WAIT_SHARE:
        return {"handoff_bound": False,
                "reason": f"decode wait_share {share:.2f} < {HANDOFF_WAIT_SHARE}: most decode-side requests are being decoded, not waiting for KV"}
    if inflight is not None and inflight < running:
        return {"handoff_bound": False,
                "reason": f"decode wait_share {share:.2f} but prefill holds only {inflight} requests in flight vs {running} decoding: the queue is upstream of the transfer"}
    held = "" if inflight is None else f"; prefill holds {inflight} finished requests in flight"
    return {"handoff_bound": True,
            "reason": f"decode wait_share {share:.2f} (mean {dec.get('transfer_req_mean')} waiting for KV vs {running} decoding){held}. TTFT is the KV handoff; prefill kernels cannot move it, whatever computed_tokens_per_seq says"}


def _kv_handoff_md(kv: dict) -> list:
    lines = ["\n## KV handoff (from the engines' own scheduler lines, same leg)\n"]
    dec, pre = kv.get("decode"), kv.get("prefill")
    if dec:
        share = dec["wait_share"]
        lines.append(
            f"- decode: {dec['steps']} steps, mean **{dec['transfer_req_mean']}** requests waiting for KV (max {dec['transfer_req_max']}) vs **{dec['running_req_mean']}** decoding → {'-' if share is None else f'{share * 100:.1f}%'} of decode-side in-flight requests are parked in the handoff, holding no GPU work on either engine.")
    if pre:
        cache = pre["prefix_cache_share"]
        lines.append(
            f"- prefill: {pre['batches']} batches, mean {pre['new_seq_mean']} new sequences per batch ({pre['single_seq_batch_share'] * 100:.0f}% admit one), {pre['computed_tokens_per_seq']} prompt tokens computed per sequence{'' if cache is None else f', {cache * 100:.1f}% served from the prefix cache'}; mean {pre['inflight_req_mean']} requests in flight to decode.")
    verdict = kv.get("verdict")
    if verdict:
        lines.append(f"- **handoff_bound = {str(verdict['handoff_bound']).lower()}**: {verdict['reason']}.")
    return lines


def _topn_md(doc: dict) -> str:
    rows = doc["top_kernels"]
    lines = [f"# Profile Top-{len(rows)} — pd_disagg, per-engine weighted union\n",
             f"- source: `{doc['source']}`",
             f"- total GPU time: **{doc['total_gpu_time_ms']:.2f} ms** over {doc['num_kernel_launches']} launches, {doc['num_distinct_kernels']} distinct kernels"]
    split = doc.get("latency_split") or {}
    for eng in doc["per_engine"]:
        e2e = eng.get("e2e_share_pct")
        lines.append(f"  - `{eng['engine']}`: {eng['total_gpu_time_ms']:.2f} ms ({eng['weight_pct']:.1f}% of the deployment's GPU time{f'; {e2e:.1f}% of E2E latency' if e2e is not None else ''}) — own table: `{eng['topN_json']}`")
    if split:
        lines.append(f"- latency split from `{split['source']}`: TTFT {split['ttft_ms']:.1f} ms vs decode {split['decode_ms']:.1f} ms (TPOT {split['tpot_ms']:.2f} × {split['osl'] - 1}). %e2e = %gpu in engine × that engine's E2E share. TTFT includes queueing and the KV transfer, so prefill's share is an upper bound.")
    if doc.get("kv_handoff"):
        lines.extend(_kv_handoff_md(doc["kv_handoff"]))
    lines.append("\n**%gpu is share of the DEPLOYMENT.** To rank within one engine read that engine's table.\n")
    lines.append("| # | engine | kernel | class | backend | edit | calls | total ms | %gpu | %gpu in engine | %e2e | avg us |")
    lines.append("|--|--------|--------|-------|---------|------|-------|----------|------|----------------|------|--------|")
    for k in rows:
        e2e = k.get("pct_e2e_est")
        lines.append(
            f"| {k['rank']} | {k['engine']} | `{k['short_name']}` | {k['classification']} | {k['backend_guess']} | {'Y' if k['editable'] else 'N'} | {k['calls']} | {k['total_ms']:.3f} | {k['pct_gpu_time']:.1f} | {k['pct_gpu_time_in_engine']:.1f} | {'-' if e2e is None else f'{e2e:.1f}'} | {k['avg_us']:.1f} |")
    lines.append("\n## Opt hints (top entries)\n")
    for k in rows[:12]:
        lines.append(f"- **{k['rank']}. [{k['engine']}] {k['short_name']}** ({k['pct_gpu_time']:.1f}% gpu): {k['opt_hint']}")
    return "\n".join(lines) + "\n"


def cmd_topn_merge(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", default=[], metavar="NAME=PATH",
                    help="one engine's profile_topN.json (repeatable)")
    ap.add_argument("--out", default="", help="prefix; writes <out>.json and <out>.md")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--bench-summary", default="",
                    help="bench_summary.json of the SAME leg the traces came from")
    ap.add_argument("--osl", type=int, default=0)
    ap.add_argument("--engine-log", action="append", default=[], metavar="NAME=PATH",
                    help="one engine's server log from the SAME leg (prefill=, decode=)")
    args = ap.parse_args(argv)
    split = latency_split(args.bench_summary, args.osl)
    engine_logs = dict(spec.partition("=")[::2] for spec in args.engine_log)
    handoff = kv_handoff(engine_logs.get("prefill", ""), engine_logs.get("decode", ""))
    if engine_logs and not handoff:
        sys.stderr.write(f"!!! mn topn-merge: no Prefill/Decode batch lines in {sorted(engine_logs.values())}; no kv_handoff block\n")
    if args.bench_summary and not split:
        sys.stderr.write(f"!!! mn topn-merge: no TTFT/TPOT in {args.bench_summary} (osl={args.osl}); merging on GPU time only\n")

    loaded = []
    for spec in args.engine:
        name, _, path = spec.partition("=")
        if not path:
            sys.stderr.write(f"!!! mn topn-merge: --engine wants NAME=PATH, got '{spec}'\n")
            return 2
        try:
            with open(path, encoding="utf-8") as handle:
                loaded.append((name, path, json.load(handle)))
        except (OSError, ValueError) as exc:
            # One engine profiled is still a usable result.
            sys.stderr.write(f"!!! mn topn-merge: skipping {name} ({path}): {exc}\n")
    if not loaded:
        sys.stderr.write("!!! mn topn-merge: no engine Top-N could be read\n")
        return 2

    total_ms = sum(float(doc.get("total_gpu_time_ms") or 0.0) for _n, _p, doc in loaded)
    if total_ms <= 0:
        sys.stderr.write("!!! mn topn-merge: every engine reports zero GPU time\n")
        return 2

    rows, per_engine = [], []
    for name, path, doc in loaded:
        engine_ms = float(doc.get("total_gpu_time_ms") or 0.0)
        weight = engine_ms / total_ms
        e2e_share = (split.get("share") or {}).get(name) if split else None
        per_engine.append({
            "engine": name, "topN_json": path,
            "total_gpu_time_ms": round(engine_ms, 4),
            "weight_pct": round(weight * 100.0, 2),
            "e2e_share_pct": None if e2e_share is None else round(e2e_share * 100.0, 2),
            "num_kernel_launches": doc.get("num_kernel_launches", 0),
            "num_distinct_kernels": doc.get("num_distinct_kernels", 0),
        })
        for entry in doc.get("top_kernels") or []:
            in_engine = float(entry.get("pct_gpu_time") or 0.0)
            rows.append({
                **entry, "engine": name,
                "pct_gpu_time_in_engine": in_engine,
                "pct_gpu_time": round(in_engine * weight, 4),
                "pct_e2e_est": None if e2e_share is None else round(in_engine * e2e_share, 4),
            })

    rows.sort(key=lambda r: r["pct_gpu_time"], reverse=True)
    rows = rows[: args.top]
    for i, row in enumerate(rows, 1):
        row["rank"] = i

    out = {
        "source": "pd-merged:" + "+".join(n for n, _p, _d in loaded),
        "topology": "pd_disagg",
        "total_gpu_time_ms": round(total_ms, 4),
        "num_kernel_launches": sum(d.get("num_kernel_launches", 0) for _n, _p, d in loaded),
        "num_distinct_kernels": sum(d.get("num_distinct_kernels", 0) for _n, _p, d in loaded),
        "per_engine": per_engine,
        "top_kernels": rows,
    }
    if split:
        out["latency_split"] = split
        out["bottleneck_engine"] = max(
            (r for r in per_engine if r["e2e_share_pct"] is not None),
            key=lambda r: r["e2e_share_pct"], default={}).get("engine", "")
    if handoff:
        out["kv_handoff"] = handoff
    markdown = _topn_md(out)
    if args.out:
        with open(args.out + ".json", "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
        with open(args.out + ".md", "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print(f">>> mn topn-merge: wrote {args.out}.json and {args.out}.md")
    for eng in per_engine:
        e2e = eng["e2e_share_pct"]
        print(f"    {eng['engine']:<10} {eng['total_gpu_time_ms']:10.2f} ms  {eng['weight_pct']:5.1f}% of deployment GPU time{f'  {e2e:5.1f}% of E2E latency' if e2e is not None else ''}")
    if not args.out:
        print(markdown)
    return 0
