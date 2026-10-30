"""Offline checks on a finished multi-node leg: no cluster, no ssh.

    mn.py trace-gate      validate the per-rank trace set before parsing
    mn.py topn-merge      fold a pd_disagg run's per-engine Top-Ns into one
    mn.py pd-mass         rewrite an AgentX pd_disagg ranking on both engines' GPU-time mass
    mn.py pd-points       plan a fixed-length pd_disagg run's sampling points from its engine logs

mn.py is the entry point for both; this module reads files and writes reports only.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import subprocess
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


def _pd_record(launch_json: str) -> dict:
    try:
        with open(launch_json, encoding="utf-8") as handle:
            return (json.load(handle) or {}).get("pd") or {}
    except (OSError, ValueError, AttributeError):
        return {}


def pd_lanes(launch_json: str) -> list:
    """(role, node_rank, engine TP or 0, node ranks it spans) per profiled engine (replica 0 of
    each role: the only one the window opens), from the launch record."""
    pd = _pd_record(launch_json)
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
    ap.add_argument("--role", default="",
                    help="with --launch-json: gate only this engine (a sampling-point probe "
                         "opens the window on one)")
    args = ap.parse_args(argv)

    if not args.profile_dir or not os.path.isdir(args.profile_dir):
        sys.stderr.write(f"!!! mn trace-gate: no profile dir at '{args.profile_dir}'\n")
        return 2
    per_node = max(args.expect_ranks // max(args.nodes, 1), 1)

    # A LANE is one directory whose TP ranks form one world. An engine whose
    # directory is missing is kept: that is the failure worth reporting.
    pd_engines = pd_lanes(args.launch_json) if args.launch_json else []
    if args.role:
        pd_engines = [lane for lane in pd_engines if lane[0] == args.role]
        if not pd_engines:
            sys.stderr.write(f"!!! mn trace-gate: no {args.role} engine in {args.launch_json}\n")
            return 2
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
        engines = _pd_record(args.launch_json).get("engines") or []
        _write_json(roles_path, {
            "verdict": report["verdict"],
            "report": report_path,
            "roles": [
                {"role": role, "node_index": node, "dir": directory,
                 "trace": selected.get(role, ""),
                 "ranks": len(by_lane[role]["found_ranks"]),
                 "instances": sum(1 for e in engines if e.get("role") == role),
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
# as a WEIGHTED UNION: a kernel at 30% of prefill, on an engine that costs 40% of
# the leg's GPU time, is 12% -- what %gpu means on one node. Rows carry their
# `engine`; a kernel on both engines stays two rows.
#
# The two profile windows differ in length and phase (decode closes at num_steps,
# prefill at the timer), so their GPU times do not add. Each engine's weight is its
# GPU time over the leg as single-node AgentX counts it: probed points' mass
# (pd_mass below) or, with no points, a step's mean time in the window x the
# analytic step count (pd_analytic). No graded axis enters: the weights do not
# change with the metric.

PHASE_ENGINE = {"PREFILL": "prefill", "DECODE": "decode"}


def latency_split(bench_summary: str, osl: float) -> dict:
    """{prefill_ms, decode_ms, share} from one bench leg, or {} without TTFT/TPOT. Explanation only.
    OSL is the summary's observed_osl, else `osl`."""
    if not bench_summary:
        return {}
    try:
        with open(bench_summary, encoding="utf-8") as handle:
            doc = json.load(handle)
    except (OSError, ValueError):
        return {}
    served = doc.get("observed_osl")
    osl = float(served) if isinstance(served, (int, float)) and served >= 1 else osl
    if osl < 1:
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


def role_ranks(roles_paths) -> dict:
    """{prefill: ranks, decode: ranks} from trace-gate's mn_trace_roles.json (one, or one per
    probed engine); ValueError if unknowable.
    A trace times TP-0 only, so an engine's GPU time is TP-0's times its ranks."""
    paths = [roles_paths] if isinstance(roles_paths, str) else list(roles_paths or [])
    if not any(paths):
        raise ValueError("no --roles (mn_trace_roles.json): the engines' rank counts are unknown")
    ranks = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                roles = (json.load(handle) or {}).get("roles") or []
        except (OSError, ValueError, AttributeError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        for entry in roles:
            role = entry.get("role")
            if role in ranks or int(entry.get("instances") or 1) > 1:
                raise ValueError(f"the {role} role has several instances (xPyD) or is in more "
                                 "than one --roles; one engine per role is supported")
            ranks[role] = int(entry.get("ranks") or 0)
    missing = [r for r in PHASE_ENGINE.values() if ranks.get(r, 0) < 1]
    if missing:
        raise ValueError(f"{', '.join(paths)} gives no ranks for {missing}")
    return {r: ranks[r] for r in PHASE_ENGINE.values()}


def pd_analytic(docs: dict, ranks: dict) -> dict:
    """Single-node AgentX's no-log fallback, per engine: the window's mean GPU time per step of
    its phase x parse_profile's analytic_calls for that phase (CONC*ceil(ISL/chunk) prefill
    steps, OSL decode steps, per wave) x its ranks. ValueError naming what is missing.
    docs: {prefill, decode} -> that engine's parse_profile Top-N."""
    per_step, calls, missing = {}, {}, []
    for phase, engine in PHASE_ENGINE.items():
        doc = docs[engine]
        ph = phase.lower()
        steps = (doc.get("serving") or {}).get(f"n_{ph}_steps") or 0
        gpu_ms = (doc.get("step_ledger") or {}).get(f"{ph}_gpu_ms") or 0
        n = ((doc.get("serving") or {}).get("analytic_calls") or {}).get(ph) or 0
        if steps < 1 or gpu_ms <= 0:
            missing.append(f"the {engine} engine's {ph} steps and their GPU time, which only step "
                           "annotations give (" + (doc.get("step_ledger_unavailable") or "none in its window") + ")")
        if n < 1:
            missing.append(f"the {engine} engine's serving.analytic_calls.{ph} "
                           "(parse_profile needs --isl/--osl/--conc)")
        if steps >= 1 and gpu_ms > 0 and n >= 1:
            per_step[phase], calls[phase] = gpu_ms / steps, n
    if missing:
        raise ValueError("; ".join(missing))
    scaled = {engine: ranks[engine] * per_step[phase] * calls[phase] for phase, engine in PHASE_ENGINE.items()}
    total = sum(scaled.values())
    return {"rule": "analytic_calls",
            "weights": {e: round(v / total, 6) for e, v in scaled.items()},
            "inputs": {"per_step_ms": {p: round(v, 4) for p, v in per_step.items()},
                       "calls_per_wave": calls, "ranks": ranks},
            "warnings": ["no probed sampling points: the steps are AgentX's analytic count, which "
                         "assumes every decode step runs CONC requests"]}


# --------------------------------------------------------------------------
# PD MASS  (mn.py pd-mass)
# --------------------------------------------------------------------------
# AgentX kernel extraction on pd_disagg. coverage_ledger.py's mass is already each
# engine's GPU time over the whole replay (a point's per-step kernel time times
# the steps it stands for), so the phases add once each is scaled by its engine's
# rank count:
#     S_k = 100 * sum_e R_e s_ke M_e / sum_e R_e M_e
# Equal ranks reduce this to single-node AgentX's rule: both phases' mass summed.

def pd_mass(coverage_paths: list, roles_path: str) -> dict:
    """{rule: gpu_time_mass, weights, inputs, warnings}; ValueError if a phase has no mass."""
    mass, warnings = {}, []
    for path in coverage_paths:
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle) or {}
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        for phase, entry in (doc.get("phases") or {}).items():
            if phase in mass:
                raise ValueError(f"phase {phase} is in more than one --coverage")
            mass[phase] = float((entry or {}).get("total_mass_us") or 0.0)
        warnings.extend(doc.get("warnings") or [])
    empty = [p for p in PHASE_ENGINE if mass.get(p, 0.0) <= 0]
    if empty:
        raise ValueError(f"no {empty} mass (phases.<P>.total_mass_us) in {coverage_paths}")
    ranks = role_ranks(roles_path)
    scaled = {engine: ranks[engine] * mass[phase] for phase, engine in PHASE_ENGINE.items()}
    total = sum(scaled.values())
    return {"rule": "gpu_time_mass",
            "weights": {e: round(v / total, 6) for e, v in scaled.items()},
            "inputs": {"mass_us": {p: mass[p] for p in PHASE_ENGINE}, "ranks": ranks},
            "warnings": warnings}


def apply_pd_mass(ranking_path: str, mass: dict) -> dict:
    """Rewrite a kernel-extract ranking's selection_pct_gpu_time as S_k; the agent's number is kept."""
    with open(ranking_path, encoding="utf-8") as handle:
        doc = json.load(handle)
    rows = doc.get("ranking") or []
    sums = {}
    for row in rows:
        shares = row.get("share_of_phase_pct") or {}
        unknown = sorted(set(shares) - set(PHASE_ENGINE))
        if unknown:
            raise ValueError(f"{row.get('id', row.get('name'))}: phase {unknown} names no pd_disagg engine "
                             f"(expected {sorted(PHASE_ENGINE)})")
        for phase, share in shares.items():
            sums[phase] = sums.get(phase, 0.0) + float(share)
    over = {p: round(s, 2) for p, s in sums.items() if s > 100.5}
    if over:
        raise ValueError(f"share_of_phase_pct sums to {over} (> 100.5): the shares are not the ledger's "
                         "phase_mass_share_pct, whose denominator includes the collective wait and the tail")
    weights = mass["weights"]
    for row in rows:
        row.setdefault("selection_pct_gpu_time_agent", row.get("selection_pct_gpu_time"))
        row["selection_pct_gpu_time"] = round(sum(
            float(share) * weights[PHASE_ENGINE[phase]]
            for phase, share in (row.get("share_of_phase_pct") or {}).items()), 4)
    doc["denominators"] = mass
    _write_json(ranking_path, doc)
    return doc


def cmd_pd_mass(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="mn.py pd-mass")
    ap.add_argument("--coverage", action="append", required=True,
                    help="coverage_ledger.py's coverage.json (repeatable: one per engine's ledger)")
    ap.add_argument("--roles", action="append", required=True,
                    help="trace-gate's mn_trace_roles.json (repeatable: one per probed engine)")
    ap.add_argument("--apply", required=True, metavar="RANKING", help="kernel_ranking.json to rewrite in place")
    args = ap.parse_args(argv)
    try:
        mass = pd_mass(args.coverage, args.roles)
        apply_pd_mass(args.apply, mass)
    except ValueError as exc:
        sys.stderr.write(f"!!! mn pd-mass: cannot mass the engines: {exc}\n")
        return 2
    w = mass["weights"]
    print(f">>> mn pd-mass: engine GPU-time mass prefill {w['prefill'] * 100:.1f}% / "
          f"decode {w['decode'] * 100:.1f}%; rewrote {args.apply}")
    for warning in mass["warnings"]:
        sys.stderr.write(f"!!! mn pd-mass: ledger warning: {warning}\n")
    return 0


# --------------------------------------------------------------------------
# PD POINTS  (mn.py pd-points)
# --------------------------------------------------------------------------
# Fixed-length pd_disagg prices its engines the way AgentX does: the kernel-extract
# skill's planner picks sampling points from each engine's own baseline log (PREFILL
# from the prefill engine, DECODE from the decode engine), bench_e2e's probe mode
# measures them, and coverage_ledger's mass feeds topn-merge --coverage.

ENGINE_PHASE = {engine: phase for phase, engine in PHASE_ENGINE.items()}
# `ServerArgs(key=v, ...)` in older sglang, `server_args={'key': v, ...}` in newer ones.
_SERVER_ARG = re.compile(r"\b(decode_log_interval|chunked_prefill_size|cuda_graph_bs(?:_decode)?|"
                         r"cuda_graph_max_bs(?:_decode)?)'?\s*[=:]\s*(\[[\d, ]*\]|\d+)")


def server_info(log: str) -> dict:
    """The planner's dispatch inputs from the server-args line sglang logs at startup."""
    with open(log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "ServerArgs(" in line or "server_args={" in line:
                info = {k: json.loads(v) for k, v in _SERVER_ARG.findall(line)}
                for key in ("cuda_graph_bs", "cuda_graph_max_bs"):
                    if key not in info and f"{key}_decode" in info:
                        info[key] = info[f"{key}_decode"]
                return info
    return {}


def pd_points(engine_logs: dict, skill_root: str, per_phase: int, out_dir: str) -> dict:
    """{points, uncovered_candidates, per_engine}: each engine's points of its own phase."""
    scripts = os.path.join(skill_root, "scripts")
    os.makedirs(out_dir, exist_ok=True)
    merged = {"points": [], "uncovered_candidates": [], "per_engine": {}}
    for engine, phase in ENGINE_PHASE.items():
        log = engine_logs.get(engine)
        if not log or not os.path.isfile(log):
            raise ValueError(f"no --engine-log {engine}= (got {log!r})")
        info = server_info(log)
        if not info.get("decode_log_interval"):
            raise ValueError(f"{log} has no sglang server-args line with decode_log_interval")
        paths = {k: os.path.join(out_dir, f"{k}_{engine}.json") for k in ("server_info", "observed", "points")}
        _write_json(paths["server_info"], info)
        for cmd in ([sys.executable, os.path.join(scripts, "observed_workload.py"), "--log", log,
                     "--server-info", paths["server_info"], "--out", paths["observed"]],
                    [sys.executable, os.path.join(scripts, "plan_sampling_points.py"),
                     "--observed", paths["observed"], "-n", str(2 * per_phase),
                     "--phase-budget", f"PREFILL={per_phase},DECODE={per_phase}",
                     "--server-info", paths["server_info"], "--out", paths["points"]]):
            run = subprocess.run(cmd, capture_output=True, text=True)
            if run.returncode:
                raise ValueError(f"{os.path.basename(cmd[1])} on the {engine} log failed: "
                                 f"{(run.stderr or run.stdout).strip()[-400:]}")
        with open(paths["points"], encoding="utf-8") as handle:
            plan = json.load(handle)
        own = [dict(p, engine=engine) for p in plan.get("points") or [] if p.get("phase") == phase]
        if not own:
            raise ValueError(f"the {engine} log gives no {phase} sampling point")
        merged["points"].extend(own)
        merged["uncovered_candidates"].extend(
            u for u in plan.get("uncovered_candidates") or [] if u.get("phase") == phase)
        merged["per_engine"][engine] = {"log": log, "phase": phase, **paths}
    return merged


def cmd_pd_points(argv: list) -> int:
    ap = argparse.ArgumentParser(prog="mn.py pd-points")
    ap.add_argument("--engine-log", action="append", required=True, metavar="NAME=PATH",
                    help="one engine's server log from the baseline leg (prefill=, decode=)")
    ap.add_argument("--skill-root", required=True, help="the kernel-extract skill directory")
    ap.add_argument("--points-per-phase", type=int, default=3)
    ap.add_argument("--out-dir", required=True, help="writes sampling_points.json and its inputs here")
    args = ap.parse_args(argv)
    try:
        merged = pd_points(dict(s.partition("=")[::2] for s in args.engine_log),
                           args.skill_root, args.points_per_phase, args.out_dir)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"!!! mn pd-points: cannot plan the engines' points: {exc}\n")
        return 2
    out = os.path.join(args.out_dir, "sampling_points.json")
    _write_json(out, merged)
    print(f">>> mn pd-points: wrote {out}")
    print("PROFILE_PROBE_IDS=" + ",".join(p["point_id"] for p in merged["points"]))
    return 0


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
    w = doc["engine_weights"]
    x = w["inputs"]
    lines = [f"# Profile Top-{len(rows)} — pd_disagg, per-engine weighted union\n",
             f"- source: `{doc['source']}`",
             f"- GPU time in the two profile windows: {doc['total_gpu_time_ms']:.2f} ms over {doc['num_kernel_launches']} launches, {doc['num_distinct_kernels']} distinct kernels (the windows differ in length; this sum weighs nothing)"]
    if w["rule"] == "gpu_time_mass":
        for eng in doc["per_engine"]:
            lines.append(f"  - `{eng['engine']}`: **{eng['weight_pct']:.1f}%** ({eng['total_gpu_time_ms']:.2f} ms in its window) — own table: `{eng['topN_json']}`")
        lines.append(
            f"- engine weights `gpu_time_mass`: each engine's whole-leg GPU time from its probed sampling points "
            f"(per-step kernel time × the steps its baseline log counts at that shape) × its ranks: "
            f"prefill {x['mass_us']['PREFILL'] / 1000:.1f} ms × {x['ranks']['prefill']}, "
            f"decode {x['mass_us']['DECODE'] / 1000:.1f} ms × {x['ranks']['decode']}")
    else:
        for eng in doc["per_engine"]:
            lines.append(f"  - `{eng['engine']}`: **{eng['weight_pct']:.1f}%** ({eng['total_gpu_time_ms']:.2f} ms in its window) — own table: `{eng['topN_json']}`")
        lines.append(
            f"- engine weights `analytic_calls` (no probed points; AgentX's no-log fallback): a step's mean "
            f"GPU time in the window × the analytic steps per wave × ranks: "
            f"prefill {x['per_step_ms']['PREFILL']:.2f} ms × {x['calls_per_wave']['PREFILL']} × {x['ranks']['prefill']}, "
            f"decode {x['per_step_ms']['DECODE']:.2f} ms × {x['calls_per_wave']['DECODE']} × {x['ranks']['decode']}")
    for warning in w.get("warnings") or []:
        lines.append(f"  - ledger warning: {warning}")
    split = doc.get("latency_split") or {}
    if split:
        lines.append(f"- latency split (explanation only, not a weight): TTFT {split['ttft_ms']:.1f} ms vs decode {split['decode_ms']:.1f} ms (TPOT {split['tpot_ms']:.2f} × {split['osl'] - 1:.1f}).")
    if doc.get("kv_handoff"):
        lines.extend(_kv_handoff_md(doc["kv_handoff"]))
    lines.append("\n**%gpu is the kernel's share of the GPU time one request costs across both engines** (share in its engine × that engine's weight), the same meaning as a single-node %gpu.\n")
    lines.append("| # | engine | kernel | class | backend | edit | calls | total ms | %gpu | %gpu in engine | avg us |")
    lines.append("|--|--------|--------|-------|---------|------|-------|----------|------|----------------|--------|")
    for k in rows:
        lines.append(
            f"| {k['rank']} | {k['engine']} | `{k['short_name']}` | {k['classification']} | {k['backend_guess']} | {'Y' if k['editable'] else 'N'} | {k['calls']} | {k['total_ms']:.3f} | {k['pct_gpu_time']:.1f} | {k['pct_gpu_time_in_engine']:.1f} | {k['avg_us']:.1f} |")
    lines.append("\n## Opt hints (top entries)\n")
    for k in rows[:12]:
        lines.append(f"- **{k['rank']}. [{k['engine']}] {k['short_name']}** ({k['pct_gpu_time']:.1f}% gpu): {k['opt_hint']}")
    return "\n".join(lines) + "\n"


def cmd_topn_merge(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", action="append", default=[], metavar="NAME=PATH",
                    help="one engine's profile_topN.json: prefill= and decode=")
    ap.add_argument("--out", default="", help="prefix; writes <out>.json and <out>.md")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--bench-summary", default="",
                    help="the baseline's bench_summary.json: its TTFT/TPOT for the latency split, "
                         "which explains and never weighs")
    ap.add_argument("--coverage", action="append", default=[],
                    help="coverage_ledger.py's coverage.json of the probed points (repeatable): "
                         "weigh the engines on their GPU-time mass; without it, on analytic step counts")
    ap.add_argument("--osl", type=float, default=0, help="the bench's OSL, for the latency split")
    ap.add_argument("--roles", default="", help="trace-gate's mn_trace_roles.json: each engine's ranks")
    ap.add_argument("--engine-log", action="append", default=[], metavar="NAME=PATH",
                    help="one engine's server log from the SAME leg (prefill=, decode=)")
    args = ap.parse_args(argv)
    engine_logs = dict(spec.partition("=")[::2] for spec in args.engine_log)
    handoff = kv_handoff(engine_logs.get("prefill", ""), engine_logs.get("decode", ""))
    if engine_logs and not handoff:
        sys.stderr.write(f"!!! mn topn-merge: no Prefill/Decode batch lines in {sorted(engine_logs.values())}; no kv_handoff block\n")

    loaded = {}
    try:
        for spec in args.engine:
            name, _, path = spec.partition("=")
            if not path or name not in PHASE_ENGINE.values() or name in loaded:
                raise ValueError(f"--engine wants prefill=PATH and decode=PATH once each, got '{spec}'")
            with open(path, encoding="utf-8") as handle:
                loaded[name] = (path, json.load(handle))
        absent = [e for e in PHASE_ENGINE.values() if e not in loaded]
        if absent:
            raise ValueError(f"no --engine for {absent}")
        if args.coverage:
            weights = pd_mass(args.coverage, args.roles)
        else:
            weights = pd_analytic({n: doc for n, (_p, doc) in loaded.items()}, role_ranks(args.roles))
        weights["inputs"]["osl"] = args.osl
        for warning in weights["warnings"]:
            sys.stderr.write(f"!!! mn topn-merge: {warning}\n")
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"!!! mn topn-merge: cannot ledger the engines: {exc}\n")
        return 2

    rows, per_engine = [], []
    for name, (path, doc) in loaded.items():
        weight = weights["weights"][name]
        per_engine.append({
            "engine": name, "topN_json": path,
            "total_gpu_time_ms": round(float(doc.get("total_gpu_time_ms") or 0.0), 4),
            "weight_pct": round(weight * 100.0, 2),
            "num_kernel_launches": doc.get("num_kernel_launches", 0),
            "num_distinct_kernels": doc.get("num_distinct_kernels", 0),
        })
        for entry in doc.get("top_kernels") or []:
            in_engine = float(entry.get("pct_gpu_time") or 0.0)
            rows.append({**entry, "engine": name, "pct_gpu_time_in_engine": in_engine,
                         "pct_gpu_time": round(in_engine * weight, 4)})

    rows.sort(key=lambda r: r["pct_gpu_time"], reverse=True)
    rows = rows[: args.top]
    for i, row in enumerate(rows, 1):
        row["rank"] = i

    out = {
        "source": "pd-merged:" + "+".join(loaded),
        "topology": "pd_disagg",
        "total_gpu_time_ms": round(sum(e["total_gpu_time_ms"] for e in per_engine), 4),
        "num_kernel_launches": sum(d.get("num_kernel_launches", 0) for _p, d in loaded.values()),
        "num_distinct_kernels": sum(d.get("num_distinct_kernels", 0) for _p, d in loaded.values()),
        "per_engine": per_engine,
        "top_kernels": rows,
        "engine_weights": weights,
        "bottleneck_engine": max(per_engine, key=lambda r: r["weight_pct"])["engine"],
    }
    split = latency_split(args.bench_summary, weights["inputs"]["osl"])
    if split:
        out["latency_split"] = split
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
        print(f"    {eng['engine']:<10} {eng['weight_pct']:5.1f}%  ({weights['rule']})")
    if not args.out:
        print(markdown)
    return 0
