#!/usr/bin/env python3
"""Reduce per-rank in-kernel phase-meter dumps into per-stage busy/wait rows.

Deterministic, stdlib-only, operator-neutral: stage and role names come from
the dump itself. Each rank writes one JSON file:

  {
    "rank": 0,
    "tick_hz": 100000000,            # constant-rate counter (e.g. s_memrealtime)
    "n_workgroups": 256,             # workgroups launched (the divisor)
    "overflow": 0,                   # dropped records; > 0 voids the dump
    "roles": [{"name": "gemm1", "stage": "stage1", "kind": "busy"},
              {"name": "dispatch_wait", "stage": "stage1", "kind": "wait"}, ...],
    "events": [[wg, role_index, t_begin, t_end], ...]
  }

Per rank and stage:
  busy_ms = sum of that stage's busy intervals over all workgroups / n_workgroups
  wait_ms = sum of that stage's wait intervals over all workgroups / n_workgroups
  span_ms = wall-clock union of the stage's busy intervals (rank-local clock)
Then every quantity is reduced with rank-MAX (same reduction as the e2e metric).
Timestamps are never compared across ranks (no shared clock).

CLI:
  python3 reduce_stage_meter.py --dump 'DIR/stage_meter_rank*.json' \
      --guard 8192_uniform --output rows.json [--evidence '<cmd/log path>']

Output: {"valid": bool, "problems": [...], "kernel_ms_rank_max": x,
         "component_breakdown": [{component, busy_ms, wait_ms, span_ms, method,
                                  evidence, guard, per_rank}]}
Exit 0 even when invalid (the caller reports `problems`); exit 2 on bad CLI.
"""

import argparse
import glob
import json
import sys


def _union_len(intervals):
    total = 0
    cur_b = cur_e = None
    for b, e in sorted(intervals):
        if cur_e is None or b > cur_e:
            if cur_e is not None:
                total += cur_e - cur_b
            cur_b, cur_e = b, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_b
    return total


def reduce_rank(dump):
    """Return (per_stage {stage: {busy_ms, wait_ms, span_ms}}, kernel_ms, problems)."""
    problems = []
    rank = dump.get("rank", "?")
    hz = float(dump.get("tick_hz") or 0)
    n_wg = int(dump.get("n_workgroups") or 0)
    roles = dump.get("roles") or []
    events = dump.get("events") or []
    if hz <= 0:
        problems.append(f"rank {rank}: tick_hz missing or <= 0")
    if n_wg <= 0:
        problems.append(f"rank {rank}: n_workgroups missing or <= 0")
    if int(dump.get("overflow") or 0) > 0:
        problems.append(f"rank {rank}: overflow={dump.get('overflow')} (records dropped)")
    if not events:
        problems.append(f"rank {rank}: no events (meter not compiled in or not armed)")
    if problems:
        return {}, 0.0, problems

    to_ms = 1000.0 / hz
    busy, wait, spans = {}, {}, {}
    t_min = t_max = None
    per_wg_total = {}
    for ev in events:
        try:
            wg, ri, tb, te = int(ev[0]), int(ev[1]), int(ev[2]), int(ev[3])
        except (TypeError, ValueError, IndexError):
            problems.append(f"rank {rank}: malformed event {ev!r}")
            continue
        if not 0 <= ri < len(roles) or te < tb:
            problems.append(f"rank {rank}: bad role index or negative interval {ev!r}")
            continue
        role = roles[ri]
        stage = str(role.get("stage") or role.get("name"))
        kind = str(role.get("kind") or "busy")
        d = te - tb
        t_min = tb if t_min is None else min(t_min, tb)
        t_max = te if t_max is None else max(t_max, te)
        per_wg_total[wg] = per_wg_total.get(wg, 0) + d
        if kind == "wait":
            wait[stage] = wait.get(stage, 0) + d
        else:
            busy[stage] = busy.get(stage, 0) + d
            spans.setdefault(stage, []).append((tb, te))
    kernel_ticks = (t_max - t_min) if t_min is not None else 0
    # A workgroup cannot spend more time in roles than the kernel lasted: overlapping
    # role intervals inside one workgroup mean the meter's enter/leave pairing is broken.
    over = [wg for wg, tot in per_wg_total.items() if kernel_ticks and tot > kernel_ticks * 1.02]
    if over:
        problems.append(f"rank {rank}: {len(over)} workgroup(s) log more role time than the "
                        f"kernel span (nested/unpaired enter-leave)")
    stages = sorted(set(busy) | set(wait))
    out = {}
    for s in stages:
        out[s] = {
            "busy_ms": busy.get(s, 0) / n_wg * to_ms,
            "wait_ms": wait.get(s, 0) / n_wg * to_ms,
            "span_ms": _union_len(spans.get(s, [])) * to_ms,
        }
    return out, kernel_ticks * to_ms, problems


def reduce_dumps(dumps, guard, evidence):
    problems = []
    per_rank = {}
    kernel_ms = []
    if not dumps:
        problems.append("no per-rank dump files found")
    for dump in dumps:
        stages, k_ms, p = reduce_rank(dump)
        problems.extend(p)
        per_rank[str(dump.get("rank", len(per_rank)))] = stages
        kernel_ms.append(k_ms)
    all_stages = sorted({s for st in per_rank.values() for s in st})
    rows = []
    for s in all_stages:
        vals = {r: st.get(s) for r, st in per_rank.items() if st.get(s)}
        rows.append({
            "component": s,
            "busy_ms": max(v["busy_ms"] for v in vals.values()),
            "wait_ms": max(v["wait_ms"] for v in vals.values()),
            "span_ms": max(v["span_ms"] for v in vals.values()),
            "method": "timestamp",
            "evidence": evidence,
            "guard": guard,
            "per_rank": {r: {k: round(x, 6) for k, x in v.items()} for r, v in sorted(vals.items())},
        })
    return {
        "valid": not problems and bool(rows),
        "problems": problems,
        "kernel_ms_rank_max": max(kernel_ms) if kernel_ms else 0.0,
        "component_breakdown": rows,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reduce per-rank phase-meter dumps.")
    ap.add_argument("--dump", required=True, help="glob of per-rank dump JSON files")
    ap.add_argument("--guard", required=True)
    ap.add_argument("--evidence", default="")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    dumps = []
    for path in sorted(glob.glob(args.dump)):
        with open(path) as f:
            dumps.append(json.load(f))
    result = reduce_dumps(dumps, args.guard, args.evidence or args.dump)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"GEAK_STAGE_METER_ROWS={args.output} valid={result['valid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
