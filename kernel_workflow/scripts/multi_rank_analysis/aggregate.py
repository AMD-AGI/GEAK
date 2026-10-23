"""Generic per-rank scalar reduction, timing, and merge primitives.

Extracted and generalized from the module-private gather / reduce / graph-timing helpers
written for one distributed MoE UT (their identifiers are deliberately not reproduced — see this
package's README for why a private name is an address, not a citation). Nothing here is named after that UT, its operator, or its
backend: every function takes plain values/callables and returns plain dicts, so it can be reused by
any distributed correctness/benchmark harness that GEAK's Kernel Workflow drives.

Two independent halves:
  * ``reduce_scalar`` / ``all_ranks_true`` / ``time_distributed`` require a live
    ``torch.distributed`` process group (torch is imported lazily, only when called) — use these
    INSIDE a running distributed UT/benchmark to produce the per-rank numbers.
  * ``merge_rank_records`` / ``classify_case_speedup`` are pure Python (no torch import at all) —
    use these OFFLINE, after per-rank records have already been gathered (e.g. via
    ``dist.all_gather_object`` in the caller, or read back from a JSON report), including from
    unit tests that have no GPU.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

_EPS = 1e-9

__all__ = [
    "reduce_scalar",
    "all_ranks_true",
    "time_distributed",
    "merge_rank_records",
    "classify_case_speedup",
]


def _require_torch_distributed():
    """Lazy import so this module has zero import-time cost/dependency for offline-only callers."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "multi_rank_analysis.aggregate: reduce_scalar/all_ranks_true/time_distributed require "
            "an initialized torch.distributed process group. Offline callers should use "
            "merge_rank_records()/classify_case_speedup() instead (no torch dependency)."
        )
    return torch, dist


def reduce_scalar(value: float, op: str = "mean", device=None) -> float:
    """All-reduce one scalar across the current distributed process group.

    ``op`` is one of "mean" | "max" | "min" | "sum". Generalizes the "gather one float per rank via
    all_reduce" pattern common to distributed UTs (previously duplicated per-operator as a private
    ``_reduce_float`` helper).
    """
    torch, dist = _require_torch_distributed()
    ops = {
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
        "mean": dist.ReduceOp.SUM,
    }
    if op not in ops:
        raise ValueError(f"reduce_scalar: unsupported op={op!r}, expected one of {sorted(ops)}")
    if not math.isfinite(float(value)):
        raise ValueError(f"reduce_scalar: value must be finite, got {value}")
    if device is None and "nccl" in str(dist.get_backend()).lower():
        device = torch.device("cuda", torch.cuda.current_device())
    result = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(result, op=ops[op])
    out = float(result.item())
    if op == "mean":
        out /= dist.get_world_size()
    return out


def all_ranks_true(value: bool, device=None) -> bool:
    """True iff every rank in the process group passed True. MIN-reduce over a 0/1 tensor."""
    return reduce_scalar(1.0 if value else 0.0, op="min", device=device) >= 1.0


def time_distributed(fn: Callable[[], Any], iters: int, device=None) -> dict:
    """Time ``fn`` locally with CUDA events over ``iters`` calls, then reduce mean/max across ranks.

    Returns ``{"local_ms", "rank_mean_ms", "rank_max_ms"}`` — the canonical shape every per-rank
    timing metric in this framework is expressed in. Generalizes the "CUDA-graph replay + all-reduce
    mean/max" pattern (previously a private, operator-named ``_time_graph`` helper).
    """
    import torch

    if iters <= 0:
        raise ValueError(f"time_distributed: iters must be positive, got {iters}")
    _, dist = _require_torch_distributed()
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    return {
        "local_ms": local_ms,
        "rank_mean_ms": reduce_scalar(local_ms, op="mean", device=device),
        "rank_max_ms": reduce_scalar(local_ms, op="max", device=device),
    }


def _get_path(record: Mapping[str, Any], dotted_path: str):
    node: Any = record
    for part in dotted_path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _span_pct(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("span requires at least one value")
    mean = sum(values) / len(values)
    return (max(values) - min(values)) / max(abs(mean), _EPS) * 100.0


def _ranked_records(
    records: Sequence[Mapping[str, Any]],
    expected_ranks: Iterable[int] | None = None,
) -> tuple[list[tuple[int, Mapping[str, Any]]], list[int]]:
    ranked = []
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"rank record at index {index} must be a mapping")
        rank = record.get("rank", index)
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError(f"rank record at index {index} has invalid rank={rank!r}")
        if rank in seen:
            raise ValueError(f"duplicate rank record: rank={rank}")
        seen.add(rank)
        ranked.append((rank, record))
    expected = set(expected_ranks or [])
    invalid_expected = [
        rank
        for rank in expected
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
    ]
    if invalid_expected:
        raise ValueError(
            "expected_ranks must contain non-negative integers, "
            f"got invalid values {invalid_expected!r}"
        )
    unexpected = seen - expected if expected else set()
    if unexpected:
        raise ValueError(f"rank records contain unexpected ranks: {sorted(unexpected)}")
    return ranked, sorted(expected - seen)


def merge_rank_records(
    records: Sequence[Mapping[str, Any]],
    metric_paths: Iterable[str],
    repetitions: Sequence[Sequence[Mapping[str, Any]]] | None = None,
    expected_ranks: Iterable[int] | None = None,
) -> dict:
    """Merge N per-rank records (already gathered by the caller) into rank_mean/max/min/tail stats.

    ``records``: one dict per rank for a single run, e.g. ``[{"timing_ms": {"stage1": 0.42}}, ...]``.
    ``metric_paths``: dotted paths into each record, e.g. ``["timing_ms.stage1", "timing_ms.e2e"]``.
    ``repetitions``: OPTIONAL — a list of independent repetitions, each itself a list of per-rank
        records shaped like ``records``. When given, also emits ``*_runs``/``*_span_pct`` (the
        cross-repetition noise band), matching the field names already used by the one existing
        analysis artifact this framework generalizes (see README's "Ground truth" section).

    Returns ``{metric_path: {"rank_mean", "rank_max", "rank_min", "rank_tail_spread_pct",
    ["rank_mean_runs", "rank_max_runs", "rank_mean_span_pct", "rank_max_span_pct"]}}``.

    Never raises on a missing metric path in some ranks — those ranks are simply excluded from that
    metric's aggregate (recorded in ``missing_ranks``), so a partially-populated record set still
    produces a usable (if incomplete) report; callers/skills decide how to react to `missing_ranks`.
    """
    ranked_records, absent_ranks = _ranked_records(records, expected_ranks)
    paths = list(metric_paths)
    out: dict[str, dict] = {}
    for path in paths:
        values = []
        missing_ranks = list(absent_ranks)
        for rank, rec in ranked_records:
            v = _get_path(rec, path)
            if (
                v is None
                or isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
            ):
                missing_ranks.append(rank)
                continue
            values.append(float(v))
        entry: dict[str, Any] = {"missing_ranks": sorted(missing_ranks)}
        if values:
            vmax, vmin = max(values), min(values)
            entry["rank_mean"] = sum(values) / len(values)
            entry["rank_max"] = vmax
            entry["rank_min"] = vmin
            entry["rank_tail_spread_pct"] = _span_pct(values)
        else:
            entry["rank_mean"] = None
            entry["rank_max"] = None
            entry["rank_min"] = None
            entry["rank_tail_spread_pct"] = None
        out[path] = entry

    if repetitions:
        for path in paths:
            mean_runs = []
            max_runs = []
            min_runs = []
            tail_spread_runs = []
            missing_by_run = []
            for repetition_index, rep_records in enumerate(repetitions):
                rep_merged = merge_rank_records(
                    rep_records,
                    [path],
                    expected_ranks=expected_ranks,
                )
                m = rep_merged[path]
                if m["rank_mean"] is not None:
                    mean_runs.append(m["rank_mean"])
                if m["rank_max"] is not None:
                    max_runs.append(m["rank_max"])
                if m["rank_min"] is not None:
                    min_runs.append(m["rank_min"])
                if m["rank_tail_spread_pct"] is not None:
                    tail_spread_runs.append(m["rank_tail_spread_pct"])
                if m["missing_ranks"]:
                    missing_by_run.append(
                        {
                            "repetition": repetition_index,
                            "missing_ranks": list(m["missing_ranks"]),
                        }
                    )
            entry = out[path]
            entry["rank_mean_runs"] = mean_runs
            entry["rank_max_runs"] = max_runs
            entry["rank_min_runs"] = min_runs
            entry["rank_tail_spread_pct_runs"] = tail_spread_runs
            entry["repetition_missing_ranks"] = missing_by_run
            for headline, runs in (
                ("rank_mean", mean_runs),
                ("rank_max", max_runs),
                ("rank_min", min_runs),
                ("rank_tail_spread_pct", tail_spread_runs),
            ):
                if runs:
                    entry[headline] = sum(runs) / len(runs)
            for key, runs in (("rank_mean_span_pct", mean_runs), ("rank_max_span_pct", max_runs)):
                if len(runs) >= 2:
                    entry[key] = _span_pct(runs)
                else:
                    entry[key] = 0.0 if runs else None
    return out


def classify_case_speedup(
    candidate_rank_max: float,
    baseline_rank_max: float,
    min_pct: float | None = None,
) -> dict:
    """Speedup of a candidate vs. a baseline rank-max latency, with an optional acceptance floor.

    Generalizes the "static per-(shape,route) minimum speedup table" pattern (previously a
    MegaMoE-specific ``PERF_GUARD_MIN_SPEEDUP`` dict keyed by (tokens, route)) into a stateless
    function: the caller supplies its own threshold (or omits it for an unguarded comparison).

    Returns ``{"speedup_pct", "status": "pass"|"fail"|"unguarded"}``.
    """
    if candidate_rank_max <= 0 or not math.isfinite(candidate_rank_max):
        raise ValueError(
            "classify_case_speedup: candidate_rank_max must be positive finite, "
            f"got {candidate_rank_max}"
        )
    if baseline_rank_max <= 0 or not math.isfinite(baseline_rank_max):
        raise ValueError(
            "classify_case_speedup: baseline_rank_max must be positive finite, "
            f"got {baseline_rank_max}"
        )
    if min_pct is not None and not math.isfinite(min_pct):
        raise ValueError(f"classify_case_speedup: min_pct must be finite, got {min_pct}")
    speedup_pct = (baseline_rank_max / candidate_rank_max - 1.0) * 100.0
    if min_pct is None:
        status = "unguarded"
    else:
        status = "pass" if speedup_pct >= min_pct else "fail"
    return {"speedup_pct": speedup_pct, "status": status}
