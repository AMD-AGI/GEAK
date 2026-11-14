"""Interval overlap and dependency-DAG primitives for distributed profile analysis."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from itertools import combinations
from typing import Iterable, Mapping, Sequence

__all__ = [
    "analyze_category_overlap",
    "critical_path",
    "interval_overlap_us",
    "interval_union_us",
    "merge_intervals",
]


def merge_intervals(intervals: Iterable[Sequence[float]]) -> list[list[float]]:
    """Validate and merge overlapping ``[start_us, end_us]`` intervals."""
    normalized = []
    for index, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError(f"interval {index} must have [start_us, end_us]")
        start, end = map(float, interval)
        if not (math.isfinite(start) and math.isfinite(end) and end >= start):
            raise ValueError(f"invalid interval {interval!r}")
        if end > start:
            normalized.append([start, end])
    normalized.sort()
    merged: list[list[float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def interval_union_us(intervals: Iterable[Sequence[float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_overlap_us(
    left: Iterable[Sequence[float]],
    right: Iterable[Sequence[float]],
) -> float:
    """Return the union duration covered by both interval sets."""
    a = merge_intervals(left)
    b = merge_intervals(right)
    i = j = 0
    overlap = 0.0
    while i < len(a) and j < len(b):
        overlap += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def analyze_category_overlap(
    per_category_intervals_us: Mapping[str, Iterable[Sequence[float]]],
) -> dict:
    """Compute per-category active time and pairwise physical overlap."""
    merged = {
        category: merge_intervals(intervals)
        for category, intervals in per_category_intervals_us.items()
    }
    active = {
        category: interval_union_us(intervals) / 1000.0
        for category, intervals in merged.items()
    }
    pairwise = {}
    for left, right in combinations(sorted(merged), 2):
        overlap_ms = interval_overlap_us(merged[left], merged[right]) / 1000.0
        smaller = min(active[left], active[right])
        pairwise[f"{left}|{right}"] = {
            "overlap_ms": overlap_ms,
            "overlap_of_smaller_pct": overlap_ms / smaller * 100.0 if smaller else 0.0,
        }
    all_intervals = [interval for intervals in merged.values() for interval in intervals]
    return {
        "category_active_ms": active,
        "pairwise": pairwise,
        "categorized_kernel_union_ms": interval_union_us(all_intervals) / 1000.0,
        "summed_category_active_ms": sum(active.values()),
    }


def critical_path(
    nodes: Mapping[str, float],
    edges: Iterable[Sequence[str]],
) -> dict:
    """Return the longest weighted path in an acyclic dependency graph.

    ``nodes`` maps node ID to duration in milliseconds. ``edges`` contains
    ``[producer, consumer]`` pairs.
    """
    durations = {}
    for node, duration in nodes.items():
        value = float(duration)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"node {node!r} has invalid duration {duration!r}")
        durations[str(node)] = value
    successors: dict[str, list[str]] = defaultdict(list)
    indegree = {node: 0 for node in durations}
    for index, edge in enumerate(edges):
        if len(edge) != 2:
            raise ValueError(f"edge {index} must contain [producer, consumer]")
        source, target = map(str, edge)
        if source not in durations or target not in durations:
            raise ValueError(f"edge references unknown node: {edge!r}")
        successors[source].append(target)
        indegree[target] += 1
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    distance = {node: durations[node] for node in durations}
    previous: dict[str, str] = {}
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in successors[node]:
            candidate = distance[node] + durations[target]
            if candidate > distance[target]:
                distance[target] = candidate
                previous[target] = node
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(durations):
        raise ValueError("dependency graph contains a cycle")
    if not durations:
        return {"critical_path_ms": 0.0, "path": []}
    tail = max(distance, key=distance.get)
    path = [tail]
    while tail in previous:
        tail = previous[tail]
        path.append(tail)
    path.reverse()
    return {"critical_path_ms": distance[path[-1]], "path": path}
