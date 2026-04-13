"""Shared helpers for parsing benchmark output across GEAK.

These helpers normalize raw benchmark output from multiple backends into a
consistent latency/speedup view. In particular, they handle:

- Standardized GEAK markers (``GEAK_RESULT_LATENCY_MS=...``)
- Triton/table-style benchmark summaries
- Raw HIP/CUDA ``Perf: ... (shape_*)`` lines
- Simple shape-labeled outputs such as ``hd=128: 0.0200ms``

For multi-shape outputs without an explicit overall marker, the canonical
"overall" latency is the geometric mean of the parsed per-shape latencies.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

_NUM_RE = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_UNIT_RE = r"(?:us|µs|ms|s)"
_SUMMARY_LABEL_KEYWORDS = (
    "median",
    "geomean",
    "overall",
    "latency",
    "benchmark",
    "total",
    "speedup",
    "reported",
)

_GEAK_LATENCY_RE = re.compile(rf"GEAK_RESULT_LATENCY_MS\s*=\s*({_NUM_RE})", re.IGNORECASE)
_PAREN_SHAPE_RE = re.compile(rf"^\s*(\([^)]*\)):\s*({_NUM_RE})\s*({_UNIT_RE})\s*$", re.IGNORECASE)
_PERF_SHAPE_RE = re.compile(
    rf"^\s*Perf:\s*({_NUM_RE})\s*({_UNIT_RE})(?:/launch)?\s*\(([^)]+)\)\s*$",
    re.IGNORECASE,
)
_GENERIC_SHAPE_RE = re.compile(rf"^\s*([^:][^:]{{0,200}}?):\s*({_NUM_RE})\s*({_UNIT_RE})\s*$", re.IGNORECASE)
_PERF_LAUNCH_RE = re.compile(rf"^\s*Perf:\s*({_NUM_RE})\s*({_UNIT_RE})(?:/launch)?\s*$", re.IGNORECASE)


def _latency_to_ms(value: str, unit: str) -> float:
    scale = {
        "us": 1.0 / 1000.0,
        "µs": 1.0 / 1000.0,
        "ms": 1.0,
        "s": 1000.0,
    }[unit.lower()]
    return float(value) * scale


def _geomean_ms(values: Iterable[float]) -> float | None:
    positives = [float(value) for value in values if value and float(value) > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def parse_median_latency_ms(output: str) -> float | None:
    """Extract median latency (ms) from harness benchmark output."""
    m = re.search(
        rf"(?:[Mm]edian\s+(?:latency|time)[\w\s]*|total\s+median\s+time)\s*:\s*({_NUM_RE})\s*ms",
        output,
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def parse_total_kernel_time_ms(output: str) -> float | None:
    """Extract TOTAL_KERNEL_TIME_MS or BENCHMARK_LATENCY_MS from benchmark output."""
    m = re.search(
        rf"(?:TOTAL_KERNEL_TIME_MS|BENCHMARK_LATENCY_MS)\s*[:=]\s*({_NUM_RE})",
        output,
        re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def _parse_benchmark_metric(output: str) -> float | None:
    """Extract from BENCHMARK_METRIC:, median_latency_ms:, or Geomean (ms): lines."""
    for pat in (
        rf"BENCHMARK_METRIC:\s*median_latency_ms=({_NUM_RE})",
        rf"median_latency_ms:\s*({_NUM_RE})",
        rf"Geomean\s*\(ms\)\s*:\s*({_NUM_RE})",
    ):
        m = re.search(pat, output, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_google_benchmark_ms(output: str) -> float | None:
    """Parse Google Benchmark format: <name> <iters> <latency> ms."""
    m = re.search(rf"^\S+\s+\d+\s+({_NUM_RE})\s+ms", output, re.MULTILINE)
    return float(m.group(1)) if m else None


def parse_shape_latencies_ms(output: str) -> dict[str, float]:
    """Extract per-shape latencies from benchmark output.

    Supported formats include:
        ``(32,4096): 0.0503 ms``
        ``Perf: 0.2242 ms (shape_4_forward_backward)``
        ``hd=256: 0.0300ms``
    """
    shape_latencies: dict[str, float] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        perf_match = _PERF_SHAPE_RE.match(line)
        if perf_match:
            shape_latencies[perf_match.group(3).strip()] = _latency_to_ms(
                perf_match.group(1), perf_match.group(2)
            )
            continue

        paren_match = _PAREN_SHAPE_RE.match(line)
        if paren_match:
            shape_latencies[paren_match.group(1)] = _latency_to_ms(paren_match.group(2), paren_match.group(3))
            continue

        generic_match = _GENERIC_SHAPE_RE.match(line)
        if generic_match:
            label = generic_match.group(1).strip()
            lower = label.lower()
            if label.lower() == "perf" or any(keyword in lower for keyword in _SUMMARY_LABEL_KEYWORDS):
                continue
            shape_latencies[label] = _latency_to_ms(generic_match.group(2), generic_match.group(3))

    return shape_latencies


def parse_shape_count(output: str) -> int | None:
    """Extract shape count from harness benchmark output."""
    m = re.search(r"(\d+)\s+shapes", output, re.IGNORECASE)
    if m:
        return int(m.group(1))
    shape_latencies = parse_shape_latencies_ms(output)
    return len(shape_latencies) if shape_latencies else None


def extract_benchmark_config_lines(output: str) -> list[str] | None:
    """Extract benchmark config fingerprint lines from harness output."""
    shape_latencies = parse_shape_latencies_ms(output)
    if shape_latencies:
        return sorted(shape_latencies)

    configs: list[str] = []
    timing_pattern = re.compile(rf"{_NUM_RE}\s*(?:ms|us|µs|s|x)")
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith(("-", "=", "#", "Status", "Geometric", "GEAK_")):
            continue
        if not timing_pattern.search(line):
            continue
        if any(kw in line.lower() for kw in ("comparing", "running", "warmup", "median", "geomean", "mean")):
            continue
        config_part = re.split(rf"(?<=[=:])\s*{_NUM_RE}|\s+{_NUM_RE}", line)[0].strip()
        config_part = re.sub(r"[\s:|]+$", "", config_part)
        config_part = re.sub(r"\s*\|\s*\w+$", "", config_part)
        config_part = re.sub(r":\s*\w+=$", "", config_part)
        if config_part:
            configs.append(config_part)
    return sorted(configs) if configs else None


def _universal_latency_fallback(text: str) -> float | None:
    """Last-resort latency parser based on latency-related keywords."""
    keywords = {"median", "overall", "geomean", "latency", "total", "perf"}
    candidates: list[float] = []
    lines = text.strip().splitlines()
    for line in lines[-30:]:
        lower = line.lower()
        if not any(kw in lower for kw in keywords):
            continue
        for m in re.finditer(rf"({_NUM_RE})\s*ms", line, re.IGNORECASE):
            val = float(m.group(1))
            if 0.0001 < val < 100000:
                candidates.append(val)
    return candidates[-1] if candidates else None


def _shape_geomean_ms(text: str) -> float | None:
    shape_latencies = parse_shape_latencies_ms(text)
    if not shape_latencies:
        return None
    return _geomean_ms(shape_latencies.values())


def _perf_launch_latency_ms(text: str) -> float | None:
    latencies_ms: list[float] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _PERF_LAUNCH_RE.match(line)
        if not m:
            continue
        latencies_ms.append(_latency_to_ms(m.group(1), m.group(2)))
    if not latencies_ms:
        return None
    return _geomean_ms(latencies_ms)


def extract_latency_ms(text: str) -> float | None:
    """Extract the canonical latency metric (ms) from benchmark output.

    Priority:
    1. GEAK marker, unless multi-shape lines disagree materially.
    2. Explicit total/median/geomean-style summaries.
    3. Shape-aware geomean from per-shape output.
    4. Perf-per-launch fallback.
    5. Universal keyword-based scanner.
    """
    shape_latencies = parse_shape_latencies_ms(text)
    shape_geomean = _geomean_ms(shape_latencies.values()) if shape_latencies else None

    m = _GEAK_LATENCY_RE.search(text)
    if m:
        explicit_latency_ms = float(m.group(1))
        if shape_geomean is not None and len(shape_latencies) >= 2:
            rel_delta = abs(explicit_latency_ms - shape_geomean) / max(shape_geomean, 1e-12)
            if rel_delta > 0.05:
                return shape_geomean
        return explicit_latency_ms

    val = parse_total_kernel_time_ms(text)
    if val is not None:
        return val
    val = _parse_benchmark_metric(text)
    if val is not None:
        return val
    val = parse_median_latency_ms(text)
    if val is not None:
        return val
    val = parse_google_benchmark_ms(text)
    if val is not None:
        return val
    if shape_geomean is not None:
        return shape_geomean
    val = _perf_launch_latency_ms(text)
    if val is not None:
        return val
    return _universal_latency_fallback(text)


def extract_reported_speedup(text: str) -> float | None:
    """Extract a reported speedup scalar from benchmark output."""
    for pat in (
        rf"GEAK_RESULT_GEOMEAN_SPEEDUP=({_NUM_RE})",
        rf"GEAK_RESULT_SPEEDUP=({_NUM_RE})",
        rf"Geometric mean speedup:\s*({_NUM_RE})x",
        rf"Speedup\s*\(geomean\)\s*:\s*({_NUM_RE})x",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def compute_shape_speedups(
    baseline_shapes_ms: dict[str, float],
    candidate_shapes_ms: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Compute per-shape speedups for the overlap between baseline and candidate."""
    results: dict[str, dict[str, float]] = {}
    for shape, baseline_ms in baseline_shapes_ms.items():
        candidate_ms = candidate_shapes_ms.get(shape)
        if candidate_ms is None or baseline_ms <= 0 or candidate_ms <= 0:
            continue
        results[shape] = {
            "baseline_ms": round(baseline_ms, 6),
            "candidate_ms": round(candidate_ms, 6),
            "speedup": round(baseline_ms / candidate_ms, 6),
        }
    return results
