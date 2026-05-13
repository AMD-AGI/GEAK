"""Profile backend policy and validity checks for preprocess."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProfileBackendDecision:
    backend: str
    explicit: bool
    reason: str
    fallback_backend: str | None = None
    rocm_version: str | None = None

    def to_dict(self) -> dict[str, str | bool | None]:
        return asdict(self)


def _parse_version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    parts = re.findall(r"\d+", value)
    if not parts:
        return None
    return tuple(int(part) for part in parts[:4])


def _version_at_least(value: str | None, minimum: tuple[int, ...]) -> bool:
    parsed = _parse_version_tuple(value)
    if parsed is None:
        return False
    width = max(len(parsed), len(minimum))
    return parsed + (0,) * (width - len(parsed)) >= minimum + (0,) * (width - len(minimum))


def parse_rocprofv3_rocm_version(output: str) -> str | None:
    """Extract the ROCm version from common rocprofv3 --version formats."""
    patterns = (
        r"rocm[_\s-]*version\s*[:=]\s*([0-9]+(?:\.[0-9]+){1,3})",
        r"ROCm\s+version\s*[:=]?\s*([0-9]+(?:\.[0-9]+){1,3})",
        r"ROCm\s+([0-9]+(?:\.[0-9]+){1,3})",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def detect_rocprofv3_rocm_version() -> str | None:
    """Return the ROCm version reported by rocprofv3, or None when unavailable."""
    candidates = (
        ("rocprofv3", "--version"),
        ("/opt/rocm/bin/rocprofv3", "--version"),
    )
    for cmd in candidates:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        version = parse_rocprofv3_rocm_version(output)
        if version:
            return version
    return None


def choose_profile_backend(
    kernel_type: str,
    *,
    env: Mapping[str, str] | None = None,
) -> ProfileBackendDecision:
    """Choose the profiler backend for a discovered kernel.

    TileLang has two viable ROCm paths in PR3:
    - Metrix/rocprofv3 on the ROCm 7.2 container stack, which gives counters.
    - legacy rocprof on older or mixed Python GPU stacks, which gives robust timing/resource data.
    """
    environ = env if env is not None else os.environ
    explicit = (environ.get("GEAK_PROFILE_BACKEND") or "").strip()
    if explicit and explicit.lower() != "auto":
        return ProfileBackendDecision(
            backend=explicit,
            explicit=True,
            reason="GEAK_PROFILE_BACKEND override",
            fallback_backend=None,
        )

    normalized_type = (kernel_type or "").strip().lower()
    if normalized_type == "tilelang":
        rocm_version = detect_rocprofv3_rocm_version()
        if _version_at_least(rocm_version, (7, 2, 0)):
            return ProfileBackendDecision(
                backend="metrix",
                explicit=False,
                reason="TileLang on rocprofv3 ROCm >= 7.2; prefer counter-rich Metrix profile",
                fallback_backend="rocprof-legacy",
                rocm_version=rocm_version,
            )
        return ProfileBackendDecision(
            backend="rocprof-legacy",
            explicit=False,
            reason="TileLang on ROCm < 7.2 or unknown rocprofv3 stack; use stable legacy profiler",
            fallback_backend=None,
            rocm_version=rocm_version,
        )

    return ProfileBackendDecision(
        backend="metrix",
        explicit=False,
        reason="default profiler backend",
        fallback_backend=None,
    )


def profile_result_has_kernels(profiling: object) -> bool:
    if not isinstance(profiling, dict):
        return False
    if profiling.get("success") is False:
        return False
    results = profiling.get("results")
    if not isinstance(results, list):
        return False
    for result in results:
        if not isinstance(result, dict):
            continue
        kernels = result.get("kernels")
        if isinstance(kernels, list) and any(isinstance(kernel, dict) for kernel in kernels):
            return True
    return False


def baseline_has_profiler_metrics(baseline_metrics: object) -> bool:
    if not isinstance(baseline_metrics, dict):
        return False
    if baseline_metrics.get("profiling_failed") or baseline_metrics.get("profiling_skipped"):
        return False
    top_kernels = baseline_metrics.get("top_kernels")
    metrics = baseline_metrics.get("metrics")
    return isinstance(top_kernels, list) and bool(top_kernels) and isinstance(metrics, dict) and bool(metrics)


def validate_required_profile(
    profiling: object,
    baseline_metrics: object,
) -> tuple[bool, str]:
    if not profile_result_has_kernels(profiling):
        return False, "profile.json is missing successful kernel records"
    if not baseline_has_profiler_metrics(baseline_metrics):
        return False, "baseline_metrics.json is missing profiler-derived metrics"
    return True, ""


def env_truthy(name: str, *, env: Mapping[str, str] | None = None) -> bool:
    environ = env if env is not None else os.environ
    value = (environ.get(name) or "").strip().lower()
    return value in {"1", "true", "yes", "on"}
