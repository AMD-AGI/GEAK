"""Shared helpers for profiler-mcp Metrix invocations."""

from __future__ import annotations

from typing import Any

DEFAULT_METRIX_NUM_REPLAYS = 3
DEFAULT_METRIX_QUICK = False


def build_metrix_profile_kwargs(
    command: str,
    gpu_devices: str | int,
    *,
    quick: bool = DEFAULT_METRIX_QUICK,
    num_replays: int = DEFAULT_METRIX_NUM_REPLAYS,
    workdir: str | None = None,
) -> dict[str, Any]:
    """Build the canonical profiler-mcp kwargs for Metrix profiling.

    GEAK uses one shared Metrix profile shape across preprocessing, evaluation,
    and per-patch save_and_test profiling so knobs like replay count and quick
    mode cannot silently drift between call sites.
    """

    kwargs: dict[str, Any] = {
        "command": command,
        "backend": "metrix",
        "num_replays": num_replays,
        "quick": quick,
        "gpu_devices": str(gpu_devices),
    }
    if workdir is not None:
        kwargs["workdir"] = workdir
    return kwargs
