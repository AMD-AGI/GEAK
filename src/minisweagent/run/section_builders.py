"""Shared COMMANDMENT section body builders.

Both Path A (``commandment_from_user_command``) and Path B
(``_generate_simple`` / ``_generate_inner_kernel``) call these helpers
to produce the body strings for each COMMANDMENT section, ensuring
structural parity across paths.

This module is deliberately dependency-free (stdlib only) so it can be
imported from both ``preprocess/`` and ``preprocess_v3/`` without
creating circular imports.
"""

from __future__ import annotations

import re

_MODE_FLAG_PATTERN = re.compile(r"\s+--(?:full-benchmark|benchmark|correctness|profile)\b")


def strip_mode_flags(cmd: str) -> str:
    """Remove all known harness mode flags from *cmd*.

    Uses regex with longest-first alternation so ``--full-benchmark`` is
    matched whole before ``--benchmark`` gets a chance.
    """
    return _MODE_FLAG_PATTERN.sub("", cmd).strip()


def warmup_block(command: str, warmup_runs: int) -> str:
    """Build the warmup sub-section for the PROFILE block."""
    if warmup_runs <= 0:
        return ""
    if warmup_runs == 1:
        return command
    return f"for _i in $(seq 1 {warmup_runs}); do {command}; done"


def build_correctness_body(base_cmd: str) -> str:
    """Generate the CORRECTNESS section body."""
    return f"{base_cmd} --correctness"


def build_profile_body(
    base_cmd: str,
    *,
    warmup_runs: int = 2,
    profile_replays: int = 3,
) -> str:
    """Generate the PROFILE section body (warmup + kernel-profile)."""
    profile_cmd = f"{base_cmd} --profile"
    warmup = warmup_block(
        f"{profile_cmd} > /dev/null 2>&1 || true",
        warmup_runs,
    )
    kernel_profile = (
        f'kernel-profile "{profile_cmd}"'
        f" --gpu-devices ${{GEAK_GPU_DEVICE}}"
        f" --replays {profile_replays}"
        f" --json -o ${{GEAK_WORK_DIR}}/profile.json"
    )
    return f"{warmup}\n{kernel_profile}" if warmup else kernel_profile


def build_benchmark_body(base_cmd: str) -> str:
    """Generate the BENCHMARK section body (uses ``--full-benchmark``)."""
    return f"{base_cmd} --full-benchmark ${{GEAK_BENCHMARK_EXTRA_ARGS:-}}"


def build_full_benchmark_body(base_cmd: str) -> str:
    """Generate the FULL_BENCHMARK section body (uses ``--full-benchmark``)."""
    return f"{base_cmd} --full-benchmark ${{GEAK_BENCHMARK_EXTRA_ARGS:-}}"
