"""Triton heterogeneous path — 12 invariant log markers.

Captured from live run: gemm_a16w16_atomic_canonical-rocm700_memon_20260422_083415.log
(see docs/refactor/INVARIANTS.md for line-number citations).

These markers are what the refactor must preserve. PR-2 introduces phase-based
markers alongside during transition; PR-2 extends the EXPECTED_MARKERS list to
include BOTH forms. After AKA migration, old markers may be dropped.

Running this test REQUIRES a GPU + GEAK_LLM_API_KEY environment. Marked slow.
In CI, we use the FIXTURE_LOG mode which parses a recorded log against the
markers (no GPU needed).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Patterns are case-insensitive where the live log uses mixed case (the
# audit's original list was slightly wrong: step 4 is "Baseline", not
# "Harness Validation"; steps 2/3/5/6 use lowercase second-word).
# Intent: assert each pipeline stage ran, not exact string match.
EXPECTED_MARKERS_TRITON = [
    r"Normalized kernel_type from task content:\s*triton",
    r"Step 1/7:\s*Resolve kernel URL",
    r"Step 2/7:\s*Codebase [Cc]ontext",
    r"Step 3/7:\s*Test [Dd]iscovery",
    r"Step 4/7:\s*Baseline",
    r"Step 5/7:\s*Kernel [Pp]rofiling",
    r"Step 6/7:\s*Baseline [Mm]etrics",
    r"Step 7/7:\s*Commandment",
    r"Using (heterogeneous|planned) mode based on discovery",
    r"run_orchestrator:",                      # line continues on next log line
    r"mode=planned",                           # log line: run_orchestrator: ... mode=planned
    r"Cross-session memory",
    r"Exploration Phase",
]


# --- Fixture mode: validate a pre-captured log file without running GEAK ---
# Allows the test to run in CI without GPU / LLM access.
FIXTURE_LOG_ENV = "GEAK_REFACTOR_SMOKE_TRITON_LOG"


def _assert_markers(log_text: str, markers: list[str], label: str) -> None:
    missing: list[str] = []
    for m in markers:
        if not re.search(m, log_text):
            missing.append(m)
    assert not missing, (
        f"[{label}] log is missing {len(missing)} of {len(markers)} invariant markers:\n"
        + "\n".join(f"  - {m}" for m in missing)
    )


def test_triton_hetero_invariant_markers_from_fixture_log() -> None:
    """If GEAK_REFACTOR_SMOKE_TRITON_LOG is set, validate that log file.

    Production use: CI exports this to a canonical captured log; the test just greps.
    """
    fixture_path = os.environ.get(FIXTURE_LOG_ENV)
    if not fixture_path:
        pytest.skip(f"{FIXTURE_LOG_ENV} not set — skipping fixture-based check. "
                    f"To run locally: export {FIXTURE_LOG_ENV}=/path/to/triton_run.log")
    p = Path(fixture_path)
    assert p.exists(), f"log fixture {p} does not exist"
    _assert_markers(p.read_text(errors="ignore"), EXPECTED_MARKERS_TRITON, "fixture log")


@pytest.mark.slow
@pytest.mark.gpu
def test_triton_hetero_invariant_markers_live() -> None:
    """Run a 1-round Triton optimization and assert all 12 markers appear.

    Requires GPU + GEAK_LLM_API_KEY. Skipped in CI without --run-slow.

    This test is a placeholder stub until PR-2 lands the fixture kernel + `geak -t`
    invocation helper. For now it only runs if explicitly opted in via env.
    """
    if os.environ.get("GEAK_REFACTOR_SMOKE_RUN_LIVE") != "1":
        pytest.skip("Set GEAK_REFACTOR_SMOKE_RUN_LIVE=1 to run the 1-round live smoke")

    # Placeholder — PR-2 will supply the fixture kernel path and full `geak -t` call.
    # For now this demonstrates the intended shape.
    result = subprocess.run(
        ["geak", "--help"],  # placeholder command
        capture_output=True, text=True, timeout=300,
    )
    log = result.stdout + result.stderr
    # Can't assert the 12 Triton markers against `geak --help`; this is a stub.
    # PR-2 replaces this body with a real 1-round invocation.
    assert "geak" in log.lower(), "geak CLI not responding"
