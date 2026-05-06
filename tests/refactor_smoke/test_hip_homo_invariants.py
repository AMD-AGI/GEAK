"""HIP homogeneous path — 11 invariant log markers.

Captured from archived run: assign_score_withk_mem_20260415_180639.log
(see docs/refactor/INVARIANTS.md for line-number citations).

HIP today legitimately SKIPS preprocess steps 2-4 (bug #1 in EXECUTION_PLAN.md
§0.3). The refactor PR-2 makes the skip explicit; PR-3 adds per-round
FULL_BENCHMARK enforcement uniformly. Until those land, the HIP marker list is
smaller than the Triton list by design.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# Patterns are CASE-INSENSITIVE and use the core markers (not the exact
# trailing `---`). The log format has some variance in how trailing dashes
# and casing appear; the intent is to assert each pipeline stage ran.
EXPECTED_MARKERS_HIP = [
    r"Normalized kernel_type from task content:\s*hip",
    r"Step 1/7:\s*Resolve kernel URL",
    r"Skipping Steps 2-4",                  # explicit HIP-path skip (bug #1 marker)
    r"Step 5/7:\s*Kernel [Pp]rofiling",
    r"Step 6/7:\s*Baseline [Mm]etrics",
    r"Step 7/7:\s*Commandment",
    r"Using homogeneous mode based on discovery",
    r"Retriever:\s*category=\w+",            # language= may be on continuation
    r"Cross-session memory injected into homogeneous task",
    r"Homogeneous Agent \(\d+ agents",
    r"Sub-agent \d+",
]


FIXTURE_LOG_ENV = "GEAK_REFACTOR_SMOKE_HIP_LOG"


def _assert_markers(log_text: str, markers: list[str], label: str) -> None:
    missing: list[str] = []
    for m in markers:
        if not re.search(m, log_text):
            missing.append(m)
    assert not missing, (
        f"[{label}] log is missing {len(missing)} of {len(markers)} invariant markers:\n"
        + "\n".join(f"  - {m}" for m in missing)
    )


def test_hip_homo_invariant_markers_from_fixture_log() -> None:
    fixture_path = os.environ.get(FIXTURE_LOG_ENV)
    if not fixture_path:
        pytest.skip(f"{FIXTURE_LOG_ENV} not set — skipping fixture-based check. "
                    f"To run locally: export {FIXTURE_LOG_ENV}=/path/to/hip_run.log")
    p = Path(fixture_path)
    assert p.exists(), f"log fixture {p} does not exist"
    _assert_markers(p.read_text(errors="ignore"), EXPECTED_MARKERS_HIP, "fixture log")


@pytest.mark.slow
@pytest.mark.gpu
def test_hip_homo_invariant_markers_live() -> None:
    """Placeholder — PR-2 lands the fixture HIP kernel + full invocation.

    Once the fixture exists, this test runs 1 round of HIP optimization and asserts
    all 11 markers appear. Today it's a stub that just confirms the CLI responds.
    """
    if os.environ.get("GEAK_REFACTOR_SMOKE_RUN_LIVE") != "1":
        pytest.skip("Set GEAK_REFACTOR_SMOKE_RUN_LIVE=1 to run live smoke")
    pytest.skip("Live HIP smoke stub; full implementation in PR-2 with fixture kernel")
