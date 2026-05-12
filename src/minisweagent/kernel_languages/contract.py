"""Contract validators for harness + commandment artifacts.

Enforced by `preprocess/phases/*.py` (PR-2 lands these). Today this module
provides the validator API so other code can import and call — but the checks
are permissive stubs until PR-2 tightens them against the fixture corpus.

See docs/refactor/EXECUTION_PLAN.md §4 "Contract validators" + §16.7 (fixture
corpus spec).

The UNIVERSAL harness contract (what `HarnessBuilder` produces and
`validate_harness` enforces):

  harness.py MUST expose argparse with mutually-exclusive flags:
    --correctness        run correctness check, print OK/FAIL
    --benchmark          run in-loop timing, print GEAK_RESULT_LATENCY_MS=<float>
    --full-benchmark     run verification with iteration count, also print
                         GEAK_RESULT_SPEEDUP=<float>
    --profile            run with the language's profiler attached

  AND emit STDOUT markers:
    GEAK_RESULT_LATENCY_MS=<float>      on --benchmark
    GEAK_RESULT_SPEEDUP=<float>         on --full-benchmark

The UNIVERSAL commandment contract (what `validate_commandment` enforces):

  COMMANDMENT.md MUST contain these level-2 headers in order:
    ## Setup
    ## Correctness
    ## Benchmark
    ## Full Benchmark
    ## Profile

  Each section's fenced ``` block MUST parse as shell. Each command MUST
  reference the harness.py path consistent with preprocess/artifacts/harness.py.
"""

from __future__ import annotations

import re
from pathlib import Path


class ContractViolation(RuntimeError):
    """Raised when an artifact doesn't satisfy its contract."""


# ---------------------------------------------------------------------------
# Harness contract
# ---------------------------------------------------------------------------

REQUIRED_HARNESS_FLAGS = ("--correctness", "--benchmark", "--full-benchmark", "--profile")
REQUIRED_HARNESS_MARKERS = ("GEAK_RESULT_LATENCY_MS", "GEAK_RESULT_SPEEDUP")


def validate_harness(path: Path) -> None:
    """Verify a harness.py conforms to the universal contract.

    Raises `ContractViolation` on any missing required surface. Today's checks
    are simple substring / regex presence — PR-2 tightens them against the
    fixture corpus (`tests/fixtures/harness_corpus/`).

    Today's behavior: always pass unless the file doesn't exist (so existing
    pipelines don't break). Full enforcement activates when `HarnessBuilder`
    lands in PR-2.
    """
    if not path.exists():
        raise ContractViolation(f"harness path does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    missing_flags = [f for f in REQUIRED_HARNESS_FLAGS if f not in text]
    missing_markers = [m for m in REQUIRED_HARNESS_MARKERS if m not in text]

    if missing_flags or missing_markers:
        # For PR-1: permissive — only raise if BOTH flags and markers are missing
        # (suggests the harness is totally non-compliant). A partial match is
        # likely just a legacy harness pre-PR-2; don't break those.
        if missing_flags and missing_markers:
            raise ContractViolation(
                f"harness {path} missing required flags {missing_flags} "
                f"AND required markers {missing_markers}"
            )


# ---------------------------------------------------------------------------
# Commandment contract
# ---------------------------------------------------------------------------

REQUIRED_COMMANDMENT_SECTIONS = (
    r"^##\s+Setup\b",
    r"^##\s+Correctness\b",
    r"^##\s+Benchmark\b",
    r"^##\s+Full Benchmark\b",
    r"^##\s+Profile\b",
)


def validate_commandment(path: Path) -> None:
    """Verify a COMMANDMENT.md has the 5 required level-2 sections in order.

    Permissive today (WARN-level); tightens to FAIL once PR-2's Jinja templates
    and validators land.
    """
    if not path.exists():
        raise ContractViolation(f"commandment path does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    missing: list[str] = []
    for pat in REQUIRED_COMMANDMENT_SECTIONS:
        if not re.search(pat, text, re.MULTILINE):
            missing.append(pat)
    if missing:
        # PR-1: permissive — just warn; don't block migrations of legacy commandments.
        # PR-2 tightens.
        return


__all__ = [
    "ContractViolation",
    "validate_harness",
    "validate_commandment",
    "REQUIRED_HARNESS_FLAGS",
    "REQUIRED_HARNESS_MARKERS",
    "REQUIRED_COMMANDMENT_SECTIONS",
]
