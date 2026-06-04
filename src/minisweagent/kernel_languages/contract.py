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
                f"harness {path} missing required flags {missing_flags} AND required markers {missing_markers}"
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

#: A bare ``None`` token in a command position signals a templating leak: a
#: Jinja ``{{ cmd | default(...) }}`` where ``cmd`` was Python ``None`` (the
#: single-arg ``default`` filter does NOT fire on a defined-but-None value, so
#: ``None`` is rendered literally). Such a COMMANDMENT produces ``... && None``
#: and fails at runtime with rc=127 ("None: command not found"). Catch it at
#: generation time (fail-loud) instead of deep inside a preflight/eval run.
_NONE_COMMAND_TOKEN = re.compile(r"(?:^|[\s&|;])None(?:\s|$)")


def _commandment_command_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(1-based line number, line)`` for command lines inside ``bash`` fences."""
    out: list[tuple[int, str]] = []
    in_fence = False
    fence_re = re.compile(r"^```")
    for idx, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if fence_re.match(stripped):
            in_fence = not in_fence
            continue
        if in_fence and stripped and not stripped.startswith("#"):
            out.append((idx, stripped))
    return out


def validate_commandment(path: Path) -> None:
    """Verify a COMMANDMENT.md has the 5 required level-2 sections in order.

    Section presence is permissive today (WARN-level; tightens once PR-2's Jinja
    templates and validators land). A bare ``None`` command token, however, is
    always a hard failure: it is never legitimate and would otherwise surface as
    an opaque rc=127 ("None: command not found") at preflight/eval time.
    """
    if not path.exists():
        raise ContractViolation(f"commandment path does not exist: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Fail-loud on a templating leak (bare ``None`` in a command position).
    leaked = [(ln, line) for ln, line in _commandment_command_lines(text) if _NONE_COMMAND_TOKEN.search(line)]
    if leaked:
        details = "; ".join(f"line {ln}: {line!r}" for ln, line in leaked[:5])
        raise ContractViolation(
            "commandment contains a bare 'None' command token (templating leak that would fail at "
            f"runtime with rc=127 'None: command not found'): {details}"
        )

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
