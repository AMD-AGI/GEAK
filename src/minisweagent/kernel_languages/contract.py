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

# Hard cap on numeric tolerances in correctness checks.
#
# Why 2e-2?  bfloat16 has ~3 decimal digits of mantissa precision, so
# operations like ``silu(x) * y`` on ``randn`` inputs produce typical
# absolute error around 1e-2.  Setting the cap at 2e-2 leaves a comfort
# factor for legitimate fp16/bf16 kernels while rejecting the
# ``hip_act_and_mul_20260528`` failure mode where the LLM wrote
# ``assert_close(..., atol=5e-2, rtol=5e-2)`` — that 5% relative slop
# silently classifies broken kernels as correct.
#
# Override via ``GEAK_HARNESS_MAX_TOLERANCE`` for kernels whose numerics
# genuinely require larger atol (fp8 quantization, accumulation-heavy
# reductions, etc.).
import os as _os

_TOLERANCE_HARD_CAP = float(_os.environ.get("GEAK_HARNESS_MAX_TOLERANCE", "2e-2"))

# Match ``atol=<float>`` / ``rtol=<float>`` in kwargs, capturing the
# numeric literal.  Supports decimal and scientific notation.  Will
# overmatch comments mentioning ``atol=`` but the values inside comments
# are still part of the harness source, so a false positive there
# would mean the LLM put a misleading tolerance in a comment — which
# we'd want to know about anyway.
_TOLERANCE_LITERAL_RE = re.compile(
    r"\b(atol|rtol)\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)"
)


def _scan_tolerances(text: str) -> list[tuple[str, float, int]]:
    """Return ``(kind, value, lineno)`` for every atol=/rtol= literal in ``text``.

    Lineno is 1-based so error messages can point the LLM at the
    exact offending line in its harness retry.
    """
    findings: list[tuple[str, float, int]] = []
    for m in _TOLERANCE_LITERAL_RE.finditer(text):
        kind = m.group(1)
        raw = m.group(2)
        try:
            value = float(raw)
        except ValueError:
            continue
        lineno = text.count("\n", 0, m.start()) + 1
        findings.append((kind, value, lineno))
    return findings


# Match ``sys.path.insert(<int>, "<absolute-path-literal>")`` where the
# inserted path is a hardcoded POSIX-absolute string literal (starts with
# ``/``).  Dynamic inserts such as ``sys.path.insert(0, os.environ[...])``
# or ``sys.path.insert(0, os.path.dirname(__file__))`` do NOT match because
# their second argument is not a quoted literal beginning with ``/``.
#
# Why this is a hard contract violation: a literal absolute path at the
# front of ``sys.path`` (index 0) shadows BOTH the GEAK worktree that the
# COMMANDMENT SETUP prepends to ``PYTHONPATH`` AND the editable install
# GEAK re-points at the worktree.  The harness then imports the BASELINE
# package regardless of the agent's edits, so every optimization round
# measures baseline-vs-baseline (~1.00x) — a silent false-negative.
# Observed in rotary_embedding_kernel_202605290819 where
# ``sys.path.insert(0, "/sgl-workspace/sglang/python")`` pinned every run
# to the baseline sglang checkout.
_HARDCODED_SYSPATH_RE = re.compile(
    r"""sys\.path\.insert\(\s*\d+\s*,\s*(['"])(?P<path>/[^'"]*)\1\s*\)"""
)


def find_hardcoded_syspath_inserts(text: str) -> list[tuple[int, str]]:
    """Return ``(lineno, path)`` for every hardcoded-absolute ``sys.path.insert``.

    ``lineno`` is 1-based.  Used by both harness validators to reject
    harnesses that pin imports to a fixed (typically baseline) location
    and thereby bypass the GEAK worktree.
    """
    findings: list[tuple[int, str]] = []
    for m in _HARDCODED_SYSPATH_RE.finditer(text):
        lineno = text.count("\n", 0, m.start()) + 1
        findings.append((lineno, m.group("path")))
    return findings


def _hardcoded_syspath_violation_message(path: Path, offenders: list[tuple[int, str]]) -> str:
    """Build the shared, actionable violation message for hardcoded inserts."""
    listed = ", ".join(f"'{p}' on line {lineno}" for lineno, p in offenders)
    return (
        f"harness {path} hardcodes absolute path(s) via sys.path.insert: {listed}.\n"
        "A literal absolute path at the front of sys.path shadows BOTH the GEAK "
        "worktree (prepended to PYTHONPATH by the COMMANDMENT SETUP section) AND "
        "the editable install GEAK re-points at the worktree. The harness then "
        "imports the BASELINE package no matter what the agent edits, so every "
        "speedup measures baseline-vs-baseline (~1.00x) — a silent false-negative "
        "(observed in rotary_embedding_kernel_202605290819).\n"
        "Fix: REMOVE the hardcoded sys.path.insert and import the kernel via the "
        "package path, relying on the PYTHONPATH set by the COMMANDMENT SETUP "
        "section and the editable install GEAK manages for the worktree. If you "
        "genuinely must adjust sys.path at runtime, derive it from "
        "os.environ['GEAK_WORK_DIR'] (NOT a literal path)."
    )


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

    # Tolerance gate: catch the ``atol=5e-2`` slop that lets broken
    # kernels pass correctness checks.  We scan EVERY tolerance literal
    # in the harness (not just the first), because LLMs sometimes set a
    # tight default at the top of the file and override it with a wide
    # one inside a specific shape loop.  Reports every offender on a
    # single violation so the retry prompt can fix them all at once.
    too_loose: list[tuple[str, float, int]] = [
        (kind, value, lineno)
        for kind, value, lineno in _scan_tolerances(text)
        if value > _TOLERANCE_HARD_CAP
    ]
    if too_loose:
        offenders = ", ".join(
            f"{kind}={value:g} on line {lineno}" for kind, value, lineno in too_loose
        )
        raise ContractViolation(
            f"harness {path} uses correctness tolerance(s) above the {_TOLERANCE_HARD_CAP:g} hard cap: "
            f"{offenders}.\n"
            "Tolerances this loose silently classify broken kernels as correct "
            "(observed regression: hip_act_and_mul_20260528_0919 used atol=5e-2 "
            "and let agent patches pass without actually matching the reference).\n"
            "Use atol/rtol = 1e-3 for fp16/bf16 and 1e-4 for fp32 by default; "
            "tighten further when the kernel is exact.  If your kernel's numerics "
            "genuinely require larger atol (fp8 quantization, accumulation-heavy "
            "reductions), override GEAK_HARNESS_MAX_TOLERANCE for this run only "
            "and document the reason in the harness."
        )

    # Worktree-bypass gate: reject hardcoded absolute paths in
    # sys.path.insert, which shadow the GEAK worktree and make every
    # speedup measure baseline-vs-baseline (~1.00x).
    hardcoded = find_hardcoded_syspath_inserts(text)
    if hardcoded:
        raise ContractViolation(_hardcoded_syspath_violation_message(path, hardcoded))


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
    "find_hardcoded_syspath_inserts",
    "REQUIRED_HARNESS_FLAGS",
    "REQUIRED_HARNESS_MARKERS",
    "REQUIRED_COMMANDMENT_SECTIONS",
]
