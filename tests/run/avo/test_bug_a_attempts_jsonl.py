"""Regression tests for Bug A — ``attempts.jsonl`` showing 100% false despite
real commits in ``lineage.json``.

Two root causes, both fixed by this change set:

1. **Correctness detection finds nothing** — ``_parse_correctness`` (newest, by
   boss) looks for ``Return code:`` / ``Test status:`` header lines but
   ``save_and_test`` was never writing them to ``patch_*_test.txt``; the
   legacy substring fallback also doesn't match the current HIP harness
   format ``[correctness] B=N, H=M: PASS``. Fixed by:
   - ``save_and_test`` now prepends the exit-code header to the on-disk log
     (Direction A — closes the gap in boss's 807b502e design)
   - the legacy fallback marker tuples gain four entries that recognise the
     new harness format (Direction B — belt-and-braces)

2. **Verify-then-record ordering** — ``controller.py`` called
   ``lineage.record_attempts(result)`` *before* ``_apply_verified_score(result)``;
   the authoritative ``AttemptRecord`` that ``_apply_verified_score`` appends
   to ``result.attempts`` never reached the jsonl. Fixed by moving
   ``record_attempts`` to after ``_apply_verified_score``.
"""

from __future__ import annotations

import json
from pathlib import Path

from minisweagent.run.avo.lineage_store import LineageStore
from minisweagent.run.avo.result import AttemptRecord, VariationResult
from minisweagent.run.avo.variation_step import _parse_correctness
from minisweagent.tools.save_and_test import _test_log_with_header


# ---------------------------------------------------------------------------
# Direction A + B — _parse_correctness covers the structured header AND the
# new HIP harness format via the legacy fallback.
# ---------------------------------------------------------------------------


def test_parse_correctness_uses_exit_code_header_when_present():
    """Authoritative path: a ``Return code:`` header wins regardless of body."""
    text = (
        "Test status: PASSED ✓\n"
        "Return code: 0\n"
        "\n"
        "[harness] kernel: /opt/GEAK/.../silu.hip\n"
        "[correctness] B=1, H=6400: FAIL\n"  # body says FAIL but exit code is 0
    )
    assert _parse_correctness(text) is True


def test_parse_correctness_exit_code_failure_wins_over_pass_body():
    text = (
        "Test status: FAILED ✗\n"
        "Return code: 1\n"
        "\n"
        "[harness] kernel: /opt/GEAK/.../silu.hip\n"
        "[correctness] B=1, H=6400: PASS\n"  # body says PASS but exit code is 1
    )
    assert _parse_correctness(text) is False


def test_parse_correctness_falls_back_on_new_harness_format_pass():
    """No structured header (older log / non-save_and_test producer); fallback
    must still recognise the current HIP harness format."""
    text = (
        "[harness] kernel: /opt/GEAK/.../silu.hip\n"
        "[correctness] B=1, H=6400: PASS\n"
        "  Check: max_abs=0  max_rel=0  -> PASS\n"
        "[correctness] B=32, H=6400: PASS\n"
        "  Check: max_abs=0  max_rel=0  -> PASS\n"
    )
    assert _parse_correctness(text) is True


def test_parse_correctness_falls_back_on_new_harness_format_fail():
    text = (
        "[harness] kernel: /opt/GEAK/.../silu.hip\n"
        "[correctness] B=1, H=6400: FAIL\n"
        "  Check: max_abs=99  max_rel=99  -> FAIL\n"
    )
    assert _parse_correctness(text) is False


def test_parse_correctness_strict_on_mixed_new_format():
    """9 PASS + 1 FAIL must be FAIL — boss's c67fb2a3 ``negative wins`` policy
    is now expressed for the new format via ``]: fail``."""
    lines = [f"[correctness] B={b}: PASS" for b in (1, 32, 64, 128, 256, 512, 1024, 2048, 4096)]
    lines.append("[correctness] B=8192: FAIL")
    assert _parse_correctness("\n".join(lines)) is False


def test_parse_correctness_legacy_old_format_still_passes():
    """Don't regress on the historical ``Correctness: PASS`` style."""
    assert _parse_correctness("Correctness: PASS\n") is True
    assert _parse_correctness("Correctness: FAIL\n") is False
    assert _parse_correctness("all tests passed\n") is True


# Real-world sample: the historical log that triggered Bug A. If we ever
# regenerate similar logs, this guards against silently re-breaking the
# fallback path.
def test_parse_correctness_on_real_historical_log():
    log = Path(
        "optimization_logs/avo_silu_20260605_045820/results/round_2/avo-worker/patch_4_test.txt"
    )
    if not log.exists():  # the log may not be present in every checkout
        return
    assert _parse_correctness(log.read_text(encoding="utf-8")) is True


# ---------------------------------------------------------------------------
# Direction A — save_and_test prepends the header to the on-disk log.
# ---------------------------------------------------------------------------


def test_test_log_with_header_pass():
    body = "[harness] kernel: /opt/GEAK/.../silu.hip\n[correctness] B=1: PASS\n"
    out = _test_log_with_header(body, test_passed=True, returncode=0)
    # Header is at the very top so the consumer's regex
    # ``re.compile(r"^Return code:", re.MULTILINE)`` matches without scanning.
    assert out.startswith("Test status: PASSED ✓\nReturn code: 0\n\n")
    # Body preserved verbatim afterwards — downstream parsers
    # (``parse_speedup_report``, ``benchmark_parsing``) must not break.
    assert body in out


def test_test_log_with_header_fail():
    body = "[harness] kernel: ...\nAssertionError: shapes differ\n"
    out = _test_log_with_header(body, test_passed=False, returncode=1)
    assert out.startswith("Test status: FAILED ✗\nReturn code: 1\n\n")
    assert body in out


def test_test_log_header_is_parsed_by_parse_correctness():
    """The two halves of Direction A meet here: the header that save_and_test
    writes must be the one that ``_parse_correctness`` reads."""
    body = "[correctness] B=1: FAIL\n"  # body would say FAIL on its own
    pass_log = _test_log_with_header(body, test_passed=True, returncode=0)
    fail_log = _test_log_with_header(body, test_passed=False, returncode=2)
    # Exit code is authoritative — body content is ignored when the header is present.
    assert _parse_correctness(pass_log) is True
    assert _parse_correctness(fail_log) is False


# ---------------------------------------------------------------------------
# Layer 2 — ``record_attempts`` must be called after the verify step, so that
# the authoritative AttemptRecord appended by ``_apply_verified_score`` is
# persisted to ``attempts.jsonl``.
# ---------------------------------------------------------------------------


def test_attempts_jsonl_records_post_verify_append(tmp_path: Path):
    """Models the fixed sequence:

      result.attempts := [collected wrong record]       (from _collect_result)
      _apply_verified_score(result)                     (appends authoritative)
      lineage.record_attempts(result)                   (NOW the write — fix)

    Before the fix, ``record_attempts`` ran before the verified-score append,
    so the jsonl was missing the authoritative record.
    """
    store = LineageStore(tmp_path / "avo_state")
    (tmp_path / "baseline_metrics.json").write_text(
        json.dumps({"latency_ms": 10.0}), encoding="utf-8"
    )
    store.seed_from_baseline(tmp_path)

    result = VariationResult(
        step_index=1,
        step_dir=tmp_path,
        strategy="test-strategy",
        # The collected (wrong) record that _collect_result writes when the
        # log doesn't match anything — pre-fix this was the only thing in the
        # jsonl, hence the 100% false report.
        attempts=[
            AttemptRecord(
                strategy="test-strategy",
                returncode=1,
                correctness_passed=False,
                verified_speedup=None,
                patch_hash="aaaa",
            )
        ],
    )

    # Simulate _apply_verified_score appending the authoritative record.
    result.attempts.append(
        AttemptRecord(
            strategy="test-strategy",
            returncode=0,
            correctness_passed=True,
            verified_speedup=1.5,
            patch_hash="aaaa",
        )
    )
    # Layer 2 fix: record AFTER the verify-time append.
    store.record_attempts(result)

    rows = [
        json.loads(line)
        for line in (tmp_path / "avo_state" / "attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == 2, f"expected both collected and verified attempts, got {rows}"
    assert rows[0]["correctness_passed"] is False
    assert rows[1]["correctness_passed"] is True
    assert rows[1]["verified_speedup"] == 1.5
