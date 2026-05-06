"""Regression tests for ``run/dispatch.py::_read_commandment_section``.

Protects against the production parity failure observed in the
Triton parity run (April 2026): the refactor's commandment.j2 uses
title-case multi-word section names like ``## Full Benchmark`` while
legacy callers of ``build_eval_script`` (post-round evaluation) query
``FULL_BENCHMARK``.  The old parser regex ``^##\\s+(\\w+)`` captured
only the first word ("Full"), so the ``FULL_BENCHMARK`` query never
matched and downstream code reported ``"No FULL_BENCHMARK commands
found in COMMANDMENT"`` even though the section was present.

The fix normalizes both section query and header-line capture to
upper-snake-case, making matching case- and whitespace-insensitive.
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.run.dispatch import _normalize_section_name, _read_commandment_section


class TestNormalizeSectionName:
    def test_title_case_space_normalizes_to_upper_underscore(self) -> None:
        assert _normalize_section_name("Full Benchmark") == "FULL_BENCHMARK"

    def test_upper_underscore_is_idempotent(self) -> None:
        assert _normalize_section_name("FULL_BENCHMARK") == "FULL_BENCHMARK"

    def test_mixed_case_normalizes(self) -> None:
        assert _normalize_section_name("full benchmark") == "FULL_BENCHMARK"

    def test_multiple_spaces_collapse(self) -> None:
        assert _normalize_section_name("Full    Benchmark") == "FULL_BENCHMARK"

    def test_leading_trailing_whitespace_stripped(self) -> None:
        assert _normalize_section_name("  Setup  ") == "SETUP"


class TestReadCommandmentSection:
    def test_title_case_matches_uppercase_query(self, tmp_path: Path) -> None:
        """The regression: refactor's commandment uses ``## Full Benchmark``,
        but post-round evaluation queries ``FULL_BENCHMARK``.  Before the
        fix the parser silently returned None and the evaluator logged
        ``"No FULL_BENCHMARK commands found in COMMANDMENT"``."""
        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text(
            "## Setup\n"
            "export FOO=1\n"
            "\n"
            "## Full Benchmark\n"
            "```bash\n"
            "python harness.py --full-benchmark\n"
            "```\n"
            "\n"
            "## Profile\n"
            "python harness.py --profile\n"
        )
        body = _read_commandment_section(str(cm), "FULL_BENCHMARK")
        assert body is not None
        assert "python harness.py --full-benchmark" in body

    def test_uppercase_header_matches_uppercase_query(self, tmp_path: Path) -> None:
        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text(
            "## SETUP\n"
            "export BAR=2\n"
            "\n"
            "## FULL_BENCHMARK\n"
            "python harness.py --full-benchmark\n"
        )
        body = _read_commandment_section(str(cm), "FULL_BENCHMARK")
        assert body is not None
        assert "python harness.py --full-benchmark" in body

    def test_uppercase_header_matches_title_case_query(self, tmp_path: Path) -> None:
        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text("## FULL_BENCHMARK\npython harness.py --full-benchmark\n")
        body = _read_commandment_section(str(cm), "Full Benchmark")
        assert body is not None
        assert "python harness.py --full-benchmark" in body

    def test_returns_none_when_section_missing(self, tmp_path: Path) -> None:
        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text("## Setup\nexport FOO=1\n")
        assert _read_commandment_section(str(cm), "FULL_BENCHMARK") is None

    def test_strips_fence_markers(self, tmp_path: Path) -> None:
        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text(
            "## Benchmark\n"
            "```bash\n"
            "python harness.py --benchmark\n"
            "```\n"
        )
        body = _read_commandment_section(str(cm), "BENCHMARK")
        assert body == "python harness.py --benchmark"
        assert "```" not in body

    def test_sections_with_hyphens(self, tmp_path: Path) -> None:
        """Multi-word sections with hyphens (e.g. ``## Full-Benchmark``)
        should be accepted as well — the normalizer keeps hyphens as
        word-internal characters."""
        cm = tmp_path / "COMMANDMENT.md"
        cm.write_text("## Full-Benchmark\npython harness.py --full-benchmark\n")
        # ``FULL-BENCHMARK`` normalises differently from ``FULL_BENCHMARK``
        # (hyphen vs underscore), so callers MUST use matching form.
        # That matching form:
        body = _read_commandment_section(str(cm), "Full-Benchmark")
        assert body is not None
