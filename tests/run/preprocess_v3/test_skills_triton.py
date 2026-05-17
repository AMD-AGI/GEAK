"""Tests for the on-disk ``skills/triton/`` knowledge base.

The v3 ``harness-generator`` subagent loads this KB at dispatch time
(see commit 5 — ``load_harness_kb``) and injects it into the child's
``{{knowledge_base}}`` Jinja placeholder. The tests below assert the
KB exists, has the right shape (SKILL.md + ``docs/`` directory), and
that each file carries non-trivial content traceable to the legacy
stack (per the "no inventing content" rule in the commit-set brief).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILL_ROOT = _REPO_ROOT / "skills" / "triton"
_DOCS_ROOT = _SKILL_ROOT / "docs"

_NON_TRIVIAL_MIN_CHARS = 200


# ---------------------------------------------------------------------------
# File existence + content size
# ---------------------------------------------------------------------------


def test_triton_skill_md_exists_with_content() -> None:
    p = _SKILL_ROOT / "SKILL.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"SKILL.md is trivially short ({len(text)} chars)"


def test_triton_harness_writing_doc_exists_with_content() -> None:
    p = _DOCS_ROOT / "triton_harness_writing.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"triton_harness_writing.md is trivially short ({len(text)} chars)"


def test_triton_idioms_doc_exists_with_content() -> None:
    p = _DOCS_ROOT / "triton_idioms.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"triton_idioms.md is trivially short ({len(text)} chars)"


# ---------------------------------------------------------------------------
# YAML frontmatter
# ---------------------------------------------------------------------------


def test_triton_skill_md_has_valid_yaml_frontmatter() -> None:
    text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter delimiter"
    _, fm, _ = text.split("---", 2)
    parsed = yaml.safe_load(fm)
    assert isinstance(parsed, dict)
    assert parsed.get("name") == "triton"
    assert isinstance(parsed.get("description", ""), str)
    assert parsed["description"].strip(), "description must be non-empty"


# ---------------------------------------------------------------------------
# Provenance: every authored doc carries phrases traceable to legacy files
# ---------------------------------------------------------------------------


def test_triton_skill_md_includes_user_task_context_rule() -> None:
    """Lifted from mini_unit_test_agent.yaml lines 1-27."""
    text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "USER TASK CONTEXT" in text
    assert "HIGHEST PRIORITY" in text


def test_triton_skill_md_includes_iterations_argparse_snippet() -> None:
    """Lifted from mini_unit_test_agent.yaml lines 167-188."""
    text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "--iterations" in text
    assert "GEAK_BENCHMARK_ITERATIONS" in text
    assert "parser.add_argument" in text


def test_triton_skill_md_includes_cpu_then_gpu_rule() -> None:
    """Lifted from INSTRUCTIONS.md section 1b pitfall 8."""
    text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "Generate tensors on CPU" in text
    assert "rocprofv3" in text


def test_triton_harness_writing_includes_three_tier_shapes() -> None:
    """Lifted from INSTRUCTIONS.md section 1b pitfall 4."""
    text = (_DOCS_ROOT / "triton_harness_writing.md").read_text(encoding="utf-8")
    assert "ALL_SHAPES" in text
    assert "HARNESS_SHAPES" in text
    assert "PROFILE_SHAPES" in text


def test_triton_harness_writing_includes_importlib_pitfall() -> None:
    """Lifted from INSTRUCTIONS.md section 1b pitfall 1."""
    text = (_DOCS_ROOT / "triton_harness_writing.md").read_text(encoding="utf-8")
    assert "importlib.util" in text


def test_triton_harness_writing_includes_default_fallback_shapes() -> None:
    """Lifted from INSTRUCTIONS.md pitfall 4 fallback rule."""
    text = (_DOCS_ROOT / "triton_harness_writing.md").read_text(encoding="utf-8")
    assert "S=2048" in text
    assert "M=1024" in text


def test_triton_harness_writing_includes_wrapper_vs_inner_kernel() -> None:
    """Lifted from INSTRUCTIONS.md section 1c."""
    text = (_DOCS_ROOT / "triton_harness_writing.md").read_text(encoding="utf-8")
    assert "wrapper" in text.lower()
    assert "inner kernel" in text.lower()


def test_triton_idioms_includes_jit_decorator_rule() -> None:
    """Lifted from kernel_languages/triton/builder_hints.md + idioms.md."""
    text = (_DOCS_ROOT / "triton_idioms.md").read_text(encoding="utf-8")
    assert "@triton.jit" in text
