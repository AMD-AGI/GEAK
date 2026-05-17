"""Tests for the on-disk ``skills/hip/`` knowledge base.

Parallel to ``test_skills_triton.py`` — the v3 ``harness-generator``
subagent loads this KB at dispatch time and injects it into the
child's ``{{knowledge_base}}`` Jinja placeholder when the active
``KernelLanguage`` resolves to HIP (or any C++-compiled GPU language).

The tests below assert the KB exists, has the right shape (SKILL.md +
``docs/`` directory), and that each file carries non-trivial content
traceable to the legacy stack.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILL_ROOT = _REPO_ROOT / "skills" / "hip"
_DOCS_ROOT = _SKILL_ROOT / "docs"

_NON_TRIVIAL_MIN_CHARS = 200


# ---------------------------------------------------------------------------
# File existence + content size
# ---------------------------------------------------------------------------


def test_hip_skill_md_exists_with_content() -> None:
    p = _SKILL_ROOT / "SKILL.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"SKILL.md is trivially short ({len(text)} chars)"


def test_hip_harness_writing_doc_exists_with_content() -> None:
    p = _DOCS_ROOT / "hip_harness_writing.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"hip_harness_writing.md is trivially short ({len(text)} chars)"


def test_hip_idioms_doc_exists_with_content() -> None:
    p = _DOCS_ROOT / "hip_idioms.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"hip_idioms.md is trivially short ({len(text)} chars)"


def test_hip_build_modes_doc_exists_with_content() -> None:
    p = _DOCS_ROOT / "hip_build_modes.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"hip_build_modes.md is trivially short ({len(text)} chars)"


# ---------------------------------------------------------------------------
# YAML frontmatter
# ---------------------------------------------------------------------------


def test_hip_skill_md_has_valid_yaml_frontmatter() -> None:
    text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter delimiter"
    _, fm, _ = text.split("---", 2)
    parsed = yaml.safe_load(fm)
    assert isinstance(parsed, dict)
    assert parsed.get("name") == "hip"
    assert isinstance(parsed.get("description", ""), str)
    assert parsed["description"].strip(), "description must be non-empty"


# ---------------------------------------------------------------------------
# Provenance: every authored doc carries phrases traceable to legacy files
# ---------------------------------------------------------------------------


def test_hip_skill_md_includes_user_task_context_rule() -> None:
    """Lifted from mini_unit_test_agent.yaml lines 1-27."""
    text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "USER TASK CONTEXT" in text
    assert "HIGHEST PRIORITY" in text


def test_hip_skill_md_includes_iterations_argparse_snippet() -> None:
    """Lifted from mini_unit_test_agent.yaml lines 167-188."""
    text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "--iterations" in text
    assert "GEAK_BENCHMARK_ITERATIONS" in text
    assert "parser.add_argument" in text


def test_hip_skill_md_includes_cpu_then_gpu_rule() -> None:
    """Lifted from INSTRUCTIONS.md section 1b pitfall 8."""
    text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "Generate tensors on CPU" in text
    assert "rocprofv3" in text


def test_hip_common_tips_are_duplicated_verbatim_from_triton() -> None:
    """Per locked decision 1, common tips DUPLICATE between language KBs.

    The duplication is intentional — each KB stays self-contained when
    injected into a child subagent prompt. This test pins the contract
    so the next refactor that "DRYs them up" fails loudly until the
    locked decision is reconsidered.
    """
    triton_text = (_REPO_ROOT / "skills" / "triton" / "SKILL.md").read_text(encoding="utf-8")
    hip_text = (_SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    for canonical_phrase in (
        "USER TASK CONTEXT block is HIGHEST PRIORITY",
        "`--iterations` argparse rule (STRONGLY RECOMMENDED)",
        "Generate tensors on CPU, then move to GPU",
    ):
        assert canonical_phrase in triton_text, f"Triton missing {canonical_phrase!r}"
        assert canonical_phrase in hip_text, f"HIP missing {canonical_phrase!r}"


def test_hip_harness_writing_includes_per_kernel_type_bullets() -> None:
    """Lifted from unit_test_agent.py _LANGUAGE_GUIDANCE['hip','cuda','ck','asm']."""
    text = (_DOCS_ROOT / "hip_harness_writing.md").read_text(encoding="utf-8")
    assert "hipcc" in text
    assert "Composable Kernel" in text
    assert "HSACO" in text
    assert "hipEventElapsedTime" in text


def test_hip_harness_writing_includes_sys_path_insert_warning() -> None:
    """Lifted from unit_test_agent.py guidance for HIP and CK."""
    text = (_DOCS_ROOT / "hip_harness_writing.md").read_text(encoding="utf-8")
    assert "sys.path.insert" in text


def test_hip_harness_writing_includes_wrapper_script_rule() -> None:
    """Lifted from INSTRUCTIONS.md section 1c (wrapper-script-not-inline-env)."""
    text = (_DOCS_ROOT / "hip_harness_writing.md").read_text(encoding="utf-8")
    assert "wrapper script" in text.lower()
    assert "execvpe" in text or "execvpe" in (_DOCS_ROOT / "hip_build_modes.md").read_text(encoding="utf-8")


def test_hip_harness_writing_includes_hip_device_synchronize_rule() -> None:
    """Lifted from kernel_languages/hip/builder_hints.md timing loop section."""
    text = (_DOCS_ROOT / "hip_harness_writing.md").read_text(encoding="utf-8")
    assert "hipDeviceSynchronize" in text


def test_hip_idioms_includes_global_void_and_launch_macro() -> None:
    """Lifted from kernel_languages/hip/idioms.md."""
    text = (_DOCS_ROOT / "hip_idioms.md").read_text(encoding="utf-8")
    assert "__global__" in text
    assert "hipLaunchKernelGGL" in text


def test_hip_build_modes_includes_three_shapes_and_offload_arch() -> None:
    """Lifted from kernel_languages/hip/builder_hints.md (three shapes)."""
    text = (_DOCS_ROOT / "hip_build_modes.md").read_text(encoding="utf-8")
    assert "pybind11" in text
    assert "make" in text
    assert "hipcc" in text
    assert "--offload-arch=gfx942" in text
