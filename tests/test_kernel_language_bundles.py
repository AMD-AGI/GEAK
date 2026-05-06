"""Tests pinning the per-language asset bundles (Workstream B).

Per plan §I and Workstream B: every KernelLanguage instance must have
all nine path fields populated with actual files, so subagent
implementations (HarnessBuilder, KernelAnalysisAgent, etc.) can rely
on the paths resolving to non-empty content.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.kernel_languages import registry


_EXPECTED_LANGUAGES = ("triton", "hip")
_PATH_FIELDS = (
    "system_prompt_path",
    "orchestrator_system_prompt_path",
    "optimizer_hints_path",
    "builder_hints_path",
    "memory_hints_path",
    "idioms_path",
    "harness_template_path",
    "commandment_template_path",
)
_CONTENT_PROPERTIES = (
    "system_prompt",
    "orchestrator_system_prompt",
    "optimizer_hints",
    "builder_hints",
    "memory_hints",
    "idioms",
    "harness_template",
    "commandment_template",
)


@pytest.fixture(params=_EXPECTED_LANGUAGES)
def language(request):
    lang = registry.get(request.param)
    assert lang is not None, f"Language {request.param} not registered"
    return lang


class TestPathFieldsPopulated:
    @pytest.mark.parametrize("field", _PATH_FIELDS)
    def test_path_field_is_set(self, language, field: str) -> None:
        path = getattr(language, field)
        assert path is not None, f"{language.name}.{field} must be a Path, got None"
        assert isinstance(path, Path)

    @pytest.mark.parametrize("field", _PATH_FIELDS)
    def test_path_exists_on_disk(self, language, field: str) -> None:
        path: Path = getattr(language, field)
        assert path.exists(), f"{language.name}.{field} points to non-existent file: {path}"
        assert path.is_file()

    @pytest.mark.parametrize("field", _PATH_FIELDS)
    def test_file_is_non_empty(self, language, field: str) -> None:
        path: Path = getattr(language, field)
        content = path.read_text(encoding="utf-8")
        assert content.strip(), f"{language.name}.{field} is empty: {path}"


class TestLazyLoadProperties:
    """Each path field has a ``@property`` that returns the file content."""

    @pytest.mark.parametrize("prop", _CONTENT_PROPERTIES)
    def test_property_loads_content(self, language, prop: str) -> None:
        content = getattr(language, prop)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_translation_hints_dir_is_set(self, language) -> None:
        assert language.translation_hints_dir is not None
        assert language.translation_hints_dir.is_dir()


class TestTranslationHints:
    """Cross-pair translation hints resolve via ``translation_hints_for``."""

    def test_triton_to_hip_pair_exists(self) -> None:
        triton = registry.get("triton")
        assert triton is not None
        hints = triton.translation_hints_for("hip")
        assert hints, "triton_to_hip.md must exist and be non-empty"

    def test_hip_to_triton_pair_exists(self) -> None:
        hip = registry.get("hip")
        assert hip is not None
        hints = hip.translation_hints_for("triton")
        assert hints, "hip_to_triton.md must exist and be non-empty"

    def test_unknown_pair_falls_back_to_fallback_md(self) -> None:
        """Requesting an unregistered target language should return the
        generic ``_fallback.md`` content, not an empty string."""
        triton = registry.get("triton")
        assert triton is not None
        hints = triton.translation_hints_for("some_future_language")
        assert hints, "Expected _fallback.md to provide generic guidance"
        assert "fallback" in hints.lower() or "generic" in hints.lower()


class TestContentInvariants:
    """Shape checks on the markdown / Jinja content.

    Catches regressions where a file got truncated, stripped of its
    Jinja placeholders, or had its required sections removed.
    """

    def test_harness_templates_declare_universal_flags(self, language) -> None:
        """Both harness.j2 templates must reference the four contract flags."""
        template = language.harness_template
        for flag in ("--correctness", "--benchmark", "--full-benchmark", "--profile"):
            assert flag in template, (
                f"{language.name}/harness.j2 missing required flag: {flag}"
            )

    def test_harness_templates_emit_geak_markers(self, language) -> None:
        template = language.harness_template
        assert "GEAK_RESULT_LATENCY_MS" in template
        assert "GEAK_RESULT_SPEEDUP" in template

    def test_commandment_templates_have_five_sections(self, language) -> None:
        template = language.commandment_template
        for heading in ("## Setup", "## Correctness", "## Benchmark", "## Full Benchmark", "## Profile"):
            assert heading in template, (
                f"{language.name}/commandment.j2 missing required section: {heading}"
            )

    def test_system_prompts_reference_rag_slot(self, language) -> None:
        """System prompts must contain the ``{rag_tools_description}``
        placeholder so the RAG section injection stays wired."""
        assert "{rag_tools_description}" in language.system_prompt
        assert "{rag_tools_description}" in language.orchestrator_system_prompt
