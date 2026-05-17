"""Per-language harness knowledge-base loader.

Loads the on-disk ``skills/<language>/SKILL.md`` + ``skills/<language>/docs/*.md``
files and concatenates them into a single markdown blob the v3
``harness-generator`` subagent receives via its ``{{knowledge_base}}``
Jinja placeholder.

Design mirror of ``minisweagent.tools.translation_registry.load_translation_kb``:

* YAML frontmatter is stripped from each file (the ``---`` block at the
  top of ``SKILL.md`` carries metadata for the Cursor / Anthropic skill
  runtime; the model doesn't need it inside its system prompt).
* Files are concatenated with ``"\\n\\n---\\n\\n"`` as the section
  separator so each section is visually distinct in the rendered prompt.
* ``docs/*.md`` files are sorted alphabetically before concatenation so
  the loader is fully deterministic — two calls on the same input
  produce byte-identical output, which the test suite asserts.
* An unknown language (no ``skills/<name>/`` folder, or a folder with no
  authored content) returns the empty string. This is graceful
  degradation — the dispatcher injects ``{{knowledge_base}} = ""`` and
  the subagent still runs (just without language-specific guidance).
"""

from __future__ import annotations

import logging
from pathlib import Path

from minisweagent.kernel_languages.base import KernelLanguage

logger = logging.getLogger(__name__)


_SECTION_SEPARATOR = "\n\n---\n\n"
"""Markdown horizontal-rule separator inserted between concatenated sections."""


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter (``---`` delimited) from markdown content.

    Mirrors ``minisweagent.tools.translation_registry._strip_frontmatter``
    so the two loaders stay behaviour-compatible.
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return content


def _default_skills_root() -> Path:
    """Resolve ``<repo>/skills`` by walking up from this file.

    Same shape as ``SubagentRegistry._default_root``: prefer a parent
    that has both ``pyproject.toml`` and a ``skills/`` directory, fall
    back to four-levels-up.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "skills").is_dir():
            return candidate / "skills"
    return here.parents[4] / "skills"


def load_harness_kb(
    language: KernelLanguage,
    *,
    skills_root: Path | None = None,
) -> str:
    """Load and concatenate the harness KB for the given kernel language.

    Returns the concatenated markdown content of
    ``skills/<language.name>/SKILL.md`` plus all files under
    ``skills/<language.name>/docs/*.md``, with YAML frontmatter stripped
    from each file and separated by ``"\\n\\n---\\n\\n"``.

    Returns an empty string if the language has no KB folder (graceful
    degradation — the dispatcher still injects the empty string into the
    child's ``{{knowledge_base}}`` placeholder).

    Args:
        language:
            The active :class:`KernelLanguage`. Only the ``name``
            attribute is consulted; the loader does not inspect any of
            the other ``KernelLanguage`` fields.
        skills_root:
            Optional override of the ``<repo>/skills`` directory. Tests
            inject a ``tmp_path``-based root here; production callers
            use the default (resolved via :func:`_default_skills_root`).
    """
    root = skills_root if skills_root is not None else _default_skills_root()
    lang_dir = root / language.name
    if not lang_dir.is_dir():
        logger.debug("load_harness_kb: no skills folder for language %r at %s", language.name, lang_dir)
        return ""

    sections: list[str] = []

    skill_md = lang_dir / "SKILL.md"
    if skill_md.is_file():
        sections.append(_strip_frontmatter(skill_md.read_text(encoding="utf-8")))

    docs_dir = lang_dir / "docs"
    if docs_dir.is_dir():
        for doc_path in sorted(docs_dir.glob("*.md")):
            sections.append(_strip_frontmatter(doc_path.read_text(encoding="utf-8")))

    if not sections:
        return ""

    return _SECTION_SEPARATOR.join(sections)


__all__ = [
    "load_harness_kb",
]
