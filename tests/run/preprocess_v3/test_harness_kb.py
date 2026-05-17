"""Tests for ``minisweagent.run.preprocess_v3.harness_kb.load_harness_kb``.

The loader is the seam between the on-disk ``skills/<lang>/`` KB and the
v3 ``harness-generator`` subagent's ``{{knowledge_base}}`` Jinja
placeholder. The tests below cover:

* the real on-disk Triton + HIP KBs are loaded with concrete sentinel
  phrases preserved;
* YAML frontmatter is stripped (the model doesn't need the skill-runtime
  metadata);
* sections are concatenated with the documented separator
  (``"\\n\\n---\\n\\n"``);
* unknown languages return ``""`` (graceful degradation, never raises);
* the loader is deterministic — two calls on the same input produce
  byte-identical output.
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.harness_kb import load_harness_kb

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKILLS_ROOT = _REPO_ROOT / "skills"


def _mk_lang(name: str) -> KernelLanguage:
    """Build a minimal KernelLanguage for the loader (only ``name`` is read)."""
    return KernelLanguage(name=name, file_extensions=frozenset({".py"}), kb_namespace=name)


# ---------------------------------------------------------------------------
# Real on-disk Triton KB
# ---------------------------------------------------------------------------


def test_load_harness_kb_triton_includes_skill_md_phrases() -> None:
    """The Triton SKILL.md common-tips section makes it into the loaded blob."""
    text = load_harness_kb(_mk_lang("triton"))
    assert "USER TASK CONTEXT" in text
    assert "--iterations" in text
    assert "Generate tensors on CPU" in text


def test_load_harness_kb_triton_includes_harness_writing_doc() -> None:
    """docs/triton_harness_writing.md phrases are concatenated in."""
    text = load_harness_kb(_mk_lang("triton"))
    assert "ALL_SHAPES" in text
    assert "HARNESS_SHAPES" in text
    assert "PROFILE_SHAPES" in text
    assert "importlib.util" in text


def test_load_harness_kb_triton_includes_idioms_doc() -> None:
    """docs/triton_idioms.md phrases are concatenated in."""
    text = load_harness_kb(_mk_lang("triton"))
    assert "@triton.jit" in text


def test_load_harness_kb_triton_strips_frontmatter() -> None:
    """The YAML frontmatter (``---`` delimited block) is removed from each file."""
    text = load_harness_kb(_mk_lang("triton"))
    # The frontmatter's ``name: triton`` line is YAML metadata; it must
    # not leak into the prompt body. The first ``# Triton`` heading from
    # the body should be the start of the returned content.
    assert text.lstrip().startswith("# Triton — Harness-Generation KB"), text[:200]
    # But the section separator inserts ``---`` between concatenated
    # files, so a bare ``---`` followed by ``name:`` would still be
    # invalid — assert no ``name: triton`` line at the top of the blob.
    assert "\nname: triton\n" not in text[:500]


# ---------------------------------------------------------------------------
# Real on-disk HIP KB
# ---------------------------------------------------------------------------


def test_load_harness_kb_hip_includes_skill_md_phrases() -> None:
    text = load_harness_kb(_mk_lang("hip"))
    assert "USER TASK CONTEXT" in text
    assert "--iterations" in text
    assert "Generate tensors on CPU" in text


def test_load_harness_kb_hip_includes_per_kernel_type_bullets() -> None:
    """docs/hip_harness_writing.md phrases are concatenated in."""
    text = load_harness_kb(_mk_lang("hip"))
    assert "hipcc" in text
    assert "Composable Kernel" in text
    assert "HSACO" in text
    assert "sys.path.insert" in text


def test_load_harness_kb_hip_includes_build_modes_doc() -> None:
    text = load_harness_kb(_mk_lang("hip"))
    assert "--offload-arch=gfx942" in text


def test_load_harness_kb_hip_includes_idioms_doc() -> None:
    text = load_harness_kb(_mk_lang("hip"))
    assert "__global__" in text
    assert "hipLaunchKernelGGL" in text


# ---------------------------------------------------------------------------
# Section separator
# ---------------------------------------------------------------------------


def test_load_harness_kb_uses_horizontal_rule_separator() -> None:
    """Sections are separated by the documented ``\\n\\n---\\n\\n`` separator."""
    text = load_harness_kb(_mk_lang("triton"))
    # At least one horizontal-rule separator between the SKILL.md and
    # the first docs/*.md file.
    assert "\n\n---\n\n" in text


# ---------------------------------------------------------------------------
# Unknown language
# ---------------------------------------------------------------------------


def test_load_harness_kb_unknown_language_returns_empty_string() -> None:
    """A language with no on-disk KB returns ``""`` — never raises."""
    text = load_harness_kb(_mk_lang("definitely-not-a-real-language"))
    assert text == ""


def test_load_harness_kb_language_with_empty_folder_returns_empty(tmp_path: Path) -> None:
    """A folder that exists but has no SKILL.md / docs returns ``""``."""
    (tmp_path / "ghost").mkdir()
    text = load_harness_kb(_mk_lang("ghost"), skills_root=tmp_path)
    assert text == ""


# ---------------------------------------------------------------------------
# Determinism: identical output on repeated calls
# ---------------------------------------------------------------------------


def test_load_harness_kb_is_deterministic_for_triton() -> None:
    a = load_harness_kb(_mk_lang("triton"))
    b = load_harness_kb(_mk_lang("triton"))
    assert a == b
    assert len(a) > 1000, "expected a non-trivial KB; sanity-check the on-disk content"


def test_load_harness_kb_is_deterministic_for_hip() -> None:
    a = load_harness_kb(_mk_lang("hip"))
    b = load_harness_kb(_mk_lang("hip"))
    assert a == b
    assert len(a) > 1000


# ---------------------------------------------------------------------------
# tmp_path-based loader integration (no dependency on the real skills/)
# ---------------------------------------------------------------------------


def test_load_harness_kb_with_custom_skills_root(tmp_path: Path) -> None:
    """The loader honours an injected ``skills_root`` so tests stay hermetic."""
    lang_dir = tmp_path / "fake"
    docs_dir = lang_dir / "docs"
    docs_dir.mkdir(parents=True)

    (lang_dir / "SKILL.md").write_text(
        "---\nname: fake\ndescription: fixture\n---\n\nSKILL_BODY_MARKER",
        encoding="utf-8",
    )
    (docs_dir / "b_doc.md").write_text("DOC_B_MARKER", encoding="utf-8")
    (docs_dir / "a_doc.md").write_text("DOC_A_MARKER", encoding="utf-8")

    text = load_harness_kb(_mk_lang("fake"), skills_root=tmp_path)

    assert "SKILL_BODY_MARKER" in text
    assert "DOC_A_MARKER" in text
    assert "DOC_B_MARKER" in text
    # Frontmatter from the fixture SKILL.md is stripped.
    assert "name: fake" not in text
    # Sort order is alphabetic — DOC_A_MARKER appears before DOC_B_MARKER.
    assert text.index("DOC_A_MARKER") < text.index("DOC_B_MARKER")
    # Section separator is present between the SKILL body and the first doc.
    assert "\n\n---\n\n" in text


def test_load_harness_kb_real_skills_root_constant_points_to_repo() -> None:
    """Sanity check: the production ``skills/`` root has the two authored languages."""
    assert (_SKILLS_ROOT / "triton").is_dir()
    assert (_SKILLS_ROOT / "hip").is_dir()
