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


# ---------------------------------------------------------------------------
# commit-set-6.5 enrichment: triton_test_structure.md (AKA-generalising patterns)
#
# Authored from the subset of AKA patterns that generalise across Triton
# and HIP. HIP-specific patterns (pybind11 + forward, separate dirs for
# ref/opt, hipEventRecord on standalone binaries, MMCV autograd.Function
# wrapper) intentionally do NOT appear here — they live in
# skills/hip/docs/hip_test_structure.md. The five lifts here keep AKA's
# rule numbering so the same number points at the same pattern in both
# language docs.
# ---------------------------------------------------------------------------


def test_triton_test_structure_doc_exists_with_content() -> None:
    """The new ``triton_test_structure.md`` is on disk and non-trivially sized."""
    p = _DOCS_ROOT / "triton_test_structure.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"triton_test_structure.md is trivially short ({len(text)} chars)"


def test_triton_test_structure_doc_has_contract_precedence_note() -> None:
    """The top-of-file GEAK contract precedence note must be present.

    Same shape as ``hip_test_structure.md`` — without this note a future
    maintainer might read these tactical patterns as a replacement for
    GEAK's harness contract (4 CLI modes, geomean speedup, etc.). The
    note pins the precedence explicitly, above the first horizontal
    rule (``---``) that separates the preamble from the first rule
    heading.
    """
    text = (_DOCS_ROOT / "triton_test_structure.md").read_text(encoding="utf-8")
    preamble = text.split("\n---\n", 1)[0]
    assert "GEAK contract precedence" in preamble, "missing precedence-note label"
    # Markdown blockquote line wraps mean the literal phrase may be
    # split across a ``> `` continuation; normalise blockquote linebreaks
    # before checking so prose reflow does not silently break this test.
    flattened = " ".join(line.lstrip("> ").rstrip() for line in preamble.splitlines())
    assert "GEAK's contract WINS" in flattened, "missing 'GEAK's contract WINS' phrase in the precedence preamble"
    assert "TACTICAL" in flattened, "missing TACTICAL qualifier in the precedence note"


def test_triton_test_structure_doc_points_to_hip_companion() -> None:
    """The preamble must link out to the HIP-specific companion doc.

    HIP-only patterns (pybind11, separate dirs, hipEventRecord, MMCV
    autograd.Function) live in ``skills/hip/docs/hip_test_structure.md``
    — surfacing the cross-link in the preamble keeps the maintainer
    from re-inventing the HIP rules here when they only generalise to
    Triton on paper.
    """
    text = (_DOCS_ROOT / "triton_test_structure.md").read_text(encoding="utf-8")
    preamble = text.split("\n---\n", 1)[0]
    assert "hip_test_structure.md" in preamble, "missing cross-link to HIP companion doc"


def test_triton_test_structure_doc_has_all_five_rule_headings() -> None:
    """The five lifted-into-Triton rule headings are present, AKA numbering preserved.

    The numbers (7, 12, 13, 14, 17) match the HIP doc so the same rule
    number points at the same pattern across both language docs. Rules
    5, 6, 8, 15, 16 are HIP-specific and intentionally absent here.
    """
    text = (_DOCS_ROOT / "triton_test_structure.md").read_text(encoding="utf-8")
    expected_headings = (
        "## Rule 7: paired `cuda.Event` timing protocol",
        "## Rule 12: PyTorch module + functional + `fn=` injection pattern",
        "## Rule 13: `get_inputs()` / `get_init_inputs()` convention",
        "## Rule 14: golden `.pt` save / load convention",
        "## Rule 17: anti-patterns",
    )
    for heading in expected_headings:
        assert heading in text, f"missing heading {heading!r}"


def test_triton_test_structure_doc_omits_hip_only_rules() -> None:
    """HIP-specific rule numbers (5, 6, 8, 15, 16) must NOT appear as headings.

    Pins the "no HIP-specific patterns here" decision so a future edit
    that lifts a HIP rule into this doc breaks loudly.
    """
    text = (_DOCS_ROOT / "triton_test_structure.md").read_text(encoding="utf-8")
    for hip_only_rule_number in (5, 6, 8, 15, 16):
        forbidden = f"## Rule {hip_only_rule_number}:"
        assert forbidden not in text, f"HIP-only rule heading leaked into Triton doc: {forbidden!r}"


def test_triton_test_structure_doc_carries_aka_citation_markers() -> None:
    """Every cited AKA file:line range appears as a provenance comment in the doc.

    Reuses the hip2hip citations because the patterns generalise across
    backends — the investigator's brief observed the same shapes in
    triton2triton, so the hip2hip file is the source-of-truth for the
    canonical idiom even when the consumer is Triton.
    """
    text = (_DOCS_ROOT / "triton_test_structure.md").read_text(encoding="utf-8")
    expected_citations = (
        # Rule 7 — paired cuda.Event timing
        "tasks/hip2hip/gpumode/SimpleMatmulModule/eval_tools/cal_kernel_perf.py:144-160",
        # Rule 12 — module + functional + fn= injection
        "tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_module/py_3267_SimpleMatmulModule.py:8-15",
        "tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_functional/py_3267_SimpleMatmulModule_func.py:20-25",
        # Rule 13 — get_inputs / get_init_inputs
        "tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_module/py_3267_SimpleMatmulModule.py:17-52",
        # Rule 14 — golden .pt save/load
        "tasks/hip2hip/others/ball_query/test_ball_query.py:49-50, 80-87",
    )
    for citation in expected_citations:
        assert citation in text, f"missing AKA citation {citation!r}"


def test_triton_test_structure_doc_lists_three_anti_patterns() -> None:
    """Rule 17 lists each of the three canonical anti-patterns.

    Same three as the HIP doc — wall-clock timing, GPU-side
    test-tensor allocation inside the timed region, and treating
    compile success as kernel-callable. The Triton-side phrasing for
    the third bullet calls out ``@triton.jit`` specifically.
    """
    text = (_DOCS_ROOT / "triton_test_structure.md").read_text(encoding="utf-8")
    rule17_start = text.index("## Rule 17: anti-patterns")
    rule17_body = text[rule17_start:]
    assert "time.time()" in rule17_body
    assert "device='cuda'" in rule17_body or 'device="cuda"' in rule17_body
    assert "@triton.jit" in rule17_body
    assert "compilation success" in rule17_body
    assert "module loadability" in rule17_body
