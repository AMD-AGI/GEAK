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


# ---------------------------------------------------------------------------
# commit-set-6.5 enrichment: hip_test_structure.md (AKA-derived tactical patterns)
#
# Authored from AKA's tasks/hip2hip/ patterns: pybind11 + forward entry,
# separate dirs for ref/opt module loading, paired cuda.Event timing,
# hipEvent timing for standalone binaries, PyTorch module + functional
# + fn= injection, get_inputs / get_init_inputs convention, golden .pt
# save/load, C++ standalone argv parsing + stdout markers, MMCV
# autograd.Function wrapper, and an anti-patterns block. The lifts are
# strictly TACTICAL — GEAK's harness contract (4 CLI modes, geomean
# speedup, WARMUP=50, 3-tier shape lists) still wins on conflict.
# ---------------------------------------------------------------------------


def test_hip_test_structure_doc_exists_with_content() -> None:
    """The new ``hip_test_structure.md`` is on disk and non-trivially sized."""
    p = _DOCS_ROOT / "hip_test_structure.md"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > _NON_TRIVIAL_MIN_CHARS, f"hip_test_structure.md is trivially short ({len(text)} chars)"


def test_hip_test_structure_doc_has_contract_precedence_note() -> None:
    """The top-of-file GEAK contract precedence note must be present.

    Without this note a future maintainer might read these tactical
    patterns as a replacement for GEAK's harness contract (4 CLI modes,
    geomean speedup, etc.). The note pins the precedence explicitly,
    above the first horizontal rule (``---``) that separates the
    preamble from the first rule heading.
    """
    text = (_DOCS_ROOT / "hip_test_structure.md").read_text(encoding="utf-8")
    preamble = text.split("\n---\n", 1)[0]
    assert "GEAK contract precedence" in preamble, "missing precedence-note label"
    # Markdown blockquote line wraps mean the literal phrase may be
    # split across a ``> `` continuation; normalise blockquote linebreaks
    # before checking so prose reflow does not silently break this test.
    flattened = " ".join(line.lstrip("> ").rstrip() for line in preamble.splitlines())
    assert "GEAK's contract WINS" in flattened, "missing 'GEAK's contract WINS' phrase in the precedence preamble"
    assert "TACTICAL" in flattened, "missing TACTICAL qualifier in the precedence note"


def test_hip_test_structure_doc_has_all_ten_rule_headings() -> None:
    """All ten lifted rule headings are present, with AKA's numbering preserved.

    The rule numbers (5, 6, 7, 8, 12, 13, 14, 15, 16, 17) match the
    investigator's brief so anyone cross-referencing the brief lands
    on the right section.
    """
    text = (_DOCS_ROOT / "hip_test_structure.md").read_text(encoding="utf-8")
    expected_headings = (
        "## Rule 5: pybind11 + `forward` entry pattern",
        "## Rule 6: separate build directories when loading two pybind extensions",
        "## Rule 7: paired `cuda.Event` timing protocol",
        "## Rule 8: HIP standalone-binary timing with `hipEventRecord`",
        "## Rule 12: PyTorch module + functional + `fn=` injection pattern",
        "## Rule 13: `get_inputs()` / `get_init_inputs()` convention",
        "## Rule 14: golden `.pt` save / load convention",
        "## Rule 15: C++ / HIP standalone argv parsing + stdout markers",
        "## Rule 16: MMCV `autograd.Function` wrapper",
        "## Rule 17: anti-patterns",
    )
    for heading in expected_headings:
        assert heading in text, f"missing heading {heading!r}"


def test_hip_test_structure_doc_carries_aka_citation_markers() -> None:
    """Every cited AKA file:line range appears as a provenance comment in the doc.

    The investigator's brief pins these citations as the provenance for
    each lifted pattern. Asserting them here makes the trace traversable:
    grep ``AKA:`` in this file gives the maintainer the original source
    location for the standard idiom each snippet illustrates.
    """
    text = (_DOCS_ROOT / "hip_test_structure.md").read_text(encoding="utf-8")
    expected_citations = (
        # Rule 5 — pybind11 + forward (.hip macro side + Python load side)
        "tasks/hip2hip/gpumode/SimpleMatmulModule/hip/hip_3267_SimpleMatmulModule.hip:139-141",
        "tasks/hip2hip/gpumode/SimpleMatmulModule/eval_tools/utils.py:58-66",
        # Rule 6 — separate dirs / module names
        "tasks/hip2hip/gpumode/SimpleMatmulModule/eval_tools/cal_kernel_perf.py:199-214",
        # Rule 7 — paired cuda.Event
        "tasks/hip2hip/gpumode/SimpleMatmulModule/eval_tools/cal_kernel_perf.py:144-160",
        # Rule 8 — hipEventRecord on standalone HIP
        "tasks/hip2hip/others/silu/silu.hip:71-79",
        # Rule 12 — module + functional + fn=
        "tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_module/py_3267_SimpleMatmulModule.py:8-15",
        "tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_functional/py_3267_SimpleMatmulModule_func.py:20-25",
        # Rule 13 — get_inputs / get_init_inputs
        "tasks/hip2hip/gpumode/SimpleMatmulModule/pytorch_code_module/py_3267_SimpleMatmulModule.py:17-52",
        # Rule 14 — golden .pt save/load
        "tasks/hip2hip/others/ball_query/test_ball_query.py:49-50, 80-87",
        # Rule 15 — argv + stdout markers
        "tasks/hip2hip/others/silu/silu.hip:82-91, 117, 124",
        # Rule 16 — MMCV autograd.Function wrapper
        "tasks/hip2hip/others/ball_query/ball_query_wrapper.py:8-46",
    )
    for citation in expected_citations:
        assert citation in text, f"missing AKA citation {citation!r}"


def test_hip_test_structure_doc_lists_three_anti_patterns() -> None:
    """Rule 17 lists each of the three canonical HIP-harness anti-patterns.

    The three are: wall-clock timing, GPU-side test-tensor allocation
    inside the timed region, and treating ``cpp_extension.load`` success
    as kernel-symbol availability.
    """
    text = (_DOCS_ROOT / "hip_test_structure.md").read_text(encoding="utf-8")
    rule17_start = text.index("## Rule 17: anti-patterns")
    rule17_body = text[rule17_start:]
    assert "time.time()" in rule17_body
    assert "device='cuda'" in rule17_body or 'device="cuda"' in rule17_body
    assert "compilation success" in rule17_body
    assert "module loadability" in rule17_body
