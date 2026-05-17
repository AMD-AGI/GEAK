"""Integration tests for the on-disk v3 preprocess subagent YAMLs.

Unlike :mod:`test_registry`, which exercises the registry against synthetic
``tmp_path`` fixtures, this module loads the **real** YAMLs that ship under
``subagents/preprocess/<name>/SUBAGENT.yaml`` and asserts the contract that
commit set 3 ports/authors them under:

* every always-on subagent is discoverable via :class:`SubagentRegistry`,
* the ``system_prompt`` field is populated and (for verbatim ports) matches
  the legacy ``subagents/<name>/SYSTEM_PROMPT.md`` body byte-for-byte, and
* the ``description`` field is non-empty and stays under the 200-char cap.

Tests for the freshly-authored ``harness-verifier`` YAML additionally assert
that the ``# source: authored fresh`` comment marker on line 1 is preserved
in the raw file, since YAML comments are stripped at parse time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minisweagent.run.preprocess_v3.registry import SubagentRegistry, SubagentSpec

# Resolve the repo root once: this file lives at
# ``<repo>/tests/run/preprocess_v3/test_subagent_yamls.py`` so four
# ``parent`` hops land on the repo root, mirroring the registry's own
# default-root resolution rule.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_V3_ROOT = _REPO_ROOT / "subagents" / "preprocess"
_LEGACY_ROOT = _REPO_ROOT / "subagents"

_DESCRIPTION_MAX_CHARS = 200
_VERBATIM_PREFIX_CHARS = 80


@pytest.fixture(scope="module")
def registry_specs() -> dict[str, SubagentSpec]:
    """Discover the on-disk v3 subagents once per module."""
    return SubagentRegistry(root=_V3_ROOT).discover()


# ---------------------------------------------------------------------------
# harness-generator (verbatim port)
# ---------------------------------------------------------------------------


def test_harness_generator_is_registered(registry_specs: dict[str, SubagentSpec]) -> None:
    assert "harness-generator" in registry_specs
    assert "harness-generator" in SubagentRegistry(root=_V3_ROOT).names()


def test_harness_generator_prompt_role_prefix_matches_legacy(registry_specs: dict[str, SubagentSpec]) -> None:
    """The opening Role section is still a verbatim lift from the legacy SYSTEM_PROMPT.md.

    Commit set 6 *additively* enriches the prompt with a
    ``## Language-Specific Knowledge Base`` block between Role and Goal;
    the surrounding legacy text is preserved unchanged. This test pins
    the opening so anyone who paraphrases the Role definition (the
    canonical TestHarnessAgent contract) sees a loud failure.
    """
    legacy_prompt = (_LEGACY_ROOT / "harness-generator" / "SYSTEM_PROMPT.md").read_text(encoding="utf-8")

    spec = registry_specs["harness-generator"]
    assert spec.system_prompt, "system_prompt must be non-empty"
    assert spec.system_prompt[:_VERBATIM_PREFIX_CHARS] == legacy_prompt[:_VERBATIM_PREFIX_CHARS], (
        "system_prompt prefix drifted from legacy SYSTEM_PROMPT.md"
    )


def test_harness_generator_prompt_preserves_legacy_body(registry_specs: dict[str, SubagentSpec]) -> None:
    """Every non-trivial legacy paragraph still appears in the enriched prompt.

    The enrichment only INSERTS a KB block between Role and Goal; it does
    not remove anything. We assert a handful of canonical legacy phrases
    that span the breadth of the legacy file so the additive contract is
    pinned.
    """
    spec = registry_specs["harness-generator"]
    sp = spec.system_prompt
    for canonical_phrase in (
        "TestHarnessAgent",
        "MINI_SWE_AGENT_FINAL_OUTPUT",
        "TEST_COMMAND",
        "--correctness",
        "--profile",
        "--benchmark",
        "--full-benchmark",
        "harness_shapes_source.txt",
        "torch.manual_seed(42)",
        "GEAK_RESULT_LATENCY_MS",
        "GEAK_SHAPES_USED",
    ):
        assert canonical_phrase in sp, f"missing legacy phrase {canonical_phrase!r}"


def test_harness_generator_has_kb_placeholder(registry_specs: dict[str, SubagentSpec]) -> None:
    """The literal ``{{knowledge_base}}`` placeholder is present before rendering.

    The dispatcher fills it in at child-spawn time via
    ``load_harness_kb(self.kernel_language)``. The KB block is wrapped
    in a clearly-labelled markdown heading so the rendered prompt makes
    the injection point visually obvious to the model.
    """
    spec = registry_specs["harness-generator"]
    assert "## Language-Specific Knowledge Base" in spec.system_prompt
    assert "{{knowledge_base}}" in spec.system_prompt


def test_harness_generator_kb_placeholder_renders_after_substitution(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """After rendering with a fake KB string, the placeholder is replaced."""
    from jinja2 import StrictUndefined, Template

    spec = registry_specs["harness-generator"]
    rendered = Template(spec.system_prompt, undefined=StrictUndefined).render(knowledge_base="MY_FAKE_KB_BODY")
    assert "{{knowledge_base}}" not in rendered
    assert "MY_FAKE_KB_BODY" in rendered


def test_harness_generator_uses_from_kernel_language_kb_template(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """``knowledge_base_template`` is set to the recognised tag."""
    spec = registry_specs["harness-generator"]
    assert spec.knowledge_base_template == "from_kernel_language"


def test_harness_generator_temperature_pinned_to_zero(registry_specs: dict[str, SubagentSpec]) -> None:
    """``model_kwargs.temperature == 0.0`` for determinism (no seed on Claude)."""
    spec = registry_specs["harness-generator"]
    assert spec.model_kwargs.get("temperature") == 0.0


def test_harness_generator_description_is_concise(registry_specs: dict[str, SubagentSpec]) -> None:
    spec = registry_specs["harness-generator"]
    assert spec.description, "description must be non-empty"
    assert len(spec.description) <= _DESCRIPTION_MAX_CHARS, (
        f"description is {len(spec.description)} chars, max is {_DESCRIPTION_MAX_CHARS}"
    )
    assert "\n" not in spec.description, "description must be a single line"


def test_harness_generator_source_comment_marker_preserved() -> None:
    """The ``# source:`` comment on line 1 must survive in the raw file.

    YAML comments are dropped on parse, so we read the raw text directly.
    Commit set 5 cleanup keys off this marker to find ports mechanically.
    """
    raw = (_V3_ROOT / "harness-generator" / "SUBAGENT.yaml").read_text(encoding="utf-8")
    assert raw.startswith("# source: subagents/harness-generator/SYSTEM_PROMPT.md"), raw[:120]


# ---------------------------------------------------------------------------
# commit-set-6.5 enrichment: Consistency & Robustness Requirements block
#
# Quality requirements live in the SUBAGENT YAML (single source of truth,
# language-agnostic, always-on) — they are deliberately NOT duplicated into
# the per-language KBs under ``skills/<lang>/``. The block sits between the
# commit-6 ``{{knowledge_base}}`` injection and the legacy ``Goal`` section,
# inside the "additive enrichment" zone marked by ``## (Existing content
# continues below)``.
# ---------------------------------------------------------------------------


def test_harness_generator_includes_consistency_robustness_heading(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """The ``## Consistency & Robustness Requirements`` heading is present exactly once."""
    spec = registry_specs["harness-generator"]
    sp = spec.system_prompt
    heading = "## Consistency & Robustness Requirements"
    assert heading in sp, "missing Consistency & Robustness Requirements heading"
    assert sp.count(heading) == 1, f"heading must appear exactly once, found {sp.count(heading)}"


def test_harness_generator_consistency_block_has_seven_numbered_rules(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """All seven numbered rule headings are present inside the block.

    The numbered headings are the contract anchors for the block; if a
    future edit drops, renumbers, or paraphrases one we want a loud
    failure rather than a quiet drift in the deterministic-generation
    requirements.
    """
    spec = registry_specs["harness-generator"]
    sp = spec.system_prompt
    block_start = sp.index("## Consistency & Robustness Requirements")
    block_end = sp.index("## (Existing content continues below)", block_start)
    block_body = sp[block_start:block_end]

    expected_rule_headings = (
        "1. **Same input → same harness.**",
        "2. **Same shapes across all 4 CLI modes.**",
        "3. **Hard-fail with clear messages. No silent skips.**",
        "4. **Idempotent artifact production.**",
        "5. **Deterministic algorithms where applicable.**",
        "6. **Single command produces a complete, verifiable artifact.**",
        "7. **Same harness must work across all baseline-vs-optimized swaps.**",
    )
    for heading in expected_rule_headings:
        assert heading in block_body, f"missing rule heading {heading!r} in block body"


def test_harness_generator_consistency_block_lands_before_goal(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """The block sits AFTER the ``{{knowledge_base}}`` injection and BEFORE ``Goal``.

    This pins the structural insertion point chosen in commit set 6.5:
    KB block (commit 6) → Consistency block (commit 6.5) → seam marker →
    legacy ``Goal`` section. Pinning the order keeps the model's reading
    flow stable: quality requirements appear immediately after the
    language-specific KB and before any language-agnostic procedural rules.
    """
    spec = registry_specs["harness-generator"]
    sp = spec.system_prompt

    idx_kb = sp.index("## Language-Specific Knowledge Base")
    idx_block = sp.index("## Consistency & Robustness Requirements")
    idx_seam = sp.index("## (Existing content continues below)")
    idx_goal = sp.index("\nGoal\n")

    assert idx_kb < idx_block < idx_seam < idx_goal, (
        f"expected order kb<block<seam<goal, got {idx_kb=} {idx_block=} {idx_seam=} {idx_goal=}"
    )


def test_harness_generator_consistency_block_marker_in_yaml_header() -> None:
    """The raw file carries a ``# commit-set-6.5`` provenance comment.

    YAML comments are dropped on parse so we read the raw text. Mirrors
    the commit-set-6 marker convention so a future maintainer can trace
    each additive enrichment back to the commit that introduced it.
    """
    raw = (_V3_ROOT / "harness-generator" / "SUBAGENT.yaml").read_text(encoding="utf-8")
    assert "# commit-set-6.5 additive enrichment:" in raw
    assert "Consistency & Robustness Requirements" in raw


# ---------------------------------------------------------------------------
# speedup-verify (verbatim port)
# ---------------------------------------------------------------------------


def test_speedup_verify_is_registered(registry_specs: dict[str, SubagentSpec]) -> None:
    assert "speedup-verify" in registry_specs
    assert "speedup-verify" in SubagentRegistry(root=_V3_ROOT).names()


def test_speedup_verify_prompt_is_verbatim_lift(registry_specs: dict[str, SubagentSpec]) -> None:
    legacy_prompt = (_LEGACY_ROOT / "speedup-verify" / "SYSTEM_PROMPT.md").read_text(encoding="utf-8")

    spec = registry_specs["speedup-verify"]
    assert spec.system_prompt, "system_prompt must be non-empty"
    assert spec.system_prompt[:_VERBATIM_PREFIX_CHARS] == legacy_prompt[:_VERBATIM_PREFIX_CHARS], (
        "system_prompt prefix drifted from legacy SYSTEM_PROMPT.md"
    )
    assert spec.system_prompt == legacy_prompt, "system_prompt body drifted from legacy SYSTEM_PROMPT.md"


def test_speedup_verify_description_is_concise(registry_specs: dict[str, SubagentSpec]) -> None:
    spec = registry_specs["speedup-verify"]
    assert spec.description, "description must be non-empty"
    assert len(spec.description) <= _DESCRIPTION_MAX_CHARS, (
        f"description is {len(spec.description)} chars, max is {_DESCRIPTION_MAX_CHARS}"
    )
    assert "\n" not in spec.description, "description must be a single line"


def test_speedup_verify_marker_contract_preserved(registry_specs: dict[str, SubagentSpec]) -> None:
    """The ``GEAK_RESULT_GEOMEAN_SPEEDUP=<float>`` marker is a hard contract.

    The orchestrator (commit set 4) parses this exact token off the
    speedup-verify subagent's output, so it must remain in the prompt
    body verbatim. Asserting it here keeps a future reflow from silently
    dropping the marker name.
    """
    spec = registry_specs["speedup-verify"]
    assert "GEAK_RESULT_GEOMEAN_SPEEDUP=" in spec.system_prompt, (
        "GEAK_RESULT_GEOMEAN_SPEEDUP marker missing from speedup-verify prompt"
    )


def test_speedup_verify_source_comment_marker_preserved() -> None:
    raw = (_V3_ROOT / "speedup-verify" / "SUBAGENT.yaml").read_text(encoding="utf-8")
    assert raw.startswith("# source: subagents/speedup-verify/SYSTEM_PROMPT.md"), raw[:120]


# ---------------------------------------------------------------------------
# harness-verifier (authored fresh — no legacy SYSTEM_PROMPT.md exists)
# ---------------------------------------------------------------------------


def test_harness_verifier_is_registered(registry_specs: dict[str, SubagentSpec]) -> None:
    assert "harness-verifier" in registry_specs
    assert "harness-verifier" in SubagentRegistry(root=_V3_ROOT).names()


def test_harness_verifier_prompt_encodes_contract(registry_specs: dict[str, SubagentSpec]) -> None:
    """The fresh prompt must encode the validate_harness + execute_harness_validation contract.

    These substring assertions pin the contract elements that the
    orchestrator (commit set 4) relies on. They are deliberately loose
    enough to allow rewording, but strict enough to catch a future edit
    that drops a whole section (e.g. the determinism rule, or the
    GEAK_RESULT_LATENCY_MS marker requirement).
    """
    spec = registry_specs["harness-verifier"]
    sp = spec.system_prompt
    assert sp, "system_prompt must be non-empty"

    # The verifier explicitly references the source-of-truth functions so
    # that a maintainer who lands here knows what to read first.
    assert "validate_harness" in sp
    assert "execute_harness_validation" in sp

    # All four required harness CLI flags are mentioned in the static
    # validation section (mirroring REQUIRED_HARNESS_FLAGS).
    for flag in ("--correctness", "--profile", "--benchmark", "--full-benchmark"):
        assert flag in sp, f"missing flag {flag!r} in harness-verifier prompt"

    # The two output marker contracts the orchestrator parses off the
    # subagent: success token + failure token + escalation token.
    assert "HARNESS_VERIFIED=true" in sp
    assert "HARNESS_VERIFIED=false" in sp
    assert "ESCALATE=true" in sp

    # Runtime contract elements callers rely on.
    assert "GEAK_RESULT_LATENCY_MS" in sp, "missing latency marker contract"
    assert "GEAK_SHAPES_USED" in sp, "missing determinism marker contract"

    # KernelLanguage handoff: the prompt must reference the template
    # paths so the verifier knows where the language-specific harness
    # shape is defined.
    assert "harness_template_path" in sp
    assert "commandment_template_path" in sp


def test_harness_verifier_description_is_concise(registry_specs: dict[str, SubagentSpec]) -> None:
    spec = registry_specs["harness-verifier"]
    assert spec.description, "description must be non-empty"
    assert len(spec.description) <= _DESCRIPTION_MAX_CHARS, (
        f"description is {len(spec.description)} chars, max is {_DESCRIPTION_MAX_CHARS}"
    )
    assert "\n" not in spec.description, "description must be a single line"


def test_harness_verifier_authored_fresh_marker_preserved() -> None:
    """The ``# source: authored fresh`` comment must survive in the raw file.

    Unlike the verbatim ports (harness-generator, speedup-verify), this
    YAML was written from scratch — the marker variant signals that to
    commit set 5 cleanup so it does not try to diff against a
    non-existent legacy SYSTEM_PROMPT.md.
    """
    raw = (_V3_ROOT / "harness-verifier" / "SUBAGENT.yaml").read_text(encoding="utf-8")
    assert raw.startswith("# source: authored fresh"), raw[:120]


def test_harness_verifier_includes_shape_fixer_provenance_tag() -> None:
    """Commit set 6 enrichment carries a ``# source: ...shape_fixer_agent.py`` comment.

    The YAML comment is dropped on parse but lives in the raw file so a
    later maintainer can trace the lifted block back to its origin.
    """
    raw = (_V3_ROOT / "harness-verifier" / "SUBAGENT.yaml").read_text(encoding="utf-8")
    assert "shape_fixer_agent.py::SYSTEM_PROMPT" in raw, "missing source provenance tag for the lifted ShapeFixer block"


def test_harness_verifier_temperature_pinned_to_zero(registry_specs: dict[str, SubagentSpec]) -> None:
    """``model_kwargs.temperature == 0.0`` for determinism (no seed on Claude)."""
    spec = registry_specs["harness-verifier"]
    assert spec.model_kwargs.get("temperature") == 0.0


def test_harness_verifier_does_not_set_kb_template(registry_specs: dict[str, SubagentSpec]) -> None:
    """The verifier's tips are language-agnostic and inline; no KB injection."""
    spec = registry_specs["harness-verifier"]
    assert spec.knowledge_base_template is None


def test_harness_verifier_includes_shape_fixer_step_protocol(registry_specs: dict[str, SubagentSpec]) -> None:
    """Lifted Step 1-3 source-vs-harness compare protocol from ShapeFixer."""
    spec = registry_specs["harness-verifier"]
    sp = spec.system_prompt
    assert "Step 1:" in sp
    assert "Step 2:" in sp
    assert "Step 3:" in sp
    assert "SHAPE SOURCE FILE" in sp
    assert "HARNESS FILE" in sp
    assert "SOURCE-ORDERED full case stream" in sp
    assert "HARNESS-ORDERED full case stream" in sp


def test_harness_verifier_includes_same_values_same_count_rule(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """The canonical ShapeFixer order-drift rule, verbatim where it fits."""
    spec = registry_specs["harness-verifier"]
    assert "SAME VALUES / SAME COUNT" in spec.system_prompt
    assert "_pick()" in spec.system_prompt


def test_harness_verifier_includes_per_tensor_execution_contract_checklist(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """Lifted per-tensor execution-contract checklist from ShapeFixer."""
    spec = registry_specs["harness-verifier"]
    sp = spec.system_prompt
    for token in (
        "dtype",
        "device",
        "layout",
        "contiguity",
        "auxiliary",
        "index dtypes",
        "helper-side preprocessing",
        "supported flags",
    ):
        assert token in sp, f"missing per-tensor contract token {token!r}"


# ---------------------------------------------------------------------------
# commit-set-6.5 enrichment: Verification Robustness Requirements block
#
# Symmetric counterpart to the generator's Consistency block. Lives in the
# verifier SUBAGENT YAML, sits after the source-tagged ShapeFixer Phase 3
# block (commit 6) and before the legacy "Output rules" section.
# ---------------------------------------------------------------------------


def test_harness_verifier_includes_verification_robustness_heading(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """The ``## Verification Robustness Requirements`` heading is present exactly once."""
    spec = registry_specs["harness-verifier"]
    sp = spec.system_prompt
    heading = "## Verification Robustness Requirements"
    assert heading in sp, "missing Verification Robustness Requirements heading"
    assert sp.count(heading) == 1, f"heading must appear exactly once, found {sp.count(heading)}"


def test_harness_verifier_robustness_block_has_six_numbered_rules(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """All six numbered rule headings are present inside the block.

    The numbered headings are the contract anchors; a future edit that
    drops, renumbers, or paraphrases one should fail loudly rather than
    silently weaken the deterministic-verification requirements.
    """
    spec = registry_specs["harness-verifier"]
    sp = spec.system_prompt
    block_start = sp.index("## Verification Robustness Requirements")
    block_end = sp.index("Output rules", block_start)
    block_body = sp[block_start:block_end]

    expected_rule_headings = (
        "1. **Same harness → same verdict.**",
        "2. **Consistent rejection criteria.**",
        "3. **No permissive defaults.**",
        "4. **Runtime-phase determinism.**",
        "5. **Reject loop bound is non-negotiable.**",
        "6. **No silent rewrite.**",
    )
    for heading in expected_rule_headings:
        assert heading in block_body, f"missing rule heading {heading!r} in block body"


def test_harness_verifier_robustness_block_lands_after_shape_fixer(
    registry_specs: dict[str, SubagentSpec],
) -> None:
    """The block sits AFTER Phase 3 (commit-6 ShapeFixer lift) and BEFORE ``Output rules``.

    This pins the structural insertion point chosen in commit set 6.5:
    Phase 1 / Phase 2 (legacy) -> Phase 3 ShapeFixer block (commit 6) ->
    Verification Robustness block (commit 6.5) -> legacy Output rules.
    Mirrors the symmetric placement in the harness-generator YAML
    (KB block -> Consistency block -> seam -> Goal).
    """
    spec = registry_specs["harness-verifier"]
    sp = spec.system_prompt

    idx_phase3 = sp.index("Phase 3 — Shape-source comparison")
    idx_block = sp.index("## Verification Robustness Requirements")
    idx_output = sp.index("Output rules")

    assert idx_phase3 < idx_block < idx_output, (
        f"expected order phase3<block<output, got {idx_phase3=} {idx_block=} {idx_output=}"
    )


def test_harness_verifier_robustness_block_marker_in_yaml_header() -> None:
    """The raw file carries a ``# commit-set-6.5`` provenance comment.

    YAML comments are dropped on parse so we read the raw text directly,
    mirroring the marker convention used for prior commit-set additions
    so each enrichment is traceable back to the commit that introduced it.
    """
    raw = (_V3_ROOT / "harness-verifier" / "SUBAGENT.yaml").read_text(encoding="utf-8")
    assert "# commit-set-6.5 additive enrichment:" in raw
    assert "Verification Robustness Requirements" in raw


# ---------------------------------------------------------------------------
# Cross-cutting: the always-on set
# ---------------------------------------------------------------------------


def test_all_three_always_on_subagents_present(registry_specs: dict[str, SubagentSpec]) -> None:
    """Commit set 3 ships exactly these three v3 always-on subagents.

    ``pytorch-to-flydsl`` is deliberately NOT shipped here — translation
    is dispatched as a deterministic tool call, not via subagent dispatch
    (commit set 4 decision). This test pins that contract so an accidental
    drop-in of an extra YAML fails loudly before it reaches main.
    """
    expected = {"harness-generator", "harness-verifier", "speedup-verify"}
    discovered = set(registry_specs)
    assert discovered == expected, f"expected exactly {sorted(expected)} v3 subagents, got {sorted(discovered)}"


# ---------------------------------------------------------------------------
# tools: enumeration per subagent (commit set 4)
#
# The v3 orchestrator restricts a child subagent's tool surface to the set
# declared in its YAML. The exact registry names here matter — the
# orchestrator looks them up against the tool registry verbatim, so a typo
# (or a divergence from the registry) silently disables a tool.
#
# The conceptual names from the design doc (``read_file``, ``write_file``,
# ``run_command``) don't exist 1:1 in the registry. The mapping the
# orchestrator commits to is:
#
#   read_file   -> str_replace_editor (view sub-command)
#   write_file  -> str_replace_editor (create / str_replace / insert)
#   run_command -> bash
#   save_and_test -> save_and_test (1:1)
# ---------------------------------------------------------------------------


def test_harness_generator_tools(registry_specs: dict[str, SubagentSpec]) -> None:
    """harness-generator can read/write/exec/save-and-test (full toolbox).

    Order isn't part of the contract — we compare as sets so YAML reordering
    doesn't break the suite.
    """
    spec = registry_specs["harness-generator"]
    assert set(spec.tools) == {"bash", "str_replace_editor", "save_and_test"}, (
        f"unexpected tools for harness-generator: {spec.tools!r}"
    )


def test_harness_verifier_tools(registry_specs: dict[str, SubagentSpec]) -> None:
    """harness-verifier is the read-only set: bash + str_replace_editor.

    The prompt forbids edits to kernel/harness sources; the registry has no
    "view-only" editor today, so we ship ``str_replace_editor`` and rely on
    the prompt's read-only contract. If a future commit adds a true
    read-only viewer, this test should be updated to require it.
    """
    spec = registry_specs["harness-verifier"]
    assert set(spec.tools) == {"bash", "str_replace_editor"}, f"unexpected tools for harness-verifier: {spec.tools!r}"


def test_speedup_verify_tools(registry_specs: dict[str, SubagentSpec]) -> None:
    """speedup-verify needs read+write (script generation) + bash (verification).

    The prompt's workflow ends with running the generated script against
    the baseline output to confirm a ~1.0x speedup; bash is required for
    that verification step.
    """
    spec = registry_specs["speedup-verify"]
    assert set(spec.tools) == {"bash", "str_replace_editor"}, f"unexpected tools for speedup-verify: {spec.tools!r}"


# ---------------------------------------------------------------------------
# max_steps configuration per subagent (commit set 4)
# ---------------------------------------------------------------------------


def test_harness_generator_uses_unlimited_steps(registry_specs: dict[str, SubagentSpec]) -> None:
    """harness-generator opts in to the unlimited-steps sentinel.

    Harness generation can take many tool-call rounds (README, deps, tests,
    iteration against verifier feedback). Capping it at 30 steps would
    pessimise slow-but-correct runs; the orchestrator gates retries via
    ``harness-verifier``'s ESCALATE token instead.
    """
    spec = registry_specs["harness-generator"]
    assert spec.max_steps == -1
    assert spec.is_unlimited_steps is True


def test_harness_verifier_uses_default_steps(registry_specs: dict[str, SubagentSpec]) -> None:
    """harness-verifier sticks to the 30-step default.

    Verification is mechanical (static checks + 4 timed harness invocations)
    and should never need many tool-call rounds. The default cap protects
    against runaway loops if a model keeps re-running validation.
    """
    spec = registry_specs["harness-verifier"]
    assert spec.max_steps == 30
    assert spec.is_unlimited_steps is False


def test_speedup_verify_uses_default_steps(registry_specs: dict[str, SubagentSpec]) -> None:
    """speedup-verify sticks to the 30-step default.

    Writing one Python script + verifying it should be a small handful of
    steps; the default cap is a safety net.
    """
    spec = registry_specs["speedup-verify"]
    assert spec.max_steps == 30
    assert spec.is_unlimited_steps is False
