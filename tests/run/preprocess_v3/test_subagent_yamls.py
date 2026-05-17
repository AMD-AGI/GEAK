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


def test_harness_generator_prompt_is_verbatim_lift(registry_specs: dict[str, SubagentSpec]) -> None:
    """The v3 ``system_prompt`` must equal the legacy SYSTEM_PROMPT.md body.

    We compare the first :data:`_VERBATIM_PREFIX_CHARS` characters strictly
    so the test name promise ("verbatim port") still fails loudly if anyone
    later paraphrases the opening role definition. We then assert the full
    body matches too — the prefix check just gives a tighter failure
    message when only the start drifts.
    """
    legacy_prompt = (_LEGACY_ROOT / "harness-generator" / "SYSTEM_PROMPT.md").read_text(encoding="utf-8")

    spec = registry_specs["harness-generator"]
    assert spec.system_prompt, "system_prompt must be non-empty"
    assert spec.system_prompt[:_VERBATIM_PREFIX_CHARS] == legacy_prompt[:_VERBATIM_PREFIX_CHARS], (
        "system_prompt prefix drifted from legacy SYSTEM_PROMPT.md"
    )
    assert spec.system_prompt == legacy_prompt, "system_prompt body drifted from legacy SYSTEM_PROMPT.md"


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


# ---------------------------------------------------------------------------
# Cross-cutting: the always-on set
# ---------------------------------------------------------------------------


def test_all_three_always_on_subagents_present(registry_specs: dict[str, SubagentSpec]) -> None:
    """Commit set 3 ships exactly these three v3 always-on subagents.

    ``pytorch-to-flydsl`` is deliberately NOT shipped here — it lands in
    commit set 3.5 / 4 once the parallel investigation completes. This
    test pins that contract so an accidental drop-in of an extra YAML
    fails loudly before it reaches main.
    """
    expected = {"harness-generator", "harness-verifier", "speedup-verify"}
    discovered = set(registry_specs)
    assert discovered == expected, f"expected exactly {sorted(expected)} v3 subagents, got {sorted(discovered)}"
