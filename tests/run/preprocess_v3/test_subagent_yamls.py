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
