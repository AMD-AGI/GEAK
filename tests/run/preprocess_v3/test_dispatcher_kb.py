"""Tests for the per-language KB injection in :class:`PreprocessSubagentDispatcher`.

The dispatcher is the seam between the on-disk ``skills/<lang>/`` KB
loaded by :func:`load_harness_kb` and the v3 ``harness-generator``
subagent's ``{{knowledge_base}}`` Jinja placeholder.

The tests below cover four cases (per the commit-set brief):

* Triton dispatch with a Triton-flagged spec injects non-empty Triton
  KB content into the child's system prompt.
* HIP dispatch with a HIP-flagged spec injects non-empty HIP KB content
  into the child's system prompt.
* Dispatch of a verifier-style spec (``knowledge_base_template == None``)
  does NOT inject any KB.
* Dispatch with an unknown language (no ``skills/<name>/`` folder)
  gracefully gets an empty KB (no exception).
"""

from __future__ import annotations

from pathlib import Path

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.registry import (
    SubagentRegistry,
    SubagentSpec,
)
from minisweagent.run.preprocess_v3.tools import (
    ALLOWED_SUBAGENT_NAMES,
    PreprocessSubagentDispatcher,
)

assert "harness-generator" in ALLOWED_SUBAGENT_NAMES


# ---------------------------------------------------------------------------
# Fixture: a fake subagent factory that captures the child's rendered prompt
# ---------------------------------------------------------------------------


class _CapturingAgent:
    """Stand-in for :class:`PreprocessSubagent`.

    Captures the ``system_prompt`` template + the ``extra_template_vars``
    so each test can render the prompt and assert on the resulting text.
    """

    def __init__(self, *, system_prompt: str, extra_template_vars: dict) -> None:
        self.system_prompt = system_prompt
        self.extra_template_vars = dict(extra_template_vars)

    def run(self, _task: str) -> tuple[str, str]:
        return ("Submitted", "ok")

    def rendered_system_prompt(self) -> str:
        """Render the system prompt the same way PreprocessSubagent would.

        Mirrors ``PreprocessSubagent.render_template`` so the assertions
        below see exactly what the LLM would have received.
        """
        from jinja2 import StrictUndefined, Template

        return Template(self.system_prompt, undefined=StrictUndefined).render(**self.extra_template_vars)


def _capturing_factory(captured: dict):
    """Return a factory that records the spec and returns a _CapturingAgent."""

    def _factory(*, spec, model, cwd, extra_template_vars=None, **_kwargs):
        agent = _CapturingAgent(
            system_prompt=spec.system_prompt,
            extra_template_vars=extra_template_vars or {},
        )
        captured["agent"] = agent
        captured["spec"] = spec
        return agent

    return _factory


def _write_spec(tmp_path: Path, name: str, *, body: str) -> None:
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SUBAGENT.yaml").write_text(body, encoding="utf-8")


_GENERATOR_BODY = """\
name: harness-generator
description: Test generator with KB injection.
system_prompt: |
  ## Language KB

  {{knowledge_base}}

  ## Static rules
  Always be helpful.
knowledge_base_template: from_kernel_language
max_steps: -1
"""

_VERIFIER_BODY = """\
name: harness-verifier
description: Test verifier without KB injection.
system_prompt: |
  ## Static verifier rules

  Always check the contract.
"""


def _mk_lang(name: str) -> KernelLanguage:
    return KernelLanguage(name=name, file_extensions=frozenset({".py"}), kb_namespace=name)


# ---------------------------------------------------------------------------
# Triton: KB is injected and rendered into the child's system prompt
# ---------------------------------------------------------------------------


def test_dispatcher_injects_triton_kb_into_generator(tmp_path: Path) -> None:
    _write_spec(tmp_path, "harness-generator", body=_GENERATOR_BODY)

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        SubagentRegistry(root=tmp_path),
        agent_factory=_capturing_factory(captured),
        kernel_language=_mk_lang("triton"),
    )
    result = dispatcher(name="harness-generator", task="Generate it", model=object())

    assert result["success"] is True

    agent: _CapturingAgent = captured["agent"]
    rendered = agent.rendered_system_prompt()

    # Triton-specific sentinel phrases must be present after rendering.
    assert "USER TASK CONTEXT" in rendered
    assert "ALL_SHAPES" in rendered
    assert "@triton.jit" in rendered
    # No HIP-specific phrases — KBs do not cross-contaminate.
    assert "hipLaunchKernelGGL" not in rendered
    assert "--offload-arch=gfx942" not in rendered


# ---------------------------------------------------------------------------
# HIP: KB is injected and rendered into the child's system prompt
# ---------------------------------------------------------------------------


def test_dispatcher_injects_hip_kb_into_generator(tmp_path: Path) -> None:
    _write_spec(tmp_path, "harness-generator", body=_GENERATOR_BODY)

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        SubagentRegistry(root=tmp_path),
        agent_factory=_capturing_factory(captured),
        kernel_language=_mk_lang("hip"),
    )
    dispatcher(name="harness-generator", task="Generate it", model=object())

    agent: _CapturingAgent = captured["agent"]
    rendered = agent.rendered_system_prompt()

    # HIP-specific sentinel phrases must be present after rendering.
    assert "hipLaunchKernelGGL" in rendered
    assert "--offload-arch=gfx942" in rendered
    assert "Composable Kernel" in rendered
    # No Triton-specific phrases — KBs do not cross-contaminate.
    assert "@triton.jit" not in rendered


# ---------------------------------------------------------------------------
# Verifier-style spec: no KB injection
# ---------------------------------------------------------------------------


def test_dispatcher_does_not_inject_kb_for_verifier(tmp_path: Path) -> None:
    _write_spec(tmp_path, "harness-verifier", body=_VERIFIER_BODY)

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        SubagentRegistry(root=tmp_path),
        agent_factory=_capturing_factory(captured),
        kernel_language=_mk_lang("triton"),
    )
    dispatcher(name="harness-verifier", task="Verify it", model=object())

    agent: _CapturingAgent = captured["agent"]
    # The verifier's spec has no knowledge_base_template; the dispatcher
    # therefore does not seed a ``knowledge_base`` template var.
    assert "knowledge_base" not in agent.extra_template_vars


# ---------------------------------------------------------------------------
# Unknown language: graceful degradation (empty KB, no exception)
# ---------------------------------------------------------------------------


def test_dispatcher_handles_unknown_language_gracefully(tmp_path: Path) -> None:
    _write_spec(tmp_path, "harness-generator", body=_GENERATOR_BODY)

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        SubagentRegistry(root=tmp_path),
        agent_factory=_capturing_factory(captured),
        kernel_language=_mk_lang("definitely-not-a-real-language"),
    )
    result = dispatcher(name="harness-generator", task="Generate it", model=object())

    assert result["success"] is True
    agent: _CapturingAgent = captured["agent"]
    # The KB resolves to the empty string but the placeholder is still
    # provided so Jinja's StrictUndefined renderer does not crash.
    assert agent.extra_template_vars["knowledge_base"] == ""
    rendered = agent.rendered_system_prompt()
    # The static rules portion of the prompt survives intact.
    assert "Always be helpful" in rendered


# ---------------------------------------------------------------------------
# Dispatcher constructed without a kernel_language: still works
# ---------------------------------------------------------------------------


def test_dispatcher_handles_missing_kernel_language_attribute(tmp_path: Path) -> None:
    """When kernel_language is None but the spec requests KB, inject ``""``.

    This is the "preprocess pipeline didn't resolve a language" failure
    mode — the dispatcher must not raise; it logs and proceeds with an
    empty KB so the subagent still runs.
    """
    _write_spec(tmp_path, "harness-generator", body=_GENERATOR_BODY)

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        SubagentRegistry(root=tmp_path),
        agent_factory=_capturing_factory(captured),
        kernel_language=None,
    )
    result = dispatcher(name="harness-generator", task="Generate it", model=object())

    assert result["success"] is True
    agent: _CapturingAgent = captured["agent"]
    assert agent.extra_template_vars["knowledge_base"] == ""


# ---------------------------------------------------------------------------
# Direct loader-tag resolution (covers the open-ended tag set)
# ---------------------------------------------------------------------------


def test_dispatcher_ignores_unknown_kb_template_tag(tmp_path: Path, caplog) -> None:
    """A spec with an unknown ``knowledge_base_template`` is logged + ignored."""
    body = """\
name: harness-generator
description: Test generator with an unknown KB tag.
system_prompt: |
  Static-only prompt.
knowledge_base_template: from_some_future_routing_key
max_steps: -1
"""
    _write_spec(tmp_path, "harness-generator", body=body)

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        SubagentRegistry(root=tmp_path),
        agent_factory=_capturing_factory(captured),
        kernel_language=_mk_lang("triton"),
    )
    result = dispatcher(name="harness-generator", task="Generate", model=object())

    assert result["success"] is True
    agent: _CapturingAgent = captured["agent"]
    # No KB was injected for the unrecognised tag.
    assert "knowledge_base" not in agent.extra_template_vars


# ---------------------------------------------------------------------------
# Dispatcher attribute exposure
# ---------------------------------------------------------------------------


def test_dispatcher_stores_kernel_language_on_instance(tmp_path: Path) -> None:
    """``self.kernel_language`` is accessible for callers / future tooling."""
    lang = _mk_lang("triton")
    dispatcher = PreprocessSubagentDispatcher(
        SubagentRegistry(root=tmp_path),
        kernel_language=lang,
    )
    assert dispatcher.kernel_language is lang


def test_dispatcher_default_kernel_language_is_none(tmp_path: Path) -> None:
    dispatcher = PreprocessSubagentDispatcher(SubagentRegistry(root=tmp_path))
    assert dispatcher.kernel_language is None


# ---------------------------------------------------------------------------
# Direct construction of the SubagentSpec (skip the YAML round-trip)
# ---------------------------------------------------------------------------


def test_dispatcher_with_directly_constructed_spec(tmp_path: Path) -> None:
    """Smoke test using a SubagentSpec passed via a custom factory.

    Confirms the KB injection works the same whether the spec came from
    the YAML on disk or was built in code.
    """
    spec = SubagentSpec(
        name="harness-generator",
        description="x",
        system_prompt="KB={{knowledge_base}}",
        knowledge_base_template="from_kernel_language",
        max_steps=-1,
    )

    captured: dict = {}

    class _Registry:
        def get(self, name: str) -> SubagentSpec:
            assert name == "harness-generator"
            return spec

    dispatcher = PreprocessSubagentDispatcher(
        _Registry(),  # type: ignore[arg-type]
        agent_factory=_capturing_factory(captured),
        kernel_language=_mk_lang("triton"),
    )
    result = dispatcher(name="harness-generator", task="x", model=object())
    assert result["success"] is True

    agent: _CapturingAgent = captured["agent"]
    rendered = agent.rendered_system_prompt()
    assert rendered.startswith("KB=")
    assert "USER TASK CONTEXT" in rendered
