"""End-to-end integration test for per-language KB injection.

This is the contract-level test that ties together:

* the on-disk ``subagents/preprocess/harness-generator/SUBAGENT.yaml``
  (now carrying ``knowledge_base_template: from_kernel_language`` and
  the ``{{knowledge_base}}`` placeholder),
* the on-disk ``skills/<language>/`` knowledge bases (Triton + HIP),
* :func:`minisweagent.run.preprocess_v3.harness_kb.load_harness_kb`
  (loader),
* :class:`minisweagent.run.preprocess_v3.tools.PreprocessSubagentDispatcher`
  (injector), and
* :func:`minisweagent.run.preprocess_v3.tools.register_default_tools`
  (wiring).

The test mocks the child agent so we can capture and assert on the
exact system prompt the LLM would have received. No real model calls,
no real LLM dispatch.

If a more granular integration mark exists (e.g. ``@pytest.mark.integration``)
the brief allows tagging this; the project does not currently register
that mark, so the test stays unmarked.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import StrictUndefined, Template

from minisweagent.kernel_languages.base import KernelLanguage
from minisweagent.run.preprocess_v3.orchestrator import PreprocessOrchestratorAgent
from minisweagent.run.preprocess_v3.registry import SubagentRegistry
from minisweagent.run.preprocess_v3.tools import (
    PreprocessSubagentDispatcher,
    register_default_tools,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_V3_SUBAGENTS_ROOT = _REPO_ROOT / "subagents" / "preprocess"


# ---------------------------------------------------------------------------
# Test harness: capturing child-agent factory + stub orchestrator model
# ---------------------------------------------------------------------------


class _CapturingChildAgent:
    """Stand-in for :class:`PreprocessSubagent`; records what it would render."""

    def __init__(self, *, system_prompt: str, extra_template_vars: dict) -> None:
        self.system_prompt = system_prompt
        self.extra_template_vars = dict(extra_template_vars)
        self.tasks: list[str] = []

    def run(self, task: str) -> tuple[str, str]:
        self.tasks.append(task)
        return (
            "Submitted",
            "TEST_COMMAND: cd /tmp/repo && python harness.py --correctness && python harness.py --benchmark",
        )

    def rendered_system_prompt(self) -> str:
        return Template(self.system_prompt, undefined=StrictUndefined).render(**self.extra_template_vars)


def _capturing_factory(captured: dict):
    def _factory(*, spec, model, cwd, extra_template_vars=None, **_kwargs):
        agent = _CapturingChildAgent(
            system_prompt=spec.system_prompt,
            extra_template_vars=extra_template_vars or {},
        )
        captured["agent"] = agent
        captured["spec"] = spec
        captured["cwd"] = cwd
        return agent

    return _factory


class _StubOrchestratorModel:
    """Minimal model stub for the parent orchestrator agent."""

    n_calls = 0
    cost = 0.0


def _mk_lang(name: str) -> KernelLanguage:
    return KernelLanguage(name=name, file_extensions=frozenset({".py"}), kb_namespace=name)


def _write_fixture_kernel(tmp_path: Path) -> Path:
    """Drop a tiny synthetic Triton kernel under ``tmp_path`` for realism."""
    kernel_dir = tmp_path / "kernel_repo"
    kernel_dir.mkdir()
    kernel_path = kernel_dir / "kernel.py"
    kernel_path.write_text(
        """\
import triton
import triton.language as tl


@triton.jit
def my_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def my_op(x, y):
    import torch
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
    my_kernel[grid](x, y, out, N, BLOCK_SIZE=1024)
    return out
""",
        encoding="utf-8",
    )
    return kernel_path


# ---------------------------------------------------------------------------
# Integration: dispatch harness-generator with Triton language
# ---------------------------------------------------------------------------


def test_orchestrator_dispatch_injects_triton_kb_end_to_end(tmp_path: Path) -> None:
    """Triton dispatch through the real wiring injects Triton tips, NOT HIP tips."""
    _write_fixture_kernel(tmp_path)

    registry = SubagentRegistry(root=_V3_SUBAGENTS_ROOT)
    triton_lang = _mk_lang("triton")

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        registry,
        agent_factory=_capturing_factory(captured),
        kernel_language=triton_lang,
    )

    agent = PreprocessOrchestratorAgent(model=_StubOrchestratorModel())
    register_default_tools(
        agent,
        kernel_language=triton_lang,
        registry=registry,
        dispatcher=dispatcher,
    )

    dispatch_tool = agent._tools["dispatch_subagent"].callable
    result = dispatch_tool(name="harness-generator", task="Generate the harness")

    assert result["success"] is True
    assert "agent" in captured, "the dispatcher must have constructed a child agent"
    child: _CapturingChildAgent = captured["agent"]

    rendered = child.rendered_system_prompt()

    # --- Common-tips text (lifted into skills/triton/SKILL.md from
    #     legacy mini_unit_test_agent.yaml USER TASK CONTEXT rule).
    assert "USER TASK CONTEXT" in rendered

    # --- Triton-specific tip text (the importlib pitfall sentence lives
    #     in skills/triton/docs/triton_harness_writing.md).
    assert "importlib.util" in rendered
    assert "@triton.jit" in rendered

    # --- HIP-specific tip text MUST NOT appear (no cross-contamination).
    assert "hipLaunchKernelGGL" not in rendered
    assert "--offload-arch=gfx942" not in rendered
    assert "Composable Kernel" not in rendered

    # --- The legacy harness-generator body (preserved verbatim around
    #     the KB block) also still appears in the rendered prompt.
    assert "TestHarnessAgent" in rendered
    assert "TEST_COMMAND" in rendered

    # --- The KB injection point is clearly labelled.
    assert "## Language-Specific Knowledge Base" in rendered


# ---------------------------------------------------------------------------
# Integration: dispatch harness-generator with HIP language
# ---------------------------------------------------------------------------


def test_orchestrator_dispatch_injects_hip_kb_end_to_end(tmp_path: Path) -> None:
    """HIP dispatch through the real wiring injects HIP tips, NOT Triton tips."""
    _write_fixture_kernel(tmp_path)

    registry = SubagentRegistry(root=_V3_SUBAGENTS_ROOT)
    hip_lang = _mk_lang("hip")

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        registry,
        agent_factory=_capturing_factory(captured),
        kernel_language=hip_lang,
    )

    agent = PreprocessOrchestratorAgent(model=_StubOrchestratorModel())
    register_default_tools(
        agent,
        kernel_language=hip_lang,
        registry=registry,
        dispatcher=dispatcher,
    )

    dispatch_tool = agent._tools["dispatch_subagent"].callable
    result = dispatch_tool(name="harness-generator", task="Generate the harness")

    assert result["success"] is True
    child: _CapturingChildAgent = captured["agent"]
    rendered = child.rendered_system_prompt()

    # --- Common-tips text (also lifted into skills/hip/SKILL.md per
    #     locked decision 1: language-agnostic tips duplicate per KB).
    assert "USER TASK CONTEXT" in rendered

    # --- HIP-specific tip text (the three build shapes, MFMA-adjacent
    #     idioms, and per-kernel-type bullets live in skills/hip/docs/).
    assert "hipLaunchKernelGGL" in rendered
    assert "--offload-arch=gfx942" in rendered
    assert "Composable Kernel" in rendered
    assert "sys.path.insert" in rendered  # the NEVER-use warning

    # --- Triton-specific tip text MUST NOT appear (no cross-contamination).
    assert "@triton.jit" not in rendered
    assert "tl.dot" not in rendered

    # --- The legacy harness-generator body still appears.
    assert "TestHarnessAgent" in rendered
    assert "TEST_COMMAND" in rendered


# ---------------------------------------------------------------------------
# Integration: dispatch harness-verifier — no KB, no cross-contamination
# ---------------------------------------------------------------------------


def test_orchestrator_dispatch_verifier_does_not_inject_kb(tmp_path: Path) -> None:
    """The harness-verifier spec carries no KB tag → no KB injected.

    Defends against a future refactor that accidentally seeds
    ``{"knowledge_base": <triton blob>}`` for every dispatched subagent.
    """
    _write_fixture_kernel(tmp_path)

    registry = SubagentRegistry(root=_V3_SUBAGENTS_ROOT)
    triton_lang = _mk_lang("triton")

    captured: dict = {}
    dispatcher = PreprocessSubagentDispatcher(
        registry,
        agent_factory=_capturing_factory(captured),
        kernel_language=triton_lang,
    )

    agent = PreprocessOrchestratorAgent(model=_StubOrchestratorModel())
    register_default_tools(
        agent,
        kernel_language=triton_lang,
        registry=registry,
        dispatcher=dispatcher,
    )

    dispatch_tool = agent._tools["dispatch_subagent"].callable
    dispatch_tool(name="harness-verifier", task="Verify the harness")

    child: _CapturingChildAgent = captured["agent"]
    assert "knowledge_base" not in child.extra_template_vars
    # The verifier's own static prompt body still renders, of course.
    rendered = child.rendered_system_prompt()
    assert "HarnessVerifierAgent" in rendered
    # And no Triton KB phrases leaked into it.
    assert "@triton.jit" not in rendered
