"""SubagentBase — peer class of OptimizationAgent (NOT a subclass of it).

This module is intentionally small. The contract is:

  - `class MySubagent(SubagentBase)` — every subagent inherits from this ABC.
  - Each subclass overrides EXACTLY ONE of:
      * `run(**inputs) -> str | dict`  (one-shot)
      * `loop(max_attempts, verify_fn, **inputs) -> Result`  (multi-round)
    Overriding both, or neither, fails CI gate
    `scripts/refactor_ci/check_subagent_base_contract.py`.

  - A subagent that needs a full LLM tool loop calls
    `self._make_optimization_agent(tools)` — this COMPOSES a short-lived
    `OptimizationAgent` and runs it. SubagentBase never INHERITS from
    OptimizationAgent; they are peer classes.

See docs/refactor/EXECUTION_PLAN.md §16.2 for the full design rationale.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from minisweagent.kernel_languages.base import KernelLanguage


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SubagentConfig:
    """Configuration for a single subagent invocation.

    Loaded from YAML at `pipeline_workers/<area>/configs/<name>.yaml`. Each subagent
    has its own config — step_limit + model are tuned per narrow-task
    so that a one-shot HarnessBuilder doesn't accidentally inherit the
    30-turn budget of a long optimization round.
    """

    name: str                                                  # subagent's name (e.g. "harness_builder")
    model_name: str                                            # which LLM model to use
    system_template: str                                       # str.format-style; e.g. with {language_name}, {system_prompt}
    instance_template: str                                     # per-invocation user prompt template
    step_limit: int = 20                                       # max LLM turns before LimitsExceeded
    cost_limit: float = 3.0                                    # USD cost ceiling per invocation
    temperature: float = 0.2
    extra: Dict[str, Any] = field(default_factory=dict)         # subagent-specific extras


# ---------------------------------------------------------------------------
# SubagentBase — the ONE base class for every subagent
# ---------------------------------------------------------------------------

class SubagentBase(ABC):
    """Base for narrow-task LLM subagents. Peer of OptimizationAgent (NOT a subclass).

    Lifecycle (for both run() and loop()):
      1. Caller instantiates: `MySubagent(language, config_path)`
      2. Caller invokes:      `subagent.run(**inputs)` or `subagent.loop(...)`
      3. Subagent internally composes OptimizationAgent(s) via `_make_optimization_agent`
         when it needs a tool loop.
      4. Subagent returns a structured artifact (file path, dict, markdown string).

    Subclass contract (enforced by CI):
      - Override EXACTLY ONE of `run` / `loop`.
      - Do NOT call `super().run(...)` or `super().loop(...)` — they raise.
      - Output MUST be structured: a path, dict, or string that downstream code
        consumes programmatically. Unstructured chat is not the job of a subagent.
    """

    def __init__(self, language: "KernelLanguage", config_path: Optional[Path] = None,
                 config: Optional[SubagentConfig] = None):
        self.language = language
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            raise ValueError("SubagentBase requires either config or config_path")

    # ------------------------------------------------------------------
    # Execution methods — subclass overrides EXACTLY ONE
    # ------------------------------------------------------------------

    def run(self, **inputs: Any) -> str | dict:
        """One-shot execution. Composes a single OptimizationAgent call.

        Override in subclasses that produce their result in one pass
        (e.g. HarnessBuilder, KernelAnalysisAgent, UnitTestAgent, ShapeFixerAgent,
        CrossSessionMemoryAnalysisAgent).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override run() OR loop() — not both, not neither. "
            f"See docs/refactor/EXECUTION_PLAN.md §16.2."
        )

    def loop(
        self,
        *,
        max_attempts: int,
        verify_fn: Callable[[Any], bool],
        **inputs: Any,
    ) -> Any:
        """Multi-round execution. Composes an OptimizationAgent PER ATTEMPT until
        `verify_fn` returns ok OR `max_attempts` exhausted.

        Override in subclasses that need retry+verify semantics (e.g. TranslationLoop).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override loop() OR run() — not both, not neither. "
            f"See docs/refactor/EXECUTION_PLAN.md §16.2."
        )

    # ------------------------------------------------------------------
    # Shared helpers (used by subclasses' run/loop implementations)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> SubagentConfig:
        """Load a YAML config file into a SubagentConfig dataclass.

        Format expected:
            name: ...
            model_name: ...
            system_template: |
              ...
            instance_template: |
              ...
            step_limit: 20
            cost_limit: 3.0
            temperature: 0.2
            extra:
              ...
        """
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "PyYAML required to load subagent configs. Install with "
                "`pip install pyyaml`."
            ) from e

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return SubagentConfig(
            name=data.get("name", path.stem),
            model_name=data["model_name"],
            system_template=data["system_template"],
            instance_template=data["instance_template"],
            step_limit=int(data.get("step_limit", 20)),
            cost_limit=float(data.get("cost_limit", 3.0)),
            temperature=float(data.get("temperature", 0.2)),
            extra=data.get("extra", {}) or {},
        )

    def _compose_prompt(self, **template_vars: Any) -> tuple[str, str]:
        """Render (system_prompt, instance_prompt) from the config templates.

        Subclasses pass their task-specific variables (plus they get
        `{language_name}` and related common vars for free).
        """
        common: Dict[str, Any] = {
            "language_name": self.language.name,
        }
        common.update(template_vars)
        sys_p = self.config.system_template.format(**common)
        inst_p = self.config.instance_template.format(**common)
        return sys_p, inst_p

    def _make_optimization_agent(self, tools: list, **kwargs: Any) -> Any:
        """COMPOSITION boundary. Instantiate a short-lived OptimizationAgent.

        This is the ONLY place a subagent touches the OptimizationAgent class.
        Subagents never subclass it — they borrow its tool-loop by instantiation.

        Note: PR-3 lands OptimizationAgent. Pre-PR-3, callers should expect
        this method to raise NotImplementedError — use a temporary stub (return
        a dict mimicking the agent's interface) in early subagent implementations
        OR defer subagent integration until PR-3.
        """
        try:
            from minisweagent.agents.optimization_agent import OptimizationAgent
        except ImportError as e:
            raise NotImplementedError(
                "OptimizationAgent is not yet available (lands in PR-3). "
                "Subagents that need a tool loop must wait for PR-3, or provide "
                "a mock OptimizationAgent via the `_make_optimization_agent` "
                "override."
            ) from e

        # OptimizationAgent signature is finalized in PR-3; this call site will
        # be adjusted then. For now this is a forward-declaration.
        return OptimizationAgent(
            config=self.config,
            tools=tools,
            **kwargs,
        )
