"""SubAgentRegistry -- discover and manage subagent definitions.

Subagents are defined in ``subagents/<folder>/SUBAGENT.yaml`` under the GEAK
repository root.  The registry scans this directory at startup, parses each
definition, and provides:

1. A catalogue of available subagents (name + description) for system-prompt
   injection.
2. JSON tool schemas so the LLM can invoke registered subagents via tool
   calling.
3. Lookup by name for the ``SubAgentTool`` dispatcher.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from minisweagent import get_repo_root

logger = logging.getLogger(__name__)


@dataclass
class SubAgentParameter:
    """A single parameter accepted by a subagent."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False


@dataclass
class SubAgentDescriptor:
    """Parsed representation of a SUBAGENT.yaml file."""

    name: str
    description: str
    execution_mode: str = "inprocess"  # "inprocess" | "subprocess"
    entry_script: str | None = None
    step_limit: int = 0
    cost_limit: float = 0.0
    parameters: list[SubAgentParameter] = field(default_factory=list)
    path: Path = field(default_factory=lambda: Path("."))
    # Embedded configuration sections (from the same SUBAGENT.yaml)
    agent_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    env_config: dict[str, Any] = field(default_factory=dict)
    tools_config: dict[str, Any] = field(default_factory=dict)


class SubAgentRegistry:
    """Discover, register, and query subagent definitions."""

    DEFINITION_FILE = "SUBAGENT.yaml"

    def __init__(self, subagents_dir: Path | None = None):
        if subagents_dir is None:
            subagents_dir = get_repo_root() / "subagents"
        self._subagents_dir = subagents_dir
        self.subagents: dict[str, SubAgentDescriptor] = self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> dict[str, SubAgentDescriptor]:
        """Scan ``subagents/`` for folders containing SUBAGENT.yaml."""
        if not self._subagents_dir.is_dir():
            logger.debug("Subagents directory not found: %s", self._subagents_dir)
            return {}

        found: dict[str, SubAgentDescriptor] = {}
        for folder in sorted(self._subagents_dir.iterdir()):
            if not folder.is_dir():
                continue
            yaml_path = folder / self.DEFINITION_FILE
            if not yaml_path.exists():
                continue
            try:
                desc = self._parse_yaml(yaml_path, folder)
                if desc.name in found:
                    logger.warning(
                        "Duplicate subagent name %r in %s (already registered from %s); skipping.",
                        desc.name,
                        folder,
                        found[desc.name].path,
                    )
                    continue
                found[desc.name] = desc
                logger.debug("Registered subagent: %s (%s)", desc.name, desc.execution_mode)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", yaml_path, exc)

        return found

    @staticmethod
    def _parse_yaml(yaml_path: Path, folder: Path) -> SubAgentDescriptor:
        """Parse a single SUBAGENT.yaml into a descriptor."""
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

        name = raw.get("name")
        description = raw.get("description")
        if not name or not description:
            raise ValueError(f"SUBAGENT.yaml must have 'name' and 'description': {yaml_path}")

        params = []
        for p in raw.get("parameters") or []:
            params.append(
                SubAgentParameter(
                    name=p["name"],
                    type=p.get("type", "string"),
                    description=p.get("description", ""),
                    required=p.get("required", False),
                )
            )

        return SubAgentDescriptor(
            name=name,
            description=str(description).strip(),
            execution_mode=raw.get("execution_mode", "inprocess"),
            entry_script=raw.get("entry_script"),
            step_limit=int(raw.get("step_limit", 0)),
            cost_limit=float(raw.get("cost_limit", 0.0)),
            parameters=params,
            path=folder,
            agent_config=dict(raw.get("agent") or {}),
            model_config=dict(raw.get("model") or {}),
            env_config=dict(raw.get("env") or {}),
            tools_config=dict(raw.get("tools") or {}),
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> SubAgentDescriptor | None:
        """Look up a subagent by name."""
        return self.subagents.get(name)

    def list_names(self) -> list[str]:
        """Return sorted list of registered subagent names."""
        return sorted(self.subagents.keys())

    # ------------------------------------------------------------------
    # Programmatic registration
    # ------------------------------------------------------------------

    def register_from_dict(self, definition: dict[str, Any]) -> SubAgentDescriptor:
        """Register a subagent from a dictionary (same schema as SUBAGENT.yaml).

        Useful when the top agent programmatically creates a custom subagent
        at runtime without writing a YAML file to disk.

        Returns the created descriptor.
        """
        name = definition.get("name")
        if not name or not definition.get("description"):
            msg = "Definition must include 'name' and 'description'"
            raise ValueError(msg)
        if name in self.subagents:
            msg = f"Subagent {name!r} already registered"
            raise ValueError(msg)

        params = []
        for p in definition.get("parameters") or []:
            params.append(
                SubAgentParameter(
                    name=p["name"],
                    type=p.get("type", "string"),
                    description=p.get("description", ""),
                    required=p.get("required", False),
                )
            )

        desc = SubAgentDescriptor(
            name=name,
            description=str(definition["description"]).strip(),
            execution_mode=definition.get("execution_mode", "inprocess"),
            entry_script=definition.get("entry_script"),
            step_limit=int(definition.get("step_limit", 0)),
            cost_limit=float(definition.get("cost_limit", 0.0)),
            parameters=params,
            path=Path("."),
            agent_config=dict(definition.get("agent") or {}),
            model_config=dict(definition.get("model") or {}),
            env_config=dict(definition.get("env") or {}),
            tools_config=dict(definition.get("tools") or {}),
        )
        self.subagents[name] = desc
        logger.info("Programmatically registered subagent: %s", name)
        return desc

    def create_subagent(
        self,
        name: str,
        description: str,
        *,
        execution_mode: str = "inprocess",
        agent_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        env_config: dict[str, Any] | None = None,
        parameters: list[dict[str, Any]] | None = None,
        persist: bool = False,
    ) -> SubAgentDescriptor:
        """Create and register a new subagent, optionally persisting to disk.

        Args:
            name: Unique subagent identifier.
            description: When to use this subagent.
            execution_mode: "inprocess" or "subprocess".
            agent_config: Agent section (system_template, etc.).
            model_config: Model section (model_class, model_name, etc.).
            env_config: Environment section.
            parameters: Parameter definitions.
            persist: If True, write ``subagents/<name>/SUBAGENT.yaml`` to disk.

        Returns:
            The created and registered SubAgentDescriptor.
        """
        definition: dict[str, Any] = {
            "name": name,
            "description": description,
            "execution_mode": execution_mode,
        }
        if agent_config:
            definition["agent"] = agent_config
        if model_config:
            definition["model"] = model_config
        if env_config:
            definition["env"] = env_config
        if parameters:
            definition["parameters"] = parameters

        desc = self.register_from_dict(definition)

        if persist:
            subagent_dir = self._subagents_dir / name
            subagent_dir.mkdir(parents=True, exist_ok=True)
            yaml_path = subagent_dir / self.DEFINITION_FILE
            yaml_path.write_text(
                yaml.dump(definition, default_flow_style=False, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            desc.path = subagent_dir
            logger.info("Persisted subagent %r to %s", name, yaml_path)

        return desc

    # ------------------------------------------------------------------
    # System prompt injection
    # ------------------------------------------------------------------

    def build_system_prompt_section(self) -> str:
        """Build an XML block listing available subagents for the system prompt."""
        if not self.subagents:
            return ""

        lines = ["\n<available_subagents>"]
        for name in sorted(self.subagents):
            desc = self.subagents[name]
            lines.append(
                f"  <subagent>\n"
                f"    <name>{desc.name}</name>\n"
                f"    <description>{desc.description}</description>\n"
                f"    <execution_mode>{desc.execution_mode}</execution_mode>\n"
                f"  </subagent>"
            )
        lines.append("</available_subagents>")
        lines.append(
            "\nYou can delegate tasks to the above subagents by calling the `sub_agent` tool "
            "with the `agent_name` parameter set to the subagent's name. "
            "You can also spawn ad-hoc child agents by calling `sub_agent` without `agent_name`."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Task-generator catalog
    # ------------------------------------------------------------------

    def build_taskgen_catalog(self) -> str:
        """Build a Markdown catalog of available subagents for the task generator prompt.

        This is injected into the ``TASKGEN_SYSTEM_PROMPT`` so the planning
        LLM knows which agent types it can assign tasks to.
        """
        if not self.subagents:
            return (
                "1. **strategy_agent** (default) -- General-purpose LLM-guided agent "
                "with bash, editor, save_and_test, submit, profile_kernel, "
                "baseline_metrics, and strategy_manager.\n"
            )

        lines: list[str] = []
        for i, name in enumerate(sorted(self.subagents), 1):
            desc = self.subagents[name]
            tool_profile = desc.agent_config.get("tool_profile", "swe")
            lines.append(
                f"{i}. **{desc.name}** -- {desc.description} "
                f"(execution_mode: {desc.execution_mode}, tool_profile: {tool_profile})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Language matching
    # ------------------------------------------------------------------

    def match_language(self, kernel_path: str) -> str | None:
        """Auto-select the best subagent for a kernel based on file extension and markers.

        Checks each subagent's ``language_match`` config (if present) against
        the kernel file.  Returns the name of the best-matching subagent, or
        ``None`` if no subagent has a ``language_match`` section.
        """
        from pathlib import Path as _Path

        kp = _Path(kernel_path)
        ext = kp.suffix.lower()

        best_name: str | None = None
        best_score: float = -1.0

        for name, desc in self.subagents.items():
            lang_match = desc.agent_config.get("language_match") or {}
            if not lang_match:
                continue

            extensions = lang_match.get("extensions", [])
            if ext not in extensions:
                continue

            score = 1.0
            markers = lang_match.get("markers", [])
            if markers:
                try:
                    content = kp.read_text(errors="ignore")[:8192]
                    matched = sum(1 for m in markers if m in content)
                    if matched == 0:
                        continue
                    score += matched * lang_match.get("confidence_boost", 0.1)
                except OSError:
                    pass

            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    # ------------------------------------------------------------------
    # System prompt loading
    # ------------------------------------------------------------------

    def load_system_prompt(self, desc: SubAgentDescriptor) -> str | None:
        """Load the system prompt for a subagent descriptor.

        Checks (in order):
        1. ``system_template_file`` in agent_config — reads from a .md file
           relative to the subagent's directory.
        2. ``system_template`` in agent_config — returns the inline string.
        3. Returns ``None`` if neither is set.
        """
        agent_cfg = desc.agent_config or {}

        template_file = agent_cfg.get("system_template_file")
        if template_file:
            prompt_path = desc.path / template_file
            if prompt_path.exists():
                return prompt_path.read_text(encoding="utf-8")
            logger.warning(
                "system_template_file %r not found at %s for subagent %r",
                template_file,
                prompt_path,
                desc.name,
            )

        template = agent_cfg.get("system_template")
        if template:
            return str(template)

        return None

    # ------------------------------------------------------------------
    # Tool schema generation
    # ------------------------------------------------------------------

    def build_tool_schema(self) -> dict[str, Any]:
        """Build the JSON tool schema for the ``sub_agent`` tool.

        The schema includes the base sub_agent parameters plus an ``agent_name``
        enum listing all registered subagents.
        """
        agent_names = self.list_names()

        schema: dict[str, Any] = {
            "name": "sub_agent",
            "type": "function",
            "description": (
                "Spawn a child agent to perform a focused sub-task. "
                "Set agent_name to use a pre-defined subagent, or omit it to create an ad-hoc child agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The sub-task description for the child agent.",
                    },
                    "agent_name": {
                        "type": "string",
                        "description": (
                            f"Name of a registered subagent to use. Available: {', '.join(agent_names)}."
                            if agent_names
                            else "No subagents registered."
                        ),
                    },
                    "step_limit": {
                        "type": "integer",
                        "description": "Max steps for the child agent (default 150, minimum 150).",
                    },
                    "cost_limit": {
                        "type": "number",
                        "description": "Max cost in dollars for the child agent (default 0.0 = disabled).",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Override the child agent's system prompt (optional, ignored for registered subagents).",
                    },
                },
                "required": ["task"],
            },
        }

        if agent_names:
            schema["parameters"]["properties"]["agent_name"]["enum"] = agent_names

        return schema
