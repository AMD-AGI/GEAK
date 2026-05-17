"""Subagent registry for the v3 preprocess pipeline.

The registry discovers ``SUBAGENT.yaml`` definitions under
``subagents/preprocess/<name>/`` and exposes them as :class:`SubagentSpec`
dataclasses. PR 3 will populate the directory with the actual subagent
bodies (``harness-generator``, ``harness-verifier``, ``speedup-verify``,
``pytorch-to-flydsl``); this PR ships only the discovery / validation
skeleton plus tests.

Design notes
------------

* The schema is deliberately **slimmer** than the existing top-level
  ``subagents/<name>/SUBAGENT.yaml`` (which is mini-swe-agent-shaped:
  nested ``agent``, ``env``, ``model`` blocks, ``system_template_file``,
  etc.). The v3 preprocess subagents are lighter-weight delegations from
  the orchestrator and only need ``name`` / ``description`` /
  ``system_prompt`` to be functional. Unknown YAML keys are preserved
  in :attr:`SubagentSpec.extras` so we can extend the contract later
  without rev-locking the registry.

* The default root is ``<repo>/subagents/preprocess``, resolved
  *relative to the repository* rather than the CWD, so the registry is
  usable from arbitrary callers (CLI, tests, orchestrator) without
  caring about working directory.

* Discovery is best-effort: a folder without a ``SUBAGENT.yaml`` is
  silently skipped (it might be a README-only placeholder, like the
  one this PR ships). Folders **with** a ``SUBAGENT.yaml`` that fails
  validation raise :class:`SubagentSpecError` immediately — silent
  corruption is worse than a loud crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


_REQUIRED_FIELDS = ("name", "description", "system_prompt")
_KNOWN_FIELDS = (*_REQUIRED_FIELDS, "model", "tools", "max_steps")
_SUBAGENT_FILE = "SUBAGENT.yaml"

#: Sentinel value for :attr:`SubagentSpec.max_steps` meaning "unlimited steps".
#: Selected because:
#:
#: * ``0`` is a common "default disabled" pattern in other config surfaces
#:   in this codebase (e.g. ``OptimizationAgent.cost_limit == 0`` disables
#:   the cost gate); reusing ``0`` here would silently disable the step
#:   budget on every YAML that omits the field. We want the ``max_steps``
#:   sentinel to be an *opt-in* signal, not a default.
#: * ``None`` would force the field type to ``int | None`` and ripple
#:   through to every consumer; an in-band ``int`` sentinel keeps the
#:   contract uniform.
#: * ``-1`` is the conventional "unlimited" sentinel in argparse / shell
#:   tooling and is unambiguous (no positive integer means "unlimited").
#:
#: Use :attr:`SubagentSpec.is_unlimited_steps` rather than comparing to
#: this constant directly so call sites stay readable.
UNLIMITED_MAX_STEPS: int = -1


def _default_root() -> Path:
    """Resolve the default discovery root to ``<repo>/subagents/preprocess``.

    The repository root is inferred by walking up from this file until
    we find one with a ``pyproject.toml`` next to a ``subagents/`` dir.
    Falls back to four-levels-up if that lookup fails — matches the
    on-disk layout of GEAK (``src/minisweagent/run/preprocess_v3/registry.py``
    is exactly 4 ``parent``s deep from the repo root).
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "subagents").is_dir():
            return candidate / "subagents" / "preprocess"
    return here.parents[4] / "subagents" / "preprocess"


class SubagentSpecError(ValueError):
    """Raised when a ``SUBAGENT.yaml`` is malformed (missing required key, bad YAML…)."""


@dataclass
class SubagentSpec:
    """In-memory representation of one ``SUBAGENT.yaml``.

    The shape is intentionally minimal — only fields the v3 orchestrator
    actually consumes get first-class attributes. Anything else the YAML
    carries (custom routing hints, future runtime knobs) survives in
    :attr:`extras` so writers don't get punished for being expressive.

    ``max_steps`` accepts the sentinel :data:`UNLIMITED_MAX_STEPS` (``-1``)
    to opt out of the step budget — the v3 ``harness-generator`` subagent
    uses this because legitimate harness generation can take many tool-call
    rounds (read README, install package, inspect tests, write harness,
    iterate against errors) and a hard cap pessimises slow-but-correct
    runs. All other negative values and ``0`` are rejected at parse time
    so the sentinel meaning stays unambiguous.
    """

    name: str
    description: str
    system_prompt: str
    model: str | None = None
    tools: list[str] = field(default_factory=list)
    max_steps: int = 30
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def is_unlimited_steps(self) -> bool:
        """``True`` when :attr:`max_steps` carries the unlimited sentinel."""
        return self.max_steps == UNLIMITED_MAX_STEPS


def _validate_and_build(name_hint: str, data: dict[str, Any], source: Path) -> SubagentSpec:
    """Validate one parsed YAML dict and project it into a :class:`SubagentSpec`."""
    if not isinstance(data, dict):
        raise SubagentSpecError(f"{source}: expected a YAML mapping at the top level, got {type(data).__name__}")

    missing = [k for k in _REQUIRED_FIELDS if k not in data or data[k] in (None, "")]
    if missing:
        raise SubagentSpecError(f"{source}: missing required field(s): {', '.join(missing)}")

    name = str(data["name"]).strip()
    if not name:
        raise SubagentSpecError(f"{source}: 'name' must be a non-empty string")

    if name != name_hint:
        logger.warning(
            "SubagentSpec name %r does not match directory name %r (using %r from YAML)",
            name,
            name_hint,
            name,
        )

    tools = data.get("tools", []) or []
    if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
        raise SubagentSpecError(f"{source}: 'tools' must be a list of strings (got {tools!r})")

    max_steps_raw = data.get("max_steps", 30)
    try:
        max_steps = int(max_steps_raw)
    except (TypeError, ValueError) as exc:
        raise SubagentSpecError(f"{source}: 'max_steps' must be an integer (got {max_steps_raw!r})") from exc

    # ``max_steps`` is either a positive integer step cap or the
    # :data:`UNLIMITED_MAX_STEPS` sentinel. Anything else is a YAML
    # mistake and we surface it loudly so writers get a precise error
    # rather than mysterious behaviour at run time (e.g. ``0`` would
    # otherwise be interpreted as "no budget" in some agent loops, but
    # silently halt before any step in others).
    if max_steps != UNLIMITED_MAX_STEPS and max_steps <= 0:
        raise SubagentSpecError(
            f"{source}: 'max_steps' must be a positive integer or the unlimited sentinel "
            f"{UNLIMITED_MAX_STEPS} (got {max_steps_raw!r})"
        )

    extras = {k: v for k, v in data.items() if k not in _KNOWN_FIELDS}

    return SubagentSpec(
        name=name,
        description=str(data["description"]).strip(),
        system_prompt=str(data["system_prompt"]),
        model=(str(data["model"]).strip() if data.get("model") else None),
        tools=[str(t) for t in tools],
        max_steps=max_steps,
        extras=extras,
    )


class SubagentRegistry:
    """Discovery & lookup for v3 preprocess subagent definitions.

    Lazily walks ``root`` on the first :meth:`discover` call and caches
    the result. Call :meth:`discover` again (e.g. from a test) to force
    a re-scan; the lookup helpers (:meth:`get`, :meth:`names`) populate
    the cache on first use too.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root: Path = Path(root) if root is not None else _default_root()
        self._cache: dict[str, SubagentSpec] | None = None

    def discover(self) -> dict[str, SubagentSpec]:
        """Walk :attr:`root` and load every ``<name>/SUBAGENT.yaml``.

        Returns a fresh dict ``name -> SubagentSpec``. A missing or
        non-directory :attr:`root` yields an empty dict — that's the
        expected state for this PR (the directory exists, but contains
        only a README, no YAMLs yet).

        Raises:
            SubagentSpecError: When a discovered YAML is malformed (missing
                required field, wrong types, syntax error).
        """
        specs: dict[str, SubagentSpec] = {}
        if not self.root.is_dir():
            logger.debug("SubagentRegistry root %s is not a directory; no subagents discovered.", self.root)
            self._cache = specs
            return specs

        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            yaml_path = entry / _SUBAGENT_FILE
            if not yaml_path.is_file():
                continue

            try:
                raw = yaml_path.read_text(encoding="utf-8")
                data = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise SubagentSpecError(f"{yaml_path}: invalid YAML: {exc}") from exc

            spec = _validate_and_build(entry.name, data, yaml_path)
            if spec.name in specs:
                raise SubagentSpecError(
                    f"{yaml_path}: duplicate subagent name {spec.name!r} "
                    f"(already loaded from another folder under {self.root})"
                )
            specs[spec.name] = spec

        self._cache = specs
        return specs

    def _ensure_loaded(self) -> dict[str, SubagentSpec]:
        if self._cache is None:
            self.discover()
        assert self._cache is not None  # for type checkers
        return self._cache

    def get(self, name: str) -> SubagentSpec:
        """Return the spec for ``name`` or raise :class:`KeyError`."""
        specs = self._ensure_loaded()
        try:
            return specs[name]
        except KeyError as exc:
            available = ", ".join(sorted(specs)) or "<none discovered>"
            raise KeyError(f"SubagentRegistry: no subagent named {name!r} (available: {available})") from exc

    def names(self) -> list[str]:
        """Return all discovered subagent names, sorted ascending."""
        return sorted(self._ensure_loaded().keys())


__all__ = [
    "SubagentRegistry",
    "SubagentSpec",
    "SubagentSpecError",
    "UNLIMITED_MAX_STEPS",
]
