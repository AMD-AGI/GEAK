"""Build a deterministic effective launch configuration from a GEAK handoff.

Schema-v2 handoffs contain overlapping records of the launch command.  This
module reconciles those records without depending on ``run_e2e.py`` so callers
can inspect (and persist) the exact command before launching a server.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union

import yaml

from e2e_workflow.scripts.adapters import server_args
from e2e_workflow.scripts.adapters.extra_env import (
    _protect_bare_json as _protect_bare_json,
)
from e2e_workflow.scripts.adapters.extra_env import _shell_tokens, parse_unset_envs
from e2e_workflow.scripts.adapters.server_args import (
    _Flag,
    _flag_map,
    _parse_flags,
    _remove_flags,
    _render_flags,
    resolve_remove_args,
)

_canonical_value = server_args._canonical_value
_looks_like_flag = server_args._looks_like_flag

_RECIPE_ARG_ENVS = {
    "vllm": "EXTRA_VLLM_ARGS",
    "sglang": "EXTRA_SGLANG_ARGS",
}
_ALL_RECIPE_ARG_ENVS = frozenset(
    {"EXTRA_SERVER_ARGS", "EXTRA_VLLM_ARGS", "EXTRA_SGLANG_ARGS"}
)


@dataclass(frozen=True)
class EffectiveConfig:
    """Canonical, auditable serving configuration."""

    final_server_args: str
    final_env: dict[str, str]
    base_overlay_pythonpath: str
    source_snapshots: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    digest: str
    manifest: dict[str, Any]
    unset_envs: tuple[str, ...] = ()
    remove_args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-serialisable representation."""

        return copy.deepcopy(asdict(self))


def _parse_env(value: Any) -> "OrderedDict[str, str]":
    if value is None or value == "":
        return OrderedDict()
    if isinstance(value, Mapping):
        return OrderedDict((str(key), str(item)) for key, item in value.items())
    result: "OrderedDict[str, str]" = OrderedDict()
    # Environment values are opaque strings, including any embedded JSON text.
    for token in _shell_tokens(value, canonicalize_json=False):
        key, separator, item = token.partition("=")
        if not separator or not key:
            raise ValueError(f"environment entry must be KEY=VALUE: {token!r}")
        result[key] = item
    return result


def resolve_unset_envs(names: Any, *environments: Any) -> tuple[str, ...]:
    """Keep explicit removals that current assignments have not re-enabled."""
    unsets = set(parse_unset_envs(names))
    for env in environments:
        unsets.difference_update(_parse_env(env))
    return tuple(sorted(unsets))


def _reconcile(
    extra: MutableMapping[str, Any],
    accepted: MutableMapping[str, Any],
    *,
    kind: str,
) -> "OrderedDict[str, Any]":
    result: "OrderedDict[str, Any]" = OrderedDict(extra)
    for key, value in accepted.items():
        if key in result and result[key] != value:
            left = result[key]
            raise ValueError(
                f"conflicting {kind} {key!r} between extra ({left!r}) "
                f"and accepted ({value!r})"
            )
        result[key] = value
    return result


def _merge_layer(
    target: "OrderedDict[str, Any]",
    incoming: Mapping[str, Any],
    *,
    lower_source: dict[str, str],
    source: str,
    kind: str,
    conflicts: list[dict[str, Any]],
) -> None:
    for key, value in incoming.items():
        if key in target and target[key] != value:
            conflicts.append(
                {
                    "kind": kind,
                    "key": key,
                    "lower_source": lower_source[key],
                    "lower_value": (
                        target[key].value if isinstance(target[key], _Flag) else target[key]
                    ),
                    "higher_source": source,
                    "higher_value": value.value if isinstance(value, _Flag) else value,
                }
            )
        target[key] = value
        lower_source[key] = source


def _find_envs(node: Any) -> dict[str, Any]:
    if not isinstance(node, Mapping):
        return {}
    envs = node.get("envs")
    if isinstance(envs, Mapping):
        return dict(envs)
    for value in node.values():
        found = _find_envs(value)
        if found:
            return found
    return {}


def _recipe_envs(path: Any) -> "OrderedDict[str, str]":
    if not path:
        return OrderedDict()
    recipe_path = Path(str(path))
    try:
        document = yaml.load(recipe_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse launch recipe {recipe_path}: {exc}") from exc
    return OrderedDict(
        (str(key), str(value))
        for key, value in _find_envs(document).items()
        if not isinstance(value, (Mapping, list))
    )


def _load_handoff(handoff: Union[Mapping[str, Any], str, Path]) -> dict[str, Any]:
    if isinstance(handoff, Mapping):
        return copy.deepcopy(dict(handoff))
    path = Path(handoff)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse handoff {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("handoff must be a JSON object")
    return loaded


def resolve_effective_config(
    handoff: Union[Mapping[str, Any], str, Path],
) -> EffectiveConfig:
    """Resolve a handoff into one canonical argument string and environment.

    For schema v2 a nonempty ``server_launch_flags`` is the complete argument
    base; the recipe supplies arguments only when that snapshot is unavailable.
    Explicit ``args_mode=replace`` also suppresses recipe fallback when the
    observed snapshot is unavailable. Removals precede current assignments.
    Schema v1 is a legacy pass-through.
    """

    data = _load_handoff(handoff)
    schema_version = int(data.get("schema_version", 1) or 1)
    baseline = data.get("baseline_env_spec") or {}
    baseline_config = baseline.get("config") or {}
    legacy_server_args: Optional[str] = None
    unset_envs: tuple[str, ...] = ()
    remove_args: tuple[str, ...] = ()

    if schema_version < 2:
        # Do not canonicalise or merge old handoffs: legacy consumers forwarded
        # this string byte-for-byte, including its quoting and equals spelling.
        legacy_server_args = str(data.get("accepted_flags") or "")
        final_flags: "OrderedDict[str, _Flag]" = OrderedDict()
        final_env = _parse_env(data.get("accepted_env", ""))
        snapshots: list[dict[str, Any]] = []
        overlay = ""
        conflicts: list[dict[str, Any]] = []
        recipe_path = ""
    else:
        framework = str(data.get("framework") or "").strip().lower()
        recipe_path = str(
            data.get("launch_recipe") or baseline.get("launch_recipe") or ""
        )
        raw_recipe_env = _recipe_envs(recipe_path)
        recipe_args = raw_recipe_env.get("EXTRA_SERVER_ARGS", "")
        backend_arg_name = _RECIPE_ARG_ENVS.get(framework)
        if backend_arg_name:
            backend_args = raw_recipe_env.get(backend_arg_name, "")
            recipe_args = " ".join(part for part in (recipe_args, backend_args) if part)

        launch_flags = _flag_map(baseline_config.get("server_launch_flags", ""))
        # A complete argv records removals by absence. Merging recipe-only flags
        # would restore options the caller removed from its best configuration.
        # Empty launch flags mean unavailable evidence in existing handoffs.
        mode = str(baseline_config.get("args_mode") or "append").strip().lower()
        if mode not in ("append", "replace"):
            raise ValueError(f"unsupported args_mode: {mode!r}")
        recipe_flags = (
            _flag_map(recipe_args) if not launch_flags and mode != "replace" else OrderedDict()
        )
        extra_flags = _flag_map(baseline_config.get("extra_server_args", ""))
        accepted_flags = _flag_map(data.get("accepted_flags", ""))
        delta_flags = _reconcile(
            extra_flags, accepted_flags, kind="server flag"
        )

        conflicts = []
        final_flags = OrderedDict()
        flag_sources: dict[str, str] = {}
        _merge_layer(
            final_flags,
            recipe_flags,
            lower_source=flag_sources,
            source="launch_recipe",
            kind="server_flag",
            conflicts=conflicts,
        )
        _merge_layer(
            final_flags,
            launch_flags,
            lower_source=flag_sources,
            source="server_launch_flags",
            kind="server_flag",
            conflicts=conflicts,
        )
        requested_removals = resolve_remove_args(baseline_config.get("remove_args"))
        _remove_flags(final_flags, requested_removals)
        remove_args = resolve_remove_args(requested_removals, _render_flags(delta_flags.values()))
        for spec in sorted(set(requested_removals) - set(remove_args)):
            flag = _parse_flags(spec)[0]
            conflicts.append({
                "kind": "server_flag", "key": flag.name,
                "lower_source": "remove_args", "lower_value": spec,
                "higher_source": "current_best_delta", "higher_value": delta_flags[flag.name].value,
            })
        _merge_layer(
            final_flags,
            delta_flags,
            lower_source=flag_sources,
            source="current_best_delta",
            kind="server_flag",
            conflicts=conflicts,
        )

        removed_envs = parse_unset_envs(baseline_config.get("unset_envs"))
        recipe_env = OrderedDict(
            (key, value)
            for key, value in raw_recipe_env.items()
            if key not in _ALL_RECIPE_ARG_ENVS and key not in removed_envs
        )
        extra_env = _parse_env(baseline_config.get("extra_envs", {}))
        accepted_env = _parse_env(data.get("accepted_env", ""))
        delta_env = _reconcile(extra_env, accepted_env, kind="environment variable")
        final_env = OrderedDict()
        env_sources: dict[str, str] = {}
        _merge_layer(
            final_env,
            recipe_env,
            lower_source=env_sources,
            source="launch_recipe",
            kind="environment",
            conflicts=conflicts,
        )
        _merge_layer(
            final_env,
            delta_env,
            lower_source=env_sources,
            source="current_best_delta",
            kind="environment",
            conflicts=conflicts,
        )
        # Absence in the map alone cannot clear inherited launcher settings.
        # Explicit current assignments may re-add a previously removed name.
        unset_envs = resolve_unset_envs(removed_envs, final_env)
        snapshots = copy.deepcopy(list(baseline.get("source_snapshots") or []))
        overlay = str(baseline.get("overlay_pythonpath") or "")

    final_server_args = (
        legacy_server_args
        if legacy_server_args is not None
        else _render_flags(final_flags.values())
    )
    final_env_dict = dict(final_env)
    manifest = {
        "schema_version": schema_version,
        "framework": str(data.get("framework") or ""),
        "launch_recipe": recipe_path,
        "final_server_args": final_server_args,
        "final_env": final_env_dict,
        "base_overlay_pythonpath": overlay,
        "source_snapshots": snapshots,
        "conflicts": conflicts,
        "remove_args": list(remove_args),
    }
    if unset_envs:
        manifest["unset_envs"] = list(unset_envs)
    digest = hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    return EffectiveConfig(
        final_server_args=final_server_args,
        final_env=final_env_dict,
        base_overlay_pythonpath=overlay,
        source_snapshots=snapshots,
        conflicts=conflicts,
        digest=digest,
        manifest=manifest,
        unset_envs=unset_envs,
        remove_args=remove_args,
    )


build_effective_config = resolve_effective_config


__all__ = ["EffectiveConfig", "build_effective_config", "resolve_effective_config"]
