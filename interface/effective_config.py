"""Build a deterministic effective launch configuration from a GEAK handoff.

Schema-v2 handoffs contain overlapping records of the launch command.  This
module reconciles those records without depending on ``run_e2e.py`` so callers
can inspect (and persist) the exact command before launching a server.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shlex
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Union

import yaml


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

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-serialisable representation."""

        return copy.deepcopy(asdict(self))


@dataclass(frozen=True)
class _Flag:
    name: str
    value: Optional[str]


def _protect_bare_json(text: str) -> tuple[str, dict[str, str]]:
    """Replace balanced bare JSON values before POSIX ``shlex`` removes quotes."""

    protected: dict[str, str] = {}
    out: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char not in "[{" or (i and not (text[i - 1].isspace() or text[i - 1] == "=")):
            out.append(char)
            i += 1
            continue

        opening = char
        closing = "}" if opening == "{" else "]"
        depth = 0
        quoted = False
        escaped = False
        end = i
        while end < len(text):
            current = text[end]
            if quoted:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    quoted = False
            elif current == '"':
                quoted = True
            elif current == opening:
                depth += 1
            elif current == closing:
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        if depth:
            out.append(char)
            i += 1
            continue

        candidate = text[i:end]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            out.append(char)
            i += 1
            continue
        token = f"__GEAK_JSON_{len(protected)}__"
        protected[token] = json.dumps(
            parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        out.append(token)
        i = end
    return "".join(out), protected


def _shell_tokens(text: Any) -> list[str]:
    """Split shell text while retaining and canonicalising bare JSON values."""

    rendered = str(text or "").strip()
    if not rendered:
        return []
    protected_text, protected = _protect_bare_json(rendered)
    tokens = shlex.split(protected_text, posix=True)
    for index, token in enumerate(tokens):
        for marker, value in protected.items():
            if marker in token:
                token = token.replace(marker, value)
        tokens[index] = token
    return tokens


def _looks_like_flag(token: str) -> bool:
    if token == "-" or not token.startswith("-"):
        return False
    try:
        float(token)
    except ValueError:
        return True
    return False


def _parse_flags(text: Any) -> list[_Flag]:
    """Parse long/unknown flags, equals forms, booleans, and JSON values."""

    tokens = _shell_tokens(text)
    flags: list[_Flag] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not _looks_like_flag(token):
            raise ValueError(f"server argument has no flag: {token!r}")
        if "=" in token:
            name, value = token.split("=", 1)
            flags.append(_Flag(name, _canonical_value(value)))
            index += 1
            continue
        value: Optional[str] = None
        if index + 1 < len(tokens) and not _looks_like_flag(tokens[index + 1]):
            value = _canonical_value(tokens[index + 1])
            index += 1
        flags.append(_Flag(token, value))
        index += 1
    return flags


def _canonical_value(value: str) -> str:
    stripped = value.strip()
    if stripped[:1] in "[{" and stripped[-1:] in "]}":
        try:
            return json.dumps(
                json.loads(stripped),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except json.JSONDecodeError:
            pass
    return value


def _flag_map(text: Any) -> "OrderedDict[str, _Flag]":
    result: "OrderedDict[str, _Flag]" = OrderedDict()
    for flag in _parse_flags(text):
        result[flag.name] = flag
    return result


def _render_flags(flags: Iterable[_Flag]) -> str:
    tokens: list[str] = []
    for flag in flags:
        tokens.append(flag.name)
        if flag.value is not None:
            tokens.append(flag.value)
    return shlex.join(tokens)


def _parse_env(value: Any) -> "OrderedDict[str, str]":
    if value is None or value == "":
        return OrderedDict()
    if isinstance(value, Mapping):
        return OrderedDict((str(key), str(item)) for key, item in value.items())
    result: "OrderedDict[str, str]" = OrderedDict()
    for token in _shell_tokens(value):
        key, separator, item = token.partition("=")
        if not separator or not key:
            raise ValueError(f"environment entry must be KEY=VALUE: {token!r}")
        result[key] = item
    return result


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

    For schema v2 the precedence is recipe, then ``server_launch_flags``, then
    the reconciled current-best delta.  Schema v1 remains a legacy pass-through:
    only its top-level accepted values are represented.
    """

    data = _load_handoff(handoff)
    schema_version = int(data.get("schema_version", 1) or 1)
    baseline = data.get("baseline_env_spec") or {}
    baseline_config = baseline.get("config") or {}
    legacy_server_args: Optional[str] = None

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

        recipe_flags = _flag_map(recipe_args)
        launch_flags = _flag_map(baseline_config.get("server_launch_flags", ""))
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
        _merge_layer(
            final_flags,
            delta_flags,
            lower_source=flag_sources,
            source="current_best_delta",
            kind="server_flag",
            conflicts=conflicts,
        )

        recipe_env = OrderedDict(
            (key, value)
            for key, value in raw_recipe_env.items()
            if key not in _ALL_RECIPE_ARG_ENVS
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
    }
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
    )


build_effective_config = resolve_effective_config


__all__ = ["EffectiveConfig", "build_effective_config", "resolve_effective_config"]
