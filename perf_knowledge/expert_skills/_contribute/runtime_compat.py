#!/usr/bin/env python3
"""Validate and probe machine-readable expert-skill runtime contracts."""

import argparse
import importlib
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

import yaml
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILLS_DIR = os.path.join(ROOT, "skills")
FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)

SUPPORTED_PROVIDERS = {
    "aiter_vendored_flydsl",
    "standalone_source_flydsl",
}
SUPPORTED_PROFILE_STATUSES = {
    "validated",
    "revalidation_required",
    "stale",
}
SUPPORTED_PROVISIONING_POLICIES = {
    "reuse_only",
    "build_isolated",
}


def _string_list(value: Any, field: str, errors: List[str]) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        errors.append(f"{field} must be a list of non-empty strings")
        return []
    return value


def _has_future_upper_bound(specifiers: SpecifierSet) -> bool:
    operators = {item.operator for item in specifiers}
    return bool(operators & {"<", "<=", "==", "===", "~="})


def _is_module_or_child(name: str, namespace: str) -> bool:
    return name == namespace or name.startswith(namespace + ".")


def _validate_symbols(symbols: List[str], field: str, errors: List[str]) -> List[str]:
    modules = []
    for symbol in symbols:
        if ":" not in symbol or symbol.startswith(":") or symbol.endswith(":"):
            errors.append(f"{field} entry {symbol!r} must use module:attribute")
            continue
        modules.append(symbol.split(":", 1)[0])
    return modules


def _validate_provider_capabilities(
    provider: str,
    capability_modules: List[str],
    field: str,
    errors: List[str],
) -> None:
    if provider == "aiter_vendored_flydsl":
        if not any(
            _is_module_or_child(name, "aiter.ops.flydsl") for name in capability_modules
        ):
            errors.append(
                f"{field}: aiter_vendored_flydsl requires an "
                "aiter.ops.flydsl import capability"
            )
        if any(_is_module_or_child(name, "kernels") for name in capability_modules):
            errors.append(
                f"{field}: aiter_vendored_flydsl must not require standalone "
                "kernels.* imports or symbols"
            )
    elif provider == "standalone_source_flydsl":
        if not any(_is_module_or_child(name, "kernels") for name in capability_modules):
            errors.append(
                f"{field}: standalone_source_flydsl requires a standalone "
                "kernels.* import capability"
            )
        if any(
            _is_module_or_child(name, "aiter.ops.flydsl") for name in capability_modules
        ):
            errors.append(
                f"{field}: standalone_source_flydsl must not require "
                "aiter.ops.flydsl imports or symbols"
            )


def validate_runtime_contract(runtime: Dict[str, Any]) -> List[str]:
    """Return human-readable schema errors; an empty list means statically valid."""
    errors: List[str] = []
    if not isinstance(runtime, dict) or not runtime:
        return ["runtime must be a non-empty mapping"]

    if runtime.get("language") != "flydsl":
        errors.append("runtime.language must be 'flydsl'")

    provider = runtime.get("provider")
    if provider not in SUPPORTED_PROVIDERS:
        errors.append(
            "runtime.provider must be one of " + ", ".join(sorted(SUPPORTED_PROVIDERS))
        )

    required_imports = _string_list(
        runtime.get("required_imports"), "runtime.required_imports", errors
    )
    required_symbols = _string_list(
        runtime.get("required_symbols"), "runtime.required_symbols", errors
    )
    required_symbol_modules = _validate_symbols(
        required_symbols, "runtime.required_symbols", errors
    )

    profiles = runtime.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        errors.append("runtime.profiles must be a non-empty list")
        profiles = []

    seen_names = set()
    for index, profile in enumerate(profiles):
        prefix = f"runtime.profiles[{index}]"
        if not isinstance(profile, dict):
            errors.append(f"{prefix} must be a mapping")
            continue
        name = profile.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{prefix}.name must be a non-empty string")
        elif name in seen_names:
            errors.append(f"{prefix}.name duplicates {name!r}")
        else:
            seen_names.add(name)

        specifier = profile.get("specifier")
        if not isinstance(specifier, str) or not specifier:
            errors.append(f"{prefix}.specifier must be a non-empty string")
        else:
            try:
                parsed = SpecifierSet(specifier)
                if not _has_future_upper_bound(parsed):
                    errors.append(
                        f"{prefix}.specifier must include an upper bound for future API safety"
                    )
            except InvalidSpecifier as exc:
                errors.append(f"{prefix}.specifier is invalid: {exc}")

        status = profile.get("validation_status")
        if status not in SUPPORTED_PROFILE_STATUSES:
            errors.append(
                f"{prefix}.validation_status must be one of "
                + ", ".join(sorted(SUPPORTED_PROFILE_STATUSES))
            )
        profile_imports = _string_list(
            profile.get("required_imports"), f"{prefix}.required_imports", errors
        )
        symbols = _string_list(
            profile.get("required_symbols"), f"{prefix}.required_symbols", errors
        )
        profile_symbol_modules = _validate_symbols(
            symbols, f"{prefix}.required_symbols", errors
        )
        _validate_provider_capabilities(
            provider,
            required_imports
            + required_symbol_modules
            + profile_imports
            + profile_symbol_modules,
            prefix,
            errors,
        )

    provisioning = runtime.get("provisioning")
    if not isinstance(provisioning, dict):
        errors.append("runtime.provisioning must be a mapping")
    elif provisioning.get("policy") not in SUPPORTED_PROVISIONING_POLICIES:
        errors.append(
            "runtime.provisioning.policy must be one of "
            + ", ".join(sorted(SUPPORTED_PROVISIONING_POLICIES))
        )
    return errors


def select_runtime_profile(
    runtime: Dict[str, Any], installed_version: str
) -> Optional[Dict[str, Any]]:
    """Select the single profile matching an installed FlyDSL version."""
    try:
        version = Version(installed_version)
    except InvalidVersion:
        return None
    matches = []
    for profile in runtime.get("profiles") or []:
        try:
            specifier = SpecifierSet(str(profile.get("specifier", "")))
        except InvalidSpecifier:
            continue
        allow_prereleases = specifier.prereleases is True
        if specifier.contains(version, prereleases=allow_prereleases):
            matches.append(profile)
    return matches[0] if len(matches) == 1 else None


def discover_flydsl_version(
    importer: Callable[[str], Any] = importlib.import_module,
) -> str:
    """Return the version of the flydsl module that the process actually imports."""
    try:
        module = importer("flydsl")
    except Exception as exc:
        raise RuntimeError("cannot import the effective flydsl module") from exc
    version = getattr(module, "__version__", "")
    if not version:
        raise RuntimeError(
            "the effective flydsl module has no __version__; refusing distribution metadata fallback"
        )
    return str(version)


def _module_origin(module: Any) -> str:
    return str(getattr(module, "__file__", "") or "")


def probe_runtime(
    runtime: Dict[str, Any],
    version_getter: Callable[[], str] = discover_flydsl_version,
    importer: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, Any]:
    """Probe one runtime contract without installing or mutating packages."""
    errors = validate_runtime_contract(runtime)
    result: Dict[str, Any] = {
        "compatible": False,
        "provider": runtime.get("provider", ""),
        "language": runtime.get("language", ""),
        "version": "",
        "profile": "",
        "validation_status": "",
        "module_origins": {},
        "errors": list(errors),
    }
    if errors:
        return result

    try:
        version = str(version_getter())
    except Exception as exc:
        result["errors"].append(f"cannot determine flydsl version: {type(exc).__name__}: {exc}")
        return result
    result["version"] = version

    profile = select_runtime_profile(runtime, version)
    if profile is None:
        result["errors"].append(
            f"flydsl {version} does not match exactly one declared runtime profile"
        )
        return result
    result["profile"] = profile["name"]
    result["validation_status"] = profile["validation_status"]

    modules: Dict[str, Any] = {}
    required_imports = list(runtime.get("required_imports") or [])
    required_imports.extend(profile.get("required_imports") or [])
    for module_name in dict.fromkeys(required_imports):
        try:
            module = importer(module_name)
            modules[module_name] = module
            result["module_origins"][module_name] = _module_origin(module)
        except Exception as exc:
            result["errors"].append(
                f"cannot import {module_name}: {type(exc).__name__}: {exc}"
            )

    required_symbols = list(runtime.get("required_symbols") or [])
    required_symbols.extend(profile.get("required_symbols") or [])
    for requirement in dict.fromkeys(required_symbols):
        module_name, attribute_path = requirement.split(":", 1)
        try:
            module = modules.get(module_name)
            if module is None:
                module = importer(module_name)
                modules[module_name] = module
                result["module_origins"][module_name] = _module_origin(module)
            value = module
            for part in attribute_path.split("."):
                value = getattr(value, part)
        except Exception as exc:
            result["errors"].append(
                f"missing required symbol {requirement}: {type(exc).__name__}: {exc}"
            )

    result["compatible"] = not result["errors"]
    return result


def load_skill_runtime(skill_id: str) -> Dict[str, Any]:
    path = os.path.join(SKILLS_DIR, skill_id, "skill.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such expert skill: {path}")
    with open(path) as handle:
        text = handle.read()
    match = FM_RE.match(text)
    if not match:
        raise ValueError(f"{path}: no YAML frontmatter")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    runtime = frontmatter.get("runtime")
    if not runtime:
        raise ValueError(f"{path}: no runtime contract")
    return runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill_id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        runtime = load_skill_runtime(args.skill_id)
        result = probe_runtime(runtime)
    except Exception as exc:
        result = {
            "compatible": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("compatible" if result.get("compatible") else "incompatible")
        for key in ("provider", "version", "profile", "validation_status"):
            if result.get(key):
                print(f"{key}: {result[key]}")
        for error in result.get("errors") or []:
            print(f"error: {error}", file=sys.stderr)
    return 0 if result.get("compatible") else 2


if __name__ == "__main__":
    raise SystemExit(main())
