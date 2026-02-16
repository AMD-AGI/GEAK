"""Programmatic validation for COMMANDMENT.md files.

OpenEvolve's evaluator only recognizes three section headers:
  ## SETUP
  ## CORRECTNESS
  ## PROFILE

Any other header (e.g., ``## Test Command``, ``## Benchmark``) is silently
ignored, resulting in 0 iterations and ``Metrix: N/A``.

Additionally, ``rocprofv3`` uses ``os.execvpe()`` to run commands, which
means shell built-ins like ``cd``, ``source``, and ``export`` cannot be
used as command prefixes -- they will crash with ``FileNotFoundError``.

This module provides ``validate_commandment()`` which can be called:
1. As a standalone tool by the agent
2. Automatically by the ``str_replace_editor`` hook when a COMMANDMENT.md is written
"""

from __future__ import annotations

import re


REQUIRED_SECTIONS = {"SETUP", "CORRECTNESS", "PROFILE"}
SHELL_BUILTINS = {"cd", "source", "export", "alias", "ulimit", "pushd", "popd"}


def validate_commandment(content: str) -> dict:
    """Validate a COMMANDMENT.md file's content.

    Returns:
        dict with keys:
          - valid (bool): True if no errors found
          - errors (list[str]): Critical issues that will cause OpenEvolve failure
          - warnings (list[str]): Non-critical issues worth noting
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- Check section headers ---
    found_sections: set[str] = set()
    for line in content.splitlines():
        m = re.match(r"^##\s+(\w+)", line.strip())
        if m:
            found_sections.add(m.group(1))

    missing = REQUIRED_SECTIONS - found_sections
    if missing:
        errors.append(
            f"Missing required section(s): {', '.join(f'## {s}' for s in sorted(missing))}. "
            f"COMMANDMENT.md MUST contain exactly: ## SETUP, ## CORRECTNESS, ## PROFILE."
        )

    unknown = found_sections - REQUIRED_SECTIONS
    if unknown:
        errors.append(
            f"Unknown section(s): {', '.join(f'## {s}' for s in sorted(unknown))}. "
            f"These will be SILENTLY IGNORED by OpenEvolve. "
            f"Only ## SETUP, ## CORRECTNESS, ## PROFILE are recognized."
        )

    # --- Check for shell built-ins in commands ---
    in_code_block = False
    current_section = None
    for line in content.splitlines():
        stripped = line.strip()

        # Track section headers
        m = re.match(r"^##\s+(\w+)", stripped)
        if m:
            current_section = m.group(1)
            continue

        # Track code block boundaries
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue

        # Skip markdown headers, comments, and empty lines
        if stripped.startswith("#") or not stripped:
            continue

        # Only check lines inside a recognized section (outside code blocks
        # they are free-form text; inside code blocks they are commands)
        if not current_section or current_section not in REQUIRED_SECTIONS:
            continue

        # Check command lines for shell built-ins
        for builtin in SHELL_BUILTINS:
            if re.match(rf"^{re.escape(builtin)}\s", stripped):
                errors.append(
                    f"Command starts with shell built-in '{builtin}': {stripped!r}. "
                    f"rocprofv3 uses os.execvpe() and cannot execute shell built-ins. "
                    f"Use absolute paths instead, or wrap the command in: "
                    f"bash -c \"{stripped}\""
                )

    # --- Check for common mistakes ---
    if "HIP_VISIBLE_DEVICES" in content and not re.search(r'\$\{?HIP_VISIBLE_DEVICES', content):
        warnings.append(
            "COMMANDMENT.md contains a hardcoded HIP_VISIBLE_DEVICES value. "
            "Consider using $HIP_VISIBLE_DEVICES to inherit from the environment."
        )

    # Check that each section has at least one non-empty command
    current_section = None
    section_has_content: dict[str, bool] = {}
    for line in content.splitlines():
        m = re.match(r"^##\s+(\w+)", line.strip())
        if m:
            current_section = m.group(1)
            section_has_content[current_section] = False
            continue
        if current_section and line.strip() and not line.strip().startswith("#"):
            section_has_content[current_section] = True

    for section in REQUIRED_SECTIONS:
        if section in section_has_content and not section_has_content[section]:
            errors.append(
                f"Section ## {section} exists but contains no commands. "
                f"Each section must have at least one executable command."
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def format_validation_message(result: dict) -> str:
    """Format validation result as a human-readable message for the agent."""
    if result["valid"] and not result["warnings"]:
        return "COMMANDMENT.md validation: OK"

    parts = []
    if result["errors"]:
        parts.append("COMMANDMENT.md VALIDATION ERRORS (must fix):")
        for err in result["errors"]:
            parts.append(f"  ERROR: {err}")

    if result["warnings"]:
        parts.append("COMMANDMENT.md warnings:")
        for warn in result["warnings"]:
            parts.append(f"  WARNING: {warn}")

    return "\n".join(parts)
