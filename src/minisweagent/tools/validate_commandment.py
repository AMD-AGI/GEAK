"""Programmatic validation for COMMANDMENT.md files.

COMMANDMENT.md is the single source of truth for kernel evaluation.  It
must contain exactly five section headers:
  ## SETUP          -- environment preparation
  ## CORRECTNESS    -- correctness gate
  ## PROFILE        -- deep hardware analysis (Metrix on PROFILE_SHAPES)
  ## BENCHMARK      -- wall-clock latency (HARNESS_SHAPES, iterative feedback)
  ## FULL_BENCHMARK -- wall-clock latency (all shapes, final evaluation)

Any other header (e.g., ``## Test Command``) is flagged as an error.

Additionally, ``rocprofv3`` uses ``os.execvpe()`` to run commands, which
means shell built-ins like ``cd``, ``source``, and ``export`` cannot be
used as command prefixes -- they will crash with ``FileNotFoundError``.
Inline environment variable prefixes (``VAR=value command ...``) also
crash ``rocprofv3`` for the same reason.

This module provides ``validate_commandment()`` which can be called:
1. As a standalone tool by the agent
2. Automatically by the ``str_replace_editor`` hook when a COMMANDMENT.md is written
"""

from __future__ import annotations

import re

REQUIRED_SECTIONS = {"SETUP", "CORRECTNESS", "PROFILE", "BENCHMARK", "FULL_BENCHMARK"}
SHELL_BUILTINS = {"cd", "source", "export", "alias", "ulimit", "pushd", "popd"}


def _extract_section_text(content: str, section: str) -> str | None:
    """Return the raw text of a ``## <section>`` block, or *None* if absent."""
    lines: list[str] = []
    in_section = False
    for raw_line in content.splitlines():
        header = re.match(r"^##\s+(\w+)", raw_line.strip())
        if header:
            if header.group(1) == section:
                in_section = True
                continue
            elif in_section:
                break
            continue
        if in_section:
            lines.append(raw_line)
    if not lines:
        return None
    return "\n".join(lines)


_REQUIRED_HARNESS_FLAGS = ("--profile", "--correctness", "--benchmark", "--full-benchmark")


def _validate_harness_flags(harness_path: str) -> list[str]:
    """Check that a harness script defines the required CLI flags.

    Returns a (possibly empty) list of warning strings.  This is a
    secondary safety net; the primary validation lives in the preprocessor
    where the harness can be re-generated via the UnitTestAgent retry loop.
    """
    from pathlib import Path as _Path

    harness = _Path(harness_path)
    warnings: list[str] = []
    if not harness.is_file():
        warnings.append(f"Harness file not found: {harness}")
        return warnings

    source = harness.read_text()
    has_parser = (
        "argparse" in source
        or "ArgumentParser" in source
        or "click" in source
        or "typer" in source
    )
    if not has_parser:
        warnings.append(
            f"Harness '{harness.name}' does not use argparse/click/typer -- "
            "CLI flags referenced in COMMANDMENT will be silently ignored"
        )
    for flag in _REQUIRED_HARNESS_FLAGS:
        if flag not in source:
            warnings.append(
                f"Harness '{harness.name}' does not define '{flag}' flag "
                "but COMMANDMENT references it"
            )
    return warnings


def validate_commandment(content: str, *, harness_path: str | None = None) -> dict:
    """Validate a COMMANDMENT.md file's content.

    Parameters
    ----------
    content:
        The raw COMMANDMENT.md text.
    harness_path:
        Optional path to the test harness script.  When provided the
        validator performs additional static checks to ensure the harness
        supports the CLI flags used in CORRECTNESS / PROFILE / BENCHMARK /
        FULL_BENCHMARK sections.

    Returns
    -------
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
            f"COMMANDMENT.md MUST contain exactly: ## SETUP, ## CORRECTNESS, ## PROFILE, ## BENCHMARK, ## FULL_BENCHMARK."
        )

    unknown = found_sections - REQUIRED_SECTIONS
    if unknown:
        errors.append(
            f"Unknown section(s): {', '.join(f'## {s}' for s in sorted(unknown))}. "
            f"Only ## SETUP, ## CORRECTNESS, ## PROFILE, ## BENCHMARK, ## FULL_BENCHMARK are recognized."
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
                    f'bash -c "{stripped}"'
                )

        # Check for inline env var prefixes (VAR=value command ...)
        # These work in a shell but NOT with os.execvpe() used by rocprofv3
        env_prefix = re.match(r"^(\w+=\S+)\s+(.+)", stripped)
        if env_prefix and current_section in ("CORRECTNESS", "PROFILE", "BENCHMARK", "FULL_BENCHMARK"):
            var_assign = env_prefix.group(1)
            errors.append(
                f"Command uses inline env var prefix '{var_assign}' in "
                f"## {current_section}: {stripped!r}. "
                f"rocprofv3 uses os.execvpe() and treats '{var_assign}' as "
                f"the executable name, causing FileNotFoundError. "
                f"Set the variable in ## SETUP (via a wrapper script) or "
                f'use: bash -c "{stripped}"'
            )

    # --- Check that SETUP configures PYTHONPATH ---
    setup_text = _extract_section_text(content, "SETUP")
    if setup_text is not None and "PYTHONPATH" not in setup_text:
        errors.append(
            "## SETUP does not configure PYTHONPATH. "
            "The COMMANDMENT's SETUP section MUST set PYTHONPATH so that "
            "test harnesses can import the package under optimization. "
            "All agents execute COMMANDMENT commands verbatim; without "
            "PYTHONPATH the imports will fail."
        )

    # --- Check for common mistakes ---
    if "HIP_VISIBLE_DEVICES" in content and not re.search(r"\$\{?HIP_VISIBLE_DEVICES", content):
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

    # --- Validate harness supports the flags used in COMMANDMENT ---
    if harness_path:
        warnings.extend(_validate_harness_flags(harness_path))

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Validate one or more COMMANDMENT.md files from the command line."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m minisweagent.tools.validate_commandment <file> [<file> ...]", file=sys.stderr)
        sys.exit(2)

    any_invalid = False
    for path_str in sys.argv[1:]:
        from pathlib import Path

        path = Path(path_str)
        if not path.is_file():
            print(f"ERROR: {path_str}: file not found", file=sys.stderr)
            any_invalid = True
            continue

        content = path.read_text()
        result = validate_commandment(content)
        message = format_validation_message(result)
        print(f"--- {path_str} ---")
        print(message)

        if not result["valid"]:
            any_invalid = True

    sys.exit(1 if any_invalid else 0)


if __name__ == "__main__":
    main()
