"""Parse recorded launch-recipe environments without applying launch policy.

PyYAML preserves scalar spellings. The indentation scanner is only a fallback
for hosts without PyYAML; callers decide whether that degraded path is allowed.
"""
from __future__ import annotations

from typing import Any


class _RecipeYAMLError(Exception):
    """Invalid YAML; unlike missing PyYAML, this must not use the fallback."""


def _find_envs_map(node: Any) -> dict | None:
    """Find the first nonempty ``envs`` map, searching depth-first.

    Return ``{}`` if only empty maps exist, or ``None`` if none exists. Empty
    metadata maps must not hide a later launch environment.
    """
    saw_empty = False
    if isinstance(node, dict):
        cand = node.get("envs")
        saw_empty = isinstance(cand, dict)
        if saw_empty and cand:
            return cand
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find_envs_map(child)
        if found:
            return found
        if found is not None:
            saw_empty = True
    return {} if saw_empty else None


def _stringify_env_value(value: Any) -> str | None:
    """Render scalars for the shell; reject nulls, mappings and lists.

    Preserve explicit empty strings and remove only trailing newlines. PyYAML's
    BaseLoader also represents a bare ``KEY:`` as an empty string.
    """
    if value is None or isinstance(value, (dict, list)):
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value.rstrip("\n")
    return str(value)


def _recipe_envs_from_yaml(text: str) -> dict[str, str] | None:
    """Parse a recipe's environment, preserving literal scalar spellings.

    Return ``{}`` for no environment and ``None`` only when PyYAML is absent.
    Invalid YAML raises ``_RecipeYAMLError``. BaseLoader keeps values such as
    ``yes`` and ``0x10`` unchanged and handles multiline arguments containing
    JSON colons without truncating them.
    """
    try:
        import yaml
    except ImportError:
        return None
    try:
        doc = yaml.load(text, Loader=yaml.BaseLoader)
    except yaml.YAMLError as exc:
        raise _RecipeYAMLError(str(exc)) from exc
    envs = _find_envs_map(doc)
    if envs is None:
        return {}
    out: dict[str, str] = {}
    for key, value in envs.items():
        rendered = _stringify_env_value(value)
        if rendered is not None:
            out[str(key)] = rendered
    return out


def _recipe_env_block_scan(text: str) -> dict[str, str]:
    """Fallback for missing PyYAML, covering recorded recipe shapes.

    Preserve literal/folded blocks and plain-scalar continuation; skip nested
    maps. This is not a full YAML parser and cannot validate malformed input.
    """
    envs: dict[str, str] = {}
    block_indent: int | None = None
    nested_indent: int | None = None
    lines = text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if block_indent is None:
            if line.strip() == "envs:":
                block_indent = indent
            continue
        if indent <= block_indent:
            break  # dedented out of the block
        if nested_indent is not None:
            if indent > nested_indent:
                continue  # still inside a nested map
            nested_indent = None
        key, sep, value = line.strip().partition(":")
        if not sep or not key:
            continue
        key = key.strip()
        value = value.strip()
        # Explicit YAML block scalar: `|` (literal) / `>` (folded), optionally
        # followed by chomping/indent indicators (+, -, digits). Consume the
        # more-indented block, joining folded scalars with spaces and literal
        # scalars with newlines (both word-split harmlessly as server args).
        if value and value[0] in "|>" and (
            len(value) == 1 or set(value[1:]) <= set("+-0123456789")
        ):
            folded = value[0] == ">"
            block_lines: list[str] = []
            content_indent: int | None = None
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    block_lines.append("")
                    i += 1
                    continue
                nxt_indent = len(nxt) - len(nxt.lstrip())
                if content_indent is None:
                    if nxt_indent <= indent:
                        break  # nothing more-indented than the key => empty scalar
                    content_indent = nxt_indent
                elif nxt_indent < content_indent:
                    break  # dedented out of the scalar content
                block_lines.append(nxt[content_indent:])
                i += 1
            while block_lines and not block_lines[-1]:
                block_lines.pop()  # drop trailing padding the dumper may add
            joined = (" " if folded else "\n").join(block_lines)
            envs[key] = joined.strip() if folded else joined
            continue
        if not value:
            # BaseLoader represents a bare `KEY:` as an empty string unless the
            # following significant line is more indented (a nested map). Look
            # ahead so the fallback preserves that same set-but-empty semantic
            # while still skipping nested values and their children.
            j = i
            while j < n and (
                not lines[j].strip() or lines[j].lstrip().startswith("#")
            ):
                j += 1
            if j < n and (len(lines[j]) - len(lines[j].lstrip())) > indent:
                nested_indent = indent
            else:
                envs[key] = ""
            continue
        # Inline plain scalar. Fold any IMPLICIT continuation: lines indented
        # deeper than the key with no block indicator are part of this value
        # (YAML plain-scalar folding), e.g. a multi-line EXTRA_<BE>_ARGS. This is
        # unambiguous inside envs: a key with an inline value can never be
        # followed by a nested map, so a deeper line must be a continuation.
        while i < n:
            nxt = lines[i]
            if not nxt.strip():
                break  # a blank line ends the plain scalar
            if (len(nxt) - len(nxt.lstrip())) <= indent:
                break  # dedent -> next key or end of block
            value = f"{value} {nxt.strip()}"
            i += 1
        envs[key] = value.strip("'\"")
    return envs
