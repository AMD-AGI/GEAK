"""Decode EXTRA_ENV into literal NUL-delimited env operands, without a shell."""

from __future__ import annotations

import json
import re
import shlex
import sys
from typing import Any


def _protect_bare_json(
    text: str, *, canonicalize_json: bool = True
) -> tuple[str, dict[str, str]]:
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
        while token in text:
            token += "_"
        protected[token] = (
            json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if canonicalize_json else candidate
        )
        out.append(token)
        i = end
    return "".join(out), protected


def _shell_tokens(text: Any, *, canonicalize_json: bool = True) -> list[str]:
    """Split shell text while protecting bare JSON values."""

    rendered = str(text or "").strip()
    if not rendered:
        return []
    protected_text, protected = _protect_bare_json(
        rendered, canonicalize_json=canonicalize_json
    )
    tokens = shlex.split(protected_text, posix=True)
    for index, token in enumerate(tokens):
        for marker, value in protected.items():
            if marker in token:
                token = token.replace(marker, value)
        tokens[index] = token
    return tokens


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[2] == "--unset":
        names = parse_unset_envs(json.loads(sys.argv[1]))
        sys.stdout.buffer.write(b"".join(b"-u\0" + name.encode() + b"\0" for name in names))
        return
    tokens = _shell_tokens(sys.argv[1], canonicalize_json=False)
    assignments = []
    for token in tokens:
        key, separator, _ = token.partition("=")
        if separator and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) and "\0" not in token:
            assignments.append(token)
        else:
            raise ValueError(f"EXTRA_ENV entry must be KEY=VALUE without NUL: {token!r}")
    sys.stdout.buffer.write(b"".join(token.encode() + b"\0" for token in assignments))


def parse_unset_envs(value: Any) -> list[str]:
    """Validate explicit removals; omitted assignments never imply deletion."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError("unset_envs must be a name or a list of names")
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
           for name in value):
        raise ValueError("unset_envs entries must be environment identifiers")
    return sorted(set(value))


if __name__ == "__main__":
    try:
        main()
    except (TypeError, ValueError) as error:
        print(f"invalid EXTRA_ENV: {error}", file=sys.stderr)
        sys.exit(2)
