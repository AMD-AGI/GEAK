#!/usr/bin/env python3
"""Split a server-argument string into argv, NUL-delimited on stdout.

Adapters used to expand ``$EXTRA_SERVER_ARGS`` unquoted. Word splitting keeps
shell quotes as literal characters and pathname-expands ``*`` / ``[...]``, so a
JSON value -- which GEAK renders shell-quoted (``--online_quant_config
'{"exclude_layer":["*.mlp.gate"]}'``) -- reached the server with its quotes
still attached, or rewritten by a glob that happened to match. This applies the
shell's quoting rules without evaluating anything, and keeps a bare JSON value
(``--x {"a": 1}``, the spelling recipes and hand-written launchers use) as one
token instead of letting shlex strip its double quotes.

Usage: split_args.py "<args>"  -> each token followed by NUL. Exit 2 on input
that does not split (unbalanced quotes); the caller falls back loudly.
"""

import json
import shlex
import sys


def _protect_bare_json(text):
    """Swap balanced bare JSON values for markers so shlex leaves them whole."""
    protected = {}
    out = []
    i = 0
    while i < len(text):
        char = text[i]
        if char not in "[{" or (i and not (text[i - 1].isspace() or text[i - 1] == "=")):
            out.append(char)
            i += 1
            continue
        closing = "}" if char == "{" else "]"
        depth, quoted, escaped, end = 0, False, False, i
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
            elif current == char:
                depth += 1
            elif current == closing:
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        candidate = text[i:end]
        try:
            if depth:
                raise ValueError
            json.loads(candidate)
        except ValueError:
            out.append(char)
            i += 1
            continue
        marker = f"__GEAK_JSON_{len(protected)}__"
        protected[marker] = candidate
        out.append(marker)
        i = end
    return "".join(out), protected


def split_args(text):
    """Return the argv a shell would build from ``text``, bare JSON kept intact."""
    protected_text, protected = _protect_bare_json(text.strip())
    tokens = shlex.split(protected_text, posix=True)
    for marker, value in protected.items():
        tokens = [token.replace(marker, value) for token in tokens]
    return tokens


def main(argv):
    if len(argv) != 2:
        print("usage: split_args.py '<args>'", file=sys.stderr)
        return 2
    try:
        tokens = split_args(argv[1])
    except ValueError as exc:
        print(f"split_args: cannot split server args: {exc}", file=sys.stderr)
        return 2
    sys.stdout.write("".join(token + "\0" for token in tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
