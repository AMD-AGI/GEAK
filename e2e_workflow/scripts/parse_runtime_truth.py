#!/usr/bin/env python3
"""Extract the effective attention backend from a serving server log."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any


_PATTERNS = (
    # Decoder-specific decisions are strongest and take precedence over encoder
    # or vision attention messages in the same server log.
    (
        "decoder_override",
        re.compile(
            r"AttentionType\.DECODER[^\n]*?Overriding with\s+([A-Za-z0-9_]+)",
            re.I,
        ),
        "high",
    ),
    (
        "backend_override",
        re.compile(r"Overriding with\s+([A-Za-z0-9_]+)\s+out of potential backends", re.I),
        "high",
    ),
    (
        "decoder_selected",
        re.compile(
            r"(?:decoder|self)[^\n]{0,120}?attention[^\n]{0,80}?"
            r"(?:backend|implementation)[\s:=]+(?:AttentionBackendEnum\.)?([A-Za-z0-9_]+)",
            re.I,
        ),
        "medium",
    ),
    (
        "using_attention_backend",
        re.compile(
            r"\b(?:selected|using)\s+(?:AttentionBackendEnum\.)?([A-Za-z0-9_]+)"
            r"\s+(?:flash\s+)?attention\s+backend\b",
            re.I,
        ),
        "medium",
    ),
    (
        "attention_backend_value",
        re.compile(
            r"\battention\s+backend\s*(?:is|:|=)\s*"
            r"(?:AttentionBackendEnum\.)?([A-Za-z0-9_]+)",
            re.I,
        ),
        "medium",
    ),
    (
        "default_attention_backend",
        re.compile(r"\buse\s+([A-Za-z0-9_]+)\s+backend\s+by\s+default\b", re.I),
        "medium",
    ),
)
_NON_DECODER_CONTEXT = re.compile(
    r"\b(?:vit|vision|encoder|mmencoder)(?:[_-]?attention)?\b",
    re.I,
)


def _normalize(value: str) -> str:
    return str(value or "").strip().upper()


def parse_runtime_truth(text: str, requested: str = "", framework: str = "") -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(str(text or "").splitlines(), 1):
        for kind, pattern, confidence in _PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            if confidence != "high" and _NON_DECODER_CONTEXT.search(line):
                continue
            matches.append(
                {
                    "backend": _normalize(match.group(1)),
                    "kind": kind,
                    "confidence": confidence,
                    "line_number": line_number,
                    "evidence": line.strip(),
                }
            )

    high = [entry for entry in matches if entry["confidence"] == "high"]
    chosen = (high or matches)[-1] if matches else None
    observed = chosen["backend"] if chosen else ""
    requested_norm = _normalize(requested)
    match = (
        requested_norm == observed
        if requested_norm and observed
        else None
    )
    evidence = []
    for entry in matches[-8:]:
        if entry["evidence"] not in evidence:
            evidence.append(entry["evidence"])
    return {
        "schema": "runtime-truth-v1",
        "framework": str(framework or ""),
        "attention_backend": {
            "requested": requested_norm,
            "observed": observed,
            "effective": observed or requested_norm,
            "match": match,
            "verified": bool(observed),
            "confidence": chosen["confidence"] if chosen else "unknown",
            "evidence": evidence,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse the effective runtime attention backend from a server log."
    )
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--requested", default="")
    parser.add_argument("--framework", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    with open(args.server_log, encoding="utf-8", errors="replace") as fh:
        result = parse_runtime_truth(fh.read(), args.requested, args.framework)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(json.dumps(result["attention_backend"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
