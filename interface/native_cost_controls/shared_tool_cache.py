# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add an opt-in cache marker to a registered shared tool prefix.

This module transforms request bytes only. The caller owns the provider
transport and supplies the shared tool definitions explicitly. The module does
not discover tools, change model settings, send requests, or record usage.

The supported request has the registered shared tools followed by one
StructuredOutput tool. It has three five-minute cache markers: system blocks
1 and 2, and the final conversation block. The policy adds a fourth marker to
the final shared tool. Other requests retain their original bytes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

_MARKER_BYTES = b',"cache_control":{"type":"ephemeral","ttl":"5m"}'
_MAX_MARKERS = 4


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("A JSON object contains a duplicate key.")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("The JSON value is not finite.")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("The JSON number is not finite.")
    return number


_DECODER = json.JSONDecoder(
    object_pairs_hook=_unique_object,
    parse_constant=_reject_constant,
    parse_float=_finite_float,
)


def _canonical(value):
    # Encoding also rejects unpaired Unicode surrogates in parsed JSON strings.
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _skip_space(text, index):
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _shared_tool_end(text, count):
    """Locate the last shared tool's closing brace in validated JSON text."""
    cursor = _skip_space(text, 0) + 1
    while True:
        cursor = _skip_space(text, cursor)
        key, cursor = _DECODER.raw_decode(text, cursor)
        cursor = _skip_space(text, cursor) + 1  # Skip the field's colon.
        cursor = _skip_space(text, cursor)
        if key == "tools":
            cursor += 1  # Skip the array's opening bracket.
            for index in range(count):
                cursor = _skip_space(text, cursor)
                _, end = _DECODER.raw_decode(text, cursor)
                if index == count - 1:
                    return end - 1
                cursor = _skip_space(text, end) + 1  # Skip the array's comma.
        _, cursor = _DECODER.raw_decode(text, cursor)
        cursor = _skip_space(text, cursor) + 1  # Skip the object's comma.


def _tool_definition(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and bool(value["name"])
        and isinstance(value.get("input_schema"), dict)
        and ("description" not in value or isinstance(value["description"], str))
        and "cache_control" not in value
    )


def _structured_output(value):
    return (
        _tool_definition(value)
        and value["name"] == "StructuredOutput"
        and value["input_schema"].get("type") == "object"
        and set(value) <= {"name", "description", "input_schema", "strict"}
        and ("strict" not in value or type(value["strict"]) is bool)
    )


def _marker_positions(body):
    positions = []
    if "cache_control" in body:
        positions.append((("request",), body["cache_control"]))
    for section in ("tools", "system"):
        for index, block in enumerate(body[section]):
            if isinstance(block, dict) and "cache_control" in block:
                positions.append(((section, index), block["cache_control"]))
    pending = [(("messages", index), message.get("content")) for index, message in enumerate(body["messages"])]
    while pending:
        path, content = pending.pop()
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                if isinstance(block, dict):
                    block_path = (*path, block_index)
                    if "cache_control" in block:
                        positions.append((block_path, block["cache_control"]))
                    if block.get("type") == "tool_result":
                        pending.append(((*block_path, "content"), block.get("content")))
    return positions


def _supported_layout(body, positions):
    if not all(
        isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        for block in body["system"]
    ):
        return False
    messages = body["messages"]
    last_content = messages[-1].get("content")
    if not isinstance(last_content, list) or not last_content:
        return False
    last_block = last_content[-1]
    if not isinstance(last_block, dict) or last_block.get("type") not in {
        "text", "tool_result", "image", "document"
    }:
        return False
    expected = {("system", 1), ("system", 2), ("messages", len(messages) - 1, len(last_content) - 1)}
    if len(positions) != 3 or {path for path, _ in positions} != expected:
        return False
    return all(marker in ({"type": "ephemeral"}, {"type": "ephemeral", "ttl": "5m"}) for _, marker in positions)


@dataclass(frozen=True)
class CachePolicyResult:
    """Return the request bytes and a content-free policy decision."""

    body: bytes
    applied: bool
    reason: str


class SharedToolCachePolicy:
    """Keep the policy disabled until the caller explicitly enables it.

    Register the complete shared tool definitions from a trusted configuration.
    Their order matters. The policy accepts one additional StructuredOutput
    tool after this exact prefix. Definitions remain fixed for this instance.

    The caller must use a provider that supports this Anthropic-compatible
    cache layout. A marker does not guarantee a cache hit or a lower bill.
    """

    def __init__(self, shared_tools: Sequence[Any] | None = None, *, enabled: bool = False):
        if type(enabled) is not bool:
            raise ValueError("The enabled setting must be a boolean.")
        self.enabled = enabled
        self._prefix = None
        self._count = 0
        if not enabled:
            return
        if not isinstance(shared_tools, (list, tuple)) or not shared_tools:
            raise ValueError("Supply a nonempty list of shared tool definitions.")
        if not all(_tool_definition(tool) and tool["name"] != "StructuredOutput" for tool in shared_tools):
            raise ValueError("A shared tool definition is unsupported.")
        if len({tool["name"] for tool in shared_tools}) != len(shared_tools):
            raise ValueError("Shared tool names must be unique.")
        try:
            self._prefix = _canonical(list(shared_tools))
        except (TypeError, ValueError, UnicodeError, RecursionError) as error:
            raise ValueError("Shared tool definitions must contain valid JSON values.") from error
        self._count = len(shared_tools)

    def apply(self, raw_body: bytes, *, method: str = "POST", path: str = "/v1/messages") -> CachePolicyResult:
        """Preserve the exact input bytes when the policy cannot add a marker."""
        if not isinstance(raw_body, bytes):
            raise TypeError("The request body must be bytes.")
        if not self.enabled:
            return CachePolicyResult(raw_body, False, "disabled")
        if method != "POST" or not isinstance(path, str) or path.split("?", 1)[0] != "/v1/messages":
            return CachePolicyResult(raw_body, False, "unsupported_endpoint")
        try:
            text = raw_body.decode("utf-8")
            body = _DECODER.decode(text)
            _canonical(body)
        except (ValueError, UnicodeError, RecursionError):
            return CachePolicyResult(raw_body, False, "invalid_json")
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("model"), str)
            or not body["model"]
            or not isinstance(body.get("tools"), list)
            or not isinstance(body.get("system"), list)
            or not isinstance(body.get("messages"), list)
            or not body["messages"]
            or not all(
                isinstance(message, dict)
                and message.get("role") in ("user", "assistant", "system")
                and isinstance(message.get("content"), (str, list))
                for message in body["messages"]
            )
        ):
            return CachePolicyResult(raw_body, False, "unsupported_request")
        tools = body["tools"]
        if len(tools) != self._count + 1 or _canonical(tools[:self._count]) != self._prefix:
            return CachePolicyResult(raw_body, False, "shared_tools_mismatch")
        if not _structured_output(tools[-1]):
            return CachePolicyResult(raw_body, False, "unsupported_output_tool")
        positions = _marker_positions(body)
        if len(positions) >= _MAX_MARKERS:
            return CachePolicyResult(raw_body, False, "marker_limit")
        if not _supported_layout(body, positions):
            return CachePolicyResult(raw_body, False, "unsupported_cache_layout")
        try:
            end = _shared_tool_end(text, self._count)
            offset = len(text[:end].encode("utf-8"))
            forwarded = raw_body[:offset] + _MARKER_BYTES + raw_body[offset:]
            restored = _DECODER.decode(forwarded.decode("utf-8"))
            del restored["tools"][self._count - 1]["cache_control"]
            if _canonical(restored) != _canonical(body):
                return CachePolicyResult(raw_body, False, "unsupported_json_layout")
        except (ValueError, UnicodeError, KeyError, IndexError, RecursionError):
            return CachePolicyResult(raw_body, False, "unsupported_json_layout")
        return CachePolicyResult(forwarded, True, "marked")
