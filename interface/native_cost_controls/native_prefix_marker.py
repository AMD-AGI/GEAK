"""Apply the registered native prefix marker policy without external access."""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import re


__all__ = [
    "DECODER",
    "MARKER",
    "Decision",
    "client_tool",
    "mark_registered_prefix",
    "markers",
    "reject",
    "sha",
    "shared_end",
    "ttl",
    "unique",
    "wire",
]


MARKER = b',"cache_control":{"type":"ephemeral","ttl":"5m"}'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_key")
        value[key] = item
    return value


def reject(_):
    raise ValueError("nonfinite_number")


DECODER = json.JSONDecoder(
    object_pairs_hook=unique, parse_float=Decimal, parse_constant=reject
)


def wire(value):
    if isinstance(value, dict):
        return (
            b"{"
            + b",".join(wire(key) + b":" + wire(value[key]) for key in sorted(value))
            + b"}"
        )
    if isinstance(value, list):
        return b"[" + b",".join(wire(item) for item in value) + b"]"
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("nonfinite_number")
        return str(value).encode("ascii")
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def markers(body):
    found = []
    if "cache_control" in body:
        found.append((["request"], body["cache_control"]))
    for section in ["tools", "system"]:
        values = body.get(section, [])
        if isinstance(values, list):
            for i, value in enumerate(values):
                if isinstance(value, dict) and "cache_control" in value:
                    found.append(([section, i], value["cache_control"]))

    def blocks(content, path):
        if not isinstance(content, list):
            return
        for i, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            here = path + [i]
            if "cache_control" in block:
                found.append((here, block["cache_control"]))
            if block.get("type") == "tool_result":
                blocks(block.get("content"), here + ["content"])

    for i, message in enumerate(body.get("messages", [])):
        if isinstance(message, dict):
            blocks(message.get("content"), ["messages", i, "content"])
    return found


def ttl(marker):
    if not isinstance(marker, dict) or marker.get("type") != "ephemeral":
        return "unsupported"
    if set(marker) - {"type", "ttl"}:
        return "unsupported"
    value = marker.get("ttl", "5m")
    return value if value in ("5m", "1h") else "unsupported"


def client_tool(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("name"), str)
        and bool(value["name"])
        and isinstance(value.get("input_schema"), dict)
        and ("description" not in value or isinstance(value["description"], str))
    )


def shared_end(text, count):
    def space(i):
        while text[i] in " \t\r\n":
            i += 1
        return i

    cursor = space(0) + 1
    while True:
        key, cursor = DECODER.raw_decode(text, space(cursor))
        cursor = space(cursor) + 1
        cursor = space(cursor)
        if key == "tools":
            cursor += 1
            for i in range(count):
                _, end = DECODER.raw_decode(text, space(cursor))
                if i == count - 1:
                    return end - 1
                cursor = space(end) + 1
        _, cursor = DECODER.raw_decode(text, cursor)
        cursor = space(cursor) + 1


@dataclass(frozen=True)
class Decision:
    body: bytes
    applied: bool
    reason: str


def mark_registered_prefix(
    raw,
    *,
    count,
    prefix_sha256,
    enabled=False,
    method="POST",
    path="/v1/messages",
):
    """Insert one 5m marker after an exact, pre-registered tool prefix."""
    if not isinstance(raw, bytes):
        raise TypeError("The request must contain bytes.")
    if not enabled:
        return Decision(raw, False, "disabled")
    if (
        type(count) is not int
        or count < 1
        or not isinstance(prefix_sha256, str)
        or not re.fullmatch("[0-9a-f]{64}", prefix_sha256)
    ):
        raise ValueError("The prefix registration is invalid.")
    if method != "POST" or path.split("?", 1)[0] != "/v1/messages":
        return Decision(raw, False, "unsupported_endpoint")
    try:
        text = raw.decode("utf-8")
        body = DECODER.decode(text)
        if not isinstance(body, dict):
            return Decision(raw, False, "unsupported_request")
        tools = body.get("tools")
        if not isinstance(tools, list) or len(tools) < count:
            return Decision(raw, False, "prefix_missing")
        if not all(client_tool(item) for item in tools[:count]):
            return Decision(raw, False, "unsupported_prefix")
        if len({item["name"] for item in tools[:count]}) != count:
            return Decision(raw, False, "duplicate_tool_names")
        if sha(wire(tools[:count])) != prefix_sha256:
            return Decision(raw, False, "prefix_mismatch")
        if "output_format" in body or "format" in (body.get("output_config") or {}):
            return Decision(raw, False, "unqualified_output_format")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            return Decision(raw, False, "unsupported_request")
        existing = markers(body)
        if len(existing) >= 4:
            return Decision(raw, False, "marker_limit")
        if any(path[0] == "tools" for path, _ in existing):
            return Decision(raw, False, "existing_tool_marker")
        if any(path == ["request"] or ttl(value) != "5m" for path, value in existing):
            return Decision(raw, False, "unqualified_cache_layout")
        end = shared_end(text, count)
        offset = len(text[:end].encode("utf-8"))
        output = raw[:offset] + MARKER + raw[offset:]
        checked = DECODER.decode(output.decode("utf-8"))
        inserted = checked["tools"][count - 1].pop("cache_control")
        if inserted != {"type": "ephemeral", "ttl": "5m"} or wire(checked) != wire(body):
            return Decision(raw, False, "content_mismatch")
        if output[:offset] + output[offset + len(MARKER) :] != raw:
            return Decision(raw, False, "byte_mismatch")
        return Decision(output, True, "marked")
    except (ValueError, TypeError, UnicodeError, KeyError, IndexError, RecursionError):
        return Decision(raw, False, "unsupported_json")
