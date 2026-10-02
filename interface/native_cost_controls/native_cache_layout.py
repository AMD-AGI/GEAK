# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative API-content guard for the registered prefix policy."""

from .native_prefix_marker import DECODER, markers


def _api_content_blocks(body):
    """Visit API content containers, not schemas, tool inputs, or arbitrary data."""
    supported = {
        "text",
        "image",
        "document",
        "search_result",
        "thinking",
        "redacted_thinking",
        "tool_use",
        "tool_result",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
        "tool_reference",
        "container_upload",
        "mcp_tool_use",
        "mcp_tool_result",
    }
    pending = [body["system"]] if "system" in body else []
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("The request messages are invalid.")  # noqa: TRY004 - Keep the frozen protocol error and guard AST.
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("A request message is invalid.")  # noqa: TRY004 - Keep the frozen protocol error and guard AST.
        pending.append(message.get("content"))
    while pending:
        content = pending.pop()
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("A request content container is invalid.")  # noqa: TRY004 - Keep the frozen protocol error and guard AST.
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("A request content block is invalid.")  # noqa: TRY004 - Keep the frozen protocol error and guard AST.
            kind = block.get("type")
            if kind == "compaction":
                raise ValueError(
                    "Server-side compaction requires separate iteration accounting."
                )
            if not isinstance(kind, str) or kind not in supported:
                raise ValueError("The request content block type is unsupported.")
            yield block
            if kind in ("tool_result", "search_result", "mcp_tool_result"):
                if "content" in block:
                    pending.append(block["content"])
            elif kind == "document":
                source = block.get("source")
                if isinstance(source, dict) and source.get("type") == "content":
                    pending.append(source.get("content"))
            elif kind == "web_fetch_tool_result":
                result = block.get("content")
                if not isinstance(result, dict):
                    raise ValueError("The web fetch result content is invalid.")
                if result.get("type") == "web_fetch_result":
                    pending.append([result.get("content")])
                elif result.get("type") != "web_fetch_tool_result_error":
                    raise ValueError("The web fetch result type is unsupported.")
            elif kind == "tool_search_tool_result":
                result = block.get("content")
                if not isinstance(result, dict):
                    raise ValueError("The tool search result content is invalid.")
                if result.get("type") == "tool_search_tool_search_result":
                    pending.append(result.get("tool_references"))
                elif result.get("type") != "tool_search_tool_result_error":
                    raise ValueError("The tool search result type is unsupported.")


def content_layout_eligible(raw):
    """Decline markers outside the assay policy's content traversal."""
    try:
        body = DECODER.decode(raw.decode("utf-8"))
        controls = [body, *body.get("tools", []), *_api_content_blocks(body)]
        complete_count = sum("cache_control" in value for value in controls)
        return complete_count == len(markers(body))
    except (ValueError, TypeError, UnicodeError, KeyError, IndexError, RecursionError):
        return False
