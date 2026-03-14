from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

_DEBUG_LOG_PATH = Path(os.environ.get("GEAK_DEBUG_LOG_PATH", "/home/sapmajum/.cursor/debug-f9f3d1.log"))
_DEBUG_SESSION_ID = os.environ.get("GEAK_DEBUG_SESSION_ID", "f9f3d1")
_DEFAULT_RUN_ID = os.environ.get("GEAK_DEBUG_RUN_ID", "post-fix")


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        if "name" in tool:
            return str(tool["name"])
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"])
    return str(tool)


def tool_names(tools: Any) -> list[str] | None:
    if tools is None:
        return None
    return [_tool_name(tool) for tool in tools]


def model_tools_snapshot(model: Any) -> dict[str, Any]:
    impl = getattr(model, "_impl", None)
    wrapper_tools = getattr(model, "tools", None)
    impl_tools = getattr(impl, "tools", None) if impl is not None else None
    return {
        "model_type": type(model).__name__,
        "model_id": id(model),
        "impl_type": type(impl).__name__ if impl is not None else None,
        "impl_id": id(impl) if impl is not None else None,
        "wrapper_tools": tool_names(wrapper_tools),
        "impl_tools": tool_names(impl_tools),
    }


def emit_debug_log(
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    hypothesis_id: str,
    run_id: str | None = None,
) -> None:
    try:
        payload = {
            "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "runId": run_id or _DEFAULT_RUN_ID,
            "hypothesisId": hypothesis_id,
        }
        if _DEBUG_SESSION_ID:
            payload["sessionId"] = _DEBUG_SESSION_ID
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
