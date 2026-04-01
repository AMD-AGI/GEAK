"""GEAK MCP HTTP Server.

Provides MCP server over HTTP transport with API key authentication via headers.
This allows AI agents to connect via HTTP and pass authentication in headers.
Supports both Streamable HTTP and SSE transport for Cursor compatibility.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from server.mcp.tools import GEAKTools
from server.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_geak_local() -> bool:
    """True when running in local-all-in-one mode (no SaFE user API key required for MCP)."""
    return os.getenv("GEAK_LOCAL", "false").lower() in ("true", "1", "yes")


def _local_mcp_effective_api_key(header_api_key: str | None) -> str | None:
    """Return API key for downstream REST calls; in local mode, substitute a placeholder if missing."""
    if header_api_key:
        return header_api_key
    if _is_geak_local():
        return os.getenv("GEAK_MCP_LOCAL_API_KEY", "local-mcp")
    return None


class MCPRequest(BaseModel):
    """MCP JSON-RPC request."""
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] | None = None


def make_mcp_response(id: int | str | None, result: Any = None, error: dict | None = None) -> dict:
    """Create MCP JSON-RPC response with only result OR error (not both)."""
    response = {"jsonrpc": "2.0"}
    if id is not None:
        response["id"] = id
    if error is not None:
        response["error"] = error
    else:
        response["result"] = result if result is not None else {}
    return response


def create_mcp_http_app() -> FastAPI:
    """Create FastAPI app for MCP HTTP server."""
    
    app = FastAPI(
        title="GEAK MCP Server",
        description="MCP (Model Context Protocol) server for GEAK optimization service",
        version="1.0.0",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    tools = GEAKTools()
    
    # Tool definitions
    TOOLS = {
        "geak_get_model_config": {
            "description": "Get user's default model configuration for GEAK tasks",
            "handler": lambda api_key, **kwargs: tools.get_model_config(api_key),
        },
        "geak_set_model_config": {
            "description": "Set user's default model configuration for GEAK tasks",
            "handler": lambda api_key, **kwargs: tools.set_model_config(api_key, **kwargs),
        },
        "geak_delete_model_config": {
            "description": "Delete user's default model configuration",
            "handler": lambda api_key, **kwargs: tools.delete_model_config(api_key),
        },
        "geak_create_task": {
            "description": "Create a new GPU/HIP kernel optimization task",
            "handler": lambda api_key, **kwargs: tools.create_task(api_key, **kwargs),
        },
        "geak_get_task": {
            "description": "Get task details and status",
            "handler": lambda api_key, **kwargs: tools.get_task(api_key, **kwargs),
        },
        "geak_list_tasks": {
            "description": "List user's optimization tasks",
            "handler": lambda api_key, **kwargs: tools.list_tasks(api_key, **kwargs),
        },
        "geak_submit_task": {
            "description": "Submit a pending task for execution on SaFE platform",
            "handler": lambda api_key, **kwargs: tools.submit_task(api_key, **kwargs),
        },
        "geak_cancel_task": {
            "description": "Cancel a running task",
            "handler": lambda api_key, **kwargs: tools.cancel_task(api_key, **kwargs),
        },
        "geak_get_outputs": {
            "description": "Get list of output files from a completed task",
            "handler": lambda api_key, **kwargs: tools.get_outputs(api_key, **kwargs),
        },
        "geak_download_file": {
            "description": "Download a specific output file from a task",
            "handler": lambda api_key, **kwargs: tools.download_file(api_key, **kwargs),
        },
    }
    
    def get_tool_schemas() -> list[dict]:
        """Get tool schemas for MCP tools/list."""
        return [
            {
                "name": "geak_get_model_config",
                "description": "Get user's default model configuration for GEAK tasks",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "geak_set_model_config",
                "description": "Set user's default model configuration for GEAK tasks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_class": {"type": "string", "description": "Model class: litellm, amd_llm, etc."},
                        "model_name": {"type": "string", "description": "Model name, e.g., openai/claude-opus-4.5"},
                        "model_kwargs": {
                            "type": "object",
                            "description": "Model parameters including api_base, api_key, max_tokens, temperature"
                        }
                    },
                    "required": ["model_class", "model_name", "model_kwargs"]
                }
            },
            {
                "name": "geak_delete_model_config",
                "description": "Delete user's default model configuration",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "geak_create_task",
                "description": "Create a new GPU/HIP kernel optimization task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input_type": {"type": "string", "enum": ["file", "repo"]},
                        "files": {
                            "type": "array",
                            "description": "List of files for file input (can include .hip, Makefile, headers, etc.)",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "filename": {"type": "string", "description": "File name, e.g., silu.hip or Makefile"},
                                    "content": {"type": "string", "description": "File content"}
                                },
                                "required": ["filename", "content"]
                            }
                        },
                        "repo_url": {"type": "string", "description": "Git repository URL"},
                        "repo_branch": {"type": "string", "description": "Git branch"},
                        "prompt": {"type": "string", "description": "Optimization instructions"},
                        "step_limit": {"type": "integer", "description": "Max agent steps"},
                        "gpu_count": {"type": "integer", "description": "Number of GPUs"},
                        "image": {"type": "string", "description": "Custom Docker image to use for task execution. If not provided, uses the server default image."},
                        "workspace_id": {"type": "string", "description": "SaFE workspace ID (defaults to user's first workspace)"}
                    },
                    "required": ["input_type"]
                }
            },
            {
                "name": "geak_get_task",
                "description": "Get task details and status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID"}
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "geak_list_tasks",
                "description": "List user's optimization tasks",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["pending", "running", "completed", "failed", "cancelled"]},
                        "limit": {"type": "integer", "default": 20}
                    },
                    "required": []
                }
            },
            {
                "name": "geak_submit_task",
                "description": "Submit a pending task for execution",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID to submit"}
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "geak_cancel_task",
                "description": "Cancel a running task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID to cancel"}
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "geak_get_outputs",
                "description": "Get list of output files from a completed task",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID"}
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "geak_download_file",
                "description": "Download a file from task outputs. Returns download URL. For small text files, also includes content directly.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "Task ID"},
                        "file_path": {"type": "string", "description": "Path to file (e.g., execution.log, modified_repo.tar.gz)"}
                    },
                    "required": ["task_id", "file_path"]
                }
            },
        ]
    
    # Session storage for SSE connections
    sessions: dict[str, asyncio.Queue] = {}
    
    def _get_message_endpoint(request: Request, session_id: str) -> str:
        """Build the full message endpoint URL for SSE clients.
        
        When behind a reverse proxy with a path prefix (e.g.
        /control-plane/.../geak-agent-xdk2z), the SSE endpoint event must
        return a path that includes the full prefix so that the client POSTs
        to the correct URL.
        
        Priority:
        1. EXTERNAL_BASE_URL from settings (most reliable for proxy setups)
        2. Derive from request URL path
        """
        settings = get_settings()
        if settings.external_base_url:
            base = settings.external_base_url.rstrip("/")
            return f"{base}/mcp/message?sessionId={session_id}"
        
        path = str(request.url.path)
        for suffix in ("/sse", "/"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                break
        base_path = path.rstrip("/")
        return f"{base_path}/message?sessionId={session_id}"
    
    @app.get("/")
    async def root_get(
        request: Request,
        authorization: str = Header(None),
        x_api_key: str = Header(None, alias="X-API-Key"),
    ):
        """Root GET endpoint - returns SSE stream or info based on Accept header."""
        accept = request.headers.get("accept", "")
        
        # If client accepts text/event-stream, return SSE
        if "text/event-stream" in accept:
            session_id = str(uuid.uuid4())
            response_queue: asyncio.Queue = asyncio.Queue()
            sessions[session_id] = response_queue
            message_endpoint = _get_message_endpoint(request, session_id)
            
            async def event_generator():
                try:
                    yield f"event: endpoint\ndata: {message_endpoint}\n\n"
                    while True:
                        try:
                            response = await asyncio.wait_for(response_queue.get(), timeout=30)
                            yield f"event: message\ndata: {json.dumps(response)}\n\n"
                        except asyncio.TimeoutError:
                            yield ": keepalive\n\n"
                finally:
                    sessions.pop(session_id, None)
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        
        # Otherwise return server info
        return {
            "name": "geak-mcp-server",
            "version": "1.0.0",
            "protocol": "mcp",
            "description": "GEAK GPU/HIP Kernel Optimization MCP Server",
            "endpoints": {
                "mcp": "/mcp",
                "sse": "/mcp/sse",
                "tools": "/mcp/tools",
            }
        }
    
    @app.get("/info")
    async def info():
        """MCP server info."""
        return {
            "name": "geak-mcp-server",
            "version": "1.0.0",
            "protocol": "mcp",
            "description": "GEAK GPU/HIP Kernel Optimization MCP Server",
            "endpoints": {
                "mcp": "/mcp",
                "tools": "/mcp/tools",
            }
        }
    
    @app.get("/tools")
    async def list_tools():
        """List available MCP tools (for discovery)."""
        return {"tools": get_tool_schemas()}
    
    # SSE endpoint for Cursor MCP client compatibility
    # Also support POST to /sse for Streamable HTTP transport at this endpoint
    @app.post("/sse")
    async def sse_post_endpoint(
        request: MCPRequest,
        authorization: str = Header(None),
        x_api_key: str = Header(None, alias="X-API-Key"),
    ):
        """Handle POST to /sse - redirect to main MCP handler."""
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:]
        elif x_api_key:
            api_key = x_api_key
        return await process_mcp_request(request, api_key, tools, TOOLS, get_tool_schemas)
    
    @app.get("/sse")
    async def sse_endpoint(
        request: Request,
        authorization: str = Header(None),
        x_api_key: str = Header(None, alias="X-API-Key"),
    ):
        """SSE endpoint for MCP client connections.
        
        This provides Server-Sent Events transport for MCP protocol.
        The client connects here to receive responses via SSE stream.
        """
        session_id = str(uuid.uuid4())
        response_queue: asyncio.Queue = asyncio.Queue()
        sessions[session_id] = response_queue
        message_endpoint = _get_message_endpoint(request, session_id)
        
        # Extract API key and store it
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:]
        elif x_api_key:
            api_key = x_api_key
        
        async def event_generator():
            try:
                # Send endpoint event to tell client where to POST messages
                yield f"event: endpoint\ndata: {message_endpoint}\n\n"
                
                # Wait for responses and send them via SSE
                while True:
                    try:
                        # Wait for response with timeout
                        response = await asyncio.wait_for(response_queue.get(), timeout=30)
                        yield f"event: message\ndata: {json.dumps(response)}\n\n"
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield ": keepalive\n\n"
            finally:
                # Cleanup session
                sessions.pop(session_id, None)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    
    @app.post("/message")
    async def message_endpoint(
        request: Request,
        sessionId: str = None,
        authorization: str = Header(None),
        x_api_key: str = Header(None, alias="X-API-Key"),
    ):
        """Message endpoint for SSE transport.
        
        Client POSTs JSON-RPC requests here, responses are sent via SSE stream.
        """
        # Extract API key from headers
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:]
        elif x_api_key:
            api_key = x_api_key
        
        try:
            body = await request.json()
            mcp_request = MCPRequest(**body)
        except Exception as e:
            error_response = make_mcp_response(None, error={"code": -32700, "message": f"Parse error: {e}"})
            if sessionId and sessionId in sessions:
                await sessions[sessionId].put(error_response)
            return {"status": "error", "message": str(e)}
        
        # Process the request
        response = await process_mcp_request(mcp_request, api_key, tools, TOOLS, get_tool_schemas)
        
        # Send response via SSE stream if session exists
        if sessionId and sessionId in sessions:
            await sessions[sessionId].put(response)
            return {"status": "accepted"}
        else:
            # Fallback: return response directly (for testing)
            return response
    
    async def process_mcp_request(request: MCPRequest, api_key: str | None, tools: GEAKTools, TOOLS: dict, get_tool_schemas):
        """Process MCP JSON-RPC request."""
        logger.info(f"MCP request: {request.method}")
        
        try:
            if request.method == "initialize":
                return make_mcp_response(
                    id=request.id,
                    result={
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {
                            "name": "geak-mcp-server",
                            "version": "1.0.0",
                        },
                        "capabilities": {
                            "tools": {"listChanged": False},
                        },
                    }
                )
            
            elif request.method == "tools/list":
                return make_mcp_response(
                    id=request.id,
                    result={"tools": get_tool_schemas()}
                )
            
            elif request.method == "tools/call":
                effective_key = _local_mcp_effective_api_key(api_key)
                if not effective_key:
                    return make_mcp_response(
                        id=request.id,
                        error={
                            "code": -32001,
                            "message": "Authentication required. Provide API key via Authorization or X-API-Key header."
                        }
                    )
                
                params = request.params or {}
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if tool_name not in TOOLS:
                    return make_mcp_response(
                        id=request.id,
                        error={
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        }
                    )
                
                # Call the tool
                handler = TOOLS[tool_name]["handler"]
                result = await handler(effective_key, **arguments)
                
                return make_mcp_response(
                    id=request.id,
                    result={
                        "content": [
                            {"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}
                        ]
                    }
                )
            
            elif request.method == "ping":
                return make_mcp_response(id=request.id, result={})
            
            elif request.method == "notifications/initialized":
                return make_mcp_response(id=request.id, result={})
            
            else:
                return make_mcp_response(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {request.method}"
                    }
                )
        
        except Exception as e:
            logger.exception(f"Error handling MCP request: {request.method}")
            return make_mcp_response(
                id=request.id,
                error={
                    "code": -32603,
                    "message": str(e)
                }
            )
    
    @app.post("/")
    async def mcp_endpoint(
        request: MCPRequest,
        authorization: str = Header(None),
        x_api_key: str = Header(None, alias="X-API-Key"),
    ):
        """Main MCP endpoint for JSON-RPC requests (Streamable HTTP transport).
        
        Supports both Authorization header (Bearer token) and X-API-Key header.
        """
        # Extract API key from headers
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:]
        elif x_api_key:
            api_key = x_api_key
        
        return await process_mcp_request(request, api_key, tools, TOOLS, get_tool_schemas)
    
    # Simplified REST-like endpoints for easier testing
    @app.post("/tools/{tool_name}")
    async def call_tool_rest(
        tool_name: str,
        request: Request,
        authorization: str = Header(None),
        x_api_key: str = Header(None, alias="X-API-Key"),
    ):
        """REST-like endpoint for calling individual tools.
        
        This provides an easier interface for testing tools directly.
        """
        # Extract API key
        api_key = None
        if authorization and authorization.startswith("Bearer "):
            api_key = authorization[7:]
        elif x_api_key:
            api_key = x_api_key
        
        api_key = _local_mcp_effective_api_key(api_key)
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        
        if tool_name not in TOOLS:
            raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
        
        # Get request body
        try:
            body = await request.json()
        except Exception:
            body = {}
        
        # Call the tool
        handler = TOOLS[tool_name]["handler"]
        result = await handler(api_key, **body)
        
        return result
    
    return app


# Create app instance
mcp_app = create_mcp_http_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    # MCP_PORT: dedicated env var for MCP server binding port (default 8001)
    # Keeps PORT env var free for GEAKTools to reference the REST API
    mcp_port = int(os.environ.get("MCP_PORT", 8001))
    uvicorn.run(
        "server.mcp.http_server:mcp_app",
        host=settings.host,
        port=mcp_port,
        reload=False,
    )
