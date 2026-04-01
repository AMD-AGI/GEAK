"""GEAK MCP Server Implementation.

Provides MCP tools for AI agents to interact with GEAK optimization service.
Supports HTTP transport with API key authentication via headers.
"""

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from server.mcp.tools import GEAKTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mcp_server(api_key: str | None = None) -> Server:
    """Create and configure MCP server instance.
    
    Args:
        api_key: Optional API key for authentication. If not provided,
                 tools will require api_key parameter in each call.
    
    Returns:
        Configured MCP Server instance.
    """
    server = Server("geak-mcp-server")
    tools = GEAKTools(default_api_key=api_key)
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available GEAK tools."""
        return [
            # User Configuration
            Tool(
                name="geak_get_model_config",
                description="Get user's default model configuration for GEAK tasks",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="geak_set_model_config",
                description="Set user's default model configuration for GEAK tasks",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        },
                        "model_class": {
                            "type": "string",
                            "description": "Model class: litellm, amd_llm, etc.",
                            "default": "litellm"
                        },
                        "model_name": {
                            "type": "string",
                            "description": "Model name, e.g., openai/claude-opus-4.5"
                        },
                        "model_kwargs": {
                            "type": "object",
                            "description": "Model parameters including api_base, api_key, max_tokens, temperature",
                            "properties": {
                                "api_base": {"type": "string"},
                                "api_key": {"type": "string"},
                                "max_tokens": {"type": "integer"},
                                "temperature": {"type": "number"}
                            }
                        }
                    },
                    "required": ["model_class", "model_name", "model_kwargs"]
                }
            ),
            Tool(
                name="geak_delete_model_config",
                description="Delete user's default model configuration",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        }
                    },
                    "required": []
                }
            ),
            
            # Task Management
            Tool(
                name="geak_create_task",
                description="Create a new GPU/HIP kernel optimization task",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        },
                        "input_type": {
                            "type": "string",
                            "enum": ["file", "repo"],
                            "description": "Input type: 'file' for single file, 'repo' for git repository"
                        },
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
                        "repo_url": {
                            "type": "string",
                            "description": "Git repository URL"
                        },
                        "repo_branch": {
                            "type": "string",
                            "description": "Git branch"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Optimization instructions"
                        },
                        "step_limit": {
                            "type": "integer",
                            "description": "Max agent steps"
                        },
                        "gpu_count": {
                            "type": "integer",
                            "description": "Number of GPUs"
                        },
                        "image": {
                            "type": "string",
                            "description": "Custom Docker image to use for task execution. If not provided, uses the server default image."
                        },
                        "workspace_id": {
                            "type": "string",
                            "description": "SaFE workspace ID (defaults to user's first workspace)"
                        }
                    },
                    "required": ["input_type"]
                }
            ),
            Tool(
                name="geak_get_task",
                description="Get task details and status",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to query"
                        }
                    },
                    "required": ["task_id"]
                }
            ),
            Tool(
                name="geak_list_tasks",
                description="List user's optimization tasks",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "running", "completed", "failed", "cancelled"],
                            "description": "Filter by status"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tasks to return",
                            "default": 20
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="geak_submit_task",
                description="Submit a pending task for execution on SaFE platform",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to submit"
                        }
                    },
                    "required": ["task_id"]
                }
            ),
            Tool(
                name="geak_cancel_task",
                description="Cancel a running task",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to cancel"
                        }
                    },
                    "required": ["task_id"]
                }
            ),
            
            # Task Outputs
            Tool(
                name="geak_get_outputs",
                description="Get list of output files from a completed task",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to get outputs for"
                        }
                    },
                    "required": ["task_id"]
                }
            ),
            Tool(
                name="geak_download_file",
                description="Download a file from task outputs. Returns download URL. For small text files, also includes content directly.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": "SaFE platform API key (ak-xxx). Optional if default key is set."
                        },
                        "task_id": {
                            "type": "string",
                            "description": "Task ID"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file within task outputs (e.g., execution.log, modified_repo.tar.gz)"
                        }
                    },
                    "required": ["task_id", "file_path"]
                }
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        logger.info(f"Tool call: {name} with args: {list(arguments.keys())}")
        
        try:
            # Get API key from arguments or use default
            api_key = arguments.pop("api_key", None) or tools.default_api_key
            if not api_key:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": "API key required. Provide 'api_key' parameter or set default key."
                    })
                )]
            
            # Route to appropriate tool handler
            if name == "geak_get_model_config":
                result = await tools.get_model_config(api_key)
            elif name == "geak_set_model_config":
                result = await tools.set_model_config(api_key, **arguments)
            elif name == "geak_delete_model_config":
                result = await tools.delete_model_config(api_key)
            elif name == "geak_create_task":
                result = await tools.create_task(api_key, **arguments)
            elif name == "geak_get_task":
                result = await tools.get_task(api_key, **arguments)
            elif name == "geak_list_tasks":
                result = await tools.list_tasks(api_key, **arguments)
            elif name == "geak_submit_task":
                result = await tools.submit_task(api_key, **arguments)
            elif name == "geak_cancel_task":
                result = await tools.cancel_task(api_key, **arguments)
            elif name == "geak_get_outputs":
                result = await tools.get_outputs(api_key, **arguments)
            elif name == "geak_download_file":
                result = await tools.download_file(api_key, **arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
            
            return [TextContent(type="text", text=json.dumps(result, indent=2, ensure_ascii=False))]
            
        except Exception as e:
            logger.exception(f"Error in tool {name}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
    
    return server


async def run_stdio_server(api_key: str | None = None):
    """Run MCP server with stdio transport."""
    server = create_mcp_server(api_key)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    import os
    
    # Get API key from environment
    api_key = os.getenv("GEAK_API_KEY")
    
    # Run stdio server
    asyncio.run(run_stdio_server(api_key))
