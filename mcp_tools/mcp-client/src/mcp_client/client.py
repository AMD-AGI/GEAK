"""
MCP Protocol Client - implements JSON-RPC 2.0 over stdio.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from .config import get_server_config
from .transport import StdioTransport

logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP protocol client for communicating with MCP servers.
    
    Implements JSON-RPC 2.0 protocol over stdio transport.
    """
    
    def __init__(self, server_name: str, server_config: dict[str, Any] | None = None):
        """
        Initialize MCP client.
        
        Args:
            server_name: Name of the MCP server to connect to
            server_config: Optional custom server configuration (overrides default)
        """
        self.server_name = server_name
        self.server_config = server_config or get_server_config(server_name)
        self.process = None
        self.transport = None
        self._initialized = False
    
    async def start(self) -> None:
        """
        Start MCP server process and initialize connection.
        
        Raises:
            RuntimeError: If server fails to start
        """
        if self.process:
            logger.warning("MCP server already running")
            return
        
        cmd = self.server_config["command"]
        cwd = self.server_config.get("cwd")
        env = os.environ.copy()
        
        # Add custom environment variables
        if "env" in self.server_config:
            env.update(self.server_config["env"])
        
        logger.info(f"Starting MCP server: {self.server_name}")
        logger.debug(f"Command: {' '.join(cmd)}")
        logger.debug(f"CWD: {cwd}")
        
        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                # Raise the default 64KB readline limit to 16MB as a safety net
                # for MCP servers that haven't adopted the file-based transport
                # convention for large responses (e.g. profiler output with
                # extremely long C++ mangled kernel names).
                limit=16 * 1024 * 1024,
            )
            
            self.transport = StdioTransport(self.process)
            
            # Initialize MCP session
            await self._initialize()
            
            logger.info(f"MCP server {self.server_name} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            if self.process:
                self.process.kill()
                await self.process.wait()
            raise RuntimeError(f"Failed to start MCP server {self.server_name}: {e}")
    
    async def _initialize(self) -> None:
        """
        Send initialize request to MCP server.
        
        JSON-RPC method: initialize
        """
        request = {
            "jsonrpc": "2.0",
            "id": self.transport.get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "mcp-client",
                    "version": "0.1.0"
                }
            }
        }
        
        response = await self.transport.send_and_receive(request)
        
        if "error" in response:
            raise RuntimeError(f"Initialize failed: {response['error']}")
        
        self._initialized = True
        logger.debug("MCP session initialized")
    
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments as dict
            
        Returns:
            Tool result dict
            
        Raises:
            RuntimeError: If not initialized or tool call fails
        """
        if not self._initialized:
            raise RuntimeError("MCP client not initialized. Call start() first.")
        
        logger.info(f"Calling tool: {tool_name}")
        logger.debug(f"Arguments: {arguments}")
        
        request = {
            "jsonrpc": "2.0",
            "id": self.transport.get_next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self.transport.send_and_receive(request)
        
        if "error" in response:
            error = response["error"]
            logger.error(f"Tool call failed: {error}")
            raise RuntimeError(f"Tool {tool_name} failed: {error.get('message', str(error))}")
        
        result = response.get("result", {})

        # Handle file-based large responses: when an MCP server writes a
        # large result to disk instead of inlining it in the JSON-RPC
        # response, it returns {"_result_file": "/path/to/result.json"}.
        # We transparently read the file, delete it, and return the contents.
        if isinstance(result, dict) and "_result_file" in result:
            result_path = Path(result["_result_file"])
            if result_path.exists():
                try:
                    with open(result_path) as f:
                        result = json.load(f)
                    result_path.unlink(missing_ok=True)
                    logger.debug("Loaded large result from file: %s", result_path)
                except Exception as exc:
                    logger.warning("Failed to load _result_file %s: %s", result_path, exc)
            else:
                logger.warning("_result_file does not exist: %s", result_path)

        logger.debug(f"Tool result: {result}")
        
        return result
    
    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List available tools from MCP server.
        
        Returns:
            List of tool definitions
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self._initialized:
            raise RuntimeError("MCP client not initialized. Call start() first.")
        
        request = {
            "jsonrpc": "2.0",
            "id": self.transport.get_next_id(),
            "method": "tools/list",
            "params": {}
        }
        
        response = await self.transport.send_and_receive(request)
        
        if "error" in response:
            raise RuntimeError(f"List tools failed: {response['error']}")
        
        tools = response.get("result", {}).get("tools", [])
        logger.info(f"Available tools: {[t.get('name') for t in tools]}")
        
        return tools
    
    async def stop(self) -> None:
        """
        Stop MCP server and close connection.
        """
        if not self.process:
            return
        
        logger.info(f"Stopping MCP server: {self.server_name}")
        
        try:
            if self.transport:
                await self.transport.close()
        except Exception as e:
            logger.warning(f"Error closing transport: {e}")
        
        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("MCP server did not terminate, killing process")
            self.process.kill()
            await self.process.wait()
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
        
        self.process = None
        self.transport = None
        self._initialized = False
        
        logger.info("MCP server stopped")
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()
        return False


async def call_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    server_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Convenience function to call an MCP tool.
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to call
        arguments: Tool arguments
        server_config: Optional custom server configuration
        
    Returns:
        Tool result dict
        
    Example:
        >>> result = await call_mcp_tool(
        ...     "openevolve-mcp",
        ...     "optimize_kernel",
        ...     {"kernel_path": "kernel.py", "max_iterations": 10}
        ... )
    """
    async with MCPClient(server_name, server_config) as client:
        return await client.call_tool(tool_name, arguments)
