# MCP Client

Proper MCP protocol client for communicating with MCP servers via JSON-RPC 2.0.

## Overview

This package provides a real MCP (Model Context Protocol) client that communicates with MCP servers using the JSON-RPC 2.0 protocol over stdio transport. Unlike direct Python imports, this uses the actual MCP protocol for server communication.

## Features

- **Real Protocol**: JSON-RPC 2.0 over stdio transport
- **Async/Await**: Full asyncio support for non-blocking operations
- **Server Registry**: Pre-configured registry of available MCP servers
- **Lifecycle Management**: Automatic server startup/shutdown
- **Context Managers**: Clean resource management with `async with`
- **Type Hints**: Full type annotations for better IDE support
- **Logging**: Comprehensive logging for debugging

## Installation

```bash
pip install -e mcp_tools/mcp-client/
```

## Architecture

```
┌─────────────┐         JSON-RPC 2.0         ┌──────────────┐
│  MCPClient  │ ◄─────────────────────────► │  MCP Server  │
│  (Python)   │       over stdio            │  (FastMCP)   │
└─────────────┘                              └──────────────┘
```

**Communication Flow:**
1. Client starts MCP server subprocess
2. Client sends JSON-RPC requests via stdin
3. Server processes requests and responds via stdout
4. Client parses responses and returns results

## Usage

### Basic Usage

```python
import asyncio
from mcp_client import MCPClient

async def main():
    # Create client for openevolve-mcp server
    client = MCPClient("openevolve-mcp")
    
    # Start server
    await client.start()
    
    # Call a tool
    result = await client.call_tool("optimize_kernel", {
        "kernel_path": "examples/add_kernel/kernel.py",
        "max_iterations": 10
    })
    
    print(f"Optimization result: {result}")
    
    # Stop server
    await client.stop()

asyncio.run(main())
```

### Using Context Manager (Recommended)

```python
import asyncio
from mcp_client import MCPClient

async def main():
    async with MCPClient("openevolve-mcp") as client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[t['name'] for t in tools]}")
        
        # Call a tool
        result = await client.call_tool("optimize_kernel", {
            "kernel_path": "kernel.py",
            "max_iterations": 10
        })

asyncio.run(main())
```

### Convenience Function

```python
import asyncio
from mcp_client import call_mcp_tool

async def main():
    # One-shot tool call
    result = await call_mcp_tool(
        server_name="openevolve-mcp",
        tool_name="optimize_kernel",
        arguments={
            "kernel_path": "kernel.py",
            "max_iterations": 10
        }
    )

asyncio.run(main())
```

### Custom Server Configuration

```python
from mcp_client import MCPClient

custom_config = {
    "command": ["python3", "-m", "custom_server"],
    "cwd": "/path/to/server",
    "env": {
        "PYTHONPATH": "/path/to/server/src",
        "CUSTOM_VAR": "value"
    }
}

async with MCPClient("custom-server", custom_config) as client:
    result = await client.call_tool("my_tool", {})
```

## Available Servers

The client comes pre-configured with the following MCP servers:

- **openevolve-mcp**: OpenEvolve optimizer - LLM-guided kernel evolution
- **kernel-profiler**: GPU kernel profiler using rocprof-compute
- **kernel-evolve**: LLM-based kernel mutation and crossover
- **kernel-ercs**: Kernel evaluation, reflection, and compatibility checking
- **automated-test-discovery**: Automated test and benchmark discovery

List all servers:

```python
from mcp_client import list_servers

servers = list_servers()
for name, description in servers.items():
    print(f"{name}: {description}")
```

## API Reference

### MCPClient

```python
class MCPClient:
    def __init__(self, server_name: str, server_config: Optional[Dict] = None)
    async def start() -> None
    async def stop() -> None
    async def call_tool(tool_name: str, arguments: Dict) -> Dict
    async def list_tools() -> List[Dict]
```

### Functions

```python
async def call_mcp_tool(
    server_name: str,
    tool_name: str,
    arguments: Dict,
    server_config: Optional[Dict] = None
) -> Dict
```

## Integration with GEAK Agent

The GEAK agent optimizer automatically uses MCP protocol when the client is installed:

```python
from minisweagent.optimizer import optimize_kernel, OptimizerType

# This will use MCP protocol if mcp-client is installed
result = optimize_kernel(
    kernel_code=my_kernel,
    optimizer=OptimizerType.OPENEVOLVE,
    max_iterations=10
)

# Falls back to direct import if mcp-client not installed
```

## Protocol Details

### JSON-RPC 2.0 Format

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "optimize_kernel",
    "arguments": {
      "kernel_path": "kernel.py",
      "max_iterations": 10
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "success": true,
    "optimized_code": "...",
    "metrics": {...}
  }
}
```

### Supported Methods

- `initialize`: Initialize MCP session
- `tools/list`: List available tools
- `tools/call`: Execute a tool

## Error Handling

```python
import asyncio
from mcp_client import MCPClient

async def main():
    try:
        async with MCPClient("openevolve-mcp") as client:
            result = await client.call_tool("optimize_kernel", {
                "kernel_path": "nonexistent.py"
            })
    except RuntimeError as e:
        print(f"Tool call failed: {e}")
    except KeyError as e:
        print(f"Unknown server: {e}")

asyncio.run(main())
```

## Logging

Enable debug logging to see protocol messages:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp_client")
logger.setLevel(logging.DEBUG)
```

## Comparison: MCP Protocol vs Direct Import

### MCP Protocol (This Client)
```python
# Real protocol communication
async with MCPClient("openevolve-mcp") as client:
    result = await client.call_tool("optimize_kernel", {...})
```

**Pros:**
- Uses actual MCP protocol (JSON-RPC 2.0)
- Server runs in separate process
- Can connect to remote servers
- Clean separation of concerns
- Testable protocol communication

**Cons:**
- Requires async/await
- Slight overhead from process communication

### Direct Import (Fallback)
```python
# Direct Python function call
from openevolve_mcp.server import _optimize_kernel_impl
result = _optimize_kernel_impl(...)
```

**Pros:**
- Simpler synchronous code
- No process overhead

**Cons:**
- Not real MCP protocol
- Tight coupling
- Can't use remote servers
- Harder to test

## Testing

Test the client with a server:

```bash
# Terminal 1: Start server manually
cd mcp_tools/openevolve-mcp
python3 -m openevolve_mcp.server

# Terminal 2: Test client
python3 << EOF
import asyncio
from mcp_client import MCPClient

async def test():
    async with MCPClient("openevolve-mcp") as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")

asyncio.run(test())
EOF
```

## Contributing

When adding new MCP servers, update the `MCP_SERVERS` registry in `src/mcp_client/config.py`.

## License

Part of the GEAK Agent project.
