# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.

"""
MCP transport layer - handles JSON-RPC communication over stdio.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class StdioTransport:
    """Stdio transport for MCP protocol using JSON-RPC 2.0."""

    def __init__(self, process):
        """
        Initialize transport with a subprocess.

        Args:
            process: asyncio subprocess with stdin/stdout/stderr
        """
        self.process = process
        self._request_id = 0

    def get_next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id

    async def send(self, request: dict[str, Any]) -> None:
        """
        Send JSON-RPC request via stdin.

        Args:
            request: JSON-RPC request dict
        """
        message = json.dumps(request) + "\n"
        logger.debug(f"Sending request: {request}")
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        """
        Receive JSON-RPC response from stdout.

        Returns:
            JSON-RPC response dict
        """
        line = await self.process.stdout.readline()
        if not line:
            raise EOFError("MCP server closed connection")

        response = json.loads(line.decode())
        logger.debug(f"Received response: {response}")
        return response

    async def send_and_receive(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Send request and wait for response.

        Args:
            request: JSON-RPC request dict

        Returns:
            JSON-RPC response dict
        """
        await self.send(request)
        return await self.receive()

    async def close(self) -> None:
        """Close the transport."""
        if self.process and self.process.stdin:
            self.process.stdin.close()
            await self.process.wait()
