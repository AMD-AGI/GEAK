"""Unit tests for ``minisweagent.tools.mcp_client``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from minisweagent.tools.mcp_client.client import MCPClient, call_mcp_tool
from minisweagent.tools.mcp_client.config import MCP_SERVERS, get_server_config, list_servers
from minisweagent.tools.mcp_client.transport import StdioTransport


@pytest.fixture
def dummy_server_config(tmp_path: Path) -> dict:
    return {
        "command": ["python3", "-c", "import sys; sys.exit(0)"],
        "cwd": str(tmp_path),
        "env": {},
    }


class TestMCPConfig:
    def test_get_server_config_returns_registered_server(self) -> None:
        cfg = get_server_config("rag-mcp")
        assert "command" in cfg
        assert "tools" in cfg
        assert isinstance(cfg["command"], list)

    def test_get_server_config_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="Unknown MCP server"):
            get_server_config("not-a-real-server-xyz")

    def test_list_servers_covers_all_mcp_servers_keys(self) -> None:
        desc = list_servers()
        assert set(desc.keys()) == set(MCP_SERVERS.keys())
        for name in MCP_SERVERS:
            assert isinstance(desc[name], str)


class TestStdioTransport:
    def test_get_next_id_increments(self) -> None:
        proc = MagicMock()
        t = StdioTransport(proc)
        assert t.get_next_id() == 1
        assert t.get_next_id() == 2

    def test_send_writes_json_line(self) -> None:
        async def _run() -> None:
            stdin = MagicMock()
            stdin.write = MagicMock()
            stdin.drain = AsyncMock()
            proc = MagicMock()
            proc.stdin = stdin
            t = StdioTransport(proc)
            await t.send({"jsonrpc": "2.0", "id": 1})
            stdin.write.assert_called_once()
            written = stdin.write.call_args[0][0].decode()
            assert json.loads(written.strip())["id"] == 1
            stdin.drain.assert_awaited_once()

        asyncio.run(_run())

    def test_receive_parses_json_line(self) -> None:
        async def _run() -> None:
            proc = MagicMock()
            proc.stdout = AsyncMock()
            proc.stdout.readline = AsyncMock(return_value=b'{"jsonrpc":"2.0","result":{}}\n')
            t = StdioTransport(proc)
            out = await t.receive()
            assert out["result"] == {}

        asyncio.run(_run())

    def test_receive_eof_raises(self) -> None:
        async def _run() -> None:
            proc = MagicMock()
            proc.stdout = AsyncMock()
            proc.stdout.readline = AsyncMock(return_value=b"")
            t = StdioTransport(proc)
            with pytest.raises(ConnectionError, match="closed connection"):
                await t.receive()

        asyncio.run(_run())

    def test_send_and_receive(self) -> None:
        async def _run() -> None:
            proc = MagicMock()
            stdin = MagicMock()
            stdin.write = MagicMock()
            stdin.drain = AsyncMock()
            proc.stdin = stdin
            proc.stdout = AsyncMock()
            proc.stdout.readline = AsyncMock(return_value=b'{"id":99}\n')
            t = StdioTransport(proc)
            resp = await t.send_and_receive({"id": 1})
            assert resp["id"] == 99

        asyncio.run(_run())

    def test_close_closes_stdin(self) -> None:
        async def _run() -> None:
            stdin = MagicMock()
            stdin.close = MagicMock()
            proc = MagicMock()
            proc.stdin = stdin
            t = StdioTransport(proc)
            await t.close()
            stdin.close.assert_called_once()

        asyncio.run(_run())


class TestMCPClient:
    def test_init_uses_explicit_config(self, dummy_server_config: dict) -> None:
        c = MCPClient("custom", dummy_server_config)
        assert c.server_config is dummy_server_config
        assert c.server_name == "custom"

    def test_call_tool_requires_initialization(self, dummy_server_config: dict) -> None:
        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            with pytest.raises(RuntimeError, match="not initialized"):
                await c.call_tool("t", {})

        asyncio.run(_run())

    def test_list_tools_requires_initialization(self, dummy_server_config: dict) -> None:
        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            with pytest.raises(RuntimeError, match="not initialized"):
                await c.list_tools()

        asyncio.run(_run())

    def test_call_tool_inline_result(self, dummy_server_config: dict) -> None:
        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            c._initialized = True
            c.transport = MagicMock()
            c.transport.get_next_id = MagicMock(return_value=1)
            c.transport.send_and_receive = AsyncMock(return_value={"result": {"ok": True}})
            out = await c.call_tool("my_tool", {"a": 1})
            assert out == {"ok": True}

        asyncio.run(_run())

    def test_call_tool_server_error(self, dummy_server_config: dict) -> None:
        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            c._initialized = True
            c.transport = MagicMock()
            c.transport.get_next_id = MagicMock(return_value=1)
            c.transport.send_and_receive = AsyncMock(
                return_value={"error": {"message": "boom", "code": -1}}
            )
            with pytest.raises(RuntimeError, match="boom"):
                await c.call_tool("t", {})

        asyncio.run(_run())

    def test_call_tool_reads_result_file_and_unlinks(
        self, tmp_path: Path, dummy_server_config: dict
    ) -> None:
        result_file = tmp_path / "large.json"
        result_file.write_text(json.dumps({"data": [1, 2, 3]}))

        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            c._initialized = True
            c.transport = MagicMock()
            c.transport.get_next_id = MagicMock(return_value=1)
            c.transport.send_and_receive = AsyncMock(
                return_value={"result": {"_result_file": str(result_file)}}
            )
            out = await c.call_tool("t", {})
            assert out == {"data": [1, 2, 3]}
            assert not result_file.exists()

        asyncio.run(_run())

    def test_call_tool_missing_result_file_keeps_wrapper(
        self, tmp_path: Path, dummy_server_config: dict
    ) -> None:
        missing = tmp_path / "nope.json"

        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            c._initialized = True
            c.transport = MagicMock()
            c.transport.get_next_id = MagicMock(return_value=1)
            c.transport.send_and_receive = AsyncMock(
                return_value={"result": {"_result_file": str(missing)}}
            )
            out = await c.call_tool("t", {})
            assert out == {"_result_file": str(missing)}

        asyncio.run(_run())

    def test_list_tools_success(self, dummy_server_config: dict) -> None:
        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            c._initialized = True
            c.transport = MagicMock()
            c.transport.get_next_id = MagicMock(return_value=1)
            c.transport.send_and_receive = AsyncMock(
                return_value={"result": {"tools": [{"name": "alpha"}, {"name": "beta"}]}}
            )
            tools = await c.list_tools()
            assert [t["name"] for t in tools] == ["alpha", "beta"]

        asyncio.run(_run())

    def test_list_tools_error(self, dummy_server_config: dict) -> None:
        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            c._initialized = True
            c.transport = MagicMock()
            c.transport.get_next_id = MagicMock(return_value=1)
            c.transport.send_and_receive = AsyncMock(return_value={"error": {"message": "nope"}})
            with pytest.raises(RuntimeError, match="List tools failed"):
                await c.list_tools()

        asyncio.run(_run())

    def test_stop_no_process_is_safe(self, dummy_server_config: dict) -> None:
        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            c.process = None
            await c.stop()

        asyncio.run(_run())


class TestCallMcpTool:
    def test_convenience_wraps_client(self) -> None:
        cfg = {"command": ["true"], "cwd": ".", "env": {}}

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.call_tool = AsyncMock(return_value={"r": 1})

        async def _run() -> None:
            with patch("minisweagent.tools.mcp_client.client.MCPClient", return_value=mock_client):
                out = await call_mcp_tool("srv", "tool1", {"x": 2}, cfg)
            assert out == {"r": 1}
            mock_client.call_tool.assert_awaited_once_with("tool1", {"x": 2})

        asyncio.run(_run())


class TestMCPClientStart:
    def test_start_already_running_returns_early(self, dummy_server_config: dict) -> None:
        async def _run() -> None:
            c = MCPClient("x", dummy_server_config)
            fake_proc = MagicMock()
            c.process = fake_proc
            await c.start()
            assert c.process is fake_proc

        asyncio.run(_run())
