"""Tests for velari_ai.integrations.fastmcp.client_fastmcp."""

import asyncio

import pytest
from omegaconf import OmegaConf


def _run(coro):
    return asyncio.run(coro)


def _make_server():
    from velari_ai.integrations.fastmcp.server import MCPServer
    from velari_ai.integrations.fastmcp.types import ResourceSpec

    def get_time() -> str:
        """Return the current time."""
        return "now"

    def fail_tool() -> str:
        """A tool that always fails."""
        raise ValueError("boom")

    def read_config() -> dict:
        return {"key": "value"}

    server = MCPServer.from_config(OmegaConf.create({"name": "test-server", "instructions": ""}))
    server.register_tools([get_time, fail_tool])
    server.register_resources([
        ResourceSpec(uri="config://settings", fn=read_config, name="read_config", mime_type="application/json"),
    ])
    return server


def test_connect_and_is_connected():
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        client = FastMCPClient(_make_server())
        assert client.is_connected() is False
        await client.connect()
        assert client.is_connected() is True
        await client.close()

    _run(_body())


def test_close_disconnects():
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        client = FastMCPClient(_make_server())
        await client.connect()
        await client.close()
        return client.is_connected()

    assert _run(_body()) is False


def test_async_context_manager_connects_and_closes():
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        client = FastMCPClient(_make_server())
        async with client as entered:
            assert entered is client
            assert client.is_connected() is True
        return client.is_connected()

    assert _run(_body()) is False


def test_discover_tools_lists_search_transform_tools():
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        async with FastMCPClient(_make_server()) as client:
            return await client.discover_tools()

    tools = _run(_body())

    # BM25SearchTransform/CodeMode (added unconditionally in MCPServer.__init__) expose
    # a search-based interface rather than raw registered tool names.
    assert {t.name for t in tools} == {"search", "get_schema", "execute"}


def test_discover_resources_returns_registered_resource():
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        async with FastMCPClient(_make_server()) as client:
            return await client.discover_resources()

    resources = _run(_body())

    assert [r.name for r in resources] == ["read_config"]


def test_discover_capabilities_returns_server_capabilities():
    from mcp.types import ServerCapabilities
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        async with FastMCPClient(_make_server()) as client:
            return await client.discover_capabilities()

    caps = _run(_body())

    assert isinstance(caps, ServerCapabilities)


def test_discover_capabilities_raises_when_not_connected():
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    client = FastMCPClient(_make_server())

    with pytest.raises(RuntimeError):
        _run(client.discover_capabilities())


def test_read_resource_returns_contents():
    from mcp.types import TextResourceContents
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        async with FastMCPClient(_make_server()) as client:
            return await client.read_resource("config://settings")

    contents = _run(_body())

    assert len(contents) == 1
    assert isinstance(contents[0], TextResourceContents)
    assert "key" in contents[0].text


def test_call_tool_returns_result_with_data():
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        async with FastMCPClient(_make_server()) as client:
            return await client.call_tool("get_time", {})

    result = _run(_body())

    assert result.is_error is False
    assert result.data == "now"


def test_call_tool_error_returns_is_error_true_not_exception():
    from velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient

    async def _body():
        async with FastMCPClient(_make_server()) as client:
            return await client.call_tool("fail_tool", {})

    # raise_on_error=False parity with MCPClient.call_tool(): a tool-level error comes
    # back as a flagged result, not a raised exception.
    result = _run(_body())

    assert result.is_error is True


def test_multiservermcpclient_wraps_servers_into_mcpconfig(monkeypatch):
    import velari_ai.integrations.fastmcp.client_fastmcp as client_module
    from velari_ai.integrations.fastmcp.client_fastmcp import MultiServerMCPClient

    captured = {}

    def _fake_client(transport, **kwargs):
        captured["transport"] = transport
        return object()

    monkeypatch.setattr(client_module, "Client", _fake_client)
    servers = {"search": {"transport": "stdio", "command": "python", "args": ["server.py"]}}

    MultiServerMCPClient(servers)

    assert captured["transport"] == {"mcpServers": servers}
