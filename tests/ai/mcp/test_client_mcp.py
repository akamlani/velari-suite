"""Tests for velari_ai.integrations.fastmcp.client_mcp."""

import asyncio
import logging
from types import SimpleNamespace

import pytest


def _run(coro):
    return asyncio.run(coro)


class _FakeAsyncContextManager:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, tools=None, resources=None, prompts=None, capabilities=None, contents=None, tool_result=None):
        self.tools = tools or []
        self.resources = resources or []
        self.prompts = prompts or []
        self.capabilities = capabilities
        self.contents = contents or []
        self.tool_result = tool_result
        self.read_resource_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return SimpleNamespace(capabilities=self.capabilities)

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def list_resources(self):
        return SimpleNamespace(resources=self.resources)

    async def list_prompts(self):
        return SimpleNamespace(prompts=self.prompts)

    async def read_resource(self, uri):
        self.read_resource_calls.append(uri)
        return SimpleNamespace(contents=self.contents)

    async def call_tool(self, name, arguments):
        return self.tool_result


def _connected_client(monkeypatch, session):
    import velari_ai.integrations.fastmcp.client_mcp as client_mcp_module

    monkeypatch.setattr(client_mcp_module, "stdio_client", lambda params: _FakeAsyncContextManager((object(), object())))
    monkeypatch.setattr(client_mcp_module, "ClientSession", lambda read, write: session)

    client = client_mcp_module.MCPClient(command="python", args=["server.py"])
    _run(client.connect())
    return client


def test_is_connected_false_before_connect():
    from velari_ai.integrations.fastmcp.client_mcp import MCPClient

    client = MCPClient(command="python", args=["server.py"])

    assert client.is_connected() is False


def test_connect_establishes_session_and_capabilities(monkeypatch):
    from mcp.types import ServerCapabilities

    caps = ServerCapabilities()
    client = _connected_client(monkeypatch, _FakeSession(capabilities=caps))

    assert client.is_connected() is True


def test_close_resets_session(monkeypatch):
    client = _connected_client(monkeypatch, _FakeSession())

    _run(client.close())

    assert client.is_connected() is False


def test_discover_tools_not_connected_raises():
    from velari_ai.integrations.fastmcp.client_mcp import MCPClient

    client = MCPClient(command="python", args=["server.py"])

    with pytest.raises(RuntimeError):
        _run(client.discover_tools())


def test_discover_tools_returns_tools(monkeypatch):
    from mcp.types import Tool

    tools = [Tool(name="get_time", description="Return the current time.", inputSchema={"type": "object", "properties": {}})]
    client = _connected_client(monkeypatch, _FakeSession(tools=tools))

    result = _run(client.discover_tools())

    assert result == tools


def test_discover_resources_returns_resources(monkeypatch):
    from mcp.types import Resource

    resources = [Resource(uri="config://settings", name="read_config")]
    client = _connected_client(monkeypatch, _FakeSession(resources=resources))

    result = _run(client.discover_resources())

    assert result == resources


def test_discover_prompts_returns_prompts(monkeypatch):
    from mcp.types import Prompt

    prompts = [Prompt(name="greet")]
    client = _connected_client(monkeypatch, _FakeSession(prompts=prompts))

    result = _run(client.discover_prompts())

    assert result == prompts


def test_discover_capabilities_not_connected_raises():
    from velari_ai.integrations.fastmcp.client_mcp import MCPClient

    client = MCPClient(command="python", args=["server.py"])

    with pytest.raises(RuntimeError):
        _run(client.discover_capabilities())


def test_discover_capabilities_returns_capabilities(monkeypatch):
    from mcp.types import ServerCapabilities

    caps = ServerCapabilities()
    client = _connected_client(monkeypatch, _FakeSession(capabilities=caps))

    result = _run(client.discover_capabilities())

    assert result is caps


def test_read_resource_not_connected_raises():
    from velari_ai.integrations.fastmcp.client_mcp import MCPClient

    client = MCPClient(command="python", args=["server.py"])

    with pytest.raises(RuntimeError):
        _run(client.read_resource("config://settings"))


def test_read_resource_converts_uri_to_anyurl(monkeypatch):
    from pydantic import AnyUrl

    session = _FakeSession(contents=[SimpleNamespace(text="hello")])
    client = _connected_client(monkeypatch, session)

    result = _run(client.read_resource("config://settings"))

    assert result[0].text == "hello"
    assert isinstance(session.read_resource_calls[0], AnyUrl)
    assert str(session.read_resource_calls[0]) == "config://settings"


def test_call_tool_returns_result(monkeypatch):
    tool_result = SimpleNamespace(content=[], isError=False, structuredContent=None, meta=None)
    client = _connected_client(monkeypatch, _FakeSession(tool_result=tool_result))

    result = _run(client.call_tool("get_time", {}))

    assert result is tool_result


def test_call_tool_verbose_logs_error_and_structured_content(monkeypatch, caplog):
    tool_result = SimpleNamespace(
        content=[SimpleNamespace(text="boom")],
        isError=True,
        structuredContent={"result": "boom"},
        meta={"source": "test"},
    )
    client = _connected_client(monkeypatch, _FakeSession(tool_result=tool_result))

    with caplog.at_level(logging.INFO, logger="velari_ai.integrations.fastmcp.client_mcp"):
        _run(client.call_tool("get_time", {}, verbose=True))

    assert "isError=True" in caplog.text
    assert "boom" in caplog.text


def test_transform_tools_builds_dictconfigs():
    from mcp.types import Tool
    from velari_ai.integrations.fastmcp.client_mcp import MCPClient

    tools = [Tool(name="get_time", description="Return the current time.", inputSchema={"type": "object", "properties": {}})]

    result = MCPClient.transform_tools(tools)

    assert result[0]["name"] == "get_time"
    assert result[0]["description"] == "Return the current time."
