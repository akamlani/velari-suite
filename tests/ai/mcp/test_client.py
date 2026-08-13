"""Tests for velari_ai.integrations.fastmcp.client."""

import asyncio
import logging


def _run(coro):
    return asyncio.run(coro)


def _make_dummy_client():
    from velari_ai.integrations.fastmcp.client import BaseMCPClient
    from mcp.types import ServerCapabilities

    class _DummyClient(BaseMCPClient):
        def __init__(self):
            self.connected = False
            self.closed = False

        async def connect(self) -> None:
            self.connected = True

        async def close(self) -> None:
            self.closed = True

        def is_connected(self) -> bool:
            return self.connected

        async def discover_tools(self, verbose: bool = False):
            return []

        async def discover_resources(self, verbose: bool = False):
            return []

        async def discover_prompts(self, verbose: bool = False):
            return []

        async def discover_capabilities(self, verbose: bool = False):
            return ServerCapabilities()

        async def read_resource(self, uri: str, verbose: bool = False):
            return []

        async def call_tool(self, name: str, arguments=None, verbose: bool = False):
            return None

    return _DummyClient()


def test_aenter_calls_connect_and_returns_self():
    client = _make_dummy_client()

    result = _run(client.__aenter__())

    assert client.connected is True
    assert result is client


def test_aexit_calls_close():
    client = _make_dummy_client()

    _run(client.__aexit__(None, None, None))

    assert client.closed is True


def test_log_preview_logs_dataframe_with_selected_columns(caplog):
    from velari_ai.integrations.fastmcp.client import BaseMCPClient
    from mcp.types import Tool

    tools = [Tool(name="get_time", description="Return the current time.", inputSchema={"type": "object", "properties": {}})]

    with caplog.at_level(logging.INFO, logger="velari_ai.integrations.fastmcp.client"):
        BaseMCPClient._log_preview(tools, ["display_name", "name", "description"])

    assert "get_time" in caplog.text
    assert "Return the current time." in caplog.text


def test_log_preview_skips_empty_items(caplog):
    from velari_ai.integrations.fastmcp.client import BaseMCPClient

    with caplog.at_level(logging.INFO, logger="velari_ai.integrations.fastmcp.client"):
        BaseMCPClient._log_preview([])

    assert caplog.text == ""
