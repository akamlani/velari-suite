"""Tests for velari_ai.integrations.fastmcp.server."""

import asyncio

from omegaconf import OmegaConf


def _run(coro):
    return asyncio.run(coro)


def test_from_config_builds_server_from_dictconfig():
    from velari_ai.integrations.fastmcp.server import MCPServer

    cfg = OmegaConf.create({"name": "test-server", "instructions": "test instructions"})

    server = MCPServer.from_config(cfg)

    assert server.name == "test-server"
    assert server.instructions == "test instructions"


def test_register_tools_registers_description_and_meta():
    from velari_ai.integrations.fastmcp.server import MCPServer

    def get_time() -> str:
        """Return the current time.

        Raises:
            RuntimeError: if the system clock is unavailable.
        """
        return "now"

    async def _body():
        server = MCPServer.from_config(OmegaConf.create({"name": "test-server", "instructions": ""}))
        server.register_tools([get_time])
        return await server.get_tool("get_time")

    tool = _run(_body())

    assert tool is not None
    assert tool.name == "get_time"
    assert tool.description == "Return the current time.\n\nRaises:\n    RuntimeError: if the system clock is unavailable."
    assert tool.meta == {"raises": [{"exception": "RuntimeError", "description": "if the system clock is unavailable."}]}


def test_register_tools_multiple_functions():
    from velari_ai.integrations.fastmcp.server import MCPServer

    def get_time() -> str:
        """Return the current time."""
        return "now"

    def get_date() -> str:
        """Return the current date."""
        return "today"

    async def _body():
        server = MCPServer.from_config(OmegaConf.create({"name": "test-server", "instructions": ""}))
        server.register_tools([get_time, get_date])
        return await server.get_tool("get_time"), await server.get_tool("get_date")

    time_tool, date_tool = _run(_body())

    assert time_tool is not None
    assert date_tool is not None
    assert time_tool.name == "get_time"
    assert date_tool.name == "get_date"


def test_register_resources_registers_uri_and_kwargs():
    from velari_ai.integrations.fastmcp.server import MCPServer
    from velari_ai.integrations.fastmcp.types import ResourceSpec

    def read_config() -> dict:
        return {"key": "value"}

    async def _body():
        server = MCPServer.from_config(OmegaConf.create({"name": "test-server", "instructions": ""}))
        server.register_resources([
            ResourceSpec(
                uri="config://settings",
                fn=read_config,
                name="read_config",
                mime_type="application/json",
                tags={"config"},
            ),
        ])
        return await server.get_resource("config://settings")

    resource = _run(_body())

    assert resource is not None
    assert resource.name == "read_config"
    assert str(resource.uri) == "config://settings"
    assert resource.mime_type == "application/json"
