"""Tests for velari_ai.integrations.langchain.tools.tool."""


def test_mcp_tools_to_langchain_converts_each_tool(monkeypatch):
    from mcp.types import Tool
    from langchain_mcp_adapters.sessions import StdioConnection
    import velari_ai.integrations.langchain.tools.tool as tool_module
    from velari_ai.integrations.langchain.tools.tool import mcp_tools_to_langchain

    captured = []

    def _fake_convert(*, session, tool, connection, server_name):
        captured.append((tool, connection, server_name))
        return object()

    monkeypatch.setattr(tool_module, "convert_mcp_tool_to_langchain_tool", _fake_convert)

    tool = Tool(name="get_time", description="Return the current time.", inputSchema={"type": "object", "properties": {}})
    connection: StdioConnection = {"transport": "stdio", "command": "python", "args": ["server.py"]}
    result = mcp_tools_to_langchain([tool], connection, server_name="search")

    assert len(result) == 1
    tool, conn, server_name = captured[0]
    assert tool.name == "get_time"
    assert conn == connection
    assert server_name == "search"
