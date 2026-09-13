import  pandas as pd
from    typing import List, Optional, Sequence, TypedDict

from    langchain_core.tools import BaseTool

from    mcp.types import Tool
from    langchain_mcp_adapters.sessions import Connection
from    langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool


class ToolArtifact(TypedDict):
    name: str
    description: str

def get_tool_info(tools: List[BaseTool]) -> pd.DataFrame:
    """Return a DataFrame of tool names and descriptions for logging or display."""
    return pd.DataFrame([{"name": t.name, "description": t.description} for t in tools])

def get_tool_artifacts(tools: List[BaseTool]) -> List[ToolArtifact]:
    return [ToolArtifact({"name": t.name, "description": t.description}) for t in tools]


def mcp_tools_to_langchain(
    tools: Sequence[Tool],
    connection: Connection,
    server_name: Optional[str] = None,
) -> List[BaseTool]:
    """Convert MCP tool schemas to LangChain tools via langchain_mcp_adapters.

    Each returned tool reconnects through `connection` fresh per call — no persistent
    MCP session is kept between calls.

    Args:
        tools (Sequence[Tool]): Raw MCP tool schemas, e.g. from `MultiServerMCPClient.discover_tools()`.
        connection (Connection): Stdio/remote connection config used to reach the server.
        server_name (Optional[str]): Attributed to each tool for error messages/logging.

    Returns:
        List[BaseTool]: One LangChain tool per MCP tool, ready for `Agent.build(tools=...)`.

    Examples:
        >>> connection = {"transport": "stdio", "command": sys.executable, "args": ["search_server.py"]}
        >>> async with MultiServerMCPClient({"search": connection}) as client:
        ...     raw_tools = await client.discover_tools()
        >>> tools = mcp_tools_to_langchain(raw_tools, connection, server_name="search")
    """
    return [
        convert_mcp_tool_to_langchain_tool(session=None, tool=tool, connection=connection, server_name=server_name)
        for tool in tools
    ]
