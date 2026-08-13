from __future__ import annotations

import  logging
from    typing  import Any, Dict, Optional, Sequence

from    fastmcp                import Client
from    fastmcp.client.client  import CallToolResult
from    fastmcp.exceptions     import FastMCPError
from    mcp.types              import Tool, Resource, Prompt, ResourceContents, ServerCapabilities
# package modules
from    .client import BaseMCPClient

logger = logging.getLogger(__name__)


class FastMCPClient(BaseMCPClient):
    """MCP client built on fastmcp 3.x's `Client`, matching `MCPClient`'s discovery/call
    surface but delegating transport handling (stdio/HTTP/in-process/etc.) to fastmcp.
    """

    def __init__(self, transport: Any, **client_kwargs: Any) -> None:
        # transport: a ClientTransport, in-process FastMCP instance, URL, Path,
        # MCPConfig/dict, or a plain str (e.g. a local ".py" script path — fastmcp infers
        # PythonStdioTransport for those automatically via infer_transport()).
        self._client: Client = Client(transport, **client_kwargs)

    async def connect(self) -> None:
        try:
            await self._client.__aenter__()
        except Exception as e:
            logger.error(f"failed to connect to MCP server: {e}")
            raise

    async def close(self) -> None:
        try:
            await self._client.close()
        except Exception as e:
            logger.warning(f"error while closing MCP client: {e}")

    def is_connected(self) -> bool:
        return self._client.is_connected()

    async def discover_tools(self, verbose: bool = False) -> Sequence[Tool]:
        try:
            tools = await self._client.list_tools()
        except Exception as e:
            logger.error(f"failed to discover tools: {e}")
            raise
        if verbose:
            self._log_preview(tools, ["display_name", "name", "description", "inputSchema"])
        return tools

    async def discover_resources(self, verbose: bool = False) -> Sequence[Resource]:
        try:
            resources = await self._client.list_resources()
        except Exception as e:
            logger.error(f"failed to discover resources: {e}")
            raise
        if verbose:
            self._log_preview(resources, ["display_name", "uri", "name", "description"])
        return resources

    async def discover_prompts(self, verbose: bool = False) -> Sequence[Prompt]:
        try:
            prompts = await self._client.list_prompts()
        except Exception as e:
            logger.error(f"failed to discover prompts: {e}")
            raise
        if verbose:
            self._log_preview(prompts)
        return prompts

    async def discover_capabilities(self, verbose: bool = False) -> ServerCapabilities:
        result = self._client.initialize_result
        if result is None:
            raise RuntimeError("FastMCPClient is not connected — call connect() first")
        if verbose:
            logger.info(f"server capabilities:\n{result.capabilities.model_dump(exclude_none=True)}")
        return result.capabilities

    async def read_resource(self, uri: str, verbose: bool = False) -> Sequence[ResourceContents]:
        try:
            contents = await self._client.read_resource(uri)
        except Exception as e:
            logger.error(f"failed to read resource {uri}: {e}")
            raise
        if verbose:
            for content in contents:
                logger.info(f"{uri}:\n{getattr(content, 'text', content)}")
        return contents

    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> CallToolResult:
        try:
            # raise_on_error=False matches MCPClient.call_tool()'s behavior: a tool-level
            # error is returned as a flagged result, not raised as an exception.
            result = await self._client.call_tool(name, arguments, raise_on_error=False)
        except FastMCPError as e:
            logger.error(f"failed to call tool {name}: {e}")
            raise
        if verbose:
            for content in result.content:
                logger.info(f"{name}:\n{getattr(content, 'text', content)}")
            if result.is_error:
                logger.warning(f"tool {name} reported is_error=True")
            if result.structured_content:
                logger.info(f"{name} structured_content:\n{result.structured_content}")
            if result.meta:
                logger.info(f"{name} meta:\n{result.meta}")
        return result


class MultiServerMCPClient(FastMCPClient):
    """FastMCPClient connected to multiple named MCP servers at once via fastmcp's own MCPConfig routing.

    Tools/resources/prompts from every server are merged into one list — `fastmcp` only
    prefixes names by server (`{server}_{name}`) when more than one server is configured;
    with a single server, names pass through unprefixed. `call_tool()`/`read_resource()`
    route back to the originating server automatically either way.

    Args:
        servers (Dict[str, Any]): Server name -> stdio/remote connection config (e.g. a
            `langchain_mcp_adapters.sessions.StdioConnection`), e.g.
            `{"search": {"transport": "stdio", "command": "python", "args": ["server.py"]}}`.

    Examples:
        >>> client = MultiServerMCPClient({
        ...     "search": {"transport": "stdio", "command": sys.executable, "args": ["search_server.py"]},
        ... })
        >>> async with client:
        ...     tools = await client.discover_tools()
        >>> [t.name for t in tools]
        ['get_time', 'search']
    """
    def __init__(self, servers: Dict[str, Any], **client_kwargs: Any) -> None:
        super().__init__({"mcpServers": servers}, **client_kwargs)
