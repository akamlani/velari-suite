from __future__ import annotations

import logging

from typing import List, Optional, Any, Sequence
from omegaconf import OmegaConf, DictConfig
from contextlib import AsyncExitStack
from pydantic import AnyUrl

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool, Resource, ResourceContents, ServerCapabilities, Prompt
from mcp.types import CallToolRequest, CallToolResult
from mcp.shared.metadata_utils import get_display_name
# package modules
from .client import BaseMCPClient

logger = logging.getLogger(__name__)


class MCPClient(BaseMCPClient):
    def __init__(self, command: str, args: List[str], cwd: str = "") -> None:
        self._server_params = StdioServerParameters(
            command=command,
            args=args,
            cwd=cwd or None,
        )
        self._stack: AsyncExitStack = AsyncExitStack()
        self._session: Optional[ClientSession] = None
        self._capabilities: Optional[ServerCapabilities] = None

    async def connect(self) -> None:
        try:
            read_stream, write_stream = await self._stack.enter_async_context(stdio_client(self._server_params))
            self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
            result = await self._session.initialize()
            self._capabilities = result.capabilities
        except Exception as e:
            logger.error(f"failed to connect to MCP server: {e}")
            raise

    async def close(self) -> None:
        try:
            await self._stack.aclose()
        except Exception as e:
            logger.warning(f"error while closing MCP client: {e}")
        finally:
            self._session = None

    def is_connected(self) -> bool:
        return self._session is not None

    async def discover_tools(self, verbose: bool = False) -> List[Tool]:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected — call connect() first")
        try:
            result = await self._session.list_tools()
        except Exception as e:
            logger.error(f"failed to discover tools: {e}")
            raise
        if verbose:
            self._log_preview(result.tools, ["display_name", "name", "description", "inputSchema"])
        return result.tools

    async def discover_resources(self, verbose: bool = False) -> List[Resource]:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected — call connect() first")
        try:
            result = await self._session.list_resources()
        except Exception as e:
            logger.error(f"failed to discover resources: {e}")
            raise
        if verbose:
            self._log_preview(result.resources, ["display_name", "uri", "name", "description"])
        return result.resources

    async def discover_prompts(self, verbose: bool = False) -> List[Prompt]:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected — call connect() first")
        try:
            result = await self._session.list_prompts()
        except Exception as e:
            logger.error(f"failed to discover prompts: {e}")
            raise
        if verbose:
            self._log_preview(result.prompts)
        return result.prompts

    async def discover_capabilities(self, verbose: bool = False) -> ServerCapabilities:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected — call connect() first")
        if self._capabilities is None:
            raise RuntimeError("server capabilities unavailable — connect() did not complete initialization")
        if verbose:
            logger.info(f"server capabilities:\n{self._capabilities.model_dump(exclude_none=True)}")
        return self._capabilities

    async def read_resource(self, uri: str, verbose: bool = False) -> Sequence[ResourceContents]:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected — call connect() first")
        try:
            result = await self._session.read_resource(AnyUrl(uri))
        except Exception as e:
            logger.error(f"failed to read resource {uri}: {e}")
            raise
        if verbose:
            for content in result.contents:
                logger.info(f"{uri}:\n{getattr(content, 'text', content)}")
        return result.contents

    async def call_tool(self, name: str, arguments: Optional[dict] = None, verbose: bool = False) -> CallToolResult:
        if self._session is None:
            raise RuntimeError("MCPClient is not connected — call connect() first")
        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as e:
            logger.error(f"failed to call tool {name}: {e}")
            raise
        if verbose:
            for content in result.content:
                logger.info(f"{name}:\n{getattr(content, 'text', content)}")
            if result.isError:
                logger.warning(f"tool {name} reported isError=True")
            if result.structuredContent:
                logger.info(f"{name} structuredContent:\n{result.structuredContent}")
            if result.meta:
                logger.info(f"{name} meta:\n{result.meta}")
        return result

    @staticmethod
    def transform_tools(tools: List[Tool]) -> List[DictConfig]:
        """Transform discover_tools() output into DictConfigs (name/description) for prompt injection."""
        return [
            OmegaConf.create({"name": get_display_name(t), "description": t.description or "No description available."})
            for t in tools
        ]

    async def get_tools(self) -> List[Any]:
        raise NotImplementedError("get_tools() must be implemented in a derived subclass")
