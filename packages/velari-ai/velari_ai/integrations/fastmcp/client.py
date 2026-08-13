from __future__ import annotations

import  logging
from    abc     import ABC, abstractmethod
from    typing  import Any, Dict, List, Optional, Self, Sequence

import pandas as pd
from    mcp.shared.metadata_utils import get_display_name
from    mcp.types import Tool, Resource, Prompt, ResourceContents, ServerCapabilities

logger = logging.getLogger(__name__)


class BaseMCPClient(ABC):
    """Shared async lifecycle/discovery contract for MCP client implementations."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection to the MCP server."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the connection to the MCP server."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return whether the client currently holds a live connection."""

    @abstractmethod
    async def discover_tools(self, verbose: bool = False) -> Sequence[Tool]:
        """List the tools exposed by the connected server."""

    @abstractmethod
    async def discover_resources(self, verbose: bool = False) -> Sequence[Resource]:
        """List the resources exposed by the connected server."""

    @abstractmethod
    async def discover_prompts(self, verbose: bool = False) -> Sequence[Prompt]:
        """List the prompts exposed by the connected server."""

    @abstractmethod
    async def discover_capabilities(self, verbose: bool = False) -> ServerCapabilities:
        """Return the capabilities the server advertised during initialization."""

    @abstractmethod
    async def read_resource(self, uri: str, verbose: bool = False) -> Sequence[ResourceContents]:
        """Read a resource's contents by URI."""

    @abstractmethod
    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Any:
        """Call a tool by name.

        Returns:
            Any: The concrete result type differs per implementation (`mcp.types.CallToolResult`
                vs. `fastmcp.client.client.CallToolResult` — different field names, not
                interchangeable) — see each subclass's docstring for its actual return type.
        """

    @staticmethod
    def _log_preview(items: Sequence[Any], columns: Optional[List[str]] = None) -> None:
        """Log a DataFrame preview of discovered MCP items (tools/resources/prompts)."""
        if not items:
            return
        df = pd.DataFrame([item.model_dump(exclude_none=True) for item in items])
        df["display_name"] = [get_display_name(item) for item in items]
        view = df[columns] if columns else df
        logger.info(f"\n{view.to_string()}")

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
