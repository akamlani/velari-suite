from __future__ import annotations

import  sys
import  traceback
import  logging
from    contextlib   import asynccontextmanager
from    dataclasses  import dataclass, field
from    datetime     import datetime, timezone
from    typing       import Any, Callable, Dict, List, Literal, AsyncGenerator, Self
from    omegaconf    import DictConfig

from    fastmcp import FastMCP, Context
from    fastmcp.server.providers import Provider
from    fastmcp.server.transforms.search.bm25 import BM25SearchTransform
from    fastmcp.experimental.transforms.code_mode import CodeMode
# from mcp.server.fastmcp.prompts import base

# apps are providers - MCP Components
from    fastmcp import FastMCPApp
from    fastmcp.apps.generative import GenerativeUI
from    fastmcp.apps.file_upload import FileUpload

from    rich.console  import Console
from    rich.logging  import RichHandler
# package modules
from    .types import ResourceSpec, ToolDoc

logger = logging.getLogger(__name__)


@dataclass
class LifespanContext:
    """Container for information stored by LifespanProvider.lifespan()."""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class LifespanProvider(Provider):
    """Example provider demonstrating startup/shutdown hooks via Provider.lifespan()."""
    @asynccontextmanager
    # AsyncGenerator[YieldType, SendType] = [None, None]: yields nothing (bare `yield`), receives nothing (no `x = yield`)
    async def lifespan(self) -> AsyncGenerator[None, None]:
        logger.info("BaseLifespanProvider: startup")
        self.context = LifespanContext()
        try:
            yield
        finally:
            logger.info(f"BaseLifespanProvider: shutdown (context={self.context})")


def _redirect_logging_to_stderr() -> None:
    """Point stdout-bound root log handlers at stderr.

    MCP stdio transport reserves stdout exclusively for JSON-RPC messages;
    anything else on stdout corrupts the protocol stream.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichHandler):
            handler.console = Console(stderr=True)
        elif isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
            handler.stream = sys.stderr


class MCPServer(FastMCP):
    def __init__(self, config: DictConfig, name: str, instructions: str, **kwargs) -> None:
        _redirect_logging_to_stderr()
        # FastMCP Constructor
        super().__init__(name=name, instructions=instructions)
        self._config = config
        # expose tool search
        # self.add_transform(BM25SearchTransform())
        # self.add_transform(CodeMode())
        logger.info(f"MCP server '{name}' created")

    @classmethod
    def from_config(cls, cfg: DictConfig, **kwargs) -> Self:
        return cls(
            config       = cfg,
            name         = cfg.get("name", "MCP Server"),
            instructions = cfg.get("instructions", ""),
            **kwargs,
        )

    def bind(self) -> None:
        self.add_provider(FileUpload())
        self.add_provider(GenerativeUI())

    def register_tools(self, functions: List[Callable[..., Any]]) -> None:
        for fn in functions:
            tool_data: ToolDoc = ToolDoc.from_function(fn)
            kwargs: Dict[str, Any] = dict(description=tool_data.render_description())
            meta = tool_data.to_meta()
            if meta:
                kwargs["meta"] = meta
            self.tool(**kwargs)(fn)

    def register_resources(self, resources: List[ResourceSpec]) -> None:
        # resources: [protocol]://[host]/[path]
        # {file:///, docs://, workspace://, postgres://, sqlite:///, s3://, redis://, http://, https://}
        for resource in resources:
            self.resource(resource.uri, **resource.to_kwargs())(resource.fn)

    def run(
        self,
        transport: Literal["stdio", "http", "sse", "streamable-http"] | None = None,
        show_banner: bool | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        **transport_kwargs,
    ) -> None:
        """Start the MCP server. Called by the startup program."""
        transport = transport or "stdio"
        logger.info(f"Starting MCP server (transport={transport})")
        if transport == "stdio":
            super().run(transport=transport, show_banner=show_banner)
        else:
            super().run(transport=transport, show_banner=show_banner, host=host, port=port, **transport_kwargs)
