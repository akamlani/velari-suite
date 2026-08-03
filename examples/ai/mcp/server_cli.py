# uv run --extra agents python examples/ai/mcp/server_cli.py --help
# uv run --extra agents python examples/ai/mcp/server_cli.py
# uv run --extra agents python examples/ai/mcp/server_cli.py --transport http --port 8000
# Inspector: uv run mcp dev server.py

from __future__ import annotations

import  functools
import  os
import  typer
import  logging
from    typing       import Union, Literal, AsyncGenerator, Callable, Dict, Any
from    pathlib      import Path
from    omegaconf    import DictConfig, OmegaConf
from    dataclasses  import dataclass, field
from    contextlib   import asynccontextmanager

from    rich.console  import Console
from    rich.panel    import Panel

# package modules
from    velari_ai.integrations.fastmcp.server  import MCPServer, LifespanProvider, LifespanContext
from    velari_ai.integrations.fastmcp.types   import ResourceSpec
from    velari_core.core                       import read_root_dir

logger  = logging.getLogger(__name__)
console = Console(stderr=True, color_system="auto", force_terminal=True, width=120)
app     = typer.Typer(rich_markup_mode="rich", add_completion=False)


def read_config(config_path: Union[str, Path]) -> dict:
    """Read the MCP server's own YAML configuration and return it as a dict.

    Exposed as the "config://settings" resource — lets an MCP client inspect the
    same name/instructions/author configuration this server loaded at startup,
    without needing filesystem access of its own.
    """
    from velari_core.core.io.filesystem import Filesystem
    return Filesystem.read(config_path)

def get_time() -> str:
    """Return the current UTC time as a human-readable timestamp string.

    Lets an MCP client ask the server for its current wall-clock time without
    maintaining its own clock or trusting local time zone handling.

    Returns:
        str: The current UTC time, formatted as
            "The current UTC time is YYYY-MM-DD HH:MM:SS.".
    """
    import datetime
    current_time = datetime.datetime.now(datetime.timezone.utc)
    return f"The current UTC time is {current_time.strftime('%Y-%m-%d %H:%M:%S')}."


@dataclass
class ServerLifespanContext(LifespanContext):
    """LifespanContext extended with CLI-specific information."""
    pid: int = field(default_factory=os.getpid)


class ServerLifespanProvider(LifespanProvider):
    """Provider implementation overriding LifespanProvider's lifespan() for this CLI."""
    @asynccontextmanager
    # AsyncGenerator[YieldType, SendType] = [None, None]: yields nothing (bare `yield`), receives nothing (no `x = yield`)
    async def lifespan(self) -> AsyncGenerator[None, None]:
        logger.info("LifespanProvider: startup")
        self.context = ServerLifespanContext()
        try:
            yield
        finally:
            logger.info(f"LifespanProvider: shutdown (context={self.context})")


class MCPServerImpl(MCPServer):
    """MCP server exposing this tutorial's demo tools, resources, and providers."""
    def __init__(self, name: str, instructions: str, **kwargs) -> None:
        super().__init__(name=name, instructions=instructions, **kwargs)
        self._author = kwargs.get("author", "unknown")

    def register_providers(self) -> None:
        # provider registered after construction — its lifespan() fires at startup/shutdown
        self.add_provider(ServerLifespanProvider())

    def announce(self, transport: str) -> None:
        console.print(
            Panel(
                f"[bold cyan]{self.name}[/bold cyan]\n"
                f"[dim]{self.instructions}[/dim]\n"
                f"[green]transport[/green] = {transport}",
                expand=False,
            )
        )



@app.command()
def main(
    transport: Literal["stdio", "http", "sse", "streamable-http"] = typer.Option("stdio", help="MCP transport"),
    host:      str = typer.Option("127.0.0.1", help="Host to bind (http/sse/streamable-http transports only)"),
    port:      int = typer.Option(8000, help="Port to bind (http/sse/streamable-http transports only)"),
) -> None:
    server.announce(transport)
    server.run(transport=transport, host=host, port=port)
    server.register_providers()
    server.register_tools([get_time])
    handlers: Dict[str, Callable[..., Any]] = {"read_config": functools.partial(read_config, cfg_path)}
    server.register_resources([ResourceSpec.from_config(entry, handlers) for entry in cfg_data.get("resources", [])])


if __name__ == "__main__":
    cfg_path: Path       = Path(__file__).parent / "conf" / "mcp_config.yaml"
    cfg_data: DictConfig = OmegaConf.create(read_config(cfg_path))
    server = MCPServerImpl.from_config(cfg_data)
    app()
