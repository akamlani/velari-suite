# uv run --extra agents python examples/ai/mcp/client_cli.py --help
# uv run --extra agents python examples/ai/mcp/client_cli.py
# uv run --extra agents python examples/ai/mcp/client_cli.py --client-type mcp
# uv run --extra agents python examples/ai/mcp/client_cli.py read-resource config://settings
# uv run --extra agents python examples/ai/mcp/client_cli.py call-tool get_time
# uv run --extra agents python examples/ai/mcp/client_cli.py call-tool <tool_name> --arguments '{"key": "value"}'

from __future__ import annotations

import  asyncio
import  json
import  sys
import  typer
import  logging
from    pathlib import Path
from    typing  import Optional

# package modules
from    velari_ai.integrations.fastmcp.client import BaseMCPClient
from    velari_ai.integrations.fastmcp.client_fastmcp import FastMCPClient
from    velari_ai.integrations.fastmcp.client_mcp import MCPClient
from    velari_ai.integrations.fastmcp.types import ClientType
from    velari_core.core import read_root_dir
from    velari_core.core.experiment import Experiment

logger = logging.getLogger(__name__)
app    = typer.Typer(rich_markup_mode="rich", add_completion=False)

SERVER_PATH = Path(__file__).parent / "server_cli.py"


def build_experiment(exp_name: str) -> tuple:
    exp = Experiment(root_path=read_root_dir())
    seed = exp.seed_init()
    exp_config = exp.create(experiment_name=exp_name, tags=exp_name.split("-")).experiment
    logger.info(f"experiment: {exp_config.name} (seed={seed}, dir={exp_config.install.dir})")
    return exp, exp_config, seed


def build_client(client_type: ClientType = ClientType.FASTMCP) -> BaseMCPClient:
    """Build an MCP client that launches server_cli.py over stdio.

    Args:
        client_type: FASTMCP for FastMCPClient (fastmcp 3.x's `Client`), MCP for
            MCPClient (the traditional `mcp` SDK's `ClientSession`).
    """
    if client_type == ClientType.MCP:
        return MCPClient(command=sys.executable, args=[str(SERVER_PATH)])
    return FastMCPClient(str(SERVER_PATH))


@app.command()
def discover(
    client_type: ClientType = typer.Option(ClientType.FASTMCP, help="MCP client implementation to use"),
) -> None:
    """Discover the tools and resources available on the MCP server."""
    async def _run() -> None:
        client = build_client(client_type)
        async with client:
            await client.discover_tools(verbose=True)
            await client.discover_resources(verbose=True)
            await client.discover_capabilities(verbose=True)
            await client.discover_prompts(verbose=True)

    asyncio.run(_run())


@app.command(name="read-resource")
def read_resource(
    uri: str = typer.Argument(..., help="Resource URI to read, e.g. config://settings"),
    client_type: ClientType = typer.Option(ClientType.FASTMCP, help="MCP client implementation to use"),
) -> None:
    """Read a resource's contents by URI from the MCP server."""
    async def _run() -> None:
        client = build_client(client_type)
        async with client:
            await client.read_resource(uri, verbose=True)

    asyncio.run(_run())


@app.command(name="call-tool")
def call_tool(
    name: str = typer.Argument(..., help="Tool name to call, e.g. get_time"),
    arguments: Optional[str] = typer.Option(None, help='Tool arguments as a JSON object, e.g. \'{"key": "value"}\''),
    client_type: ClientType = typer.Option(ClientType.FASTMCP, help="MCP client implementation to use"),
) -> None:
    """Call a tool by name on the MCP server."""
    async def _run() -> None:
        client = build_client(client_type)
        async with client:
            await client.call_tool(name, json.loads(arguments) if arguments else None, verbose=True)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
