import  logging
import  os
from    pathlib import Path
from    typing import Any, List, Optional, Sequence, Tuple
from    pydantic import BaseModel, Field

# specific modules
from    langchain.chat_models import init_chat_model
from    langchain_core.language_models import BaseChatModel
from    langchain_core.messages import BaseMessage
from    langchain_core.runnables import Runnable, RunnableConfig
from    langchain_core.tools import BaseTool

from    mcp.types import Tool
from    langchain_mcp_adapters.sessions import Connection
from    langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool

# package modules
from    ...ai.types import ModelConfig
from    .types import ResponseInfo

logger = logging.getLogger(__name__)


class Configuration(BaseModel):
    """Configurable fields for runtime graph injection behavior."""
    # TBD: specific fields can be added here as needed, e.g.:

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {
            field: os.environ.get(field.upper(), configurable.get(field)) for field in cls.model_fields.keys()
        }
        return cls(**{k: v for k, v in values.items() if v is not None})


def build_chat_model(model_config: Optional[ModelConfig], **kwargs: Any) -> Tuple[ModelConfig, BaseChatModel]:
    """Resolve a ModelConfig (defaulting if omitted) and construct its chat model.

    Args:
        model_config (Optional[ModelConfig]): Provider:model + kwargs; defaults to `ModelConfig()`.
        **kwargs (Any): Extra provider kwargs (e.g. `api_key`); wins over `model_config.extra` on collision.

    Returns:
        Tuple[ModelConfig, BaseChatModel]: The resolved config and its constructed chat model —
            assign both, e.g. `self._model_config, self._model = build_chat_model(model_config, **kwargs)`.

    Examples:
        >>> model_config, model = build_chat_model(ModelConfig(model="openai:gpt-4o-mini"), api_key="sk-...")
    """
    model_config = model_config or ModelConfig()
    model_kwargs = {**model_config.extra, **kwargs}
    return model_config, init_chat_model(model_config.model, **model_kwargs)


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


def render_graph(cxt: Optional[Runnable], path: Optional[str] = None, use_ascii: bool = False) -> str:
    """Render a Mermaid or ASCII diagram of a built agent's execution graph.

    Works on anything exposing `.get_graph()` — typically an `Agent`'s `self._agent` (from `build()`).

    Args:
        cxt (Optional[Runnable]): The built agent graph to render.
        path (Optional[str]): If given, also writes a Mermaid PNG here.
        use_ascii (bool): Return ASCII art instead of Mermaid syntax.

    Returns:
        str: The graph as Mermaid syntax, or ASCII art if `use_ascii=True`.

    Raises:
        RuntimeError: If `cxt` is `None`.

    Examples:
        >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
        >>> agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
        >>> diagram = render_graph(agent._agent, path="docs/agent_graph.png")
        >>> print(diagram)
        graph TD;
            __start__ --> agent;
            agent --> lookup_account_balance;
            lookup_account_balance --> agent;
            agent --> __end__;
    """
    if cxt is None:
        raise RuntimeError("Agent is not built — call build() first")
    graph = cxt.get_graph()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        graph.draw_mermaid_png(output_file_path=path)
    if use_ascii:
        return graph.draw_ascii()
    return graph.draw_mermaid()


def log_response(response_info: ResponseInfo, history: bool = False) -> None:
    """Log a ResponseInfo's response (or full thread history) via `logger.info()` — never `print()`.

    Args:
        response_info (ResponseInfo): The result of `Agent.run()`/`arun()`.
        history (bool): Log every message in `response_info.messages` (full thread,
            including prior turns) instead of just the final response.

    Examples:
        >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
        >>> agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
        >>> result = agent.run("What's the balance on ACC-10293?", thread_id="acc-10293-session")
        >>> log_response(result)               # just this call's final response
        >>> log_response(result, history=True) # every message in the thread so far
    """
    for msg in (response_info.messages if history else [response_info.response]):
        if isinstance(msg, BaseMessage):
            logger.info(f"{type(msg).__name__}: {msg.content[:100]}")
            msg.pretty_print()
        else:
            logger.info(f"{type(msg).__name__}: {msg!r}")
