from __future__ import annotations

from typing import Any, Callable, List, Optional, Self

from agno.agent import Agent as AgnoAgent
from agno.run.agent import RunOutput
from agno.models.openai import OpenAIChat
from agno.tools import Function

from ...ai.types import AgentConfig, ModelConfig, ProviderName


def make_tool(fn: Callable[..., Any], name: Optional[str] = None) -> Function:
    """Wrap a plain Python callable as an agno `Function` tool for `build()`'s `tools=`.

    Args:
        fn (Callable[..., Any]): The function to expose as a tool; its name and
            docstring become the tool's name/description unless overridden.
        name (Optional[str]): Override the tool's name; defaults to `fn.__name__`.

    Returns:
        Function: agno's tool wrapper, ready for `Agent.build(tools=[...])`.

    Examples:
        >>> def lookup_account_balance(account_id: str) -> str:
        ...     "Look up the outstanding balance for a billing account."
        ...     return f"Account {account_id} balance: $1,204.50"
        >>> balance_tool = make_tool(lookup_account_balance)
        >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
        >>> agent.build(tools=[balance_tool], instructions="You are a billing support assistant.")
    """
    return Function.from_callable(fn, name=name)


class Agent(object):
    """Minimal agno-based agent wrapper — build a tool-using agent, then run it.

    Args:
        model_config (Optional[ModelConfig]): Provider:model string plus provider
            kwargs; defaults to `ModelConfig()`.
        agent_config (Optional[AgentConfig]): Identity settings; `.name` is forwarded
            to the underlying `agno.agent.Agent`.

    Examples:
        >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
        >>> agent.build(instructions="You are a billing support assistant.")
        >>> response = agent.run("What's the balance on ACC-10293?")
        >>> response.content
        'Account ACC-10293 has an outstanding balance of $1,204.50.'
    """
    def __init__(self, model_config: Optional[ModelConfig] = None, agent_config: Optional[AgentConfig] = None) -> None:
        self._model_config = model_config or ModelConfig()
        self._agent_config = agent_config or AgentConfig()
        self._agent: Optional[AgnoAgent] = None

    def _resolve_model(self) -> Any:
        provider, _, model_id = self._model_config.model.partition(":")
        if provider == ProviderName.OPENAI:
            return OpenAIChat(id=model_id, **self._model_config.extra)
        raise RuntimeError(f"_resolve_model() does not support provider {provider!r}")

    def build(self, tools: Optional[List[Any]] = None, instructions: Optional[str] = None) -> Self:
        """Build the underlying agno agent.

        Args:
            tools (Optional[List[Any]]): Tools the agent may call.
            instructions (Optional[str]): System-level instructions for the agent.

        Returns:
            Self: This `Agent`, for chaining.

        Examples:
            >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
            >>> agent.build(
            ...     tools=[make_tool(lookup_account_balance)],
            ...     instructions="You are a billing support assistant.",
            ... )
        """
        self._agent = AgnoAgent(
            name=self._agent_config.name,
            model=self._resolve_model(),
            tools=tools or [],
            instructions=instructions,
            tool_call_limit=self._agent_config.max_tool_calls,
        )
        return self

    def run(self, message: str) -> RunOutput:
        """Ask the built agent a question.

        Args:
            message (str): The user's question or instruction.

        Returns:
            RunOutput: agno's own response object (`.content` holds the text reply).

        Raises:
            RuntimeError: If `build()` hasn't been called yet.

        Examples:
            >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
            >>> agent.build(instructions="You are a billing support assistant.")
            >>> response = agent.run("What's the balance on ACC-10293?")
            >>> response.content
            'Account ACC-10293 has an outstanding balance of $1,204.50.'
        """
        if self._agent is None:
            raise RuntimeError("run() requires build() to be called first — no agent is built")
        return self._agent.run(message)
