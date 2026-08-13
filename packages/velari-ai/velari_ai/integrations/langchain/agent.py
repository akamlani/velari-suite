import time
import uuid
from typing import Any, AsyncIterator, Iterator, List, Optional, Self, Tuple

from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    SummarizationMiddleware,
    before_model,
    after_model
)

from langgraph.types import StreamMode
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

# local files
from . import types
from .utils import build_chat_model
from ...ai.types import AgentConfig, ModelConfig


class Agent(object):
    """LangChain chat-model wrapper — build a create_agent() graph, then run() or stream() it.

    Args:
        model_config (Optional[ModelConfig]): Provider:model + kwargs; defaults to `ModelConfig()`.
        agent_config (Optional[AgentConfig]): Identity/tool-loop settings; `.name` forwarded to `build()`.
        **kwargs (Any): Extra provider kwargs (e.g. `api_key`); wins over `model_config.extra` on collision.

    Examples:
        >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"), api_key="sk-...")
    """
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        **kwargs: Any,
    ) -> None:
        self._model_config, self._model = build_chat_model(model_config, **kwargs)
        self._agent_config = agent_config or AgentConfig()
        self._name = self._agent_config.name
        self._agent = None

    def build(
        self,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        context_schema: Optional[type] = None,
        state_schema: Optional[type] = None,
        middleware: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Self:
        """Build a full tool-calling agent graph via LangChain's create_agent().

        Returns `self` (not the compiled graph), so calls chain: `Agent(...).build(...).run(...)`.

        Args:
            tools (List[BaseTool]): Tools the agent may call.
            system_prompt (Optional[str]): Instructions prepended as a `SystemMessage`.
            checkpointer (Optional[BaseCheckpointSaver]): Memory store; defaults to `MemorySaver()`.
            context_schema (Optional[type]): Type `run()` will pass via `context=`; must match.
            state_schema (Optional[type]): `TypedDict` extending `AgentState`; defaults to messages only.
            middleware (Optional[List[Any]]): Defaults to the retry/limit/summarization stack below; pass `[]` to disable.
            **kwargs (Any): Additional `create_agent()` params, forwarded through.

        Returns:
            Self: This `Agent`, for chaining.

        Raises:
            RuntimeError: If `create_agent()` fails to construct the graph.

        Examples:
            >>> from langchain.agents import AgentState
            >>> class BillingState(AgentState):
            ...     account_id: str
            >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
            >>> agent.build(
            ...     [lookup_account_balance],
            ...     system_prompt="You are a billing support assistant.",
            ...     context_schema=ContextSchema,
            ...     state_schema=BillingState,
            ... )
            >>> context = ContextSchema(experiment_name="billing-support-agent-chat-session", seed=42)
            >>> result = agent.run("What's the balance on ACC-10293?", thread_id="acc-10293-session", context=context)
            >>> result.text
            'Account ACC-10293 has an outstanding balance of $1,204.50.'
        """
        checkpointer = checkpointer or MemorySaver()
        if middleware is None:
            middleware = [
                ModelRetryMiddleware(),
                ToolRetryMiddleware(),
                ToolCallLimitMiddleware(run_limit=self._agent_config.max_tool_calls),
                SummarizationMiddleware(
                    model=self._model,          # can use a cheaper SLM for summarization
                    trigger=("tokens", 1024),   # trigger summarization when messages exceed 1024 tokens
                    keep=("messages", 3),       # keep the last 3 messages in full detail
                ),
            ]
        try:
            self._agent = create_agent(
                name=self._name,
                model=self._model,
                tools=tools,
                system_prompt=system_prompt,
                checkpointer=checkpointer,
                context_schema=context_schema,
                state_schema=state_schema,
                middleware=middleware,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"build() failed to construct the agent graph: {e}") from e
        return self

    def _prepare_call(self, method_name: str, thread_id: Optional[str]) -> Tuple[Any, str, bool, RunnableConfig]:
        # Returns self._agent (narrowed to non-Optional) since Optional narrowing doesn't cross
        # method boundaries. A missing thread_id gets a fresh uuid so history just doesn't persist.
        if self._agent is None:
            raise RuntimeError(f"{method_name}() requires build() to be called first — no agent graph is built")
        is_new_thread = thread_id is None
        thread_id = thread_id or str(uuid.uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        return self._agent, thread_id, is_new_thread, config

    @staticmethod
    def _to_response_info(
        all_messages: List[BaseMessage],
        prior_messages: List[BaseMessage],
        start: float,
    ) -> types.AgentResponseInfo:
        new_messages = all_messages[len(prior_messages):]
        metrics = types.ToolCallMetrics(
            latency_sec=round(time.perf_counter() - start, 3),
            message_stats=types.ToolCallMessageStats.from_messages(
                new_messages, cnt_total_messages=len(all_messages),
            ),
            usage_stats=types.UsageStats.from_messages(new_messages),
        )
        # The graph's final message is always the model's answer once the tool-call loop
        # ends — a real runtime invariant of create_agent()'s loop, not statically provable.
        return types.AgentResponseInfo(
            response=all_messages[-1], metrics=metrics, messages=all_messages,
        )

    def run(
        self, message: str, thread_id: Optional[str] = None, context: Optional[Any] = None,
    ) -> types.AgentResponseInfo:
        """Ask the built agent graph a question, abstracting away message/config structure.

        Wraps `self._agent.invoke()`, building the message/config structure for you.

        Args:
            message (str): The user's question or instruction.
            thread_id (Optional[str]): Checkpointer thread id; omitted → a fresh id is
                generated per call, so no history persists across calls.
            context (Optional[Any]): Forwarded to `.invoke()`'s `context=`; must match `context_schema`.

        Returns:
            AgentResponseInfo: `response` (final `AIMessage`), `messages` (full thread), and
                `metrics` — all scoped to this call except `cnt_total_messages`.

        Raises:
            RuntimeError: If `build()` hasn't been called yet.

        Examples:
            >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
            >>> agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
            >>> result = agent.run("What's the balance on ACC-10293?", thread_id="acc-10293-session")
            >>> result.text
            'Account ACC-10293 has an outstanding balance of $1,204.50.'
            >>> result.metrics.latency_sec
            0.842
        """
        agent, thread_id, is_new_thread, config = self._prepare_call("run", thread_id)
        prior_messages: List[BaseMessage] = (
            [] if is_new_thread else list(agent.get_state(config).values.get("messages", []))
        )
        start = time.perf_counter()
        result = agent.invoke(
            {"messages": [HumanMessage(message)]},
            config=config,
            context=context,
        )
        return self._to_response_info(result["messages"], prior_messages, start)

    async def arun(
        self, message: str, thread_id: Optional[str] = None, context: Optional[Any] = None,
    ) -> types.AgentResponseInfo:
        """Async counterpart to `run()` — same behavior, via `ainvoke()`/`aget_state()`.

        Args:
            message (str): The user's question or instruction.
            thread_id (Optional[str]): Checkpointer thread id; omitted → a fresh id is
                generated per call, so no history persists across calls.
            context (Optional[Any]): Forwarded to `.ainvoke()`'s `context=`; must match `context_schema`.

        Returns:
            AgentResponseInfo: `response` (final `AIMessage`), `messages` (full thread), and
                `metrics` — all scoped to this call except `cnt_total_messages`.

        Raises:
            RuntimeError: If `build()` hasn't been called yet.

        Examples:
            >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
            >>> agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
            >>> result = await agent.arun("What's the balance on ACC-10293?", thread_id="acc-10293-session")
            >>> result.text
            'Account ACC-10293 has an outstanding balance of $1,204.50.'
        """
        agent, thread_id, is_new_thread, config = self._prepare_call("arun", thread_id)
        prior_messages: List[BaseMessage] = (
            [] if is_new_thread else list((await agent.aget_state(config)).values.get("messages", []))
        )
        start = time.perf_counter()
        result = await agent.ainvoke(
            {"messages": [HumanMessage(message)]},
            config=config,
            context=context,
        )
        return self._to_response_info(result["messages"], prior_messages, start)

    def stream(
        self,
        message: str,
        thread_id: Optional[str] = None,
        context: Optional[Any] = None,
        stream_mode: StreamMode = "messages",
    ) -> Iterator[Any]:
        """Stream the built agent graph's response incrementally instead of buffering it.

        Args:
            message (str): The user's question or instruction.
            thread_id (Optional[str]): Checkpointer thread id; omitted → a fresh id is
                generated per call, so no history persists across calls.
            context (Optional[Any]): Forwarded to `.stream()`'s `context=`; must match `context_schema`.
            stream_mode (StreamMode): LangGraph stream mode; defaults to `"messages"`
                (token-by-token deltas). Use `"values"` for the full accumulated state after
                each step instead — take the last yielded item for the complete response.

        Returns:
            Iterator[Any]: Chunks per `stream_mode` — `(AIMessageChunk, metadata)` tuples for `"messages"`.

        Raises:
            RuntimeError: If `build()` hasn't been called yet.

        Examples:
            >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
            >>> agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
            >>> for chunk, metadata in agent.stream("What's the balance on ACC-10293?", thread_id="acc-10293-session"):
            ...     print(chunk.content, end="")

            >>> # full response at once, instead of piecing together token deltas:
            >>> chunks = list(agent.stream(
            ...     "What's the balance on ACC-10293?", thread_id="acc-10293-session", stream_mode="values",
            ... ))
            >>> chunks[-1]["messages"][-1].content
            'Account ACC-10293 has an outstanding balance of $1,204.50.'
        """
        agent, _, _, config = self._prepare_call("stream", thread_id)
        return agent.stream(
            {"messages": [HumanMessage(message)]},
            config=config,
            context=context,
            stream_mode=stream_mode,
        )

    async def astream(
        self,
        message: str,
        thread_id: Optional[str] = None,
        context: Optional[Any] = None,
        stream_mode: StreamMode = "messages",
    ) -> AsyncIterator[Any]:
        """Async counterpart to `stream()` — same behavior, via `self._agent.astream()`.

        Args:
            message (str): The user's question or instruction.
            thread_id (Optional[str]): Checkpointer thread id; omitted → a fresh id is
                generated per call, so no history persists across calls.
            context (Optional[Any]): Forwarded to `.astream()`'s `context=`; must match `context_schema`.
            stream_mode (StreamMode): LangGraph stream mode; defaults to `"messages"`
                (token-by-token deltas). Use `"values"` for the full accumulated state
                after each step instead.

        Returns:
            AsyncIterator[Any]: Chunks per `stream_mode` — `(AIMessageChunk, metadata)` tuples for `"messages"`.

        Raises:
            RuntimeError: If `build()` hasn't been called yet.

        Examples:
            >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
            >>> agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
            >>> async for chunk, metadata in await agent.astream(
            ...     "What's the balance on ACC-10293?", thread_id="acc-10293-session",
            ... ):
            ...     print(chunk.content, end="")
        """
        agent, _, _, config = self._prepare_call("astream", thread_id)
        return agent.astream(
            {"messages": [HumanMessage(message)]},
            config=config,
            context=context,
            stream_mode=stream_mode,
        )
