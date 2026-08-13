import time
from typing import Any, Dict, Iterator, List, Optional, Type, Union, cast
from pydantic import BaseModel

from langchain_core.language_models import LanguageModelInput
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

# local files
from . import types
from .utils import build_chat_model
from ...ai.types import AgentConfig, ModelConfig


class LLM(object):
    """Bare-bones chat-model wrapper — invoke with a system prompt and a single user message.

    No tools, no memory, no middleware — for the common case of a single system+human
    call. See `ToolCallingLLM` for manual tool-calling, or `Agent` for a full
    create_agent() graph (build/run).

    Args:
        model_config (Optional[ModelConfig]): Provider:model + kwargs; defaults to `ModelConfig()`.
        **kwargs (Any): Extra provider kwargs (e.g. `api_key`); wins over `model_config.extra` on collision.

    Examples:
        >>> llm = LLM(model_config=ModelConfig(model="openai:gpt-4o-mini"), api_key="sk-...")
    """
    def __init__(self, model_config: Optional[ModelConfig] = None, **kwargs: Any) -> None:
        self._model_config, self._model = build_chat_model(model_config, **kwargs)

    def _build_messages(self, system_prompt: str, message: str) -> List[BaseMessage]:
        return [SystemMessage(system_prompt), HumanMessage(message)]

    def run(
        self,
        system_prompt: str,
        message: str,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> types.LLMResponseInfo:
        """Invoke the model with a system prompt and a single human message.

        Args:
            system_prompt (str): Instructions sent as a `SystemMessage`.
            message (str): The user's question or instruction, sent as a `HumanMessage`.
            response_model (Optional[Type[BaseModel]]): If given, the response is parsed into
                this Pydantic model via `with_structured_output()` instead of `AIMessage`. In
                that case `messages` holds only the request turn — the parsed object isn't a
                `BaseMessage`, so it can't be appended.

        Returns:
            types.LLMResponseInfo: `response` (a parsed `response_model` instance if one was given,
                otherwise the raw `AIMessage`), `messages`, and `metrics` (incl. `message_stats`
                turn/assistant counts — no tool fields, since `LLM` never calls tools).

        Examples:
            >>> llm = LLM(model_config=ModelConfig(model="openai:gpt-4o-mini"))
            >>> result = llm.run(
            ...     "You are a billing support assistant.",
            ...     "What's the outstanding balance on account ACC-10293?",
            ... )
            >>> result.text
            'Account ACC-10293 has an outstanding balance of $1,204.50.'
            >>> result.metrics.latency_sec
            0.412
        """
        model = self._model.with_structured_output(response_model) if response_model else self._model
        messages = self._build_messages(system_prompt, message)
        start = time.perf_counter()
        response = cast(Union[AIMessage, BaseModel], model.invoke(messages))
        final_messages = messages + [response] if isinstance(response, AIMessage) else messages
        return types.LLMResponseInfo(
            response=response,
            messages=final_messages,
            metrics=types.Metrics(
                latency_sec=round(time.perf_counter() - start, 3),
                usage_stats=types.UsageStats.from_messages([response] if isinstance(response, AIMessage) else []),
                message_stats=types.MessageStats.from_messages(final_messages),
            ),
        )

    def batch(
        self,
        system_prompt: str,
        messages: List[str],
        response_model: Optional[Type[BaseModel]] = None,
    ) -> List[types.LLMResponseInfo]:
        """Invoke the model over many user messages sharing one system prompt, in parallel.

        Args:
            system_prompt (str): Instructions sent as a `SystemMessage`, shared by every request.
            messages (List[str]): User messages; each becomes its own independent call.
            response_model (Optional[Type[BaseModel]]): If given, each response is parsed into
                this Pydantic model via `with_structured_output()` instead of `AIMessage`.

        Returns:
            List[types.LLMResponseInfo]: One per input message, in the same order. Since the whole
                batch runs as a single parallel call, every item's `metrics.latency_sec` is the
                same whole-batch duration — only `response`/`usage_stats` are genuinely per-item.

        Examples:
            >>> llm = LLM(model_config=ModelConfig(model="openai:gpt-4o-mini"))
            >>> results = llm.batch(
            ...     "Classify the sentiment of this support ticket as positive, neutral, or negative.",
            ...     ["The product arrived damaged and support has ignored me for a week.", "Great experience, thanks!"],
            ... )
            >>> [r.text for r in results]
            ['negative', 'positive']
        """
        model = self._model.with_structured_output(response_model) if response_model else self._model
        per_item_messages = [self._build_messages(system_prompt, message) for message in messages]
        inputs: List[LanguageModelInput] = list(per_item_messages)
        start = time.perf_counter()
        responses = cast(Union[List[AIMessage], List[BaseModel]], model.batch(inputs))
        latency_sec = round(time.perf_counter() - start, 3)
        results = []
        for item_messages, response in zip(per_item_messages, responses):
            final_messages = item_messages + [response] if isinstance(response, AIMessage) else item_messages
            results.append(types.LLMResponseInfo(
                response=response,
                messages=final_messages,
                metrics=types.Metrics(
                    latency_sec=latency_sec,
                    usage_stats=types.UsageStats.from_messages([response] if isinstance(response, AIMessage) else []),
                    message_stats=types.MessageStats.from_messages(final_messages),
                ),
            ))
        return results

    def stream(
        self,
        system_prompt: str,
        message: str,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> Iterator[Any]:
        """Stream the model's response token-by-token for a single system+human call.

        Args:
            system_prompt (str): Instructions sent as a `SystemMessage`.
            message (str): The user's question or instruction, sent as a `HumanMessage`.
            response_model (Optional[Type[BaseModel]]): If given, streamed via
                `with_structured_output()` instead of the raw chat model — most providers
                yield the fully parsed object as a single chunk rather than incremental deltas.

        Returns:
            Iterator[Any]: Message chunks as the model generates them.

        Examples:
            >>> llm = LLM(model_config=ModelConfig(model="openai:gpt-4o-mini"))
            >>> for chunk in llm.stream(
            ...     "You are a research assistant.",
            ...     "What were Q3's key findings in the churn-analysis report?",
            ... ):
            ...     print(chunk.content, end="")
        """
        model = self._model.with_structured_output(response_model) if response_model else self._model
        return model.stream(self._build_messages(system_prompt, message))


class ToolCallingLLM(object):
    """Chat-model wrapper with manual tool-binding and an automatic tool-call loop.

    Bind tools with `bind()`, then either drive the loop yourself off the returned
    Runnable, or let `query()` run it automatically. See `Agent` for a full
    create_agent()-backed graph with memory/middleware instead.

    Args:
        model_config (Optional[ModelConfig]): Provider:model + kwargs; defaults to `ModelConfig()`.
        agent_config (Optional[AgentConfig]): Tool-loop settings; `.max_tool_calls` bounds `query()`.
        **kwargs (Any): Extra provider kwargs (e.g. `api_key`); wins over `model_config.extra` on collision.

    Examples:
        >>> tool_llm = ToolCallingLLM()
        >>> tool_llm.bind([lookup_account_balance])
        >>> result = tool_llm.query("What's the outstanding balance on account ACC-10293?")
        >>> result.text
        'Account ACC-10293 has an outstanding balance of $1,204.50.'
    """
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        **kwargs: Any,
    ) -> None:
        self._model_config, self._model = build_chat_model(model_config, **kwargs)
        self._agent_config = agent_config or AgentConfig()
        self._bound_llm = None
        # set by bind() — name -> BaseTool lookup so query() can execute requested tool_calls.
        self._tools_by_name: Dict[str, BaseTool] = {}

    def bind(self, tools: List[BaseTool]) -> Runnable:
        """Bind tools to the chat model for a manual, standalone tool-call loop.

        Drive the loop yourself: invoke the returned Runnable, inspect
        `response.tool_calls`, execute, and repeat. See `query()` for the automatic version.

        Args:
            tools (List[BaseTool]): Tools the model may choose to call.

        Returns:
            Runnable: The tool-bound chat model, also stored on `self._bound_llm`.

        Examples:
            >>> tool_llm = ToolCallingLLM()
            >>> bound_llm = tool_llm.bind([lookup_account_balance])
            >>> response = bound_llm.invoke("What's the outstanding balance on account ACC-10293?")
            >>> response.tool_calls
            [{'name': 'lookup_account_balance', 'args': {'account_id': 'ACC-10293'}, 'id': '...'}]
        """
        self._tools_by_name = {t.name: t for t in tools}
        self._bound_llm = self._model.bind_tools(tools)
        return self._bound_llm

    def query(
        self, message: str, response_model: Optional[Type[BaseModel]] = None,
    ) -> types.ToolCallingLLMResponseInfo:
        """Ask the bound model a question, automatically executing any requested tool calls.

        Automatic version of `bind()`'s loop — stops once the model returns no more
        tool calls, or `max_tool_calls` is exceeded.

        Args:
            message (str): The user's question or instruction.
            response_model (Optional[Type[BaseModel]]): If given, the final answer (once no
                more tool calls are requested) is parsed into this Pydantic model via
                `with_structured_output()` instead of returned as `AIMessage`. Tool-calling
                turns themselves are unaffected — only the final turn is structured. In that
                case `messages` holds the loop trace up to the last tool result, since the
                parsed object isn't a `BaseMessage`.

        Returns:
            types.ToolCallingLLMResponseInfo: `response` (a parsed `response_model` instance if one
                was given, otherwise the raw final `AIMessage`), `messages` (the full loop
                trace), and `metrics` (incl. `message_stats` tool-call counts).

        Raises:
            RuntimeError: If `bind()` hasn't been called, or `max_tool_calls` is exceeded.

        Examples:
            >>> tool_llm = ToolCallingLLM()
            >>> tool_llm.bind([lookup_account_balance])
            >>> result = tool_llm.query("What's the outstanding balance on account ACC-10293?")
            >>> result.text
            'Account ACC-10293 has an outstanding balance of $1,204.50.'
            >>> result.metrics.message_stats.cnt_tool_requests
            1
        """
        if self._bound_llm is None:
            raise RuntimeError("query() requires bind() to be called first — no tools are bound")

        start = time.perf_counter()
        messages = self._run_tool_loop(self._bound_llm, message)
        # cast is only needed for the structured branch — with_structured_output()'s stub always
        # includes dict[str, Any]; messages[-1] needs none, since BaseMessage is already a BaseModel.
        response = (
            cast(Union[AIMessage, BaseModel], self._model.with_structured_output(response_model).invoke(messages))
            if response_model is not None
            else messages[-1]
        )
        return types.ToolCallingLLMResponseInfo(
            response=response,
            messages=messages,
            metrics=types.ToolCallMetrics(
                latency_sec=round(time.perf_counter() - start, 3),
                message_stats=types.ToolCallMessageStats.from_messages(messages),
                usage_stats=types.UsageStats.from_messages(messages),
            ),
        )

    def _run_tool_loop(self, bound_llm: Runnable, message: str) -> List[BaseMessage]:
        messages: List[BaseMessage] = [HumanMessage(message)]
        for _ in range(self._agent_config.max_tool_calls):
            response = bound_llm.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                return messages
            for call in response.tool_calls:
                tool = self._tools_by_name[call["name"]]
                messages.append(tool.invoke(call))

        max_tool_calls = self._agent_config.max_tool_calls
        raise RuntimeError(f"query() exceeded max_tool_calls={max_tool_calls} without a final answer")
