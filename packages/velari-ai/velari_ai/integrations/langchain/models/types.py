from    __future__ import annotations
from    typing import List, Union, Optional, Self, Any
from    dataclasses import dataclass, field
from    pydantic import BaseModel, Field
# specific modules
from    langchain_core.messages import AIMessage, BaseMessage, ToolMessage, AnyMessage



@dataclass(frozen=True)
class UsageStats:
    """Token usage for one call, summed across every model turn in it."""
    input_tokens:     int
    output_tokens:    int
    reasoning_tokens: int

    @classmethod
    def from_messages(cls, messages: List[BaseMessage]) -> Self:
        """Sum token usage across every assistant turn in one call's new messages.

        Args:
            messages (List[BaseMessage]): Messages produced by a single call —
                excludes any prior thread history.

        Returns:
            Self: Input/output/reasoning token totals; zero when the model
                provider doesn't report `usage_metadata`.
        """
        usages = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
        return cls(
            input_tokens=sum(u["input_tokens"] for u in usages),
            output_tokens=sum(u["output_tokens"] for u in usages),
            reasoning_tokens=sum(u.get("output_token_details", {}).get("reasoning", 0) for u in usages),
        )


@dataclass(frozen=True)
class MessageStats:
    """Message counts produced by one call, common to every LLM/ToolCallingLLM/Agent call."""
    cnt_total_messages: int   # size of the full thread state after this call — includes
                              # prior history when `thread_id` carries memory forward
    cnt_turn_messages:  int   # total messages produced by this call
    cnt_assistant:      int   # number of assistant messages in the response

    @classmethod
    def from_messages(cls, messages: List[BaseMessage], cnt_total_messages: Optional[int] = None) -> Self:
        """Count assistant messages in one call's new messages.

        Args:
            messages (List[BaseMessage]): Messages produced by a single call — excludes
                any prior thread history.
            cnt_total_messages (Optional[int]): Size of the full graph state after this
                call, when it differs from `len(messages)` — e.g. prior thread history
                carried forward by `thread_id`. Defaults to `len(messages)`.

        Returns:
            Self: Assistant message count, plus the turn/thread totals.
        """
        cnt_total_messages = cnt_total_messages if cnt_total_messages is not None else len(messages)
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        return cls(
            cnt_total_messages=cnt_total_messages,
            cnt_turn_messages=len(messages),
            cnt_assistant=len(ai_messages),
        )


@dataclass(frozen=True)
class ToolCallMessageStats(MessageStats):
    """Message and tool-call counts produced by a call that runs a tool-calling loop."""
    cnt_tool_results:       int   # completed ToolMessage results
    cnt_tool_request_turns: int   # assistant turns that requested >=1 tool call
    cnt_tool_requests:      int   # total individual tool calls requested (summed across turns)

    @classmethod
    def from_messages(cls, messages: List[BaseMessage], cnt_total_messages: Optional[int] = None) -> Self:
        """Count assistant/tool messages and tool-call requests in one call's new messages.

        Args:
            messages (List[BaseMessage]): Messages produced by a single call — excludes
                any prior thread history.
            cnt_total_messages (Optional[int]): Size of the full graph state after this
                call, when it differs from `len(messages)` — e.g. prior thread history
                carried forward by `thread_id`. Defaults to `len(messages)`.

        Returns:
            Self: Assistant/tool-result/tool-request counts, plus the thread total.
        """
        cnt_total_messages = cnt_total_messages if cnt_total_messages is not None else len(messages)
        ai_messages        = [m for m in messages if isinstance(m, AIMessage)]
        tool_request_turns = [m for m in ai_messages if m.tool_calls]
        return cls(
            cnt_total_messages=cnt_total_messages,
            cnt_turn_messages=len(messages),
            cnt_assistant=len(ai_messages),
            cnt_tool_results=sum(1 for m in messages if isinstance(m, ToolMessage)),
            cnt_tool_request_turns=len(tool_request_turns),
            cnt_tool_requests=sum(len(m.tool_calls) for m in tool_request_turns),
        )


@dataclass(frozen=True)
class LLMMetrics:
    """Latency/usage/message metrics common to every LLM/ToolCallingLLM/Agent call."""
    latency_sec:   float
    usage_stats:   UsageStats
    message_stats: MessageStats

@dataclass(frozen=True)
class ToolCallMetrics(LLMMetrics):
    """Metrics for a call that runs a tool-calling loop (ToolCallingLLM, Agent)."""
    message_stats: ToolCallMessageStats



@dataclass(frozen=True)
class ResponseInfo:
    """Result of an LLM/ToolCallingLLM/Agent call — the response plus this call's metrics."""
    response: Union[AIMessage, BaseModel]
    messages: List[BaseMessage]
    metrics:  LLMMetrics

    @staticmethod
    def _extract_content(content: Any) -> str:
        return content.content if hasattr(content, "content") else str(content)

    @property
    def text(self) -> str:
        """The response's text content, when it's a plain (non-`response_model`) `AIMessage`.

        Raises:
            RuntimeError: If `response_model` was used — `.response` is a parsed object then,
                not a message; access its fields on `.response` directly instead.
        """
        if isinstance(self.response, AIMessage):
            return self._extract_content(self.response.content)
        raise RuntimeError(
            f"'{type(self.response).__name__}' response has no text content — it was parsed via "
            "response_model; access its fields on .response directly instead."
        )


@dataclass(frozen=True)
class LLMResponseInfo(ResponseInfo):
    """Result of LLM.run()/batch() — the response plus this call's latency/usage metrics."""

@dataclass(frozen=True)
class ToolCallResponseInfo(ResponseInfo):
    """ResponseInfo for a call that runs a tool-calling loop (ToolCallingLLM, Agent)."""
    metrics: ToolCallMetrics

@dataclass(frozen=True)
class ToolCallingLLMResponseInfo(ToolCallResponseInfo):
    """Result of ToolCallingLLM.query() — the response plus this call's tool-loop metrics."""

@dataclass(frozen=True)
class AgentResponseInfo(ToolCallResponseInfo):
    """Result of Agent.run() — the agent's response plus this call's metrics."""
