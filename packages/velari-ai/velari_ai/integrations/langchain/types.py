from __future__ import annotations

from    dataclasses import dataclass, field
from    typing import Any, Dict, List, NotRequired, TypedDict, Optional, Self, Annotated, Sequence, Union
from    langchain.agents import AgentState
from    langchain_core.documents import Document
from    langchain_core.messages import AIMessage, BaseMessage, ToolMessage, AnyMessage
from    langgraph.graph.message import add_messages, MessagesState
from    pydantic import BaseModel




### Retrieval
class IngressState(TypedDict):
    "The state of a web-content ingestion pipeline feeding a retrieval vectorstore."
    uris: List[str]                         # URLs to fetch via Loader
    documents: NotRequired[List[Document]]  # loaded Documents, one per URL
    chunks: NotRequired[List[Document]]     # documents after text-splitting, ready to index
    num_documents: NotRequired[int]         # number of documents loaded from the URLs
    num_chunks: NotRequired[int]            # number of chunks after splitting the documents

class RoutingSource(TypedDict):
    """The state of a single router collection execution."""
    collection: str                         # the name of the table or collection to route query based on intent
    reasoning:  str                         # the reasoning behind the routing decision
    metadata:   Dict[str, Any]              # additional metadata about the routing process, e.g. source, timestamp, etc.

class RetrievalState(TypedDict):
    "The state of a single retrieval execution."
    query: str                              # user question
    context: List[str]                      # relative list of documents context retrieved for the query
    answer: str                             # generated response
    metadata: Dict[str, Any]                # additional metadata about the retrieval process, e.g. source, timestamp.
    collection: NotRequired[RoutingSource]  # routing: the name of the table or collection to route query based on intent








class ActorCritic(TypedDict):
    """A single actor-critic pair review, revisions, recommendations of an actor task execution."""
    actor:      str
    critic:     str
    sufficient: bool # whether the actor's output is sufficient to complete the task
    quality:    str  # acceptance ["accepted", "rejected", "review", "improve"]
                     # acceptance can go to next stage, else repeat actor task
    reason:     str  # feedback explanation of the critic's assessment of the actor's output
    iteration:  int  # number of actor-critic revision iterations performed for this task
                     # bail out at max iterations to avoid infinite loops


class TaskState(TypedDict):
    "The state of a single task execution."
    task_name:  str
    source:     str # input source (e.g. user query, tool output, etc.)
    draft:      str
    critique:   str
    revised:    str
    quality:    str # acceptance ["accepted", "rejected", "review", "improve"]

class ClassifierState(TypedDict):
    messages:   Annotated[Sequence[BaseMessage], add_messages]  # reducer for accumulation of messages
    intent:     NotRequired[str]                                # user intent classification or category (if any)
    route:      NotRequired[str]                                # the current flow the agent is executing (if any)
    result:     NotRequired[Any]                                # the result of the route execution (if any)






# class RuntimeAgentState(AgentState):
class RuntimeAgentState(TypedDict):
    "The runtime state of an agent lifecycle"
    # default from AgentState: messages (reducer), remaining_steps
    messages:    Annotated[Sequence[BaseMessage], add_messages] # reducer for accumulation of messages
    intent:      NotRequired[str]                               # user query intent classification (if any)
    active_flow: NotRequired[str]                               # the current flow the agent is executing (if any)
    iterations:  NotRequired[int]                               # per-turn LLM<->tool round-trip counter (guard)
    steps:       NotRequired[List[Dict[str, Any]]]
    result:      NotRequired[Any]
    metadata:    NotRequired[Dict[str, Any]]






@dataclass
class ContextSchema:
    """Runtime context for Agent.run() — passed through to LangGraph's `context=` at invoke time."""
    experiment_name: Optional[str]  = field(default=None)
    seed:            Optional[int]  = field(default=None)
    install_dir:     Optional[str]  = field(default=None)
    username:        Optional[str]  = field(default=None)
    agent_id:        Optional[str]  = field(default=None)
    extra:           Dict[str, Any] = field(default_factory=dict)


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
class Metrics:
    """Latency/usage/message metrics common to every LLM/ToolCallingLLM/Agent call."""
    latency_sec:   float
    usage_stats:   UsageStats
    message_stats: MessageStats


@dataclass(frozen=True)
class ToolCallMetrics(Metrics):
    """Metrics for a call that runs a tool-calling loop (ToolCallingLLM, Agent)."""
    message_stats: ToolCallMessageStats


@dataclass(frozen=True)
class ResponseInfo:
    """Result of an LLM/ToolCallingLLM/Agent call — the response plus this call's metrics."""
    response: Union[AIMessage, BaseModel]
    messages: List[BaseMessage]
    metrics:  Metrics

    @property
    def text(self) -> str:
        """The response's text content, when it's a plain (non-`response_model`) `AIMessage`.

        Raises:
            RuntimeError: If `response_model` was used — `.response` is a parsed object then,
                not a message; access its fields on `.response` directly instead.
        """
        if isinstance(self.response, AIMessage):
            return str(self.response.content)
        raise RuntimeError(
            f"'{type(self.response).__name__}' response has no text content — it was parsed via "
            "response_model; access its fields on .response directly instead."
        )


@dataclass(frozen=True)
class ToolCallResponseInfo(ResponseInfo):
    """ResponseInfo for a call that runs a tool-calling loop (ToolCallingLLM, Agent)."""
    metrics: ToolCallMetrics


@dataclass(frozen=True)
class AgentResponseInfo(ToolCallResponseInfo):
    """Result of Agent.run() — the agent's response plus this call's metrics."""


@dataclass(frozen=True)
class LLMResponseInfo(ResponseInfo):
    """Result of LLM.run()/batch() — the response plus this call's latency/usage metrics."""


@dataclass(frozen=True)
class ToolCallingLLMResponseInfo(ToolCallResponseInfo):
    """Result of ToolCallingLLM.query() — the response plus this call's tool-loop metrics."""
