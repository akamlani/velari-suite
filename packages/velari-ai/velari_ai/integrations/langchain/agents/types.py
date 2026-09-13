from __future__ import annotations

from    dataclasses import dataclass, field
from    typing import Any, Dict, List, NotRequired, TypedDict, Optional, Annotated, Sequence

from    langchain_core.messages import BaseMessage
from    langgraph.graph.message import add_messages


class ClassifierState(TypedDict):
    messages:   Annotated[Sequence[BaseMessage], add_messages]  # reducer for accumulation of messages
    intent:     NotRequired[str]                                # user intent classification or category (if any)
    route:      NotRequired[str]                                # the current flow the agent is executing (if any)
    result:     NotRequired[Any]                                # the result of the route execution (if any)


class RuntimeAgentState(TypedDict):
    "The runtime state of an agent lifecycle"
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
