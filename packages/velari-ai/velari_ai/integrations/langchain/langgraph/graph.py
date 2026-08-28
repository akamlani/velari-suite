from __future__ import annotations

from    dataclasses import dataclass
from    typing       import Any, List, Dict, Callable, Generic, Protocol, Type, Literal, Optional
from    pathlib      import Path
from    collections.abc import Hashable
import  sqlite3

from    langgraph.runtime import Runtime
from    langgraph.typing import StateT, ContextT
from    langgraph.graph import StateGraph
from    langgraph.graph import START, END
from    langgraph.graph.state import CompiledStateGraph
from    langgraph.checkpoint.memory import MemorySaver
from    langgraph.checkpoint.sqlite import SqliteSaver
from    langgraph.pregel import Pregel
# package modules
from    ..types import ContextSchema


class GraphNode(Protocol):
    def __call__(
        self,
        state: Any,
        *,
        runtime: Runtime[Any],
    ) -> dict: ...

@dataclass(frozen=True)
class EdgeSpec:
    source: str
    target: str

@dataclass(frozen=True)
class ConditionalEdgeSpec(Generic[StateT]):
    source:   str
    route_fn:  Callable[[StateT], str]
    route_map: Dict[Hashable, str]


class Graph(object):
    def __init__(self, **kwargs):
        self._checkpointer = self.checkpointer(
            ckpt_type=kwargs.get("checkpoint_type", "memory"),
            ckpt_path=kwargs.get("checkpoint_path", "")
        )
        self._steps: Dict[str, GraphNode] = {}
        self._edges: List[EdgeSpec] = []
        self._conditional_edges: List[ConditionalEdgeSpec] = []

    def build(self, name: str, state: Type[StateT], context: Type[ContextT] = ContextSchema) -> CompiledStateGraph[StateT, ContextT, StateT, StateT]:
        if not self._steps:
            raise ValueError("At least one step is required.")

        self._validate_graph()
        self._graph = self._build_graph(name, state, context)
        assert isinstance(self._graph, Pregel)
        return self._graph

    def _build_graph(
        self, name: str, state: Type[StateT], context: Type[ContextT],
    ) -> CompiledStateGraph[StateT, ContextT, StateT, StateT]:
        # state_schema:   main schema your nodes read and write
        # context_schema: per-run runtime context, e.g., user_id, database handles, execution-time dependencies.
        builder = StateGraph(state, context_schema=context)
        # compile graph steps
        for name, node in self._steps.items():
            builder.add_node(name, node)

        for edge in self._edges:
            builder.add_edge(edge.source, edge.target)

        for edge in self._conditional_edges:
            builder.add_conditional_edges(
                edge.source,
                edge.route_fn,
                edge.route_map,
            )
        self._graph =  builder.compile(name=name, checkpointer=self._checkpointer)
        return self._graph

    def add_step(
        self,
        *,
        name: str,
        node: GraphNode,
    ) -> Graph:
        if name in self._steps:
            raise ValueError(f"Duplicate step name: {name}")

        self._steps[name] = node
        return self

    def add_edge(
        self,
        *,
        source: str,
        target: str,
    ) -> Graph:
        self._edges.append(EdgeSpec(source=source, target=target))
        return self

    def add_conditional_edge(
        self,
        *,
        source:    str,
        route_fn:  Callable[[StateT], str],
        route_map: Dict[Hashable, str],
    ) -> Graph:
        self._conditional_edges.append(
            ConditionalEdgeSpec(
                source=source,
                # decision function: conditional router function for returned 'decision'
                # given current state, what should happen next?
                # e.g., decision = route_fn(state)
                route_fn=route_fn,
                # mapping of possible decision outputs to next steps
                # maps decision output -> actual node name (route node destination)
                # e.g., next_node = route_map[decision]
                route_map=route_map,
            )
        )
        return self

    def checkpointer(self,
        ckpt_type: Literal["memory", "sqlite"],
        ckpt_path: Optional[Path] = None
    ) -> None:
        if ckpt_type == "sqlite":
            saver = SqliteSaver(sqlite3.connect(str(ckpt_path), check_same_thread=False))
        else:
            saver = MemorySaver()

        self._checkpointer = saver

    def _validate_graph(self) -> None:
        valid_sources = set(self._steps.keys()) | {START}
        valid_targets = set(self._steps.keys()) | {END}

        for edge in self._edges:
            if edge.source not in valid_sources:
                raise ValueError(f"Unknown edge source: {edge.source}")
            if edge.target not in valid_targets:
                raise ValueError(f"Unknown edge target: {edge.target}")

        for edge in self._conditional_edges:
            if edge.source not in self._steps:
                raise ValueError(f"Conditional edge source must be a step name: {edge.source}")
            for route_target in edge.route_map.values():
                if route_target not in valid_targets:
                    raise ValueError(f"Unknown conditional edge target: {route_target}")
