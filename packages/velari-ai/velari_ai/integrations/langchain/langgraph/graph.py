from __future__ import annotations

import  sqlite3
from    dataclasses import dataclass
from    typing       import Any, List, Dict, Callable, Generic, Protocol, Type, Optional, TypeVar, Union
from    pathlib      import Path
from    collections.abc import Hashable

from    langgraph.typing import StateT, ContextT, InputT, OutputT
from    langgraph.graph import StateGraph
from    langgraph.graph import START, END
from    langgraph.graph.state import CompiledStateGraph
from    langgraph.pregel import Pregel
from    langgraph.runtime import Runtime
# langgraph memory
from    langgraph.checkpoint.base import BaseCheckpointSaver
from    langgraph.checkpoint.memory import MemorySaver
from    langgraph.checkpoint.sqlite import SqliteSaver
from    langgraph.store.base import BaseStore
from    langgraph.store.memory import InMemoryStore
from    langgraph.store.sqlite import SqliteStore

# package modules
from    ..types import ContextSchema
from    ....ai.types import PersistenceBackend


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
            ckpt_type=PersistenceBackend(kwargs.get("checkpoint_type", PersistenceBackend.MEMORY)),
            ckpt_path=kwargs.get("checkpoint_path", "")
        )
        self._store = self.memory_store(
            store_type=PersistenceBackend(kwargs.get("store_type", PersistenceBackend.MEMORY)),
            store_path=kwargs.get("store_path", "")
        )
        self._steps: Dict[str, Union[GraphNode, CompiledStateGraph[Any, Any]]] = {}
        self._edges: List[EdgeSpec] = []
        self._conditional_edges: List[ConditionalEdgeSpec] = []

    def build(
        self,
        name: str,
        state:   Type[StateT],
        context: Type[ContextT] = ContextSchema,
        input:   Optional[Type[InputT]]  = None,
        output:  Optional[Type[OutputT]] = None,
    ) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
        if not self._steps:
            raise ValueError("At least one step is required.")

        self._validate_graph()
        self._graph = self._build_graph(name, state, context, input, output)
        assert isinstance(self._graph, Pregel)
        return self._graph

    def _build_graph(
        self,
        name: str,
        state:   Type[StateT],
        context: Type[ContextT],
        input:   Optional[Type[InputT]]  = None,
        output:  Optional[Type[OutputT]] = None,
    ) -> CompiledStateGraph[StateT, ContextT, InputT, OutputT]:
        # state_schema:   main schema your nodes read and write
        # context_schema: per-run runtime context, e.g., user_id, database handles, execution-time dependencies.
        # input_schema/output_schema: default to state_schema when omitted (LangGraph's own InputT/OutputT
        # TypeVar defaults) — only meaningfully differ from state when a caller opts into a narrower one.
        builder = StateGraph(state, context_schema=context, input_schema=input, output_schema=output)
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
        self._graph =  builder.compile(name=name, checkpointer=self._checkpointer, store=self._store)
        return self._graph

    def add_step(
        self,
        *,
        name: str,
        node: Union[GraphNode, CompiledStateGraph[StateT, ContextT]],
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
        ckpt_type: PersistenceBackend,
        ckpt_path: Optional[Path] = None
    ) -> BaseCheckpointSaver:
        if ckpt_type == PersistenceBackend.SQLITE:
            saver = SqliteSaver(sqlite3.connect(str(ckpt_path), check_same_thread=False))
        else:
            saver = MemorySaver()

        self._checkpointer = saver
        return saver

    def memory_store(self,
        store_type: PersistenceBackend,
        store_path: Optional[Path] = None
    ) -> BaseStore:
        if store_type == PersistenceBackend.SQLITE:
            # SqliteStore manages its own BEGIN/COMMIT per operation — isolation_level=None
            # (autocommit) stops sqlite3's own implicit transaction handling from conflicting with it.
            conn = sqlite3.connect(str(store_path), check_same_thread=False, isolation_level=None)
            store_backend = SqliteStore(conn)
            store_backend.setup()
        else:
            store_backend = InMemoryStore()

        self._store = store_backend
        return store_backend

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
