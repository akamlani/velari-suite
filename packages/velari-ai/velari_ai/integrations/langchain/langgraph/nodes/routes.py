from    typing import Any, Generic, Mapping, TypeVar
from    langgraph.types import Send, Command

# Bound to Mapping[str, Any] — every real graph state in this codebase is a TypedDict, which is
# structurally a Mapping, so route()'s state[...] access below type-checks without needing Any.
StateT = TypeVar("StateT", bound=Mapping[str, Any])


class Router(Generic[StateT]):
    """Reads a routing decision from state — composed into a node, not inherited from one.

    A conditional edge needs two things: a step that writes a decision into state (a regular
    node's `__call__`), and a `route_fn` that reads that decision back out. Written separately,
    the `route_fn` (often a standalone lambda) has to independently know which state key the node
    happens to write its decision to — two places agreeing on one string, with nothing enforcing
    it. `Router` closes that gap: construct one `Router(route_field)`, pass it into the node that
    writes the decision, and wire that *same instance*'s `route()` as the conditional edge's
    `route_fn` — one shared source of truth for the field name, not two.

    Not a `NodeBase` — it doesn't need a graph step's own machinery (`name`, `__call__`,
    `is_message_node`), only the one state key it reads.

    Args:
        route_field (str): State key this router's decision is read from.

    Examples:
        >>> class GradeContextNode(NodeBase[RetrievalState, None]):
        ...     def __init__(self, router: Router, **kwargs):
        ...         super().__init__(**kwargs)
        ...         self._router = router
        ...     def __call__(self, state, *, runtime):
        ...         verdict = "RELEVANT" if is_relevant(state["query"], state["candidates"]) else "IRRELEVANT"
        ...         return {self._router.route_field: verdict}
        >>> router = Router("verdict")
        >>> grade_node = GradeContextNode(router=router)
        >>> graph.add_step(name="grade", node=grade_node)
        >>> graph.add_conditional_edge(
        ...     source="grade",
        ...     route_fn=router.route,
        ...     route_map={"RELEVANT": "generate", "IRRELEVANT": "rewrite_query"},
        ... )
    """
    def __init__(self, route_field: str) -> None:
        self._route_field = route_field

    @property
    def route_field(self) -> str:
        return self._route_field

    def route(self, state: StateT) -> Any:
        """Read this router's decision — pass directly as `Graph.add_conditional_edge`'s `route_fn`.

        Raises:
            KeyError: If `route_field` isn't present in `state` — the node holding this router
                hasn't run yet, or ran but didn't write the field it declared.
        """
        return state[self._route_field]
