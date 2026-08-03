from __future__ import annotations

import  functools
import  inspect
from    dataclasses import dataclass, field, fields
from    typing      import Any, Callable, Dict, List, Optional, Self, Set, Tuple
from    omegaconf   import DictConfig
from    griffe      import Docstring, DocstringSectionKind, DocstringParameter, DocstringRaise, DocstringReturn


@dataclass
class ToolDoc:
    """Structured content parsed from a tool function's docstring."""
    @dataclass
    class Parameter:
        """A single documented parameter from a docstring's Args: section."""
        name:        str
        annotation:  Optional[str]
        description: str

    @dataclass
    class Returns:
        """The documented return value from a docstring's Returns: section."""
        annotation:  Optional[str]
        description: str

    @dataclass
    class Raise:
        """A single documented exception from a docstring's Raises: section."""
        exception:   str
        description: str

    summary:    str                       = field(default="")
    parameters: List[ToolDoc.Parameter]   = field(default_factory=list)
    returns:    Optional[ToolDoc.Returns] = field(default=None)
    raises:     List[ToolDoc.Raise]       = field(default_factory=list)
    examples:   List[str]                 = field(default_factory=list)

    @classmethod
    def _parse_parameters(cls, values: List[DocstringParameter]) -> List[ToolDoc.Parameter]:
        """Convert griffe's parsed Args: entries into ToolDoc.Parameter objects."""
        return [
            cls.Parameter(
                name=p.name,
                annotation=str(p.annotation) if p.annotation is not None else None,
                description=p.description,
            )
            for p in values
        ]

    @classmethod
    def _parse_returns(cls, values: List[DocstringReturn]) -> Optional[ToolDoc.Returns]:
        """Convert griffe's parsed Returns: entry, if any, into a ToolDoc.Returns object."""
        return next(
            (
                cls.Returns(
                    annotation=str(r.annotation) if r.annotation is not None else r.name,
                    description=r.description,
                )
                for r in values
            ),
            None,
        )

    @classmethod
    def _parse_raises(cls, values: List[DocstringRaise]) -> List[ToolDoc.Raise]:
        """Convert griffe's parsed Raises: entries into ToolDoc.Raise objects."""
        return [
            cls.Raise(exception=str(r.annotation) if r.annotation is not None else "", description=r.description)
            for r in values
        ]

    @staticmethod
    def _parse_examples(values: List[Tuple[DocstringSectionKind, str]]) -> List[str]:
        """Convert griffe's parsed Examples: blocks into their raw text."""
        return [text for _, text in values]

    @classmethod
    def from_function(cls, fn: Callable[..., Any]) -> Self:
        """Parse a tool function's docstring into structured documentation data.

        Splits the free-text summary from Args:/Returns:/Raises:/Examples: via griffe,
        since fastmcp's own parsing only reads Args:/the return annotation itself.

        Args:
            fn (Callable[..., Any]): The tool function whose docstring to parse.

        Returns:
            Self: The parsed content; `summary` falls back to the raw docstring if no
                section was found.

        Examples:
            >>> def get_time() -> str:
            ...     '''Return the current time.
            ...
            ...     Raises:
            ...         RuntimeError: if the system clock is unavailable.
            ...     '''
            >>> parsed = ToolDoc.from_function(get_time)
            >>> parsed.summary
            'Return the current time.'
            >>> parsed.raises[0].exception
            'RuntimeError'
        """
        doc = inspect.getdoc(fn) or ""
        kind = DocstringSectionKind
        parsed = (
            {s.kind: s.value for s in Docstring(doc, lineno=1, parser=parser).parse()}
            for parser in ("google", "numpy", "sphinx")
        )
        sections = next((s for s in parsed if s.get(kind.text) is not None), {})
        return cls(
            summary=sections.get(kind.text, doc),
            parameters=cls._parse_parameters(sections.get(kind.parameters, [])),
            returns=cls._parse_returns(sections.get(kind.returns, [])),
            raises=cls._parse_raises(sections.get(kind.raises, [])),
            examples=cls._parse_examples(sections.get(kind.examples, [])),
        )

    def render_description(self) -> str:
        """Render this structured tool documentation into a client-facing description.

        Returns:
            str: Summary plus any documented exceptions as a "Raises:" block.

        Examples:
            >>> parsed = ToolDoc(summary="Return the current time.", raises=[
            ...     ToolDoc.Raise(exception="RuntimeError", description="if the system clock is unavailable."),
            ... ])
            >>> parsed.render_description()
            'Return the current time.\\n\\nRaises:\\n    RuntimeError: if the system clock is unavailable.'
        """
        lines = "\n".join(f"    {r.exception}: {r.description}" for r in self.raises)
        return f"{self.summary}\n\nRaises:\n{lines}" if self.raises else self.summary

    def to_meta(self) -> Dict[str, Any]:
        """Render this structured tool documentation into MCP tool metadata.

        Returns:
            Dict[str, Any]: parameters/returns/raises/examples for MCP clients wanting
                structured data beyond the description text; keys omitted when unset.

        Examples:
            >>> parsed = ToolDoc(summary="Square a number.", parameters=[
            ...     ToolDoc.Parameter(name="x", annotation="int", description="The number to square."),
            ... ])
            >>> parsed.to_meta()
            {'parameters': [{'name': 'x', 'annotation': 'int', 'description': 'The number to square.'}]}
        """
        meta = {
            "parameters": [
                {"name": p.name, "annotation": p.annotation, "description": p.description}
                for p in self.parameters
            ],
            "returns": (
                {"annotation": self.returns.annotation, "description": self.returns.description}
                if self.returns else None
            ),
            "raises": [
                {"exception": r.exception, "description": r.description}
                for r in self.raises
            ],
            "examples": list(self.examples),
        }
        return {k: v for k, v in meta.items() if v}


@dataclass
class ResourceSpec:
    """Registration parameters for a single MCP resource."""
    uri:         str
    fn:          Callable[..., Any]
    name:        Optional[str]            = field(default=None)
    description: Optional[str]            = field(default=None)
    mime_type:   Optional[str]            = field(default=None)
    tags:        Optional[Set[str]]       = field(default=None)
    meta:        Optional[Dict[str, Any]] = field(default=None)

    def to_kwargs(self) -> Dict[str, Any]:
        """Render this spec's optional fields into `.resource()`-ready kwargs.

        Returns:
            Dict[str, Any]: Every set field except `uri`/`fn`, which the caller passes
                to `.resource()` separately.
        """
        exclude = {"uri", "fn"}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in exclude and getattr(self, f.name) is not None
        }

    @classmethod
    def from_config(cls, entry: DictConfig, handlers: Dict[str, Callable[..., Any]]) -> Self:
        """Build a ResourceSpec from one entry of a config's `resources:` list.

        Args:
            entry (DictConfig): One resource entry — `handler` (a key into `handlers`)
                and `uri` are required; `name`/`description`/`mime_type`/`tags`/`meta`
                are optional.
            handlers (Dict[str, Callable[..., Any]]): Handler name → callable registry.

        Returns:
            Self: Ready for `MCPServer.register_resources()`.
        """
        fn = handlers[entry["handler"]]
        description = entry.get("description")
        if description is None:
            # functools.partial has no __doc__ of its own — unwrap to the real
            # function so ToolDoc parses its docstring, not partial()'s.
            unwrapped   = fn.func if isinstance(fn, functools.partial) else fn
            description = ToolDoc.from_function(unwrapped).render_description()
        return cls(
            uri=entry["uri"],
            fn=fn,
            name=entry.get("name"),
            description=description,
            mime_type=entry.get("mime_type"),
            tags=set(entry["tags"]) if entry.get("tags") else None,
            meta={str(k): v for k, v in entry["meta"].items()} if entry.get("meta") else None,
        )
