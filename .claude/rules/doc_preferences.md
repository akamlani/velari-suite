# Documentation Preferences

Preferences for docstrings and other in-code documentation in this project. AI agents
and contributors should follow these.

## Docstrings

- Use **Google-style** docstrings throughout — configured via `.vscode/settings.json` (`autoDocstring.docstringFormat: google`).
- One-line summary on the opening line of the docstring; no multi-line summary blocks.
- Standard section order: summary → `Args:` → `Returns:` → `Raises:` (if applicable) → `Examples:` (always last).
- Use `Examples:` (plural) — the canonical Google spec name. Never `Example:` (singular).
- In `Args:`, include the type in parentheses: `param (Type): description.`
- In `Examples:`, prefix every line with `>>>` (doctest style), indented 4 spaces under the `Examples:` label.

```python
def cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance between two embedding vectors.

    Args:
        a (List[float]): Query embedding vector.
        b (List[float]): Document embedding vector.

    Returns:
        float: Cosine distance in [0, 2]; 0.0 = identical, 1.0 = orthogonal, 2.0 = opposite.

    Examples:
        >>> corpus_embeddings = [[0.2, 0.7, 0.4, 0.6], [0.5, 0.3, 0.8, 0.1]]
        >>> query_embedding   = [0.1, 0.8, 0.3, 0.5]
        >>> scores = [cosine_distance(query_embedding, doc) for doc in corpus_embeddings]
    """
```

## Docstring Concision

- Keep `Args:`/`Returns:` entries short, but don't compress away information a reader
  actually needs — what a parameter is looked up against, which fields are required vs
  optional, or why a fallback exists. Cut restated type names and generic filler
  ("registry mapping X to Y that implement Z"); keep the one or two facts that aren't
  obvious from the signature alone.

```python
# correct — short, but keeps the lookup relationship and required/optional split
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

# wrong — technically shorter, but generic enough to fit any method
def from_config(cls, entry: DictConfig, handlers: Dict[str, Callable[..., Any]]) -> Self:
    """Build a ResourceSpec from one entry of a config's `resources:` list.

    Args:
        entry (DictConfig): One resource entry.
        handlers (Dict[str, Callable[..., Any]]): Handler name to implementing callable.

    Returns:
        Self: The resource spec, ready to register.
    """
```

## Docstring Spacing

- No blank line between a class or function's docstring and the first line of code that
  follows it (a field, nested class, decorator, or statement) — regardless of whether the
  docstring is a single line or a full multi-paragraph Google-style block with
  Args:/Returns:/Examples: sections. The docstring should sit directly against the code
  it documents.

```python
# correct — one-line docstring directly against the first field
@dataclass
class Config:
    """Runtime configuration for the service."""
    host: str
    port: int = field(default=8080)

# correct — same rule for a full multi-paragraph docstring
class Filesystem(object):
    """Filesystem operations: read, write, move, delete, ...

    Examples:
        >>> Filesystem.read("notes.txt")
    """
    @staticmethod
    def get_mime_type(file_path: Union[str, Path]) -> str: ...

# wrong — stray blank line between docstring and the code it documents
@dataclass
class Config:
    """Runtime configuration for the service."""

    host: str
    port: int = field(default=8080)
```

## Docstring Deviation for Tests

- One-line module docstring only (`"""Tests for <module>."""`). Test functions/methods
  do not get docstrings — the test name should describe the behavior on its own. This
  overrides the general `## Docstrings` rule above for test code specifically; see
  `.claude/rules/test_preferences.md` for the rest of the test-writing conventions.
