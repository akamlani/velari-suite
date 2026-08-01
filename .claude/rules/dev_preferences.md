# Coding Preferences

Preferences established for this project. AI agents and contributors should follow these.

## Logging

- Always use f-strings in logger calls — never `%s` lazy interpolation.

```python
logger.info(f"Loaded {n} records")   # correct
logger.info("Loaded %s records", n)  # wrong
```

## Type Annotations

- Use `List`, `Callable`, `Literal`, `TypeVar`, etc. from `typing` — not built-in lowercase forms (`list`, `tuple`, etc.) — in all function signatures, return types, and dataclass fields.

```python
from typing import List

def sample(self, n: int = 5) -> List[DatasetExample]: ...   # correct
def sample(self, n: int = 5) -> list[DatasetExample]: ...  # wrong
```

- Do **not** use string-quoted annotations (e.g. `"Foo | List[Foo]"`) unless a true forward reference requires it. If a forward reference is unavoidable, add `from __future__ import annotations` at the top of the file instead of quoting individual annotations.

```python
# correct — type defined above in the same file
def __getitem__(self, index: int) -> DatasetExample: ...

# wrong — unnecessary quotes
def __getitem__(self, index: int) -> "DatasetExample": ...
```

- Annotate every parameter. When a function genuinely accepts multiple unrelated types (e.g. `str`, `Path`, `list`, `DataFrame`), use `Any` from `typing` rather than leaving the parameter bare.

```python
from typing import Any

def load(self, path: Any, fmt: Optional[str] = None) -> List[Document]: ...  # correct
def load(self, path, fmt: Optional[str] = None) -> List[Document]: ...       # wrong — path untyped
```

- Declare a return type on every function and method, including private helpers (`_name`).

```python
def _infer_fmt(self, path: Any) -> LoaderFormat: ...                                  # correct
def _normalize(self, path: Any, fmt: LoaderFormat) -> Tuple[Any, LoaderFormat]: ...   # correct
def _infer_fmt(self, path: Any): ...                                                   # wrong — missing return type
```

- For `dict` and `tuple` return types, use the parameterised `typing` forms (`Dict[str, Any]`, `Tuple[X, Y]`), not bare `dict` or `tuple`. This is consistent with the existing `List` rule.

```python
def to_dict(self) -> Dict[str, Any]: ...   # correct
def to_dict(self) -> dict: ...             # wrong — bare builtin
```

## DataFrame Iteration

- Prefer `zip(df[col_a], df[col_b])` over `iterrows()` when building lists from DataFrame rows — it avoids per-row Series construction and is significantly faster.

```python
# correct
[DatasetExample(input=inp, target=tgt)
 for inp, tgt in zip(rows[input_col], rows[target_col])]

# avoid
[DatasetExample(input=row[input_col], target=row[target_col])
 for _, row in rows.iterrows()]
```

## DataFrame Column Assignment

- Prefer `.assign()` chaining over direct column mutation when adding computed columns to a DataFrame before returning it. Chaining keeps the operation functional and avoids mutating an intermediate object.

```python
# correct — .assign() chaining
return (
    df
    .assign(
        n_vocab   = df["encoding"].map(lambda e: enc_meta[e].n_vocab),
        eot_token = df["encoding"].map(lambda e: enc_meta[e].eot_token),
    )
    .sort_values("model")
    .reset_index(drop=True)
)

# wrong — direct mutation before return
df["n_vocab"]   = df["encoding"].map(lambda e: enc_meta[e].n_vocab)
df["eot_token"] = df["encoding"].map(lambda e: enc_meta[e].eot_token)
return df.sort_values("model").reset_index(drop=True)
```

## DataFrame Naming

- All `pd.DataFrame` parameters and variables must start with the `df_` prefix so the type is
  immediately visible at the call site.
- Use a descriptive suffix after the prefix: `df_emb`, `df_summary_stats`, `df_record_stats`,
  `df_plot`, `df_emb_filtered`.
- Short-lived local throwaway DataFrames (never passed to another function) may omit the prefix,
  but named parameters and returned DataFrames must always use it.

```python
# correct — prefix makes type obvious
def show_metrics(self, df_summary_stats: pd.DataFrame, df_record_stats: pd.DataFrame) -> None: ...

# wrong — no prefix; type not visible from name
def show_metrics(self, summary_stats: pd.DataFrame, record_stats: pd.DataFrame) -> None: ...
```

## Comprehensions

- Prefer list, dict, and set comprehensions over building a temporary accumulator variable in an explicit loop, as long as the expression remains readable. A double-loop comprehension with a guard is acceptable; three or more nested loops or complex conditionals are a signal to use an explicit loop instead.

```python
# correct — dict comprehension with filter
values = {k: v for k, v in items.items() if v is not None}

# correct — double-loop dict comprehension to flatten nested dicts
return {
    k: v
    for sub in nested.values()
    for k, v in sub.items()
    if v is not None
}

# wrong — temporary accumulator
result = {}
for sub in nested.values():
    result.update({k: v for k, v in sub.items() if v is not None})
return result
```

## Attribute Access

- Prefer dot notation over dictionary string key access wherever the attribute is statically
  defined — dataclasses and objects with well-known fields.
- Use dictionary string key access (`d["key"]`) only when the structure is open or dynamic
  (e.g. `TypedDict` instances that merge with arbitrary runtime keys).

```python
# correct — dataclass with statically known fields
result.score
result.index

# wrong — string key on a statically-defined attribute
result["score"]

# correct — open TypedDict that merges with arbitrary document metadata at runtime
search_result["text"]
search_result["_score"]
```

## Optional Function Arguments

- When a function has optional keyword arguments, omit them entirely rather than passing empty or sentinel values (`[]`, `None`, `False`). Build a `kwargs` dict and add keys only when the value is known to be meaningful.

```python
# correct — output_keys omitted when absent; callee uses its own default
kwargs = dict(name=cfg.name, input_keys=list(cfg.input_keys))
if "output_keys" in cfg:
    kwargs["output_keys"] = list(cfg.output_keys)
fn(**kwargs)

# wrong — passes empty list even when there is nothing to say
fn(name=cfg.name, input_keys=list(cfg.input_keys), output_keys=[])
```

## Imports

- Never use `if TYPE_CHECKING:` guards to defer imports. All imports — including type-only ones — must appear at the top of the file unconditionally. This keeps type information fully available at runtime, keeps static analysis honest, and removes the two-import-path maintenance burden. If a module is an optional heavy dependency, install it in the appropriate `uv` dependency group (`evaluation`, `apps`, etc.) rather than hiding it behind a guard.

- Within a package, always use **relative imports** (`from .module import X`, `from ..sibling import Y`). Never use absolute package-name imports from inside the package itself.
- Outside the package (`projects/`, `examples/`, scripts), use absolute imports.

```python
# correct — relative import inside mypackage/subpkg/module.py
from .utils import helper

# wrong — absolute import inside the package
from mypackage.subpkg.utils import helper

# correct — absolute import from outside the package
from mypackage.subpkg.module import MyClass
```

- When importing multiple symbols from one module, count them: **4 or more** symbols → import the module itself and reference each symbol through it (`module.Symbol`); **3 or fewer** symbols → import each one directly by name. This keeps import blocks short for modules with many types without adding qualification noise for the common small case.

```python
# correct — 5+ symbols from one module: import the module, reference qualified
from .schemas import types

def handler(req: types.CreateRequest) -> types.Response:
    ...

# correct — 2-3 symbols from one module: import directly by name
from .schemas.types import Request, Response

def handler(req: Request) -> Response:
    ...

# wrong — qualifying access for only 2 symbols adds noise without benefit
from .schemas import types

def handler(req: types.Request) -> types.Response:
    ...
```

- When importing a module under its bare name would collide with a local variable, parameter, or another imported module of the same name, alias it to something unambiguous (e.g. `from .other_pkg import thing as other_thing`) rather than picking a name that shadows or gets shadowed.

- `from __future__ import annotations` must always be the **first line** of any Python file that includes it — before all other imports, before module docstrings.
- Bare `import X` statements come **before** `from X import Y` statements within each import block.
- After the `import` or `from` keyword, align module names with a **tab** so the module column lines up across all lines in the block.

```python
# correct
from __future__ import annotations
import  os
import  pandas as pd
from    typing import List, Union
from    .module import MyClass

# wrong — __future__ not first, from before import, no alignment
from typing import List
import pandas as pd
from __future__ import annotations
from .module import MyClass
```

## Exception Handling

- Prefer `try/except` over `if`-checks for recoverable error paths — let the operation attempt and catch failure rather than guarding with a condition.
- Place all success-path logic inside the `try` block so it is skipped on error without needing an explicit guard.
- Use the most specific exception type you can name. Fall back to `except Exception` only when the concrete type is unknown, may change across SDK versions, or would couple the code to an implementation detail.

```python
# correct — success path inside try; specific type when known
try:
    result = client.fetch(name)
    table  = build_table(result)
    console.print(table)
except SomeSDKError:
    console.print(f"[yellow]{name!r} not found[/yellow]")

# correct — generic fallback when SDK exception type is unstable/unknown
try:
    result = client.fetch(name)
    console.print(result)
except Exception:
    console.print(f"[yellow]{name!r} not found[/yellow]")

# wrong — if-check instead of exception handling
result = client.fetch_or_none(name)
if result is None:
    console.print(f"[yellow]{name!r} not found[/yellow]")
else:
    console.print(result)
```

## Package Structure

- Do **not** create `__init__.py` in subpackage directories. Python 3.3+ supports namespace packages natively — a directory without `__init__.py` is a valid, importable package.
- Do **not** create `__init__.py` solely to re-export symbols or shorten import paths. Always import directly from the module that defines the symbol.
- Only add `__init__.py` when the file contains genuine package-level initialisation logic (e.g. version assignment, plugin registration) — not mere re-exports or `__all__`.
- **Exception**: a subpackage's `__init__.py` may re-export a small, curated set of its most-used symbols for ergonomic convenience when explicitly decided for that package (e.g. `velari_core/core/__init__.py` re-exports `read_root_dir`/`read_env`/`read_cache_dir` from `.utils.env_utils`). This remains the exception, not the default — most subpackages should still follow the two rules above.

```python
# correct — import from the defining module
from mypackage.subpkg.module import MyClass

# wrong — __init__.py created just to enable a shorter import path
from mypackage.subpkg import MyClass

# exception — velari_core/core/__init__.py explicitly re-exports a curated set
from velari_core.core import read_root_dir
```

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

## .gitignore

- Anchor root-level ignore rules with a leading `/` so they match only the repo root, not nested directories with the same name.
- Never hardcode the package name in ignore rules. Use path-agnostic patterns or root-anchored paths instead.

```gitignore
# correct — only ignores /stores/ at the repo root
/stores/

# wrong — ignores any stores/ directory anywhere in the tree, including inside the package
stores/

# wrong — hardcodes the package name
mypackage/ai/stores/
```

## Enumerations

- Use `StrEnum` (from `enum`) for string-valued enumerations.
- Always use `auto()` instead of typing the string value explicitly — `StrEnum` + `auto()` produces the lowercase member name, so `OBJECT_STORAGE = auto()` yields `"object_storage"`.
- Member names must be UPPER_SNAKE_CASE; the lowercase string value is derived automatically.
- Align `=` signs at the same column across **all enumerations in the same file** — the column is set by the longest member name across all enums, plus a 2-space minimum gap.
- When a new enum is added or a member is renamed, re-check and re-align all enums in the file.

```python
from enum import StrEnum, auto

# correct — = column is global; driven by OBJECT_STORAGE (longest across both enums)
class SourceType(StrEnum):
    DATABASE         = auto()
    OBJECT_STORAGE   = auto()

class Priority(StrEnum):
    LOW              = auto()
    HIGH             = auto()

# wrong — explicit string duplicates the member name
class SourceType(StrEnum):
    DATABASE         = "database"
    OBJECT_STORAGE   = "object_storage"

# wrong — each enum uses its own local column instead of the global one
class SourceType(StrEnum):
    DATABASE       = auto()
    OBJECT_STORAGE = auto()

class Priority(StrEnum):
    LOW  = auto()
    HIGH = auto()
```

## Class Inheritance

- Normal classes (no `@dataclass` decorator) with no explicit parent must always inherit from `object` explicitly.
- `@dataclass`-decorated classes and classes that already inherit from a meaningful base (e.g. `StrEnum`, a library base class) are exempt.

```python
# correct — explicit object base for a plain class
class WebSearcher(object):
    ...

# wrong — implicit base omitted
class WebSearcher:
    ...

# exempt — @dataclass already implies object
@dataclass
class DatasetProfile(object):
    ...

# exempt — meaningful base class
class ContentFormat(StrEnum):
    ...
```

## Jupyter Notebook Editing

- Always **read the notebook** with the `Read` tool before inserting or replacing cells.
  Confirm the target `cell_id` is present in the output before calling `NotebookEdit`.
- `NotebookEdit` with a missing `cell_id` does **not** raise an error — it silently inserts
  the new cell at position 0 (the top of the notebook), which is almost never correct.
- After any `NotebookEdit`, re-read the notebook to verify the cell landed in the expected
  position before reporting the change as complete.

## Dataclass Namespace Field Order

- When a `@dataclass` also acts as a namespace (inner `@dataclass` classes nested inside it), declare the outer class's instance field annotations **after** all inner class definitions — not before.
- This keeps the class readable top-to-bottom: inner types are fully defined before the fields that reference them.

```python
# correct — inner types first, fields last
@dataclass
class DatasetProfile(object):
    @dataclass
    class Info:
        name: str

    @dataclass
    class Stats:
        summary: pd.DataFrame

    info:  DatasetProfile.Info   # ← fields at bottom
    stats: DatasetProfile.Stats

# wrong — fields declared before the types they reference
@dataclass
class DatasetProfile(object):
    info:  DatasetProfile.Info   # ← fields at top, before inner classes
    stats: DatasetProfile.Stats

    @dataclass
    class Info:
        name: str
```

## Dataclass Field Declarations

- Use `field()` from `dataclasses` whenever an attribute carries a default value or any field option (repr, compare, metadata, etc.). Do **not** use `field()` when the field is required and has no options — leave it as a bare annotation.
  - **Required (no default):** `name: Type` — bare annotation, no `field()`.
  - **Scalar default:** `name: Type = field(default=value)`
  - **Mutable / complex default:** `name: Type = field(default_factory=Callable)`
- Add `field` to the `from dataclasses import ...` line in any file that uses it.

```python
from dataclasses import dataclass, field
from typing import List, Optional

# correct
@dataclass
class Config:
    host:  str            # required — bare annotation
    port:  int            = field(default=8080)
    tags:  List[str]      = field(default_factory=list)
    label: Optional[str]  = field(default=None)

# wrong — plain assignment used instead of field() for defaults
@dataclass
class Config:
    host:  str            # this one is fine
    port:  int            = 8080          # should be field(default=8080)
    tags:  List[str]      = []            # mutable — must be field(default_factory=list)
    label: Optional[str]  = None          # should be field(default=None)
```
