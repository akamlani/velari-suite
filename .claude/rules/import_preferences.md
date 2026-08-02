# Import Preferences

Preferences for how imports are structured in this project. AI agents and contributors
should follow these.

## Type-Checking Imports

- Never use `if TYPE_CHECKING:` guards to defer imports. All imports — including
  type-only ones — must appear at the top of the file unconditionally. This keeps type
  information fully available at runtime, keeps static analysis honest, and removes the
  two-import-path maintenance burden. If a module is an optional heavy dependency,
  install it in the appropriate `uv` dependency group (`evaluation`, `apps`, etc.)
  rather than hiding it behind a guard.

## Relative vs. Absolute Imports

- Within a package, always use **relative imports** (`from .module import X`,
  `from ..sibling import Y`). Never use absolute package-name imports from inside the
  package itself.
- Outside the package (`projects/`, `examples/`, scripts), use absolute imports.

```python
# correct — relative import inside mypackage/subpkg/module.py
from .utils import helper

# wrong — absolute import inside the package
from mypackage.subpkg.utils import helper

# correct — absolute import from outside the package
from mypackage.subpkg.module import MyClass
```

## Import Qualification

- When importing multiple symbols from one module, count them: **4 or more** symbols →
  import the module itself and reference each symbol through it (`module.Symbol`);
  **3 or fewer** symbols → import each one directly by name. This keeps import blocks
  short for modules with many types without adding qualification noise for the common
  small case.

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

## Aliasing

- When importing a module under its bare name would collide with a local variable,
  parameter, or another imported module of the same name, alias it to something
  unambiguous (e.g. `from .other_pkg import thing as other_thing`) rather than picking a
  name that shadows or gets shadowed.

## Ordering, Alignment, and Grouping

- `from __future__ import annotations` must always be the **first line** of any Python
  file that includes it — before all other imports, before module docstrings.
- `from __future__ import annotations` also lets a classmethod return its own enclosing
  class, unquoted, even though the class isn't fully defined yet at the point the method
  signature is parsed — a common case for factory-style classmethods
  (`from_config`/`from_defaults`/etc.). Without the future import, that return
  annotation would need to be quoted instead.

```python
# correct — future import in place, no quoting needed for the self-reference
from __future__ import annotations

class Config:
    @classmethod
    def from_defaults(cls) -> Config:
        return cls()

# wrong — without the future import, the forward reference must be quoted
class Config:
    @classmethod
    def from_defaults(cls) -> "Config":
        return cls()
```

- Bare `import X` statements come **before** `from X import Y` statements within each
  import block.
- After the `import` or `from` keyword, align module names with a **tab** so the module
  column lines up across all lines in the block. For `from X import Y` lines
  specifically, also align the `import` keyword itself at a second, consistent column —
  sized to the longest module name in the block plus a 2-space minimum gap — so it isn't
  left ragged even though the module names differ in length.
- When a file's imports span more than one logical category (e.g. standard-library and
  third-party imports vs. local/relative ones), separate the groups with a short comment
  header rather than leaving them interleaved or unlabeled — `# package modules` is the
  most common label, for the group of relative/local imports.

```python
# correct — module names align at one column; `import` aligns at a second column,
# sized to the longest module name (`omegaconf`) plus a 2-space gap; a comment header
# separates the local/relative import from the third-party ones above it
from __future__ import annotations
import  os
import  pandas as pd
from    abc        import ABC
from    omegaconf  import DictConfig
from    typing     import List, Union
# package modules
from    .module import MyClass

# wrong — __future__ not first, from before import, no column alignment,
# and no grouping signal for the local import
from typing import List
import pandas as pd
from __future__ import annotations
from .module import MyClass
```
