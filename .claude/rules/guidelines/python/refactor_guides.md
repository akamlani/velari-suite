# Refactoring Guidelines

Preferences for how AI agents and contributors should approach *refactoring* work in this
repository — changing existing code's structure, types, or internal design without an explicit new
feature request. This is distinct from `dev_guides.md` (what the resulting code should look
like), `test_guides.md` (how tests are written), and `agent_guides.md` (general working
habits) — this file governs the *process* of getting from "working but rough" to "correct and
clean," and the judgment calls that come up along the way.

## Verify Before Claiming Something Isn't Possible

Before telling the user a further simplification, reduction, or fix "isn't possible," check whether
that conclusion rests on a constraint *they* actually stated, or one you silently added yourself.
If pushed back on, re-examine your own assumptions before re-asserting the limitation — the fix is
often available once an over-restrictive self-imposed constraint is lifted.

```python
# wrong — concluding "no further reduction possible" because bare-Python accumulation was assumed
# to be the only option, when the actual constraint was narrower ("don't delegate the whole
# operation to one builtin", not "don't use any stdlib helper at all")
def batch_iterable(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# correct — re-examined the actual constraint, found a low-level primitive (not the banned
# batching abstraction itself) that removes the accumulator entirely
from itertools import islice

def batch_iterable(iterable, batch_size):
    it = iter(iterable)
    for first in it:
        yield [first, *islice(it, batch_size - 1)]
```

## Stay Scoped to What Was Asked

Fix the thing that was asked about, not everything adjacent to it that also looks improvable. When
you notice a separate issue while working (an unused-variable hint, a dead code path, a naming
inconsistency in a sibling function), name it explicitly to the user rather than folding it
silently into the current diff — let them decide if it's in scope.

```python
# wrong — asked to fix a typing error in _extract_texts_and_metadatas; also "cleans up" the
# unrelated unused `batch_size` hint in a different method three functions down, unasked
def upsert_merge(self, texts, metadatas, embeddings, batch_size):  # <- also touched, out of scope
    ...

# correct — fixes only the reported function; the unrelated hint gets a one-line callout instead
# ("_upsert_merge's unused batch_size param is a separate, pre-existing issue — want that fixed
# too, or leave it?"), not a silent edit
```

## Consolidate Duplication Into a Single Source of Truth

When you find the same construction, kwargs, or logic block repeated three or more times (e.g. the
same four kwargs passed to every `Chroma(...)`/`Chroma.from_documents(...)`/`Chroma.from_texts(...)`
call site), extract a shared property or helper rather than patching each copy individually. Do
this as its own refactor, verified independently — don't fold it into an unrelated bug fix.

```python
# wrong — same 3 kwargs duplicated across 4 construction sites; a future change to any of them
# means remembering to update all 4, and they will eventually drift
Chroma(client=self._client, collection_name=self._collection_name,
       collection_metadata={"hnsw:space": "cosine"}, embedding_function=self._embedding_fn)
# ...(repeated 3 more times with only embedding_function/embedding differing)

# correct — one source of truth, reused everywhere
@property
def _collection_kwargs(self) -> Dict[str, Any]:
    return {
        "client":              self._client,
        "collection_name":     self._collection_name,
        "collection_metadata": {"hnsw:space": self._distance_metric},
    }

Chroma(embedding_function=self._embedding_fn, **self._collection_kwargs)
```

## Prefer Standard-Library Solutions, but Respect Explicit "Hand-Rolled" Constraints

Default to delegating to a well-tested standard-library or framework primitive over reimplementing
it by hand. When the user explicitly asks for a hand-rolled version (for teaching, for avoiding a
specific dependency, or to keep control of the exact algorithm), that constraint is about not
delegating the *whole operation* to one builtin — it does not automatically forbid every low-level
primitive that helps you author the logic yourself. Confirm which is meant rather than assuming the
strictest possible reading if the user pushes back on an over-strict interpretation.

## No Explicit Casting as a Typing Shortcut

`cast()` and `# type: ignore` silence the type checker without fixing anything — treat them as a
last resort, not a first response to a Pylance/Pyright error. Prefer, in order: (1) correcting an
overly defensive type (e.g. `Optional[Any]` on a field that's never actually `None` once
constructed becomes plain `Any`); (2) real narrowing via an `isinstance`/`is None` check, including
inside a comprehension's filter clause; (3) a small helper that raises on a genuine precondition
violation and returns a non-`Optional` type on success. Only reach for `cast()` when none of those
apply and the type checker is provably wrong.

```python
# wrong — silences the checker without addressing why it's unhappy
return cast(List[Document], vectorstore.similarity_search(query, k=k))

# correct — the field's type was overly defensive; corrected instead of worked around
self._client: Any = None   # was Optional[Any], but every concrete subclass sets it immediately
```

## Reconcile Runtime Safety with Static Typing — try/except Alone Doesn't Narrow

A `try`/`except` around a risky attribute access fixes the *runtime* behavior but does not give the
type checker any static narrowing — it still sees the same `Optional[...]` type inside the `try`
block regardless of what's caught around it. When both exception-based control flow and a clean
type-check are required, use a small helper that raises on the `None` case and returns the
non-`Optional` type on success; call it inside the `try` and catch what it raises.

```python
# wrong — runtime-safe, but Pylance still flags every access inside the try block
def retrieve_candidates(self, query, k=3):
    try:
        return self._vectorstore.similarity_search(query, k=k)  # still Optional[Any] to the checker
    except AttributeError:
        return []

# correct — the raise happens once, in a helper with an honest return type; the checker is happy
# because `vectorstore` is `Any`, not `Optional[Any]`, from that point on
def _require_vectorstore(self) -> Any:
    if self._vectorstore is None:
        raise RuntimeError("Vector store not loaded — call load() first")
    return self._vectorstore

def retrieve_candidates(self, query, k=3):
    try:
        vectorstore = self._require_vectorstore()
        return vectorstore.similarity_search(query, k=k)
    except RuntimeError:
        return []
```

## Don't Let Strategy-Specific Parameters Bloat a Shared Function Signature

When a function dispatches between several strategies/branches and only one branch needs a given
parameter, don't add that parameter to the function's named signature — every caller using the
other branches carries dead, meaningless arguments. Forward `**kwargs` to whichever underlying
call the active branch makes instead, and let that callee's own defaults apply when the caller
doesn't override them.

```python
# wrong — fetch_k/lambda_mult are MMR-only, but sit in the signature for every strategy
def retrieve_candidates(self, query, strategy=..., k=3, fetch_k=20, lambda_mult=0.5): ...

# correct — no strategy-specific params in the signature; each branch forwards what it needs,
# and MMR's own sensible defaults live where they're used, not duplicated into the signature
def retrieve_candidates(self, query, strategy=..., k=3, **kwargs):
    if strategy == RetrieverStrategy.VECTORSTORE_MMR:
        mmr_defaults = {"fetch_k": 20, "lambda_mult": 0.5}
        return vectorstore.max_marginal_relevance_search(query, k=k, **{**mmr_defaults, **kwargs})
    ...
```

Keep constants like `mmr_defaults` scoped to the function or class that uses them, not hoisted to
module level, unless genuinely shared across the whole module.

## Verify Every Refactor Live, Not Just Statically

A refactor isn't done when the diagnostics are clean. For every change: (1) check
`mcp__ide__getDiagnostics` on the touched file(s); (2) run the actual code — a real call, not just
an import — and confirm the output matches what any docstring `Examples:` claims, character for
character where practical; (3) run the targeted tests, then the full suite, and confirm the only
failures are pre-existing ones you already know about. When a claim about *why* something fails
(a Pyright error, a runtime exception) matters to the fix, reproduce it directly — via a standalone
`pyright` run or a live call — rather than reasoning about it from memory.

## Clean Up Verification Side Effects

Any file, directory, or registered state created purely to verify a refactor (a scratch script, a
temporary `.sqlite`/cache file, an ephemeral collection) gets removed in the same turn once
verification is complete — see `agent_guides.md` for the full rule and the out-of-repo case.

## Re-derive Alignment/Formatting Precisely When Structure Changes

When a refactor adds, removes, or renames an enum member, dataclass field, or aligned import, don't
eyeball the new spacing — recompute it (e.g. via a one-line script using `str.ljust`) against the
actual current longest name in that block, per `dev_guides.md`'s alignment rules. An
approximately-aligned block is more misleading than an unaligned one.

## Rename and Reference-Check Repo-Wide

Before finishing a rename (a function, an enum member, a class), grep the whole repo — including
notebooks (`.ipynb`), not just `.py` files — for every reference, and update tests and docs in the
same change. A rename that leaves one notebook cell referencing the old name is a broken rename,
not a finished one.
