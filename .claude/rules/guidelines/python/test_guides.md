# Test Guides

Preferences established for this project's test suite. AI agents and contributors should follow these when writing or reviewing tests. These apply on top of `.claude/rules/guidelines/python/dev_guides.md` — general coding conventions still apply to test code except where explicitly overridden below.

## Location and Naming

- Tests are centralized under the root `tests/` directory (`[tool.pytest.ini_options] testpaths = ["tests"]`), never co-located next to source files, regardless of how many packages the workspace has.
- Mirror the package/module structure: `tests/<package-shortname>/test_<module>.py`.

```
# correct
tests/core/test_filesystem.py     # tests velari_core/core/io/filesystem.py
tests/data/test_storage.py        # tests velari_data/storage.py

# wrong — not centralized under tests/
packages/velari-core/velari_core/core/io/test_filesystem.py
```

## Organizing Tests Within a File

- Default to flat `test_*` functions when a file covers one cohesive subject.
- Group into plain classes (`class TestSomething:` — no `unittest.TestCase`, no `__init__`) when a single test file covers multiple distinct classes or clearly separate concerns. Drop the now-redundant subject prefix from each method name once the class disambiguates it.

```python
# correct — one file testing two unrelated classes
class TestLocalKeyValueStore:
    def test_put_get(self): ...
    def test_filter(self): ...

class TestLocalDocumentStore:
    def test_upsert_and_get_dict(self): ...

# wrong — redundant prefix once grouped in a class
class TestLocalKeyValueStore:
    def test_local_key_value_store_put_get(self): ...
```

## Imports

- Generic testing tools (`pytest`, stdlib helpers like `json`, `dataclasses.dataclass`, `contextlib`, `io`) go at the top of the file, same as any other module.
- Import the module/class **under test** inside each test function/method, not at module top — keeps every test's dependency explicit and self-contained.

```python
# correct
import pytest  # generic tool, top of file

def test_put_get():
    from velari_data.storage import LocalKeyValueStore  # module under test, inside the test
    ...

# wrong — module under test imported at module top
from velari_data.storage import LocalKeyValueStore

def test_put_get():
    ...
```

## Docstrings

- See `.claude/rules/guidelines/python/doc_guides.md`'s "Docstring Deviation for Tests" —
  one-line module docstring only, no docstrings on test functions/methods.

## Naming

- Name tests `test_<behavior>_<condition>`, describing what's being verified, not the mechanics: `test_delete_missing_raises_filenotfounderror`, `test_upsert_without_id_generates_one`, `test_get_properties_remote_head_failure_falls_back`.

## Test Granularity

- One behavior per test. Prefer several small, focused tests over one large test asserting many unrelated things — makes failures immediately legible.
- Use `@pytest.mark.parametrize` instead of copy-pasting near-identical tests for different inputs.

```python
# correct
@pytest.mark.parametrize("archive_name", ["bundle.zip", "bundle.tar.gz"])
def test_compress_extract_roundtrip(self, tmp_path, archive_name): ...

# wrong — copy-pasted body, only the archive extension differs
def test_compress_extract_roundtrip_zip(self, tmp_path): ...
def test_compress_extract_roundtrip_targz(self, tmp_path): ...
```

## Minimize Test Count and Verbosity

- Prefer fewer, higher-signal tests over exhaustive coverage of every getter, default, or
  trivial code path. Before adding a test, check whether it would fail for any reason other
  than the declaration itself being wrong — if not, it's testing the language/framework, not
  your code, and isn't worth the maintenance cost.
- Don't test the same code path twice at different layers with equivalent fake data. If a
  private helper's behavior is already fully exercised by a higher-level test (e.g. a public
  method that calls it) using the same fake response shape, keep the test at whichever layer
  is more informative on failure and drop the other — don't verify both.
- When two subclasses share a base class's method unchanged (neither overrides it), test that
  method's behavior once, through either subclass — not once per subclass.
- If a fake/mock class is copy-pasted verbatim across two test files, that's a signal the two
  tests cover the same scenario, not a signal to extract a shared fixture — remove the
  redundant test instead.

```python
# wrong — tests a Pydantic field's declared default, not any logic of ours
def test_gateway_message_tool_calls_default_to_none():
    message = GatewayMessage(role=Role.ASSISTANT, content="")
    assert message.tool_calls is None

# wrong — EmbeddingAdapter.embed_texts() is already covered directly (unit-level, below);
# re-verifying the identical translation through LLMGateway.embed() with the same fake
# response class adds no new coverage, since embed() is a one-line delegation
def test_embed_resolves_adapter_and_returns_gateway_embedding_response(monkeypatch): ...
def test_openai_embedding_adapter_embed_texts_returns_gateway_response(monkeypatch): ...

# wrong — ModelRegistry and EmbeddingRegistry both inherit _AdapterRegistry.resolve()
# unchanged; testing the shared "unregistered provider raises" path via both subclasses
# tests the same lines of code twice
def test_resolve_unregistered_provider_raises_valueerror(): ...
def test_embedding_registry_resolve_unregistered_provider_raises_valueerror(): ...

# correct — one test per distinct behavior, parametrized where the behavior itself repeats
@pytest.mark.parametrize("provider, adapter_cls", [(OPENAI, OpenAIAdapter), (ANTHROPIC, AnthropicAdapter)])
def test_resolve_returns_adapter_registered_for_provider(provider, adapter_cls): ...
```

## Exercise the Public API

- Test through the public entry points (e.g. `upsert()`/`get()`), not private helpers (`_to_dict()`), even when the goal is covering a private method's branches — keeps tests resilient to internal refactors.

## Fixtures Over Mocking Libraries

- Prefer pytest's built-in fixtures — `tmp_path` for filesystem work, `caplog` for logging assertions, `monkeypatch` for patching module-level functions/attributes — over adding a mocking dependency (no `respx`, no `unittest.mock`, unless a genuine need arises that these can't cover).
- When a dependency needs faking (e.g. `httpx.stream`/`put`/`head`), write a small local fake (a plain class or function) scoped to exactly what the code under test touches, rather than a general-purpose mock object.

```python
# correct — minimal hand-written fake, patched via monkeypatch
class _FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self): ...
    def iter_bytes(self):
        yield self._content

monkeypatch.setattr(httpx, "put", lambda *a, **kw: _FakeResponse())

# avoid — reaching for unittest.mock.MagicMock for something this small
```

## Assertions

- Use `pytest.raises(ExceptionType)` for expected errors, not a manual `try/except` + `assert False`.

```python
# correct
with pytest.raises(KeyError):
    store["missing"]

# wrong
try:
    store["missing"]
    assert False, "expected KeyError"
except KeyError:
    pass
```

## Typing Deviation from `dev_guides.md`

- Test functions/methods do not need return-type annotations (`-> None`) or full parameter typing, unlike normal application code. Fixture parameters (`tmp_path`, `monkeypatch`, `caplog`) are left unannotated, matching this project's existing test suite.
