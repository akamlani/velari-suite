# Documentation Guides

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

## Realistic Examples

- `Examples:` should use realistic, domain-relevant scenarios — not generic placeholders
  like `hello_world()`, `foo`/`bar`, `my_tool`, or toy examples like weather lookups.
  Ground examples in this project's actual domain (data services, MCP tools/resources,
  agent tool-calling, embeddings, etc.) so the example itself teaches something about how
  the function is actually used, not just its call syntax.

```python
# correct — grounded in a realistic enterprise scenario
Examples:
    >>> agent = Agent()
    >>> bound_llm = agent.bind([lookup_account_balance])
    >>> response = bound_llm.invoke("What's the outstanding balance on account ACC-10293?")
    >>> response.tool_calls
    [{'name': 'lookup_account_balance', 'args': {'account_id': 'ACC-10293'}, 'id': '...'}]

# wrong — generic toy example that could belong to any codebase
Examples:
    >>> agent = Agent()
    >>> bound_llm = agent.bind([my_tool])
    >>> response = bound_llm.invoke("What's the weather?")
    >>> response.tool_calls
    [{'name': 'my_tool', 'args': {...}, 'id': '...'}]
```

- Vary the scenario across different docstrings — don't reuse the same use case (e.g.
  the same account lookup) in every `Examples:` block throughout a file or module.
  Draw from different parts of the domain (billing, support tickets, research queries,
  document retrieval, etc.) so each example also demonstrates a different facet of how
  the codebase is actually used, not just the same story restated with a new method name.

```python
# wrong — every method's Examples reuses the identical billing lookup, even though
# each method does something different
def bind(self, tools): ...
    """
    Examples:
        >>> response = bound_llm.invoke("What's the outstanding balance on account ACC-10293?")
    """

def query(self, message): ...
    """
    Examples:
        >>> response = agent.query("What's the outstanding balance on account ACC-10293?")
    """

def run(self, message): ...
    """
    Examples:
        >>> result = agent.run("What's the balance on ACC-10293?")
    """

# correct — still enterprise-grounded, but each method's Examples draws from a
# different part of the domain
def bind(self, tools): ...
    """
    Examples:
        >>> response = bound_llm.invoke("What's the outstanding balance on account ACC-10293?")
    """

def query(self, message): ...
    """
    Examples:
        >>> response = agent.query("Summarize the top 3 open support tickets for customer ORG-4471.")
    """

def run(self, message): ...
    """
    Examples:
        >>> result = agent.run("What were Q3's key findings in the churn-analysis report?")
    """
```

## Examples Show the Setup Flow

- `Examples:` should show the sequence leading up to the call being documented —
  construct the object and run any required setup calls first — not just the isolated
  call in isolation. A reader needs to see where the object came from, not just the
  syntax for calling one method on it once they already have one.

```python
# correct — shows the construct → build → run flow before the method being documented
Examples:
    >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
    >>> agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
    >>> result = agent.run("What's the balance on ACC-10293?", thread_id="acc-10293-session")
    >>> result.log()               # just this call's final response
    >>> result.log(history=True)   # every message in the thread so far

# wrong — bare calls with no indication of where `result` came from
Examples:
    >>> result.log()
    >>> result.log(history=True)
```

## Docstring Concision

- Keep `Args:`/`Returns:` entries short, but don't compress away information a reader
  actually needs — what a parameter is looked up against, which fields are required vs
  optional, or why a fallback exists. Cut restated type names and generic filler
  ("registry mapping X to Y that implement Z"); keep the one or two facts that aren't
  obvious from the signature alone.
- Default to **one line per `Args:`/`Returns:` entry**. Wrap to a second line only when
  a fact genuinely doesn't fit on one — not to preserve phrasing that could be tightened.
- Cut prose paragraphs between the one-line summary and `Args:` — implementation
  rationale ("unlike the old approach, this delegates to...") belongs in the code, not
  the docstring. Keep a lead-in sentence only if it states a single fact the caller
  needs and isn't obvious from the signature (e.g. "returns `self`, so calls chain"),
  and keep that fact to one line too.
- Don't restate behavior that's already visible in the code being documented. If a
  default's behavior is already spelled out via short comments in the function body,
  point at it (`"defaults to the stack below"`) instead of re-describing each item in
  the docstring too — one place to keep in sync, not two.
- Default to **one scenario in `Examples:`**. A second example demonstrating an optional
  or advanced parameter rarely teaches more than the `Args:` entry for that parameter
  already did — add a second example only when the parameter changes the *shape* of the
  return value in a way `Args:` genuinely can't convey.
- When an example's result is a structured object, print **1-2 representative fields**,
  not every field on it. A `>>> result.metrics.<field>` line per attribute stops being
  illustrative past the second one — it's a field dump, not an example.

```python
# correct — one line per entry, no rationale paragraph; middleware entry points at the
# code instead of re-listing each default's behavior a second time; one Examples scenario
def build(self, tools, checkpointer=None, middleware=None, **kwargs) -> Self:
    """Build a full tool-calling agent graph via LangChain's create_agent().

    Returns `self` (not the compiled graph), so calls chain: `Agent(...).build(...).run(...)`.

    Args:
        tools (List[BaseTool]): Tools the agent may call.
        checkpointer (Optional[BaseCheckpointSaver]): Memory store; defaults to `InMemorySaver()`.
        middleware (Optional[List[Any]]): Defaults to the retry/limit/summarization stack
            below; pass `[]` to disable.

    Examples:
        >>> agent = Agent(agent_config=AgentConfig(name="billing-support-agent"))
        >>> agent.build([lookup_account_balance], system_prompt="You are a billing support assistant.")
        >>> result = agent.run("What's the balance on ACC-10293?", thread_id="acc-10293-session")
        >>> result.response.content
        'Account ACC-10293 has an outstanding balance of $1,204.50.'
    """
    middleware = middleware or [
        ModelRetryMiddleware(),
        ToolCallLimitMiddleware(run_limit=self._agent_config.max_tool_calls),
    ]

# wrong — rationale paragraph restates what create_agent() already does, the middleware
# entry re-describes each default a second time instead of pointing at the code, and a
# second Examples scenario + a field-dump of result.metrics adds nothing Args: didn't say
def build(self, tools, checkpointer=None, middleware=None, **kwargs) -> Self:
    """Build a full tool-calling agent graph via LangChain's create_agent().

    Unlike bind()/query()'s hand-rolled loop, this delegates to a compiled LangGraph
    state graph that runs the model/tool loop internally — after building, use run()
    to query it, or render_graph() since self._agent is a Runnable...

    Args:
        middleware (Optional[List[Any]]): create_agent() middleware stack; defaults to
            ModelRetryMiddleware() (retry failed model calls), ToolCallLimitMiddleware
            (caps tool-call volume at agent_config.max_tool_calls — a limit, not a
            retry), and SummarizationMiddleware (triggered every 50 messages). Pass
            [] to disable.

    Examples:
        >>> result = agent.run("What's the balance on ACC-10293?", thread_id="acc-10293-session")
        >>> result.response.content
        'Account ACC-10293 has an outstanding balance of $1,204.50.'
        >>> result.metrics.latency_sec
        0.842
        >>> result.metrics.message_stats.cnt_tool_requests
        1
        >>> result.metrics.message_stats.cnt_total_messages
        4
        >>> result.metrics.usage_stats.input_tokens
        128

        >>> result = (
        ...     Agent(agent_config=AgentConfig(name="billing-support-agent"))
        ...     .build([lookup_account_balance], system_prompt="You are a billing support assistant.")
        ...     .run("What's the balance on ACC-10293?", thread_id="acc-10293-session")
        ... )
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

## Constructor Docstrings

- Document `__init__` parameters via a **class-level docstring**, placed directly above
  `__init__` per `Docstring Spacing` above — not scattered inline comments on each
  `self.x = ...` assignment. Use a normal `Args:` section listing the constructor's
  parameters; this codebase has no separate `Attributes:` section, so don't introduce one.
- Keep an inline comment on a `self._x = ...` assignment only when that attribute's
  purpose isn't documented anywhere else in the class — e.g. a value some other method's
  own docstring already explains needs no repeat comment in `__init__`.

```python
# correct — constructor params documented once, on the class; only _tools_by_name gets
# an inline comment, since nothing else in the class documents what it's for
class Agent(object):
    """LangChain chat-model wrapper — manual tool-calling (bind/query) or a create_agent() graph (build/run).

    Args:
        model_config (Optional[ModelConfig]): Provider:model + kwargs; defaults to `ModelConfig()`.
        agent_config (Optional[AgentConfig]): Identity/tool-loop settings; `.name` forwarded to `build()`.
        **kwargs (Any): Extra provider kwargs (e.g. `api_key`); wins over `model_config.extra` on collision.
    """
    def __init__(self, model_config=None, agent_config=None, **kwargs) -> None:
        self._model_config = model_config or ModelConfig()
        self._name = self._agent_config.name
        self._bound_llm = None
        # set by bind() — name -> BaseTool lookup so query() can execute requested tool_calls.
        self._tools_by_name: Dict[str, BaseTool] = {}

# wrong — no class docstring; every attribute's purpose explained via a scattered
# inline comment instead, including ones already documented on other methods
class Agent(object):
    def __init__(self, model_config=None, agent_config=None, **kwargs) -> None:
        self._model_config = model_config or ModelConfig()
        # identifies this agent in logs/tracing and multi-agent setups; forwarded to
        # create_agent()'s own `name` param in build().
        self._name = self._agent_config.name
        # set by bind() — a tool-bound Runnable for standalone use outside the
        # create_agent graph below; independent of self._model / build().
        self._bound_llm = None
```

## Comments

- Don't add an inline comment that only restates what a well-named call or class already
  says. If the comment and the identifier next to it would read the same to someone who
  doesn't know the library, the comment isn't adding information.
- Don't leave placeholder comments describing functionality that has no corresponding
  code (a TODO-style comment sitting above nothing). If the feature isn't built, a
  comment gesturing at it isn't documentation — track it outside the file (an issue, a
  plan) instead of leaving it in a list or function body.
- Keep comments that explain a genuinely non-obvious "why" (a hidden constraint, a
  workaround, a runtime invariant the type checker can't see) — this is the one case a
  comment earns its place, per the top-level `CLAUDE.md`/`AGENTS.md` guidance. But keep
  it to the facts a reader actually needs; a comment spread across 5-6 lines usually has
  a 1-2 line version that says the same thing.

```python
# correct — no redundant restating, and the one comment that remains explains a real
# invariant the type checker can't see, in as few lines as the fact needs
middleware = [
    ModelRetryMiddleware(),
    ToolRetryMiddleware(),
    ToolCallLimitMiddleware(run_limit=self._agent_config.max_tool_calls),
    SummarizationMiddleware(model=self._model, trigger=("messages", 50)),
]

def _prepare_call(self, method_name, thread_id):
    # Returns self._agent (narrowed to non-Optional) since Optional narrowing doesn't
    # cross method boundaries. A missing thread_id gets a fresh uuid so history just
    # doesn't persist.
    ...

# wrong — each comment just restates the class name next to it, two comments describe
# features that don't exist anywhere in the code, and the invariant comment below takes
# 6 lines to say what the correct version above says in 2
middleware = [
    # Fault Tolerance at infrastructure level - retry on transient failures
    ModelRetryMiddleware(),  # retry failed model calls
    ToolRetryMiddleware(),  # retry failed tool calls
    ToolCallLimitMiddleware(run_limit=self._agent_config.max_tool_calls),  # cap tool-call volume
    SummarizationMiddleware(model=self._model, trigger=("messages", 50)),  # condense long threads
    # PII Detection and Redaction
    # Steering Human in-the-loop for critical decisions
]

def _prepare_call(self, method_name, thread_id):
    # Returns self._agent (rather than just validating it) so callers get a
    # non-Optional local to call .invoke()/.stream()/etc. on — self._agent's own type
    # stays Optional, which static analysis can't narrow across a separate method call.
    # LangGraph also requires a thread_id whenever a checkpointer is attached (build()'s
    # default) — generate a fresh one so an omitted thread_id still means "no history
    # persists," instead of failing outright.
    ...
```

## Docstring Deviation for Tests

- One-line module docstring only (`"""Tests for <module>."""`). Test functions/methods
  do not get docstrings — the test name should describe the behavior on its own. This
  overrides the general `## Docstrings` rule above for test code specifically; see
  `.claude/rules/guidelines/test_guides.md` for the rest of the test-writing conventions.
