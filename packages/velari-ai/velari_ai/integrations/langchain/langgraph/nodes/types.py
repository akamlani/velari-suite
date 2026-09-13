
from    typing  import Any, Dict, Generic, Mapping, Optional, Type, TypeVar
from    pydantic import BaseModel
# specific modules
from    langchain_core.language_models import LanguageModelInput
from    langchain_core.messages import AIMessage
from    langchain_core.prompts import ChatPromptTemplate
from    langchain_core.prompt_values import PromptValue
from    langgraph.typing import ContextT
from    langgraph.runtime import Runtime
# package modules
from    .....ai.types import ModelConfig
from    ...utils  import build_chat_model

# Bound to Mapping[str, Any] rather than left unbound — every real graph state in this codebase is a
# TypedDict, which is structurally a Mapping, so this keeps is_message_node's state.get(...) call
# type-checked without needing Any. ContextT is reused from langgraph.typing (same TypeVar graph.py
# already parametrizes Runtime/StateGraph with), not redefined here.
StateT    = TypeVar("StateT", bound=Mapping[str, Any])
ResponseT = TypeVar("ResponseT", bound=BaseModel)


class NodeBase(Generic[StateT, ContextT]):
    def __init__(self, name: Optional[str] = None, **kwargs) -> None:
        self._name = name or type(self).__name__
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, state: StateT, *, runtime: Runtime[ContextT]) -> dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"

    def is_message_node(self, state: StateT) -> bool:
        return bool(state.get("messages"))


class LLMNode(NodeBase[StateT, ContextT]):
    """A NodeBase with one bound chat model — usable directly as a graph node.

    `__call__` invokes the bound model on `state["messages"]` and returns the new message for
    LangGraph's `add_messages` reducer to append — the standard "call the model" node shape, so
    this class can be passed straight to `Graph.add_step()` without a wrapping subclass.

    Args:
        model_config (Optional[ModelConfig]): Provider:model + kwargs; defaults to `ModelConfig()`.
        **kwargs (Any): Forwarded to `NodeBase.__init__` (e.g. `name`).
    """
    def __init__(self, model_config: Optional[ModelConfig] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_config, self._model = build_chat_model(model_config)

    def invoke_messages(self, messages: LanguageModelInput) -> AIMessage:
        """Call the bound model directly with the given messages."""
        return self._model.invoke(messages)

    def __call__(self, state: StateT, *, runtime: Runtime[ContextT]) -> dict:
        """Invoke the bound model on `state["messages"]`; the reducer appends the returned message."""
        return {"messages": [self.invoke_messages(state["messages"])]}


class LLMTaskNode(LLMNode[StateT, ContextT]):
    """An LLMNode bound to one already-built prompt template, invoked via render_prompt()/invoke_text().

    Accepts a `ChatPromptTemplate` built ahead of time — this class only knows how to *invoke* a
    prompt, not how to obtain one, so it isn't tied to any particular prompt source. Build the
    prompt with whichever `PromptBuilderLangChain` method fits (`.from_file()` for a prompt YAML,
    `.from_strings()`/`.from_template()` for one built entirely in memory, `.compose()` to
    concatenate several), fetch one from elsewhere (e.g. a remote prompt registry), or construct a
    `ChatPromptTemplate` directly — then pass the result in here.

    Overrides `LLMNode.__call__`'s `messages`-in/`messages`-out shape: this node's `__call__`
    spreads the whole graph `state` as this prompt's template variables (any keys the template
    doesn't reference are simply ignored) and writes the resulting text to `output_field` instead
    — set `output_field` to use this node directly as a graph step; leave it unset to only use
    `render_prompt()`/`invoke_text()` standalone (e.g. from another node's own `__call__`).

    Args:
        prompt (ChatPromptTemplate): The prompt to invoke — already built.
        prompt_meta (Optional[Dict[str, Any]]): Merged into every `render_prompt()` call alongside
            that call's own `**template_vars` — makes a prompt's own metadata (e.g. a YAML's
            `meta: {task: ...}`) available to the template body too, for a template that happens to
            reference it directly; harmless to pass when the template doesn't. Defaults to `{}`.
        output_field (Optional[str]): State key `__call__` writes this node's text result to;
            required only when using this node directly as a graph step (see above).
        model_config (Optional[ModelConfig]): Provider:model + kwargs; defaults to `ModelConfig()`.
        **kwargs (Any): Forwarded to `NodeBase.__init__` (e.g. `name`).

    Examples:
        >>> from velari_ai.integrations.langchain.prompt import PromptBuilderLangChain
        >>> builder = PromptBuilderLangChain()

        >>> # standalone use — invoke_text() directly, no graph involved
        >>> meta, prompt = builder.from_file("analysis.yaml", task_name="revise_query", prefix="analysis")
        >>> node = LLMTaskNode(prompt=prompt, prompt_meta=meta)
        >>> node.invoke_text(query="wut r agents")
        'Revised: "What are agents?"'

        >>> # as a graph step — output_field set, __call__ reads query straight from state
        >>> revise_node = LLMTaskNode(prompt=prompt, prompt_meta=meta, output_field="query_revision")
        >>> graph.add_step(name="revise_query", node=revise_node)
        >>> compiled.invoke({"query": "wut r agents"}, config=...)
        {'query': 'wut r agents', 'query_revision': 'Revised: "What are agents?"'}
    """
    def __init__(
        self,
        prompt:       ChatPromptTemplate,
        prompt_meta:  Optional[Dict[str, Any]] = None,
        output_field: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(model_config=model_config, **kwargs)
        self._prompt       = prompt
        self._prompt_meta  = prompt_meta or {}
        self._output_field = output_field

    def render_prompt(self, **template_vars: Any) -> PromptValue:
        """Render the bound prompt with the given template variables plus the prompt's own meta."""
        return self._prompt.invoke({**self._prompt_meta, **template_vars})

    def invoke_text(self, **template_vars: Any) -> str:
        """Render the bound prompt and run a plain (non-structured) call, returning its text."""
        response = self.invoke_messages(self.render_prompt(**template_vars))
        return response.text

    def __call__(self, state: StateT, *, runtime: Runtime[ContextT]) -> dict:
        """Render the bound prompt from `state` and write the model's text to `output_field`.

        Raises:
            ValueError: If `output_field` wasn't set at construction.
        """
        if self._output_field is None:
            raise ValueError(f"{self!r} has no output_field set — required to use as a graph node")
        return {self._output_field: self.invoke_text(**state)}

class LLMStructuredTaskNode(LLMTaskNode[StateT, ContextT], Generic[StateT, ResponseT, ContextT]):
    """An LLMTaskNode bound to a structured-output schema instead of returning plain text.

    Same `prompt`/`prompt_meta` contract as `LLMTaskNode` — accepts an already-built
    `ChatPromptTemplate`, not a prompt source — but every call is parsed into `response_schema`
    instead of returning raw text, via `invoke_task()` (this class's structured analogue of
    `LLMTaskNode.invoke_text()`). `__call__` writes the whole parsed model to `output_field`, not
    just one of its fields — a downstream node reads `state[output_field].<field>` for whichever
    part it needs.

    Args:
        prompt (ChatPromptTemplate): The prompt to invoke — already built.
        response_schema (Type[ResponseT]): Pydantic model the model's output is parsed into.
        prompt_meta (Optional[Dict[str, Any]]): Merged into every `render_prompt()` call alongside
            that call's own `**template_vars`; see `LLMTaskNode`. Defaults to `{}`.
        output_field (Optional[str]): State key `__call__` writes the parsed `response_schema`
            instance to; required only when using this node directly as a graph step.
        model_config (Optional[ModelConfig]): Provider:model + kwargs; defaults to `ModelConfig()`.
        **kwargs (Any): Forwarded to `NodeBase.__init__` (e.g. `name`).

    Examples:
        >>> from velari_ai.integrations.langchain.prompt import PromptBuilderLangChain
        >>> from velari_ai.ai.response import ResponseRelevanceGrade
        >>> builder = PromptBuilderLangChain()
        >>> meta, prompt = builder.from_file("analysis.yaml", task_name="grade_context", prefix="analysis")

        >>> # standalone use — invoke_task() directly, no graph involved
        >>> node = LLMStructuredTaskNode(prompt=prompt, prompt_meta=meta, response_schema=ResponseRelevanceGrade)
        >>> grade = node.invoke_task(query="How do I reset my password?", context="Go to Settings > Security > Reset Password.")
        >>> grade.is_relevant
        True

        >>> # as a graph step — output_field set, __call__ reads query/context straight from state
        >>> grade_node = LLMStructuredTaskNode(
        ...     prompt=prompt, prompt_meta=meta, response_schema=ResponseRelevanceGrade, output_field="grade",
        ... )
        >>> graph.add_step(name="grade_context", node=grade_node)
        >>> compiled.invoke({"query": "How do I reset my password?", "context": "..."}, config=...)
        {'query': 'How do I reset my password?', 'context': '...', 'grade': ResponseRelevanceGrade(is_relevant=True, ...)}
    """
    def __init__(
        self,
        prompt:          ChatPromptTemplate,
        response_schema: Type[ResponseT],
        prompt_meta:     Optional[Dict[str, Any]] = None,
        output_field:    Optional[str] = None,
        model_config:    Optional[ModelConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(prompt=prompt, prompt_meta=prompt_meta, output_field=output_field, model_config=model_config, **kwargs)
        self._response_schema  = response_schema
        self._structured_model = self._model.with_structured_output(response_schema)

    def invoke_task(self, **template_vars: Any) -> ResponseT:
        """Render the bound prompt, run the structured-output call, and narrow the result."""
        result = self._structured_model.invoke(self.render_prompt(**template_vars))
        if not isinstance(result, self._response_schema):
            raise TypeError(
                f"Expected {self._response_schema.__name__} from structured output, got {type(result).__name__}"
            )
        return result

    def __call__(self, state: StateT, *, runtime: Runtime[ContextT]) -> dict:
        """Render the bound prompt from `state` and write the parsed result to `output_field`.

        Raises:
            ValueError: If `output_field` wasn't set at construction.
        """
        if self._output_field is None:
            raise ValueError(f"{self!r} has no output_field set — required to use as a graph node")
        return {self._output_field: self.invoke_task(**state)}