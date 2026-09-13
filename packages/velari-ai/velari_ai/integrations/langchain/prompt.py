from    typing import Any, Dict, Optional, Tuple
from    functools import reduce
import  operator
# specific modules
from    langchain_core.prompts import ChatPromptTemplate
from    langchain_core.prompts.string import PromptTemplateFormat
# package modules
from    ...ai.prompt.builder import PromptBuilder


class PromptBuilderLangChain(PromptBuilder):
    """Builds LangChain ChatPromptTemplates from loaded prompt sources or raw strings.

    Args:
        template_format (PromptTemplateFormat): Template syntax used for every prompt this builder
            constructs; defaults to `"jinja2"`, matching every prompt YAML in `config/prompts/templates/`.
        **kwargs (Any): Forwarded to `PromptBuilder.__init__` (e.g. `prompt_dir`).

    Examples:
        >>> builder = PromptBuilderLangChain()
        >>> meta, prompt = builder.from_file("analysis.yaml", task_name="revise_query", prefix="analysis")
        >>> meta
        {'task': 'query revision for clarity'}
        >>> # next: render the template with meta + the caller's own variables, then hand the
        >>> # resulting messages to a chat model (e.g. self._model.invoke(messages) in LLMTaskNode)
        >>> messages = prompt.invoke({**meta, "query": "wut r agents"})
        >>> [m.__class__.__name__ for m in messages.to_messages()]
        ['SystemMessage', 'HumanMessage']
    """
    def __init__(self, template_format: PromptTemplateFormat = "jinja2", **kwargs) -> None:
        super().__init__(**kwargs)
        self._template_format: PromptTemplateFormat = template_format

    def from_file(
        self, prompt_name: str, task_name: Optional[str] = None, prefix: str = "tasks"
    ) -> Tuple[Dict[str, Any], ChatPromptTemplate]:
        """Load a prompt YAML file and build its ChatPromptTemplate.

        Args:
            prompt_name (str): File name within this builder's `prompt_dir`, e.g. `"tasks.yaml"`.
            task_name (Optional[str]): When given, look up `<prefix>.<task_name>` inside the file
                instead of reading `meta`/`system`/`user` from the file root — see `PromptBuilder.load_file`.
            prefix (str): Top-level key `task_name` is looked up under when given; defaults to `"tasks"`.

        Returns:
            Tuple[Dict[str, Any], ChatPromptTemplate]: The resolved entry's `meta` dict, and the built template.

        Examples:
            >>> builder = PromptBuilderLangChain()
            >>> meta, prompt = builder.from_file("tasks.yaml", task_name="classification_task")
            >>> meta
            {'task': 'content and query classification'}
        """
        source = self.load_file(prompt_name, task_name=task_name, prefix=prefix)
        return source.meta, self.from_strings(source.system, source.user)

    def from_strings(self, system: str, user: str) -> ChatPromptTemplate:
        """Build a ChatPromptTemplate directly from system/user template strings — no file involved.

        Args:
            system (str): System-turn template string (Jinja2 by default) — task instructions.
            user (str): User-turn template string, interpolating the caller's own variables.

        Returns:
            ChatPromptTemplate: A two-message (system, user) template, ready to `.invoke(...)`.

        Examples:
            >>> builder = PromptBuilderLangChain()
            >>> prompt = builder.from_strings(
            ...     system="Decide whether this support ticket needs to be escalated to a human agent.",
            ...     user="Ticket:\\n{{ ticket_text }}",
            ... )
            >>> # next: render with the caller's own variables — the result is ready to pass
            >>> # straight to a chat model's own .invoke(messages)
            >>> messages = prompt.invoke({"ticket_text": "My payment was charged twice, please help."})
            >>> [m.__class__.__name__ for m in messages.to_messages()]
            ['SystemMessage', 'HumanMessage']
        """
        return ChatPromptTemplate.from_messages(
            [("system", system), ("user", user)],
            template_format=self._template_format,
        )

    def from_template(self, template: str) -> ChatPromptTemplate:
        """Build a single human-turn ChatPromptTemplate from one template string.

        Args:
            template (str): User-turn-only template string (Jinja2 by default) — no system message.

        Returns:
            ChatPromptTemplate: A single-message (human) template, ready to `.invoke(...)`.

        Examples:
            >>> builder = PromptBuilderLangChain()
            >>> prompt = builder.from_template("Summarize the key findings from this report:\\n{{ report_text }}")
            >>> messages = prompt.invoke({"report_text": "Q3 revenue grew 12% year-over-year."})
            >>> messages.to_messages()[0].content
            'Summarize the key findings from this report:\\nQ3 revenue grew 12% year-over-year.'
        """
        return ChatPromptTemplate.from_template(template, template_format=self._template_format)

    def compose(self, *prompts: ChatPromptTemplate) -> ChatPromptTemplate:
        """Concatenate multiple ChatPromptTemplates' messages into one, in order (via '+').

        Useful for appending a reusable instruction (e.g. an output-format reminder) after a
        task's own template, without duplicating that instruction inside every task's own YAML —
        `prompts[0]`'s messages come first, then `prompts[1]`'s, and so on.

        Args:
            *prompts (ChatPromptTemplate): Templates to concatenate, earliest first.

        Returns:
            ChatPromptTemplate: One template whose messages are every input template's messages,
                in the same order the templates were passed in.

        Examples:
            >>> builder = PromptBuilderLangChain()
            >>> meta, topic_analysis = builder.from_file("tasks.yaml", task_name="topic_analysis_task")
            >>> reminder = builder.from_template("Return the topics as a bulleted list, nothing else.")
            >>> prompt = builder.compose(topic_analysis, reminder)
            >>> messages = prompt.invoke({
            ...     **meta, "content_type": "article",
            ...     "content": "A deep dive into vector databases and approximate nearest neighbor search.",
            ... })
            >>> [m.__class__.__name__ for m in messages.to_messages()]
            ['SystemMessage', 'HumanMessage', 'HumanMessage']
        """
        return reduce(operator.add, prompts)
