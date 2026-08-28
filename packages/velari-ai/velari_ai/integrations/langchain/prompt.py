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
        """Build a ChatPromptTemplate directly from system/user template strings — no file involved."""
        return ChatPromptTemplate.from_messages(
            [("system", system), ("user", user)],
            template_format=self._template_format,
        )

    def from_template(self, template: str) -> ChatPromptTemplate:
        """Build a single human-turn ChatPromptTemplate from one template string."""
        return ChatPromptTemplate.from_template(template, template_format=self._template_format)

    def compose(self, *prompts: ChatPromptTemplate) -> ChatPromptTemplate:
        """Concatenate multiple ChatPromptTemplates' messages into one, in order (via '+').

        Args:
            *prompts (ChatPromptTemplate): Templates to concatenate, earliest first.

        Returns:
            ChatPromptTemplate: One template whose messages are every input template's messages, in order.
        """
        return reduce(operator.add, prompts)
