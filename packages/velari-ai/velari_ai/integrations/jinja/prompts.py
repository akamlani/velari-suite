from __future__ import annotations

from typing import Any

from jinja2 import Template


def render_prompt(template: str, **context: Any) -> str:
    """Render a Jinja2 prompt template string with the given context variables.

    Args:
        template (str): Jinja2 template source, e.g. a task's `system`/`user` prompt.
        **context (Any): Template variables — plain values, lists, or dicts (e.g.
            `categories: List[str]`, `category_map: Dict[str, str]`).

    Returns:
        str: The rendered prompt text.

    Examples:
        >>> template = "Categories:\\n{% for c in categories %}- {{ c }}\\n{% endfor %}"
        >>> render_prompt(template, categories=["billing", "technical"])
        'Categories:\\n- billing\\n- technical\\n'
    """
    return Template(template, trim_blocks=True, lstrip_blocks=True).render(**context)
