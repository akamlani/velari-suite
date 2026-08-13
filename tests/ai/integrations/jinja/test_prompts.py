"""Tests for velari_ai.integrations.jinja.prompts."""


def test_render_prompt_substitutes_scalar_variables():
    from velari_ai.integrations.jinja.prompts import render_prompt

    result = render_prompt(
        "You are a {{ assistant_type }} assistant responsible for {{ task }}.",
        assistant_type="customer support ticket",
        task="classifier",
    )

    assert result == "You are a customer support ticket assistant responsible for classifier."


def test_render_prompt_renders_list_one_item_per_line():
    from velari_ai.integrations.jinja.prompts import render_prompt

    template = "Categories:\n{% for category in categories %}- {{ category }}\n{% endfor %}"

    result = render_prompt(template, categories=["billing", "technical", "general"])

    assert result == "Categories:\n- billing\n- technical\n- general\n"


def test_render_prompt_renders_dict_one_key_value_per_line():
    from velari_ai.integrations.jinja.prompts import render_prompt

    template = (
        "Descriptions:\n"
        "{% for category, description in category_map.items() %}"
        "- {{ category }}: {{ description }}\n"
        "{% endfor %}"
    )
    category_map = {
        "billing": "Questions about invoices, charges, or payments.",
        "technical": "Questions about product functionality or bugs.",
    }

    result = render_prompt(template, category_map=category_map)

    assert result == (
        "Descriptions:\n"
        "- billing: Questions about invoices, charges, or payments.\n"
        "- technical: Questions about product functionality or bugs.\n"
    )
