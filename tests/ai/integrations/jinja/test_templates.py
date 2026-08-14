"""Tests for velari_ai.integrations.jinja.templates."""

import pytest


def test_render_from_string_substitutes_variables():
    from velari_ai.integrations.jinja.templates import TemplateJ2

    result = TemplateJ2.render_from_string("Hello, {{ name }}!", {"name": "Ada"})

    assert result == "Hello, Ada!"


def test_render_from_string_merges_kwargs_into_context():
    from velari_ai.integrations.jinja.templates import TemplateJ2

    result = TemplateJ2.render_from_string("{{ greeting }}, {{ name }}!", {"name": "Ada"}, greeting="Hi")

    assert result == "Hi, Ada!"


def test_render_from_string_renders_list_one_item_per_line():
    from velari_ai.integrations.jinja.templates import TemplateJ2

    template = "Categories:\n{% for category in categories %}- {{ category }}\n{% endfor %}"

    result = TemplateJ2.render_from_string(template, {"categories": ["billing", "technical", "general"]})

    assert result == "Categories:\n- billing\n- technical\n- general\n"


def test_render_from_string_renders_dict_one_key_value_per_line():
    from velari_ai.integrations.jinja.templates import TemplateJ2

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

    result = TemplateJ2.render_from_string(template, {"category_map": category_map})

    assert result == (
        "Descriptions:\n"
        "- billing: Questions about invoices, charges, or payments.\n"
        "- technical: Questions about product functionality or bugs.\n"
    )


def test_load_template_returns_compiled_template(tmp_path):
    from velari_ai.integrations.jinja.templates import TemplateJ2

    tmp_path.joinpath("child.j2").write_text("Hello, {{ name }}!")
    prompt_exec = TemplateJ2(template_path=str(tmp_path))

    template = prompt_exec.load_template("child.j2")

    assert template.render(name="John") == "Hello, John!"


def test_load_text_returns_raw_stripped_content(tmp_path):
    from velari_ai.integrations.jinja.templates import TemplateJ2

    tmp_path.joinpath("child.j2").write_text("  Hello, {{ name }}!  \n")
    prompt_exec = TemplateJ2(template_path=str(tmp_path))

    result = prompt_exec.load_text("child.j2")

    assert result == "Hello, {{ name }}!"


def test_load_source_returns_raw_stripped_content(tmp_path):
    from velari_ai.integrations.jinja.templates import TemplateJ2

    tmp_path.joinpath("child.j2").write_text("  Hello, {{ name }}!  \n")
    prompt_exec = TemplateJ2(template_path=str(tmp_path))

    result = prompt_exec.load_source("child.j2")

    assert result == "Hello, {{ name }}!"


def test_render_substitutes_content_into_template(tmp_path):
    from velari_ai.integrations.jinja.templates import TemplateJ2

    tmp_path.joinpath("child.j2").write_text("Hello, {{ name }}! You are {{ age }}.")
    prompt_exec = TemplateJ2(template_path=str(tmp_path))

    result = prompt_exec.render("child.j2", content={"name": "John", "age": 30})

    assert result == "Hello, John! You are 30."


def test_render_exposes_len_and_type_globals(tmp_path):
    from velari_ai.integrations.jinja.templates import TemplateJ2

    tmp_path.joinpath("child.j2").write_text("{{ len(tags) }}:{{ type(tags).__name__ }}")
    prompt_exec = TemplateJ2(template_path=str(tmp_path))

    result = prompt_exec.render("child.j2", content={"tags": ["billing", "support"]})

    assert result == "2:list"


def test_render_strips_trailing_whitespace(tmp_path):
    from velari_ai.integrations.jinja.templates import TemplateJ2

    tmp_path.joinpath("child.j2").write_text("Hello, {{ name }}!\n\n\n")
    prompt_exec = TemplateJ2(template_path=str(tmp_path))

    result = prompt_exec.render("child.j2", content={"name": "John"})

    assert result == "Hello, John!"


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.load_text("child.j2"),
        lambda t: t.load_source("child.j2"),
        lambda t: t.load_template("child.j2"),
        lambda t: t.render("child.j2", content={}),
    ],
)
def test_file_based_methods_without_template_path_raise_runtimeerror(call):
    from velari_ai.integrations.jinja.templates import TemplateJ2

    prompt_exec = TemplateJ2()

    with pytest.raises(RuntimeError):
        call(prompt_exec)
