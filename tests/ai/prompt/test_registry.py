"""Tests for velari_ai.ai.prompt.registry."""

import pytest


def _write_catalog(tmp_path, entries):
    import yaml

    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump({"prompts": entries}))
    return str(catalog_path)


def test_init_without_uri_leaves_catalog_empty():
    from velari_ai.ai.prompt.registry import PromptRegistry

    registry = PromptRegistry()

    with pytest.raises(KeyError):
        registry.get_template("summarise")


def test_get_template_returns_raw_template_string(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, [
        {"name": "summarise", "template": "Summarize the following topic in {length} words: {topic}"},
    ])
    registry = PromptRegistry(uri=uri, key="prompts")

    result = registry.get_template("summarise")

    assert result == "Summarize the following topic in {length} words: {topic}"


def test_get_template_finds_entry_after_the_first_row(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, [
        {"name": "billing_reminder", "template": "Reminder: balance due for {account_id}."},
        {"name": "summarise", "template": "Summarize {topic} in {length} words."},
        {"name": "classify", "template": "Classify the intent of: {message}"},
    ])
    registry = PromptRegistry(uri=uri, key="prompts")

    assert registry.get_template("summarise") == "Summarize {topic} in {length} words."
    assert registry.get_template("classify") == "Classify the intent of: {message}"


def test_get_template_missing_name_raises_keyerror(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, [
        {"name": "summarise", "template": "Summarize {topic}."},
    ])
    registry = PromptRegistry(uri=uri, key="prompts")

    with pytest.raises(KeyError):
        registry.get_template("does_not_exist")


def test_format_template_substitutes_kwargs(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, [
        {"name": "summarise", "template": "Summarize the following topic in {length} words: {topic}"},
    ])
    registry = PromptRegistry(uri=uri, key="prompts")

    result = registry.format_template("summarise", topic="climate change", length=200)

    assert result == "Summarize the following topic in 200 words: climate change"


def test_load_catalog_key_pointing_at_empty_list_is_empty_catalog(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, [])
    registry = PromptRegistry(uri=uri, key="prompts")

    with pytest.raises(KeyError):
        registry.get_template("summarise")


def test_load_catalog_missing_key_returns_empty_dict(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, [{"name": "summarise", "template": "Summarize {topic}."}])
    registry = PromptRegistry(uri=uri, key="does_not_exist")

    with pytest.raises(KeyError):
        registry.get_template("summarise")
