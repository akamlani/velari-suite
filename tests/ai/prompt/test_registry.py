"""Tests for velari_ai.ai.prompt.registry."""

import pytest


def _write_catalog(tmp_path, mapping):
    import yaml

    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(mapping))
    return str(catalog_path)


def test_init_without_uri_leaves_catalog_empty():
    from velari_ai.ai.prompt.registry import PromptRegistry

    registry = PromptRegistry()

    with pytest.raises(KeyError):
        registry.get("summarise")


def test_get_returns_raw_template_string(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, {
        "summarise": "Summarize the following topic in {length} words: {topic}",
    })
    registry = PromptRegistry(uri=uri)

    result = registry.get("summarise")

    assert result == "Summarize the following topic in {length} words: {topic}"


def test_get_finds_entry_after_the_first_row(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, {
        "billing_reminder": "Reminder: balance due for {account_id}.",
        "summarise": "Summarize {topic} in {length} words.",
        "classify": "Classify the intent of: {message}",
    })
    registry = PromptRegistry(uri=uri)

    assert registry.get("summarise") == "Summarize {topic} in {length} words."
    assert registry.get("classify") == "Classify the intent of: {message}"


def test_get_missing_name_raises_keyerror(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, {"summarise": "Summarize {topic}."})
    registry = PromptRegistry(uri=uri)

    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_list_returns_registered_names(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, {
        "billing_reminder": "Reminder: balance due for {account_id}.",
        "summarise": "Summarize {topic} in {length} words.",
    })
    registry = PromptRegistry(uri=uri)

    assert registry.list() == ["billing_reminder", "summarise"]


def test_has_and_len_work_through_inheritance(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, {
        "billing_reminder": "Reminder: balance due for {account_id}.",
        "summarise": "Summarize {topic} in {length} words.",
    })
    registry = PromptRegistry(uri=uri)

    assert registry.has("summarise") is True
    assert registry.has("does_not_exist") is False
    assert "billing_reminder" in registry
    assert len(registry) == 2


def test_to_dataframe_converts_catalog(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, {
        "billing_reminder": "Reminder: balance due for {account_id}.",
        "summarise": "Summarize {topic} in {length} words.",
    })
    registry = PromptRegistry(uri=uri)

    df = registry.to_dataframe()

    assert list(df.columns) == ["name", "value"]
    assert list(df["name"]) == ["billing_reminder", "summarise"]


def test_create_returns_dataframe(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, {
        "summarise": "Summarize {topic} in {length} words.",
    })
    registry = PromptRegistry()

    df = registry.create(uri=uri)

    assert list(df.columns) == ["name", "value"]
    assert list(df["name"]) == ["summarise"]
    assert df.loc[df["name"] == "summarise", "value"].iloc[0] == "Summarize {topic} in {length} words."
    assert df.loc[df["name"] == "summarise", "value"].iloc[0] == "Summarize {topic} in {length} words."


def test_list_returns_empty_list_for_empty_catalog():
    from velari_ai.ai.prompt.registry import PromptRegistry

    registry = PromptRegistry()

    assert registry.list() == []


def test_format_template_substitutes_kwargs(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    uri = _write_catalog(tmp_path, {
        "summarise": "Summarize the following topic in {length} words: {topic}",
    })
    registry = PromptRegistry(uri=uri)

    result = registry.format_template("summarise", topic="climate change", length=200)

    assert result == "Summarize the following topic in 200 words: climate change"


def test_create_empty_file_is_empty_catalog(tmp_path):
    from velari_ai.ai.prompt.registry import PromptRegistry

    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text("")
    registry = PromptRegistry(uri=str(catalog_path))

    with pytest.raises(KeyError):
        registry.get("summarise")
