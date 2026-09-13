"""Tests for velari_data.registry."""

import pandas as pd
from omegaconf import OmegaConf


def test_prompt_registry_implements_registry():
    from velari_ai.ai.prompt.registry import PromptRegistry
    from velari_data.registry import Registry

    assert issubclass(PromptRegistry, Registry)


def test_arize_prompt_registry_implements_registry():
    from velari_ai.integrations.arize.registry import PromptRegistry
    from velari_data.registry import Registry

    assert issubclass(PromptRegistry, Registry)


def test_arize_dataset_registry_implements_registry():
    from velari_ai.integrations.arize.registry import DatasetRegistry
    from velari_data.registry import Registry

    assert issubclass(DatasetRegistry, Registry)


def _make_fake_registry():
    from velari_data.registry import Registry

    class _FakeRegistry(Registry):
        def __init__(self, items):
            self._items = items

        def get(self, name):
            return self._items[name]

        def list(self, **kwargs):
            return list(self._items.values())

    return _FakeRegistry({"a": "alpha", "b": "beta"})


def test_has_returns_true_for_existing_name():
    registry = _make_fake_registry()

    assert registry.has("a") is True


def test_has_returns_false_for_missing_name():
    registry = _make_fake_registry()

    assert registry.has("missing") is False


def test_contains_operator_matches_has():
    registry = _make_fake_registry()

    assert "a" in registry
    assert "missing" not in registry


def test_len_matches_list_length():
    registry = _make_fake_registry()

    assert len(registry) == 2


def test_iter_yields_list_items():
    registry = _make_fake_registry()

    assert list(registry) == ["alpha", "beta"]


def test_create_saves_config_to_catalog():
    from velari_data.storage import LocalKeyValueStore

    registry = _make_fake_registry()
    cfg = OmegaConf.create({"uri": "config/prompts/catalog.yaml", "key": "prompts"})

    result = registry.create(cfg)

    assert isinstance(registry._catalog, LocalKeyValueStore)
    assert registry._catalog.get("uri") == "config/prompts/catalog.yaml"
    assert registry._catalog.get("key") == "prompts"
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["name", "value"]
    assert len(result) == 2


def test_to_dataframe_converts_catalog():
    registry = _make_fake_registry()
    cfg = OmegaConf.create({"uri": "config/prompts/catalog.yaml", "key": "prompts"})
    registry.create(cfg)

    df = registry.to_dataframe()

    assert list(df.columns) == ["name", "value"]
    assert len(df) == 2
    assert set(df["name"]) == {"uri", "key"}
    assert df.loc[df["name"] == "uri", "value"].iloc[0] == "config/prompts/catalog.yaml"
