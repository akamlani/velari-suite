"""Tests for velari_data.registry."""

import pandas as pd
import pytest
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


@pytest.mark.parametrize("class_name", ["PromptRegistry", "ArtifactRegistry"])
def test_mlflow_registries_implement_registry(class_name):
    from velari_ai.integrations.mlflow import registry as mlflow_registry
    from velari_data.registry import Registry

    assert issubclass(getattr(mlflow_registry, class_name), Registry)


@pytest.mark.parametrize(
    "module_path",
    ["velari_ai.integrations.arize.registry", "velari_ai.integrations.mlflow.registry"],
)
def test_dataset_registries_implement_dataset_registry_interface(module_path):
    import importlib
    from velari_data.registry import DatasetRegistryInterface

    assert issubclass(importlib.import_module(module_path).DatasetRegistry, DatasetRegistryInterface)


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


def _scorers():
    return pd.DataFrame({
        "name": ["exact_match", "llm_judge", "tool_call_accuracy"],
        "kind": ["heuristic", "model", "heuristic"],
        "cost": ["low", "high", None],
    })


def test_record_registry_implements_registry():
    from velari_data.registry import RecordRegistry, Registry

    assert issubclass(RecordRegistry, Registry)


def test_record_registry_get_returns_the_row_as_a_dict_with_none_for_a_missing_value():
    from velari_data.registry import RecordRegistry

    assert RecordRegistry(_scorers()).get("tool_call_accuracy") == {"name": "tool_call_accuracy", "kind": "heuristic", "cost": None}


def test_record_registry_get_unknown_name_raises_keyerror():
    from velari_data.registry import RecordRegistry

    with pytest.raises(KeyError):
        RecordRegistry(_scorers()).get("missing")


@pytest.mark.parametrize("filters, expected", [
    ({},                                  ["exact_match", "llm_judge", "tool_call_accuracy"]),
    ({"kind": "heuristic"},               ["exact_match", "tool_call_accuracy"]),
    ({"kind": "heuristic", "cost": "low"}, ["exact_match"]),
])
def test_record_registry_list_returns_the_names_that_match_every_filter(filters, expected):
    from velari_data.registry import RecordRegistry

    assert RecordRegistry(_scorers()).list(**filters) == expected


def test_record_registry_list_unknown_filter_column_raises_keyerror():
    from velari_data.registry import RecordRegistry

    with pytest.raises(KeyError):
        RecordRegistry(_scorers()).list(color="red")


def test_record_registry_has_contains_len_and_iteration_work_through_the_base_class():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry(_scorers())

    assert (registry.has("llm_judge"), "missing" in registry, len(registry), list(registry)) == (True, False, 3, ["exact_match", "llm_judge", "tool_call_accuracy"])


def test_record_registry_to_dataframe_returns_a_copy():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry(_scorers())
    df_copy = registry.to_dataframe()
    df_copy.loc[0, "name"] = "changed"

    assert registry.get("exact_match")["name"] == "exact_match"


@pytest.mark.parametrize("records", [
    pd.DataFrame({"kind": ["heuristic"]}),
    pd.DataFrame({"name": [""]}),
    pd.DataFrame({"name": [1]}),
    pd.DataFrame({"name": ["a", "a"]}),
])
def test_record_registry_create_invalid_records_raises_valueerror(records):
    from velari_data.registry import RecordRegistry

    with pytest.raises(ValueError):
        RecordRegistry(records)


def test_record_registry_empty_has_no_names_and_keeps_the_key_column():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry()

    assert (len(registry), list(registry.to_dataframe().columns)) == (0, ["name"])


def test_record_registry_uses_a_custom_key_column():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry(pd.DataFrame({"id": ["a", "b"], "kind": ["x", "y"]}), key="id")

    assert registry.list(kind="y") == ["b"]


def test_record_registry_to_xml_renders_the_chosen_columns_one_row_per_record():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry(pd.DataFrame({"name": ["exact_match", "llm_judge"], "kind": ["heuristic", "model"], "cost": ["low", "high"]}))

    assert registry.to_xml(["name", "kind"], root="scorers", row="scorer") == (
        "<scorers>\n"
        "  <scorer>\n"
        "    <name>exact_match</name>\n"
        "    <kind>heuristic</kind>\n"
        "  </scorer>\n"
        "  <scorer>\n"
        "    <name>llm_judge</name>\n"
        "    <kind>model</kind>\n"
        "  </scorer>\n"
        "</scorers>"
    )


def test_record_registry_to_xml_leaves_out_a_missing_value_and_a_column_the_records_lack():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry(pd.DataFrame({"name": ["exact_match"], "kind": [None]}))

    assert registry.to_xml(["name", "kind", "color"]) == "<records>\n  <record>\n    <name>exact_match</name>\n  </record>\n</records>"


def test_record_registry_to_xml_uses_the_part_of_a_dotted_column_after_the_last_dot_as_the_tag():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry(pd.DataFrame({"name": ["a"], "meta.category": ["support"]}))

    assert "<category>support</category>" in registry.to_xml(["name", "meta.category"])


def test_record_registry_to_xml_escapes_markup_characters():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry(pd.DataFrame({"name": ["a"], "note": ["a & b < c"]}))

    assert "<note>a &amp; b &lt; c</note>" in registry.to_xml(["name", "note"])


def test_record_registry_to_xml_with_names_renders_only_those_records_in_table_order():
    from velari_data.registry import RecordRegistry

    registry = RecordRegistry(pd.DataFrame({"name": ["a", "b", "c"]}))

    xml = registry.to_xml(["name"], names=["c", "a"])

    assert "<name>a</name>" in xml and "<name>b</name>" not in xml and xml.index("<name>a</name>") < xml.index("<name>c</name>")


def test_record_registry_to_xml_with_an_unknown_name_raises_keyerror():
    from velari_data.registry import RecordRegistry

    with pytest.raises(KeyError):
        RecordRegistry(pd.DataFrame({"name": ["a"]})).to_xml(["name"], names=["missing"])


def test_record_registry_to_xml_without_records_is_empty():
    from velari_data.registry import RecordRegistry

    assert RecordRegistry().to_xml(["name"]) == ""

