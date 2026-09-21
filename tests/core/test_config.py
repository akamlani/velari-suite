"""Tests for velari_core.core.config."""
from dataclasses import dataclass, field
from typing import Any, Dict

from omegaconf import OmegaConf


def _make_config_cls():
    from velari_core.core.config import ConfigBase

    @dataclass
    class ChatConfig(ConfigBase):
        model:      str             = field(default="gpt-4o-mini")
        parameters: Dict[str, Any]  = field(default_factory=dict)

    return ChatConfig


def test_from_section_converts_nested_values_to_plain_dicts():
    section = OmegaConf.create({"model": "gpt-4o", "parameters": {"temperature": 0.2}})

    config = _make_config_cls().from_section(section)

    assert type(config.parameters) is dict
    assert config.parameters == {"temperature": 0.2}


def test_from_section_resolves_interpolations():
    section = OmegaConf.create({"base": "gpt-4o", "model": "${base}"})

    config = _make_config_cls().from_section(section)

    assert config.model == "gpt-4o"


def test_from_section_dict_override_is_unioned_into_section_dict():
    section = OmegaConf.create({"parameters": {"temperature": 0.2, "seed": 1}})

    config = _make_config_cls().from_section(section, parameters={"seed": 42})

    assert config.parameters == {"temperature": 0.2, "seed": 42}


def test_from_section_scalar_override_replaces_section_value():
    section = OmegaConf.create({"model": "gpt-4o-mini"})

    config = _make_config_cls().from_section(section, model="gpt-4o")

    assert config.model == "gpt-4o"


def test_from_section_unknown_override_lands_in_extra():
    config = _make_config_cls().from_section(OmegaConf.create({}), api_key="test-key")

    assert config.extra == {"api_key": "test-key"}


def test_from_section_non_mapping_raises_typeerror():
    import pytest

    with pytest.raises(TypeError):
        _make_config_cls().from_section(OmegaConf.create([1, 2]))
