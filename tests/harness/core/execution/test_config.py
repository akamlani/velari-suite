"""Tests for velari_harness.core.execution.config."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from omegaconf import OmegaConf


def test_from_config_none_keeps_defaults():
    from velari_harness.core.execution.config import HarnessConfig

    config = HarnessConfig.from_config(None, defaults={"name": "ticket-triage"})

    assert config.name == "ticket-triage"


def test_from_config_cfg_wins_over_defaults():
    from velari_harness.core.execution.config import HarnessConfig

    config = HarnessConfig.from_config(OmegaConf.create({"name": "from-cfg"}), defaults={"name": "from-defaults"})

    assert config.name == "from-cfg"


def test_from_config_overrides_win_over_cfg():
    from velari_harness.core.execution.config import HarnessConfig

    config = HarnessConfig.from_config(OmegaConf.create({"name": "from-cfg"}), name="from-override")

    assert config.name == "from-override"


def test_from_config_registers_project_resolvers():
    from velari_harness.core.execution.config import HarnessConfig

    HarnessConfig.from_config(OmegaConf.create({"name": "churn-review"}))

    assert OmegaConf.has_resolver("as_tuple")


def test_from_config_unknown_field_raises_typeerror():
    from velari_harness.core.execution.config import HarnessConfig

    with pytest.raises(TypeError):
        HarnessConfig.from_config(OmegaConf.create({"name": "x", "sed": 1}))


def test_from_config_non_mapping_raises_typeerror():
    from velari_harness.core.execution.config import HarnessConfig

    not_a_mapping: Any = OmegaConf.create([1, 2])

    with pytest.raises(TypeError):
        HarnessConfig.from_config(not_a_mapping)


def test_init_empty_name_raises_valueerror():
    from velari_harness.core.execution.config import HarnessConfig

    with pytest.raises(ValueError):
        HarnessConfig(name="")


def test_init_non_str_name_raises_typeerror():
    from velari_harness.core.execution.config import HarnessConfig

    not_a_name: Any = 5

    with pytest.raises(TypeError):
        HarnessConfig(name=not_a_name)


def test_config_is_immutable():
    from velari_harness.core.execution.config import HarnessConfig

    config = HarnessConfig(name="ticket-triage")

    with pytest.raises(FrozenInstanceError):
        setattr(config, "name", "other")
