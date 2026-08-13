"""Tests for velari_ai.ai.types."""


def test_modelconfig_from_config_routes_known_and_unknown_keys():
    from velari_ai.ai.types import ModelConfig

    cfg = ModelConfig.from_config({"model": "openai:gpt-4o-mini", "temperature": 0.3, "api_key": "test-key"})

    assert cfg.model == "openai:gpt-4o-mini"
    assert cfg.extra == {"temperature": 0.3, "api_key": "test-key"}


def test_modelconfig_from_config_defaults_model_when_omitted():
    from velari_ai.ai.types import ModelConfig

    cfg = ModelConfig.from_config({})

    assert cfg.model == "openai:gpt-4o-mini"
    assert cfg.extra == {}


def test_agentconfig_from_config_routes_known_and_unknown_keys():
    from velari_ai.ai.types import AgentConfig

    cfg = AgentConfig.from_config({
        "name": "billing-support-agent",
        "max_tool_calls": 3,
        "default_thread_id": "cli-session",
    })

    assert cfg.name == "billing-support-agent"
    assert cfg.max_tool_calls == 3
    assert cfg.extra == {"default_thread_id": "cli-session"}
