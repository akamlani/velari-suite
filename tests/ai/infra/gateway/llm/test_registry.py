"""Tests for velari_ai.infra.gateway.llm.registry."""

import pytest


@pytest.mark.parametrize(
    "provider, model, adapter_cls_path",
    [
        ("openai", "gpt-4o-mini", "velari_ai.infra.gateway.llm.adapters.openai.provider.OpenAIAdapter"),
        ("anthropic", "claude-sonnet-5", "velari_ai.infra.gateway.llm.adapters.anthropic.provider.AnthropicAdapter"),
    ],
)
def test_resolve_returns_adapter_registered_for_provider(monkeypatch, provider, model, adapter_cls_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from importlib import import_module
    from velari_ai.ai.types import ModelConfig
    from velari_ai.infra.gateway.llm.registry import ModelRegistry

    module_path, cls_name = adapter_cls_path.rsplit(".", 1)
    adapter_cls = getattr(import_module(module_path), cls_name)

    registry = ModelRegistry()
    adapter = registry.resolve(ModelConfig(provider=provider, model=model))

    assert isinstance(adapter, adapter_cls)


def test_resolve_caches_adapter_instance_per_provider_and_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.ai.types import ModelConfig
    from velari_ai.infra.gateway.llm.registry import ModelRegistry

    registry = ModelRegistry()
    model_config = ModelConfig(provider="openai", model="gpt-4o-mini")

    assert registry.resolve(model_config) is registry.resolve(model_config)


def test_resolve_unregistered_provider_raises_valueerror():
    from velari_data.storage import LocalKeyValueStore
    from velari_ai.ai.types import ModelConfig
    from velari_ai.infra.gateway.llm.registry import ModelRegistry

    registry = ModelRegistry(adapters=LocalKeyValueStore())

    with pytest.raises(ValueError):
        registry.resolve(ModelConfig(provider="openai", model="gpt-4o-mini"))


@pytest.mark.parametrize("provider", ["openai", "huggingface", "sentence_transformers"])
def test_embedding_registry_resolve_returns_registered_adapter(monkeypatch, provider):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.ai.types import ModelConfig
    from velari_ai.infra.gateway.llm.adapters.stf import embedding as st_module
    from velari_ai.infra.gateway.llm.adapters.openai.embedding import OpenAIEmbeddingAdapter
    from velari_ai.infra.gateway.llm.registry import EmbeddingRegistry
    from velari_ai.integrations.stf import client as st_client_module

    monkeypatch.setattr(st_client_module, "SentenceTransformer", lambda model_name, **kwargs: object())
    expected_cls = OpenAIEmbeddingAdapter if provider == "openai" else st_module.SentenceTransformersEmbeddingAdapter

    registry = EmbeddingRegistry()
    adapter = registry.resolve(ModelConfig(provider=provider, model="sentence-transformers/all-mpnet-base-v2"))

    assert isinstance(adapter, expected_cls)
