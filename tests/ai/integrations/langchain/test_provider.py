"""Tests for velari_ai.integrations.langchain.provider."""

import pytest


def test_default_provider_is_openai():
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory()

    assert factory.provider == ProviderName.OPENAI
    assert factory.model is None


def test_from_config_separates_known_and_extra_fields():
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory.from_config({
        "provider": ProviderName.HUGGINGFACE,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "cache_folder": "/tmp/hf-cache",
    })

    assert factory.provider == ProviderName.HUGGINGFACE
    assert factory.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert factory.extra == {"cache_folder": "/tmp/hf-cache"}


def test_get_config_openai_includes_model_and_wraps_raw_api_key():
    from pydantic import SecretStr
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory(provider=ProviderName.OPENAI, extra={"api_key": "sk-test"})

    config = factory.get_config()

    assert config["model"] == "text-embedding-3-small"
    assert isinstance(config["api_key"], SecretStr)
    assert config["api_key"].get_secret_value() == "sk-test"


def test_get_config_huggingface_includes_model_and_provider_defaults():
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory(provider=ProviderName.HUGGINGFACE)

    config = factory.get_config()

    assert config["model"] == "sentence-transformers/all-mpnet-base-v2"
    assert config["encode_kwargs"] == {"normalize_embeddings": True}
    assert config["model_kwargs"] == {"trust_remote_code": False}
    assert "cache_folder" in config


def test_get_config_unsupported_provider_does_not_raise():
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory(provider=ProviderName.ANTHROPIC, model="voyage-3", extra={"api_key": "sk-test"})

    config = factory.get_config()

    assert config == {"model": "voyage-3", "api_key": "sk-test"}


def test_build_openai_returns_openaiembeddings_with_configured_model():
    from langchain_openai import OpenAIEmbeddings
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory(
        provider=ProviderName.OPENAI, model="text-embedding-3-large", extra={"api_key": "sk-test"},
    )

    embeddings = factory.build()

    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "text-embedding-3-large"


def test_build_openai_defaults_to_text_embedding_3_small():
    from langchain_openai import OpenAIEmbeddings
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory(provider=ProviderName.OPENAI, extra={"api_key": "sk-test"})

    embeddings = factory.build()

    assert isinstance(embeddings, OpenAIEmbeddings)
    assert embeddings.model == "text-embedding-3-small"


def test_build_openai_wraps_raw_api_key_in_secretstr():
    from langchain_openai import OpenAIEmbeddings
    from pydantic import SecretStr
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory(provider=ProviderName.OPENAI, extra={"api_key": "sk-test"})

    embeddings = factory.build()

    assert isinstance(embeddings, OpenAIEmbeddings)
    assert isinstance(embeddings.openai_api_key, SecretStr)
    assert embeddings.openai_api_key.get_secret_value() == "sk-test"


def test_build_huggingface_constructs_with_resolved_model_name(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    factory = ProviderEmbeddingFactory(provider=ProviderName.HUGGINGFACE, extra={"cache_folder": "/tmp/hf-cache"})

    embeddings = factory.build()

    assert isinstance(embeddings, _FakeHuggingFaceEmbeddings)
    assert captured["model_name"] == "sentence-transformers/all-mpnet-base-v2"
    assert captured["kwargs"]["cache_folder"] == "/tmp/hf-cache"


def test_build_huggingface_defaults_cache_folder_to_read_cache_dir(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    monkeypatch.setattr(provider_module, "read_cache_dir", lambda author, app: "/fake/cache/huggingface/hub")
    factory = ProviderEmbeddingFactory(provider=ProviderName.HUGGINGFACE)

    factory.build()

    assert captured["kwargs"]["cache_folder"] == "/fake/cache/huggingface/hub"


def test_build_huggingface_defaults_encode_kwargs_to_normalize_embeddings(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    factory = ProviderEmbeddingFactory(provider=ProviderName.HUGGINGFACE)

    factory.build()

    assert captured["kwargs"]["encode_kwargs"] == {"normalize_embeddings": True}


def test_build_huggingface_merges_caller_encode_kwargs_with_default(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    factory = ProviderEmbeddingFactory(
        provider=ProviderName.HUGGINGFACE, extra={"encode_kwargs": {"batch_size": 32}},
    )

    factory.build()

    assert captured["kwargs"]["encode_kwargs"] == {"normalize_embeddings": True, "batch_size": 32}


def test_build_huggingface_caller_can_override_normalize_embeddings(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    factory = ProviderEmbeddingFactory(
        provider=ProviderName.HUGGINGFACE, extra={"encode_kwargs": {"normalize_embeddings": False}},
    )

    factory.build()

    assert captured["kwargs"]["encode_kwargs"] == {"normalize_embeddings": False}


def test_build_huggingface_defaults_model_kwargs_to_trust_remote_code_false(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    factory = ProviderEmbeddingFactory(provider=ProviderName.HUGGINGFACE)

    factory.build()

    assert captured["kwargs"]["model_kwargs"] == {"trust_remote_code": False}


def test_build_huggingface_merges_caller_model_kwargs_with_default(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    factory = ProviderEmbeddingFactory(
        provider=ProviderName.HUGGINGFACE, extra={"model_kwargs": {"device": "cpu"}},
    )

    factory.build()

    assert captured["kwargs"]["model_kwargs"] == {"trust_remote_code": False, "device": "cpu"}


def test_build_huggingface_caller_can_override_trust_remote_code(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    factory = ProviderEmbeddingFactory(
        provider=ProviderName.HUGGINGFACE, extra={"model_kwargs": {"trust_remote_code": True}},
    )

    factory.build()

    assert captured["kwargs"]["model_kwargs"] == {"trust_remote_code": True}


def test_build_sentence_transformers_uses_same_huggingface_backend(monkeypatch):
    import velari_ai.integrations.langchain.provider as provider_module
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    captured = {}

    class _FakeHuggingFaceEmbeddings:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name

    monkeypatch.setattr(provider_module, "HuggingFaceEmbeddings", _FakeHuggingFaceEmbeddings)
    factory = ProviderEmbeddingFactory(
        provider=ProviderName.SENTENCE_TRANSFORMERS, model="sentence-transformers/all-MiniLM-L6-v2",
    )

    embeddings = factory.build()

    assert isinstance(embeddings, _FakeHuggingFaceEmbeddings)
    assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_build_unsupported_provider_raises_valueerror():
    from velari_ai.integrations.langchain.provider import ProviderEmbeddingFactory
    from velari_ai.ai.types import ProviderName

    factory = ProviderEmbeddingFactory(provider=ProviderName.ANTHROPIC)

    with pytest.raises(ValueError):
        factory.build()
