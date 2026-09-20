"""Tests for velari_ai.infra.gateway.llm.adapters embedding adapters."""


def test_openai_embedding_adapter_embed_texts_maps_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.ai import types
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.adapters.openai.embedding import OpenAIEmbeddingAdapter
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayEmbeddingRequest

    adapter = OpenAIEmbeddingAdapter(types.ModelConfig(provider="openai", model="text-embedding-3-small"))
    result = types.EmbeddingResult(
        result=types.EmbeddingVectors(embeddings=[[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
        metrics=types.ProviderMetrics(
            perf=PerfMetrics(latency_sec=0.0),
            usage=UsageMetrics(input_tokens=10, total_tokens=10),
        ),
    )
    monkeypatch.setattr(adapter._client, "create_embeddings", lambda **kwargs: result)

    response = adapter.embed_texts(GatewayEmbeddingRequest(texts=["a", "bb", "ccc"]))

    assert response.result.embeddings == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert response.metrics.usage.total_tokens == 10
    assert response.model.model == "text-embedding-3-small"


def test_openai_embedding_adapter_get_embeddings_reuses_embed_texts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from velari_ai.ai import types
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.adapters.openai.embedding import OpenAIEmbeddingAdapter

    adapter = OpenAIEmbeddingAdapter(types.ModelConfig(provider="openai", model="text-embedding-3-small"))
    monkeypatch.setattr(
        adapter._client, "create_embeddings",
        lambda *, model, texts, **kwargs: types.EmbeddingResult(
            result=types.EmbeddingVectors(embeddings=[[float(len(t)), 1.0] for t in texts]),
            metrics=types.ProviderMetrics(
                perf=PerfMetrics(latency_sec=0.0),
                usage=UsageMetrics(input_tokens=10, total_tokens=10),
            ),
        ),
    )

    assert adapter.get_embeddings("hi") == [2.0, 1.0]
    assert adapter.get_embeddings(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]


def test_sentence_transformers_embedding_adapter_embed_texts_maps_result(monkeypatch):
    from velari_ai.ai import types
    from velari_ai.ai.schemas.response import PerfMetrics, UsageMetrics
    from velari_ai.infra.gateway.llm.adapters.stf import embedding as adapter_module
    from velari_ai.infra.gateway.llm.schemas.requests import GatewayEmbeddingRequest
    from velari_ai.integrations.stf import client as client_module

    monkeypatch.setattr(client_module, "SentenceTransformer", lambda model_name, **kwargs: object())
    adapter = adapter_module.SentenceTransformersEmbeddingAdapter(
        types.ModelConfig(provider="sentence_transformers", model="sentence-transformers/all-mpnet-base-v2"),
    )
    result = types.EmbeddingResult(
        result=types.EmbeddingVectors(embeddings=[[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
        metrics=types.ProviderMetrics(
            perf=PerfMetrics(latency_sec=0.0),
            usage=UsageMetrics(input_tokens=6, total_tokens=6),
        ),
    )
    monkeypatch.setattr(adapter._client, "embed", lambda texts, **kwargs: result)

    response = adapter.embed_texts(GatewayEmbeddingRequest(texts=["a", "bb", "ccc"]))

    assert response.result.embeddings == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert response.metrics.usage.total_tokens == 6
    assert response.metrics.usage.input_tokens == 6
