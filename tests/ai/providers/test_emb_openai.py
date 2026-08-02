"""Tests for OpenAIProviderEmbeddings."""

import pytest
from omegaconf import OmegaConf


class _FakeEmbedding:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeCreateEmbeddingResponse:
    def __init__(self, data):
        self.data = data


def _fake_create(*, model, input):
    return _FakeCreateEmbeddingResponse(data=[_FakeEmbedding([float(len(text)), 1.0]) for text in input])


def test_get_embeddings_single_string_returns_vector(monkeypatch):
    from velari_ai.integrations.openai.provider import OpenAIProviderEmbeddings

    embedder = OpenAIProviderEmbeddings(api_key="test-key", embedding_model="text-embedding-3-small")
    monkeypatch.setattr(embedder._client.embeddings, "create", _fake_create)

    assert embedder.get_embeddings("hi") == [2.0, 1.0]


def test_get_embeddings_list_returns_matrix(monkeypatch):
    from velari_ai.integrations.openai.provider import OpenAIProviderEmbeddings

    embedder = OpenAIProviderEmbeddings(api_key="test-key", embedding_model="text-embedding-3-small")
    monkeypatch.setattr(embedder._client.embeddings, "create", _fake_create)

    assert embedder.get_embeddings(["a", "bb", "ccc"]) == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]


def test_from_config_uses_default_base_url():
    from velari_ai.integrations.openai.provider import OpenAIProviderEmbeddings

    cfg = OmegaConf.create({"embedding": "text-embedding-3-small"})
    embedder = OpenAIProviderEmbeddings.from_config(cfg, api_key="test-key")

    assert embedder._embedding_model == "text-embedding-3-small"
    assert str(embedder._client.base_url) == "https://api.openai.com/v1/"


def test_from_config_uses_custom_base_url():
    from velari_ai.integrations.openai.provider import OpenAIProviderEmbeddings

    cfg = OmegaConf.create({"embedding": "text-embedding-3-small", "base_url": "https://custom.example/v1"})
    embedder = OpenAIProviderEmbeddings.from_config(cfg, api_key="test-key")

    assert str(embedder._client.base_url) == "https://custom.example/v1/"


def test_embed_populates_emb_column_via_inherited_method(monkeypatch):
    import pandas as pd
    from velari_ai.integrations.openai.provider import OpenAIProviderEmbeddings

    embedder = OpenAIProviderEmbeddings(api_key="test-key", embedding_model="text-embedding-3-small")
    monkeypatch.setattr(embedder._client.embeddings, "create", _fake_create)

    df = pd.DataFrame({"text": ["a", "bb", "ccc"]})
    result = embedder.embed(df, col="text")

    assert result["emb"].tolist() == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]


def test_score_ranks_by_cosine_similarity(monkeypatch):
    import pandas as pd
    from velari_ai.integrations.openai.provider import OpenAIProviderEmbeddings

    embedder = OpenAIProviderEmbeddings(api_key="test-key", embedding_model="text-embedding-3-small")
    monkeypatch.setattr(embedder._client.embeddings, "create", _fake_create)

    df = pd.DataFrame({"text": ["a", "bb", "ccc"]})
    df_embedded = embedder.embed(df, col="text")

    result = embedder.score(df_embedded, search_query="bb")

    assert result["text"].tolist() == ["bb", "ccc", "a"]
    assert result.loc[0, "cosine_similarity"] == pytest.approx(1.0)
