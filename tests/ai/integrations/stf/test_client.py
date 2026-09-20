"""Tests for velari_ai.integrations.stf.client."""

import numpy as np


class _FakeSentenceTransformer:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        return np.array([[float(len(text)), 1.0] for text in texts])

    def preprocess(self, texts, **kwargs):
        return {"attention_mask": np.array([[1, 1] for _ in texts])}


def test_sentence_transformer_client_embed_returns_result(monkeypatch):
    from velari_ai.integrations.stf import client as client_module

    monkeypatch.setattr(client_module, "SentenceTransformer", _FakeSentenceTransformer)
    client = client_module.SentenceTransformerClient("sentence-transformers/all-mpnet-base-v2")

    result = client.embed(["a", "bb", "ccc"])

    assert result.result.embeddings == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert result.metrics.usage.total_tokens == 6
