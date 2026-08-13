"""Tests for velari_ai.integrations.tavily.search."""

import pytest


def test_search_returns_parsed_results_and_answer(monkeypatch):
    from velari_ai.integrations.tavily.search import TavilySearch

    def _fake_search(*, query, max_results, search_depth, include_answer):
        return {
            "query": query,
            "answer": "RAG combines retrieval with generation to ground LLM outputs in real data.",
            "results": [
                {
                    "title": "RAG in 2026: A Practical Overview",
                    "url": "https://example.com/rag-2026",
                    "content": "Retrieval-augmented generation pairs a retriever with an LLM...",
                    "score": 0.92,
                },
            ],
        }

    searcher = TavilySearch(api_key="tvly-test-key")
    monkeypatch.setattr(searcher._client, "search", _fake_search)

    response = searcher.search("retrieval-augmented generation", include_answer=True)

    assert response.query == "retrieval-augmented generation"
    assert response.answer == "RAG combines retrieval with generation to ground LLM outputs in real data."
    assert len(response.results) == 1
    assert response.results[0].title == "RAG in 2026: A Practical Overview"
    assert response.results[0].url == "https://example.com/rag-2026"
    assert response.results[0].score == 0.92


def test_search_wraps_client_errors_in_runtimeerror(monkeypatch):
    from velari_ai.integrations.tavily.search import TavilySearch

    def _raise(*args, **kwargs):
        raise ValueError("boom")

    searcher = TavilySearch(api_key="tvly-test-key")
    monkeypatch.setattr(searcher._client, "search", _raise)

    with pytest.raises(RuntimeError):
        searcher.search("anything")
